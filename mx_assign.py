"""Turn per-layer sensitivity scores into a precision assignment.

The scorer answers "how much does this layer suffer at low precision?"; this
module answers "so what format does it get?". It is a pure function of the
sensitivity artifact plus the config — no model, no forward passes — so an
assignment can be recomputed, reviewed and hand-edited offline without paying
for scoring again.

Ladders are written low precision first, e.g. ["int4", "int6", "int8"]. Layers
sort ascending by sensitivity, so the least sensitive fill the lowest rung.
"""

import json
import math
import os

from mx_sensitivity import UNMEASURED, clean_name, elem_bits, spec_bits


# =============================================================================
# Spec key ownership — which rung owns which knob in a w/a-separable merge
# =============================================================================

W_KEYS = ("w_elem_format", "block_size_wt", "flatten_wt",
          "block_axes_wt", "block_shape_wt")
A_KEYS = ("a_elem_format", "block_axes_act", "block_shape_act")
# Keys that describe the block geometry both operands share. They must agree
# across rungs, because a merged spec has exactly one of each.
SHARED_KEYS = ("block_axes", "block_shape", "scale_bits",
               "shared_exp_method", "round_mx_output", "custom_cuda")


class AssignError(ValueError):
    """Configuration is inconsistent — raised before any expensive work."""


# =============================================================================
# Validation — everything checkable before a single forward pass
# =============================================================================

def validate(auto_cfg, groups, quant_probe=None, all_groups=None, raw_groups=None):
    """Check an auto_mixed block against the available groups.

    Runs before scoring on purpose: a typo'd group name or an element format the
    library does not implement should cost a second, not a twenty-minute scoring
    run followed by a crash.

    Args:
        groups: the ladder rungs; every constraint about formats and shared keys
            applies to these.
        all_groups: every group a pin may name. A pin is allowed to point at a
            group that is not on the ladder — that is how a layer opts out of
            the ladder's block geometry or accumulator model entirely, e.g. a
            ConvTranspose2d pinned to a plain-MX group while the ladder carries
            xblock_accum. Defaults to `groups`.
        raw_groups: the ladder groups as written in the config, before the
            deployment spec was merged in. The structural checks below are about
            what the *user* put on a rung, and `groups` has by then inherited
            block geometry and xblock_accum from the deployment spec — checking
            the merged specs would reject every accumulator-model config.
            Defaults to `groups`.
    """
    ladder = auto_cfg.get("ladder") or []
    if not ladder:
        raise AssignError("auto_mixed.ladder must list at least one group, "
                          "lowest precision first")

    unknown = [g for g in ladder if g not in groups]
    if unknown:
        raise AssignError(
            f"auto_mixed.ladder references undefined group(s) {unknown}; "
            f"defined groups are {sorted(groups)}")

    raw = raw_groups if raw_groups is not None else groups
    for name, spec in ((g, raw[g]) for g in ladder if g in raw):
        if spec.get("xblock_accum") is not None:
            raise AssignError(
                f"ladder group '{name}' carries xblock_accum. That selects the "
                f"layer class (MXConv2dHW / MXConv2dBlocked / MXConv2d), not a "
                f"number format, so letting a rung set it would silently change "
                f"the hardware model. Put xblock_accum on the deployment group "
                f"instead.")

    for key in SHARED_KEYS:
        seen = {raw[g][key] for g in ladder if g in raw and key in raw[g]}
        if len(seen) > 1:
            raise AssignError(
                f"ladder groups disagree on '{key}' ({sorted(map(str, seen))}). "
                f"It is shared by both operands, so a merged w/a spec cannot "
                f"honour both values.")

    strategy = auto_cfg.get("strategy", "quantile")
    if strategy == "quantile":
        fracs = auto_cfg.get("quantile") or {}
        missing = [g for g in ladder if g not in fracs]
        if missing:
            raise AssignError(f"auto_mixed.quantile is missing a fraction for {missing}")
        total = sum(float(fracs[g]) for g in ladder)
        if abs(total - 1.0) > 1e-6:
            raise AssignError(
                f"auto_mixed.quantile fractions sum to {total:.6f}, expected 1.0")
    elif strategy == "threshold":
        if not auto_cfg.get("threshold"):
            raise AssignError("strategy 'threshold' needs an auto_mixed.threshold block")
    elif strategy == "cost_budget":
        if not (auto_cfg.get("cost_budget") or {}).get("target_avg_bits"):
            raise AssignError(
                "strategy 'cost_budget' needs auto_mixed.cost_budget.target_avg_bits")
    else:
        raise AssignError(
            f"Unknown auto_mixed strategy {strategy!r}. "
            f"Use 'quantile', 'threshold' or 'cost_budget'.")

    pin_groups = all_groups if all_groups is not None else groups
    for layer, pin in (auto_cfg.get("pins") or {}).items():
        for g in ([pin] if isinstance(pin, str) else list(pin.values())):
            if g not in pin_groups:
                raise AssignError(
                    f"pin for '{layer}' references undefined group '{g}'")

    # Formats must actually exist, and must survive a real quantization call.
    for g in ladder:
        spec = groups[g]
        for key in ("w_elem_format", "a_elem_format"):
            if key in spec:
                try:
                    elem_bits(spec[key])
                except Exception as exc:
                    raise AssignError(
                        f"group '{g}' uses {key}={spec[key]!r}, which this "
                        f"microxcaling build does not support ({exc})")
        if quant_probe is not None:
            try:
                quant_probe(spec)
            except Exception as exc:
                raise AssignError(f"group '{g}' failed a trial quantization: {exc}")


# =============================================================================
# Apportionment
# =============================================================================

def apportion(n, fractions):
    """Split `n` items into per-rung counts by largest remainder.

    Plain truncation loses layers (10/20/70 of 47 gives 4+9+32 = 45) and naive
    rounding can overshoot. Largest remainder always sums to exactly `n`, so
    every layer lands on exactly one rung.

    Args:
        n: number of items to distribute.
        fractions: list of (key, fraction) in ladder order.

    Returns:
        dict: key -> count, summing to n.
    """
    exact = [(k, n * float(f)) for k, f in fractions]
    counts = {k: int(math.floor(v)) for k, v in exact}
    remainder = n - sum(counts.values())
    # Ties broken by ladder order, so the result is reproducible.
    order = sorted(range(len(exact)),
                   key=lambda i: (-(exact[i][1] - math.floor(exact[i][1])), i))
    for i in order[:remainder]:
        counts[exact[i][0]] += 1
    return counts


# =============================================================================
# Assignment
# =============================================================================

def _sort_key(row):
    """Ascending by sensitivity, ties broken by name for reproducibility."""
    return (row["sensitivity"], row["name"])


def assign(rows, auto_cfg, groups, log=None):
    """Map scored layers to ladder rungs.

    Args:
        rows: list of {"name", "sensitivity", "status", "cost": {"macs", ...}}.
        auto_cfg: the auto_mixed config block.
        groups: {group_name: spec_dict}.

    Returns:
        (assignments, notes) where assignments is {layer: group} and notes
        records what happened to the layers that were not ranked.
    """
    ladder = list(auto_cfg["ladder"])
    top = ladder[-1]
    strategy = auto_cfg.get("strategy", "quantile")
    pins = auto_cfg.get("pins") or {}

    assignments = {}
    notes = {"pinned": [], "unmeasured": [], "tied": {}}

    pool = []
    for row in rows:
        name = row["name"]
        if name in pins:
            pin = pins[name]
            assignments[name] = pin if isinstance(pin, str) else pin
            notes["pinned"].append(name)
            continue
        if row.get("status") in UNMEASURED or row.get("sensitivity") is None:
            # Not "insensitive" — unmeasured. Safest rung, and kept out of the
            # denominator so it cannot shift the requested fractions.
            assignments[name] = top
            notes["unmeasured"].append(name)
            continue
        s = row["sensitivity"]
        if isinstance(s, float) and math.isnan(s):
            assignments[name] = top
            notes["unmeasured"].append(name)
            continue
        pool.append(row)

    pool.sort(key=_sort_key)

    if strategy == "quantile":
        fracs = [(g, float(auto_cfg["quantile"][g])) for g in ladder]
        counts = apportion(len(pool), fracs)
        i = 0
        for g, c in ((g, counts[g]) for g in ladder):
            for row in pool[i:i + c]:
                assignments[row["name"]] = g
            i += c
    elif strategy == "threshold":
        cuts = auto_cfg["threshold"]
        for row in pool:
            chosen = ladder[0]
            for g in ladder:
                if g in cuts and row["sensitivity"] > float(cuts[g]):
                    continue
                chosen = g
                break
            else:
                chosen = ladder[-1]
            assignments[row["name"]] = chosen
    elif strategy == "cost_budget":
        assignments.update(_cost_budget(pool, ladder, groups, auto_cfg, log))

    # Surface ties: a run of identical scores was split by name, which is
    # deterministic but arbitrary, and the reviewer should know.
    by_score = {}
    for row in pool:
        by_score.setdefault(row["sensitivity"], []).append(row["name"])
    notes["tied"] = {str(k): v for k, v in by_score.items() if len(v) > 1}

    return assignments, notes


def _cost_budget(pool, ladder, groups, auto_cfg, log=None):
    """Demote layers while the cost target allows, cheapest damage first.

    Sorted by bits saved per dB lost — not by sensitivity. Ranking on
    sensitivity alone spends the whole budget upgrading one huge sensitive
    layer while a long tail of tiny layers gets demoted for no real saving.
    Needs per-rung scores; a rank-only artifact cannot support it.
    """
    cfg = auto_cfg["cost_budget"]
    target = float(cfg["target_avg_bits"])
    top = ladder[-1]

    missing = [r["name"] for r in pool if not (r.get("per_rung") or {})]
    if missing:
        raise AssignError(
            f"strategy 'cost_budget' needs a per-rung score curve, but "
            f"{len(missing)} layer(s) have none (scored with probe='bottom'). "
            f"Re-score with probe='all_rungs'.")

    bits = {g: spec_bits(groups[g]) for g in ladder}
    assign_map = {r["name"]: top for r in pool}
    weights = {r["name"]: max(1.0, float((r.get("cost") or {}).get("macs", 1)))
               for r in pool}

    def avg_bits():
        num = sum(bits[assign_map[n]] * weights[n] for n in assign_map)
        return num / sum(weights.values())

    # Candidate demotions: one step down the ladder at a time.
    while avg_bits() > target:
        best, best_ratio = None, None
        for row in pool:
            name = row["name"]
            cur = ladder.index(assign_map[name])
            if cur == 0:
                continue
            nxt = ladder[cur - 1]
            d_bits = (bits[assign_map[name]] - bits[nxt]) * weights[name]
            d_sens = ((row["per_rung"].get(nxt) or 0.0)
                      - (row["per_rung"].get(assign_map[name]) or 0.0))
            ratio = d_bits / max(d_sens, 1e-6)
            if best_ratio is None or ratio > best_ratio:
                best, best_ratio = (name, nxt), ratio
        if best is None:
            if log is not None:
                log(f"cost_budget | every layer is at the lowest rung; "
                    f"realized {avg_bits():.2f} bits vs target {target:.2f}")
            break
        assign_map[best[0]] = best[1]

    return assign_map


# =============================================================================
# w/a-separable merge
# =============================================================================

def split_wa_supported(deploy_spec):
    """Can this deployment path run a different format for weights and acts?

    Not everywhere. MXConv2dHW models one fixed-point datapath: both operands
    are shifted onto a single shared accumulator grid, so it requires
    `a_elem_format == w_elem_format` and raises at forward time otherwise. A
    w/a-separable assignment is simply not deployable there, and finding that
    out mid-probe wastes the whole scoring run.

    Returns:
        (ok, reason): `reason` is None when supported.
    """
    cfg = (deploy_spec or {}).get("xblock_accum") or {}
    if cfg.get("enabled") and cfg.get("mode") == "hw_fixed_point":
        return False, ("MXConv2dHW requires a_elem_format == w_elem_format — "
                       "the fixed-point datapath shifts both operands onto one "
                       "shared accumulator grid")
    return True, None


def merge_wa(groups, w_group, a_group, deploy_group=None):
    """Build one spec that takes its weight format from `w_group` and its
    activation format from `a_group`.

    Two layers can share an output SQNR while one is weight-underflow driven and
    the other activation-underflow driven; spending bits on the wrong operand
    buys nothing. Hence the split — but only the format keys split. Block
    geometry is shared by both operands and the accumulator model belongs to the
    deployment group, so both are taken whole rather than mixed.

    Returns:
        (group_name, spec_dict)
    """
    if w_group == a_group:
        return w_group, dict(groups[w_group])

    w, a = groups[w_group], groups[a_group]
    base = dict(groups[deploy_group] if deploy_group else a)

    ok, why = split_wa_supported(base)
    if not ok and w.get("w_elem_format") != a.get("a_elem_format"):
        raise AssignError(
            f"cannot merge '{w_group}' and '{a_group}': {why}.")

    for key in SHARED_KEYS:
        if key in w and key in a and w[key] != a[key]:
            raise AssignError(
                f"cannot merge '{w_group}' and '{a_group}': they disagree on "
                f"shared key '{key}' ({w[key]!r} vs {a[key]!r})")

    spec = dict(base)
    for key in A_KEYS:
        if key in a:
            spec[key] = a[key]
        else:
            spec.pop(key, None)
    for key in W_KEYS:
        if key in w:
            spec[key] = w[key]
        elif key != "w_elem_format":
            spec.pop(key, None)

    # block_size is one knob shared by both operands. When the rungs want
    # different weight blocking, the difference is expressed as block_size_wt —
    # which the Blocked/HW conv classes ignore, so with an accumulator model in
    # play the request is simply not buildable and must fail here rather than be
    # silently dropped at replace time.
    w_bs = w.get("block_size_wt", w.get("block_size"))
    a_bs = a.get("block_size")
    if w_bs is not None and a_bs is not None and w_bs != a_bs:
        if (base.get("xblock_accum") or {}).get("enabled"):
            raise AssignError(
                f"cannot merge '{w_group}' and '{a_group}': they need different "
                f"weight/activation block sizes ({w_bs} vs {a_bs}), but the "
                f"deployment group enables xblock_accum, whose conv classes "
                f"ignore block_size_wt.")
        spec["block_size"] = a_bs
        spec["block_size_wt"] = w_bs
    else:
        spec.pop("block_size_wt", None)

    return group_name_for(spec, w_group, a_group), spec


def group_name_for(spec, w_group, a_group):
    """A readable name for a merged group, e.g. 'w4a8'.

    Named for what it is rather than 'merged_17', because a human reviews and
    hand-edits the resolved config.
    """
    def short(fmt):
        return str(fmt).replace("_elem_format", "").replace("int", "").replace("fp", "f")
    w = short(spec.get("w_elem_format", w_group))
    a = short(spec.get("a_elem_format", a_group))
    return f"w{w}a{a}"


def resolve_groups(assignments, groups, deploy_group=None):
    """Expand w/a-separable assignments into concrete named groups.

    Returns:
        (flat_assignments {layer: group_name}, extra_groups {name: spec})
    """
    flat, extra = {}, {}
    for layer, target in assignments.items():
        if isinstance(target, str):
            flat[layer] = target
            continue
        name, spec = merge_wa(groups, target["w"], target["a"], deploy_group)
        flat[layer] = name
        if name not in groups:
            extra[name] = spec
    return flat, extra


# =============================================================================
# Cost reporting
# =============================================================================

def cost_summary(assignments, groups, extra_groups, costs):
    """Average bits, plain and MAC-weighted, plus the per-rung head count."""
    all_groups = dict(groups)
    all_groups.update(extra_groups)
    per_rung = {}
    total_bits = 0.0
    weighted_bits = 0.0
    total_macs = 0
    for layer, group in assignments.items():
        per_rung[group] = per_rung.get(group, 0) + 1
        bits = spec_bits(all_groups[group])
        total_bits += bits
        macs = int((costs.get(layer) or {}).get("macs", 0))
        weighted_bits += bits * macs
        total_macs += macs
    n = max(1, len(assignments))
    return {
        "per_rung_counts": dict(sorted(per_rung.items())),
        "avg_bits": total_bits / n,
        "macs_weighted_avg_bits": (weighted_bits / total_macs) if total_macs else None,
        "total_macs": total_macs,
    }


# =============================================================================
# Reuse — replaying a saved score set instead of re-measuring it
# =============================================================================

SCHEMA_VERSION = 1

# What a cached score set is allowed to be reused across. Anything not listed
# here is free to change between runs, and changing it is the whole point of
# reuse: ladder fractions, thresholds, budgets, pins and strategy are decided
# at assignment time and cost nothing to redo.
FINGERPRINT_KEYS = ("model", "layers", "probe_spec", "reference_spec")

_FINGERPRINT_WHY = {
    "model": "model weights changed",
    "layers": "the candidate layer set changed",
    "probe_spec": "the probe rung's spec changed",
    "reference_spec": "the reference rung's spec changed",
    "input_shape": "the calibration input shape changed",
}


def load_sensitivity(path):
    """Read back a sensitivity.json written by `write_sensitivity`.

    Returns (meta, rows, summary). Rows come back in the shape `assign()` takes,
    which is the same shape they were written in — the artifact is a verbatim
    dump of them, so a replay is a replay and not a reconstruction.
    """
    with open(path) as f:
        blob = json.load(f)
    if not isinstance(blob, dict) or not isinstance(blob.get("layers"), list):
        raise AssignError(f"{path}: not a sensitivity artifact "
                          f"(expected an object with a 'layers' list)")
    version = blob.get("schema_version")
    if version != SCHEMA_VERSION:
        raise AssignError(f"{path}: schema_version {version!r}, this build "
                          f"writes {SCHEMA_VERSION}. Delete it and re-score.")
    rows = [dict(row) for row in blob["layers"]]
    for row in rows:
        if "name" not in row:
            raise AssignError(f"{path}: a row has no 'name'")
        row["name"] = clean_name(row["name"])
    return dict(blob.get("meta") or {}), rows, dict(blob.get("summary") or {})


def costs_from_rows(rows):
    """Rebuild the {layer: {macs, params}} table `cost_summary` needs.

    This is why `cost` is written per row: MACs come from shape hooks on a real
    forward, and carrying them in the artifact is what makes a replay need no
    model and no data.
    """
    return {row["name"]: dict(row.get("cost") or {}) for row in rows}


def reuse_verdict(meta, expected, rows, candidates, auto_cfg):
    """Decide whether a cached score set may be replayed.

    Returns (ok, reasons). `reasons` is never empty when ok is False, and is
    phrased for a log line a human reads six weeks later.

    The check is deliberately conservative: a false reject costs one scoring
    run, a false accept costs a wrong precision assignment that looks entirely
    reasonable and is never questioned again.
    """
    reasons = []
    cached = dict(meta.get("fingerprint") or {})
    if not cached:
        return False, ["artifact predates fingerprinting (no meta.fingerprint)"]

    for key in FINGERPRINT_KEYS:
        want = expected.get(key)
        got = cached.get(key)
        if want is None:
            continue
        if got is None:
            reasons.append(f"cached artifact has no {key} fingerprint")
        elif got != want:
            reasons.append(_FINGERPRINT_WHY[key])

    # Input shape is checked only when both sides know it. A replay with no
    # calibration data cannot compute it, and refusing on that would make the
    # data-free path — the cheapest and most useful one — impossible.
    want_shape, got_shape = expected.get("input_shape"), cached.get("input_shape")
    if want_shape is not None and got_shape is not None and want_shape != got_shape:
        reasons.append(_FINGERPRINT_WHY["input_shape"])

    scored = {row["name"] for row in rows}
    missing = sorted(clean_name(n) for n in candidates
                     if clean_name(n) not in scored)
    if missing:
        head = ", ".join(missing[:3])
        more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
        reasons.append(f"{len(missing)} candidate layer(s) are not in the "
                       f"artifact: {head}{more}")

    # cost_budget prices a demotion in dB per rung, so a rank alone will not do.
    # A "bottom"-probed artifact has null per_rung and must be refused here
    # rather than produce a budget built on missing data.
    if auto_cfg.get("strategy") == "cost_budget":
        ladder = list(auto_cfg["ladder"])
        need = set(ladder[:-1])
        for row in rows:
            if row.get("status") in UNMEASURED or row.get("sensitivity") is None:
                continue
            have = {k for k, v in (row.get("per_rung") or {}).items()
                    if v is not None}
            if not need <= have:
                reasons.append(
                    f"strategy 'cost_budget' needs a per-rung curve for "
                    f"{sorted(need)}, and the artifact has "
                    f"{sorted(have) or 'none'} for '{row['name']}' "
                    f"(re-score with probe: \"all_rungs\")")
                break

    return (not reasons), reasons


def rows_have_wa(rows):
    """True if the artifact carries the w-only / a-only scores a separable
    assignment needs. A replay cannot produce them — they come from extra
    probes — so a run that wants them and does not have them must be told."""
    for row in rows:
        if row.get("status") in UNMEASURED:
            continue
        scores = row.get("scores") or {}
        if scores.get("w") is not None and scores.get("a") is not None:
            return True
    return False


# =============================================================================
# Artifacts
# =============================================================================

def write_sensitivity(path, meta, rows, summary):
    """Write sensitivity.json — the reviewable record of why each layer got
    the format it got. Rows are pre-sorted worst-first so the file reads top
    down and diffs stay stable between runs."""
    blob = {"schema_version": SCHEMA_VERSION, "meta": meta, "layers": rows,
            "summary": summary}
    with open(path, "w") as f:
        json.dump(_round_floats(blob), f, indent=2)
    return path


def _round_floats(obj, ndigits=4):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def build_resolved_config(base_config, assignments, groups, extra_groups,
                          provenance=None):
    """Emit a plain groups+layers config with no auto_mixed key.

    Re-running this config is a deterministic replay: no scoring, no randomness,
    nothing to re-derive. It is also the thing a human edits when they disagree
    with one of the assignments.
    """
    used = sorted(set(assignments.values()))
    out_groups = {}
    for g in used:
        out_groups[g] = dict(extra_groups[g] if g in extra_groups else groups[g])

    layers = []
    for entry in base_config.get("layers", []):
        if isinstance(entry, dict) and entry.get("kind") in ("act_quant", "out_quant"):
            layers.append(entry)          # carried through untouched

    for name in sorted(assignments):
        layers.append({"name": name, "group": assignments[name]})

    resolved = {"groups": out_groups, "layers": layers}
    for key in ("ptq", "collect_stats", "measure_error"):
        if key in base_config:
            resolved[key] = base_config[key]
    if provenance:
        resolved["_provenance"] = provenance
    return resolved


def write_resolved_config(path, resolved):
    with open(path, "w") as f:
        json.dump(resolved, f, indent=2)
    return path
