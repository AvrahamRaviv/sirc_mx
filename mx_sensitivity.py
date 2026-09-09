"""Per-layer quantization sensitivity scoring for automated mixed precision.

A scorer answers one question: *how much does the network suffer if this layer
runs at a lower precision?* Every scorer shares one signature and one sign
convention, so a new algorithm — ours or a third party's — plugs in without
touching MXQuantizer:

    score_layers(model, candidates, ...) -> {layer_name: Score}

    Score = {"sensitivity": float | None,   # HIGHER = MORE SENSITIVE
             "status": str,                 # ok | unreached | zero_output | no_change
             "scores": {...}, "metrics": {...}, "cost": {...}, "meta": {...}}

Note the sign: the existing code stores SQNR, where *higher is better*. Here
sensitivity is normalized so higher always means "needs more bits", which is
what an assignment ladder wants. Layers with no usable measurement carry
`sensitivity: None` and a `status` saying why; they are never silently ranked.
"""

import math
import os
import sys

sys.path.append('/Users/avrahamraviv/PycharmProjects')
sys.path.append('/home/avrahamra/PycharmProjects')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from microxcaling.mx.formats import _get_format_params


# =============================================================================
# Status taxonomy — why a layer has no usable score
# =============================================================================

STATUS_OK = "ok"                    # measured normally
STATUS_UNREACHED = "unreached"      # module never ran during calibration
STATUS_ZERO_OUTPUT = "zero_output"  # reference signal energy is 0
STATUS_NO_CHANGE = "no_change"      # probe spec == reference spec for this layer

# Statuses that must NOT enter an apportionment denominator: they are not
# "insensitive layers", they are layers we failed to measure. Counting them
# would silently shift every requested fraction.
UNMEASURED = (STATUS_UNREACHED, STATUS_ZERO_OUTPUT, STATUS_NO_CHANGE)


# =============================================================================
# Name handling
# =============================================================================

def clean_name(name):
    """Strip the DataParallel `module.` prefix.

    The single naming rule for the whole pipeline. Config files
    (`configs/mx_config_*.json`) carry unprefixed names while a model wrapped in
    nn.DataParallel reports `module.x` from named_modules() — and
    MXQuantizer._replace_layers looks up the *stripped* name. Any code path that
    produced prefixed keys therefore matched nothing, silently.
    """
    return name[len("module."):] if name.startswith("module.") else name


def unwrap_parallel(model):
    """Return (inner_model, was_wrapped) for DataParallel / DDP.

    Scoring never runs under DataParallel: replicas are rebuilt per forward and
    scatter/gather makes the output comparison depend on batch splitting, which
    is exactly the signal we are trying to measure.
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module, True
    return model, False


# =============================================================================
# Calibration data
# =============================================================================

def materialize(data, max_batches):
    """Take up to `max_batches` batches from `data` and return them as a list.

    `data` is frequently a generator or a DataLoader iterator. Scoring, PTQ,
    error measurement and stats collection each iterate it in turn, so a
    one-shot iterable silently starves every phase after the first. Anything
    already re-iterable (list/tuple) is returned as-is.
    """
    if data is None:
        return None
    if isinstance(data, (list, tuple)):
        return list(data) if max_batches <= 0 else list(data)[:max_batches]
    out = []
    for i, batch in enumerate(data):
        if max_batches > 0 and i >= max_batches:
            break
        out.append(batch)
    return out


def default_forward(model, batch, forward_fn=None):
    """Run one calibration batch, matching MXQuantizer's dispatch rules."""
    if forward_fn is not None:
        return forward_fn(model, batch)
    if isinstance(batch, (list, tuple)):
        return model(batch[0])
    if isinstance(batch, dict):
        return model(**batch)
    return model(batch)


# =============================================================================
# Output reduction
# =============================================================================

def flatten_outputs(out):
    """Reduce a model's return value to a deterministic list of float tensors.

    Real models here return more than a tensor: dvnr is multi-frame temporal,
    DOF returns several heads. Non-float tensors (indices, masks, counts) are
    dropped — an SQNR over int64 indices is meaningless — and the traversal
    order is fixed so reference and probe line up element by element.
    """
    acc = []

    def walk(x):
        if torch.is_tensor(x):
            if x.is_floating_point():
                acc.append(x.detach())
            return
        if isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
            return
        if isinstance(x, dict):
            for k in sorted(x.keys(), key=str):
                walk(x[k])
            return
        if hasattr(x, "__dataclass_fields__"):
            for f in x.__dataclass_fields__:
                walk(getattr(x, f))
            return
        # anything else (None, scalars, strings) carries no signal

    walk(out)
    return acc


# =============================================================================
# Error accumulation
# =============================================================================

class ErrAcc:
    """Online signal/noise accumulator over a stream of (ref, probe) pairs.

    Same math as MXQuantizer._measure_error, factored out so the scorer and the
    error measurement cannot drift apart. Everything is scalar, so memory is
    O(1) regardless of how many batches or how large the tensors are.
    """

    def __init__(self):
        self.signal_sq = 0.0
        self.noise_sq = 0.0
        self.n_elem = 0
        self.max_abs_err = 0.0
        self.dot = 0.0
        self.probe_sq = 0.0
        self.shape_mismatch = False

    def update(self, ref, probe):
        if ref.shape != probe.shape:
            # A probe that changes the output *shape* (e.g. a detection head
            # whose NMS keeps a different number of boxes) is maximally
            # sensitive, not an error to raise.
            self.shape_mismatch = True
            return
        ref = ref.detach().float()
        probe = probe.detach().float()
        err = ref - probe
        self.signal_sq += ref.pow(2).sum().item()
        self.noise_sq += err.pow(2).sum().item()
        self.dot += (ref * probe).sum().item()
        self.probe_sq += probe.pow(2).sum().item()
        self.max_abs_err = max(self.max_abs_err, err.abs().max().item())
        self.n_elem += ref.numel()

    def summary(self):
        """Return (status, metrics). `sqnr_db` is None when undefined."""
        if self.shape_mismatch:
            return STATUS_OK, {"sqnr_db": -math.inf, "mse": math.inf,
                               "cos_sim": 0.0, "max_abs_err": math.inf,
                               "shape_mismatch": True}
        if self.n_elem == 0:
            return STATUS_UNREACHED, {}
        if self.signal_sq == 0:
            return STATUS_ZERO_OUTPUT, {}
        mse = self.noise_sq / self.n_elem
        denom = math.sqrt(self.signal_sq * self.probe_sq)
        cos_sim = max(-1.0, min(1.0, self.dot / denom)) if denom > 0 else float("nan")
        if self.noise_sq == 0:
            sqnr_db = math.inf
        else:
            sqnr_db = 10 * math.log10(self.signal_sq / self.noise_sq)
        return STATUS_OK, {"sqnr_db": sqnr_db, "mse": mse, "cos_sim": cos_sim,
                           "max_abs_err": self.max_abs_err}


def sqnr_to_sensitivity(sqnr_db):
    """SQNR (higher = better) -> sensitivity (higher = worse).

    An infinite SQNR means the demotion changed nothing, so the layer is as
    insensitive as it gets.
    """
    if sqnr_db is None or (isinstance(sqnr_db, float) and math.isnan(sqnr_db)):
        return None
    if sqnr_db == math.inf:
        return -math.inf
    return -float(sqnr_db)


# =============================================================================
# Layer cost — for reporting and for cost-aware assignment
# =============================================================================

def elem_bits(fmt):
    """Storage bits for one element of an MX element format."""
    ebits, mbits, _, _, _ = _get_format_params(fmt)
    # int formats: mbits counts sign + magnitude, no implicit bit.
    # float formats: mbits counts sign + implicit + explicit mantissa.
    return mbits if ebits == 0 else ebits + mbits - 1


def spec_bits(spec_dict):
    """Average bits per weight element, including the shared block scale."""
    fmt = spec_dict.get("w_elem_format", "int8")
    block = spec_dict.get("block_size", 32) or 32
    scale = spec_dict.get("scale_bits", 8)
    return elem_bits(fmt) + float(scale) / float(block)


def conv_macs(module, out_shape):
    """MACs for one forward of `module` given its output shape."""
    if isinstance(module, nn.Linear):
        return int(module.in_features * module.out_features)
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        spatial = 1
        for d in out_shape[2:]:
            spatial *= int(d)
        k = 1
        for d in module.kernel_size:
            k *= int(d)
        return int(module.in_channels // module.groups * module.out_channels
                   * k * spatial)
    return 0


# =============================================================================
# Module bindings and the swap primitive
# =============================================================================

def get_parent(model, full_name):
    """Resolve 'a.b.c' to (module a.b, 'c'); (None, None) if unreachable."""
    parent = model
    parts = full_name.split(".")
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None, None
        parent = getattr(parent, p)
    return parent, parts[-1]


def resolve_bindings(model, names, types):
    """Map each requested layer name to every place it is bound in `model`.

    Resolved against the *built* model, after MXQuantizer has replaced layers,
    wrapped param-free ops and installed hooks — `_wrap_act_layers` inserts an
    `.inner` level into module paths, so a path derived from the original model
    can be stale by the time we want to swap.

    Bindings are collected by module *identity*, not by name: a module reachable
    as both `body` and `body_again` must be swapped at both sites even when the
    caller only asked for `body`, or the probe runs for one call site while the
    original still runs for the other — a half-demoted layer whose score is
    quietly too optimistic.

    Returns:
        dict: name -> {"module": m, "sites": [(parent, leaf), ...],
                       "paths": [...], "aliases": [other names for the same module]}
    """
    wanted = set(names)

    # Pass 1: which module object does each requested name point at?
    by_name = {}
    for full_name, module in model.named_modules(remove_duplicate=False):
        if not isinstance(module, types):
            continue
        name = clean_name(full_name)
        if name in wanted and name not in by_name:
            by_name[name] = module

    # Two requested names for one module: probe it once, under the first name.
    owner = {}
    found = {}
    for name, module in by_name.items():
        key = id(module)
        if key in owner:
            found[owner[key]]["aliases"].append(name)
            continue
        owner[key] = name
        found[name] = {"module": module, "sites": [], "paths": [], "aliases": []}

    # Pass 2: every binding of those module objects, whatever it is called.
    for full_name, module in model.named_modules(remove_duplicate=False):
        name = owner.get(id(module))
        if name is None:
            continue
        parent, leaf = get_parent(model, full_name)
        if parent is None:
            continue
        found[name]["sites"].append((parent, leaf))
        found[name]["paths"].append(full_name)

    return found


class swap_module:
    """Temporarily bind `probe` everywhere `entry`'s module is bound.

    Restores the exact original objects on exit, then asserts the restore
    actually took — a probe left installed would silently poison every
    subsequent layer's score.
    """

    def __init__(self, entry, probe):
        self.entry = entry
        self.probe = probe
        self.orig = entry["module"]

    def __enter__(self):
        for parent, leaf in self.entry["sites"]:
            setattr(parent, leaf, self.probe)
        return self.probe

    def __exit__(self, *exc):
        for parent, leaf in self.entry["sites"]:
            setattr(parent, leaf, self.orig)
        for parent, leaf in self.entry["sites"]:
            assert getattr(parent, leaf) is self.orig, (
                f"failed to restore module at {leaf} after probing")
        return False


# =============================================================================
# Rank correlation (no scipy dependency)
# =============================================================================

def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman rank correlation of two equal-length sequences."""
    pairs = [(x, y) for x, y in zip(a, b)
             if x is not None and y is not None
             and not math.isnan(x) and not math.isnan(y)
             and not math.isinf(x) and not math.isinf(y)]
    if len(pairs) < 3:
        return None
    ra = _ranks([p[0] for p in pairs])
    rb = _ranks([p[1] for p in pairs])
    n = len(ra)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da > 0 and db > 0 else None


# =============================================================================
# Scorer registry
# =============================================================================

SCORERS = {}


def register(name):
    """Register a scorer under `name`. The plug-in point for new algorithms."""
    def deco(fn):
        SCORERS[name] = fn
        return fn
    return deco


def get_scorer(name):
    if name not in SCORERS:
        raise ValueError(
            f"Unknown sensitivity scorer {name!r}. Available: "
            f"{', '.join(sorted(SCORERS))}")
    return SCORERS[name]


# =============================================================================
# OAT — one layer at a time, measured at the network output
# =============================================================================

def _run_reference(model, batches, forward_fn, output_fn):
    """Run the reference net once, keeping its flattened outputs on CPU."""
    outs = []
    with torch.no_grad():
        for batch in batches:
            y = default_forward(model, batch, forward_fn)
            outs.append([t.detach().float().cpu() for t in output_fn(y)])
    return outs


def check_determinism(model, batch, ref_out, forward_fn, output_fn):
    """Re-run one batch and require bit-identical outputs.

    OAT is a difference of two forward passes. If the model is nondeterministic
    (dropout left on, a stateful/recurrent buffer, nondeterministic kernels),
    the differences are noise and the resulting ranking looks entirely
    plausible while being meaningless. Better to refuse.
    """
    with torch.no_grad():
        y = default_forward(model, batch, forward_fn)
    got = [t.detach().float().cpu() for t in output_fn(y)]
    if len(got) != len(ref_out):
        return False, "output structure changed between two identical runs"
    for a, b in zip(ref_out, got):
        if a.shape != b.shape or not torch.equal(a, b):
            return False, "outputs differ between two identical runs"
    return True, None


def score_oat(ref_model, entries, batches, *, build_probe, forward_fn=None,
              output_fn=None, log=None, verbose=True):
    """One-at-a-time scoring against the network output.

    For each candidate: swap in a lower-precision copy of that layer alone, run
    the calibration batches, and measure how far the network output moved. The
    rest of the network stays exactly as the reference has it, so the score is
    the *marginal* cost of demoting that one layer in the net we actually ship.

    Args:
        ref_model: the reference network, already built (all layers at the
            reference precision, or plain FP32 for an absolute reference).
        entries: output of `resolve_bindings` — name -> module/sites/paths.
        batches: materialized calibration batches.
        build_probe: callable(name, orig_module) -> nn.Module, the demoted layer.
        forward_fn / output_fn: how to run a batch and reduce its return value.
        verbose: print progress and the half-split rank correlation.

    Returns:
        dict: name -> {"sensitivity", "status", "metrics", "n_calls"}, plus a
        "__meta__" entry carrying the determinism check and half-split Spearman.
    """
    output_fn = output_fn or flatten_outputs
    was_training = ref_model.training
    ref_model.eval()

    try:
        ref_out = _run_reference(ref_model, batches, forward_fn, output_fn)
        ok, why = check_determinism(ref_model, batches[0], ref_out[0],
                                    forward_fn, output_fn)
        if not ok:
            raise RuntimeError(
                f"OAT scoring needs a deterministic forward pass, but {why}. "
                f"Check for dropout or BatchNorm in train mode, stateful buffers "
                f"updated during forward, or nondeterministic kernels.")

        results = {}
        halves = ([], [])       # (even-batch scores, odd-batch scores) per layer
        names = list(entries)
        for idx, name in enumerate(names, 1):
            entry = entries[name]
            probe = build_probe(name, entry["module"])
            if probe is None:
                results[name] = {"sensitivity": None, "status": STATUS_NO_CHANGE,
                                 "metrics": {}, "n_calls": 0}
                halves[0].append(None)
                halves[1].append(None)
                continue

            calls = {"n": 0}

            def _count(mod, inp, out):
                calls["n"] += 1

            handle = probe.register_forward_hook(_count)
            acc, acc_a, acc_b = ErrAcc(), ErrAcc(), ErrAcc()
            try:
                with swap_module(entry, probe), torch.no_grad():
                    for i, batch in enumerate(batches):
                        y = default_forward(ref_model, batch, forward_fn)
                        got = output_fn(y)
                        half = acc_a if i % 2 == 0 else acc_b
                        for ref_t, got_t in zip(ref_out[i], got):
                            got_c = got_t.detach().float().cpu()
                            acc.update(ref_t, got_c)
                            half.update(ref_t, got_c)
            finally:
                handle.remove()

            status, metrics = acc.summary()
            if calls["n"] == 0:
                # The layer never ran during calibration. Detected by counter,
                # not by "the output did not change" — those look identical here
                # but mean opposite things.
                status, metrics = STATUS_UNREACHED, {}
            sens = (sqnr_to_sensitivity(metrics.get("sqnr_db"))
                    if status == STATUS_OK else None)
            results[name] = {"sensitivity": sens, "status": status,
                             "metrics": metrics, "n_calls": calls["n"]}
            halves[0].append(sqnr_to_sensitivity(acc_a.summary()[1].get("sqnr_db")))
            halves[1].append(sqnr_to_sensitivity(acc_b.summary()[1].get("sqnr_db")))

            if verbose and log is not None and idx % 25 == 0:
                log(f"  scored {idx}/{len(names)} layers")

        half_rho = spearman(halves[0], halves[1]) if len(batches) > 1 else None
        results["__meta__"] = {"determinism_check": "pass",
                               "half_split_spearman": half_rho,
                               "n_batches": len(batches),
                               "n_probes": len(names)}
        return results
    finally:
        if was_training:
            ref_model.train()


# =============================================================================
# Zero-compute and offline scorers
# =============================================================================

def score_weight_only(entries, build_probe):
    """Weight-only SQNR: quantize each layer's weights and compare.

    No activations, no forward pass. The cheapest possible signal and the
    fallback for layers a calibration pass never reaches.
    """
    results = {}
    for name, entry in entries.items():
        mod = entry["module"]
        w = getattr(mod, "weight", None)
        if w is None:
            results[name] = {"sensitivity": None, "status": STATUS_ZERO_OUTPUT,
                             "metrics": {}}
            continue
        probe = build_probe(name, mod)
        w = w.data.float()
        with torch.no_grad():
            w_q = _quantize_like(probe, w)
        acc = ErrAcc()
        acc.update(w, w_q)
        status, metrics = acc.summary()
        results[name] = {
            "sensitivity": sqnr_to_sensitivity(metrics.get("sqnr_db"))
            if status == STATUS_OK else None,
            "status": status, "metrics": metrics}
    return results


def _quantize_like(probe, w):
    """Fake-quantize `w` with the probe layer's own weight spec."""
    from microxcaling.mx.mx_ops import quantize_mx_op
    sp = probe.mx_specs
    return quantize_mx_op(w, sp, elem_format=sp["w_elem_format"], axes=[-1],
                          round=sp.get("round_mx_output", "nearest"))


def score_from_stats(stats, names):
    """Reuse the per-layer out-SQNR that collect_stats already measured.

    Free: `quant_stats.json` is written on every run with collect_stats enabled,
    and its isolated output error is the same quantity the existing auto_mixed
    scorer computes. Isolated, so it does not see error propagation — cheap
    first pass, not a substitute for OAT.
    """
    layers = stats.get("layers", stats) if isinstance(stats, dict) else {}
    by_clean = {clean_name(k): v for k, v in layers.items()}
    results = {}
    for name in names:
        entry = by_clean.get(clean_name(name))
        if entry is None:
            results[name] = {"sensitivity": None, "status": STATUS_UNREACHED,
                             "metrics": {}}
            continue
        err = (entry.get("output_error") or {}).get("isolated") or {}
        sqnr = err.get("sqnr_db")
        why = {}
        w = entry.get("weight") or {}
        a = entry.get("activation") or {}
        if w.get("underflow_rate") is not None:
            why["w_underflow"] = w["underflow_rate"]
        if a.get("underflow_rate") is not None:
            why["a_underflow"] = a["underflow_rate"]
        dyn = ((a.get("block") or {}).get("dyn_range") or {})
        if dyn.get("p50") is not None:
            why["a_dyn_p50"] = dyn["p50"]
        if (w.get("error") or {}).get("sqnr_db") is not None:
            why["w_sqnr_db"] = w["error"]["sqnr_db"]
        if (a.get("error") or {}).get("sqnr_db") is not None:
            why["a_sqnr_db"] = a["error"]["sqnr_db"]
        sens = sqnr_to_sensitivity(sqnr)
        results[name] = {
            "sensitivity": sens,
            "status": STATUS_OK if sens is not None else STATUS_ZERO_OUTPUT,
            "metrics": {"sqnr_db": sqnr}, "why": why}
    return results


def score_from_file(path, names):
    """Read scores produced elsewhere.

    Accepts either a bare mapping {layer: score} or a full sensitivity.json
    artifact. This is the offline half of the black box: any external algorithm
    that can emit per-layer numbers plugs in here, including the sweep
    generated by mixed_precision_sweep.py.
    """
    import json
    with open(path) as f:
        blob = json.load(f)

    table = {}
    if isinstance(blob, dict) and "layers" in blob and isinstance(blob["layers"], list):
        for row in blob["layers"]:
            table[clean_name(row["name"])] = row.get("sensitivity")
    elif isinstance(blob, dict):
        for k, v in blob.items():
            table[clean_name(k)] = v.get("sensitivity") if isinstance(v, dict) else v
    else:
        raise ValueError(f"{path}: expected an object mapping layer names to scores")

    results = {}
    for name in names:
        val = table.get(clean_name(name))
        if val is None:
            results[name] = {"sensitivity": None, "status": STATUS_UNREACHED,
                             "metrics": {}}
        else:
            results[name] = {"sensitivity": float(val), "status": STATUS_OK,
                             "metrics": {}}
    return results


def load_callable(spec):
    """Import 'package.module:function' — the plug-in escape hatch."""
    import importlib
    mod_path, _, fn_name = spec.partition(":")
    if not fn_name:
        raise ValueError(f"scorer callable must be 'package.module:function', got {spec!r}")
    sys.path.insert(0, os.getcwd())
    try:
        return getattr(importlib.import_module(mod_path), fn_name)
    finally:
        sys.path.pop(0)


# =============================================================================
# Isolated scoring — each layer replayed on its own captured inputs
# =============================================================================

def score_isolated(ref_model, entries, batches, *, build_probe, forward_fn=None,
                   capture_chunk=32, log=None):
    """Score each layer on its own inputs, without upstream error propagation.

    Two phases. Capture the reference inputs and outputs of a group of layers
    during one pass over the calibration batches, then replay those inputs
    through a demoted copy of each layer. Cheap in compute — one pass covers
    every layer — but the score ignores how the error travels to the network
    output, which is what OAT measures.

    `capture_chunk` bounds memory: the captured tensors are the expensive part
    (inputs *and* outputs for every layer in the chunk, for every batch), so
    layers are processed `capture_chunk` at a time, re-running the batches per
    chunk. Without it a real network at a realistic batch count needs tens of
    gigabytes of host RAM.
    """
    names = list(entries)
    results = {}
    was_training = ref_model.training
    ref_model.eval()

    try:
        for start in range(0, len(names), max(1, capture_chunk)):
            chunk = names[start:start + max(1, capture_chunk)]
            captured = {n: [] for n in chunk}      # name -> [(in, out), ...] FIFO

            handles = []
            for name in chunk:
                def make_hook(n):
                    def hook(mod, inp, out):
                        # Offloaded immediately: holding these on the accelerator
                        # for every layer and batch is what blows up memory.
                        captured[n].append((inp[0].detach().cpu(),
                                            out.detach().cpu()))
                    return hook
                handles.append(entries[name]["module"].register_forward_hook(
                    make_hook(name)))

            with torch.no_grad():
                for batch in batches:
                    default_forward(ref_model, batch, forward_fn)
            for h in handles:
                h.remove()

            for name in chunk:
                pairs = captured.pop(name)
                if not pairs:
                    results[name] = {"sensitivity": None,
                                     "status": STATUS_UNREACHED, "metrics": {}}
                    continue
                mod = entries[name]["module"]
                device = next(iter(mod.parameters()), torch.tensor(0.0)).device
                probe = build_probe(name, mod)
                if probe is None:
                    results[name] = {"sensitivity": None,
                                     "status": STATUS_NO_CHANGE, "metrics": {}}
                    continue
                probe = probe.to(device)
                acc = ErrAcc()
                with torch.no_grad():
                    for x_cpu, y_cpu in pairs:
                        y_q = probe(x_cpu.to(device)).detach().float().cpu()
                        acc.update(y_cpu.float(), y_q)
                        del y_q
                status, metrics = acc.summary()
                results[name] = {
                    "sensitivity": sqnr_to_sensitivity(metrics.get("sqnr_db"))
                    if status == STATUS_OK else None,
                    "status": status, "metrics": metrics}

            if log is not None:
                log(f"  isolated: {min(start + len(chunk), len(names))}/{len(names)} layers")

        return results
    finally:
        if was_training:
            ref_model.train()


# =============================================================================
# Registry adapters — one call shape for every scorer
# =============================================================================

class ScoreContext:
    """Everything a scorer may need. Most scorers use a small subset.

    Attributes:
        ref_model: reference network, already built at the reference precision.
        entries: resolve_bindings() output for the candidate layers.
        batches: materialized calibration batches (may be None).
        build_probe: callable(name, module) -> demoted nn.Module.
        forward_fn / output_fn: how to run a batch, how to reduce its result.
        options: the config's scorer block (paths, chunking, verbosity...).
        log: one-argument callable for progress messages.
    """

    def __init__(self, ref_model=None, entries=None, batches=None,
                 build_probe=None, forward_fn=None, output_fn=None,
                 options=None, log=None):
        self.ref_model = ref_model
        self.entries = entries or {}
        self.batches = batches
        self.build_probe = build_probe
        self.forward_fn = forward_fn
        self.output_fn = output_fn or flatten_outputs
        self.options = options or {}
        self.log = log

    @property
    def names(self):
        return list(self.entries)


@register("oat_output")
def _scorer_oat(ctx):
    if not ctx.batches:
        raise ValueError("scorer 'oat_output' needs calibration data")
    return score_oat(ctx.ref_model, ctx.entries, ctx.batches,
                     build_probe=ctx.build_probe, forward_fn=ctx.forward_fn,
                     output_fn=ctx.output_fn, log=ctx.log)


@register("weight_only")
def _scorer_weight_only(ctx):
    return score_weight_only(ctx.entries, ctx.build_probe)


@register("from_stats")
def _scorer_from_stats(ctx):
    import json
    path = ctx.options.get("path")
    if path is None:
        raise ValueError("scorer 'from_stats' needs options.path "
                         "(usually <save_dir>/quant_stats.json)")
    with open(path) as f:
        stats = json.load(f)
    return score_from_stats(stats, ctx.names)


@register("from_file")
def _scorer_from_file(ctx):
    path = ctx.options.get("path")
    if path is None:
        raise ValueError("scorer 'from_file' needs options.path")
    return score_from_file(path, ctx.names)


@register("callable")
def _scorer_callable(ctx):
    target = ctx.options.get("target")
    if target is None:
        raise ValueError("scorer 'callable' needs options.target "
                         "('package.module:function')")
    return load_callable(target)(ctx)


@register("isolated_sqnr")
def _scorer_isolated(ctx):
    if not ctx.batches:
        raise ValueError("scorer 'isolated_sqnr' needs calibration data")
    return score_isolated(ctx.ref_model, ctx.entries, ctx.batches,
                          build_probe=ctx.build_probe, forward_fn=ctx.forward_fn,
                          capture_chunk=int(ctx.options.get("capture_chunk", 32)),
                          log=ctx.log)
