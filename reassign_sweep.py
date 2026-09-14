#!/usr/bin/env python3
"""Re-assign an existing sensitivity.json under different ladder shapes.

Zero forward passes: `mx_assign.assign` is a pure function of the scored rows,
and `plan_mixed_precision` already wrote each layer's MACs into the artifact so
the MAC-weighted average can be recomputed offline. Use this to shop for a
ladder before spending GPU on a re-plan.

    python reassign_sweep.py <save_dir>/sensitivity.json configs/<config>.json
"""
import argparse
import copy
import json
import re
import sys

import mx_assign as A


def freeze_explicit_layers(config, auto_cfg, groups):
    """Mirror of MXQuantizer._freeze_explicit_layers.

    A `{"name": ..., "group": ...}` entry in `layers` is a hand-written decision
    the ladder must not overwrite, so `plan_mixed_precision` converts it into a
    pin before assigning. That happens in memory, not on disk, so anything
    re-assigning offline has to redo it — otherwise those layers fall into the
    pool and get handed a ladder rung they were explicitly kept out of.
    """
    frozen, table = {}, dict(groups)
    for entry in config.get("layers", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") in ("act_quant", "out_quant"):
            continue
        name = entry.get("name")
        if not name:
            continue
        if "mx_specs" in entry:
            gname = "fixed_" + re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
            table[gname] = dict(entry["mx_specs"])
            frozen[name] = gname
        elif "group" in entry:
            gname = entry["group"]
            if gname not in groups:
                raise ValueError(f"layer '{name}' references undefined group '{gname}'")
            table.setdefault(gname, dict(groups[gname]))
            frozen[name] = gname
    if not frozen:
        return auto_cfg, table
    pins = dict(frozen)
    pins.update(auto_cfg.get("pins") or {})
    return dict(auto_cfg, pins=pins), table


def evaluate(rows, costs, auto_cfg, groups):
    assignments, notes = A.assign(rows, auto_cfg, groups)
    flat, extra = A.resolve_groups(assignments, groups, deploy_group=None)
    return A.cost_summary(flat, groups, extra, costs), notes


def sweep(artifact, config, extra_rungs=None):
    rows = artifact["layers"]
    costs = {r["name"]: r.get("cost") or {} for r in rows}
    base, groups = freeze_explicit_layers(config, config["auto_mixed"],
                                          dict(config["groups"]))
    pins = base.get("pins") or {}
    if pins:
        print(f"{len(pins)} layer(s) pinned by an explicit \"group\" in the config; "
              f"they keep that group and stay out of the ladder pool.")

    ranked = [r for r in rows
              if r["name"] not in pins
              and r.get("status") not in A.UNMEASURED
              and r.get("sensitivity") is not None]
    ranked.sort(key=lambda r: r["sensitivity"])
    print(f"{len(rows)} layers in artifact, {len(ranked)} ranked "
          f"(the rest are pinned or unmeasured and stay at their fixed group)\n")

    if ranked:
        lo, hi = ranked[0]["sensitivity"], ranked[-1]["sensitivity"]
        mid = ranked[len(ranked) // 2]["sensitivity"]
        print(f"sensitivity spread: min {lo:.2f}  median {mid:.2f}  max {hi:.2f}")
        print("(threshold cutoffs below are read against this scale)\n")

    trials = []

    def add(label, **over):
        cfg = copy.deepcopy(base)
        cfg.update(over)
        trials.append((label, cfg))

    ladders = [["int4", "int6", "int8"], ["int6", "int8"]]
    if extra_rungs:
        ladders.append(extra_rungs)
    for ladder in ladders:
        tag = "/".join(x.replace("int", "") for x in ladder)
        if len(ladder) == 3:
            for q in ((0.10, 0.20, 0.70), (0.20, 0.30, 0.50),
                      (0.30, 0.30, 0.40), (0.40, 0.30, 0.30)):
                add(f"{tag} quantile {int(q[0]*100)}/{int(q[1]*100)}/{int(q[2]*100)}",
                    ladder=ladder, strategy="quantile",
                    quantile=dict(zip(ladder, q)))
        else:
            for q in ((0.20, 0.80), (0.40, 0.60), (0.60, 0.40)):
                add(f"{tag} quantile {int(q[0]*100)}/{int(q[1]*100)}",
                    ladder=ladder, strategy="quantile",
                    quantile=dict(zip(ladder, q)))

    # threshold cuts are UPPER BOUNDS on the rung they name (mx_assign.py:229),
    # and a rung absent from the dict is taken immediately -- so the cuts belong
    # on the LOW rungs and the top rung is the fallthrough.
    if ranked:
        for lo_f, hi_f in ((0.2, 0.5), (0.35, 0.65), (0.5, 0.8)):
            lo = ranked[int(len(ranked) * lo_f)]["sensitivity"]
            hi = ranked[int(len(ranked) * hi_f)]["sensitivity"]
            add(f"4/6/8 threshold int4<={lo:.1f} int6<={hi:.1f}",
                ladder=["int4", "int6", "int8"], strategy="threshold",
                threshold={"int4": round(lo, 2), "int6": round(hi, 2)})

    width = max(len(t[0]) for t in trials) + 2
    print(f"{'ladder':<{width}} {'avg bits':>9} {'MAC-wt bits':>12}  counts")
    print("-" * (width + 26 + 30))
    for label, cfg in trials:
        try:
            summary, _ = evaluate(rows, costs, cfg, groups)
        except A.AssignError as exc:
            print(f"{label:<{width}} {'-':>9} {'-':>12}  rejected: {exc}")
            continue
        counts = ", ".join(f"{g}:{n}" for g, n in summary["per_rung_counts"].items())
        mw = summary["macs_weighted_avg_bits"]
        print(f"{label:<{width}} {summary['avg_bits']:>9.2f} "
              f"{(f'{mw:.2f}' if mw is not None else '-'):>12}  {counts}")

    top = base["ladder"][-1]
    all_top = {r["name"]: pins.get(r["name"], top) for r in rows}
    ref = A.cost_summary(all_top, groups, {}, costs)
    print("-" * (width + 26 + 30))
    print(f"{'all ' + top + ' (reference)':<{width}} {ref['avg_bits']:>9.2f} "
          f"{ref['macs_weighted_avg_bits']:>12.2f}")
    print("\nBits are per weight element and include the shared block scale "
          "(scale_bits/block_size).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifact", help="sensitivity.json from plan_mixed_precision")
    ap.add_argument("config", help="the mx_config.json the plan was run from")
    ap.add_argument("--rungs", nargs="+", default=None,
                    help="extra ladder to try, low to high (e.g. --rungs int4 int5 int6 int8)")
    args = ap.parse_args(argv)
    sweep(json.load(open(args.artifact)), json.load(open(args.config)), args.rungs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
