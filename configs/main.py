"""
main.py - run a case through both sides and compare tap by tap.

    python main.py -m conv3x3
    python main.py -m warp
    python main.py -m all
    python main.py -m conv --config mx_config_dof_npe_int6.json --backend python

Integer taps must match exactly; float taps report max |delta|. Taps are
compared in ALG order and the first integer tap that differs is named - every
later stage is downstream of it.

Verdict per case: bit-exact (every shared tap equal), float-diff (integer taps
equal, a float tap differs - reported, not failed), MISMATCH (an integer tap or
a shape differs).

Exit code: 0 all bit-exact, 1 a mismatch, 2 HW side not implemented, 3 only
float differences.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cases as C
import main_ALG as ALG
import main_HW as HW
from taps import is_int


def _unravel(flat, shape):
    """Flat index -> per-dim index."""
    idx = []
    for n in reversed(tuple(shape)):
        idx.append(flat % n)
        flat //= n
    return tuple(reversed(idx))


def compare(alg, hw):
    """Shared taps, ALG order -> rows (name, verdict, n_bad, first_bad, max_d)."""
    rows = []
    for k, a in alg.items():
        if k not in hw:
            continue
        h = hw[k]
        if a.shape != h.shape:
            rows.append((k, "SHAPE", -1, f"{tuple(a.shape)} vs {tuple(h.shape)}", 0))
            continue
        d = (a.double() - h.double()).abs()
        bad = d > 0
        n = int(bad.sum())
        first = _unravel(int(bad.flatten().nonzero()[0]), a.shape) if n else "-"
        if is_int(a):
            verdict = "bit-exact" if n == 0 else "MISMATCH"
        else:
            verdict = "float" if n == 0 else "float-diff"
        rows.append((k, verdict, n, first, float(d.max()) if d.numel() else 0.0))
    return rows


def report(name, rows, alg, hw):
    print(f"\n=== {name} ===")
    print(f"  {'tap':<12} {'verdict':<11} {'n_bad':>6} {'first':>16} {'max_d':>12}")
    for k, v, n, first, md in rows:
        print(f"  {k:<12} {v:<11} {n:>6} {str(first):>16} {md:>12.6g}")
    only_alg = [k for k in alg if k not in hw]
    only_hw = [k for k in hw if k not in alg]
    if only_alg:
        print(f"  ALG only: {', '.join(only_alg)}")
    if only_hw:
        print(f"  HW only:  {', '.join(only_hw)}")
    bad = [r for r in rows if r[1] in ("MISMATCH", "SHAPE")]
    if bad:
        print(f"  -> first diverging tap: {bad[0][0]}")
        return "mismatch"
    if any(r[1] == "float-diff" for r in rows):
        return "float-diff"
    return "bit-exact"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-m", default="all", help="case name, op (conv/convt/warp) or all")
    ap.add_argument("--config", default=ALG.DEFAULT_CONFIG,
                    help="framework mx_config.json (default mx_config_dof.json)")
    ap.add_argument("--layer", default=None,
                    help="entry of the config's layers to use (its group/kind)")
    ap.add_argument("--backend", default=None,
                    help="override xblock_accum.backend (python / triton)")
    args = ap.parse_args()

    cfg = ALG.load_config(args.config, args.layer, args.backend)
    results = {}
    for name in C.select(args.m):
        case = C.build(name)
        alg = ALG.run(case, cfg)
        try:
            hw = HW.run(case)
        except NotImplementedError as e:
            print(f"\n=== {name} ===\n  HW not implemented: {e}  "
                  f"(ALG: {len(alg)} taps)")
            results[name] = None
            continue
        results[name] = report(name, compare(alg, hw), alg, hw)

    ok = [k for k, v in results.items() if v == "bit-exact"]
    fd = [k for k, v in results.items() if v == "float-diff"]
    bad = [k for k, v in results.items() if v == "mismatch"]
    skip = [k for k, v in results.items() if v is None]
    print(f"\n{len(ok)}/{len(results)} bit-exact"
          + (f", float-diff: {', '.join(fd)}" if fd else "")
          + (f", mismatched: {', '.join(bad)}" if bad else "")
          + (f", HW missing: {len(skip)}" if skip else ""))
    sys.exit(1 if bad else 2 if skip else 3 if fd else 0)


if __name__ == "__main__":
    main()
