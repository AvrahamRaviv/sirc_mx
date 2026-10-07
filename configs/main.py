"""
main.py - run a case through both sides and compare.

Triggers `main_ALG.run` (MX lib) and `main_HW.run` (HW implementation) on the
same model, the same weights, the same test vectors and the same config, then
compares them tap by tap.

    python main.py -m conv1x1_ramp           # one case
    python main.py -m corner                 # the 12 corner cases
    python main.py -m all --dump taps.pt

`main_HW` is still a skeleton, so this reports "HW not implemented" for every
case and exits 2. The ALG side already runs - `python main_ALG.py -m all` works
today - so this file is wired and waiting for the HW bodies.

Exit code: 0 all taps bit-exact, 1 a mismatch, 2 the HW side did not run.

Comparison rules
----------------
Integer taps (a_int / a_exp / w_int / w_exp / out_int / out_exp / acc_*, and
out_code / out_clip from the static fixed-point output stage) are compared
**exactly**. A single differing element is a failure, and the report
names the stage, the count, and the first offending flat index.

Float taps (/out, model/out) are compared by max abs delta and SQNR, because
the ALG side carries them in FP32 - until the HW side reports its integer
output, a float comparison is the best available there.

Taps only one side emits (the HW-only accumulator stages) are listed as
HW-only, not as a mismatch.
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import models as M
import main_ALG as ALG
import main_HW as HW

_INT_SUFFIXES = ("a_int", "a_exp", "w_int", "w_exp", "out_int", "out_exp",
                 "acc_raw", "acc_shf", "acc_sat", "sat_cnt", "sat_total",
                 # static fixed-point output stage: the word itself is an
                 # integer, so it is the bit-exact comparable there too.
                 "out_code", "out_clip")


# =============================================================================
# Compare
# =============================================================================

def _is_int_tap(name):
    return name.rsplit("/", 1)[-1] in _INT_SUFFIXES


def _sqnr_db(ref, other):
    err = (ref - other).double()
    num = ref.double().pow(2).sum()
    den = err.pow(2).sum()
    if den == 0:
        return float("inf")
    return float(10 * torch.log10(num / den))


def compare_tap(name, alg, hw):
    """Compare one tap. Returns a row dict; `exact` is None for float taps."""
    row = {"tap": name, "shape": tuple(alg.shape), "exact": None,
           "n_bad": 0, "first_bad": None, "max_delta": 0.0, "sqnr": None,
           "note": ""}

    if tuple(alg.shape) != tuple(hw.shape):
        row["note"] = f"shape mismatch: ALG {tuple(alg.shape)} vs HW {tuple(hw.shape)}"
        row["exact"] = False
        return row

    a, h = alg.flatten(), hw.flatten()
    if _is_int_tap(name):
        bad = (a != h)
        row["n_bad"] = int(bad.sum())
        row["exact"] = row["n_bad"] == 0
        if row["n_bad"]:
            row["first_bad"] = int(bad.nonzero()[0])
            d = (a.long() - h.long()).abs()
            row["max_delta"] = int(d.max())
    else:
        d = (a.double() - h.double()).abs()
        row["max_delta"] = float(d.max())
        row["sqnr"] = _sqnr_db(a, h)
        row["n_bad"] = int((d > 0).sum())
        if row["n_bad"]:
            row["first_bad"] = int((d > 0).nonzero()[0])

    return row


def compare(alg_taps, hw_taps):
    """Compare two tap dicts. Returns (rows, alg_only, hw_only)."""
    shared = [k for k in alg_taps if k in hw_taps]
    rows = [compare_tap(k, alg_taps[k], hw_taps[k]) for k in sorted(shared)]
    alg_only = sorted(k for k in alg_taps if k not in hw_taps)
    hw_only = sorted(k for k in hw_taps if k not in alg_taps)
    return rows, alg_only, hw_only


# =============================================================================
# Report
# =============================================================================

def report(case_name, rows, alg_only, hw_only):
    """Print the per-tap table. Returns True if every integer tap matched."""
    print(f"\n=== {case_name} ===")
    hdr = (f"  {'tap':<26} {'shape':<18} {'verdict':<12} "
           f"{'n_bad':>7} {'first':>7} {'max_d':>12} {'SQNR dB':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    ok = True
    for r in rows:
        if r["exact"] is True:
            verdict = "bit-exact"
        elif r["exact"] is False:
            verdict = "MISMATCH"
            ok = False
        else:
            verdict = "float" if r["max_delta"] == 0 else "float-diff"
        sqnr = "-" if r["sqnr"] is None else (
            "inf" if r["sqnr"] == float("inf") else f"{r['sqnr']:.2f}")
        first = "-" if r["first_bad"] is None else str(r["first_bad"])
        print(f"  {r['tap']:<26} {str(r['shape']):<18} {verdict:<12} "
              f"{r['n_bad']:>7} {first:>7} {r['max_delta']:>12.6g} {sqnr:>9}"
              + (f"   {r['note']}" if r["note"] else ""))

    if hw_only:
        print(f"  HW-only taps (no ALG equivalent): {', '.join(hw_only)}")
    if alg_only:
        print(f"  ALG-only taps (HW did not emit):  {', '.join(alg_only)}")

    # The first failing stage is the one worth looking at: everything after it
    # is downstream of the same root cause.
    bad = [r for r in rows if r["exact"] is False]
    if bad:
        print(f"  -> first diverging stage: {bad[0]['tap']} "
              f"({bad[0]['n_bad']} elements, first at flat index "
              f"{bad[0]['first_bad']})")
    return ok


# =============================================================================
# Run
# =============================================================================

def run_case(case, config, verbose=False):
    """Run both sides on one case. Returns (ok, alg_taps, hw_taps).

    `ok` is None when the HW side could not run at all - distinct from False,
    which means it ran and disagreed.
    """
    alg_taps = ALG.run(case, config, verbose=verbose)
    try:
        hw_taps = HW.run(case, config, verbose=verbose)
    except NotImplementedError as e:
        print(f"\n=== {case['meta']['case']} ===")
        print(f"  HW side not available: {e}")
        print(f"  ALG side ran: {len(alg_taps)} taps, "
              f"model/out {tuple(alg_taps['model/out'].shape)}")
        return None, alg_taps, None

    rows, alg_only, hw_only = compare(alg_taps, hw_taps)
    return report(case["meta"]["case"], rows, alg_only, hw_only), alg_taps, hw_taps


def main():
    parser = M.add_model_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--config", default=None,
                        help="config JSON driving both sides (default: built-in)")
    parser.add_argument("--dump", default=None,
                        help="save both sides' taps to a .pt file")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = ALG.load_config(args.config, bs=args.bs)
    results, dump = {}, {}

    for case in M.cases_from_args(args):
        name = case["meta"]["case"]
        ok, alg_taps, hw_taps = run_case(case, config, verbose=args.verbose)
        results[name] = ok
        dump[name] = {"alg": alg_taps, "hw": hw_taps, "meta": case["meta"]}

    # Summary
    n = len(results)
    exact = [k for k, v in results.items() if v is True]
    diff = [k for k, v in results.items() if v is False]
    skipped = [k for k, v in results.items() if v is None]
    print(f"\n{'-' * 60}")
    print(f"{len(exact)}/{n} bit-exact", end="")
    if diff:
        print(f", {len(diff)} mismatched: {', '.join(diff)}", end="")
    if skipped:
        print(f", {len(skipped)} not compared (HW side missing)", end="")
    print()

    if args.dump:
        torch.save({"config": config, "cases": dump}, args.dump)
        print(f"wrote {args.dump}")

    if skipped:
        sys.exit(2)
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
