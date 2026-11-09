"""
main.py - run a case through both sides and compare tap by tap.

    python main.py -m conv3x3
    python main.py -m warp
    python main.py -m all
    python main.py -m conv --config mx_config_dof_npe_int6.json --backend python
    python main.py -m conv3x3_p0 --input both     # float input vs integer input

--input float : x as is (what the MX lib sees in training)
--input int   : x snapped to the grid 2^-x_frac first - an integer tensor, as
                the NPE receives it. Both sides get the same tensor.
--input both  : run both and print how much of HW-ALG is the input rounding.

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

import torch

import cases as C
import main_ALG as ALG
import main_HW as HW
import sim_config as SC
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


# ALG's out_pre is float32: its acc in code units carries <= ~2e-5 of float
# error (measured, phase A), so the acc stage counts differences above this.
WARP_ACC_TOL = 1e-3


def _row(stage, q, a, h, tol=0.0, mask=None):
    """One stage of warp_stages: compare a and h (same shape) where mask."""
    a, h = a.double(), h.double()
    if mask is not None:
        a, h = a[mask], h[mask]
    n = a.numel()
    d = (a - h).abs()
    return {"stage": stage, "q": q, "n": n, "n_diff": int((d > tol).sum()),
            "max_d": float(d.max()) if n else 0.0}


def warp_stages(case, alg, hw):
    """The warp, ALG vs HW, per stage where both sides have one, in units both
    share. Rows {stage, q (the HW question it bears on), n, n_diff, max_d}:
      ref code     ALG int8 ref_code vs their uint8 code (padding dropped)
      grid px      sample position each grid word stands for: ALG mesh +
                   grid_code / 2^frac_cfg vs theirs hw_grid_code / 2^hw_frac_bits
      acc interior the bilinear sum in ref code units, ALG out_pre / ref_scale
      acc border   vs their hw_acc / 2^(frx+fry); interior = all 4 corners of
                   ALG's position inside the frame, border = the rest
    `out` is in the generic tap compare. ALG's int8 out_code has no HW
    counterpart: their PCU is bypassed (HW Q7)."""
    B, _, H, W = case["x"].shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    mesh = torch.stack([xx, yy])                               # [2, H, W]
    fa = hw["frac_cfg"].double().view(1, 2, 1, 1)
    fh = hw["hw_frac_bits"].double().view(1, 2, 1, 1)
    a_pos = mesh + alg["grid_code"].double() / 2.0 ** fa       # [B, 2, H, W]
    h_pos = (hw["hw_grid_code"].double() / 2.0 ** fh).expand_as(a_pos)
    k = int(hw["frac_sum"])
    a_acc = alg["out_pre"].double() / case["warp"]["ref_scale"]
    h_acc = hw["hw_acc"].double() / 2.0 ** k
    x0, y0 = torch.floor(a_pos[:, 0]), torch.floor(a_pos[:, 1])
    inside = (x0 >= 0) & (x0 + 1 < W) & (y0 >= 0) & (y0 + 1 < H)   # [B, H, W]
    inside = inside.unsqueeze(1).expand_as(a_acc)
    return [_row("ref code", "Q5 Q6", alg["ref_code"], hw["hw_ref_code"][..., :W]),
            _row("grid px", "Q3 Q5", a_pos, h_pos),
            _row("acc interior", "Q3", a_acc, h_acc, WARP_ACC_TOL, inside),
            _row("acc border", "Q1 Q2 Q10", a_acc, h_acc, WARP_ACC_TOL, ~inside)]


def warp_float32_store(hw):
    """(n, n above 2^24, n rounded): their float32 output vs their own int64
    acc. out_raw * 2^(frx+fry) is exact where |acc| <= 2^24 (HW Q9)."""
    acc = hw["hw_acc"]
    back = hw["out_raw"].double() * 2.0 ** int(hw["frac_sum"])
    return (acc.numel(), int((acc.abs() > 2 ** 24).sum()),
            int((back != acc.double()).sum()))


def report_warp(case, alg, hw):
    print("  warp per stage (HW question):")
    for r in warp_stages(case, alg, hw):
        print(f"    {r['stage']:<13} {r['q']:<10} {r['n_diff']:>6}/{r['n']:<6} "
              f"differ, max {r['max_d']:.6g}")
    n, big, rounded = warp_float32_store(hw)
    print(f"    their float32 output vs their int64 acc (Q9): {big}/{n} acc above "
          f"2^24, {rounded} rounded")
    print("    ALG out_code: no HW counterpart, their PCU is bypassed (Q7)")
    for line in HW.warp_notes(hw):
        print(f"  {line}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-m", default="all", help="case name, op (conv/convt/warp) or all")
    ap.add_argument("--config", default=SC.DEFAULT_CONFIG,
                    help="framework mx_config.json (default mx_config_dof.json)")
    ap.add_argument("--layer", default=None,
                    help="entry of the config's layers to use (its group/kind)")
    ap.add_argument("--backend", default=None,
                    help="override xblock_accum.backend (python / triton)")
    ap.add_argument("--input", default="float", choices=("float", "int", "both"),
                    help="float input, integer input (grid 2^-x_frac), or both")
    ap.add_argument("--x-frac", type=int, default=4,
                    help="integer input grid: x in units of 2^-x_frac (default 4)")
    args = ap.parse_args()

    cfg = SC.load_config(args.config, args.layer, args.backend)
    modes = {"float": [None], "int": [args.x_frac],
             "both": [None, args.x_frac]}[args.input]
    results, outd = {}, {}
    for name in C.select(args.m):
        for xf in modes:
            key = name if xf is None else f"{name} [int 2^-{xf}]"
            case = C.build(name, x_frac=xf)
            alg = ALG.run(case, cfg)
            try:
                hw = HW.run(case, cfg)
            except NotImplementedError as e:
                print(f"\n=== {key} ===\n  HW not implemented: {e}  "
                      f"(ALG: {len(alg)} taps)")
                results[key] = None
                continue
            results[key] = report(key, compare(alg, hw), alg, hw)
            if case["op"] == "warp":
                report_warp(case, alg, hw)
            if "pre_round_moved" in hw:
                print(f"  HW input off its integer grid (x*2^x_cal): "
                      f"{int(hw['pre_round_moved'])}/{case['x'].numel()} elements"
                      f" (not rounded: not an input layer)")
            if "out_bins" in hw:
                # PLA. HW: their one-bin PCU (as it is) and our bins stand-in
                # (main_HW, until their emulator has bins). PCU alone: the
                # lib's PLA + window on HW's own acc words, same 2^-12 grid.
                k = hw["hw_pla_bin"]
                neg = k == 0
                d1 = (alg["out"].double() - hw["out"].double()).abs()
                d2 = (alg["out"].double() - hw["out_bins"].double()).abs()
                print(f"  PLA: lower bin {int(neg.sum())}/{neg.numel()} elements")
                for lab, d_ in (("their PCU (1 bin)", d1), ("bins stand-in (ours)", d2)):
                    print(f"    HW-ALG {lab:21s} max {float(d_.max()):.4g}  "
                          f"lower bin max {float(d_[neg].max()) if neg.any() else 0:.4g}  "
                          f"upper bin max {float(d_[~neg].max()) if (~neg).any() else 0:.4g}")
                try:
                    o, _ = ALG.pcu_on(hw["hw_acc"], cfg, case, -int(hw["out_scale_log2"]))
                    nd = int((o != hw["hw_out_raw_bins"]).sum())
                    print(f"    PCU alone (same acc): lib bins vs stand-in "
                          f"{nd}/{o.numel()} differ")
                except (NotImplementedError, SystemExit) as e:
                    print(f"    PCU alone: skipped ({e})")
            d = (alg["out"].double() - hw["out"].double()).abs()
            outd.setdefault(name, {})[xf] = (float(d.max()),
                                             float((d ** 2).mean().sqrt()))

    if args.input == "both" and outd:
        print(f"\n  HW-ALG out      {'float in':>18} {'int in':>18}")
        print(f"  {'case':<14} {'max':>8} {'rms':>9} {'max':>8} {'rms':>9}")
        for n, r in outd.items():
            if len(r) == 2:
                (fm, fr), (im, ir) = r[None], r[args.x_frac]
                print(f"  {n:<14} {fm:>8.4f} {fr:>9.5f} {im:>8.4f} {ir:>9.5f}")

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
