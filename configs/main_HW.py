"""
main_HW.py - run one case through the HW implementation (the HW side).

PLACEHOLDER. HW fills the three op bodies. Same interface as main_ALG:

    from main_HW import run
    taps = run(cases.build("conv3x3"))   # {tap_name: tensor}, same names as ALG

Rules:
  1. No MX lib import (microxcaling / mx_quantizer / mx_layers_*). Integer ops,
     shifts and a saturating accumulator only - if both sides shared the
     quantizer, agreeing would prove nothing.
  2. Take the float inputs from the case (x, w, b, flow) and quantize them here.
  3. Log every stage with `tap`, same names as ALG. main.py compares shared
     names in order and blames the first that differs; extra HW-only taps are
     fine (listed, not failed).
  4. No RNG: the case already holds all the data.
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cases as C
from taps import tap, show   # noqa: F401  (HW uses tap in the bodies)


def conv(case, taps):
    """MX conv on the HW datapath.

    Steps:
      1. x -> a_int int8 [B,Cin,H,W], a_exp int16 [B,Cin,H,ceil(W/32)]
         blocks of 32 along W. E = floor(log2 max|block|), code = E + 127,
         mant = round_half_away(x * 2^(6 - E)), clamp [-127, 127].
      2. w -> each filter flattened C-major to [Cin*kH*kW], blocks of 32 along it.
         w_int int8 [O,Cin,kH,kW], w_exp int16 [O, ceil(Cin*kH*kW/32)].
      3. per output, for c in Cin, for (kh, kw): p = a_int * w_int,
         shift by (Ea + Ew - 12 - e_layer_min), e_layer_min = -20,
         add into a 48-bit accumulator, saturate after every product.
         Zero padding / stride / dilation from case["params"].
      4. bias: round(b * 2^20) added after the last product, saturate again.

    Taps: a_int a_exp w_int w_exp acc (int64) sat (uint8) out (= acc * 2^-20).
    """
    raise NotImplementedError("HW conv")


def convt(case, taps):
    """ConvTranspose as zero-insert + conv.

    Steps:
      1. dilate x: insert (stride - 1) zeros between pixels in H and W.
         tap "x_dil".
      2. pad by (k - 1 - padding) each side (+ output_padding at bottom/right).
      3. kernel: w [Cin,O,k,k] -> flip in kH, kW, swap to [O,Cin,k,k].
      4. run `conv` steps 1-4 with stride 1.

    Open: quantize x before the zero insert (blocks on the original W) or after
    (blocks on the dilated W)? ALG (lib) today blocks along Cin and accumulates
    in fp32, so only `out` is comparable, as float.

    Taps: x_dil, then the conv taps, out.
    """
    raise NotImplementedError("HW convt")


def warp(case, taps):
    """Bilinear backward warp, integer.

    Steps:
      1. ref_code  = clamp(round_half_away(x / ref_scale), -128, 127)    int8
      2. grid_code = clamp(round_half_away(flow * 256), -32768, 32767)    int16,
                     Q8.8 displacement in px, channel 0 = x
      3. X = (x << 8) + grid_code[:,0];  x0 = X >> 8;  fx = X & 255  (same for y)
      4. acc = sum of 4 neighbours ref_code * wx * wy,
               wx in {256 - fx, fx}, wy in {256 - fy, fy}; outside image -> 0
      5. out_code = clamp(round_half_away(acc * ref_scale / out_scale / 65536),
                          -128, 127)                                     int8

    Taps: ref_code grid_code (acc) out_code out (= out_code * out_scale).
    """
    raise NotImplementedError("HW warp")


OPS = {"conv": conv, "convt": convt, "warp": warp}


def run(case):
    """One case -> {tap_name: tensor}."""
    taps = {}
    OPS[case["op"]](case, taps)
    return taps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-m", default="all", help="case name, op (conv/convt/warp) or all")
    args = ap.parse_args()
    for name in C.select(args.m):
        show(run(C.build(name)), f"{name}  HW")


if __name__ == "__main__":
    main()
