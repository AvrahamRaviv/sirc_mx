"""
main_HW.py - run one case through the HW implementation (the HW side).

HW code lives in hw/npe_mx_emu/, copied verbatim from the HW team - never
edited here. This file only wires it in: it builds their layer, calls their
calibration and forward, and taps what their code produced (via a forward
pre-hook and a pass-through wrapper on their bcu - no logic of theirs is
reimplemented). Status, per their guide: conv done; warp, convT not yet.

Same interface as main_ALG:

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

import contextlib

import torch

_HW_DIR = os.path.join(_HERE, "hw")
if _HW_DIR not in sys.path:
    sys.path.insert(0, _HW_DIR)

import cases as C
from taps import tap, show

# --- HW team code (hw/npe_mx_emu, verbatim) ---------------------------------
from npe_mx_emu import CalibratedMXConv2d, LayerMXQParams, MXObserver
from npe_mx_emu import hw_pipe as _hw_pipe


@contextlib.contextmanager
def _capture_bcu():
    """Record every bcu() result (one output line: p_sum[out_width]) in call
    order. Pass-through: their bcu runs unchanged, we only keep its output."""
    seen = []
    orig = _hw_pipe.bcu

    def rec(*a, **k):
        p = orig(*a, **k)
        seen.append(p.detach().clone())
        return p

    _hw_pipe.bcu = rec
    try:
        yield seen
    finally:
        _hw_pipe.bcu = orig


def conv(case, taps):
    """MX conv on the HW team's NPE emulator (hw/npe_mx_emu).

    Flow exactly as their notebook (mx_conv_npe_emu_example):
      MXObserver(mode="mx") on x and on w -> x_cal, w_cal
      LayerMXQParams(x_cal, w_cal) -> CalibratedMXConv2d (weight/bias copied)
      calibrate() -> forward(x) -> raw int, float = raw * 2^-output_scale_log2

    Their format differs from ALG by design (e4bm8: per-tensor calibrate
    offset, ceil log2, unsigned 4-bit exponent with wrap, mantissa truncate,
    46-bit acc with floor shift, PCU round + 24-bit clamp), so their integer
    taps carry an `hw_` prefix and are not compared to ALG's; `out` is.

    Not supported by their code: padding (they raise), dilation (they have no
    such parameter - refused here so it can't run silently as dilation 1),
    groups.

    Taps: x w | x_cal w_cal | hw_a_mant hw_a_exp | hw_w_mant hw_w_exp |
          hw_acc | out_raw | out
    """
    x, w, b, p = case["x"], case["w"], case["b"], case["params"]
    O, Cin, kH, kW = w.shape
    if p["dilation"] != 1:
        raise NotImplementedError("HW conv: npe_mx_emu has no dilation")

    # calibration (theirs)
    obs_x = MXObserver(mode="mx"); obs_x.update(x)
    obs_w = MXObserver(mode="mx"); obs_w.update(w)
    qp = LayerMXQParams(x_cal=obs_x.calibrate_offset(),
                        w_cal=obs_w.calibrate_offset())

    layer = CalibratedMXConv2d(Cin, O, (kH, kW), stride=p["stride"],
                               padding=p["padding"], qparams=qp)
    with torch.no_grad():
        layer.mx_conv.weight.copy_(w)
        layer.mx_conv.bias.copy_(b if b is not None else torch.zeros(O))
    layer.calibrate()

    seen_mx = []
    hook = layer.mx_conv.register_forward_pre_hook(
        lambda m, args: seen_mx.append(args[0]))
    try:
        with torch.no_grad(), _capture_bcu() as lines:
            out_raw = layer(x)
    finally:
        hook.remove()
    mx_a, mx_w = seen_mx[0], layer.mx_conv.mx_weight

    B, _, Ho, Wo = out_raw.shape
    tap(taps, "x", x)
    tap(taps, "w", w)
    tap(taps, "x_cal", qp.x_cal)
    tap(taps, "w_cal", qp.w_cal)
    tap(taps, "hw_a_mant", mx_a.data)        # [B, Cin, H, nblk, bs] int32
    tap(taps, "hw_a_exp", mx_a.shared_exp)   # [B, Cin, H, nblk]
    tap(taps, "hw_w_mant", mx_w.data)        # [O, nblk, bs]
    tap(taps, "hw_w_exp", mx_w.shared_exp)   # [O, nblk]
    # bcu is called per (b, oc, oh) in that loop order -> [B, O, Ho, Wo]
    tap(taps, "hw_acc", torch.stack(lines).reshape(B, O, Ho, Wo))
    tap(taps, "out_raw", out_raw)
    tap(taps, "out", out_raw * 2.0 ** -layer.output_scale_log2)
    return taps["out"]


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
