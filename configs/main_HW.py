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
import sim_config as SC
from taps import tap, show

# --- HW team code (hw/npe_mx_emu, verbatim) ---------------------------------
from npe_mx_emu import CalibratedMXConv2d, LayerMXQParams, MXObserver
from npe_mx_emu import mx_config_factory, SUPPORTED_ACC_WIDTHS
from npe_mx_emu import Cstruct, pcu
from npe_mx_emu import hw_pipe as _hw_pipe

# Widths fixed inside npe_mx_emu - their API has no parameter for them, so the
# config can't reach them (listed so a config that disagrees is visible):
HW_FIXED = {
    "pcu_clamp_bits": 24,  # calib.build_c_struct_arr default; calibrate() never passes it
}


def hw_format(cfg):
    """Config mx_specs -> npe_mx_emu (mx_type, block_size).

    mx_type is their name "mx-e{scale_bits}bm{mantissa bits}"; the exponent
    wrap mask comes with it (their MXTensorConfig.overflow_mask). Refused when
    their code can't run it as configured, rather than silently running
    something else.
    """
    sp = cfg["mx_specs"]
    a, w = sp["a_elem_format"], sp["w_elem_format"]
    if a != w or not str(a).startswith("int"):
        raise NotImplementedError(f"HW: needs a == w == intN, got a={a!r} w={w!r}")
    ebits, mbits = int(sp["scale_bits"]), int(a[3:])
    mx_type = f"mx-e{ebits}bm{mbits}"
    if mx_type not in mx_config_factory:
        raise NotImplementedError(
            f"HW: {mx_type} (scale_bits={ebits}, {a}) not in npe_mx_emu "
            f"mx_config_factory {sorted(mx_config_factory)}")
    return mx_type, int(sp["block_size"])


def hw_acc_bits(cfg):
    """Accumulator width: config xblock_accum.bits, refused unless their
    MXConv2D_Loops supports it (SUPPORTED_ACC_WIDTHS). No xblock_accum -> their
    default (48)."""
    xa = cfg["mx_specs"].get("xblock_accum")
    bits = int(xa.get("bits", 48)) if isinstance(xa, dict) else 48
    if bits not in SUPPORTED_ACC_WIDTHS:
        raise NotImplementedError(
            f"HW: xblock_accum.bits={bits}, npe_mx_emu supports "
            f"{SUPPORTED_ACC_WIDTHS}")
    return bits


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


# --- PLA (activation folded into the PCU) -----------------------------------
# HW (their answer to Q10): per channel, bins split by boundaries on the
# accumulator word; bin k has an int8 signed scale S_k and a bias B_k:
# out = S_k * acc + B_k -> pcu_div (shift, round half up) -> relux clamp.
# PReLU / leaky = 2 bins at boundary acc = -b. Their emulator today is ONE bin
# (Cstruct has one scale and one bias); they will extend Cstruct / pcu_pla.
# All values come from the config (hw_grids pla_frac, pla_bin_bounds /
# pla_bin_scales / pla_bin_biases, xblock_accum.pla_bin_tie) - the same values
# the ALG side reads.

def _pla_bins(case, cfg, qp, O):
    """The config's PLA bins for this layer, on their accumulator grid."""
    g = SC.hw_grid(cfg, case) or {}
    keys = ("pla_frac", "pla_bin_bounds", "pla_bin_scales", "pla_bin_biases")
    miss = [k for k in keys if k not in g]
    if miss:
        raise SystemExit(f"{os.path.basename(cfg['file'])}: hw_grids "
                         f"{SC.grid_key(cfg, case)!r} lacks {miss}; the HW PCU "
                         f"bins come from the config")
    # bounds / biases sit on the grid of the config's e_layer_min; their
    # accumulator LSB is 2^-(x_cal + w_cal). Same grid needed, else the values
    # would have to be re-derived (= calibration, which the sim does not do).
    if -qp.b_cal != int(g["e_layer_min"]):
        raise NotImplementedError(
            f"HW PLA: their accumulator grid 2^-{qp.b_cal} (x_cal + w_cal) != "
            f"config e_layer_min {g['e_layer_min']}; the config's bins don't apply")
    xa = cfg["mx_specs"].get("xblock_accum") or {}
    bits = HW_FIXED["pcu_clamp_bits"]
    return {"frac": int(g["pla_frac"]),
            "bounds": [list(map(int, v)) for v in g["pla_bin_bounds"]],
            "scales": [list(map(int, v)) for v in g["pla_bin_scales"]],
            "biases": [list(map(int, v)) for v in g["pla_bin_biases"]],
            "tie": xa.get("pla_bin_tie", "upper"),
            "lo": -2.0 ** (bits - 1), "hi": 2.0 ** (bits - 1) - 1}


def _cs(pla, scale, bias):
    return Cstruct(bias=bias, scale=scale, shift=-pla["frac"], round_type="round",
                   min_val=pla["lo"], max_val=pla["hi"])


def _pla_single(pla, O):
    """Their emulator as it is: ONE bin per channel. Given the upper bin
    (S = m_pos, B its bias): exact for the positive side, no PReLU."""
    return {o: _cs(pla, pla["scales"][o][-1], pla["biases"][o][-1]) for o in range(O)}


def bins_pcu(n, pla, o, e):
    """OUR ASSUMPTION UNTIL HW EMULATOR HAS BINS, one channel `o`: bin of
    each acc word n (bounds ascending; tie 'upper': n == bound goes up), then
    their pcu with that bin's S_k, B_k. Returns (out int64, bin uint8)."""
    k = torch.zeros_like(n, dtype=torch.uint8)
    for j, t in enumerate(pla["bounds"][o]):
        k = torch.where(n >= t if pla["tie"] == "upper" else n > t,
                        torch.full_like(k, j + 1), k)
    y = torch.zeros_like(n)
    for j, (sk, bk) in enumerate(zip(pla["scales"][o], pla["biases"][o])):
        y = torch.where(k == j, pcu(n, c_struct=_cs(pla, sk, bk), e_min_layer=e).long(), y)
    return y, k


def _pla_bins_standin(taps, pla, layer, mx_a):
    """OUR ASSUMPTION UNTIL HW EMULATOR HAS BINS: per channel pick the bin by
    acc vs the boundaries (tie rule from the config), then their pcu
    (pcu_pla -> pcu_div -> relux) with that bin's S_k, B_k. Runs on the
    accumulator lines their bcu produced."""
    acc = taps["hw_acc"]
    e = int(layer.mx_conv.mx_weight.e_min_layer + mx_a.e_min_layer)
    out = torch.empty_like(acc)
    bins = torch.zeros_like(acc, dtype=torch.uint8)
    for o in range(acc.shape[1]):
        out[:, o], bins[:, o] = bins_pcu(acc[:, o], pla, o, e)
    tap(taps, "pla_frac", pla["frac"])
    tap(taps, "pla_bin_scales", torch.tensor(pla["scales"]))
    tap(taps, "pla_bin_biases", torch.tensor(pla["biases"]))
    tap(taps, "pla_bin_bounds", torch.tensor(pla["bounds"]))
    tap(taps, "hw_pla_bin", bins)
    tap(taps, "hw_out_raw_bins", out)
    tap(taps, "out_bins", out * 2.0 ** -layer.output_scale_log2)

def conv(case, cfg, taps):
    """MX conv on the HW team's NPE emulator (hw/npe_mx_emu).

    Flow exactly as their notebook (mx_conv_npe_emu_example):
      MXObserver(mode="mx") on x and on w -> x_cal, w_cal
      LayerMXQParams(x_cal, w_cal) -> CalibratedMXConv2d (weight/bias copied)
      calibrate() -> forward(x) -> raw int, float = raw * 2^-output_scale_log2

    Their format differs from ALG by design (e4bm8: per-tensor calibrate
    offset, unsigned exponent with wrap, mantissa truncate, int64 acc with
    two's-complement wrap, PCU round half up + 24-bit clamp), so their integer
    taps carry an `hw_` prefix and are not compared to ALG's; `out` is.

    Not supported by their code: padding (they raise), dilation (they have no
    such parameter - refused here so it can't run silently as dilation 1),
    groups.

    Taps: x w | x_cal w_cal pre_round_moved | hw_a_mant hw_a_exp |
          hw_w_mant hw_w_exp | hw_acc | out_raw | out
    pre_round_moved: how many input elements a round(x * 2^x_cal) would
    change. Their forward rounds only an input layer (is_input_layer); this
    layer is not one, so x * 2^x_cal goes to their quantizer unrounded.
    """
    x, w, b, p = case["x"], case["w"], case["b"], case["params"]
    O, Cin, kH, kW = w.shape
    if p["dilation"] != 1:
        raise NotImplementedError("HW conv: npe_mx_emu has no dilation")
    mx_type, bs = hw_format(cfg)
    acc_bits = hw_acc_bits(cfg)

    # calibration (theirs); weights blocked per flattened kernel, as they say
    obs_x = MXObserver(block_size=bs, mx_type=mx_type, mode="mx"); obs_x.update(x)
    obs_w = MXObserver(block_size=bs, mx_type=mx_type, mode="mx")
    obs_w.update(w, per_row_flatten=True)
    qp = LayerMXQParams(x_cal=obs_x.calibrate_offset(),
                        w_cal=obs_w.calibrate_offset(),
                        block_size=bs, mx_type=mx_type)

    layer = CalibratedMXConv2d(Cin, O, (kH, kW), stride=p["stride"],
                               padding=p["padding"], qparams=qp)
    # their CalibratedMXConv2d doesn't forward acc_bit_width; MXConv2D_Loops
    # keeps it as an attribute and passes it to bcu
    layer.mx_conv.acc_bit_width = acc_bits
    with torch.no_grad():
        layer.mx_conv.weight.copy_(w)
        layer.mx_conv.bias.copy_(b if b is not None else torch.zeros(O))
    layer.calibrate()
    pla = _pla_bins(case, cfg, qp, O) if case.get("act") else None
    if pla is not None:                 # their one-bin PCU, as it is today
        layer.mx_conv.load_c_struct(_pla_single(pla, O))

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
    xs = x.double() * 2.0 ** qp.x_cal
    tap(taps, "pre_round_moved", int((xs != xs.round()).sum()))
    tap(taps, "acc_bits", acc_bits)
    tap(taps, "hw_a_mant", mx_a.data)        # [B, Cin, H, nblk, bs] int32
    tap(taps, "hw_a_exp", mx_a.shared_exp)   # [B, Cin, H, nblk]
    tap(taps, "hw_w_mant", mx_w.data)        # [O, nblk, bs]
    tap(taps, "hw_w_exp", mx_w.shared_exp)   # [O, nblk]
    # bcu is called per (b, oc, oh) in that loop order -> [B, O, Ho, Wo]
    tap(taps, "hw_acc", torch.stack(lines).reshape(B, O, Ho, Wo))
    tap(taps, "out_raw", out_raw)
    tap(taps, "out_scale_log2", layer.output_scale_log2)   # out grid 2^-this
    tap(taps, "out", out_raw * 2.0 ** -layer.output_scale_log2)
    if pla is not None:
        _pla_bins_standin(taps, pla, layer, mx_a)
    return taps["out"]


def convt(case, cfg, taps):
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


def warp(case, cfg, taps):
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


def run(case, cfg=None):
    """One case -> {tap_name: tensor}. cfg from sim_config.load_config (the
    same one ALG reads; default file when None)."""
    cfg = cfg if cfg is not None else SC.load_config()
    taps = {}
    OPS[case["op"]](case, cfg, taps)
    return taps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-m", default="all", help="case name, op (conv/convt/warp) or all")
    ap.add_argument("--config", default=SC.DEFAULT_CONFIG,
                    help="framework mx_config.json (same file ALG reads)")
    args = ap.parse_args()
    cfg = SC.load_config(args.config)
    for name in C.select(args.m):
        show(run(C.build(name), cfg), f"{name}  HW")


if __name__ == "__main__":
    main()
