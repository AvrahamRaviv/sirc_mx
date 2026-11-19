"""
main_HW.py - run one case through the HW implementation (the HW side).

HW code lives in hw/npe_mx_emu/, the HW team's own (their drops merged with
their commits, configs/README.md "HW side") - never hand-edited here. This
file only wires it in: it builds their layer, calls their calibration and
forward, and taps what their code produced (via a forward pre-hook and
pass-through wrappers on their module functions - no logic of theirs is
reimplemented). Status: conv (dense, depthwise), convT and warp
wired in.

Same interface as main_ALG:

    from main_HW import run
    taps = run(cases.build("conv3x3"))   # {tap_name: tensor}, same names as ALG

Rules:
  1. No MX lib import (microxcaling / mx_quantizer / mx_layers_*). Integer ops,
     shifts and a saturating accumulator only - if both sides shared the
     quantizer, agreeing would prove nothing. Config reading is shared on
     purpose: per-call-site fxp lists go through fixed_point/fxp_sites (pure
     python, via sim_config), and an fxp key a config leaves out takes the
     lib's default (fixed_point/fxp_quant OUT_QUANT_DEFAULTS, a dict, never its
     quantizer), so both sides accept and refuse the same configs.
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
import dataclasses
import math

import torch

_HW_DIR = os.path.join(_HERE, "hw")
if _HW_DIR not in sys.path:
    sys.path.insert(0, _HW_DIR)

import cases as C
import sim_cli as CLI
import sim_config as SC
from taps import tap, show

# --- HW team code (hw/npe_mx_emu, theirs, never hand-edited) ----------------
from npe_mx_emu import CalibratedMXConv2d, LayerMXQParams, MXObserver
from npe_mx_emu import CalibratedMXConvTranspose2d, CalibratedMXDepthwiseConv2d
from npe_mx_emu import CalibratedMXWarp
from npe_mx_emu import mx_config_factory, SUPPORTED_ACC_WIDTHS
from npe_mx_emu import Cstruct, pcu, mantissa_offset
from npe_mx_emu import hw_pipe as _hw_pipe
from npe_mx_emu import layers as _hw_layers

from fixed_point.fxp_quant import OUT_QUANT_DEFAULTS   # config defaults only

# Widths fixed inside npe_mx_emu - their API has no parameter for them, so the
# config can't reach them (listed so a config that disagrees is visible):
HW_FIXED = {
    "pcu_clamp_bits": 24,  # calib.build_c_struct_arr default; calibrate() never passes it
}


# Their conv mantissa lane: mx_format says the 16-bit sibling (mx-e4bm16) is
# for the warp pipe's bilinear weights, not conv.
CONV_MANTISSA_BITS = 8


def hw_format(cfg):
    """Config mx_specs -> npe_mx_emu (mx_type, block_size).

    mx_type is their name "mx-e{bits}bm{mantissa bits}"; the exponent wrap
    mask comes with it (their MXTensorConfig.overflow_mask). bits = the NPE
    shared-exponent register: xblock_accum.in_exp_bits when the lib models
    that register (in_exp_mode on), else mx_specs.scale_bits. Conv runs on
    their 8-bit mantissa lane only (CONV_MANTISSA_BITS): their factory also has
    mx-e4bm16, but that lane is the warp's (grid and weights), and their conv
    on it gives wrong numbers instead of an error. Refused when their code
    can't run it as configured, rather than silently running something else.
    """
    sp = cfg["mx_specs"]
    a, w = sp["a_elem_format"], sp["w_elem_format"]
    if a != w or not str(a).startswith("int"):
        raise NotImplementedError(f"HW: needs a == w == intN, got a={a!r} w={w!r}")
    xa = sp.get("xblock_accum") if isinstance(sp.get("xblock_accum"), dict) else {}
    reg = xa.get("in_exp_bits") if xa.get("in_exp_mode", "off") != "off" else None
    ebits, mbits = int(reg if reg is not None else sp["scale_bits"]), int(a[3:])
    mx_type = f"mx-e{ebits}bm{mbits}"
    if mbits != CONV_MANTISSA_BITS:
        why = (f"their {mx_type} is the warp's 16-bit lane: wrong numbers on conv"
               if mx_type in mx_config_factory else f"npe_mx_emu has no {mx_type}")
        raise NotImplementedError(
            f"HW conv: {a} needs a {mbits}-bit mantissa lane; npe_mx_emu conv "
            f"runs int{CONV_MANTISSA_BITS} only ({why})")
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
def _pass_through(module, name, keep=lambda r: r):
    """Record every result of their function `module.name`, in call order.
    Pass-through: their function runs unchanged, we only keep `keep(result)`.
    Their callers look the name up in their module at call time, so they call
    the wrapper."""
    seen = []
    orig = getattr(module, name)

    def rec(*a, **k):
        r = orig(*a, **k)
        seen.append(keep(r))
        return r

    setattr(module, name, rec)
    try:
        yield seen
    finally:
        setattr(module, name, orig)


def _capture_bcu():
    """Every bcu() result (one output line: p_sum[out_width], their int64
    accumulator) in call order."""
    return _pass_through(_hw_pipe, "bcu", keep=lambda p: p.detach().clone())


# --- PLA (activation folded into the PCU) -----------------------------------
# HW (their answer to Q10): per channel, bins split by boundaries on the
# accumulator word; bin k has an int8 signed scale S_k and a bias B_k:
# out = S_k * acc + B_k -> pcu_div (shift, round half up) -> relux clamp.
# PReLU / leaky = 2 bins at boundary acc = -b. Their emulator today is ONE bin
# (Cstruct has one scale and one bias); they will extend Cstruct / pcu_pla.
# All values come from the config (hw_grids pla_frac, pla_bin_bounds /
# pla_bin_scales / pla_bin_biases, xblock_accum.pla_bin_tie) - the same values
# the ALG side reads.

def _pla_bins(case, cfg, qp, O, s_shift=0):
    """The config's PLA bins for this layer, on their accumulator grid."""
    g = SC.hw_grid(cfg, case) or {}
    where = f"{os.path.basename(cfg['file'])}: hw_grids {SC.grid_key(cfg, case)!r}"
    SC.require(g, ("pla_frac", "pla_bin_bounds", "pla_bin_scales", "pla_bin_biases"),
               where, "the HW PCU bins come from the config")
    SC.require(g, ("e_layer_min",), where, "the bins sit on its accumulator grid")
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
            "tie": xa.get("pla_bin_tie", "upper"), "static": s_shift,
            "lo": -2.0 ** (bits - 1), "hi": 2.0 ** (bits - 1) - 1}


def _cs(pla, scale, bias):
    return Cstruct(bias=bias, scale=scale, shift=pla.get("static", 0) - pla["frac"],
                   round_type="round",
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


def _pla_bins_standin(taps, pla, layer, engine, mx_a):
    """OUR ASSUMPTION UNTIL HW EMULATOR HAS BINS: per channel pick the bin by
    acc vs the boundaries (tie rule from the config), then their pcu
    (pcu_pla -> pcu_div -> relux) with that bin's S_k, B_k. Runs on the
    accumulator lines their bcu produced."""
    acc = taps["hw_acc"]
    e = int(engine.mx_weight.e_min_layer + mx_a.e_min_layer)
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


def _hw_setup(case, cfg, x, w_conv):
    """Their format and calibration for one conv-family layer -> dict
    {qp, acc_bits, s_shift, cal_src}.

    w_conv is the weight in the layout their quantizer blocks: [O, C, kH, kW],
    each filter flattened C-major (convT: the converted kernel, convt_weight).

    Calibration offsets: from the config's hw_grids when it has the register
    bases (offline PTQ, as deployed): their observer anchors the smallest
    non-zero block at shift 0, cal = mantissa_offset - E_min, and e_base_a /
    e_base_w ARE E_min. Else their observers on this input, weights blocked
    per flattened kernel, as their quantizer (per_row_flatten).
    """
    mx_type, bs = hw_format(cfg)
    acc_bits = hw_acc_bits(cfg)
    grid = SC.hw_grid(cfg, case) or {}
    m_off = mantissa_offset(mx_type)
    s_shift = int(grid.get("static_shift", 0))     # PCU static shift (M7)
    if case.get("input_layer"):
        # first layer (M8): their input_mx_quantizer, x_int = round(x * 2^cal),
        # zero shared exponent. cal from the config's in_scale (POT only:
        # their code scales by 2^cal). The lib folds in_scale into the weights,
        # so its weight base is of w * in_scale: E(w) = e_base_w + cal.
        if "in_scale" not in grid or "e_base_w" not in grid:
            raise SystemExit(f"{os.path.basename(cfg['file'])}: input layer needs "
                             f"hw_grids in_scale and e_base_w")
        cal_f = -math.log2(float(grid["in_scale"]))
        if cal_f != round(cal_f):
            raise NotImplementedError(
                f"HW input layer: in_scale {grid['in_scale']} is not a power of two; "
                f"their input path scales by 2^cal (in_int_scale 'pot' needed)")
        x_cal = int(round(cal_f))
        w_cal = m_off - (int(grid["e_base_w"]) + x_cal)
        cal_src = "config"
    elif "e_base_a" in grid and "e_base_w" in grid:
        x_cal, w_cal = m_off - int(grid["e_base_a"]), m_off - int(grid["e_base_w"])
        cal_src = "config"
    else:
        obs_x = MXObserver(block_size=bs, mx_type=mx_type, mode="mx"); obs_x.update(x)
        obs_w = MXObserver(block_size=bs, mx_type=mx_type, mode="mx")
        obs_w.update(w_conv, per_row_flatten=True)
        x_cal, w_cal = obs_x.calibrate_offset(), obs_w.calibrate_offset()
        cal_src = "observer"
    # out_cal decodes their output: the static shift moves the 2^-12 base grid
    qp = LayerMXQParams(x_cal=x_cal, w_cal=w_cal, block_size=bs, mx_type=mx_type,
                        out_cal=2 * m_off + s_shift,
                        is_input_layer=bool(case.get("input_layer")))
    return {"qp": qp, "acc_bits": acc_bits, "s_shift": s_shift, "cal_src": cal_src}


def _run_hw_layer(layer, engine, case, hw, pla=None):
    """Load the case's weight and bias into their layer, calibrate, run
    forward(x). engine: their loop module inside `layer` (mx_conv, mx_dw,
    mx_convt). The weight goes in as the case holds it (their layer converts
    it if its op needs that). Returns {out_raw, mx_a (the MX input their
    engine got), lines (every bcu p_sum, call order), canvas (every
    pad_dilate_streams result: the activation streams after their geometry
    zeros)}."""
    w, b = case["w"], case["b"]
    # their Calibrated* layers don't forward acc_bit_width; the engine keeps it
    # as an attribute and passes it to bcu
    engine.acc_bit_width = hw["acc_bits"]
    with torch.no_grad():
        engine.weight.copy_(w)
        engine.bias.copy_(b if b is not None else torch.zeros_like(engine.bias))
    layer.calibrate()
    if pla is not None:                 # their one-bin PCU, as it is today
        engine.load_c_struct(_pla_single(pla, engine.bias.numel()))
    elif hw["s_shift"]:                 # their Cstruct.shift = the static shift
        engine.load_c_struct(
            {o: dataclasses.replace(c, shift=hw["s_shift"])
             for o, c in engine.c_struct_arr.items()})

    seen_mx = []
    hook = engine.register_forward_pre_hook(lambda m, args: seen_mx.append(args[0]))
    try:
        with torch.no_grad(), _capture_bcu() as lines, \
                _pass_through(_hw_pipe, "pad_dilate_streams") as canvas:
            out_raw = layer(case["x"])
    finally:
        hook.remove()
    return {"out_raw": out_raw, "mx_a": seen_mx[0], "lines": lines, "canvas": canvas}


def _tap_hw(taps, case, hw, layer, engine, r):
    """The taps every conv-family HW run gives, in their order:
    x w | cal_from_config x_cal w_cal pre_round_moved acc_bits |
    hw_a_mant hw_a_exp | hw_w_mant hw_w_exp | hw_acc | out_raw out_scale_log2 out
    pre_round_moved: how many input elements a round(x * 2^x_cal) would
    change. Their forward rounds only an input layer (is_input_layer); this
    layer is not one, so x * 2^x_cal goes to their quantizer unrounded."""
    qp, mx_a, mx_w, x = hw["qp"], r["mx_a"], engine.mx_weight, case["x"]
    B, O = r["out_raw"].shape[:2]
    tap(taps, "x", x)
    tap(taps, "w", case["w"])
    tap(taps, "cal_from_config", int(hw["cal_src"] == "config"))
    tap(taps, "x_cal", qp.x_cal)
    tap(taps, "w_cal", qp.w_cal)
    xs = x.double() * 2.0 ** qp.x_cal
    tap(taps, "pre_round_moved", int((xs != xs.round()).sum()))
    tap(taps, "acc_bits", hw["acc_bits"])
    tap(taps, "hw_a_mant", mx_a.data)        # [B, Cin, H, nblk, bs] int32
    tap(taps, "hw_a_exp", mx_a.shared_exp)   # [B, Cin, H, nblk]
    tap(taps, "hw_w_mant", mx_w.data)        # [O, nblk, bs]
    tap(taps, "hw_w_exp", mx_w.shared_exp)   # [O, nblk]
    # bcu is called per (b, oc, oh) in that loop order, one output line each
    # -> [B, O, Ho, Wo] (convT: before their output crop)
    lines = r["lines"]
    tap(taps, "hw_acc", torch.stack(lines).reshape(B, O, -1, lines[0].shape[-1]))
    tap(taps, "out_raw", r["out_raw"])
    tap(taps, "out_scale_log2", layer.output_scale_log2)   # out grid 2^-this
    tap(taps, "out", r["out_raw"] * 2.0 ** -layer.output_scale_log2)


def conv(case, cfg, taps):
    """MX conv on the HW team's NPE emulator (hw/npe_mx_emu).

    Flow exactly as their notebook (mx_conv_npe_emu_example):
      MXObserver(mode="mx") on x and on w -> x_cal, w_cal  (_hw_setup)
      LayerMXQParams(x_cal, w_cal) -> CalibratedMXConv2d (weight/bias copied)
      calibrate() -> forward(x) -> raw int, float = raw * 2^-output_scale_log2

    Depthwise (case params groups == Cin == O, weight [O, 1, k, k]): their
    CalibratedMXDepthwiseConv2d, one padded block per channel filter (QA
    Q8). Any other grouping is refused: their code has dense and depthwise
    only.

    Padding and dilation are theirs, with torch's meaning: zero (0, 0) pairs
    inserted after quantization into the aligned streams (hw/README.md,
    "Geometry options"): padding at the activation border, dilation between
    the kernel taps.

    Their format differs from ALG by design (e4bm8: per-tensor calibrate
    offset, unsigned exponent with wrap, mantissa truncate, int64 acc with
    two's-complement wrap, PCU round half up + 24-bit clamp), so their integer
    taps carry an `hw_` prefix and are not compared to ALG's; `out` is.

    Taps: _tap_hw; PLA cases add the bins stand-in (_pla_bins_standin).
    """
    w, p = case["w"], case["params"]
    O, Cin_g, kH, kW = w.shape
    Cin = case["x"].shape[1]
    groups = p.get("groups", 1)
    geo = dict(kernel_size=(kH, kW), stride=p["stride"], padding=p["padding"],
               dilation=p["dilation"])
    hw = _hw_setup(case, cfg, case["x"], w)
    if groups == 1:
        layer = CalibratedMXConv2d(Cin, O, qparams=hw["qp"], **geo)
        engine = layer.mx_conv
    elif groups == Cin == O and Cin_g == 1:
        layer = CalibratedMXDepthwiseConv2d(Cin, qparams=hw["qp"], **geo)
        engine = layer.mx_dw
    else:
        raise NotImplementedError(
            f"HW conv: groups {groups} (Cin {Cin}, O {O}); npe_mx_emu has dense "
            f"(groups 1) and depthwise (groups == Cin == O) conv only")
    pla = _pla_bins(case, cfg, hw["qp"], O, hw["s_shift"]) if case.get("act") else None
    r = _run_hw_layer(layer, engine, case, hw, pla)
    _tap_hw(taps, case, hw, layer, engine, r)
    if pla is not None:
        _pla_bins_standin(taps, pla, layer, engine, r["mx_a"])
    return taps["out"]


def convt_weight(w):
    """A torch ConvTranspose2d weight [Cin, O, k, k] -> the conv kernel their
    MXConvTranspose2D_Loops.load_mx_weight quantizes: [O, Cin, k, k], flipped
    in kH, kW. Their docstring (CalibratedMXConvTranspose2d) asks the caller
    to observe the weight on this converted tensor: the conversion changes
    which elements share a block (per OC, C-major)."""
    return w.transpose(0, 1).flip(-1, -2).contiguous()


def convt(case, cfg, taps):
    """ConvTranspose on the HW team's emulator: their
    CalibratedMXConvTranspose2d (MXConvTranspose2D_Loops), nothing of theirs
    reimplemented.

    Their geometry (hw/README.md, "Geometry options"): the compact input is
    MX-quantized first; then the stride zero-insert and the border pad
    (d*(k-1) - padding, a negative one crops the output) are (0, 0) pairs
    inserted into the aligned streams after quantization (QA Q7: the zeros
    are never in a block); then their stride-1 dense conv pipe. The weight
    goes in torch layout [Cin, O, k, k]; their load_mx_weight converts it
    (convt_weight) before quantizing, and the observer sees the converted
    weight. dilation is torch's kernel dilation.

    Refused (their NotImplementedError, named): groups != 1,
    output_padding != 0. Refused here: an activation (case act), which conv
    folds into their PCU; neither side's convT applies one, so running
    without it would read as a match.

    ALG's convT, unless the lib's HW path (xblock_accum.convT "hw"), blocks
    along Cin and accumulates in fp32: only `out` compares, as float.

    Taps: _tap_hw, then x_dil (the activation mantissa stream after their
    zero insert and pad, [B, Cin, H', W'], int) and x_dil_exp (its
    per-element exponents; the inserted zeros carry 0).
    """
    if case.get("act"):
        raise NotImplementedError(
            f"HW convT: an activation ({case['act']['kind']}); the sim folds one "
            f"into the PCU on conv only")
    w, p = case["w"], case["params"]
    Cin, O, kH, kW = w.shape
    hw = _hw_setup(case, cfg, case["x"], convt_weight(w))
    try:
        layer = CalibratedMXConvTranspose2d(
            Cin, O, (kH, kW), stride=p["stride"], padding=p["padding"],
            output_padding=p.get("output_padding", 0), groups=p.get("groups", 1),
            dilation=p["dilation"], qparams=hw["qp"])
    except NotImplementedError as e:
        raise NotImplementedError(f"HW convT: npe_mx_emu: {e}") from e
    r = _run_hw_layer(layer, layer.mx_convt, case, hw)
    _tap_hw(taps, case, hw, layer, layer.mx_convt, r)
    (x_dil, x_dil_exp), = r["canvas"]
    tap(taps, "x_dil", x_dil)
    tap(taps, "x_dil_exp", x_dil_exp)
    return taps["out"]


# Warp, as the HW team answered it (hw/warp_hw_questions_and_answers.md) and
# their corrected emulator does it. Call site: SC.WARP_CALL_SITE.
WARP_REF_MX_TYPE = "mx-e4bm8"   # their reference lane: int8 codes, zero exponents
WARP_REF_BITS = 8               # pack_warp_reference: int8 [-128, 127] (Q6)
WARP_GRID_WORD_BITS = 16        # pack_warp_grid takes an int16 word only (Q3)
WARP_FRAC_RANGE = (0, 8)        # fraction bits per axis pack_warp_grid takes (Q3)
WARP_OUT_BITS = 8               # their forward calibrates relux_bits=8: int8 out (Q7)
# frx + fry = 16 makes the zero-fraction weight 2^16, which a plain unsigned
# 16-bit field does not hold; its encoding is open (Q4). Their emulator keeps
# it as an int64 weight.
WARP_WEIGHT_FIELD_BITS = 16


def _fxp_key(fxp, key):
    """An fxp spec's value, or the lib's default for it (OUT_QUANT_DEFAULTS:
    config reading, not its quantizer)."""
    return fxp.get(key, OUT_QUANT_DEFAULTS[key])


def warp_spec(cfg):
    """The config's warp entry, as their code takes it: the grid word width,
    the grid fraction bits per axis (frx, fry) at the sim's call site, how
    many call sites the config declares. Refused when their code cannot run
    what the config describes: a reference that is not a signed 8-bit word,
    an output fxp that is not int8, a grid wider than int16 or with fraction
    bits outside 0..8, and an entry lacking an input, its fxp
    (sim_config.warp_sites) or its total_bits."""
    name = SC.entry_name(cfg, "warp")
    f = os.path.basename(cfg["file"])
    e = cfg["entries"][name]
    ref, grid = (s[SC.WARP_CALL_SITE] for s in SC.warp_sites(cfg, e))
    for role, fxp in zip(SC.WARP_INPUTS, (ref, grid)):
        SC.require(fxp, ("total_bits",), f"{f}: {name} {role} fxp (call site "
                   f"{SC.WARP_CALL_SITE})", "their lane widths come from the config")
    if int(ref["total_bits"]) != WARP_REF_BITS or not _fxp_key(ref, "signed"):
        raise NotImplementedError(
            f"HW warp: {name} ref is {'' if _fxp_key(ref, 'signed') else 'un'}signed "
            f"{ref['total_bits']}-bit; main_HW feeds their reference lane signed int8 "
            f"(their uint8 path takes a uint8 tensor, whose float-to-code rounding "
            f"is open, HW Q5/Q6)")
    if int(grid["total_bits"]) > WARP_GRID_WORD_BITS:
        raise NotImplementedError(
            f"HW warp: {name} grid is {grid['total_bits']}-bit; their pack_warp_grid "
            f"takes an int{WARP_GRID_WORD_BITS} word")
    if grid.get("frac_bits") is None or grid.get("scale") is not None:
        raise SystemExit(f"{f}: {name} grid has no frac_bits (or a scale); the sim "
                         f"reads the grid's Q format from the config")
    fb = int(grid["frac_bits"])
    lo, hi = WARP_FRAC_RANGE
    if not lo <= fb <= hi:
        raise NotImplementedError(
            f"HW warp: {name} grid has {fb} fraction bits; their pack_warp_grid "
            f"takes {lo}..{hi} per axis (HW Q3)")
    _need_int8_out(cfg, e, name)
    return {"call_sites": SC.entry_call_sites(e, name), "call_site": SC.WARP_CALL_SITE,
            "grid_bits": int(grid["total_bits"]), "frac": (fb, fb)}


def _need_int8_out(cfg, e, name):
    """Their warp output is int8 (relux_bits 8, which their forward sets):
    an output fxp that is on at the sim's call site and not a signed 8-bit
    word is refused. No output fxp: their int8 still runs (ALG's float out
    then differs, said by main_ALG.warp_out_note)."""
    out = e.get("output")
    if not isinstance(out, dict) or "fxp" not in out:
        return
    site = SC.site_fxp(e, out["fxp"], SC.WARP_CALL_SITE, f"{name} output")
    if site is None or site is False:          # off, as the lib reads it
        return
    site = {} if site is True else site        # true: the lib's defaults
    if not site.get("enabled", True):
        return
    if int(_fxp_key(site, "total_bits")) != WARP_OUT_BITS or not _fxp_key(site, "signed"):
        raise NotImplementedError(
            f"HW warp: {name} output fxp is {_fxp_key(site, 'total_bits')}-bit "
            f"{'signed' if _fxp_key(site, 'signed') else 'unsigned'}; their warp "
            f"output is int{WARP_OUT_BITS} (HW Q7)")


def grid_word(flow, frac, grid_bits):
    """Relative flow [B,2,H,W] in px (channel 0 = x) -> the int16 grid word
    their CalibratedMXWarp takes: motion in Q format with `frac` = (frx, fry)
    fraction bits, as the producing layer's PCU emits it (HW Q3, Q5): round
    to nearest with ties toward +inf (Q7), saturated to the config's grid
    word width (signed `grid_bits`)."""
    f = flow.double()
    scale = torch.tensor([2.0 ** frac[0], 2.0 ** frac[1]], dtype=torch.float64).view(1, 2, 1, 1)
    q = torch.floor(f * scale + 0.5)
    lim = 2 ** (grid_bits - 1)
    return q.clamp(-lim, lim - 1).to(torch.int16)


def out_cal(s):
    """Their PCU output shift: out step = ref step * 2^-out_cal (their PCU
    scale is 1, so only a power-of-two ratio runs). Refused otherwise: the
    PCU's programmable multiplier format is open (HW Q7)."""
    r = math.log2(s["ref_scale"] / s["out_scale"])
    if r != round(r):
        raise NotImplementedError(
            f"HW warp: out step {s['out_scale']:g} / ref step {s['ref_scale']:g} is "
            f"not a power of two; their PCU only shifts (scale 1), and the PCU "
            f"multiplier format is open (HW Q7)")
    return int(round(r))


def _refuse_warp(ref, flow):
    """Refuse what their warp does not take (it raises, or would give a wrong
    number without a word): more than one frame, a grid of another size
    (HW Q8, Q10, the agreed scope), a non-finite reference or flow (their
    packer, and an int16 cast, turn NaN into some code without a word)."""
    if ref.shape[0] != 1:
        raise NotImplementedError(
            f"HW warp: batch {ref.shape[0]}; their CalibratedMXWarp takes one "
            f"frame and one grid per call (HW Q8)")
    if tuple(flow.shape[-2:]) != tuple(ref.shape[-2:]):
        raise NotImplementedError(
            f"HW warp: grid {tuple(flow.shape[-2:])} vs reference "
            f"{tuple(ref.shape[-2:])}; their warp takes a same-size grid (HW Q10)")
    for name, t in (("reference", ref), ("flow", flow)):
        if not bool(torch.isfinite(t).all()):
            raise NotImplementedError(
                f"HW warp: a non-finite {name}; it would become an integer code "
                f"without a word")


def _warp_layer(channels, grid_bits, acc_bits, out_shift):
    """Their warp layer: LayerMXQParams(x_cal=0, w_cal=0, out_cal=out_shift)
    on the reference lane WARP_REF_MX_TYPE; their forward builds the PCU
    c_structs itself from the grid's fraction bits (calibrate)."""
    qp = LayerMXQParams(x_cal=0, w_cal=0, out_cal=out_shift, mx_type=WARP_REF_MX_TYPE)
    return CalibratedMXWarp(channels=channels, grid_bit_width=grid_bits,
                            acc_bit_width=acc_bits, qparams=qp)


def run_their_warp(ref, grid, frac, grid_bits, acc_bits, out_shift=0):
    """Their CalibratedMXWarp.forward(ref, grid, frac) as is, no refusals,
    with every intermediate kept by pass-through wrappers: the packed
    reference and grid (+ the fraction bits), trc's weights and corners, and
    bcu's int64 p_sum per line. ref: [1,C,H,W] in code units (float: their
    packer rounds and clamps it); grid: grid_word."""
    layer = _warp_layer(ref.shape[1], grid_bits, acc_bits, out_shift)
    with torch.no_grad(), \
            _pass_through(_hw_layers, "pack_warp_reference") as refs, \
            _pass_through(_hw_layers, "pack_warp_grid") as grids, \
            _pass_through(_hw_pipe, "trc") as trcs, \
            _capture_bcu() as lines:
        out = layer(ref, grid, frac)
    (mx_grid, fr), = grids
    (weights, corners), = trcs
    B, C, H, W = out.shape
    return {"mx_ref": refs[0], "mx_grid": mx_grid, "frac": fr.flatten().long(),
            "weights": weights, "corners": corners,
            # bcu is called per (b, c, oh) in that loop order -> [B, C, H, W]
            "acc": torch.stack(lines).reshape(B, C, H, W), "out": out}


def warp(case, cfg, taps):
    """Backward warp on the HW team's emulator (CalibratedMXWarp ->
    MXWarp_Loops: TRC -> BCU -> PCU), as their corrected code takes it,
    nothing of theirs reimplemented.

    Inputs from the case: reference = x / ref_scale in code units (their
    packer rounds and clamps it to int8, HW Q6); grid = the flow as an int16
    relative-motion word in the config's Q format (grid_word, HW Q3); the
    fraction bits (frx, fry) from the warp entry's grid fxp at call site 0;
    the PCU output shift from the case's ref / out steps (out_cal);
    accumulator width from hw_acc_bits.

    Refused: warp_spec, out_cal and _refuse_warp say what.

    Taps (shared names compared with ALG): x flow | call_sites call_site
          frac_bits | ref_code [B,C,H,W] | grid_code [1,2,H,W] | hw_weights
          [H,W,4] hw_corners [B,C,H,W,4] | hw_acc int64 (bcu's p_sum, taken
          inside their loop) | out_shift | out_code (their PCU output, int8
          values) | out (= out_code * out_scale)
    """
    x, flow, s = case["x"], case["flow"], case["warp"]
    spec = warp_spec(cfg)
    acc_bits = hw_acc_bits(cfg)
    shift = out_cal(s)
    ref = x.double() / s["ref_scale"]
    _refuse_warp(ref, flow)
    grid = grid_word(flow, spec["frac"], spec["grid_bits"])
    r = run_their_warp(ref, grid, spec["frac"], spec["grid_bits"], acc_bits, shift)

    tap(taps, "x", x)
    tap(taps, "flow", flow)
    tap(taps, "call_sites", spec["call_sites"])
    tap(taps, "call_site", spec["call_site"])
    tap(taps, "frac_bits", r["frac"])
    tap(taps, "ref_code", r["mx_ref"].data)
    tap(taps, "grid_code", r["mx_grid"].data)
    tap(taps, "acc_bits", acc_bits)
    tap(taps, "hw_weights", r["weights"])
    tap(taps, "hw_corners", r["corners"])
    tap(taps, "hw_acc", r["acc"])
    tap(taps, "out_shift", shift)
    tap(taps, "out_code", r["out"])
    tap(taps, "out", r["out"] * s["out_scale"])
    return taps["out"]


def warp_notes(taps):
    """What a reader of an HW warp run must not miss, one line each."""
    notes = []
    w = taps["hw_weights"]
    full = 1 << WARP_WEIGHT_FIELD_BITS
    n_full = int((w >= full).sum())
    if n_full:
        notes.append(f"!! {n_full}/{w.numel()} TRC weights are 2^{WARP_WEIGHT_FIELD_BITS} "
                     f"(frx + fry = {int(taps['frac_bits'].sum())}, zero fraction): "
                     f"beyond a plain unsigned {WARP_WEIGHT_FIELD_BITS}-bit field; "
                     f"their int64 weight holds it, the HW encoding is open (HW Q4)")
    if int(taps["call_sites"]) > 1:
        notes.append(f"config call_sites {int(taps['call_sites'])}: the sim "
                     f"runs call site {int(taps['call_site'])}")
    return notes


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
    cfg = CLI.load_config(args.config)

    def run_one(name):
        taps = run(C.build(name), cfg)
        show(taps, f"{name}  HW")
        if C.CASES[name]["op"] == "warp":
            for line in warp_notes(taps):
                print(f"  {line}")

    sys.exit(CLI.run_each(C.select(args.m), "HW", run_one))


if __name__ == "__main__":
    main()
