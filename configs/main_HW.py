"""
main_HW.py - run one case through the HW implementation (the HW side).

HW code lives in hw/npe_mx_emu/, copied verbatim from the HW team - never
edited here. This file only wires it in: it builds their layer, calls their
calibration and forward, and taps what their code produced (via a forward
pre-hook and pass-through wrappers on their module functions - no logic of
theirs is reimplemented). Status: conv and warp wired in; they have no convT.

Same interface as main_ALG:

    from main_HW import run
    taps = run(cases.build("conv3x3"))   # {tap_name: tensor}, same names as ALG

Rules:
  1. No MX lib import (microxcaling / mx_quantizer / mx_layers_*). Integer ops,
     shifts and a saturating accumulator only - if both sides shared the
     quantizer, agreeing would prove nothing. Config reading is shared on
     purpose: per-call-site fxp lists go through fixed_point/fxp_sites (pure
     python, via sim_config), so both sides accept and refuse the same configs.
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

import torch

_HW_DIR = os.path.join(_HERE, "hw")
if _HW_DIR not in sys.path:
    sys.path.insert(0, _HW_DIR)

import cases as C
import sim_cli as CLI
import sim_config as SC
from taps import tap, show

# --- HW team code (hw/npe_mx_emu, verbatim) ---------------------------------
from npe_mx_emu import CalibratedMXConv2d, LayerMXQParams, MXObserver
from npe_mx_emu import CalibratedMXWarp
from npe_mx_emu import mx_config_factory, SUPPORTED_ACC_WIDTHS
from npe_mx_emu import Cstruct, pcu, mantissa_offset
from npe_mx_emu import hw_pipe as _hw_pipe
from npe_mx_emu import layers as _hw_layers

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
    # Calibration offsets: from the config's hw_grids when it has the
    # register bases (offline PTQ, as deployed): their observer anchors the
    # smallest non-zero block at shift 0, cal = mantissa_offset - E_min, and
    # e_base_a / e_base_w ARE E_min. Else their observers on this input.
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
        import math
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
        obs_w.update(w, per_row_flatten=True)
        x_cal, w_cal = obs_x.calibrate_offset(), obs_w.calibrate_offset()
        cal_src = "observer"
    # out_cal decodes their output: the static shift moves the 2^-12 base grid
    qp = LayerMXQParams(x_cal=x_cal, w_cal=w_cal, block_size=bs, mx_type=mx_type,
                        out_cal=2 * m_off + s_shift,
                        is_input_layer=bool(case.get("input_layer")))

    layer = CalibratedMXConv2d(Cin, O, (kH, kW), stride=p["stride"],
                               padding=p["padding"], qparams=qp)
    # their CalibratedMXConv2d doesn't forward acc_bit_width; MXConv2D_Loops
    # keeps it as an attribute and passes it to bcu
    layer.mx_conv.acc_bit_width = acc_bits
    with torch.no_grad():
        layer.mx_conv.weight.copy_(w)
        layer.mx_conv.bias.copy_(b if b is not None else torch.zeros(O))
    layer.calibrate()
    if s_shift:                         # their Cstruct.shift = the static shift
        layer.mx_conv.load_c_struct(
            {o: dataclasses.replace(c, shift=s_shift)
             for o, c in layer.mx_conv.c_struct_arr.items()})
    pla = _pla_bins(case, cfg, qp, O, s_shift) if case.get("act") else None
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
    tap(taps, "cal_from_config", int(cal_src == "config"))
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


# Warp. Their reference lane: uint8 codes with zero exponents, no scale
# (pack_warp_reference's default type). Call site: SC.WARP_CALL_SITE.
WARP_REF_MX_TYPE = "mx-e4bm8"
WARP_REF_RANGE = (0, 255)       # pack_warp_reference clamps to this, silently
WARP_GRID_WORD_BITS = 16        # trc casts the grid word to int16
# frx + fry above this overflows their 16-bit weight lane: their trc_warp and
# build_warp_c_struct_arr raise, the path that runs (trc) does not check (HW Q4)
WARP_FRAC_SUM_LIMIT = 14


def warp_spec(cfg):
    """The config's warp entry, as their code takes it: grid word width
    (their grid_bit_width), the grid fraction bits per axis (frx, fry) at the
    sim's call site, and how many call sites the config declares.
    Refused when their lanes can't hold what the config describes."""
    name = SC.entry_name(cfg, "warp")
    e = cfg["entries"][name]
    ref = SC.site_fxp(e, e["inputs"][0]["fxp"], SC.WARP_CALL_SITE, f"{name} input 0")
    grid = SC.site_fxp(e, e["inputs"][1]["fxp"], SC.WARP_CALL_SITE, f"{name} input 1")
    if int(ref["total_bits"]) != 8:
        raise NotImplementedError(
            f"HW warp: {name} ref is {ref['total_bits']}-bit; their reference "
            f"lane is 8-bit ({WARP_REF_MX_TYPE}, uint8)")
    if int(grid["total_bits"]) > WARP_GRID_WORD_BITS:
        raise NotImplementedError(
            f"HW warp: {name} grid is {grid['total_bits']}-bit; their trc casts "
            f"the grid word to int{WARP_GRID_WORD_BITS}")
    if grid.get("frac_bits") is None:
        raise SystemExit(f"{os.path.basename(cfg['file'])}: {name} grid has no "
                         f"frac_bits; the sim reads the grid format from the config")
    f = int(grid["frac_bits"])
    return {"call_sites": SC.entry_call_sites(e, name), "call_site": SC.WARP_CALL_SITE,
            "grid_bits": int(grid["total_bits"]), "frac_cfg": (f, f)}


def abs_coords(flow):
    """Relative flow [B,2,H,W] in px (channel 0 = x) -> what their
    CalibratedMXWarp takes: ABSOLUTE sample positions [B,H,W,2] (x, y), in
    float64 as their notebooks build them (their weights and bcu product take
    the coordinates' float dtype; float64 keeps them exact)."""
    B, _, H, W = flow.shape
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    f = flow.double()
    return torch.stack([xx + f[:, 0], yy + f[:, 1]], dim=-1)


def _refuse_warp(ref, coords, flow, grid_bits):
    """Refuse what their warp would compute wrongly without raising."""
    B, _, H, _ = ref.shape
    if B > 1 and not all(torch.equal(flow[b], flow[0]) for b in range(1, B)):
        raise NotImplementedError(
            "HW warp: batch > 1 with different flows; their pack_warp_grid and "
            "trc read batch 0's grid only (HW Q8)")
    if H < 2:
        raise NotImplementedError(
            f"HW warp: H = {H}; their get_ref_frame reads rows up to index H-2")
    lo, hi = WARP_REF_RANGE
    r = torch.round(ref)                  # the rounding their packer applies
    if not bool(torch.isfinite(ref).all()) or float(r.min()) < lo or float(r.max()) > hi:
        raise NotImplementedError(
            f"HW warp: reference codes in [{float(r.min()):g}, {float(r.max()):g}]; "
            f"their pack_warp_reference clamps to uint8 [{lo}, {hi}] without "
            f"a word (HW Q6)")
    _, _, fr = _hw_layers.grid_quantization(coords[0], grid_bits)
    if not bool(torch.isfinite(fr).all()) or float(fr.min()) < 0:
        raise NotImplementedError(
            f"HW warp: their grid_quantization gives fraction bits "
            f"{fr.flatten().tolist()} (an axis with max |position| 0 gives "
            f"inf); their trc would split the grid word wrongly")


def _warp_layer(channels, grid_bits, acc_bits):
    """Their warp layer, with the minimal setup their example notebook uses:
    LayerMXQParams(x_cal=0, w_cal=0, out_cal=0) (output in reference code
    units, reference lane WARP_REF_MX_TYPE), then calibrate() with its default
    fraction bits (7, 7). calibrate() only fills the PCU c_structs, which
    forward() requires to be non-empty and never reads (their PCU is bypassed,
    HW Q7); nothing in it comes from data."""
    qp = LayerMXQParams(x_cal=0, w_cal=0, out_cal=0, mx_type=WARP_REF_MX_TYPE)
    layer = CalibratedMXWarp(channels=channels, grid_bit_width=grid_bits,
                             acc_bit_width=acc_bits, qparams=qp)
    layer.calibrate()
    return layer


def run_their_warp(ref, coords, grid_bits, acc_bits):
    """Their CalibratedMXWarp.forward(ref, coords) as is, no refusals, with
    every intermediate kept by pass-through wrappers: the packed reference
    and grid (+ their fraction bits), trc's weights and corners, and bcu's
    int64 p_sum per line. ref: [B,C,H,W] in code units; coords from
    abs_coords."""
    layer = _warp_layer(ref.shape[1], grid_bits, acc_bits)
    with torch.no_grad(), \
            _pass_through(_hw_layers, "pack_warp_reference") as refs, \
            _pass_through(_hw_layers, "pack_warp_grid") as grids, \
            _pass_through(_hw_pipe, "trc") as trcs, \
            _capture_bcu() as lines:
        out = layer(ref, coords)
    (mx_grid, fr), = grids
    (weights, corners), = trcs
    B, C, H, W = out.shape
    return {"mx_ref": refs[0], "mx_grid": mx_grid,
            "frac": fr.flatten().long(), "weights": weights, "corners": corners,
            # bcu is called per (b, c, oh) in that loop order -> [B, C, H, W]
            "acc": torch.stack(lines).reshape(B, C, H, W), "out": out}


def warp(case, cfg, taps):
    """Backward warp on the HW team's emulator (CalibratedMXWarp ->
    MXWarp_Loops), as their notebook calls it, nothing reimplemented.

    Inputs from the case, conventions converted: reference = x / ref_scale,
    in code units (their packer rounds half to even and clamps to uint8, no
    scale); coords = absolute positions (abs_coords). Config: grid word width
    and fraction bits (the warp entry's grid fxp, call site 0), accumulator
    width (hw_acc_bits). Their fraction bits come from the data (per frame,
    from max |position|): taps `frac_cfg` (config) and `hw_frac_bits` (theirs)
    and `frac_match` show whether they agree; their code runs on its own.

    Refused (_refuse_warp): batch > 1 with different flows, a reference their
    packer would clamp, H < 2, fraction bits their trc can't use.

    Taps: x flow | call_sites call_site frac_cfg | hw_ref_code [B,C,H,W pad 32]
          hw_grid_code [1,2,H,W] hw_frac_bits frac_match frac_sum
          frac_over | hw_weights [H,W,4] hw_corners [B,C,H,W,4] |
          hw_acc int64 (bcu's p_sum, taken inside their loop) | out_raw
          (their float32 output as is: acc / 2^(frx+fry), in code units; float32
          rounds acc above 2^24) | out (= out_raw * ref_scale)
    """
    x, flow, s = case["x"], case["flow"], case["warp"]
    spec = warp_spec(cfg)
    acc_bits = hw_acc_bits(cfg)
    ref = x.double() / s["ref_scale"]
    coords = abs_coords(flow)
    _refuse_warp(ref, coords, flow, spec["grid_bits"])
    r = run_their_warp(ref, coords, spec["grid_bits"], acc_bits)

    frac_cfg = torch.tensor(spec["frac_cfg"])
    k = int(r["frac"].sum())
    tap(taps, "x", x)
    tap(taps, "flow", flow)
    tap(taps, "call_sites", spec["call_sites"])
    tap(taps, "call_site", spec["call_site"])
    tap(taps, "frac_cfg", frac_cfg)
    tap(taps, "hw_ref_code", r["mx_ref"].data)
    tap(taps, "hw_grid_code", r["mx_grid"].data)
    tap(taps, "hw_frac_bits", r["frac"])
    tap(taps, "frac_match", int(torch.equal(frac_cfg, r["frac"])))
    tap(taps, "frac_sum", k)
    tap(taps, "frac_over", int(k > WARP_FRAC_SUM_LIMIT))
    tap(taps, "acc_bits", acc_bits)
    tap(taps, "hw_weights", r["weights"])
    tap(taps, "hw_corners", r["corners"])
    tap(taps, "hw_acc", r["acc"])
    tap(taps, "out_raw", r["out"])
    tap(taps, "out", r["out"] * s["ref_scale"])
    return taps["out"]


def warp_notes(taps):
    """What a reader of an HW warp run must not miss, one line each."""
    notes = []
    fc, fh = taps["frac_cfg"].tolist(), taps["hw_frac_bits"].tolist()
    if not int(taps["frac_match"]):
        notes.append(f"!! grid fraction bits: config {fc}, their pack_warp_grid "
                     f"derived {fh} from the data; their code ran on its own (HW Q3)")
    if int(taps["frac_over"]):
        notes.append(f"!! frx + fry = {int(taps['frac_sum'])} > "
                     f"{WARP_FRAC_SUM_LIMIT}: their weight lane limit; the path "
                     f"runs anyway (HW Q4)")
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
