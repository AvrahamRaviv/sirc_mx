"""
main_ALG.py - run one case through the MX lib (the ALG side).

One function per op. Each builds the single torch layer, lets MXQuantizer
replace it, runs it, and logs every intermediate value with `tap`.

    python main_ALG.py -m conv3x3                          # default config
    python main_ALG.py -m conv3x3 --config mx_config_dof_npe_int6.json
    python main_ALG.py -m convt --layer model.block2.conv1   # its group
    python main_ALG.py -m conv3x3 --backend python         # step into the ref
    python main_ALG.py -m all

    from main_ALG import run, load_config
    taps = run(cases.build("conv3x3"), load_config("mx_config_dof.json"))

Config: a real framework mx_config.json, read as is. Its `mx_specs` and
`groups` are kept; `layers` is cut to the one layer under test, renamed "op":
    --layer NAME   copy that entry of the file's `layers` (its group, kind,
                   inputs ...). Default: conv/convt -> plain "op" (global
                   mx_specs); warp -> the file's first act_quant entry.
    --backend B    override only xblock_accum.backend (python = step-able ref).
No CUDA (Mac): custom_cuda -> False and triton falls back to python, as the
lib itself does on a CPU tensor.
Warp scales: the case's static ref/out scales replace the file's calibrate.

MX integer form (taps a_int / a_exp / w_int / w_exp), intN from the config:
    value    = mant * 2^(E - (N-2))        int8: mant in [-127, 127], 2^(E-6)
    E        = floor(log2 max|block|)
    code     = E + 127                     unsigned; all-zero block -> 0
"""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _ROOT, os.path.dirname(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from mx_quantizer import MXQuantizer
from mx_layers_blocked import MXConv2dHW
from fixed_point.fxp_quant import fxp_step, normalize_out_quant

import cases as C
import sim_cli as CLI
from taps import tap, show

# =============================================================================
# Config
# =============================================================================

from sim_config import DEFAULT_CONFIG, load_config, hw_grid   # noqa: E402  (re-exported)
from sim_config import entry_name, grid_key, require, site_fxp, sites_fxp   # noqa: E402
from sim_config import warp_sites, WARP_CALL_SITE   # noqa: E402


def _entry(cfg, op, as_name="op"):
    """The `layers` entry for the layer under test, renamed `as_name` ("op",
    or "op.0" for the conv inside a conv + activation Sequential)."""
    name = entry_name(cfg, op)
    if name is None:
        return as_name
    e = json.loads(json.dumps(cfg["entries"][name]))   # deep copy
    e["name"] = as_name
    return e


def _mx_config(cfg, entry, grid=None):
    """The mx_config.json written for MXQuantizer: file's specs + one layer,
    plus that layer's hw_grids entry (re-keyed to the sim's name) so the lib
    loads the static grids itself, as in the framework."""
    specs = json.loads(json.dumps(cfg["mx_specs"]))
    groups = json.loads(json.dumps(cfg["groups"]))
    if not torch.cuda.is_available():     # no CUDA: lib's python quantizer
        for d in (specs, *groups.values()):
            d["custom_cuda"] = False
    if cfg["backend"] is not None:
        xa = specs.get("xblock_accum")
        if isinstance(xa, dict):
            xa["backend"] = cfg["backend"]
    out = {"mx_specs": specs, "groups": groups, "layers": [entry],
           "ptq": False, "measure_error": False}
    if grid is not None:
        name = entry if isinstance(entry, str) else entry["name"]
        out["hw_grids"] = {name: grid}
    return out


class _One(nn.Module):
    """Holds the single layer as `op`, so MXQuantizer has a name to replace."""

    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, *args):
        return self.op(*args)


def quantize(layer, cfg, entry="op", grid=None):
    """MXQuantizer on a one-layer model, as the framework calls it. Returns the
    replaced layer (a Sequential stays one: conv + its activation)."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "mx_config.json"), "w") as f:
            json.dump(_mx_config(cfg, entry, grid), f)
        with contextlib.redirect_stdout(io.StringIO()):
            q = MXQuantizer(save_dir=d).quant(_One(layer))
    return q.eval().op


# =============================================================================
# MX decode: fake-quant float -> (int mantissa, exponent code per block)
# =============================================================================

def _mbits(fmt):
    """'int6' -> 6."""
    assert fmt.startswith("int"), f"MX decode needs an MXINT format, got {fmt!r}"
    return int(fmt[3:])


def _dec(sp, which):
    """mx_decode widths from the layer's mx_specs: block size, mantissa bits
    of the a/w format, shared-exponent bits (scale_bits)."""
    return {"bs": int(sp["block_size"]),
            "mbits": _mbits(sp[f"{which}_elem_format"]),
            "scale_bits": int(sp["scale_bits"])}


def mx_decode(q, axis, bs=C.BS, mbits=8, scale_bits=8):
    # callers pass bs, mbits and scale_bits from the layer's mx_specs
    """`q` is already on the MXINT lattice; read its integers back.
    intN: value = mant * 2^(E - (N-2)), mant in [-(2^(N-1)-1), 2^(N-1)-1].

    Returns (mant int8, code int16 with `axis` reduced to ceil(n/bs)).
    """
    emax = (1 << (scale_bits - 1)) - 1
    x = q.detach().double().movedim(axis, -1)
    n = x.shape[-1]
    x = nn.functional.pad(x, (0, (-n) % bs))
    xb = x.reshape(*x.shape[:-1], -1, bs)

    m = xb.abs().amax(-1, keepdim=True)
    nz = m > 0
    E = torch.floor(torch.log2(torch.where(nz, m, torch.ones_like(m))))
    E = torch.where(nz, E.clamp(min=-emax), torch.full_like(E, -emax))
    mant = xb * 2.0 ** (mbits - 2 - E)
    assert bool((mant == mant.round()).all()), "not on the MXINT lattice"

    mant = mant.reshape(x.shape)[..., :n].movedim(-1, axis).to(torch.int8)
    code = (E.squeeze(-1) + emax).to(torch.int16).movedim(-1, axis)
    return mant.contiguous(), code.contiguous()


# =============================================================================
# Ops
# =============================================================================

def _need_npe_conv(cfg, mod):
    """The conv taps decode the lib's HW conv (MXConv2dHW) on NPE blocks
    (activations along W, weights per flattened filter). A config that gives
    another layer or blocking is refused, instead of decoding it wrongly."""
    xa = cfg["mx_specs"].get("xblock_accum")
    xa = xa if isinstance(xa, dict) else {}
    blk = (xa.get("act_blockify", "channel"), xa.get("weight_blockify", "channel"))
    if not isinstance(mod, MXConv2dHW) or blk != ("xblock", "flatten"):
        raise SystemExit(
            f"{os.path.basename(cfg['file'])}: ALG conv taps need the lib's HW conv "
            f"on NPE blocks (xblock_accum mode hw_fixed_point, act_blockify xblock, "
            f"weight_blockify flatten); this config gives {type(mod).__name__}, "
            f"blockify {blk[0]} / {blk[1]}")


def conv(case, cfg, taps):
    """MXConv2dHW, NPE. Taps: x w | a_int a_exp | w_int w_exp | acc sat | out.

    Every tap is the layer's own value: `_tap_acc` makes its forward keep the
    quantized operands it fed the accumulator (qi, qw) and the acc word.
    """
    x, w, p = case["x"], case["w"], case["params"]
    O, Cin, kH, kW = w.shape
    layer = nn.Conv2d(Cin, O, (kH, kW), stride=p["stride"], padding=p["padding"],
                      dilation=p["dilation"], bias=case["b"] is not None)
    with torch.no_grad():
        layer.weight.copy_(w)
        if case["b"] is not None:
            layer.bias.copy_(case["b"])
    grid = hw_grid(cfg, case)           # static grids from the config only
    act = case.get("act")
    if act is not None:
        # conv + PReLU in a Sequential: the lib folds the activation into the
        # PCU multiply (PLA) when xblock_accum.pla is on; else it runs after.
        prelu = nn.PReLU(O)
        with torch.no_grad():
            prelu.weight.copy_(act["alpha"])
        seq = quantize(nn.Sequential(layer, prelu), cfg,
                       _entry(cfg, "conv", "op.0"), grid)
        mod = seq[0]
    else:
        mod = seq = quantize(layer, cfg, _entry(cfg, "conv"), grid)
    _need_npe_conv(cfg, mod)
    mod._tap_acc = True                 # keep qi, qw and the int64 acc word
    sp = mod.mx_specs

    with torch.no_grad():
        out = seq(x)

    tap(taps, "x", x)
    tap(taps, "w", w)
    # activation: blocks along W
    a_int, a_exp = mx_decode(mod._last_qi, axis=3, **_dec(sp, "a"))
    tap(taps, "a_int", a_int)           # [B, Cin, H, W]
    tap(taps, "a_exp", a_exp)           # [B, Cin, H, ceil(W/bs)]
    # weight: each filter flattened to [Cin*kH*kW] (C-major), blocks along it
    w_int, w_exp = mx_decode(mod._last_qw.reshape(O, -1), axis=1, **_dec(sp, "w"))
    tap(taps, "w_int", w_int.reshape(w.shape))   # [O, Cin, kH, kW]
    tap(taps, "w_exp", w_exp)                    # [O, ceil(Cin*kH*kW/bs)]
    if mod._last_acc is not None:                # python backend only
        tap(taps, "acc", mod._last_acc)          # int64 [B, O, Ho, Wo]
        tap(taps, "sat", mod._last_sat.to(torch.uint8))
    if mod.e_layer_min is not None:
        tap(taps, "e_layer_min", int(mod.e_layer_min))
    if mod._last_win is not None:                # output window on
        tap(taps, "e_qa_out", int(mod.e_qa_out))
        if mod._pla_last:                        # PLA applied in this forward
            pla = mod.pla_params()
            tap(taps, "pla_frac", int(pla.frac))
            tap(taps, "pla_m_pos", pla.m_pos)
            tap(taps, "pla_m_neg", pla.m_neg)
            if pla.bias == "bins":
                tap(taps, "pla_bin_scales", pla.scales)
                tap(taps, "pla_bin_biases", pla.biases)
                tap(taps, "pla_bin_bounds", pla.bounds)
        tap(taps, "win", mod._last_win)          # int64 O on 2^e_qa_out
        tap(taps, "win_sat", mod._last_win_sat.to(torch.uint8))
    # out: acc * 2^e_layer_min (no window) | win * 2^e_qa_out (window; after
    # PLA when folded)
    tap(taps, "out", out)
    return out


def pcu_on(acc, cfg, case, e_out):
    """The lib's PCU steps (PLA bins + output window) on an accumulator word
    from anywhere - e.g. HW's own - so the PCU is compared alone. Bins, frac,
    tie, window width, rounding and saturation word from the config; output
    grid 2^e_out.
    acc: int64 [B, O, ...] on the config's e_layer_min grid."""
    from fixed_point.mx_fixed_point_hw import (PLAParams, _apply_pla,
                                               _apply_out_window, out_window_sat_word,
                                               resolve_out_window_bits)
    g = hw_grid(cfg, case)
    f = os.path.basename(cfg["file"])
    require(g or {}, ("e_layer_min", "pla_frac", "pla_m_pos", "pla_m_neg", "pla_bin_bounds",
                      "pla_bin_scales", "pla_bin_biases"),
            f"{f}: hw_grids {grid_key(cfg, case)!r}", "the lib's PCU reads its PLA bins "
            "and grids from the config")
    require(cfg["mx_specs"], ("xblock_accum",), f"{f}: mx_specs",
            "the lib's PCU reads its window and rounding from it")
    xa = cfg["mx_specs"]["xblock_accum"]
    t = lambda k: torch.tensor(g[k], dtype=torch.int64)
    pla = PLAParams(t("pla_m_pos"), t("pla_m_neg"), int(g["pla_frac"]), "bins",
                    t("pla_bin_bounds"), t("pla_bin_scales"), t("pla_bin_biases"),
                    xa.get("pla_bin_tie", "upper"))
    B, O = acc.shape[:2]
    word = _apply_pla(acc.reshape(B, O, -1), pla)
    fmt = cfg["mx_specs"]["a_elem_format"]
    win = resolve_out_window_bits(xa, fmt)
    o, sat = _apply_out_window(word, int(g["e_layer_min"]) - pla.frac, e_out, win,
                               xa.get("out_round", "floor"),
                               out_window_sat_word(xa, fmt))
    return o.reshape(acc.shape), sat.reshape(acc.shape)


@contextlib.contextmanager
def _capture_mx(module):
    """Record every quantize_mx_op output that `module` calls, in call order.

    For layers in microxcaling (upstream, not edited here) whose forward keeps
    no handle on its quantized operands.
    """
    seen = []
    orig = module.quantize_mx_op

    def rec(*a, **k):
        q = orig(*a, **k)
        seen.append(q.detach())
        return q

    module.quantize_mx_op = rec
    try:
        yield seen
    finally:
        module.quantize_mx_op = orig


def convt(case, cfg, taps):
    """MXConvTranspose2d. Taps: x w | a_int a_exp | w_int w_exp | out.

    The lib has no HW path for convT: plain MX operands, fp32 accumulation, and
    blocks along Cin (input axis 1, weight axis 0) - not the NPE layout. HW
    quantizes the compact input, then zero-inserts and runs a conv (QA Q7);
    compare `out` as float.

    The operands are the layer's own: captured from its quantize_mx_op calls
    (input first, then weight).
    """
    from microxcaling.mx import transpose_convolution as TC

    x, w, p = case["x"], case["w"], case["params"]
    Cin, O, kH, kW = w.shape
    layer = nn.ConvTranspose2d(Cin, O, (kH, kW), stride=p["stride"],
                               padding=p["padding"],
                               output_padding=p.get("output_padding", 0),
                               dilation=p["dilation"], bias=case["b"] is not None)
    with torch.no_grad():
        layer.weight.copy_(w)
        if case["b"] is not None:
            layer.bias.copy_(case["b"])
    mod = quantize(layer, cfg, _entry(cfg, "convt"), hw_grid(cfg, case))
    sp = mod.mx_specs

    if isinstance(mod, MXConv2dHW):
        # xblock_accum.convT = "hw": the lib's MXConvTranspose2dHW quantizes
        # the compact input (NPE blocks along W), zero-inserts the extracted
        # operand and runs the HW conv. Taps as conv; w_int / w_exp are its
        # equivalent conv weight (transposed, flipped) [O, Cin, kH, kW].
        _need_npe_conv(cfg, mod)
        mod._tap_acc = True
        with torch.no_grad():
            out = mod(x)
        tap(taps, "x", x)
        tap(taps, "w", w)
        a_int, a_exp = mx_decode(mod._last_qi, axis=3, **_dec(sp, "a"))
        tap(taps, "a_int", a_int)
        tap(taps, "a_exp", a_exp)
        wc = mod._last_qw
        w_int, w_exp = mx_decode(wc.reshape(wc.shape[0], -1), axis=1, **_dec(sp, "w"))
        tap(taps, "w_int", w_int.reshape(wc.shape))
        tap(taps, "w_exp", w_exp)
        if mod._last_acc is not None:
            tap(taps, "acc", mod._last_acc)
            tap(taps, "sat", mod._last_sat.to(torch.uint8))
        tap(taps, "out", out)
        return out

    with torch.no_grad(), _capture_mx(TC) as seen:
        out = mod(x)
    assert len(seen) == 2, f"convT forward made {len(seen)} MX quantize calls"
    qa, qw = seen

    tap(taps, "x", x)
    tap(taps, "w", w)
    a_int, a_exp = mx_decode(qa, axis=1, **_dec(sp, "a"))
    tap(taps, "a_int", a_int)           # [B, Cin, H, W]
    tap(taps, "a_exp", a_exp)           # [B, ceil(Cin/bs), H, W]
    w_int, w_exp = mx_decode(qw, axis=0, **_dec(sp, "w"))
    tap(taps, "w_int", w_int)           # [Cin, O, kH, kW]
    tap(taps, "w_exp", w_exp)           # [ceil(Cin/bs), O, kH, kW]
    tap(taps, "out", out)
    return out


def _fix_scale(operand, sites, scale):
    """The case's static scale in place of the file's (calibrate or a
    frozen scale), at every call site the lib quantizes. `sites`: the
    operand's fxp per call site, read by the lib's rule (sim_config.sites_fxp),
    so a list and a single spec are accepted and refused exactly as the lib
    does; a site the lib reads as off (normalize_out_quant) stays off, and
    `true` is the lib's defaults."""
    operand["fxp"] = [dict({} if d is True else d, scale=scale, calibrate=False)
                      if normalize_out_quant(d)["enabled"] else d for d in sites]


def warp_out_note(cfg):
    """None when the config's warp entry rounds the output to an fxp word at
    the call site ALG runs (sim_config.WARP_CALL_SITE), else the line saying
    what the lib does instead: with no `output` entry, or an output fxp the lib reads as off
    (false, null, {"enabled": false}: normalize_out_quant), it leaves the
    warp result unrounded (MXActQuant.quant_output passes it through); with a
    non-fxp `output` it holds no fixed-point word either. No format is
    invented: the sim then has no out_code."""
    e = cfg["entries"][entry_name(cfg, "warp")]
    out = e.get("output")
    if out is None:
        return (f"ALG out: warp entry {e['name']!r} has no output fxp; the lib "
                f"leaves the warp output unrounded (float), so there is no out_code")
    if "fxp" not in out:
        return (f"ALG out: warp entry {e['name']!r} output is not fxp; the lib "
                f"gives no fixed-point output word, so there is no out_code")
    site = site_fxp(e, out["fxp"], WARP_CALL_SITE, f"{e['name']} output")
    if not normalize_out_quant(site)["enabled"]:
        return (f"ALG out: warp entry {e['name']!r} output fxp is off; the lib "
                f"leaves the warp output unrounded (float), so there is no out_code")
    return None


def warp(case, cfg, taps):
    """MXActQuant around cases.Warp. Taps: x flow | ref_code grid_code |
    out_pre out_code | out.

    ref: int8, static scale.  grid: Q8.8 (int16, step 1/256 px).
    out: int8, static scale.  Each rounds by its fxp's `round` and
    saturates; the border is cases.Warp's (the HW rule).
    One real forward of a fresh module: call site WARP_CALL_SITE (0) of a
    per-site config.
    `_tap` makes it keep its quantized inputs, the raw bilinear result and
    the quantized output. An entry with no output fxp: out as the lib gives
    it and no out_code (warp_out_note says why).
    """
    x, flow, s = case["x"], case["flow"], case["warp"]
    entry = _entry(cfg, "warp")
    name = entry["name"]
    # both inputs read by the lib's rule (refused when missing); the file
    # calibrates ref/out scales on data, the case fixes them
    ref_sites, _ = warp_sites(cfg, entry)
    _fix_scale(entry["inputs"][0], ref_sites, s["ref_scale"])
    out_rounded = warp_out_note(cfg) is None
    if out_rounded:
        output = entry["output"]
        _fix_scale(output, sites_fxp(entry, output["fxp"], f"{name} output"), s["out_scale"])
    mod = quantize(C.Warp(), cfg, entry)
    mod._tap = True

    with torch.no_grad():
        out = mod(x, flow)
    q_ref, q_grid = mod._last_in

    # the formats the lib applied in this forward (its call site)
    tap(taps, "x", x)
    tap(taps, "flow", flow)
    tap(taps, "ref_code", _code(q_ref, mod.fxp_cfg(0), torch.int8))
    tap(taps, "grid_code", _code(q_grid, mod.fxp_cfg(1), torch.int16))
    tap(taps, "out_pre", mod._last_pre)          # bilinear on quantized operands
    if out_rounded:
        tap(taps, "out_code", _code(out, mod.out_fxp_cfg(), torch.int8))
    tap(taps, "out", out)
    return out


def _code(q, cfg, dtype):
    """Fake-quant fixed-point value -> its integer word."""
    v = q.double() / fxp_step(cfg)
    assert bool(((v - v.round()).abs() < 1e-3).all()), "not on the fxp lattice"
    return v.round().to(dtype)


OPS = {"conv": conv, "convt": convt, "warp": warp}


def run(case, cfg=None):
    """One case -> {tap_name: tensor}. cfg from load_config (default file)."""
    cfg = cfg if cfg is not None else load_config()
    taps = {}
    OPS[case["op"]](case, cfg, taps)
    return taps


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-m", default="all", help="case name, op (conv/convt/warp) or all")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="framework mx_config.json (default mx_config_dof.json)")
    ap.add_argument("--layer", default=None,
                    help="entry of the config's layers to use (its group/kind)")
    ap.add_argument("--backend", default=None,
                    help="override xblock_accum.backend (python / triton)")
    ap.add_argument("--dump", default=None, help="save all taps to a .pt file")
    args = ap.parse_args()

    cfg = CLI.load_config(args.config, args.layer, args.backend)
    print(f"config: {cfg['file']}  layer: {args.layer or 'default'}  "
          f"backend: {args.backend or 'from file'}"
          + ("" if torch.cuda.is_available() else "  (no CUDA: custom_cuda off, python)"))
    dump = {}

    def run_one(name):
        taps = run(C.build(name), cfg)
        show(taps, f"{name}  ALG")
        note = warp_out_note(cfg) if C.CASES[name]["op"] == "warp" else None
        if note is not None:
            print(f"  {note}")
        dump[name] = taps

    code = CLI.run_each(C.select(args.m), "ALG", run_one)
    if args.dump:
        torch.save(dump, args.dump)
        print(f"\nwrote {args.dump} ({len(dump)} cases that ran)")
    sys.exit(code)


if __name__ == "__main__":
    main()
