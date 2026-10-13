"""
main_ALG.py - run one case through the MX lib (the ALG side).

One function per op. Each builds the single torch layer, lets MXQuantizer
replace it, runs it, and logs every intermediate value with `tap`.

    python main_ALG.py -m conv3x3        # one case
    python main_ALG.py -m warp           # every warp case
    python main_ALG.py -m all

    from main_ALG import run
    taps = run(cases.build("conv3x3"))   # {tap_name: tensor}

MX integer form (taps a_int / a_exp / w_int / w_exp):
    value    = mant * 2^(E - 6)            int8: mant in [-127, 127]
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
from mx_stats import _quant_operand
from fixed_point.fxp_quant import fxp_step

import cases as C
from taps import tap, show

# =============================================================================
# Config
# =============================================================================

MX_SPECS = {
    "w_elem_format": "int8",
    "a_elem_format": "int8",
    "block_size": C.BS,
    "scale_bits": 8,
    "shared_exp_method": "max",
    "custom_cuda": False,
    # conv runs on MXConv2dHW: int8 x int8 products, shifted onto a fixed
    # 48-bit accumulator with LSB 2^e_layer_min, saturating per product.
    # NPE blocking: activations along W, weights per filter flattened C-major.
    "xblock_accum": {
        "enabled": True, "mode": "hw_fixed_point", "bits": 48,
        "sat_mode": "per_product", "e_layer_min": -20,
        "act_blockify": "xblock", "weight_blockify": "flatten",
        "pad_channels": True, "backend": "python", "verbose": 0,
    },
}

E_LAYER_MIN = MX_SPECS["xblock_accum"]["e_layer_min"]


def _fxp(total_bits, scale=None, frac_bits=None):
    """Static fixed-point word: round half away, saturate."""
    d = {"total_bits": total_bits, "signed": True, "round": "half_away",
         "saturate": True, "calibrate": False}
    if scale is not None:
        d["scale"] = scale
    else:
        d["frac_bits"] = frac_bits
    return d


class _One(nn.Module):
    """Holds the single layer as `op`, so MXQuantizer has a name to replace."""

    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, *args):
        return self.op(*args)


def quantize(layer, layer_cfg="op"):
    """MXQuantizer on a one-layer model. Returns the replaced layer."""
    cfg = {"mx_specs": MX_SPECS, "layers": [layer_cfg],
           "ptq": False, "measure_error": False}
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "mx_config.json"), "w") as f:
            json.dump(cfg, f)
        with contextlib.redirect_stdout(io.StringIO()):
            q = MXQuantizer(save_dir=d).quant(_One(layer))
    return q.eval().op


# =============================================================================
# MX decode: fake-quant float -> (int mantissa, exponent code per block)
# =============================================================================

def mx_decode(q, axis, bs=C.BS, mbits=8, scale_bits=8):
    """`q` is already on the MXINT lattice; read its integers back.

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

def conv(case, taps):
    """MXConv2dHW, NPE. Taps: x w | a_int a_exp | w_int w_exp | acc sat | out."""
    x, w, p = case["x"], case["w"], case["params"]
    O, Cin, kH, kW = w.shape
    layer = nn.Conv2d(Cin, O, (kH, kW), stride=p["stride"], padding=p["padding"],
                      dilation=p["dilation"], bias=case["b"] is not None)
    with torch.no_grad():
        layer.weight.copy_(w)
        if case["b"] is not None:
            layer.bias.copy_(case["b"])
    mod = quantize(layer)
    mod._tap_acc = True                 # keep the int64 accumulator word
    sp = mod.mx_specs
    tap(taps, "x", x)
    tap(taps, "w", w)

    # activation: blocks along W
    _, qa = _quant_operand(x, sp, sp["a_elem_format"], [3], "round_output")
    a_int, a_exp = mx_decode(qa, axis=3)
    tap(taps, "a_int", a_int)           # [B, Cin, H, W]
    tap(taps, "a_exp", a_exp)           # [B, Cin, H, ceil(W/bs)]

    # weight: each filter flattened to [Cin*kH*kW] (C-major), blocks along it
    _, qw = _quant_operand(w.reshape(O, -1), sp, sp["w_elem_format"], [1],
                           "round_weight")
    w_int, w_exp = mx_decode(qw, axis=1)
    tap(taps, "w_int", w_int.reshape(w.shape))   # [O, Cin, kH, kW]
    tap(taps, "w_exp", w_exp)                    # [O, ceil(Cin*kH*kW/bs)]

    with torch.no_grad():
        out = mod(x)
    tap(taps, "acc", mod._last_acc)              # int64 [B, O, Ho, Wo]
    tap(taps, "sat", mod._last_sat.to(torch.uint8))
    tap(taps, "out", out)                        # acc * 2^e_layer_min
    return out


def convt(case, taps):
    """MXConvTranspose2d. Taps: x w | a_int a_exp | w_int w_exp | out.

    The lib has no HW path for convT: plain MX operands, fp32 accumulation, and
    blocks along Cin (input axis 1, weight axis 0) - not the NPE layout. HW does
    convT as zero-insert + conv; whether it quantizes before or after the zero
    insert is an open question. Compare `out` as float until that is settled.
    """
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
    mod = quantize(layer)
    sp = mod.mx_specs
    tap(taps, "x", x)
    tap(taps, "w", w)

    _, qa = _quant_operand(x, sp, sp["a_elem_format"], [1], "round_output")
    a_int, a_exp = mx_decode(qa, axis=1)
    tap(taps, "a_int", a_int)           # [B, Cin, H, W]
    tap(taps, "a_exp", a_exp)           # [B, ceil(Cin/bs), H, W]
    _, qw = _quant_operand(w, sp, sp["w_elem_format"], [0], "round_weight")
    w_int, w_exp = mx_decode(qw, axis=0)
    tap(taps, "w_int", w_int)           # [Cin, O, kH, kW]
    tap(taps, "w_exp", w_exp)           # [ceil(Cin/bs), O, kH, kW]

    with torch.no_grad():
        out = mod(x)
    tap(taps, "out", out)
    return out


def warp(case, taps):
    """MXActQuant around cases.Warp. Taps: x flow | ref_code grid_code |
    out_pre out_code | out.

    ref: int8, static scale.  grid: Q8.8 (int16, step 1/256 px).
    out: int8, static scale.  All round half away from zero and saturate.
    """
    x, flow, s = case["x"], case["flow"], case["warp"]
    ref_cfg = _fxp(8, scale=s["ref_scale"])
    grid_cfg = _fxp(16, frac_bits=8)
    out_cfg = _fxp(8, scale=s["out_scale"])
    mod = quantize(C.Warp(), {"name": "op", "kind": "act_quant",
                              "inputs": [{"fxp": ref_cfg}, {"fxp": grid_cfg}],
                              "output": {"fxp": out_cfg}})
    tap(taps, "x", x)
    tap(taps, "flow", flow)

    q_ref = mod.quant_input(x, 0, site=0)
    q_grid = mod.quant_input(flow, 1, site=0)
    tap(taps, "ref_code", _code(q_ref, ref_cfg, torch.int8))
    tap(taps, "grid_code", _code(q_grid, grid_cfg, torch.int16))

    with torch.no_grad():
        pre = mod.inner(q_ref, q_grid)  # bilinear on the quantized operands
        out = mod(x, flow)
    tap(taps, "out_pre", pre)
    tap(taps, "out_code", _code(out, out_cfg, torch.int8))
    tap(taps, "out", out)
    return out


def _code(q, cfg, dtype):
    """Fake-quant fixed-point value -> its integer word."""
    v = q.double() / fxp_step(cfg)
    assert bool(((v - v.round()).abs() < 1e-3).all()), "not on the fxp lattice"
    return v.round().to(dtype)


OPS = {"conv": conv, "convt": convt, "warp": warp}


def run(case):
    """One case -> {tap_name: tensor}."""
    taps = {}
    OPS[case["op"]](case, taps)
    return taps


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-m", default="all", help="case name, op (conv/convt/warp) or all")
    ap.add_argument("--dump", default=None, help="save all taps to a .pt file")
    args = ap.parse_args()

    dump = {}
    for name in C.select(args.m):
        taps = run(C.build(name))
        show(taps, f"{name}  ALG")
        dump[name] = taps
    if args.dump:
        torch.save(dump, args.dump)
        print(f"\nwrote {args.dump}")


if __name__ == "__main__":
    main()
