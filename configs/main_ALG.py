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
from fixed_point.fxp_quant import fxp_step

import cases as C
from taps import tap, show

# =============================================================================
# Config
# =============================================================================

from sim_config import DEFAULT_CONFIG, load_config   # noqa: E402  (re-exported)


def _entry(cfg, op):
    """The `layers` entry for the layer under test, renamed "op"."""
    name = cfg["layer"]
    if name is None and op == "warp":
        name = next((n for n, e in cfg["entries"].items()
                     if e.get("kind") == "act_quant"), None)
        if name is None:
            raise SystemExit(f"{cfg['file']}: no act_quant entry for warp")
    if name is None:
        return "op"
    e = json.loads(json.dumps(cfg["entries"][name]))   # deep copy
    e["name"] = "op"
    return e


def _mx_config(cfg, entry):
    """The mx_config.json written for MXQuantizer: file's specs + one layer."""
    specs = json.loads(json.dumps(cfg["mx_specs"]))
    groups = json.loads(json.dumps(cfg["groups"]))
    if not torch.cuda.is_available():     # no CUDA: lib's python quantizer
        for d in (specs, *groups.values()):
            d["custom_cuda"] = False
    if cfg["backend"] is not None:
        xa = specs.get("xblock_accum")
        if isinstance(xa, dict):
            xa["backend"] = cfg["backend"]
    return {"mx_specs": specs, "groups": groups, "layers": [entry],
            "ptq": False, "measure_error": False}


class _One(nn.Module):
    """Holds the single layer as `op`, so MXQuantizer has a name to replace."""

    def __init__(self, op):
        super().__init__()
        self.op = op

    def forward(self, *args):
        return self.op(*args)


def quantize(layer, cfg, entry="op"):
    """MXQuantizer on a one-layer model, as the framework calls it. Returns the
    replaced layer."""
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "mx_config.json"), "w") as f:
            json.dump(_mx_config(cfg, entry), f)
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
    mod = quantize(layer, cfg, _entry(cfg, "conv"))
    mod._tap_acc = True                 # keep qi, qw and the int64 acc word
    sp = mod.mx_specs

    with torch.no_grad():
        out = mod(x)

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
    tap(taps, "out", out)                        # acc * 2^e_layer_min
    return out


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
    blocks along Cin (input axis 1, weight axis 0) - not the NPE layout. HW does
    convT as zero-insert + conv; whether it quantizes before or after the zero
    insert is an open question. Compare `out` as float until that is settled.

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
    mod = quantize(layer, cfg, _entry(cfg, "convt"))
    sp = mod.mx_specs

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


def warp(case, cfg, taps):
    """MXActQuant around cases.Warp. Taps: x flow | ref_code grid_code |
    out_pre out_code | out.

    ref: int8, static scale.  grid: Q8.8 (int16, step 1/256 px).
    out: int8, static scale.  All round half away from zero and saturate.
    One real forward; `_tap` makes it keep its quantized inputs, the raw
    bilinear result and the quantized output.
    """
    x, flow, s = case["x"], case["flow"], case["warp"]
    entry = _entry(cfg, "warp")
    # the file calibrates ref/out scales on data; the case fixes them
    for d, sc in ((entry["inputs"][0]["fxp"], s["ref_scale"]),
                  (entry["output"]["fxp"], s["out_scale"])):
        d.update(scale=sc, calibrate=False)
    ref_cfg = entry["inputs"][0]["fxp"]
    grid_cfg = entry["inputs"][1]["fxp"]
    out_cfg = entry["output"]["fxp"]
    mod = quantize(C.Warp(), cfg, entry)
    mod._tap = True

    with torch.no_grad():
        out = mod(x, flow)
    q_ref, q_grid = mod._last_in

    tap(taps, "x", x)
    tap(taps, "flow", flow)
    tap(taps, "ref_code", _code(q_ref, ref_cfg, torch.int8))
    tap(taps, "grid_code", _code(q_grid, grid_cfg, torch.int16))
    tap(taps, "out_pre", mod._last_pre)          # bilinear on quantized operands
    tap(taps, "out_code", _code(out, out_cfg, torch.int8))
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

    cfg = load_config(args.config, args.layer, args.backend)
    print(f"config: {cfg['file']}  layer: {args.layer or 'default'}  "
          f"backend: {args.backend or 'from file'}"
          + ("" if torch.cuda.is_available() else "  (no CUDA: custom_cuda off, python)"))
    dump = {}
    for name in C.select(args.m):
        taps = run(C.build(name), cfg)
        show(taps, f"{name}  ALG")
        dump[name] = taps
    if args.dump:
        torch.save(dump, args.dump)
        print(f"\nwrote {args.dump}")


if __name__ == "__main__":
    main()
