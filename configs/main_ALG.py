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
from mx_stats import _quant_operand
from fixed_point.fxp_quant import fxp_step

import cases as C
from taps import tap, show

# =============================================================================
# Config
# =============================================================================

DEFAULT_CONFIG = os.path.join(_HERE, "mx_config_dof.json")


def load_config(path=DEFAULT_CONFIG, layer=None, backend=None):
    """Framework mx_config.json -> sim config dict.

    Returns {"file", "mx_specs", "groups", "entries", "layer", "backend"}.
    Nothing in the file is changed except what --layer / --backend name.
    """
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(_HERE, path)
    with open(path) as f:
        raw = json.load(f)
    entries = {}
    for e in raw.get("layers", []):
        e = {"name": e} if isinstance(e, str) else e
        entries[e["name"]] = e
    if layer is not None and layer not in entries:
        raise SystemExit(f"--layer {layer!r} not in {path} layers")
    return {"file": path, "mx_specs": raw["mx_specs"],
            "groups": raw.get("groups", {}), "entries": entries,
            "layer": layer, "backend": backend}


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


def mx_decode(q, axis, bs=C.BS, mbits=8, scale_bits=8):
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
    """MXConv2dHW, NPE. Taps: x w | a_int a_exp | w_int w_exp | acc sat | out."""
    x, w, p = case["x"], case["w"], case["params"]
    O, Cin, kH, kW = w.shape
    layer = nn.Conv2d(Cin, O, (kH, kW), stride=p["stride"], padding=p["padding"],
                      dilation=p["dilation"], bias=case["b"] is not None)
    with torch.no_grad():
        layer.weight.copy_(w)
        if case["b"] is not None:
            layer.bias.copy_(case["b"])
    mod = quantize(layer, cfg, _entry(cfg, "conv"))
    mod._tap_acc = True                 # keep the int64 accumulator word
    sp = mod.mx_specs
    tap(taps, "x", x)
    tap(taps, "w", w)

    # activation: blocks along W
    _, qa = _quant_operand(x, sp, sp["a_elem_format"], [3], "round_output")
    a_int, a_exp = mx_decode(qa, axis=3, mbits=_mbits(sp["a_elem_format"]))
    tap(taps, "a_int", a_int)           # [B, Cin, H, W]
    tap(taps, "a_exp", a_exp)           # [B, Cin, H, ceil(W/bs)]

    # weight: each filter flattened to [Cin*kH*kW] (C-major), blocks along it
    _, qw = _quant_operand(w.reshape(O, -1), sp, sp["w_elem_format"], [1],
                           "round_weight")
    w_int, w_exp = mx_decode(qw, axis=1, mbits=_mbits(sp["w_elem_format"]))
    tap(taps, "w_int", w_int.reshape(w.shape))   # [O, Cin, kH, kW]
    tap(taps, "w_exp", w_exp)                    # [O, ceil(Cin*kH*kW/bs)]

    with torch.no_grad():
        out = mod(x)
    if mod._last_acc is not None:                # python backend only
        tap(taps, "acc", mod._last_acc)          # int64 [B, O, Ho, Wo]
        tap(taps, "sat", mod._last_sat.to(torch.uint8))
    tap(taps, "out", out)                        # acc * 2^e_layer_min
    return out


def convt(case, cfg, taps):
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
    mod = quantize(layer, cfg, _entry(cfg, "convt"))
    sp = mod.mx_specs
    tap(taps, "x", x)
    tap(taps, "w", w)

    _, qa = _quant_operand(x, sp, sp["a_elem_format"], [1], "round_output")
    a_int, a_exp = mx_decode(qa, axis=1, mbits=_mbits(sp["a_elem_format"]))
    tap(taps, "a_int", a_int)           # [B, Cin, H, W]
    tap(taps, "a_exp", a_exp)           # [B, ceil(Cin/bs), H, W]
    _, qw = _quant_operand(w, sp, sp["w_elem_format"], [0], "round_weight")
    w_int, w_exp = mx_decode(qw, axis=0, mbits=_mbits(sp["w_elem_format"]))
    tap(taps, "w_int", w_int)           # [Cin, O, kH, kW]
    tap(taps, "w_exp", w_exp)           # [ceil(Cin/bs), O, kH, kW]

    with torch.no_grad():
        out = mod(x)
    tap(taps, "out", out)
    return out


def warp(case, cfg, taps):
    """MXActQuant around cases.Warp. Taps: x flow | ref_code grid_code |
    out_pre out_code | out.

    ref: int8, static scale.  grid: Q8.8 (int16, step 1/256 px).
    out: int8, static scale.  All round half away from zero and saturate.
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
