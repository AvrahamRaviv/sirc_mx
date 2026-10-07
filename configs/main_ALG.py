"""
main_ALG.py - run a benchmark case through the MX lib (the ALG side).

Takes a model from `models.py`, a checkpoint (the filled weights that come with
the case), the test vectors, and a config; replaces every Conv2d / Linear with
its MX-quantized equivalent via `MXQuantizer`; returns the output per tv plus
the intermediate taps.

This is the side we already trust to be a correct MX implementation. `main_HW`
re-implements the same arithmetic independently, and `main.py` compares them.

Usage:

    python main_ALG.py -m conv1x1_ramp           # one case, print taps
    python main_ALG.py -m corner --config my.json
    python main_ALG.py -m all --dump taps_alg.pt

    from main_ALG import run
    taps = run(case, config)                     # {tap_name: tensor}
"""

import argparse
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

from microxcaling.mx.convolution import Conv2d as MXConv2d
from microxcaling.mx.linear import Linear as MXLinear
from mx_layers_blocked import MXConv2dHW
from mx_quantizer import MXQuantizer
from mx_stats import _layer_quant_axes, _quant_operand
from fixed_point.mx_fixed_point_hw import (
    extract_mxint, extract_mxint_flatten, extract_mxint_xblock,
)
from fixed_point.fxp_quant import normalize_out_quant, fxp_format_str

import models as M


# =============================================================================
# Config
# =============================================================================

# Plain MX, CPU-friendly: no custom CUDA kernel, no PTQ, no error measurement,
# so a run is deterministic and depends on nothing but the case and this dict.
DEFAULT_CONFIG = {
    "mx_specs": {
        "w_elem_format": "int8",
        "a_elem_format": "int8",
        "block_size": M.BLOCK_SIZE,
        "scale_bits": 8,
        "shared_exp_method": "max",
        "custom_cuda": False,
        # The thing under test. `mode: hw_fixed_point` makes MXQuantizer install
        # MXConv2dHW instead of MXConv2d, so the conv runs the per-product
        # int8 x int8 multiply -> signed shift -> narrow saturating accumulator
        # that the HW datapath implements, rather than an FP32 block dot-product.
        # This is what main_HW has to reproduce; plain MXConv2d is not the target.
        "xblock_accum": {
            "enabled": True,
            "mode": "hw_fixed_point",
            "bits": 48,
            "sat_mode": "per_product",
            # Exponent of the accumulator's LSB. Low enough that no product
            # gets right-shifted, so it costs nothing on 27 of the 29 cases and
            # the accumulator datapath is what gets compared, not a truncation
            # artefact. -20 is also what mx_config_npe_*.json ships, so the sim
            # and the production configs agree. The two cases it does change:
            # `with_bias` becomes exact (finer bias grid than a calibrated -2),
            # and `flush_to_zero` flushes, which is that case's whole point.
            # null = calibrate per layer instead; --e-layer-min pins any value
            # (0 right-shifts every product by 9-12 bits - see the README).
            "e_layer_min": -20,
            # Block both operands along Cin. Matches `fill_params`, which blocks
            # weights on dim 1. NPE's flatten/xblock pair lives in
            # mx_config_npe_*.json - pass it with --config to test that geometry.
            "weight_blockify": "channel",
            "act_blockify": "channel",
            "pad_channels": True,
            "backend": "python",
            "verbose": 0,
        },
    },
    "layers": [],          # filled in per model by `_layer_names`
    "ptq": False,
    "measure_error": False,
}


def load_config(path=None, bs=None):
    """Read a config JSON, or return the default. `bs` overrides block_size so
    `--bs` stays the single knob for both the model shapes and the quantizer."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))     # deep copy
    if path:
        with open(path) as f:
            cfg.update(json.load(f))
    if bs is not None:
        cfg.setdefault("mx_specs", {})["block_size"] = bs
    return cfg


def _layer_names(model):
    """Every Conv2d / Linear in the model, in module order.

    The sim quantizes the whole toy model, so the config's `layers` list is
    derived from the model rather than written by hand.
    """
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            names.append(name)
    return names


# =============================================================================
# Quantize
# =============================================================================

def quantize(model, config, out_quant=None, verbose=False):
    """Replace every Conv2d / Linear with its MX equivalent.

    `MXQuantizer` reads `mx_config.json` from a directory, so the config dict is
    written to a temp dir with the layer list filled in from the model.

    `out_quant` adds the static fixed-point output stage - a separate quantizer
    from MX (`fixed_point/fxp_quant.py`), installed as a forward hook on the
    root module so the network output leaves in a fixed Qm.n word. "" is the
    root module's name in `named_modules()`; the production configs say "model"
    because the real net sits under a `.model` attribute.
    """
    cfg = json.loads(json.dumps(config))
    if not cfg.get("layers"):
        cfg["layers"] = _layer_names(model)
    if out_quant:
        cfg["layers"] = list(cfg["layers"]) + [
            {"name": "", "kind": "out_quant", **out_quant}]

    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "mx_config.json"), "w") as f:
            json.dump(cfg, f)
        quantizer = MXQuantizer(save_dir=d)
        if not verbose:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                qmodel = quantizer.quant(model)
        else:
            qmodel = quantizer.quant(model)
    return qmodel.eval()


# =============================================================================
# Taps
# =============================================================================
#
# Tap names follow the convention in `main.py` so the two sides line up key by
# key:
#
#   <layer>/a_int, <layer>/a_exp   quantised activations: intN + shared exponent
#   <layer>/w_int, <layer>/w_exp   quantised weights, same
#   <layer>/out                    layer output
#   model/out                      network output
#
# The int taps are the ones that can be compared bit-exactly. `/out` is FP32 on
# this side, so it is compared by SQNR until the HW side reports its own
# integer output.

def _extract_padded(q_fp, bs, axis, fmt):
    """`extract_mxint` on an axis that is not a multiple of `bs`.

    The lib's `_reshape_to_blocks` zero-pads the quant axis, and zeros do not
    move a max-abs shared exponent, so padding here reproduces the blocks the
    forward pass actually formed. The pad is sliced back off the mantissas; the
    shared exponent keeps its tail block.
    """
    axis = axis % q_fp.dim()
    C = q_fp.shape[axis]
    pad = (-C) % bs
    if pad:
        shape = list(q_fp.shape)
        shape[axis] = pad
        q_fp = torch.cat([q_fp, q_fp.new_zeros(shape)], dim=axis)
    q_int, exp = extract_mxint(q_fp, bs, axis, fmt=fmt)
    if pad:
        q_int = q_int.narrow(axis, 0, C).contiguous()
    return q_int, exp


def _ints(q_fp, module, axes, bs, fmt):
    """intN mantissas + per-block shared exponent from a fake-quant tensor.

    Which extractor applies depends on how the layer blockifies: NPE mode
    flattens the weights per filter and blocks activations along W; the default
    MX path blocks both along channels.
    """
    from mx_stats import _is_npe
    try:
        if isinstance(module, MXConv2d) and _is_npe(module):
            return (extract_mxint_flatten(q_fp, bs, fmt=fmt) if q_fp.dim() == 4
                    and axes == [1] else extract_mxint_xblock(q_fp, bs, fmt=fmt))
        return _extract_padded(q_fp, bs, axes[0], fmt)
    except (ValueError, RuntimeError) as e:
        # Non-int formats (fp8 / MXFP) have no intN representation; the float
        # tap still gets compared.
        return None, None


def _make_hook(name, taps, bs):
    """Forward hook: record the quantised operands and the output of a layer."""

    def hook(module, inputs, output):
        sp = module.mx_specs
        act_axes, wt_axes = _layer_quant_axes(module)
        a_fmt, w_fmt = sp["a_elem_format"], sp["w_elem_format"]

        x = inputs[0].detach()
        _, qa = _quant_operand(x, sp, a_fmt, act_axes, "round_output")
        _, qw = _quant_operand(module.weight.detach(), sp, w_fmt, wt_axes,
                               "round_weight")

        a_int, a_exp = _ints(qa, module, act_axes, bs, a_fmt)
        w_int, w_exp = _ints(qw, module, wt_axes, bs, w_fmt)

        if a_int is not None:
            taps[f"{name}/a_int"] = a_int.cpu()
            taps[f"{name}/a_exp"] = a_exp.cpu()
        if w_int is not None:
            taps[f"{name}/w_int"] = w_int.cpu()
            taps[f"{name}/w_exp"] = w_exp.cpu()
        taps[f"{name}/out"] = output.detach().cpu()

        # MXConv2dHW counts how many outputs its narrow accumulator clamped.
        # The counters are bumped inside forward(), so they are already current
        # by the time a forward hook runs. HW must report the same two numbers.
        if isinstance(module, MXConv2dHW):
            taps[f"{name}/sat_cnt"] = torch.tensor(module._sat_seen_life,
                                                   dtype=torch.int64)
            taps[f"{name}/sat_total"] = torch.tensor(module._sat_total_life,
                                                     dtype=torch.int64)

    return hook


# =============================================================================
# Run
# =============================================================================

def run(case, config=None, verbose=False):
    """Run one case through the MX lib.

    Args:
        case: a dict from `models.build_case`.
        config: config dict (see `DEFAULT_CONFIG`); default if None.
        verbose: let MXQuantizer print its replacement summary.

    Returns:
        dict: tap name -> tensor, including "model/out". Cases carrying an
        `out_quant` also get "model/out_pre" (before the fixed-point stage),
        "model/out_code" (the integer word) and "model/out_clip".
    """
    config = config or load_config(bs=case["meta"]["block_size"])
    bs = config["mx_specs"]["block_size"]
    out_quant = case.get("out_quant")

    qmodel = quantize(case["model"], config, out_quant=out_quant, verbose=verbose)

    # Which datapath each layer actually got. MXConv2dHW is the one under test;
    # the quantizer silently falls back to MXConv2d when a layer cannot take the
    # HW path (groups != 1 is the common one), and a case that falls back is not
    # testing the accumulator at all - so record it rather than let it pass
    # unnoticed.
    case["meta"]["datapath"] = {
        name: type(mod).__name__
        for name, mod in qmodel.named_modules()
        if isinstance(mod, (MXConv2d, MXLinear))
    }

    # e_layer_min unset -> calibrate from this case's own input. One batch is
    # exact here: the case is deterministic and single-input, so the running min
    # over blocks is the true min, not a sample of it.
    xb = _get_xblock_cfg_dict(config)
    if xb.get("enabled") and xb.get("mode") == "hw_fixed_point" \
            and xb.get("e_layer_min") is None:
        from fixed_point.mx_fixed_point_hw import calibrate_e_layer_min
        with torch.no_grad():
            calibrate_e_layer_min(qmodel, [case["x"]], num_batches=1)
    case["meta"]["e_layer_min"] = {
        name: mod.e_layer_min
        for name, mod in qmodel.named_modules()
        if isinstance(mod, MXConv2dHW)
    }

    taps, handles = {}, []
    for name, mod in qmodel.named_modules():
        if isinstance(mod, (MXConv2d, MXLinear)):
            handles.append(mod.register_forward_hook(_make_hook(name, taps, bs)))

    if out_quant:
        # prepend so this runs *before* the out_quant hook and sees the output
        # the fixed-point stage is about to quantize.
        def pre_fxp(mod, inputs, output, taps=taps):
            taps["model/out_pre"] = output.detach().cpu()
        handles.append(qmodel.register_forward_hook(pre_fxp, prepend=True))

    try:
        with torch.no_grad():
            out = qmodel(case["x"])
    finally:
        for h in handles:
            h.remove()

    taps["model/out"] = out.detach().cpu()

    if out_quant:
        # The fixed-point output IS an integer word: code = value / 2^-frac.
        # That integer is what HW emits and what can be compared bit-exactly -
        # the float tap is just the same number scaled.
        oq = normalize_out_quant(out_quant)
        step = 2.0 ** -oq["frac_bits"]
        code = (taps["model/out"].double() / step).round().to(torch.int64)
        taps["model/out_code"] = code
        # Clip mask: a static scale can overflow where MX cannot, so flag which
        # elements the saturation actually touched.
        lo, hi = _fxp_range(oq)
        pre = taps.get("model/out_pre")
        if pre is not None:
            taps["model/out_clip"] = ((pre < lo) | (pre > hi)).to(torch.uint8)

    return taps


def _get_xblock_cfg_dict(config):
    """The xblock_accum sub-dict of a config, or {} when absent."""
    return (config.get("mx_specs") or {}).get("xblock_accum") or {}


def _fxp_range(oq):
    """Representable float range of a normalized out_quant config.

    `fxp_range` returns the bounds in *code* units plus the step; the float
    bounds are code * step.
    """
    from fixed_point.fxp_quant import fxp_range
    lo, hi, step = fxp_range(total_bits=oq["total_bits"],
                             frac_bits=oq["frac_bits"], signed=oq["signed"])
    return lo * step, hi * step


# =============================================================================
# CLI
# =============================================================================

def _print_taps(name, taps, case):
    y = case["y_fp32"]
    q = taps["model/out"]
    err = (y - q).flatten()
    sqnr = 10 * torch.log10(y.pow(2).sum() / err.pow(2).sum().clamp_min(1e-30))
    print(f"\n=== {name} ===")
    print(f"  {'tap':<28} {'shape':<20} {'dtype':<9} range")
    for k in sorted(taps):
        t = taps[k]
        rng = (f"[{t.min().item():g} .. {t.max().item():g}]" if t.numel()
               else "[]")
        print(f"  {k:<28} {str(tuple(t.shape)):<20} {str(t.dtype).replace('torch.',''):<9} {rng}")
    print(f"  out vs FP32: SQNR {sqnr.item():7.2f} dB   "
          f"max|err| {err.abs().max().item():.6g}")
    dp = case["meta"].get("datapath") or {}
    if dp:
        hw = [n for n, c in dp.items() if c == "MXConv2dHW"]
        soft = {n: c for n, c in dp.items() if c != "MXConv2dHW"}
        em = case["meta"].get("e_layer_min") or {}
        print(f"  datapath: {len(hw)}/{len(dp)} layers on MXConv2dHW" +
              (f"   NOT on HW path: " +
               ", ".join(f"{n}={c}" for n, c in soft.items()) if soft else "") +
              (f"   e_layer_min " +
               ", ".join(f"{n or '.'}={v}" for n, v in em.items()) if em else ""))
    oq = case["meta"].get("out_quant")
    if oq:
        n = taps["model/out_clip"].numel()
        n_clip = int(taps["model/out_clip"].sum())
        print(f"  out_quant: {fxp_format_str(normalize_out_quant(oq))}   "
              f"clipped {n_clip}/{n} ({100.0 * n_clip / max(n, 1):.1f}%)")


def main():
    parser = M.add_model_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--config", default=None, help="config JSON (default: built-in)")
    parser.add_argument("--dump", default=None, help="save all taps to a .pt file")
    parser.add_argument("--verbose", action="store_true", help="MXQuantizer logs")
    parser.add_argument("--e-layer-min", type=int, default=None, metavar="N",
                        help="pin the accumulator LSB exponent instead of "
                             "calibrating it (negative; -12 is lossless for "
                             "int8 operands at small exponents, 0 throws away "
                             "12 bits)")
    args = parser.parse_args()

    config = load_config(args.config, bs=args.bs)
    if args.e_layer_min is not None:
        _get_xblock_cfg_dict(config)["e_layer_min"] = args.e_layer_min
    all_taps = {}
    for case in M.cases_from_args(args):
        name = case["meta"]["case"]
        taps = run(case, config, verbose=args.verbose)
        all_taps[name] = taps
        _print_taps(name, taps, case)

    if args.dump:
        torch.save({"config": config, "taps": all_taps}, args.dump)
        print(f"\nwrote {args.dump}")


if __name__ == "__main__":
    main()
