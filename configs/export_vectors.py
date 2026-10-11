"""
export_vectors.py - freeze a case into an exchangeable bundle.

Writes everything the other side needs to reproduce one run, with no PyTorch
and no MX lib required to read it:

    bundle/<case>/
      manifest.json           topology, per-layer geometry and accumulator
                              config, the MX format spec, the conventions,
                              what each file is
      param_<layer>_weight    every parameter of the model, in module order
      param_<layer>_bias      (only layers that have one)
      input                   the test vector
      ref_fp32                unquantised network output - context only
      alg_out                 our network output
      alg_<layer>_<tap>       per-layer taps: a_int, a_exp, w_int, w_exp, acc,
                              sat, out (and model_out_code / _clip for cases
                              with a fixed-point output stage)

Every tensor is written twice: `.npy` for whoever uses numpy/torch, and `.txt`
for C or Excel - one value per line, C order (last axis fastest), with the
shape in a header comment. Integer taps are written as integers, so a text diff
is a bit-exact diff.

The run goes through `main_ALG.run`, which refuses to return taps that do not
rebuild their own accumulator (`tap_check.py`), so nothing inconsistent can be
exported.

    python export_vectors.py -m conv1x1_ramp
    python export_vectors.py -m conv1x1_ramp,conv3x3_randn --out bundle
    python export_vectors.py -m corner --blockify npe --clean
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import models as M
import main_ALG as ALG
import tap_check as TC
from mx_layers_blocked import MXConv2dHW


def _write(path_base, t):
    """Write one tensor as .npy and as flat .txt with a shape header."""
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    np.save(path_base + ".npy", a)
    is_int = np.issubdtype(a.dtype, np.integer)
    prec = 17 if a.dtype == np.float64 else 9
    with open(path_base + ".txt", "w") as f:
        f.write(f"# shape={list(a.shape)} dtype={a.dtype} order=C "
                f"(last axis fastest) count={a.size}\n")
        for v in a.reshape(-1):
            # float64 taps (the warp's arbitrary-real scales) need all 17
            # digits to round-trip; fp32 needs 9
            f.write(f"{int(v)}\n" if is_int else f"{float(v):.{prec}g}\n")
    return {"npy": os.path.basename(path_base) + ".npy",
            "txt": os.path.basename(path_base) + ".txt",
            "shape": list(a.shape), "dtype": str(a.dtype)}


def _fname(key):
    return key.replace("/", "_").replace(".", "_") or "model"


def _format_spec(sp, blockify):
    mbits = TC.fmt_bits(sp["a_elem_format"])
    emax = TC.scale_emax(sp["scale_bits"])
    bs = sp["block_size"]
    if blockify == "npe":
        blk = (f"activations: blocks of {bs} along W (dim 3), per (n, c, h); "
               f"a_exp is [N, C, H, ceil(W/{bs})]. weights: each filter "
               f"flattened C-major (c, kh, kw) to Cin*kH*kW and cut into blocks "
               f"of {bs} along that stream; w_exp is [O, ceil(Cin*kH*kW/{bs})]. "
               f"A tail shorter than {bs} is its own block. Conv zero padding "
               f"is applied AFTER quantization - pad zeros join no block.")
    else:
        blk = (f"both operands: blocks of {bs} along dim 1 (Cin); a_exp is "
               f"[N, ceil(C/{bs}), H, W], w_exp [O, ceil(Cin/{bs}), kH, kW]. "
               f"A tail shorter than {bs} is its own block.")
    return {
        "elem_format": sp["a_elem_format"],
        "mantissa_bits": mbits,
        "mant_bias": mbits - 2,
        "mantissa_range": [-((1 << (mbits - 1)) - 1), (1 << (mbits - 1)) - 1],
        "scale_bits": sp["scale_bits"],
        "exp_bias": emax,
        "exp_code_range": [0, 2 * emax],
        "shared_exp_method": sp["shared_exp_method"],
        "value": "value = mantissa * 2**(code - exp_bias - mant_bias)",
        "shared_exp": (f"E = floor(log2(max |x| over the block)), clamped to "
                       f"[-{emax}, {emax}]; code = E + {emax} (unsigned)"),
        "zero_block": "an all-zero block has code 0 (its exponent is undefined)",
        "mantissa": ("round(x * 2**(mant_bias - E)), ties away from zero, then "
                     "clamped to mantissa_range (an overflow to 2**(mbits-1) "
                     "keeps E and clamps)"),
        "blockify": blk,
    }


def _accum_spec(xb, sp, blockify):
    mbits = TC.fmt_bits(sp["a_elem_format"])
    order = ("for each input channel c, for each kernel tap (kh, kw): the "
             "flattened filter stream, C-major" if blockify == "npe" else
             "for each channel block, for each kernel tap, for each lane")
    return {
        "bits": xb["bits"],
        "sat_mode": xb["sat_mode"],
        "product": f"p = a_int * w_int, exact ({2 * mbits} bits)",
        "shift": (f"shift = Ea + Ew - {2 * (mbits - 2)} - e_layer_min, with Ea, "
                  f"Ew the real exponents (code - exp_bias). shift >= 0: "
                  f"p << shift; shift < 0: arithmetic p >> -shift (rounds "
                  f"toward -inf)"),
        "accumulate": (f"acc += shifted p, clamped to the signed {xb['bits']}-bit "
                       f"range after every product ({xb['sat_mode']})"),
        "order": order,
        "bias": ("round(b * 2**-e_layer_min), ties to even, added after the "
                 "last product, then clamped once more"),
        "output": ("acc is the layer's integer result; out = acc * "
                   "2**e_layer_min in fp32 (lossy above 24 bits - compare acc)"),
    }


def _warp_spec(w):
    def word(c):
        return {k: c.get(k) for k in ("total_bits", "frac_bits", "scale",
                                      "signed", "round", "saturate")
                if c.get(k) is not None}
    return {
        "call_sites": w["call_sites"],
        "ref": word(w["ref"]), "grid": word(w["grid"]), "out": word(w["out"]),
        "scales": ("ref / out steps are in alg_warp*_ref_scale / _out_scale "
                   "(one per call site; calibrated = max|x| / 127, an "
                   "arbitrary positive real). grid step = 2^-frac_bits."),
        "quantize": "code = clamp(round_half_away(x / step), lo, hi)",
        "grid_meaning": ("grid_code [B, 2, H, W]: channel 0 = x, 1 = y, a "
                         "DISPLACEMENT in pixels added to the output pixel's "
                         "own coordinate"),
        "interpolate": ("X = (x << 8) + gx; x0 = X >> 8 (arithmetic, floor); "
                        "fx = X & 255; same for y. acc = sum over the 4 "
                        "neighbours of ref_code * wx * wy, wx in {256 - fx, "
                        "fx}, wy likewise (exact integer); a neighbour outside "
                        "the image contributes 0"),
        "output": ("out_code = clamp(round_half_away(acc * ref_step / "
                   "out_step / 65536), -128, 127)"),
    }


def export(case_name, out_dir, config=None, bs=M.BLOCK_SIZE, seed=0,
           blockify=None, clean=False):
    """Freeze one case to `out_dir/<case_name>/`. Returns the manifest."""
    blockify = blockify or M.BLOCKIFY
    case = M.build_case(case_name, bs=bs, seed=seed, blockify=blockify)
    config = config or ALG.load_config(bs=bs, blockify=blockify)
    taps = ALG.run(case, config)
    cfg = case["meta"]["config"]
    sp = cfg["mx_specs"]
    xb = sp["xblock_accum"]

    d = os.path.join(out_dir, case_name)
    if clean and os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

    files = {}
    for pname, p in case["model"].named_parameters():
        files[f"param_{_fname(pname)}"] = _write(
            os.path.join(d, f"param_{_fname(pname)}"), p)
    files["input"] = _write(os.path.join(d, "input"), case["x"])
    if case.get("flow") is not None:
        files["input_flow"] = _write(os.path.join(d, "input_flow"), case["flow"])
    files["ref_fp32"] = _write(os.path.join(d, "ref_fp32"), case["y_fp32"])
    files["alg_out"] = _write(os.path.join(d, "alg_out"), taps["model/out"])
    for k, v in taps.items():
        if k == "model/out":
            continue
        name = "alg_" + _fname(k)
        files[name] = _write(os.path.join(d, name), v)

    layers = []
    for name, mod in case["model"].named_modules():
        if not isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
        dp = case["meta"]["datapath"].get(name)
        row = {"name": name, "type": type(mod).__name__, "datapath": dp,
               "bias": mod.bias is not None}
        if isinstance(mod, torch.nn.Conv2d):
            row.update(in_channels=mod.in_channels, out_channels=mod.out_channels,
                       kernel_size=list(mod.kernel_size), stride=list(mod.stride),
                       padding=list(mod.padding), dilation=list(mod.dilation),
                       groups=mod.groups)
        else:
            row.update(in_features=mod.in_features, out_features=mod.out_features)
        if dp == "MXConv2dHW":
            row["e_layer_min"] = case["meta"]["e_layer_min"][name]
        else:
            row["note"] = ("not on the HW datapath: MX-quantized operands "
                           "(channel blocks), FP32 reduction - no acc / sat taps")
        layers.append(row)

    warp = case["meta"].get("warp_config")
    if warp:
        for name in ALG._warp_names(case["model"]):
            layers.append({"name": name, "type": "Warp", "datapath": "fxp",
                           "call_sites": warp["call_sites"],
                           "inputs": ["ref = input 0 (features)",
                                      "grid = input 1 (flow, px)"]})

    int_taps = [f"alg_{_fname(k)}" for k in taps
                if k.rsplit("/", 1)[-1] in ("a_int", "a_exp", "w_int", "w_exp",
                                            "acc", "sat", "out_code", "out_clip",
                                            "ref_code", "grid_code")]
    manifest = {
        "case": case_name,
        "note": case["meta"]["note"],
        "tags": case["meta"]["tags"],
        "blockify": blockify,
        "model": {
            "class": case["meta"]["model"],
            "doc": (type(case["model"]).__doc__ or "").strip(),
            "repr": str(case["model"]),
            "forward": "ReLU / concat / add run in FP32 between quantized layers",
        },
        "layers": layers,
        "block_size": bs,
        "seed": seed,
        "patterns": {"weight": case["meta"]["w_pattern"],
                     "input": case["meta"]["x_pattern"],
                     "bias": case["meta"]["b_pattern"],
                     "w_kw": case["meta"]["w_kw"], "x_kw": case["meta"]["x_kw"],
                     "b_kw": case["meta"]["b_kw"]},
        "out_quant": case["meta"].get("out_quant"),
        "mx_format": _format_spec(sp, blockify),
        "accumulator": _accum_spec(xb, sp, blockify),
        "warp": _warp_spec(warp) if warp else None,
        "mx_config": {**cfg, "layers": ALG._layer_names(case["model"])},
        "files": files,
        "how_to_compare": {
            "bit_exact": int_taps,
            "float": ["alg_out"] + [f"alg_{_fname(k)}" for k in taps
                                    if k.endswith("/out") and k != "model/out"],
            "order": ("compare in pipeline order - a_int, a_exp, w_int, w_exp, "
                      "acc, sat - layer by layer; the first mismatch is the "
                      "root cause, the rest are downstream of it"),
            "note": ("The int taps are the contract. acc is the bit-exact "
                     "layer output; alg_*_out is acc cast to fp32 and loses bits "
                     "above 24. ref_fp32 is the unquantised output, context "
                     "only."),
        },
    }
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", "-m", default="conv1x1_ramp",
                    help="case name(s), comma-separated, or a tag, or 'all'")
    ap.add_argument("--out", default=os.path.join(_HERE, "bundle"))
    ap.add_argument("--config", default=None,
                    help="config JSON over the default (deep-merged)")
    ap.add_argument("--blockify", choices=M.BLOCKIFY_CHOICES, default=M.BLOCKIFY)
    ap.add_argument("--bs", type=int, default=M.BLOCK_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="delete each case directory before writing it, so no "
                         "stale file from an older export survives")
    args = ap.parse_args()

    names = []
    for part in args.model.split(","):
        names.extend(M.resolve_models(part.strip()))

    config = ALG.load_config(args.config, bs=args.bs, blockify=args.blockify)
    for name in names:
        man = export(name, args.out, config=config, bs=args.bs, seed=args.seed,
                     blockify=args.blockify, clean=args.clean)
        n = len(man["files"])
        print(f"{name:<24} -> {os.path.join(args.out, name)}  ({n} tensors)")


if __name__ == "__main__":
    main()
