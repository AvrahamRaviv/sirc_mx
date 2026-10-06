"""
export_vectors.py - freeze a case into an exchangeable bundle.

Writes everything the other side needs to reproduce one run, with no PyTorch
and no MX lib required to read it:

    bundle/<case>/
      manifest.json      shapes, config, the MX format spec, what each file is
      weight.{npy,txt}   the checkpoint
      bias.{npy,txt}     only when the layer has a bias
      input.{npy,txt}    the test vector
      ref_fp32.{npy,txt} unquantised conv output - the FP reference
      alg_out.{npy,txt}  our MX output - the number to compare against
      alg_<tap>.{npy,txt}  per-tap integer intermediates (a_int, a_exp, ...)

Every tensor is written twice: `.npy` for whoever uses numpy/torch, and `.txt`
for C or Excel - one value per line, C order (last axis fastest), with the
shape in a header comment. Integer taps are written as integers, so a text diff
is a bit-exact diff.

    python export_vectors.py -m conv1x1_ramp
    python export_vectors.py -m conv1x1_ramp,conv3x3_randn --out bundle
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import models as M
import main_ALG as ALG


def _write(path_base, t):
    """Write one tensor as .npy and as flat .txt with a shape header."""
    a = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    np.save(path_base + ".npy", a)
    is_int = np.issubdtype(a.dtype, np.integer)
    with open(path_base + ".txt", "w") as f:
        f.write(f"# shape={list(a.shape)} dtype={a.dtype} order=C "
                f"(last axis fastest) count={a.size}\n")
        for v in a.reshape(-1):
            f.write(f"{int(v)}\n" if is_int else f"{float(v):.9g}\n")
    return {"npy": os.path.basename(path_base) + ".npy",
            "txt": os.path.basename(path_base) + ".txt",
            "shape": list(a.shape), "dtype": str(a.dtype)}


def export(case_name, out_dir, config=None, bs=M.BLOCK_SIZE, seed=0):
    """Freeze one case to `out_dir/<case_name>/`. Returns the manifest."""
    case = M.build_case(case_name, bs=bs, seed=seed)
    config = config or ALG.load_config(bs=bs)
    taps = ALG.run(case, config)

    d = os.path.join(out_dir, case_name)
    os.makedirs(d, exist_ok=True)

    files = {}
    # The layer under test. Single-conv cases have exactly one; the walk keeps
    # the first conv/linear so this also works on the multi-layer cases.
    layer_name, layer = next(
        (n, m) for n, m in case["model"].named_modules()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)))

    files["weight"] = _write(os.path.join(d, "weight"), layer.weight)
    if layer.bias is not None:
        files["bias"] = _write(os.path.join(d, "bias"), layer.bias)
    files["input"] = _write(os.path.join(d, "input"), case["x"])
    files["ref_fp32"] = _write(os.path.join(d, "ref_fp32"), case["y_fp32"])
    files["alg_out"] = _write(os.path.join(d, "alg_out"), taps["model/out"])
    for k, v in sorted(taps.items()):
        if k == "model/out":
            continue
        name = "alg_" + k.replace("/", "_")
        files[name] = _write(os.path.join(d, name), v)

    geom = {}
    if isinstance(layer, torch.nn.Conv2d):
        geom = {"in_channels": layer.in_channels,
                "out_channels": layer.out_channels,
                "kernel_size": list(layer.kernel_size),
                "stride": list(layer.stride), "padding": list(layer.padding),
                "dilation": list(layer.dilation), "groups": layer.groups,
                "bias": layer.bias is not None}

    manifest = {
        "case": case_name,
        "note": case["meta"]["note"],
        "layer": {"name": layer_name, "type": type(layer).__name__, **geom},
        "block_size": bs,
        "seed": seed,
        "patterns": {"weight": case["meta"]["w_pattern"],
                     "input": case["meta"]["x_pattern"]},
        "out_quant": case["meta"].get("out_quant"),
        # `layers` is empty in the stored config because the mains fill it from
        # the model; record what it resolved to so the manifest is self-contained.
        "mx_config": {**config, "layers": ALG._layer_names(case["model"])},
        "files": files,
        "mx_format": {
            "elem_format": config["mx_specs"]["w_elem_format"],
            "mantissa_bits": 8,
            "mant_bias": 6,
            "mantissa_range": [-127, 127],
            "shared_exp_method": config["mx_specs"]["shared_exp_method"],
            "value": "value = mantissa * 2**(shared_exp - mant_bias)",
            "shared_exp": "shared_exp = floor(log2(max(|x|) over the block))",
            "blockify": "activations and weights block along dim 1 (channels), "
                        "block_size elements per block; a tail shorter than "
                        "block_size is zero-padded, which does not move a "
                        "max-abs shared exponent",
        },
        "how_to_compare": {
            "bit_exact": ["alg_a_int", "alg_a_exp", "alg_w_int", "alg_w_exp"],
            "float": ["alg_out"],
            "note": "The int taps are the contract. Match those first; "
                    "alg_out follows from them. ref_fp32 is the unquantised "
                    "output, for context only - neither side should match it.",
        },
    }
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", "-m", default="conv1x1_ramp",
                    help="case name(s), comma-separated, or a tag, or 'all'")
    ap.add_argument("--out", default=os.path.join(_HERE, "bundle"))
    ap.add_argument("--bs", type=int, default=M.BLOCK_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = []
    for part in args.model.split(","):
        names.extend(M.resolve_models(part.strip()))

    config = ALG.load_config(bs=args.bs)
    for name in names:
        man = export(name, args.out, config=config, bs=args.bs, seed=args.seed)
        n = len(man["files"])
        print(f"{name:<24} -> {os.path.join(args.out, name)}  ({n} tensors)")


if __name__ == "__main__":
    main()
