"""
gen_hw_grids.py - write a sim config whose hw_grids come from an OFFLINE
calibration, as the framework does before inference. The sim itself never
calibrates; it reads these values.

    python gen_hw_grids.py --base mx_config_npe_e4.json --out mx_config_npe_e4_pla.json \
        --set out_window=infer out_round=nearest pla=infer pla_bias=pre e_layer_min=null \
        --cases conv_prelu conv_relu

For each case: build the same layer(s) main_ALG builds, replace them with the
lib's MXQuantizer, run the lib's calibrate_hw_grids on inputs from seeds
--seeds (never the test seed 0), export_hw_grids, store the entry under the
case name. Weights are the case's own (seed 0), as a deployed model's are.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cases as C
import main_ALG as A                     # sets up the lib path
import sim_config as SC
from fixed_point.mx_fixed_point_hw import calibrate_hw_grids, export_hw_grids


def _value(v):
    return json.loads(v) if v in ("null", "true", "false") or v.lstrip("-").isdigit() else v


def grids_for(name, cfg, seeds):
    case = C.build(name)
    x, w, b, p = case["x"], case["w"], case["b"], case["params"]
    O, Cin, kH, kW = w.shape
    conv = nn.Conv2d(Cin, O, (kH, kW), stride=p["stride"], padding=p["padding"],
                     dilation=p["dilation"], bias=b is not None)
    with torch.no_grad():
        conv.weight.copy_(w)
        if b is not None:
            conv.bias.copy_(b)
    layer, sub = conv, "op"
    if case["act"] is not None:
        prelu = nn.PReLU(O)
        with torch.no_grad():
            prelu.weight.copy_(case["act"]["alpha"])
        layer, sub = nn.Sequential(conv, prelu), "op.0"
    import contextlib, io, tempfile
    from mx_quantizer import MXQuantizer
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "mx_config.json"), "w") as f:
            json.dump(A._mx_config(cfg, A._entry(cfg, "conv", sub)), f)
        with contextlib.redirect_stdout(io.StringIO()):
            model = MXQuantizer(save_dir=d).quant(A._One(layer))
    data = [C.build(name, seed=s)["x"] for s in seeds]
    calibrate_hw_grids(model, data, num_batches=len(data), verbose=0)
    return export_hw_grids(model)[sub]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[],
                    help="xblock_accum key=value overrides (null/true/false/int)")
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--seeds", default="1-8", help="calibration seeds, a-b")
    args = ap.parse_args()

    base = args.base if os.path.isabs(args.base) else os.path.join(_HERE, args.base)
    raw = json.load(open(base))
    xa = raw["mx_specs"]["xblock_accum"]
    for kv in args.set:
        k, v = kv.split("=", 1)
        xa[k] = _value(v)
    raw.pop("hw_grids", None)
    lo, hi = (int(t) for t in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    if 0 in seeds:
        raise SystemExit("seed 0 is the test input; calibrate on other seeds")

    out = args.out if os.path.isabs(args.out) else os.path.join(_HERE, args.out)
    tmp = out + ".tmp"
    json.dump(raw, open(tmp, "w"), indent=1)
    cfg = SC.load_config(tmp)
    os.remove(tmp)
    raw["hw_grids"] = {n: grids_for(n, cfg, seeds) for n in args.cases}
    raw["_comment"] = (f"{os.path.basename(base)} + xblock_accum {args.set}; "
                       f"hw_grids from gen_hw_grids.py (lib calibrate_hw_grids, "
                       f"seeds {args.seeds}, not the test seed 0)")
    with open(out, "w") as f:
        json.dump(raw, f, indent=1)
        f.write("\n")
    for n, g in raw["hw_grids"].items():
        print(f"{n}: {g}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
