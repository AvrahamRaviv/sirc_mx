"""
dump_layer_shapes.py - record the real layer geometry of a network, so the
simulator can build cases in exactly those shapes.

The DOF source lives where DOF is trained, not here. Run this there, once, on
one real input, and copy the JSON into configs/dof_shapes.json:

    import sys; sys.path.insert(0, "<path to>/sirc_mx/configs")
    from dump_layer_shapes import dump_layer_shapes
    dump_layer_shapes(model, (img0, img1), "dof_shapes.json")
    # or, when the model is not called as model(*inputs):
    dump_layer_shapes(model, batch, "dof_shapes.json",
                      forward_fn=lambda m, b: m(b["img0"], b["img1"]))

Per Conv2d / ConvTranspose2d it records the full geometry plus the input shape
of its first call; per module named in `act_names` (default: anything whose
name ends in "warp") the input shapes of every call. `models.py` reads the file
and registers one `dof_<layer>` case per distinct conv geometry.
"""

import json

import torch
import torch.nn as nn


def _pair(v):
    return list(v) if isinstance(v, (tuple, list)) else [v, v]


def dump_layer_shapes(model, inputs, path, forward_fn=None, act_names=None):
    rec, handles = {"convs": [], "acts": []}, []
    by_name = {}

    def conv_hook(name):
        def hook(mod, args, out):
            if name in by_name:
                by_name[name]["calls"] += 1
                return
            row = {
                "name": name, "type": type(mod).__name__,
                "in_channels": mod.in_channels, "out_channels": mod.out_channels,
                "kernel_size": _pair(mod.kernel_size), "stride": _pair(mod.stride),
                "padding": _pair(mod.padding), "dilation": _pair(mod.dilation),
                "groups": mod.groups, "bias": mod.bias is not None,
                "input_shape": list(args[0].shape),
                "output_shape": list(out.shape), "calls": 1,
            }
            if isinstance(mod, nn.ConvTranspose2d):
                row["output_padding"] = _pair(mod.output_padding)
            by_name[name] = row
            rec["convs"].append(row)
        return hook

    def act_hook(name):
        def hook(mod, args, out):
            rec["acts"].append({
                "name": name, "type": type(mod).__name__,
                "input_shapes": [list(a.shape) for a in args
                                 if torch.is_tensor(a)],
                "output_shape": list(out.shape) if torch.is_tensor(out) else None,
            })
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(mod.register_forward_hook(conv_hook(name)))
        elif (act_names is not None and name in act_names) or \
                (act_names is None and name.endswith("warp")):
            handles.append(mod.register_forward_hook(act_hook(name)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if forward_fn is not None:
                forward_fn(model, inputs)
            elif isinstance(inputs, (tuple, list)):
                model(*inputs)
            else:
                model(inputs)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)

    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {path}: {len(rec['convs'])} convs, {len(rec['acts'])} "
          f"act calls")
    return rec
