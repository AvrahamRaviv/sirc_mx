"""
cases.py - the test cases. One op per case, tiny tensors, fixed seed.

A case is a plain dict:

    {"name": "conv3x3", "op": "conv", "params": {...},
     "x": tensor, "w": tensor or None, "b": tensor or None,
     "flow": tensor or None, "warp": {...} or None}

    python cases.py            # list them
"""

import torch
import torch.nn as nn

BS = 32   # MX block size


# =============================================================================
# Warp - the float spec of the DOF warp (the MX lib wraps it, it has no weights)
# =============================================================================

class Warp(nn.Module):
    """Backward warp: out[b,c,y,x] = bilinear(ref[b,c], x + flow_x, y + flow_y).

    flow [B,2,H,W] is a displacement in px, channel 0 = x. x0 = floor(X),
    ax = X - x0; four neighbours weighted (1-ax)(1-ay) ... ax*ay; a neighbour
    outside the image contributes 0. float64, so with the flow on the 1/256
    grid every weight is exact.
    """

    def forward(self, ref, flow):
        B, C, H, W = ref.shape
        f = flow.double()
        X = torch.arange(W, dtype=torch.float64).view(1, 1, W) + f[:, 0]
        Y = torch.arange(H, dtype=torch.float64).view(1, H, 1) + f[:, 1]
        x0, y0 = torch.floor(X), torch.floor(Y)
        ax, ay = X - x0, Y - y0
        x0, y0 = x0.long(), y0.long()
        flat = ref.double().reshape(B, C, H * W)
        out = torch.zeros(B, C, H, W, dtype=torch.float64)
        for dy, wy in ((0, 1 - ay), (1, ay)):
            for dx, wx in ((0, 1 - ax), (1, ax)):
                yy, xx = y0 + dy, x0 + dx
                inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                idx = (yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)).view(B, 1, -1)
                v = flat.gather(2, idx.expand(B, C, -1)).view(B, C, H, W)
                out += v * (wx * wy * inside).unsqueeze(1)
        return out.to(ref.dtype)


# =============================================================================
# Case table
# =============================================================================
#
# conv  : x [B,Cin,H,W], w [O,Cin,k,k]
# convt : x [B,Cin,H,W], w [Cin,O,k,k]   (torch ConvTranspose2d layout)
# warp  : x = ref [B,C,H,W], flow [B,2,H,W] in px; static scales ref/out

P2 = 2.0 ** -5     # a power-of-two step: ref code == out code on a pure shift

CASES = {
    # --- conv -----------------------------------------------------------------
    "conv1x1":      dict(op="conv", x=(1, 4, 3, 32), w=(8, 4, 1, 1)),
    "conv3x3":      dict(op="conv", x=(1, 4, 6, 40), w=(8, 4, 3, 3), padding=1),
    "conv3x3_s2":   dict(op="conv", x=(1, 4, 6, 40), w=(8, 4, 3, 3), padding=1,
                         stride=2),
    "conv3x3_d2":   dict(op="conv", x=(1, 4, 6, 40), w=(8, 4, 3, 3), padding=2,
                         dilation=2),
    "conv3x3_bias": dict(op="conv", x=(1, 4, 6, 40), w=(8, 4, 3, 3), padding=1,
                         bias=True),
    # --- convT (HW: zero-insert = dilate the input, then a plain conv) ----------
    "convt_k3_s2":  dict(op="convt", x=(1, 4, 5, 16), w=(4, 8, 3, 3), stride=2,
                         padding=1, output_padding=1),
    "convt_k4_s2":  dict(op="convt", x=(1, 4, 5, 16), w=(4, 8, 4, 4), stride=2,
                         padding=1),
    # --- warp -----------------------------------------------------------------
    "warp_int_shift":  dict(op="warp", x=(1, 4, 5, 9), flow=("const", 3.0, -1.0),
                            ref_scale=P2, out_scale=P2),
    "warp_half_pixel": dict(op="warp", x=(1, 4, 4, 9), flow=("const", 0.5, 0.0),
                            ref_scale=P2, out_scale=P2),
    "warp_neg_frac":   dict(op="warp", x=(1, 4, 4, 8), flow=("const", -0.25, -0.75),
                            ref_scale=P2, out_scale=P2),
    "warp_oob":        dict(op="warp", x=(1, 4, 6, 10), flow=("const", -0.5, 0.5),
                            ref_scale=P2, out_scale=P2),
    "warp_subpix":     dict(op="warp", x=(1, 8, 6, 12), flow=("subpix", 3.0),
                            ref_scale=0.0331, out_scale=0.0137),
}

TAGS = ("conv", "convt", "warp")


def _flow(shape, spec, gen):
    B, _, H, W = shape
    kind = spec[0]
    if kind == "const":
        f = torch.empty(B, 2, H, W)
        f[:, 0], f[:, 1] = spec[1], spec[2]
        return f
    if kind == "subpix":          # random, on the 1/256 grid
        f = (torch.rand(B, 2, H, W, generator=gen) * 2 - 1) * spec[1]
        return torch.round(f * 256) / 256
    raise ValueError(f"flow {kind!r}")


def build(name, seed=0):
    """Case name -> case dict with its tensors filled in."""
    spec = dict(CASES[name])
    gen = torch.Generator().manual_seed(seed)
    case = {"name": name, "op": spec.pop("op"),
            "x": torch.randn(spec.pop("x"), generator=gen),
            "w": None, "b": None, "flow": None, "warp": None}
    if case["op"] in ("conv", "convt"):
        w_shape = spec.pop("w")
        case["w"] = torch.randn(w_shape, generator=gen) * 0.1
        if spec.pop("bias", False):
            o = w_shape[0] if case["op"] == "conv" else w_shape[1]
            case["b"] = torch.randn(o, generator=gen) * 0.1
        case["params"] = {"stride": 1, "padding": 0, "dilation": 1, **spec}
    else:
        case["flow"] = _flow(case["x"].shape, spec.pop("flow"), gen)
        case["warp"] = {"ref_scale": spec.pop("ref_scale"),
                        "out_scale": spec.pop("out_scale")}
        case["params"] = spec
    return case


def select(arg):
    """'-m' value -> case names: a case name, an op tag, or 'all'."""
    if arg == "all":
        return list(CASES)
    if arg in TAGS:
        return [n for n, c in CASES.items() if c["op"] == arg]
    if arg in CASES:
        return [arg]
    raise SystemExit(f"unknown case {arg!r}; have all, {', '.join(TAGS)}, "
                     f"{', '.join(CASES)}")


if __name__ == "__main__":
    for n, c in CASES.items():
        print(f"  {n:<16} {c}")
