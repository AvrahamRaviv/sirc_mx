"""
models.py - benchmark of small models for the MX single-layer simulator.

Everything here is plain `torch.nn` so that both sides of the comparison can
consume it: `ALG_main` replaces the ops with the MX lib, `HW_main` replaces them
with the HW implementation, and arch can re-create the same topology in
torch / C / Excel from the printed description alone.

Three pieces:

  * Model builders (`single_conv`, `conv_relu_conv`, `dw_pw`, ...) - parametric,
    so one builder covers many corner cases by changing channels / stride /
    groups rather than by adding another model.
  * Tensor patterns (`PATTERNS`) - deterministic weight / input fillings that
    stress a specific part of the MX pipeline (block underflow, accumulator
    saturation, a single non-zero per block, ...).
  * `CASES` - a named registry pairing a builder with the shapes and the
    patterns that make it interesting. `build_case(name)` materialises one.

Usage:

    from models import CASES, build_case, list_cases

    list_cases()                       # print the table
    c = build_case("tail_block_c33")   # -> {"model", "x", "spec", ...}
    y = c["model"](c["x"])             # FP32 reference
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK_SIZE = 32  # default MX block size; cases are parametric in it

# Which way the operands are cut into blocks. Arch confirmed NPE:
#   npe      activations block along W (per b, c, y); each weight filter is
#            flattened to [Cin*kH*kW] (C-major) and blocked along that stream.
#   channel  both operands block along Cin - the plain MX lib layout.
# Patterns that shape a block ("one big element per block", "+ then -") have to
# know this, or under NPE they shape something that is not a block at all.
BLOCKIFY = "npe"
BLOCKIFY_CHOICES = ("npe", "channel")


def act_block_axis(ndim, blockify=BLOCKIFY):
    """Axis an activation is blocked along: W for NPE convs, else channels /
    features."""
    if ndim == 4:
        return 3 if blockify == "npe" else 1
    return ndim - 1


# =============================================================================
# Models
# =============================================================================

class SingleConv(nn.Module):
    """One Conv2d. Covers plain / strided / dilated / grouped / depthwise."""

    def __init__(self, cin, cout, k=3, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ConvReLU(nn.Module):
    """Conv2d -> ReLU. Checks where the clamp sits relative to the requant."""

    def __init__(self, cin, cout, k=3, padding=0, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=padding, bias=bias)

    def forward(self, x):
        return F.relu(self.conv(x))


class ConvReLUConv(nn.Module):
    """Conv2d -> ReLU -> Conv2d. The first model with a layer-to-layer handoff:
    the second conv's activations are the first conv's requantised output."""

    def __init__(self, cin, cmid, cout, k=3, padding=0, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cmid, k, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(cmid, cout, k, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class DwPw(nn.Module):
    """Depthwise 3x3 -> pointwise 1x1, the NPE backbone pattern.

    Depthwise is the hard case for blockification: each output channel reduces
    over k*k values only, so a block is never filled by the channel axis.
    """

    def __init__(self, cin, cout, k=3, padding=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, k, padding=padding, groups=cin, bias=bias)
        self.pw = nn.Conv2d(cin, cout, 1, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))


class ConvConcat(nn.Module):
    """Two parallel convs -> channel concat -> conv.

    The concat makes the consumer's channel axis a sum of two independently
    quantised tensors, so block boundaries no longer line up with the producers.
    """

    def __init__(self, cin, cb0, cb1, cout, k=3, padding=1, bias=False):
        super().__init__()
        self.b0 = nn.Conv2d(cin, cb0, k, padding=padding, bias=bias)
        self.b1 = nn.Conv2d(cin, cb1, 1, bias=bias)
        self.head = nn.Conv2d(cb0 + cb1, cout, 1, bias=bias)

    def forward(self, x):
        return self.head(torch.cat([self.b0(x), self.b1(x)], dim=1))


class ConvAdd(nn.Module):
    """Residual add of two conv branches, then a conv.

    Exercises an elementwise add of two quantised tensors - the add itself is
    outside the MX path, so this is where SW and HW disagree about what
    precision the sum is kept in.
    """

    def __init__(self, cin, cout, k=3, padding=1, bias=False):
        super().__init__()
        self.b0 = nn.Conv2d(cin, cout, k, padding=padding, bias=bias)
        self.b1 = nn.Conv2d(cin, cout, 1, bias=bias)
        self.head = nn.Conv2d(cout, cout, 1, bias=bias)

    def forward(self, x):
        return self.head(self.b0(x) + self.b1(x))


class SingleLinear(nn.Module):
    """One Linear. The flatten-blockify path without any conv geometry."""

    def __init__(self, fin, fout, bias=False):
        super().__init__()
        self.fc = nn.Linear(fin, fout, bias=bias)

    def forward(self, x):
        return self.fc(x)


# --- warp (DOF) --------------------------------------------------------------

class Warp(nn.Module):
    """Backward warp: out[b,c,y,x] = bilinear(ref[b,c], x + flow_x, y + flow_y).

    This is the spec the warp cases are built on, written out instead of going
    through grid_sample so every step is visible:

      * `flow` [B, 2, H, W] is a DISPLACEMENT in pixels (channel 0 = x,
        channel 1 = y), added to the output pixel's own integer coordinate.
      * x0 = floor(X), ax = X - x0 (same for y); the four neighbours are
        weighted (1-ax)(1-ay), ax(1-ay), (1-ax)ay, ax*ay.
      * a neighbour outside the image contributes 0 (zero padding).

    Computed in float64: with the grid on the 1/256 lattice the weights are
    exact, so the only rounding left is the output quantizer's - the float
    model and an integer bilinear agree bit for bit (`tap_check.ref_warp`).
    DOF's production warp goes through grid_sample in fp32, whose coordinate
    normalise / unnormalise round trip is not exact; see main_HW.py.
    """

    def forward(self, ref, flow):
        B, C, H, W = ref.shape
        r = ref.double()
        f = flow.double()
        X = torch.arange(W, dtype=torch.float64).view(1, 1, W) + f[:, 0]
        Y = torch.arange(H, dtype=torch.float64).view(1, H, 1) + f[:, 1]
        x0, y0 = torch.floor(X), torch.floor(Y)
        ax, ay = X - x0, Y - y0
        x0, y0 = x0.long(), y0.long()
        out = torch.zeros(B, C, H, W, dtype=torch.float64)
        flat = r.reshape(B, C, H * W)
        for dy, wy in ((0, 1 - ay), (1, ay)):
            for dx, wx in ((0, 1 - ax), (1, ax)):
                yy, xx = y0 + dy, x0 + dx
                inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                idx = (yy.clamp(0, H - 1) * W + xx.clamp(0, W - 1)).view(B, 1, -1)
                v = flat.gather(2, idx.expand(B, C, -1)).view(B, C, H, W)
                out += v * (wx * wy * inside).unsqueeze(1)
        return out


class WarpOnly(nn.Module):
    """The warp alone: ref and flow are both network inputs."""

    def __init__(self):
        super().__init__()
        self.warp = Warp()

    def forward(self, x, flow):
        return self.warp(x, flow)


class ConvWarp(nn.Module):
    """Conv2d -> warp, the DOF feature path: the warp reference is a conv's
    output, so its int8 ref code is a requant of that conv's accumulator."""

    def __init__(self, cin, cout, k=3, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=padding, bias=bias)
        self.warp = Warp()

    def forward(self, x, flow):
        return self.warp(self.conv(x), flow)


class WarpLevels(nn.Module):
    """One warp instance called once per pyramid level, coarse to fine, as in
    DOF (call_sites = 3). Level l works at 1/2^l resolution on features that
    grow 4x in magnitude per level, so a single static scale cannot fit all
    three - each call site gets its own. Output: the three warped levels,
    flattened and concatenated."""

    def __init__(self, levels=3, gain=4.0):
        super().__init__()
        self.levels, self.gain = levels, gain
        self.warp = Warp()

    def forward(self, x, flow):
        outs = []
        for lvl in range(self.levels):
            s = 2 ** (self.levels - 1 - lvl)
            xs = F.avg_pool2d(x, s) * self.gain ** lvl if s > 1 else \
                x * self.gain ** lvl
            fs = F.avg_pool2d(flow, s) / s if s > 1 else flow
            outs.append(self.warp(xs, fs).flatten(1))
        return torch.cat(outs, dim=1)


# =============================================================================
# Tensor patterns
# =============================================================================
#
# Each pattern fills a tensor so that one specific part of the MX pipeline is
# stressed. Patterns are applied block-wise along `axis` in steps of `bs`, so a
# trailing partial block is filled the same way as a full one.
#
# All patterns are deterministic given `seed`: the same case rebuilt on either
# side produces the same numbers, and the random ones are exported anyway.

def _blocks(t, axis, bs):
    """Yield writable views of `t`, one per block of `bs` along `axis`."""
    n = t.shape[axis]
    for start in range(0, n, bs):
        yield t.narrow(axis, start, min(bs, n - start))


def _randn(shape, axis, bs, gen, scale=1.0):
    return torch.randn(shape, generator=gen) * scale


def _ramp(shape, axis, bs, gen, scale=1.0):
    """arange over the flat tensor, normalised to [-1, 1]. Trivial to re-create
    on the other side - the first vector to run when nothing matches yet."""
    n = int(torch.tensor(shape).prod())
    t = torch.arange(n, dtype=torch.float32)
    t = (t / max(n - 1, 1)) * 2.0 - 1.0
    return (t * scale).reshape(shape)


def _int_ramp(shape, axis, bs, gen, scale=1.0):
    """Small integers 1..bs cycling along `axis`. Exactly representable, so any
    mismatch is a pipeline bug and not a rounding difference."""
    t = torch.zeros(shape)
    for blk in _blocks(t, axis, bs):
        n = blk.shape[axis]
        idx = torch.arange(1, n + 1, dtype=torch.float32)
        blk.copy_(idx.reshape([-1 if d == axis % t.dim() else 1
                               for d in range(t.dim())]).expand_as(blk))
    return t * scale


def _ones(shape, axis, bs, gen, scale=1.0):
    return torch.full(shape, float(scale))


def _single_nonzero(shape, axis, bs, gen, scale=1.0):
    """One non-zero per block, everything else exactly 0. The shared exponent is
    set by that single element, so this isolates the shift path from the sum."""
    t = torch.zeros(shape)
    for blk in _blocks(t, axis, bs):
        blk.narrow(axis, 0, 1).fill_(float(scale))
    return t


def _wide_dyn(shape, axis, bs, gen, ratio=2.0 ** -10, scale=1.0):
    """One large element per block, the rest `ratio` times smaller.

    The large element sets the shared exponent, so the small ones fall off the
    bottom of the mantissa and quantise to 0 - the dominant MX loss mode, and
    the case where a sign or bias error in the shift is immediately visible.
    """
    t = torch.full(shape, float(scale) * ratio)
    for blk in _blocks(t, axis, bs):
        blk.narrow(axis, 0, 1).fill_(float(scale))
    return t


def _underflow(shape, axis, bs, gen, scale=1.0):
    """Same idea as `wide_dyn` but with random small values, so the underflow
    count per block varies instead of being all-or-nothing."""
    t = torch.rand(shape, generator=gen) * (float(scale) * 2.0 ** -9)
    for blk in _blocks(t, axis, bs):
        blk.narrow(axis, 0, 1).fill_(float(scale))
    return t


def _tiny(shape, axis, bs, gen, scale=2.0 ** -30):
    """Everything below any sane `e_layer_min`: the whole block should flush to
    zero. Checks that the clamp triggers, and that it triggers on both sides."""
    return torch.full(shape, float(scale))


def _saturate(shape, axis, bs, gen, scale=1.0):
    """All elements at full magnitude with the same sign, so every product adds
    in the same direction and the narrow accumulator saturates. Whether HW
    saturates per product or per block is visible only here."""
    return torch.full(shape, float(scale))


def _half_split(shape, axis, bs, gen, scale=1.0):
    """First half of every block +max, second half -max.

    Unlike `signs`, which alternates element by element and so cancels after
    every pair, this front-loads one sign: a running accumulator climbs for
    half a block before coming back down. That is the only way to make a
    partial sum exceed the accumulator bound while the final sum stays inside
    it, which is what separates sat_mode 'per_product' (clamps the running
    value and loses the excess) from 'per_block' (sums first, clamps once).
    """
    t = torch.empty(shape, dtype=torch.float32)
    for blk in _blocks(t, axis, bs):
        n = blk.shape[axis]
        sign = torch.ones(n)
        sign[n // 2:] = -1.0
        view = [1] * blk.dim()
        view[axis] = n
        blk.copy_(sign.reshape(view).expand_as(blk) * scale)
    return t


def _signs(shape, axis, bs, gen, scale=1.0):
    """Alternating signs over a magnitude ramp. Catches two's-complement and
    round-half asymmetries, which cancel out on all-positive data."""
    t = _ramp(shape, axis, bs, gen, scale=scale).abs() + (scale * 1e-3)
    sign = torch.ones_like(t).flatten()
    sign[1::2] = -1.0
    return t * sign.reshape(t.shape)


def _ties(shape, axis, bs, gen, scale=1.0, mbits=8):
    """Every non-leading element sits exactly on a rounding tie of the MX
    mantissa grid.

    The first element of each block is `scale`, which pins the shared exponent
    to E = floor(log2 scale) and the mantissa step to 2^(E - mant_bias). The rest
    are (k + 0.5) steps for k cycling through both signs and both parities, so
    half-away-from-zero, half-to-even and truncation give three different
    mantissa sets. The MX lib rounds half away from zero. `mbits` is the
    element width the ties are laid out for (mant_bias = mbits - 2); k stays
    inside the mantissa range, so no tie is hidden by the clamp.
    """
    t = torch.zeros(shape)
    step = float(scale) * 2.0 ** -(mbits - 2)
    ks = torch.tensor([0, 1, 2, 3, -1, -2, -3, -4, 10, 11, -11, -12, 40, 41,
                       -41, -42], dtype=torch.float32)
    ks = ks[(ks + 0.5).abs() < (1 << (mbits - 1)) - 1]
    for blk in _blocks(t, axis, bs):
        n = blk.shape[axis]
        v = (ks.repeat((n + len(ks) - 1) // len(ks))[:n] + 0.5) * step
        v[0] = float(scale)
        view = [1] * blk.dim()
        view[axis] = n
        blk.copy_(v.reshape(view).expand_as(blk))
    return t


def _mant_overflow(shape, axis, bs, gen, scale=1.0):
    """Block max just under the next power of two, so its mantissa rounds to
    2^(mbits-1) = 128 - one past the int8 range.

    The MX lib keeps the exponent and clamps the mantissa to +-127. Hardware
    could instead bump the exponent (mantissa 64, E + 1), or allow -128 on the
    negative side. Lead elements: +1.99609375 (127.75 -> 128), -1.99609375,
    +1.9921875 (exact tie 127.5), -1.9921875; the rest random at half scale.
    """
    t = (torch.randn(shape, generator=gen) * 0.5).clamp(-1.5, 1.5) * float(scale)
    lead = torch.tensor([1.99609375, -1.99609375, 1.9921875, -1.9921875]) * scale
    for blk in _blocks(t, axis, bs):
        n = min(blk.shape[axis], len(lead))
        for j in range(n):
            blk.narrow(axis, j, 1).fill_(float(lead[j]))
    return t


def _zero_blocks(shape, axis, bs, gen, scale=1.0):
    """randn, with every second block along `axis` exactly zero.

    An all-zero block has no defined shared exponent (log2 0). The contract
    pins its code to 0; this is the vector that shows whether HW agrees.
    """
    t = torch.randn(shape, generator=gen) * float(scale)
    for b, blk in enumerate(_blocks(t, axis, bs)):
        if b % 2 == 1:
            blk.zero_()
    return t


def _half_ties(shape, axis, bs, gen, step=2.0 ** -20, scale=0.25):
    """Bias values lying exactly half-way between two accumulator LSBs.

    The bias is projected onto the accumulator grid as round(b / step); with
    step = 2^e_layer_min every value here is a tie, both signs and both
    parities, plus a coarse +-`scale` offset so the bias is not negligible.
    The MX lib uses torch.round (half to even) here, while its MX quantizer
    rounds half away from zero - which one HW does is an open question.
    """
    n = shape[0]
    k = torch.arange(n, dtype=torch.float64) - n // 2
    off = (torch.arange(n) % 3 - 1).to(torch.float64) * float(scale)
    return (off + (k + 0.5) * step).to(torch.float32).reshape(shape)


def _abs_randn(shape, axis, bs, gen, scale=1.0):
    return torch.randn(shape, generator=gen).abs() * float(scale)


PATTERNS: Dict[str, Callable] = {
    "randn": _randn,
    "ramp": _ramp,
    "int_ramp": _int_ramp,
    "ones": _ones,
    "single_nonzero": _single_nonzero,
    "wide_dyn": _wide_dyn,
    "underflow": _underflow,
    "tiny": _tiny,
    "saturate": _saturate,
    "signs": _signs,
    "half_split": _half_split,
    "ties": _ties,
    "mant_overflow": _mant_overflow,
    "zero_blocks": _zero_blocks,
    "half_ties": _half_ties,
    "abs_randn": _abs_randn,
}


def _step_ties(shape, axis, bs, gen, step=2.0 ** -4, scale=None):
    """Every element an exact tie (k + 0.5) * step, k over both signs and both
    parities. For a static fixed-point quantizer with that step: half-away,
    half-even and truncation give three different codes."""
    n = int(torch.tensor(shape).prod())
    ks = torch.tensor([0, 1, 2, 3, -1, -2, -3, -4, 10, 11, -11, -12, 60, 61,
                       -61, -62], dtype=torch.float64)
    k = ks.repeat((n + len(ks) - 1) // len(ks))[:n]
    return ((k + 0.5) * step).to(torch.float32).reshape(shape)


PATTERNS["step_ties"] = _step_ties


# --- flow fields [B, 2, H, W], in pixels ------------------------------------
#
# The warp grid is a Q8.8 word: 1/256 px steps, range [-128, +127.996]. These
# fill it so that one convention at a time is the only thing that differs.

def _flow_zero(shape, gen):
    return torch.zeros(shape)


def _flow_const(shape, gen, dx=0.0, dy=0.0):
    """Constant displacement. Integer values make the warp a pure shift."""
    t = torch.zeros(shape)
    t[:, 0], t[:, 1] = float(dx), float(dy)
    return t


def _flow_subpix(shape, gen, max_px=3.0, lattice=True):
    """Random displacement in [-max_px, max_px]. `lattice` snaps it to 1/256 so
    the grid quantizer is exact and only the interpolation is under test;
    otherwise the grid rounding is exercised too."""
    t = (torch.rand(shape, generator=gen) * 2 - 1) * float(max_px)
    return torch.round(t * 256) / 256 if lattice else t


def _flow_ties(shape, gen):
    """Every value an exact half-step of the Q8.8 grid, (k + 0.5)/256, both
    signs and both parities: the grid's own rounding rule decides the code."""
    n = int(torch.tensor(shape).prod())
    ks = torch.tensor([0, 1, 2, 3, -1, -2, -3, -4, 100, 101, -101, -102],
                      dtype=torch.float64)
    k = ks.repeat((n + len(ks) - 1) // len(ks))[:n]
    return ((k + 0.5) / 256).to(torch.float32).reshape(shape)


def _flow_oob(shape, gen):
    """Displacements that push every pixel across an image edge by a fraction
    of a pixel, so each output mixes in-bounds and out-of-bounds neighbours.
    Zero padding and border clamping give different values everywhere."""
    B, _, H, W = shape
    xs = torch.arange(W, dtype=torch.float32).view(1, W).expand(H, W)
    ys = torch.arange(H, dtype=torch.float32).view(H, 1).expand(H, W)
    t = torch.zeros(shape)
    # left half goes off the left edge, right half off the right one
    t[:, 0] = torch.where(xs < W / 2, -xs - 0.375, (W - 1) - xs + 0.625)
    t[:, 1] = torch.where(ys < H / 2, -ys - 0.25, (H - 1) - ys + 0.5)
    return t


def _flow_sat(shape, gen):
    """Displacements past the Q8.8 range (+-128 px) and right at its edge:
    200, -200, 127.998 (rounds to code 32767.5 -> clamps), -128.002."""
    vals = torch.tensor([200.0, -200.0, 127.998046875, -128.001953125,
                         127.99609375, -128.0, 1.5, -1.5])
    n = int(torch.tensor(shape).prod())
    return vals.repeat((n + len(vals) - 1) // len(vals))[:n].reshape(shape)


FLOW_PATTERNS: Dict[str, Callable] = {
    "zero": _flow_zero,
    "const": _flow_const,
    "subpix": _flow_subpix,
    "ties": _flow_ties,
    "oob": _flow_oob,
    "sat": _flow_sat,
}


def make_flow(shape, pattern="zero", seed=0, **kw):
    if pattern not in FLOW_PATTERNS:
        raise ValueError(f"unknown flow pattern {pattern!r}; have "
                         f"{sorted(FLOW_PATTERNS)}")
    gen = torch.Generator().manual_seed(seed)
    return FLOW_PATTERNS[pattern](tuple(shape), gen, **kw).to(torch.float32)


def make_tensor(shape, pattern="randn", axis=1, bs=BLOCK_SIZE, seed=0, **kw):
    """Build one tensor. `axis` is the blockified axis (channels for
    activations, input-features for weights / linear)."""
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}; have {sorted(PATTERNS)}")
    gen = torch.Generator().manual_seed(seed)
    return PATTERNS[pattern](tuple(shape), axis % len(shape), bs, gen, **kw)


def fill_params(model, pattern="randn", bs=BLOCK_SIZE, seed=0, bias_pattern=None,
                bias_kw=None, blockify=BLOCKIFY, **kw):
    """Overwrite every weight / bias in `model` with a pattern, in a fixed
    module order so the same seed gives the same checkpoint on both sides.

    The pattern is laid out along the axis the weight is actually blocked on:
    under NPE a conv filter is flattened to [Cin*kH*kW] and blocked along that
    stream, so the pattern is built on the flat view and reshaped back. Under
    `channel` (and for every Linear) it runs along dim 1. For k=1 the two agree.
    """
    with torch.no_grad():
        for i, (name, p) in enumerate(sorted(model.named_parameters())):
            if p.dim() == 1:  # bias
                pat = bias_pattern or "ramp"
                p.copy_(make_tensor(p.shape, pat, axis=0, bs=bs, seed=seed + i,
                                    **(bias_kw or {})))
            elif p.dim() == 4 and blockify == "npe":
                flat = (p.shape[0], int(p[0].numel()))
                p.copy_(make_tensor(flat, pattern, axis=1, bs=bs,
                                    seed=seed + i, **kw).reshape(p.shape))
            else:
                p.copy_(make_tensor(p.shape, pattern, axis=1, bs=bs,
                                    seed=seed + i, **kw))
    return model


# =============================================================================
# Case registry
# =============================================================================

@dataclass
class Case:
    """One benchmark point: a model, an input shape, and how to fill both.

    `out_quant` adds the *static fixed-point* output stage on top of MX - a
    different quantizer, not another MX format. MX derives its scale from the
    block max, so it cannot overflow; a static Qm.n scale is fixed up front and
    overflows silently. Set it to a dict of `fxp_quant.OUT_QUANT_DEFAULTS` keys
    (total_bits / frac_bits / signed / round / saturate) and the mains attach it
    to the network output.
    """
    build: Callable[[int], nn.Module]       # bs -> model
    input_shape: Callable[[int], Sequence]  # bs -> shape
    w_pattern: str = "randn"
    x_pattern: str = "randn"
    # Extra kwargs for the pattern builders, e.g. {"scale": 2**12} to push the
    # shared exponents high enough that the narrow accumulator actually clamps.
    w_kw: dict = None
    x_kw: dict = None
    note: str = ""
    tags: Tuple[str, ...] = ()
    out_quant: dict = None
    # Bias pattern (default "ramp") and its kwargs.
    b_pattern: str = None
    b_kw: dict = None
    # Axis the input pattern runs along. None = the block axis of the active
    # geometry; 1 keeps a channel ramp (e.g. conv1x1_ramp, whose sum(i^2) needs
    # the input to vary along Cin whatever the blocking).
    x_axis: int = None
    # Per-case config overrides, merged into mx_specs / mx_specs.xblock_accum.
    # Both sides apply them: element format, scale_bits, accumulator bits,
    # e_layer_min, ...
    mx: dict = None
    accum: dict = None
    # Warp cases: the second network input is a flow field [B, 2, H, W] in
    # pixels (`FLOW_PATTERNS`), and `warp` overrides the DOF warp quantizers
    # per operand - {"ref": {...}, "grid": {...}, "out": {...},
    # "call_sites": N} over `main_ALG.WARP_DEFAULT`.
    flow_pattern: str = None
    flow_kw: dict = None
    warp: dict = None


# Explicit power-of-two scales for the warp cases that pin one convention: the
# ref and out words share the step, so an identity warp returns the ref code
# unchanged and a half-pixel warp lands exactly on ties.
_WARP_POW2 = {"ref": {"scale": 2.0 ** -5, "calibrate": False},
              "out": {"scale": 2.0 ** -5, "calibrate": False}}


def _c(cin_mult=1, **kw):
    """Shorthand: channels expressed as a multiple of the block size."""
    return lambda bs: int(cin_mult * bs)


CASES: Dict[str, Case] = {

    # --- baseline: must match before anything else is worth debugging -------
    "conv1x1_ramp": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, 2),
        w_pattern="int_ramp", x_pattern="int_ramp", x_axis=1,
        note="1x1 conv, one aligned block, small integers. Hand-checkable.",
        tags=("baseline",),
    ),
    "conv1x1_single_nonzero": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 1, 1),
        w_pattern="single_nonzero", x_pattern="single_nonzero", x_axis=1,
        note="One product in the whole layer. Isolates the shift, not the sum.",
        tags=("baseline",),
    ),
    "conv3x3_randn": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=0),
        input_shape=lambda bs: (1, bs, 5, 5),
        note="Plain 3x3, aligned channels, random data. The reference point.",
        tags=("baseline",),
    ),

    # --- block alignment: channel count vs block size -----------------------
    "aligned_c64": Case(
        build=lambda bs: SingleConv(2 * bs, bs, k=1),
        input_shape=lambda bs: (1, 2 * bs, 2, 2),
        note="Two full blocks per reduction. Inter-block accumulation.",
        tags=("align",),
    ),
    "tail_block_c33": Case(
        build=lambda bs: SingleConv(bs + 1, bs, k=1),
        input_shape=lambda bs: (1, bs + 1, 2, 2),
        note="One full block + a 1-element tail. Is the tail padded, or a "
             "short block with its own shared exponent?",
        tags=("align", "corner"),
    ),
    "tail_block_c48": Case(
        build=lambda bs: SingleConv(bs + bs // 2, bs, k=1),
        input_shape=lambda bs: (1, bs + bs // 2, 2, 2),
        note="1.5 blocks. Half-full tail, the common real-network shape.",
        tags=("align", "corner"),
    ),
    "sub_block_c8": Case(
        build=lambda bs: SingleConv(max(bs // 4, 1), bs, k=1),
        input_shape=lambda bs: (1, max(bs // 4, 1), 2, 2),
        note="Fewer channels than one block. Every block is partial.",
        tags=("align", "corner"),
    ),
    "tail_x_tail": Case(
        build=lambda bs: SingleConv(bs + 1, bs + 1, k=3, padding=0),
        input_shape=lambda bs: (1, bs + 1, 4, 4),
        note="Unaligned on both input and output channels, with 3x3 geometry "
             "on top - the k*k*cin flatten is unaligned too.",
        tags=("align", "corner"),
    ),

    # --- conv geometry ------------------------------------------------------
    "stride2": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, stride=2, padding=1),
        input_shape=lambda bs: (1, bs, 8, 8),
        note="Stride 2. Output positions skip input columns.",
        tags=("geometry",),
    ),
    "dilation2": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, dilation=2, padding=0),
        input_shape=lambda bs: (1, bs, 7, 7),
        note="Dilated taps. Same block count, non-contiguous reads.",
        tags=("geometry",),
    ),
    "padded_edges": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 4, 4),
        x_pattern="wide_dyn",
        note="Zero padding with wide dynamic range: do the pad zeros join the "
             "block, shifting its alignment, or are they skipped?",
        tags=("geometry", "corner"),
    ),
    "with_bias": Case(
        build=lambda bs: SingleConv(bs, bs, k=1, bias=True),
        input_shape=lambda bs: (1, bs, 2, 2),
        w_pattern="int_ramp", x_pattern="int_ramp", x_axis=1,
        note="Bias on. Added into the fixed-point accumulator, or after the "
             "requant in FP? Different results once it saturates.",
        tags=("geometry", "corner"),
    ),
    "depthwise": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=1, groups=bs),
        input_shape=lambda bs: (1, bs, 5, 5),
        note="Depthwise: reduction is k*k=9 long, so a 32-block is never full.",
        tags=("geometry", "corner"),
    ),
    "grouped4": Case(
        build=lambda bs: SingleConv(bs, bs, k=1, groups=4),
        input_shape=lambda bs: (1, bs, 2, 2),
        note="4 groups. Reduction is cin/4, i.e. a quarter block each.",
        tags=("geometry", "corner"),
    ),

    # --- topology: more than one op ----------------------------------------
    "conv_relu": Case(
        build=lambda bs: ConvReLU(bs, bs, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 5, 5),
        x_pattern="signs",
        note="ReLU after the conv. Clamp before or after requant.",
        tags=("topology",),
    ),
    "conv_relu_conv": Case(
        build=lambda bs: ConvReLUConv(bs, bs, bs, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 6, 6),
        note="Two layers. The handoff - conv2's activations are conv1's "
             "requantised output, not FP32.",
        tags=("topology",),
    ),
    "dw_pw": Case(
        build=lambda bs: DwPw(bs, bs, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 6, 6),
        note="Depthwise then pointwise, the NPE backbone block.",
        tags=("topology",),
    ),
    "concat": Case(
        build=lambda bs: ConvConcat(bs, bs // 2 + 1, bs // 2, bs, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 5, 5),
        note="Channel concat of two unequal branches: the consumer's blocks "
             "straddle the branch boundary.",
        tags=("topology", "corner"),
    ),
    "residual_add": Case(
        build=lambda bs: ConvAdd(bs, bs, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 5, 5),
        note="Elementwise add of two quantised tensors. What precision is the "
             "sum kept in before the next layer re-quantises it?",
        tags=("topology", "corner"),
    ),
    "linear": Case(
        build=lambda bs: SingleLinear(2 * bs + 3, bs),
        input_shape=lambda bs: (4, 2 * bs + 3),
        note="Linear with an unaligned feature count. Flatten blockify with "
             "no conv geometry in the way.",
        tags=("topology", "align"),
    ),

    # --- data stress: same shapes, numerics pushed to the edges -------------
    "underflow_acts": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, 2),
        x_pattern="underflow",
        note="Activation blocks with one dominant element: most values "
             "quantise to 0. Dominant MX loss mode.",
        tags=("stress",),
    ),
    "underflow_weights": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, 2),
        w_pattern="wide_dyn",
        note="Same, driven by the weights instead.",
        tags=("stress",),
    ),
    "flush_to_zero": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, 2),
        x_pattern="tiny",
        note="Every product underflows the accumulator grid (acts at exp -30, "
             "e_layer_min -20). Does NOT reach zero: an arithmetic right shift "
             "rounds toward -inf, so each negative product deposits -1 LSB and "
             "each positive 0, leaving a systematic negative bias. HW that "
             "truncates toward zero instead gives exactly 0 - this case is the "
             "vector that tells the two apart.",
        tags=("stress", "corner"),
    ),
    "accum_saturate": Case(
        build=lambda bs: SingleConv(4 * bs, bs, k=3, padding=0),
        input_shape=lambda bs: (1, 4 * bs, 5, 5),
        w_pattern="saturate", x_pattern="saturate",
        # shift = Ew + Ea - 2*mant_bias - e_layer_min, so what fills the
        # accumulator is the shared EXPONENTS, not the mantissas: products are
        # capped at 127*127 whatever the scale. At e_layer_min=-20 and a 1152-
        # term reduction the clamp needs Ew = Ea = 12, i.e. operands around
        # 2^12. Measured: 2^8 still does not clamp, 2^12 clamps every output.
        w_kw={"scale": 2.0 ** 12}, x_kw={"scale": 2.0 ** 12},
        note="Long reduction (k*k*4*bs) at full magnitude and high exponent, "
             "all one sign: the narrow accumulator saturates on every output. "
             "Pins the clamp bound itself: out == (2^(bits-1) - 1) * "
             "2^e_layer_min. Both sat_modes agree here - see "
             "accum_sat_transient for the vector that splits them.",
        tags=("stress", "corner"),
    ),

    "accum_sat_transient": Case(
        build=lambda bs: SingleConv(4 * bs, bs, k=3, padding=0),
        input_shape=lambda bs: (1, 4 * bs, 5, 5),
        w_pattern="half_split", x_pattern="saturate",
        w_kw={"scale": 2.0 ** 12}, x_kw={"scale": 2.0 ** 12},
        note="The partial sum leaves the accumulator range but the final sum "
             "does not: each block climbs for 16 products, then the sign flips "
             "and it comes back to zero. 'per_product' clamps at the peak, "
             "loses the excess and ends pinned at the negative bound; "
             "'per_block' sums first and returns exactly 0. Measured: "
             "-1.342e+08 vs 0. The only vector that tells the two apart, so "
             "it is how the sat_mode line of the spec gets settled.",
        tags=("stress", "corner"),
    ),
    # --- static fixed-point output stage (not MX: fixed scale, can overflow) --
    "out_fxp_q8_8": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, 2),
        w_pattern="single_nonzero", x_pattern="single_nonzero", x_axis=1,
        out_quant={"total_bits": 16, "frac_bits": 8, "signed": True,
                   "round": "half_away", "saturate": True},
        note="MX layer then a Q8.8 output word (step 1/256, range "
             "[-128, +127.996] - the DOF flow-field format). One product per "
             "output, so out = 1.0 and the word is exactly 256. In range, "
             "nothing clips: the clean baseline for the fxp stage.",
        tags=("fxp",),
    ),
    "out_fxp_saturate": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, 2),
        w_pattern="int_ramp", x_pattern="int_ramp", x_axis=1,
        out_quant={"total_bits": 16, "frac_bits": 8, "signed": True,
                   "saturate": True},
        note="Same Q8.8, but int_ramp products reach ~11440 - far past +128, "
             "so the whole output clips. The failure mode MX does not have.",
        tags=("fxp", "corner"),
    ),
    "out_fxp_round_modes": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 4, 4),
        out_quant={"total_bits": 16, "frac_bits": 4, "signed": True,
                   "round": "half_away", "saturate": True},
        note="Coarse Q12.4 (step 1/16) so ties are common: half_away vs "
             "half_even vs trunc give different words here - the other two "
             "rules are out_fxp_round_half_even / out_fxp_round_trunc.",
        tags=("fxp", "corner"),
    ),
    "out_fxp_unsigned": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 5, 5),
        out_quant={"total_bits": 8, "frac_bits": 4, "signed": False,
                   "saturate": True},
        note="Unsigned 8-bit word, Q4.4, fed by a conv that emits both signs: "
             "negatives clamp to 0 at the bottom and large values clamp at "
             "15.9375 at the top, so both clamp directions are exercised.",
        tags=("fxp", "corner"),
    ),

    "sign_mix": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, 2),
        w_pattern="signs", x_pattern="signs",
        note="Alternating signs: catches two's-complement and rounding "
             "asymmetries that all-positive data hides.",
        tags=("stress",),
    ),

    # --- MX quantizer rounding: the spec lines arch still has to confirm -----
    "mx_round_ties": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, bs),
        w_pattern="ties", x_pattern="ties",
        note="Every non-leading element of every block is an exact mantissa "
             "tie ((k+0.5) steps, both signs, both parities). The lib rounds "
             "half away from zero; half-even or truncation shows up in "
             "a_int / w_int directly.",
        tags=("mxq", "corner"),
    ),
    "mx_mant_overflow": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, bs),
        w_pattern="mant_overflow", x_pattern="mant_overflow",
        note="Block max at 1.996 / 1.992 (127.75 / 127.5 steps): the mantissa "
             "rounds to 128. The lib keeps E and clamps to +-127; HW may bump "
             "the exponent or allow -128.",
        tags=("mxq", "corner"),
    ),
    "zero_blocks": Case(
        build=lambda bs: SingleConv(2 * bs, bs, k=1),
        input_shape=lambda bs: (1, 2 * bs, 2, 2 * bs),
        w_pattern="zero_blocks", x_pattern="zero_blocks",
        note="Every second block of both operands is all zero. Its shared "
             "exponent is undefined; the contract pins the code to 0.",
        tags=("mxq", "corner"),
    ),
    "exp_underflow_clamp": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, 2),
        x_pattern="tiny", x_kw={"scale": 2.0 ** -130},
        note="Activations at 2^-130 (fp32 subnormal), below the 8-bit field's "
             "floor of -127: E clamps to -127 (code 0) and the mantissa is 8, "
             "not 64. HW must clamp, not wrap.",
        tags=("mxq", "corner"),
    ),
    "scale_bits_5": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, 2),
        x_pattern="randn", x_kw={"scale": 2.0 ** -18},
        mx={"scale_bits": 5},
        note="5-bit exponent field (bias 15, floor -15). Activations sit at "
             "~2^-18, so their exponent clamps and the mantissas shrink by 3 "
             "bits. Keeps the field width a parameter, not a constant.",
        tags=("mxq", "corner"),
    ),

    # --- NPE X-blocks: W against the block size -----------------------------
    "xblock_w32": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 3, bs),
        note="W = block size: exactly one full X-block per row, no tail. With "
             "padding=1 the pad columns sit outside the block - does HW block "
             "the padded row (bs+2 wide) instead?",
        tags=("xblock", "geometry"),
    ),
    "xblock_w33": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 3, bs + 1),
        x_pattern="wide_dyn",
        note="One full X-block + a 1-wide tail per row, wide dynamic range so "
             "a tail that is merged into, or padded off, the full block gets a "
             "visibly different exponent.",
        tags=("xblock", "geometry", "corner"),
    ),
    "xblock_w64": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 3, 2 * bs),
        x_pattern="underflow",
        note="Two full X-blocks per row; the 3x3 window straddles the block "
             "boundary at x = bs-1 / bs, so one output mixes two activation "
             "exponents.",
        tags=("xblock", "geometry"),
    ),

    # --- geometry the first 30 cases never touched --------------------------
    "batch2": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=1),
        input_shape=lambda bs: (2, bs, 3, 6),
        note="Batch of 2 with different data. Catches a batch index dropped "
             "or mixed up in the HW walk.",
        tags=("geometry",),
    ),
    "kernel_1x3": Case(
        build=lambda bs: SingleConv(bs, bs, k=(1, 3), padding=(0, 1)),
        input_shape=lambda bs: (1, bs, 3, 7),
        note="Non-square kernel and non-square input: kH != kW, H != W. "
             "Swapped H/W anywhere shows up here and nowhere else.",
        tags=("geometry",),
    ),
    "stride_2x1_odd": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, stride=(2, 1), padding=1),
        input_shape=lambda bs: (1, bs, 7, 9),
        note="Asymmetric stride on odd H / W: the last row is reached only if "
             "the output-size formula rounds the way torch does.",
        tags=("geometry",),
    ),
    "kernel_5x5": Case(
        build=lambda bs: SingleConv(bs, bs, k=5, padding=2),
        input_shape=lambda bs: (1, bs, 6, 6),
        note="5x5: flatten length 25*Cin, so weight blocks start at kernel "
             "taps other than (0,0) - the C-major flatten order is exercised.",
        tags=("geometry",),
    ),

    # --- accumulator configs ------------------------------------------------
    "e_layer_min_m8": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=0),
        input_shape=lambda bs: (1, bs, 5, 5),
        accum={"e_layer_min": -8},
        note="Accumulator LSB at 2^-8: products with Ea + Ew < 4 are right-"
             "shifted and lose bits, of both signs - partial underflow, so "
             "floor (arith shift) vs truncate-toward-zero differ on a few "
             "LSBs instead of all-or-nothing.",
        tags=("accum", "corner"),
    ),
    "e_layer_min_0": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=0),
        input_shape=lambda bs: (1, bs, 5, 5),
        accum={"e_layer_min": 0},
        note="LSB at 2^0: every product loses ~10 bits. Heavy truncation; the "
             "output is a small integer.",
        tags=("accum",),
    ),
    "accum35_partial_sat": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=0),
        input_shape=lambda bs: (1, bs, 5, 5),
        w_kw={"scale": 2.0 ** 5}, x_kw={"scale": 2.0 ** 5},
        accum={"bits": 35},
        note="35-bit accumulator (DVNR config): bound 2^34 * 2^-20 = 16384, "
             "and the outputs straddle it, so some saturate and some do not.",
        tags=("accum", "corner"),
    ),
    "bias_ties": Case(
        build=lambda bs: SingleConv(bs, bs, k=1, bias=True),
        input_shape=lambda bs: (1, bs, 2, 2),
        b_pattern="half_ties", b_kw={"step": 2.0 ** -20},
        note="Every bias is an exact tie on the 2^-20 accumulator grid. The "
             "lib projects it with torch.round (half to even) - while its MX "
             "quantizer rounds half away. Which one does HW use?",
        tags=("accum", "corner"),
    ),

    # --- element formats ----------------------------------------------------
    "fmt_int10": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=0),
        input_shape=lambda bs: (1, bs, 5, 5),
        mx={"w_elem_format": "int10", "a_elem_format": "int10"},
        note="MXINT10: mant_bias 8, mantissa +-511. Checks nothing is hard-"
             "wired to int8.",
        tags=("fmt",),
    ),
    "fmt_int16": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=0),
        input_shape=lambda bs: (1, bs, 5, 5),
        mx={"w_elem_format": "int16", "a_elem_format": "int16"},
        accum={"e_layer_min": -32},
        note="MXINT16: mant_bias 14, products up to 2^30. e_layer_min -32 so "
             "no product is right-shifted (-20 would drop ~8 bits).",
        tags=("fmt",),
    ),

    # --- output-stage tie rules (the other two of out_fxp_round_modes) ------
    "out_fxp_round_half_even": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 4, 4),
        out_quant={"total_bits": 16, "frac_bits": 4, "signed": True,
                   "round": "half_even", "saturate": True},
        note="out_fxp_round_modes with round=half_even.",
        tags=("fxp", "corner"),
    ),
    "out_fxp_round_trunc": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 4, 4),
        out_quant={"total_bits": 16, "frac_bits": 4, "signed": True,
                   "round": "trunc", "saturate": True},
        note="out_fxp_round_modes with round=trunc (toward zero).",
        tags=("fxp", "corner"),
    ),

    # --- narrow element formats: int4 / int6 ----------------------------------
    "fmt_int4": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 4, bs + 8),
        mx={"w_elem_format": "int4", "a_elem_format": "int4"},
        note="MXINT4 (DOF ladder rung): mant_bias 2, mantissa +-7, so most "
             "elements of a block round to a handful of codes and ties are "
             "everywhere. Partial X-block tail on top.",
        tags=("fmt",),
    ),
    "fmt_int6": Case(
        build=lambda bs: SingleConv(bs, bs, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 4, bs + 8),
        mx={"w_elem_format": "int6", "a_elem_format": "int6"},
        note="MXINT6 (DOF ladder rung): mant_bias 4, mantissa +-31.",
        tags=("fmt",),
    ),
    "fmt_int4_ties": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, bs),
        w_pattern="ties", x_pattern="ties", w_kw={"mbits": 4},
        x_kw={"mbits": 4},
        mx={"w_elem_format": "int4", "a_elem_format": "int4"},
        note="mx_round_ties at int4: every non-leading element is an exact "
             "tie of the 3-bit mantissa grid (k + 0.5 steps, |k + 0.5| < 7). "
             "int4 is where a tie rule costs most - one step is 1/8 of the "
             "block max.",
        tags=("fmt", "mxq"),
    ),

    # --- warp (DOF): int8 ref, Q8.8 grid, int8 out -------------------------
    "warp_identity": Case(
        build=lambda bs: WarpOnly(),
        input_shape=lambda bs: (1, 4, 4, 8),
        flow_pattern="zero", warp=_WARP_POW2,
        note="Zero flow, ref and out share a 2^-5 step: out_code must equal "
             "ref_code exactly. The first warp vector to match.",
        tags=("warp", "dof", "baseline"),
    ),
    "warp_int_shift": Case(
        build=lambda bs: WarpOnly(),
        input_shape=lambda bs: (1, 4, 5, 9),
        flow_pattern="const", flow_kw={"dx": 3.0, "dy": -1.0}, warp=_WARP_POW2,
        note="Integer flow (+3 px x, -1 px y): a pure shift of the ref codes, "
             "zeros entering from the edges. Settles the grid's meaning - a "
             "displacement added to the pixel coordinate (the spec) vs an "
             "absolute source coordinate - and which channel is x.",
        tags=("warp", "dof"),
    ),
    "warp_half_pixel": Case(
        build=lambda bs: WarpOnly(),
        input_shape=lambda bs: (1, 4, 4, 9),
        flow_pattern="const", flow_kw={"dx": 0.5, "dy": 0.0}, warp=_WARP_POW2,
        note="Half-pixel flow, equal ref / out steps: out = (a + b) / 2 in "
             "codes, an exact tie wherever a + b is odd. The out quantizer's "
             "tie rule (lib: half away) is the only thing that decides it.",
        tags=("warp", "dof", "corner"),
    ),
    "warp_neg_frac": Case(
        build=lambda bs: WarpOnly(),
        input_shape=lambda bs: (1, 4, 4, 8),
        flow_pattern="const", flow_kw={"dx": -0.25, "dy": -0.75},
        warp=_WARP_POW2,
        note="Negative fractional flow: the grid word splits into integer and "
             "fraction by an arithmetic shift (x0 = floor, frac in [0, 1)). "
             "Splitting toward zero instead gives the wrong neighbour pair.",
        tags=("warp", "dof", "corner"),
    ),
    "warp_subpix": Case(
        build=lambda bs: WarpOnly(),
        input_shape=lambda bs: (1, 8, 6, 12),
        flow_pattern="subpix", flow_kw={"max_px": 3.0},
        note="Random flow on the 1/256 lattice, random features, calibrated "
             "arbitrary-real ref / out scales: the DOF configuration as is. "
             "Settles the interpolation weight precision (8 fractional bits).",
        tags=("warp", "dof"),
    ),
    "warp_grid_ties": Case(
        build=lambda bs: WarpOnly(),
        input_shape=lambda bs: (1, 4, 4, 12),
        flow_pattern="ties", warp=_WARP_POW2,
        note="Every flow value an exact half-step of the Q8.8 grid: the grid "
             "quantizer's tie rule (lib: half away) decides every grid_code.",
        tags=("warp", "dof", "corner"),
    ),
    "warp_ref_ties": Case(
        build=lambda bs: WarpOnly(),
        input_shape=lambda bs: (1, 4, 4, 8),
        x_pattern="step_ties", x_kw={"step": 2.0 ** -5},
        flow_pattern="zero", warp=_WARP_POW2,
        note="Every feature an exact tie of the 2^-5 ref step: the ref "
             "quantizer's tie rule decides every ref_code.",
        tags=("warp", "dof", "corner"),
    ),
    "warp_oob": Case(
        build=lambda bs: WarpOnly(),
        input_shape=lambda bs: (1, 4, 6, 10),
        flow_pattern="oob", warp=_WARP_POW2,
        note="Every pixel sampled a fraction of a pixel past an image edge: "
             "zero padding (the spec) and border clamping differ on every "
             "output.",
        tags=("warp", "dof", "corner"),
    ),
    "warp_grid_sat": Case(
        build=lambda bs: WarpOnly(),
        input_shape=lambda bs: (1, 4, 4, 8),
        flow_pattern="sat", warp=_WARP_POW2,
        note="Flow at and past +-128 px: the Q8.8 grid word clamps to "
             "[-32768, 32767] (127.998 rounds to 32767.5 and clamps).",
        tags=("warp", "dof", "corner"),
    ),
    "warp_out_sat": Case(
        build=lambda bs: WarpOnly(),
        input_shape=lambda bs: (1, 4, 4, 8),
        x_kw={"scale": 2.0}, flow_pattern="subpix", flow_kw={"max_px": 1.0},
        warp={"ref": {"scale": 2.0 ** -4, "calibrate": False},
              "out": {"scale": 2.0 ** -7, "calibrate": False}},
        note="Out step 8x finer than the ref's, so warped values beyond "
             "+-1 clip (the ref itself does not): the int8 out word saturates "
             "both ways. The lib allows "
             "-128 (two's complement); a symmetric +-127 HW differs.",
        tags=("warp", "dof", "corner"),
    ),
    "warp_levels": Case(
        build=lambda bs: WarpLevels(),
        input_shape=lambda bs: (1, 8, 8, 16),
        flow_pattern="subpix", flow_kw={"max_px": 2.0},
        warp={"call_sites": 3},
        note="One warp, three pyramid levels (1/4, 1/2, 1 res; features 4x "
             "larger per level): each call site calibrates its own ref and "
             "out scale, as arch specified. Taps are warp@0 .. warp@2.",
        tags=("warp", "dof"),
    ),
    "conv_warp": Case(
        build=lambda bs: ConvWarp(bs, 8, k=3, padding=1),
        input_shape=lambda bs: (1, bs, 4, 12),
        flow_pattern="subpix", flow_kw={"max_px": 2.0},
        note="MX conv -> warp: the warp's int8 ref is the conv's accumulator "
             "requantised to a static scale. The MX -> fxp handoff.",
        tags=("warp", "dof", "topology"),
    ),
}


# =============================================================================
# Entry points
# =============================================================================

def build_case(name, bs=BLOCK_SIZE, seed=0, batch=None, blockify=None):
    """Materialise one case.

    `blockify` ('npe' | 'channel', default `BLOCKIFY`) decides which axis the
    block-shaping patterns run along, so it has to match the config the case
    is run under - `main_ALG.load_config(blockify=...)` takes the same value.

    Returns a dict with:
      model   - nn.Module, weights filled, in eval mode
      x       - input tensor
      y_fp32  - FP32 reference output (what both sides are quantising away from)
      meta    - name / bs / seed / shapes / patterns / note, for the manifest
    """
    if name not in CASES:
        raise KeyError(f"unknown case {name!r}; have {sorted(CASES)}")
    case = CASES[name]

    blockify = blockify or BLOCKIFY
    if blockify not in BLOCKIFY_CHOICES:
        raise ValueError(f"blockify must be one of {BLOCKIFY_CHOICES}")
    model = case.build(bs).eval()
    fill_params(model, case.w_pattern, bs=bs, seed=seed,
                bias_pattern=case.b_pattern, bias_kw=case.b_kw,
                blockify=blockify, **(case.w_kw or {}))

    shape = list(case.input_shape(bs))
    if batch is not None:
        shape[0] = batch
    axis = case.x_axis if case.x_axis is not None else \
        act_block_axis(len(shape), blockify)
    x = make_tensor(shape, case.x_pattern, axis=axis, bs=bs, seed=seed + 1000,
                    **(case.x_kw or {}))
    flow = None
    if case.flow_pattern:
        flow = make_flow((shape[0], 2, shape[2], shape[3]), case.flow_pattern,
                         seed=seed + 2000, **(case.flow_kw or {}))

    with torch.no_grad():
        y = model(*inputs_of({"x": x, "flow": flow}))

    meta = {
        "case": name,
        "out_quant": case.out_quant,
        "block_size": bs,
        "seed": seed,
        "model": type(model).__name__,
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "w_pattern": case.w_pattern,
        "x_pattern": case.x_pattern,
        "w_kw": case.w_kw or {},
        "x_kw": case.x_kw or {},
        "b_pattern": case.b_pattern,
        "b_kw": case.b_kw or {},
        "x_axis": axis,
        "blockify": blockify,
        "mx": case.mx or {},
        "accum": case.accum or {},
        "params": {n: list(p.shape) for n, p in model.named_parameters()},
        "note": case.note,
        "tags": list(case.tags),
        "flow_pattern": case.flow_pattern,
        "flow_kw": case.flow_kw or {},
        "warp": case.warp or {},
    }
    return {"model": model, "x": x, "flow": flow, "y_fp32": y, "meta": meta,
            "out_quant": case.out_quant}


def inputs_of(case):
    """The positional network inputs of a built case: (x,) or (x, flow)."""
    return (case["x"],) if case.get("flow") is None else (case["x"], case["flow"])


def case_names(tag=None):
    """All case names, or only those carrying `tag`."""
    if tag is None:
        return list(CASES)
    return [n for n, c in CASES.items() if tag in c.tags]


def list_cases(bs=BLOCK_SIZE, names=None):
    """Print the registry as a table: name, model, shapes, patterns, note."""
    hdr = (f"{'case':<24} {'model':<14} {'input':<18} {'w/x pattern':<26} "
           f"{'out_fxp':<10} note")
    print(hdr)
    print("-" * len(hdr))
    for name in (names if names is not None else CASES):
        case = CASES[name]
        built = build_case(name, bs=bs)
        m = built["meta"]
        shapes = f"{tuple(m['input_shape'])}"
        pats = f"{m['w_pattern']}/{m['x_pattern']}"
        oq = m["out_quant"]
        # Same tag as fxp_quant.fxp_format_str: Q<total-frac>.<frac>, 'u' when
        # unsigned.
        fxp = "-" if not oq else (
            f"Q{oq['total_bits'] - oq.get('frac_bits', 8)}"
            f".{oq.get('frac_bits', 8)}{'' if oq.get('signed', True) else 'u'}")
        print(f"{name:<24} {m['model']:<14} {shapes:<18} {pats:<26} "
              f"{fxp:<10} {m['note']}")


# =============================================================================
# CLI glue
# =============================================================================
#
# Shared by `main.py`, `ALG_main.py` and `HW_main.py` so all three accept the
# same `--model` / `--bs` / `--seed` and cannot drift apart.

def add_model_args(parser):
    """Register the model-selection flags on an `argparse` parser."""
    parser.add_argument("--model", "-m", default="conv3x3_randn",
                        metavar="NAME",
                        help="case name from the registry, or 'all' / a tag "
                             "(baseline, align, geometry, topology, stress, "
                             "corner, fxp, mxq, xblock, accum, fmt, dof, warp) "
                             "to select a group. --list prints "
                             "them.")
    parser.add_argument("--bs", type=int, default=BLOCK_SIZE,
                        help=f"MX block size (default {BLOCK_SIZE})")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the weight / input patterns")
    parser.add_argument("--batch", type=int, default=None,
                        help="override the batch dim of the input")
    parser.add_argument("--blockify", choices=BLOCKIFY_CHOICES,
                        default=BLOCKIFY,
                        help="operand blocking: npe (acts along W, weights "
                             "flattened per filter - what HW does) or channel "
                             "(both along Cin). Sets the config AND the "
                             f"pattern layout (default {BLOCKIFY})")
    parser.add_argument("--list", action="store_true",
                        help="print the case registry and exit")
    return parser


def resolve_models(name):
    """`--model` value -> list of case names.

    Accepts a case name, a tag, or 'all'. Raises with the valid options listed
    rather than silently running the wrong thing.
    """
    if name in CASES:
        return [name]
    if name == "all":
        return list(CASES)
    by_tag = case_names(name)
    if by_tag:
        return by_tag
    tags = sorted({t for c in CASES.values() for t in c.tags})
    raise SystemExit(
        f"unknown --model {name!r}\n"
        f"  cases: {', '.join(CASES)}\n"
        f"  tags:  {', '.join(tags)}, all"
    )


def cases_from_args(args):
    """`argparse` namespace -> iterator of materialised cases.

    Handles `--list` by printing and exiting, so each main stays a few lines.
    """
    names = resolve_models(args.model)
    if getattr(args, "list", False):
        list_cases(bs=args.bs, names=names)
        raise SystemExit(0)
    for name in names:
        yield build_case(name, bs=args.bs, seed=args.seed, batch=args.batch,
                         blockify=getattr(args, "blockify", None))


if __name__ == "__main__":
    import argparse

    parser = add_model_args(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(model="all")
    args = parser.parse_args()
    args.list = True          # models.py standalone only prints the table
    list(cases_from_args(args))
