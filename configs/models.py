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


def _signs(shape, axis, bs, gen, scale=1.0):
    """Alternating signs over a magnitude ramp. Catches two's-complement and
    round-half asymmetries, which cancel out on all-positive data."""
    t = _ramp(shape, axis, bs, gen, scale=scale).abs() + (scale * 1e-3)
    sign = torch.ones_like(t).flatten()
    sign[1::2] = -1.0
    return t * sign.reshape(t.shape)


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
}


def make_tensor(shape, pattern="randn", axis=1, bs=BLOCK_SIZE, seed=0, **kw):
    """Build one tensor. `axis` is the blockified axis (channels for
    activations, input-features for weights / linear)."""
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}; have {sorted(PATTERNS)}")
    gen = torch.Generator().manual_seed(seed)
    return PATTERNS[pattern](tuple(shape), axis % len(shape), bs, gen, **kw)


def fill_params(model, pattern="randn", bs=BLOCK_SIZE, seed=0, bias_pattern=None):
    """Overwrite every weight / bias in `model` with a pattern, in a fixed
    module order so the same seed gives the same checkpoint on both sides.

    Weights are blockified along the input-channel axis (dim 1 for conv, dim 1
    for linear), which is what `weight_blockify=flatten` reduces to for the
    aligned cases.
    """
    with torch.no_grad():
        for i, (name, p) in enumerate(sorted(model.named_parameters())):
            if p.dim() == 1:  # bias
                pat = bias_pattern or "ramp"
                p.copy_(make_tensor(p.shape, pat, axis=0, bs=bs, seed=seed + i))
            else:
                p.copy_(make_tensor(p.shape, pattern, axis=1, bs=bs, seed=seed + i))
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
    note: str = ""
    tags: Tuple[str, ...] = ()
    out_quant: dict = None


def _c(cin_mult=1, **kw):
    """Shorthand: channels expressed as a multiple of the block size."""
    return lambda bs: int(cin_mult * bs)


CASES: Dict[str, Case] = {

    # --- baseline: must match before anything else is worth debugging -------
    "conv1x1_ramp": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, 2),
        w_pattern="int_ramp", x_pattern="int_ramp",
        note="1x1 conv, one aligned block, small integers. Hand-checkable.",
        tags=("baseline",),
    ),
    "conv1x1_single_nonzero": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 1, 1),
        w_pattern="single_nonzero", x_pattern="single_nonzero",
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
        w_pattern="int_ramp", x_pattern="int_ramp",
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
        note="Everything under e_layer_min. The whole layer should flush to "
             "zero - on both sides, at the same threshold.",
        tags=("stress", "corner"),
    ),
    "accum_saturate": Case(
        build=lambda bs: SingleConv(4 * bs, bs, k=3, padding=0),
        input_shape=lambda bs: (1, 4 * bs, 5, 5),
        w_pattern="saturate", x_pattern="saturate",
        note="Long reduction (k*k*4*bs) at full magnitude, all one sign: the "
             "narrow accumulator saturates. Per-product vs per-block "
             "saturation only differ here.",
        tags=("stress", "corner"),
    ),
    # --- static fixed-point output stage (not MX: fixed scale, can overflow) --
    "out_fxp_q8_8": Case(
        build=lambda bs: SingleConv(bs, bs, k=1),
        input_shape=lambda bs: (1, bs, 2, 2),
        w_pattern="single_nonzero", x_pattern="single_nonzero",
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
        w_pattern="int_ramp", x_pattern="int_ramp",
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
             "half_even vs trunc give different words here.",
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
}


# =============================================================================
# Entry points
# =============================================================================

def build_case(name, bs=BLOCK_SIZE, seed=0, batch=None):
    """Materialise one case.

    Returns a dict with:
      model   - nn.Module, weights filled, in eval mode
      x       - input tensor
      y_fp32  - FP32 reference output (what both sides are quantising away from)
      meta    - name / bs / seed / shapes / patterns / note, for the manifest
    """
    if name not in CASES:
        raise KeyError(f"unknown case {name!r}; have {sorted(CASES)}")
    case = CASES[name]

    model = case.build(bs).eval()
    fill_params(model, case.w_pattern, bs=bs, seed=seed)

    shape = list(case.input_shape(bs))
    if batch is not None:
        shape[0] = batch
    axis = 1 if len(shape) > 2 else 1  # channels for 4D, features for 2D
    x = make_tensor(shape, case.x_pattern, axis=axis, bs=bs, seed=seed + 1000)

    with torch.no_grad():
        y = model(x)

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
        "params": {n: list(p.shape) for n, p in model.named_parameters()},
        "note": case.note,
        "tags": list(case.tags),
    }
    return {"model": model, "x": x, "y_fp32": y, "meta": meta,
            "out_quant": case.out_quant}


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
                             "corner) to select a group. --list prints them.")
    parser.add_argument("--bs", type=int, default=BLOCK_SIZE,
                        help=f"MX block size (default {BLOCK_SIZE})")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the weight / input patterns")
    parser.add_argument("--batch", type=int, default=None,
                        help="override the batch dim of the input")
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
        yield build_case(name, bs=args.bs, seed=args.seed, batch=args.batch)


if __name__ == "__main__":
    import argparse

    parser = add_model_args(argparse.ArgumentParser(description=__doc__))
    parser.set_defaults(model="all")
    args = parser.parse_args()
    args.list = True          # models.py standalone only prints the table
    list(cases_from_args(args))
