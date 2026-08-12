"""Static-scale fixed-point fake quantization.

Unlike MX (block floating point, one shared exponent per block, scale derived
from the data), the accelerator emits the network *output* as plain fixed-point
with a scale known up front: N fractional bits, so the step is exactly
2^-frac_bits regardless of the values.

For the DOF flow field that is Q8.8 signed — 8 integer bits (whole pixels) plus
8 fractional bits (subpixels): step 1/256, range [-128, +127.99609375].

Fake quant only: FP32 in, FP32 out, values snapped to the lattice, dtype never
changes. Backward is a straight-through estimator, matching the convention in
`mx_layers_blocked.quantize_mx_op`.

The single failure mode worth watching is saturation: MX cannot overflow (the
shared exponent is set from the block max), but a static scale can, and it does
so silently. `fxp_clip_stats` exists to measure exactly that.
"""

import torch

_VALID_ROUND_MODES = ("half_away", "half_even", "trunc")
_MIN_TOTAL_BITS = 2
_MAX_TOTAL_BITS = 32


OUT_QUANT_DEFAULTS = {
    "total_bits": 16,        # full width including the sign bit
    "frac_bits": 8,          # scale = 2^-frac_bits
    "signed": True,
    "round": "half_away",    # 'half_away' | 'half_even' | 'trunc'
    "saturate": True,        # clamp to the representable range (False = lattice only)
    "clip_grad": False,      # True = zero the gradient of saturated elements
    "outputs": "all",        # 'all', or a list of indices when the module returns a tuple
}


def normalize_out_quant(value):
    """Normalize a user-supplied out_quant config into a canonical dict.

    Accepts None/False (disabled), True (defaults), or a dict merged over the
    defaults. Unknown keys raise, so a typo in mx_config.json fails loudly
    instead of silently doing nothing.
    """
    if value is None or value is False:
        return {**OUT_QUANT_DEFAULTS, "enabled": False}
    if value is True:
        return {**OUT_QUANT_DEFAULTS, "enabled": True}
    if not isinstance(value, dict):
        raise TypeError(
            f"out_quant must be bool or dict, got {type(value).__name__}"
        )

    cfg = {**OUT_QUANT_DEFAULTS, "enabled": True}
    for k, v in value.items():
        if k not in cfg:
            raise ValueError(
                f"unknown out_quant key: {k!r}; valid keys: "
                f"{sorted(k for k in cfg if k != 'enabled')}"
            )
        cfg[k] = v

    if not cfg["enabled"]:
        return cfg

    tb, fb = cfg["total_bits"], cfg["frac_bits"]
    if not isinstance(tb, int) or isinstance(tb, bool):
        raise TypeError(f"out_quant.total_bits must be int, got {type(tb).__name__}")
    if not isinstance(fb, int) or isinstance(fb, bool):
        raise TypeError(f"out_quant.frac_bits must be int, got {type(fb).__name__}")
    if tb < _MIN_TOTAL_BITS or tb > _MAX_TOTAL_BITS:
        raise ValueError(
            f"out_quant.total_bits={tb} out of range "
            f"[{_MIN_TOTAL_BITS}, {_MAX_TOTAL_BITS}]"
        )
    if not isinstance(cfg["signed"], bool):
        raise TypeError(
            f"out_quant.signed must be bool, got {type(cfg['signed']).__name__}")
    # frac_bits may exceed total_bits only in principle; in practice that leaves no
    # integer range at all, which is always a config mistake.
    int_bits = tb - fb - (1 if cfg["signed"] else 0)
    if int_bits < 0:
        raise ValueError(
            f"out_quant: frac_bits={fb} leaves no integer bits in a "
            f"{'signed' if cfg['signed'] else 'unsigned'} {tb}-bit word"
        )
    if cfg["round"] not in _VALID_ROUND_MODES:
        raise ValueError(
            f"out_quant.round must be in {_VALID_ROUND_MODES}, got {cfg['round']!r}"
        )
    for key in ("saturate", "clip_grad"):
        if not isinstance(cfg[key], bool):
            raise TypeError(
                f"out_quant.{key} must be bool, got {type(cfg[key]).__name__}")
    outs = cfg["outputs"]
    if outs != "all":
        if not isinstance(outs, (list, tuple)) or not all(
                isinstance(i, int) and not isinstance(i, bool) for i in outs):
            raise ValueError(
                f"out_quant.outputs must be 'all' or a list of ints, got {outs!r}")
        cfg["outputs"] = list(outs)

    return cfg


def fxp_range(total_bits=16, frac_bits=8, signed=True):
    """Integer code bounds (lo, hi) and the float step for a fixed-point format."""
    if signed:
        lo = -(1 << (total_bits - 1))
        hi = (1 << (total_bits - 1)) - 1
    else:
        lo = 0
        hi = (1 << total_bits) - 1
    return lo, hi, 2.0 ** -frac_bits


def _round_to_int(v, round_mode):
    if round_mode == "half_away":
        # torch has no round-half-away-from-zero; |v| + 0.5 floored gives it, and
        # copysign puts the sign back (sign() would map exact zeros to 0, which is
        # the same value here, but copysign keeps -0.0 behaviour sane).
        return torch.floor(v.abs() + 0.5).copysign(v)
    if round_mode == "half_even":
        return torch.round(v)
    if round_mode == "trunc":
        return torch.trunc(v)
    raise ValueError(
        f"round_mode must be in {_VALID_ROUND_MODES}, got {round_mode!r}")


def fake_quant_fxp(x, frac_bits=8, total_bits=16, signed=True,
                   round_mode="half_away", saturate=True, clip_grad=False):
    """Snap `x` onto a static fixed-point lattice. FP32 in, FP32 out, STE back.

        code  = round(x * 2^frac_bits)          # per `round_mode`
        code  = clamp(code, lo, hi)             # if saturate
        out   = code * 2^-frac_bits

    Args:
        x: float tensor. Non-float or non-tensor input is returned untouched.
        frac_bits: fractional bits; the scale is 2^-frac_bits.
        total_bits: full word width, including the sign bit when `signed`.
        signed: two's-complement range vs unsigned.
        round_mode: 'half_away' (HW convention) | 'half_even' (torch.round) | 'trunc'.
        saturate: clamp out-of-range values to the ends. False keeps the lattice
            but allows any magnitude — useful to isolate rounding from clipping.
        clip_grad: zero the gradient of elements that saturated. Default False
            (plain pass-through), consistent with the rest of the repo's STE.
    """
    if not torch.is_tensor(x) or not x.is_floating_point():
        return x

    lo, hi, _ = fxp_range(total_bits, frac_bits, signed)
    scale = 2.0 ** frac_bits

    v = x * scale
    code = _round_to_int(v, round_mode)
    if saturate:
        code = code.clamp(lo, hi)
    x_q = code / scale

    if not x.requires_grad:
        return x_q

    if clip_grad and saturate:
        inside = ((v >= lo) & (v <= hi)).to(x.dtype)
        return x_q.detach() + (x - x.detach()) * inside
    return x + (x_q - x).detach()


@torch.no_grad()
def fxp_clip_stats(x, x_q=None, frac_bits=8, total_bits=16, signed=True,
                   round_mode="half_away"):
    """Running-stat contribution for one tensor: (n, n_clipped, sum_sq, sum_sq_err).

    `n_clipped` counts elements whose *rounded code* fell outside the
    representable range — i.e. values the format cannot hold. Pass `x_q` to
    reuse an already-computed quantization instead of redoing it.

    The three sums come back as 0-dim tensors on the input's device, never as
    python floats: this runs on every QAT step, and calling `.item()` here
    would force a host sync per forward. Reduce them with `fxp_stats_value`
    only at report time.
    """
    if not torch.is_tensor(x) or not x.is_floating_point():
        return 0, 0, 0.0, 0.0

    lo, hi, _ = fxp_range(total_bits, frac_bits, signed)
    xf = x.detach().float()
    code = _round_to_int(xf * (2.0 ** frac_bits), round_mode)
    n_clipped = ((code < lo) | (code > hi)).sum()

    if x_q is None:
        x_q = code.clamp(lo, hi) / (2.0 ** frac_bits)
    err = xf - x_q.detach().float()

    return xf.numel(), n_clipped, (xf * xf).sum(), (err * err).sum()


def fxp_stats_value(v):
    """Collapse an accumulated stat (tensor or python number) to a float."""
    return v.item() if torch.is_tensor(v) else float(v)


def fxp_format_str(cfg):
    """Human-readable format tag, e.g. 'Q8.8 signed/16b round=half_away'."""
    tb, fb = cfg["total_bits"], cfg["frac_bits"]
    int_bits = tb - fb - (1 if cfg["signed"] else 0)
    return (f"Q{int_bits + (1 if cfg['signed'] else 0)}.{fb} "
            f"{'signed' if cfg['signed'] else 'unsigned'}/{tb}b "
            f"round={cfg['round']}")
