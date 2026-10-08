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

The scale does not have to be a power of two. `frac_bits` is the common case
(step 2^-frac_bits, a pure shift in hardware), but a post-training static
quantizer generally lands on an arbitrary positive real step, which `scale`
carries directly. `scale` wins over `frac_bits` when both are present; the
power-of-two path is kept separate so it stays bit-identical to a shift.

The single failure mode worth watching is saturation: MX cannot overflow (the
shared exponent is set from the block max), but a static scale can, and it does
so silently. `fxp_clip_stats` exists to measure exactly that.
"""

import math

import torch

_VALID_ROUND_MODES = ("half_away", "half_even", "trunc")
_MIN_TOTAL_BITS = 2
_MAX_TOTAL_BITS = 32


OUT_QUANT_DEFAULTS = {
    "total_bits": 16,        # full width including the sign bit
    "frac_bits": 8,          # step = 2^-frac_bits (ignored when `scale` is set)
    "scale": None,           # explicit step, any positive real; overrides frac_bits
    "calibrate": False,      # True = `scale` is frozen from data by a calibration pass
    "pow2": False,           # calibration rounds the step up to a power of two
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
    for key in ("calibrate", "pow2"):
        if not isinstance(cfg[key], bool):
            raise TypeError(
                f"out_quant.{key} must be bool, got {type(cfg[key]).__name__}")

    sc = cfg["scale"]
    if sc is not None:
        if isinstance(sc, bool) or not isinstance(sc, (int, float)):
            raise TypeError(
                f"out_quant.scale must be a positive number or None, got "
                f"{type(sc).__name__}")
        if not (sc > 0) or sc == float("inf"):
            raise ValueError(f"out_quant.scale must be finite and > 0, got {sc!r}")
        cfg["scale"] = float(sc)
    elif cfg["calibrate"]:
        # Scale comes from a calibration pass, so there is nothing to check yet.
        pass
    else:
        # frac_bits may exceed total_bits only in principle; in practice that leaves
        # no integer range at all, which is always a config mistake. An explicit
        # `scale` makes frac_bits unused, so this check does not apply there.
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


def fxp_code_range(total_bits=16, signed=True):
    """Integer code bounds (lo, hi) for a word width, independent of the scale."""
    if signed:
        return -(1 << (total_bits - 1)), (1 << (total_bits - 1)) - 1
    return 0, (1 << total_bits) - 1


def fxp_range(total_bits=16, frac_bits=8, signed=True, scale=None):
    """Integer code bounds (lo, hi) and the float step for a fixed-point format."""
    lo, hi = fxp_code_range(total_bits, signed)
    return lo, hi, float(scale) if scale is not None else 2.0 ** -frac_bits


def fxp_step(cfg):
    """The lattice step of a normalized out_quant dict."""
    sc = cfg.get("scale")
    return float(sc) if sc is not None else 2.0 ** -cfg["frac_bits"]


def fxp_max_abs(cfg):
    """Largest magnitude a normalized out_quant format can hold."""
    lo, hi = fxp_code_range(cfg["total_bits"], cfg["signed"])
    return max(abs(lo), abs(hi)) * fxp_step(cfg)


def fxp_scale_for_max_abs(max_abs, total_bits=8, signed=True, pow2=False):
    """Static scale that just covers `max_abs` in a `total_bits` word.

    This is the post-training step: the scale is frozen from observed data, not
    derived per block the way MX derives a shared exponent. `pow2=True` rounds
    the step up to the next power of two, which is what a shift-only datapath
    can do; the default keeps the arbitrary real step a float multiplier gives,
    which uses the full code range and so is 2-5 dB better on awkward ranges.
    """
    # The positive bound is the binding one: a signed word reaches -2^(b-1) but
    # only +2^(b-1)-1, so scaling by |lo| would clip the largest positive value.
    _lo, qmax = fxp_code_range(total_bits, signed)
    max_abs = float(max_abs)
    if not (max_abs > 0):
        # An all-zero tensor has no scale; pick the finest step so nothing clips.
        return 2.0 ** -(total_bits - (1 if signed else 0))
    step = max_abs / qmax
    if pow2:
        step = 2.0 ** math.ceil(math.log2(step))
    return step


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
                   round_mode="half_away", saturate=True, clip_grad=False,
                   scale=None):
    """Snap `x` onto a static fixed-point lattice. FP32 in, FP32 out, STE back.

        code  = round(x / step)                 # per `round_mode`
        code  = clamp(code, lo, hi)             # if saturate
        out   = code * step

    with `step = scale` when given, else the power of two 2^-frac_bits.

    Args:
        x: float tensor. Non-float or non-tensor input is returned untouched.
        frac_bits: fractional bits; the step is 2^-frac_bits.
        scale: explicit step, any positive real. Overrides `frac_bits`. This is
            what a post-training static quantizer produces when it is not
            restricted to shifts.
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

    lo, hi = fxp_code_range(total_bits, signed)

    if scale is None:
        # Pure shift: multiplying by a power of two is exact, so keep this path
        # separate from the general one rather than dividing by 2^-frac_bits.
        mul = 2.0 ** frac_bits
        v = x * mul
        code = _round_to_int(v, round_mode)
        if saturate:
            code = code.clamp(lo, hi)
        x_q = code / mul
    else:
        step = float(scale)
        v = x / step
        code = _round_to_int(v, round_mode)
        if saturate:
            code = code.clamp(lo, hi)
        x_q = code * step

    if not x.requires_grad:
        return x_q

    if clip_grad and saturate:
        inside = ((v >= lo) & (v <= hi)).to(x.dtype)
        return x_q.detach() + (x - x.detach()) * inside
    return x + (x_q - x).detach()


@torch.no_grad()
def fxp_clip_stats(x, x_q=None, frac_bits=8, total_bits=16, signed=True,
                   round_mode="half_away", scale=None):
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

    lo, hi, step = fxp_range(total_bits, frac_bits, signed, scale)
    xf = x.detach().float()
    code = _round_to_int(xf / step, round_mode)
    n_clipped = ((code < lo) | (code > hi)).sum()

    if x_q is None:
        x_q = code.clamp(lo, hi) * step
    err = xf - x_q.detach().float()

    return xf.numel(), n_clipped, (xf * xf).sum(), (err * err).sum()


def fxp_stats_value(v):
    """Collapse an accumulated stat (tensor or python number) to a float."""
    return v.item() if torch.is_tensor(v) else float(v)


def fxp_format_str(cfg):
    """Human-readable format tag, e.g. 'Q8.8 signed/16b round=half_away'.

    A non-power-of-two scale has no Qm.n name, so it prints as the step itself.
    """
    tb = cfg["total_bits"]
    sign = 'signed' if cfg["signed"] else 'unsigned'
    if cfg.get("scale") is not None:
        return (f"int{tb} {sign} step={cfg['scale']:.6g} "
                f"round={cfg['round']}")
    if cfg.get("calibrate"):
        return (f"int{tb} {sign} step=uncalibrated"
                f"{' pow2' if cfg.get('pow2') else ''} round={cfg['round']}")
    fb = cfg["frac_bits"]
    int_bits = tb - fb - (1 if cfg["signed"] else 0)
    return (f"Q{int_bits + (1 if cfg['signed'] else 0)}.{fb} "
            f"{sign}/{tb}b round={cfg['round']}")
