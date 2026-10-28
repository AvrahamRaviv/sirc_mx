"""HW-faithful fixed-point emulation for MX Conv2d.

Real hardware has a single Nb x Nb -> 2Nb multiplier per BCU (not a block
dot product). Each product is shifted by `Ew + Ea - 2*mant_bias - E_layer_min`
and pushed into a narrow saturating fixed-point accumulator,
product-by-product. Intra-block reduction happens inside the fixed-point
accumulator, not in FP32.

This module provides:
  * `extract_mxint` - recover intN values and shared exponent from a
    tensor already quantised by microxcaling's `quantize_mx_op`. Supports
    any MXINT format (int2..int16). `extract_mxint8` is a back-compat alias.
  * `_hw_fxp_conv2d_ref` - pure-torch reference that computes the same
    numerics as the kernel (parallel over B,O,L; serial over K/bs).
  * `HWFxpConv2dFn` - autograd Function with STE backward (treats forward
    as if it were `F.conv2d` for gradient purposes).
  * `hw_fxp_conv2d` - public entry point used by `MXConv2dHW`.
  * `calibrate_e_layer_min` - sweep representative inputs and freeze the
    per-layer static `e_layer_min` attribute.
  * `calibrate_hw_grids` - same sweep, also freezes the output grid
    `e_qa_out` that the static output window extracts at.

Output window (the back half): the 48b accumulator word N sits on the grid
2^e_layer_min. HW extracts O = N >> (e_qa_out - e_layer_min), keeps
`win_bits` magnitude bits and saturates at +-(2^win_bits - 1); the output is
O * 2^e_qa_out. `win_bits` is what one output MX block can address:
(mbits-1) mantissa magnitude bits + (2^out_exp_bits - 1) exponent steps,
22 for int8 with a 4b shared exponent (23 with sign). Without a window the
word is just cast to fp32 (dynamic, top-aligned ~24 bits). The next layer's MX requant then uses a full-res
shared exponent; HW stores it in 4 bits, which decodes to the same lattice
in range. TODO: model the 4b output exponent clamp (saturate/reset tails).

PLA (xblock_accum.pla, default off): the PCU's per-channel multiply between
the accumulator and the window folds the activation that follows the conv
(PReLU / LeakyReLU / ReLU). Per output channel two signed `pla_mult_bits`
multipliers, m_pos = 2^frac and m_neg = q(alpha * 2^frac), are picked by the
sign of the word, P = N * m (48 + 8 = 56b), and the unchanged window extracts
O from P on the grid 2^(e_layer_min - frac). See `PLAParams`. pla_bias='bins'
is the HW form: per channel bins split by boundaries on the bias-free
accumulator, each with its own int8 scale S_k and bias B_k, P = S_k * N + B_k
(`pla_bins`).

Only MXINT formats are supported. FP8/MXFP paths are rejected.
"""

from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F

MANTISSA_BIAS = 6  # legacy: MXINT8 mantissa bias. New code: use _int_format_params.
OUT_EXP_BITS = 4   # HW output shared-exponent width (xblock_accum.out_exp_bits)


def _int_format_params(fmt):
    """Returns (mbits, mant_bias, max_val, store_dtype) for an MXINT format.

    Raises ValueError for non-int formats.
    """
    from microxcaling.mx.formats import _get_format_params, ElemFormat

    if isinstance(fmt, str):
        try:
            elem = ElemFormat.from_str(fmt)
        except Exception as e:
            raise ValueError(f"Unknown elem_format: {fmt!r}") from e
    else:
        elem = fmt

    ebits, mbits, _, _, _ = _get_format_params(elem)
    if ebits != 0:
        raise ValueError(
            f"HW fixed-point path requires intN format; got {fmt!r} "
            f"(ebits={ebits}, not int)."
        )
    mant_bias = mbits - 2
    max_val = (1 << (mbits - 1)) - 1
    store_dtype = torch.int8 if mbits <= 8 else torch.int16
    return mbits, mant_bias, max_val, store_dtype


def extract_mxint(x_q_fp32, bs, axis, fmt='int8'):
    """Recover (intN, shared_exp_int16) from an MXINT-quantised FP32 tensor.

    `x_q_fp32` is assumed to already lie on the MXINT lattice (produced by
    `microxcaling.mx.quantize_mx_op` with the matching elem_format). The
    recovery is exact on-lattice.

    Parameters
    ----------
    fmt : str | ElemFormat — any MXINT format supported by microxcaling
          (int2, int4, int5, int6, int7, int8, int10, int12, int16).

    Returns
    -------
    int_tensor : same shape as `x_q_fp32`, dtype int8 (N<=8) or int16 (N>8)
    shared_exp : `x_q_fp32` shape with `axis` reduced from C to nb = C/bs,
                 dtype int16
    """
    _, mant_bias, max_val, store_dtype = _int_format_params(fmt)

    x = x_q_fp32.detach().to(torch.float32)
    x_last = x.movedim(axis, -1).contiguous()
    prefix = x_last.shape[:-1]
    C = x_last.shape[-1]
    assert C % bs == 0, f"channel dim {C} not divisible by block_size {bs}"
    nb = C // bs

    xb = x_last.view(*prefix, nb, bs)
    max_abs = xb.abs().amax(dim=-1, keepdim=True)
    nonzero = max_abs > 0
    safe = torch.where(nonzero, max_abs, torch.ones_like(max_abs))
    E = torch.floor(torch.log2(safe))
    E = torch.where(nonzero, E, torch.zeros_like(E))

    scale_inv = torch.pow(torch.tensor(2.0, device=x.device), mant_bias - E)
    q = torch.round(xb * scale_inv).clamp_(-max_val, max_val).to(store_dtype)

    q_full = q.view(*prefix, C).movedim(-1, axis).contiguous()
    E_full = E.squeeze(-1).to(torch.int16).movedim(-1, axis).contiguous()
    return q_full, E_full


def extract_mxint8(x_q_fp32, bs, axis):
    """Back-compat alias: extract_mxint with fmt='int8'."""
    return extract_mxint(x_q_fp32, bs, axis, fmt='int8')


def _mxint_mantissa_exp(xb, mant_bias, max_val, store_dtype):
    """Shared MXINT block decode over the last axis of `xb` ([..., blk_len]).

    Returns (q, E): integer mantissas (same shape) and the per-block shared
    exponent E (shape [..., 1]). Mirrors `extract_mxint`'s math exactly.
    """
    max_abs = xb.abs().amax(dim=-1, keepdim=True)
    nonzero = max_abs > 0
    safe = torch.where(nonzero, max_abs, torch.ones_like(max_abs))
    E = torch.floor(torch.log2(safe))
    E = torch.where(nonzero, E, torch.zeros_like(E))
    scale_inv = torch.pow(torch.tensor(2.0, device=xb.device), mant_bias - E)
    q = torch.round(xb * scale_inv).clamp_(-max_val, max_val).to(store_dtype)
    return q, E


def extract_mxint_flatten(qw_fp, bs, fmt='int8'):
    """NPE weight decode: per output filter, flatten [Cin,kH,kW] -> 1D and block
    by `bs` along that stream (blocks cross channel boundaries; tail block is
    partial, zero-padded internally — zeros do not change the block max).

    Returns
    -------
    q_full : same shape as `qw_fp` ([O,Cin,kH,kW]), intN mantissas.
    E_full : same shape, int16 — each weight element carries its flatten-block
             shared exponent (full resolution, drives the bs=1 ref).
    """
    _, mant_bias, max_val, store_dtype = _int_format_params(fmt)
    x = qw_fp.detach().to(torch.float32)
    O = x.shape[0]
    flat = x.reshape(O, -1)                       # [O, Cin*kH*kW]
    K = flat.shape[1]
    nb = (K + bs - 1) // bs
    pad = nb * bs - K
    if pad:
        flat = F.pad(flat, (0, pad))              # tail zeros; block max unchanged
    xb = flat.view(O, nb, bs)
    q, E = _mxint_mantissa_exp(xb, mant_bias, max_val, store_dtype)
    q_full = q.view(O, nb * bs)[:, :K].reshape(qw_fp.shape).contiguous()
    E_full = E.expand(O, nb, bs).reshape(O, nb * bs)[:, :K] \
        .reshape(qw_fp.shape).to(torch.int16).contiguous()
    return q_full, E_full


def extract_mxint_xblock(qi_fp, bs, fmt='int8'):
    """NPE activation decode: block along W (width, axis=3) by `bs`, per (b,c,y)
    (tail X-block partial, zero-padded internally).

    Returns
    -------
    q_full : same shape as `qi_fp` ([B,C,H,W]), intN mantissas.
    E_full : same shape, int16 — each activation element carries its X-block
             shared exponent (full resolution, drives the bs=1 ref).
    """
    _, mant_bias, max_val, store_dtype = _int_format_params(fmt)
    x = qi_fp.detach().to(torch.float32)
    B, C, H, W = x.shape
    nb = (W + bs - 1) // bs
    pad = nb * bs - W
    if pad:
        x = F.pad(x, (0, pad))                    # pad width; block max unchanged
    xb = x.view(B, C, H, nb, bs)
    q, E = _mxint_mantissa_exp(xb, mant_bias, max_val, store_dtype)
    q_full = q.view(B, C, H, nb * bs)[..., :W].reshape(qi_fp.shape).contiguous()
    E_full = E.expand(B, C, H, nb, bs).reshape(B, C, H, nb * bs)[..., :W] \
        .reshape(qi_fp.shape).to(torch.int16).contiguous()
    return q_full, E_full


def derive_out_window_bits(fmt, out_exp_bits=OUT_EXP_BITS):
    """Magnitude bits of the output window for an MXINT `fmt` output with an
    `out_exp_bits` shared exponent, all 2^out_exp_bits codes usable.

    An element is m * 2^(E - mant_bias), |m| <= 2^(mbits-1) - 1, so one block
    covers bit positions E - mant_bias .. E. E spans 2^out_exp_bits values,
    so the format addresses (mbits-1) + (2^out_exp_bits - 1) bits: int8/4b
    -> 7 + 15 = 22. With this width, floor(log2(AbsMax)) - (win_bits-1) is
    the min block exp - mant_bias of a tensor that uses the whole exponent
    range, so the guide's two E_q forms agree.
    """
    mbits, _, _, _ = _int_format_params(fmt)
    return (mbits - 1) + (1 << int(out_exp_bits)) - 1


def resolve_out_window_bits(cfg, fmt):
    """`xblock_accum.out_window_bits` if set, else derived from `fmt` and
    `out_exp_bits`."""
    wb = cfg.get("out_window_bits")
    if wb is None:
        wb = derive_out_window_bits(fmt, int(cfg.get("out_exp_bits", OUT_EXP_BITS)))
    if not 1 <= int(wb) <= 46:
        raise ValueError(
            f"output window of {wb} bits does not fit the int64 emulation "
            f"(max 46); lower out_exp_bits or set out_window_bits.")
    return int(wb)


def _window_params(e_acc, e_qa_out, win_bits, out_round):
    """Integer constants of the static output window, shared by the torch ref
    and the triton launchers.

    shift_out = e_qa_out - e_acc. Positive: right shift, drop the bits below
    the window (`round_add` turns floor into round-half-up). Negative: left
    shift, the accumulator holds fewer than `win_bits` significant bits.
    `pre_hi` is the largest |value| that survives the left shift unsaturated,
    so saturation is detected before the shift can overflow.
    """
    if win_bits is None:
        raise ValueError("output window needs win_bits; see derive_out_window_bits")
    s = int(e_qa_out) - int(e_acc)
    rshift = max(s, 0)
    lshift = max(-s, 0)
    round_add = (1 << (rshift - 1)) if (out_round == "nearest" and rshift > 0) else 0
    win_hi = (1 << int(win_bits)) - 1
    pre_hi = win_hi >> lshift
    return rshift, lshift, round_add, win_hi, pre_hi


def _apply_out_window(acc, e_acc, e_qa_out, win_bits, out_round="floor"):
    """O = acc >> (e_qa_out - e_acc), saturated to +-(2^win_bits - 1).

    acc : int64 tensor on the grid 2^e_acc. Returns (O int64 on the grid
    2^e_qa_out, saturation mask). O * 2^e_qa_out is exact in fp32 for
    win_bits <= 24.
    """
    rshift, lshift, round_add, win_hi, pre_hi = _window_params(
        e_acc, e_qa_out, win_bits, out_round)
    v = (acc + round_add) >> rshift
    over = v > pre_hi
    under = v < -pre_hi
    v = v << lshift
    v = torch.where(over, torch.full_like(v, win_hi), v)
    v = torch.where(under, torch.full_like(v, -win_hi), v)
    return v, over | under


class PLAParams(NamedTuple):
    """Integer PCU multiply of one layer (`_apply_pla`).

    m_pos, m_neg : int64 [O], signed multipliers for a word >= 0 / < 0
    frac         : fraction bits of both; P sits on the grid 2^(e_acc - frac)
    bias         : 'pre'  - bias in the accumulator, so the segment follows
                            the sign of acc + bias (what PReLU(conv) needs)
                   'post' - emulator single-bin order (pcu_pla): P = N * m +
                            bias * m_pos, segment by the sign of the bias-free
                            N. Equals 'pre' only for m_neg == m_pos; with a
                            fold it differs where N and N + bias have other
                            signs
                   'bins' - HW bin form: bias-free N, bin k where
                            bounds[k-1] <= N < bounds[k] ('upper' tie: N equal
                            to a boundary goes to the bin above), P = scales[k]
                            * N + biases[k]. See `pla_bins`
    bounds       : int64 [O, nbins - 1] ascending, on the accumulator grid
    scales       : int64 [O, nbins] signed, frac fraction bits
    biases       : int64 [O, nbins] on the grid 2^(e_acc - frac)
    tie          : 'upper' | 'lower', bin of N == boundary
    """
    m_pos: torch.Tensor
    m_neg: torch.Tensor
    frac: int
    bias: str = "pre"
    bounds: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None
    biases: Optional[torch.Tensor] = None
    tie: str = "upper"


def pla_multipliers(alpha, mult_bits=8, frac_bits=None):
    """(m_pos, m_neg, frac) of y = z (z >= 0), alpha * z (z < 0), unit scale.

    m_pos = 2^frac, m_neg = floor(alpha * 2^frac + 0.5) (offline, round half
    up as the HW bias), both signed `mult_bits`. `frac_bits` None = the
    largest frac where both fit (8b, |alpha| <= 1 -> 6).
    """
    a = torch.as_tensor(alpha).detach().to(torch.float64).cpu().flatten()
    lo, hi = -(1 << (mult_bits - 1)), (1 << (mult_bits - 1)) - 1

    def fits(f):
        mn = torch.floor(a * 2.0 ** f + 0.5)
        return (1 << f) <= hi and int(mn.min()) >= lo and int(mn.max()) <= hi

    if frac_bits is None:
        frac = next((f for f in range(mult_bits - 2, -1, -1) if fits(f)), None)
        if frac is None:
            raise ValueError(f"PLA: slopes {a.min().item():.4g}..{a.max().item():.4g} "
                             f"do not fit {mult_bits}b multipliers")
    else:
        frac = int(frac_bits)
        if not fits(frac):
            raise ValueError(f"PLA: frac {frac} overflows {mult_bits}b "
                             f"multipliers (slopes {a.min().item():.4g}.."
                             f"{a.max().item():.4g})")
    m_neg = torch.floor(a * 2.0 ** frac + 0.5).to(torch.int64)
    return torch.full_like(m_neg, 1 << frac), m_neg, frac


def _bias_to_grid(bias_fp, e_grid, half_up, device):
    """Bias on the integer grid 2^e_grid: round half to even, or with
    `half_up` floor(v + 0.5), the HW's offline rounding (used with PLA)."""
    v = bias_fp.detach().to(torch.float64).to(device) * float(2.0 ** (-int(e_grid)))
    return ((v + 0.5).floor() if half_up else v.round()).to(torch.int64)


def pla_bins(m_pos, m_neg, bias_fp, e_acc, frac):
    """(bounds, scales, biases) of the two-bin HW form of y = z (z >= 0),
    alpha * z (z < 0) on z = N * 2^e_acc + b (PReLU / LeakyReLU / ReLU).

    Bin 0 (N < -b_acc): S = m_neg, bin 1: S = m_pos, with b_acc = floor(b *
    2^-e_acc + 0.5) the accumulator bias of 'pre', so both pick the same
    segment. B_k = floor(m_k * b * 2^-e_acc + 0.5) on the grid 2^(e_acc -
    frac), offline, round half up as the HW bias. 'pre' instead has m_k *
    b_acc; the two differ by at most m_k / 2 + 1/2 there."""
    m_pos = m_pos.to(torch.int64).cpu()
    m_neg = m_neg.to(torch.int64).cpu()
    scales = torch.stack([m_neg, m_pos], 1)
    if bias_fp is None:
        z = torch.zeros_like(m_pos)
        return z.view(-1, 1), scales, torch.zeros_like(scales)
    v = bias_fp.detach().to(torch.float64).cpu().flatten() * float(2.0 ** (-int(e_acc)))
    b_acc = (v + 0.5).floor().to(torch.int64)
    biases = (scales.to(torch.float64) * v.view(-1, 1) + 0.5).floor().to(torch.int64)
    return (-b_acc).view(-1, 1), scales, biases


def _apply_pla(acc, pla, bias_post=None):
    """P = acc * (m_neg if acc < 0 else m_pos) [+ bias_post], per output
    channel (dim 1); 'bins': P = scales[k] * acc + biases[k] of acc's bin.
    int64 exact: |acc| < 2^(bits-1), |m| <= 2^(mult_bits-1)."""
    shape = (1, acc.shape[1]) + (1,) * (acc.dim() - 2)
    if pla.bias == "bins":
        sc, bi = pla.scales.to(acc.device), pla.biases.to(acc.device)
        bd = pla.bounds.to(acc.device)
        s, b = sc[:, 0].view(shape), bi[:, 0].view(shape)
        for k in range(1, sc.shape[1]):
            t = bd[:, k - 1].view(shape)
            sel = acc >= t if pla.tie == "upper" else acc > t
            s = torch.where(sel, sc[:, k].view(shape), s)
            b = torch.where(sel, bi[:, k].view(shape), b)
        return acc * s + b
    m = torch.where(acc < 0, pla.m_neg.to(acc.device).view(shape),
                    pla.m_pos.to(acc.device).view(shape))
    p = acc * m
    return p if bias_post is None else p + bias_post.view(shape)


def e_qa_out_from_absmax(absmax, win_bits, margin_bits=0):
    """Output grid from a calibrated |max|: floor(log2(absmax)) - (win_bits-1)
    + margin_bits, so absmax itself fits the window unsaturated. None for an
    all-zero output."""
    import math
    if not absmax > 0:
        return None
    _, ex = math.frexp(float(absmax))      # absmax = m * 2^ex, m in [0.5, 1)
    return (ex - 1) - (int(win_bits) - 1) + int(margin_bits)


def floor_log2(x):
    """floor(log2(x)) for x > 0 - the MX block exponent of a block whose
    |max| is x. None for x <= 0."""
    import math
    if not x > 0:
        return None
    return math.frexp(float(x))[1] - 1


def clamp_block_exp(q, e_base, exp_bits, fmt='int8'):
    """Model a `exp_bits`-wide HW shared-exponent register on an MX tensor.

    Allowed block exponents: [e_base, e_base + 2^exp_bits - 1]. A block
    below e_base is re-rounded (nearest, half away, as microxcaling) to that
    step, 2^(e_base - mant_bias); a block above the top saturates to
    +-max_int * 2^(top - mant_bias). Works per element: an in-range block's
    values are multiples of its own step, which the coarser e_base step
    divides, so they come back unchanged. (Re-rounding an already rounded
    value can differ from rounding the float once by one step at a tie.)

    Returns (clamped tensor, number of changed elements).
    """
    _, mant_bias, max_val, _ = _int_format_params(fmt)
    step = 2.0 ** (int(e_base) - mant_bias)
    hi = max_val * 2.0 ** (int(e_base) + (1 << int(exp_bits)) - 1 - mant_bias)
    y = torch.sign(q) * torch.floor(q.abs() / step + 0.5) * step
    y = y.clamp(-hi, hi)
    return y, int((y != q).sum().item())


def _pair(v):
    return tuple(int(t) for t in v) if isinstance(v, (tuple, list)) else (int(v), int(v))


def dilate_input(t, in_dil=1, in_extra=0):
    """Zero-insert on the last two (spatial) dims: in_dil - 1 zeros between
    neighbours, then in_extra zeros at the bottom/right. This is how a
    ConvTranspose2d runs as a stride-1 conv (`MXConvTranspose2dHW`).

    Applied to the extracted operand (intN values and their per-position
    shared exponents), so the blocks were formed on the real tensor by
    whatever act blockify is set; an inserted zero gives a 0 product, so the
    exponent it carries (0) never matters. Returns `t` itself when
    in_dil == 1 and in_extra == 0.
    """
    sH, sW = _pair(in_dil)
    eH, eW = _pair(in_extra)
    if (sH, sW, eH, eW) == (1, 1, 0, 0):
        return t
    *lead, H, W = t.shape
    out = t.new_zeros(*lead, (H - 1) * sH + 1 + eH, (W - 1) * sW + 1 + eW)
    out[..., :(H - 1) * sH + 1:sH, :(W - 1) * sW + 1:sW] = t
    return out


def _undilate(t, in_dil, H, W):
    """Inverse of `dilate_input` on the last two dims (keeps the real H x W)."""
    sH, sW = _pair(in_dil)
    return t[..., :(H - 1) * sH + 1:sH, :(W - 1) * sW + 1:sW]


def _sat(x, lo, hi):
    """Clamp to [lo, hi] and return (clamped, over|under mask)."""
    sat = (x < lo) | (x > hi)
    return x.clamp(lo, hi), sat


def _hw_fxp_conv2d_ref(
    qi_i8, qw_i8, Ea, Ew,
    e_layer_min, stride, padding, dilation,
    bs, bits, sat_mode,
    bias_fp=None,
    mant_bias=MANTISSA_BIAS,
    return_acc=False,
    e_qa_out=None, win_bits=None, out_round="floor", win_sink=None,
    pla=None,
):
    """Pure-torch reference for the HW fixed-point conv2d kernel.

    Parallel over (B, O, L). Serial over K-block index kb, kernel pos kk,
    and within-block lane j. Matches the Triton kernel bit-for-bit when
    `sat_mode` is the same.

    Parameters
    ----------
    qi_i8 : [B, C, H, W]   int8
    qw_i8 : [O, C, kH, kW] int8
    Ea    : [B, nb, H, W]  int16
    Ew    : [O, nb, kH, kW] int16

    Returns
    -------
    out : [B, O, L]   fp32
    sat : [B, O, L]   bool  (True if the accumulator saturated at any step)
    acc : [B, O, L]   int64, only with `return_acc` - the accumulator word
          itself. `out` is that word cast to fp32, which keeps only 24 bits,
          so a bit-exact comparison against HW has to use `acc`.

    With `e_qa_out` set, `out` is the static output window instead of the
    fp32 cast (see `_apply_out_window`); `win_sink`, if given, receives the
    window saturation map as 'win_sat' and the window word as 'win'.
    `pla` (a `PLAParams`, needs `e_qa_out`) multiplies the word before the
    window; `acc` stays the accumulator word.
    """
    if pla is not None and e_qa_out is None:
        raise ValueError("PLA needs the output window (e_qa_out)")
    if not isinstance(stride, tuple):   stride   = (int(stride),   int(stride))
    if not isinstance(padding, tuple):  padding  = (int(padding),  int(padding))
    if not isinstance(dilation, tuple): dilation = (int(dilation), int(dilation))

    B, C, H, W = qi_i8.shape
    O, _, kH, kW = qw_i8.shape
    assert C % bs == 0
    nb = C // bs
    kK = kH * kW

    # im2col needs float; cast back to int64 after.
    qi_unf = F.unfold(
        qi_i8.to(torch.float32), (kH, kW),
        dilation=dilation, padding=padding, stride=stride,
    )  # [B, C*kK, L]
    L = qi_unf.shape[-1]
    qi_unf = qi_unf.view(B, nb, bs, kK, L).to(torch.int64)

    Ea_unf = F.unfold(
        Ea.to(torch.float32), (kH, kW),
        dilation=dilation, padding=padding, stride=stride,
    )  # [B, nb*kK, L]
    Ea_unf = Ea_unf.view(B, nb, kK, L).to(torch.int64)

    qw = qw_i8.view(O, nb, bs, kK).to(torch.int64)
    Ew64 = Ew.view(O, nb, kK).to(torch.int64)  # [O, nb, kK] (flatten kH*kW)

    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    two_bias = 2 * int(mant_bias)
    e_min = int(e_layer_min)

    device = qi_i8.device
    acc = torch.zeros(B, O, L, dtype=torch.int64, device=device)
    sat = torch.zeros(B, O, L, dtype=torch.bool, device=device)

    for kb in range(nb):
        for kk in range(kK):
            shift = (
                Ew64[:, kb, kk].view(1, O, 1)
                + Ea_unf[:, kb, kk, :].view(B, 1, L)
                - two_bias - e_min
            )  # [B, O, L]
            pos_shift = shift.clamp(min=0)
            neg_shift = (-shift).clamp(min=0)

            if sat_mode == "per_block":
                block_acc = torch.zeros(B, O, L, dtype=torch.int64, device=device)

            for j in range(bs):
                qi_v = qi_unf[:, kb, j, kk, :].view(B, 1, L)
                qw_v = qw[:, kb, j, kk].view(1, O, 1)
                p = qi_v * qw_v  # [B, O, L], at most 127*127 = 16129
                s = torch.where(shift >= 0, p << pos_shift, p >> neg_shift)
                if sat_mode == "per_product":
                    acc = acc + s
                    acc, sat_step = _sat(acc, lo, hi)
                    sat = sat | sat_step
                else:
                    block_acc = block_acc + s

            if sat_mode == "per_block":
                acc = acc + block_acc
                acc, sat_step = _sat(acc, lo, hi)
                sat = sat | sat_step

    bias_post = None
    if bias_fp is not None and pla is not None and pla.bias == "post":
        # Emulator order: the bias joins after the PCU multiply, on its grid.
        bias_post = _bias_to_grid(bias_fp, e_min - pla.frac, True, device).view(1, O, 1)
    elif bias_fp is not None and not (pla is not None and pla.bias == "bins"):
        # 'bins' carries its biases per bin (`pla_bins`), after the multiply.
        # Project bias onto acc grid: bias_int = round(b * 2^(-e_layer_min)).
        bias_int = _bias_to_grid(bias_fp, e_min, pla is not None, device).view(1, O, 1)
        acc = acc + bias_int
        acc, sat_step = _sat(acc, lo, hi)
        sat = sat | sat_step

    if e_qa_out is not None:
        word, e_word = acc, e_min
        if pla is not None:
            word, e_word = _apply_pla(acc, pla, bias_post), e_min - int(pla.frac)
        O_win, win_sat = _apply_out_window(word, e_word, e_qa_out, win_bits, out_round)
        out = O_win.to(torch.float32) * float(2.0 ** int(e_qa_out))
        if win_sink is not None:
            win_sink["win_sat"] = win_sat
            win_sink["win"] = O_win
    else:
        out = acc.to(torch.float32) * float(2.0 ** e_min)
    if return_acc:
        return out, sat, acc
    return out, sat


def _hw_fxp_conv2d_ref_npe(
    qi_i8, qw_i8, Ea_full, Ew_full,
    e_layer_min, stride, padding, dilation,
    bits, sat_mode,
    bias_fp=None, mant_bias=MANTISSA_BIAS, return_acc=False,
    e_qa_out=None, win_bits=None, out_round="floor", win_sink=None,
    pla=None,
):
    """NPE reference: the two operands carry independent, full-resolution
    per-element shared exponents (weight = flatten-block, act = X-block). This
    is exactly the channel-mode ref with effective bs=1 (one lane per channel),
    so no separate kernel is needed — the shift/accumulate/saturate/de-shift
    logic is identical.

    Ea_full : [B,C,H,W]     int16 (X-block exp broadcast per element)
    Ew_full : [O,C,kH,kW]   int16 (flatten-block exp broadcast per element)
    """
    if sat_mode == "per_block":
        raise ValueError(
            "sat_mode='per_block' is not supported for NPE blockify: the weight "
            "flatten-block and activation X-block decouple, so per-block grouping "
            "is undefined. Use sat_mode='per_product'."
        )
    return _hw_fxp_conv2d_ref(
        qi_i8, qw_i8, Ea_full, Ew_full,
        e_layer_min, stride, padding, dilation,
        bs=1, bits=bits, sat_mode=sat_mode,
        bias_fp=bias_fp, mant_bias=mant_bias, return_acc=return_acc,
        e_qa_out=e_qa_out, win_bits=win_bits, out_round=out_round,
        win_sink=win_sink, pla=pla,
    )


class HWFxpConv2dFn(torch.autograd.Function):
    """HW-faithful fixed-point Conv2d forward + STE backward.

    Forward consumes FP32 operands already on the MXINT8 lattice. Internally
    extracts int8 + shared exponent, runs the kernel, returns a 4D output.

    Backward approximates the forward as a standard `F.conv2d` (STE) so
    gradients propagate through quantised operands. If `ste_mask` is set,
    grad for saturated output lanes is zeroed (accumulator saturation only;
    the output window does not touch the backward).

    `e_qa_out` (None = off) turns on the static output window; see
    `_apply_out_window`.

    `in_dil` / `in_extra` zero-insert the input after extraction (see
    `dilate_input`): a ConvTranspose2d run as a stride-1 conv. Both backends
    get the dilated operand, so they need no change. Defaults are a no-op.
    """

    @staticmethod
    def forward(
        ctx,
        qi_fp, qw_fp, bias_fp,
        e_layer_min, bs, bits, sat_mode, ste_mask,
        stride, padding, dilation, backend,
        stats_sink=None,
        fmt='int8',
        act_blockify='channel', weight_blockify='channel',
        npe_triton_variant='reuse',
        e_qa_out=None, win_bits=None, out_round='floor',
        in_dil=(1, 1), in_extra=(0, 0),
        pla=None,
    ):
        _, mant_bias, _, _ = _int_format_params(fmt)
        win_sink = {} if e_qa_out is not None else None
        win_kw = dict(
            e_qa_out=None if e_qa_out is None else int(e_qa_out),
            win_bits=None if win_bits is None else int(win_bits),
            out_round=str(out_round), win_sink=win_sink, pla=pla,
        )
        npe = (act_blockify == "xblock" and weight_blockify == "flatten")
        # A caller that taps the integer accumulator sets `want_acc` in the
        # sink. Only the python reference keeps the int64 word around.
        want_acc = bool(stats_sink is not None and stats_sink.get("want_acc"))
        acc_flat = None

        with torch.no_grad():
            if npe:
                qi_i8, Ea = extract_mxint_xblock(qi_fp, bs, fmt=fmt)
                qw_i8, Ew = extract_mxint_flatten(qw_fp, bs, fmt=fmt)
            else:
                qi_i8, Ea = extract_mxint(qi_fp, bs, axis=1, fmt=fmt)
                qw_i8, Ew = extract_mxint(qw_fp, bs, axis=1, fmt=fmt)
            # Zero-insert after extraction: blocks are formed on the real
            # input by any act blockify, then the ints and exps are dilated.
            qi_i8 = dilate_input(qi_i8, in_dil, in_extra)
            Ea = dilate_input(Ea, in_dil, in_extra)

        if npe:
            # NPE: independent per-element exponents (act X-block, weight
            # flatten-block). GPU via triton ('reuse' = existing kernel at bs=1,
            # or 'opt' = dedicated per-lane kernel); torch ref otherwise.
            use_triton = (backend == "triton" and qi_fp.is_cuda)
            if use_triton:
                if sat_mode == "per_block":
                    raise ValueError(
                        "sat_mode='per_block' is not supported for NPE blockify; "
                        "use sat_mode='per_product'."
                    )
                from .mx_fixed_point_hw_triton import (
                    _hw_fxp_conv2d_triton, _hw_fxp_conv2d_triton_npe,
                )
                if npe_triton_variant == "opt":
                    out_flat, sat_flat = _hw_fxp_conv2d_triton_npe(
                        qi_i8, qw_i8, Ea, Ew,
                        int(e_layer_min), stride, padding, dilation,
                        int(bits), sat_mode,
                        bias_fp=bias_fp, mant_bias=int(mant_bias), **win_kw,
                    )
                else:  # 'reuse': existing kernel with effective bs=1
                    out_flat, sat_flat = _hw_fxp_conv2d_triton(
                        qi_i8, qw_i8, Ea, Ew,
                        int(e_layer_min), stride, padding, dilation,
                        1, int(bits), sat_mode,
                        bias_fp=bias_fp, mant_bias=int(mant_bias), **win_kw,
                    )
            else:
                res = _hw_fxp_conv2d_ref_npe(
                    qi_i8, qw_i8, Ea, Ew,
                    int(e_layer_min), stride, padding, dilation,
                    int(bits), sat_mode,
                    bias_fp=bias_fp,
                    mant_bias=int(mant_bias),
                    return_acc=want_acc,
                    **win_kw,
                )
                out_flat, sat_flat = res[0], res[1]
                acc_flat = res[2] if want_acc else None
        else:
            use_triton = (backend == "triton" and qi_fp.is_cuda)
            if use_triton:
                from .mx_fixed_point_hw_triton import _hw_fxp_conv2d_triton
                out_flat, sat_flat = _hw_fxp_conv2d_triton(
                    qi_i8, qw_i8, Ea, Ew,
                    int(e_layer_min), stride, padding, dilation,
                    bs, int(bits), sat_mode,
                    bias_fp=bias_fp,
                    mant_bias=int(mant_bias),
                    **win_kw,
                )
            else:
                res = _hw_fxp_conv2d_ref(
                    qi_i8, qw_i8, Ea, Ew,
                    int(e_layer_min), stride, padding, dilation,
                    bs, int(bits), sat_mode,
                    bias_fp=bias_fp,
                    mant_bias=int(mant_bias),
                    return_acc=want_acc,
                    **win_kw,
                )
                out_flat, sat_flat = res[0], res[1]
                acc_flat = res[2] if want_acc else None

        B, _, H, W = qi_i8.shape          # dilated size when in_dil > 1
        O, _, kH, kW = qw_fp.shape
        sH, sW = (stride if isinstance(stride, tuple) else (stride, stride))
        pH, pW = (padding if isinstance(padding, tuple) else (padding, padding))
        dH, dW = (dilation if isinstance(dilation, tuple) else (dilation, dilation))
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        assert H_out * W_out == out_flat.shape[-1]

        # Clone so the return is not a view of `out_flat`. Downstream inplace ops
        # (e.g. F.relu_ on the next layer's input) would otherwise trigger
        # PyTorch's "view+inplace inside custom Function" guard.
        out = out_flat.view(B, O, H_out, W_out).to(qi_fp.dtype).clone()
        sat = sat_flat.view(B, O, H_out, W_out)
        # Bias is now added inside the kernel (HW-faithful: into the int
        # accumulator on the common e_layer_min grid). Do NOT add post-de-shift.

        if stats_sink is not None:
            stats_sink["sat_count"] = int(sat.sum().item())
            stats_sink["total"] = int(sat.numel())
            stats_sink["backend"] = "triton" if use_triton else "python"
            if want_acc:
                stats_sink["sat_map"] = sat.detach().clone()
                stats_sink["acc"] = (None if acc_flat is None else
                                     acc_flat.view(B, O, H_out, W_out).clone())
            if e_qa_out is not None:
                win_sat = win_sink["win_sat"].view(B, O, H_out, W_out)
                stats_sink["win_sat_count"] = int(win_sat.sum().item())
                if want_acc:
                    # The python ref hands the word over; triton keeps only the
                    # fp32 output, which holds it exactly for win_bits <= 24.
                    win = win_sink.get("win")
                    if win is None:
                        win = (out_flat.double() * 2.0 ** -int(e_qa_out)) \
                            .round().to(torch.int64)
                    stats_sink["win"] = win.view(B, O, H_out, W_out).clone()
                    stats_sink["win_sat_map"] = win_sat.clone()

        ctx.save_for_backward(qi_fp, qw_fp, sat)
        ctx.has_bias = bias_fp is not None
        ctx.stride = (sH, sW)
        ctx.padding = (pH, pW)
        ctx.dilation = (dH, dW)
        ctx.in_dil, ctx.in_extra = _pair(in_dil), _pair(in_extra)
        ctx.ste_mask = bool(ste_mask)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        qi_fp, qw_fp, sat = ctx.saved_tensors
        if ctx.ste_mask and sat.any():
            grad_out = grad_out * (~sat).to(grad_out.dtype)

        # STE on the conv the forward ran: over the dilated input, then back
        # to the real positions (no-op without in_dil / in_extra).
        qi_d = dilate_input(qi_fp, ctx.in_dil, ctx.in_extra)
        grad_qi = torch.nn.grad.conv2d_input(
            qi_d.shape, qw_fp, grad_out,
            stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=1,
        )
        grad_qi = _undilate(grad_qi, ctx.in_dil, *qi_fp.shape[-2:])
        grad_qw = torch.nn.grad.conv2d_weight(
            qi_d, qw_fp.shape, grad_out,
            stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=1,
        )
        grad_bias = grad_out.sum(dim=(0, 2, 3)) if ctx.has_bias else None

        # forward took 23 args: qi_fp, qw_fp, bias_fp, e_layer_min, bs, bits,
        # sat_mode, ste_mask, stride, padding, dilation, backend, stats_sink,
        # fmt, act_blockify, weight_blockify, npe_triton_variant,
        # e_qa_out, win_bits, out_round, in_dil, in_extra, pla
        return (
            grad_qi, grad_qw, grad_bias,
            None, None, None, None, None,
            None, None, None, None, None, None,
            None, None, None,
            None, None, None,
            None, None, None,
        )


def hw_fxp_conv2d(
    qi_fp, qw_fp, bias_fp,
    e_layer_min, bs, bits=35, sat_mode="per_product", ste_mask=False,
    stride=1, padding=0, dilation=1, backend="python",
    stats_sink=None,
    fmt='int8',
    act_blockify='channel', weight_blockify='channel',
    npe_triton_variant='reuse',
    e_qa_out=None, win_bits=None, out_round='floor',
    out_exp_bits=OUT_EXP_BITS,
    in_dil=1, in_extra=0,
    pla=None,
):
    """Public entry point; see `HWFxpConv2dFn.forward` for semantics.

    `stats_sink` is an optional mutable dict; kernel fills in 'sat_count',
    'total', and 'backend' after each forward for logging.
    `fmt` is the MXINT element format ('int8', 'int10', 'int12', 'int16', ...).
    `act_blockify`/`weight_blockify` select NPE blockify: defaults
    'channel'/'channel' (block both along Cin); NPE = 'xblock'/'flatten'
    (activation along W, weight per-filter flattened). `npe_triton_variant`
    picks the NPE+triton kernel ('reuse' existing kernel at bs=1, or 'opt'
    dedicated per-lane kernel); both are bit-identical.
    `e_qa_out` (None = off) applies the static output window: the output is
    O * 2^e_qa_out, O = acc >> (e_qa_out - e_layer_min) saturated to
    +-(2^win_bits - 1), dropped bits rounded per `out_round`. Forward only;
    the backward is the same STE either way. `win_bits=None` derives it from
    `fmt` and `out_exp_bits` (`derive_out_window_bits`).
    `in_dil` / `in_extra` zero-insert the extracted input (`dilate_input`):
    `stride`/`padding`/`dilation` then describe the stride-1 conv over it.
    `pla` (`PLAParams`, None = off) runs the PCU multiply before the window
    (forward only, needs `e_qa_out`); the caller applies the float activation
    to the gradient path itself.
    """
    if pla is not None and e_qa_out is None:
        raise ValueError("hw_fxp_conv2d: pla needs the output window (e_qa_out)")
    if pla is not None and pla.bias == "bins":
        O, nb = qw_fp.shape[0], pla.scales.shape[-1]
        if (pla.scales.shape != (O, nb) or pla.biases.shape != (O, nb)
                or pla.bounds.shape != (O, nb - 1) or pla.tie not in ("upper", "lower")):
            raise ValueError(
                f"hw_fxp_conv2d: PLA bins need scales/biases [{O}, n], bounds "
                f"[{O}, n - 1] and tie 'upper'|'lower', got "
                f"{tuple(pla.scales.shape)}/{tuple(pla.biases.shape)}/"
                f"{tuple(pla.bounds.shape)}, {pla.tie!r}")
    if e_qa_out is not None and win_bits is None:
        win_bits = derive_out_window_bits(fmt, out_exp_bits)
    if e_layer_min is None:
        raise RuntimeError(
            "hw_fxp_conv2d requires a static e_layer_min. Run "
            "`calibrate_e_layer_min(model, loader)` before inference."
        )
    return HWFxpConv2dFn.apply(
        qi_fp, qw_fp, bias_fp,
        int(e_layer_min), int(bs), int(bits), str(sat_mode), bool(ste_mask),
        stride, padding, dilation, str(backend),
        stats_sink,
        str(fmt),
        str(act_blockify), str(weight_blockify),
        str(npe_triton_variant),
        None if e_qa_out is None else int(e_qa_out),
        None if win_bits is None else int(win_bits), str(out_round),
        _pair(in_dil), _pair(in_extra),
        pla,
    )


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------

def _min_grid_exps(qi_fp, qw_fp, bs, fmt='int8',
                   act_blockify='channel', weight_blockify='channel'):
    """Return (min(Ea) - mant_bias, min(Ew) - mant_bias): the activation and
    weight grids E_qa_in, E_qw of the guide."""
    _, mant_bias, _, _ = _int_format_params(fmt)
    if act_blockify == "xblock":
        _, Ea = extract_mxint_xblock(qi_fp, bs, fmt=fmt)
    else:
        _, Ea = extract_mxint(qi_fp, bs, axis=1, fmt=fmt)
    if weight_blockify == "flatten":
        _, Ew = extract_mxint_flatten(qw_fp, bs, fmt=fmt)
    else:
        _, Ew = extract_mxint(qw_fp, bs, axis=1, fmt=fmt)
    return (int(Ea.amin().item()) - int(mant_bias),
            int(Ew.amin().item()) - int(mant_bias))


def _compute_min_shift_exp(qi_fp, qw_fp, bs, fmt='int8',
                           act_blockify='channel', weight_blockify='channel'):
    """Return layer-wide min(Ew + Ea - 2*mantissa_bias) over all block pairs.

    Ea and Ew are independent (no shared index), so the joint min is simply
    min(Ea) + min(Ew) — holds for both channel and NPE blockings.
    """
    e_a, e_w = _min_grid_exps(qi_fp, qw_fp, bs, fmt=fmt,
                              act_blockify=act_blockify,
                              weight_blockify=weight_blockify)
    return e_a + e_w


def _running(old, new, fn):
    return new if old is None else fn(old, new)


class _CalibrationState:
    """Per-layer collector: running min(Ew + Ea - 2*mant_bias) (accumulator
    grid), plus the input/output |max| and the separate input/weight grids
    for the output window and the window-fill diagnostic."""

    def __init__(self):
        self.running_min = None
        self.min_e_qa_in = None
        self.min_e_qw = None
        self.in_absmax = None
        self.w_absmax = None
        self.out_absmax = None
        self.calls_now = 0      # calls in the current forward
        self.max_calls = 0      # most calls seen in one forward

    def update(self, qi_fp, qw_fp, bs, fmt='int8',
               act_blockify='channel', weight_blockify='channel'):
        e_a, e_w = _min_grid_exps(
            qi_fp, qw_fp, bs, fmt=fmt,
            act_blockify=act_blockify, weight_blockify=weight_blockify,
        )
        self.calls_now += 1
        self.running_min = _running(self.running_min, e_a + e_w, min)
        self.min_e_qa_in = _running(self.min_e_qa_in, e_a, min)
        self.min_e_qw = _running(self.min_e_qw, e_w, min)
        self.in_absmax = _running(self.in_absmax,
                                  float(qi_fp.detach().abs().amax().item()), max)
        self.w_absmax = _running(self.w_absmax,
                                 float(qw_fp.detach().abs().amax().item()), max)

    def update_out(self, out):
        self.out_absmax = _running(self.out_absmax,
                                   float(out.detach().abs().amax().item()), max)


def report_hw_stats(model, reset=False):
    """Print and return aggregated saturation stats per MXConv2dHW layer.

    Useful as a per-epoch summary when running with verbose=0 or 1. With
    `reset=True`, lifetime counters are zeroed after reporting.
    """
    from mx_layers_blocked import MXConv2dHW
    rows = []
    win_sat = {}
    exp_cut = {}
    for m in model.modules():
        if not isinstance(m, MXConv2dHW):
            continue
        name = m._mx_layer_name or f"conv-{id(m):x}"
        sat = int(m._sat_seen_life)
        tot = int(m._sat_total_life)
        fwd = int(m._fwd_count)
        pct = (100.0 * sat / tot) if tot else 0.0
        rows.append((name, fwd, sat, tot, pct))
        win_sat[name] = int(getattr(m, "_win_sat_seen_life", 0))
        cut = getattr(m, "_in_exp_cut_life", [0, 0])
        exp_cut[name] = (int(cut[0]), int(cut[1]))
        if reset:
            m._in_exp_cut_life = [0, 0]
            m._win_sat_seen_life = 0
            m._sat_seen_life = 0
            m._sat_total_life = 0
            m._sat_seen_window = 0
            m._sat_total_window = 0
            m._fwd_count = 0

    rows.sort(key=lambda r: -r[4])   # worst first
    print(f"[MXConv2dHW] saturation report ({len(rows)} layers):")
    for name, fwd, sat, tot, pct in rows[:20]:
        a_cut, w_cut = exp_cut[name]
        print(f"  {pct:6.3f}%  sat={sat}/{tot}  win_sat={win_sat[name]}  "
              f"exp_cut a={a_cut} w={w_cut}  fwd={fwd}  {name}")
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more layers (sorted by % desc)")
    return rows


def _feed(model, batch, forward_fn):
    """Same dispatch as collect_stats / MXQuantizer: tuple -> m(b[0]),
    dict -> m(**b), tensor -> m(b)."""
    if forward_fn is not None:
        forward_fn(model, batch)
    elif isinstance(batch, (tuple, list)):
        model(batch[0])
    elif isinstance(batch, dict):
        model(**batch)
    else:
        model(batch)


def _sweep(model, layers, data_iter, num_batches, forward_fn):
    """Run the calibration forward on `layers`; return {id(m): state}. The
    layers take their FP32 short-circuit while a state is attached."""
    states = {id(m): _CalibrationState() for m in layers}
    for m in layers:
        m._calibration_state = states[id(m)]
    model.eval()
    try:
        with torch.no_grad():
            for i, batch in enumerate(data_iter):
                if i >= num_batches:
                    break
                for st in states.values():
                    st.calls_now = 0
                _feed(model, batch, forward_fn)
                for st in states.values():
                    st.max_calls = max(st.max_calls, st.calls_now)
    finally:
        for m in layers:
            if hasattr(m, "_calibration_state"):
                delattr(m, "_calibration_state")
    return states


def _set_grid(m, key, value, src):
    """Set one grid on an MXConv2dHW and record where it came from."""
    setattr(m, key, int(value))
    if getattr(m, "_grid_src", None) is None:
        m._grid_src = {}
    m._grid_src[key] = src


def _grid_src(m, key):
    return (getattr(m, "_grid_src", None) or {}).get(key)


def in_exp_on(cfg):
    """True when the a/w shared-exponent register is modelled at all."""
    return cfg.get("in_exp_mode", "off") != "off" and cfg.get("in_exp_bits") is not None


def _e_layer_min_floor(st, cfg, fmt, e_base_a, e_base_w):
    """Lowest accumulator grid the layer may use, or None for no floor.

    - in_exp: no block is below its base, so no product is below
      (e_base_a - bias) + (e_base_w - bias).
    - acc_headroom_bits h: |product| < 2^(Ea_max + Ew_max + 2) and
      |out| < 2^(floor(log2 max|out|) + 1) must fit a signed `bits` word
      with h spare bits: e >= top - (bits - 1) + h.
    """
    _, mant_bias, _, _ = _int_format_params(fmt)
    floors = []
    if in_exp_on(cfg) and e_base_a is not None and e_base_w is not None:
        floors.append(int(e_base_a) + int(e_base_w) - 2 * mant_bias)
    h = cfg.get("acc_headroom_bits")
    ea, ew = floor_log2(st.in_absmax or 0), floor_log2(st.w_absmax or 0)
    if h is not None and ea is not None and ew is not None:
        top = ea + ew + 2
        eo = floor_log2(st.out_absmax or 0)
        if eo is not None:
            top = max(top, eo + 1)
        floors.append(top - (int(cfg.get("bits", 48)) - 1) + int(h))
    return max(floors) if floors else None


def calibrate_e_layer_min(model, data_iter, num_batches=8, forward_fn=None,
                          force=False):
    """Populate `e_layer_min` on every `MXConv2dHW` layer by sweeping inputs.

    Parameters
    ----------
    model       : nn.Module holding one or more `MXConv2dHW` instances.
    data_iter   : iterable yielding input tensors or tuples; first element is fed
                  to `forward_fn(model, batch)`.
    num_batches : number of batches to sample.
    forward_fn  : callable `(model, batch) -> any`. Defaults to `model(batch)`.
    force       : recalibrate layers whose e_layer_min is already set (e.g.
                  pinned by `xblock_accum.e_layer_min` in config).
    """
    from mx_layers_blocked import MXConv2dHW  # local import to avoid cycle

    all_hw = [m for m in model.modules() if isinstance(m, MXConv2dHW)]
    if not all_hw:
        return {}

    # Generic-call friendly: skip layers whose e_layer_min is already pinned.
    layers = all_hw if force else [m for m in all_hw if m.e_layer_min is None]
    skipped = len(all_hw) - len(layers)
    if skipped:
        print(f"[calibrate_e_layer_min] {skipped}/{len(all_hw)} layers already "
              f"have e_layer_min set; skipping them (force=True recalibrates).")
    if not layers:
        return {}

    states = _sweep(model, layers, data_iter, num_batches, forward_fn)
    result = {}
    for m in layers:
        st = states[id(m)]
        if st.running_min is not None:
            _set_grid(m, "e_layer_min", st.running_min, "calib")
            result[m] = m.e_layer_min
    return result


def calibrate_hw_grids(model, data_iter, num_batches=8, forward_fn=None,
                       force=False, margin_bits=None, verbose=1):
    """Freeze both static grids of every `MXConv2dHW` layer in one sweep.

      e_layer_min = min(Ea) + min(Ew) - 2*mant_bias      accumulator grid E_acc
      e_qa_out    = floor(log2(max|out|)) - (win_bits-1) + margin_bits

    win_bits per layer from `resolve_out_window_bits` (int8 + 4b output exp
    -> 22, so e_qa_out = floor(log2(max|out|)) - 21).

    The output window then extracts O = acc >> (e_qa_out - e_layer_min). It
    always uses the calibrated e_layer_min, never the ideal E_qa_in + E_qw,
    so the two grids stay consistent even when a tensor does not fill its
    window. Turning the window on is a config choice
    (`xblock_accum.out_window`); calibrating does not.

    A config value always wins: calibration fills only grids that are still
    None (the config's "hw_grids" or xblock_accum did not set them). With
    `force=True` both grids are recalibrated - an offline step, before
    `MXQuantizer.save_hw_grids` writes them out. Inference reads the config
    and never calls this.

    Same call shape as `collect_stats(model, data, forward_fn=...)`, so a
    trainer that cannot be modified wires it through its forward_fn. The
    model's train/eval mode is restored afterwards.

    Parameters
    ----------
    margin_bits : headroom on e_qa_out. None = each layer's
                  `xblock_accum.out_margin_bits` (default 0).
    verbose     : 1 prints one line per layer: both grids, shift_out, and the
                  window-fill check (input grid from min block exp vs from max).

    Returns
    -------
    dict: layer name -> {e_layer_min, e_qa_out, shift_out, e_qa_in, e_qw,
                         in_fill_gap, win_bits, src} for every MXConv2dHW;
                         src maps each grid to 'config' or 'calib'. Layers
                         the config fully sets are not swept and have no
                         e_qa_in / e_qw / in_fill_gap (None).
    """
    from mx_layers_blocked import MXConv2dHW, _get_xblock_cfg

    all_hw = [m for m in model.modules() if isinstance(m, MXConv2dHW)]
    layers = all_hw if force else [
        m for m in all_hw
        if m.e_layer_min is None or getattr(m, "e_qa_out", None) is None
        or (in_exp_on(_get_xblock_cfg(m)) and m.e_base_a is None)]
    states = {}
    if layers:
        was_training = model.training
        try:
            states = _sweep(model, layers, data_iter, num_batches, forward_fn)
        finally:
            if was_training:
                model.train()

    result = {}
    for m in all_hw:
        cfg = _get_xblock_cfg(m)
        name = m._mx_layer_name or f"conv-{id(m):x}"
        st = states.get(id(m))
        if st is None or st.running_min is None:
            # Fully set by config, or never called in the sweep.
            if m.e_layer_min is not None and m.e_qa_out is not None:
                result[name] = dict(
                    e_layer_min=int(m.e_layer_min), e_qa_out=int(m.e_qa_out),
                    shift_out=int(m.e_qa_out) - int(m.e_layer_min),
                    e_qa_in=None, e_qw=None, in_fill_gap=None,
                    win_bits=resolve_out_window_bits(cfg, m.mx_specs['a_elem_format']),
                    src={k: _grid_src(m, k) for k in ("e_layer_min", "e_qa_out")},
                )
            continue
        # A layer called k times per forward (shared module) gets one grid
        # pooled over all k calls, and the guard allows k window calls.
        m.hw_calls = max(1, st.max_calls)
        win_bits = resolve_out_window_bits(cfg, m.mx_specs['a_elem_format'])
        margin = (int(cfg.get("out_margin_bits", 0)) if margin_bits is None
                  else int(margin_bits))
        fmt = m.mx_specs['a_elem_format']
        _, mant_bias, _, _ = _int_format_params(fmt)
        span = None
        if in_exp_on(cfg):
            span = (1 << int(cfg["in_exp_bits"])) - 1
            if force or m.e_base_a is None:
                _set_grid(m, "e_base_a", floor_log2(st.in_absmax) - span, "calib")
        e_base_w = m.e_base_w
        if span is not None and e_base_w is None:
            e_base_w = floor_log2(st.w_absmax) - span   # live in forward
        if force or m.e_layer_min is None:
            floor = _e_layer_min_floor(st, cfg, fmt, m.e_base_a if span is not None
                                       else None, e_base_w)
            e_min = st.running_min if floor is None else max(st.running_min, floor)
            _set_grid(m, "e_layer_min", e_min, "calib")
        if force or m.e_qa_out is None:
            e_out = e_qa_out_from_absmax(st.out_absmax or 0.0, win_bits, margin)
            _set_grid(m, "e_qa_out", m.e_layer_min if e_out is None else e_out,
                      "calib")
        # Window-fill check: E_q from the max equals min block exp - bias
        # only when the input fills its window. A
        # positive gap means the input spans more than win_bits bits.
        e_in_max = e_qa_out_from_absmax(st.in_absmax or 0.0, win_bits)
        gap = None if e_in_max is None else e_in_max - st.min_e_qa_in
        # Exponent codes each operand spans (a 4b register holds 16).
        ea_max, ew_max = floor_log2(st.in_absmax), floor_log2(st.w_absmax)
        a_codes = None if ea_max is None else ea_max - (st.min_e_qa_in + mant_bias) + 1
        w_codes = None if ew_max is None else ew_max - (st.min_e_qw + mant_bias) + 1
        result[name] = dict(
            e_layer_min=int(m.e_layer_min), e_qa_out=m.e_qa_out,
            shift_out=m.e_qa_out - int(m.e_layer_min),
            e_qa_in=st.min_e_qa_in, e_qw=st.min_e_qw, in_fill_gap=gap,
            win_bits=win_bits, a_codes=a_codes, w_codes=w_codes, calls=m.hw_calls,
            e_base_a=m.e_base_a, e_base_w=e_base_w if span is not None else None,
            src={k: _grid_src(m, k) for k in ("e_layer_min", "e_qa_out")},
        )
    if verbose:
        for name, r in result.items():
            src = r["src"]
            print(f"[calibrate_hw_grids] {name}: "
                  f"e_layer_min={r['e_layer_min']}[{src['e_layer_min']}] "
                  f"(E_qa_in={r['e_qa_in']} + E_qw={r['e_qw']}) "
                  f"e_qa_out={r['e_qa_out']}[{src['e_qa_out']}] "
                  f"shift_out={r['shift_out']} "
                  f"win_bits={r['win_bits']} "
                  f"in_fill_gap={r['in_fill_gap']} "
                  f"codes a={r.get('a_codes')} w={r.get('w_codes')}"
                  + (f" calls={r['calls']} (shared grid)"
                     if r.get('calls', 1) > 1 else "")
                  + (f" e_base_a={r['e_base_a']} e_base_w={r['e_base_w']}"
                     if r.get('e_base_a') is not None else ""))
    return result


def export_hw_grids(model):
    """{layer name: {e_layer_min, e_qa_out, win_bits, shift_out}} for every
    `MXConv2dHW` in `model` - the config's "hw_grids" section (see
    `fixed_point.hw_grids`), plus pla_frac / pla_m_pos / pla_m_neg (and the
    pla_bin_* lists with pla_bias 'bins') for a
    layer with PLA folded. Raises if a layer has no grid yet."""
    from mx_layers_blocked import MXConv2dHW, _get_xblock_cfg
    from .hw_grids import check_hw_grid

    grids, missing = {}, []
    for m in model.modules():
        if not isinstance(m, MXConv2dHW):
            continue
        name = m._mx_layer_name
        if name is None:
            raise ValueError("export_hw_grids: an MXConv2dHW has no layer name; "
                             "build the model through MXQuantizer.quant")
        if m.e_layer_min is None or m.e_qa_out is None:
            missing.append(name)
            continue
        win_bits = resolve_out_window_bits(_get_xblock_cfg(m),
                                           m.mx_specs['a_elem_format'])
        entry = dict(
            e_layer_min=int(m.e_layer_min), e_qa_out=int(m.e_qa_out),
            win_bits=win_bits, shift_out=int(m.e_qa_out) - int(m.e_layer_min),
        )
        if getattr(m, "hw_calls", 1) > 1:
            entry["calls"] = int(m.hw_calls)
        if in_exp_on(_get_xblock_cfg(m)):
            if m.e_base_a is None:
                missing.append(name)
                continue
            entry["e_base_a"] = int(m.e_base_a)
            entry["e_base_w"] = int(m.weight_exp_base())
        if m.pla_on():
            pla = m.pla_params()
            entry.update(pla_frac=int(pla.frac),
                         pla_m_pos=[int(v) for v in pla.m_pos.tolist()],
                         pla_m_neg=[int(v) for v in pla.m_neg.tolist()])
            if pla.bias == "bins":
                entry.update(pla_bin_bounds=pla.bounds.tolist(),
                             pla_bin_scales=pla.scales.tolist(),
                             pla_bin_biases=pla.biases.tolist())
        grids[name] = check_hw_grid(name, entry)
    if missing:
        raise ValueError(f"export_hw_grids: {len(missing)} layers have no grids "
                         f"(run calibrate_hw_grids first): {missing[:5]}")
    return grids


def calibrate_act_scales(model, data_iter, num_batches=8, forward_fn=None,
                         verbose=1):
    """Freeze the static scales of every `calibrate: true` fxp act operand.

    MX needs no calibration - its shared exponent comes from each block's max
    at run time. A static fixed-point operand is the opposite: the scale is
    fixed before inference, which is what makes it able to clip, and what makes
    it need data. This is the post-training pass that picks it.

    Per `MXActQuant` instance, per operand, per call site, the observed max|x|
    over the sampled batches sets `step = max|x| / code_max`, so nothing seen
    here clips. Call sites are kept apart on purpose: one shared module called
    at several pyramid levels sees tensors of quite different magnitude, and a
    single scale is then wrong at most of them.

    Parameters
    ----------
    model       : nn.Module holding one or more `MXActQuant` instances.
    data_iter   : iterable yielding input tensors or tuples; first element is
                  fed to `forward_fn(model, batch)`.
    num_batches : number of batches to sample.
    forward_fn  : callable `(model, batch) -> any`. Defaults to `model(batch)`.
    verbose     : 1 prints the frozen scales, 0 is silent.

    Returns
    -------
    dict: layer name -> {"in<i>@<site>" | "out@<site>": scale}
    """
    from mx_layers_act import MXActQuant  # local import to avoid a cycle

    pending = [m for m in model.modules()
               if isinstance(m, MXActQuant) and m.needs_calibration()]
    if not pending:
        return {}

    for m in pending:
        m.start_calibration()
    # HW convs that cannot run yet for lack of a grid (calibrate_hw_grids
    # runs after this, on the quantized act operands) take their FP32
    # calibration path meanwhile. The sweep is in eval mode.
    from mx_layers_blocked import MXConv2dHW, _get_xblock_cfg

    def _needs_grid(m):
        cfg = _get_xblock_cfg(m)
        e_min = m.e_layer_min if m.e_layer_min is not None else cfg.get("e_layer_min")
        win = cfg.get("out_window", "off") in ("infer", "always")
        return (e_min is None or (win and m.e_qa_out is None)
                or (in_exp_on(cfg) and m.e_base_a is None))

    hw_open = [m for m in model.modules() if isinstance(m, MXConv2dHW)
               and getattr(m, "_calibration_state", None) is None
               and _needs_grid(m)]
    for m in hw_open:
        m._calibration_state = _CalibrationState()

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for i, batch in enumerate(data_iter):
                if i >= num_batches:
                    break
                for m in pending:
                    # Call sites are identified by call order, so the counter
                    # has to restart with every forward pass.
                    m.reset_calls()
                if forward_fn is not None:
                    forward_fn(model, batch)
                elif isinstance(batch, (tuple, list)):
                    model(batch[0])
                else:
                    model(batch)
                # `call_sites` is a claim about the model, and getting it wrong
                # silently mixes two levels' tensors into one scale. Check it
                # against reality instead, on the first pass.
                for m in pending:
                    if m.n_calls != m.call_sites:
                        name = getattr(m, "_mx_layer_name",
                                       type(m.inner).__name__)
                        raise ValueError(
                            f"calibrate_act_scales: '{name}' is configured "
                            f"call_sites={m.call_sites} but was called "
                            f"{m.n_calls} times in one forward pass. Set "
                            f"call_sites={m.n_calls} in the config.")
        result = {}
        for m in pending:
            name = getattr(m, "_mx_layer_name", type(m.inner).__name__)
            result[name] = m.finish_calibration()
    finally:
        # On the error path nothing is frozen: a scale from a half-finished
        # sweep is worse than none, since the "run calibration first" error is
        # at least visible.
        for m in pending:
            m.calibrating = False
            m.reset_calls()
        for m in hw_open:
            delattr(m, "_calibration_state")
        if was_training:
            model.train()

    if verbose:
        for name, scales in result.items():
            if not scales:
                print(f"[calibrate_act_scales] {name}: nothing observed - the "
                      f"layer was never called")
                continue
            pretty = ", ".join(f"{k}={v:.6g}" for k, v in scales.items())
            print(f"[calibrate_act_scales] {name}: {pretty}")
    return result
