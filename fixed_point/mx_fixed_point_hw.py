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

Only MXINT formats are supported. FP8/MXFP paths are rejected.
"""

import torch
import torch.nn.functional as F

MANTISSA_BIAS = 6  # legacy: MXINT8 mantissa bias. New code: use _int_format_params.


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
    """
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

    if bias_fp is not None:
        # Project bias onto acc grid: bias_int = round(b * 2^(-e_layer_min)).
        scale_to_acc = float(2.0 ** (-e_min))
        bias_int = (bias_fp.detach().to(torch.float64) * scale_to_acc) \
            .round().to(torch.int64).to(device).view(1, O, 1)
        acc = acc + bias_int
        acc, sat_step = _sat(acc, lo, hi)
        sat = sat | sat_step

    out = acc.to(torch.float32) * float(2.0 ** e_min)
    return out, sat


def _hw_fxp_conv2d_ref_npe(
    qi_i8, qw_i8, Ea_full, Ew_full,
    e_layer_min, stride, padding, dilation,
    bits, sat_mode,
    bias_fp=None, mant_bias=MANTISSA_BIAS,
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
        bias_fp=bias_fp, mant_bias=mant_bias,
    )


class HWFxpConv2dFn(torch.autograd.Function):
    """HW-faithful fixed-point Conv2d forward + STE backward.

    Forward consumes FP32 operands already on the MXINT8 lattice. Internally
    extracts int8 + shared exponent, runs the kernel, returns a 4D output.

    Backward approximates the forward as a standard `F.conv2d` (STE) so
    gradients propagate through quantised operands. If `ste_mask` is set,
    grad for saturated output lanes is zeroed.
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
    ):
        _, mant_bias, _, _ = _int_format_params(fmt)
        npe = (act_blockify == "xblock" and weight_blockify == "flatten")

        if npe:
            # NPE: independent per-element exponents (act X-block, weight
            # flatten-block). Torch ref only in Phase 1 (triton is Phase 2).
            with torch.no_grad():
                qi_i8, Ea = extract_mxint_xblock(qi_fp, bs, fmt=fmt)
                qw_i8, Ew = extract_mxint_flatten(qw_fp, bs, fmt=fmt)
            use_triton = False
            out_flat, sat_flat = _hw_fxp_conv2d_ref_npe(
                qi_i8, qw_i8, Ea, Ew,
                int(e_layer_min), stride, padding, dilation,
                int(bits), sat_mode,
                bias_fp=bias_fp,
                mant_bias=int(mant_bias),
            )
        else:
            with torch.no_grad():
                qi_i8, Ea = extract_mxint(qi_fp, bs, axis=1, fmt=fmt)
                qw_i8, Ew = extract_mxint(qw_fp, bs, axis=1, fmt=fmt)

            use_triton = (backend == "triton" and qi_fp.is_cuda)
            if use_triton:
                from .mx_fixed_point_hw_triton import _hw_fxp_conv2d_triton
                out_flat, sat_flat = _hw_fxp_conv2d_triton(
                    qi_i8, qw_i8, Ea, Ew,
                    int(e_layer_min), stride, padding, dilation,
                    bs, int(bits), sat_mode,
                    bias_fp=bias_fp,
                    mant_bias=int(mant_bias),
                )
            else:
                out_flat, sat_flat = _hw_fxp_conv2d_ref(
                    qi_i8, qw_i8, Ea, Ew,
                    int(e_layer_min), stride, padding, dilation,
                    bs, int(bits), sat_mode,
                    bias_fp=bias_fp,
                    mant_bias=int(mant_bias),
                )

        B, _, H, W = qi_fp.shape
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

        ctx.save_for_backward(qi_fp, qw_fp, sat)
        ctx.has_bias = bias_fp is not None
        ctx.stride = (sH, sW)
        ctx.padding = (pH, pW)
        ctx.dilation = (dH, dW)
        ctx.ste_mask = bool(ste_mask)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        qi_fp, qw_fp, sat = ctx.saved_tensors
        if ctx.ste_mask and sat.any():
            grad_out = grad_out * (~sat).to(grad_out.dtype)

        grad_qi = torch.nn.grad.conv2d_input(
            qi_fp.shape, qw_fp, grad_out,
            stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=1,
        )
        grad_qw = torch.nn.grad.conv2d_weight(
            qi_fp, qw_fp.shape, grad_out,
            stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=1,
        )
        grad_bias = grad_out.sum(dim=(0, 2, 3)) if ctx.has_bias else None

        # forward took 16 args: qi_fp, qw_fp, bias_fp, e_layer_min, bs, bits,
        # sat_mode, ste_mask, stride, padding, dilation, backend, stats_sink,
        # fmt, act_blockify, weight_blockify
        return (
            grad_qi, grad_qw, grad_bias,
            None, None, None, None, None,
            None, None, None, None, None, None,
            None, None,
        )


def hw_fxp_conv2d(
    qi_fp, qw_fp, bias_fp,
    e_layer_min, bs, bits=35, sat_mode="per_product", ste_mask=False,
    stride=1, padding=0, dilation=1, backend="python",
    stats_sink=None,
    fmt='int8',
    act_blockify='channel', weight_blockify='channel',
):
    """Public entry point; see `HWFxpConv2dFn.forward` for semantics.

    `stats_sink` is an optional mutable dict; kernel fills in 'sat_count',
    'total', and 'backend' after each forward for logging.
    `fmt` is the MXINT element format ('int8', 'int10', 'int12', 'int16', ...).
    `act_blockify`/`weight_blockify` select NPE blockify: defaults
    'channel'/'channel' (block both along Cin); NPE = 'xblock'/'flatten'
    (activation along W, weight per-filter flattened). NPE uses the torch ref
    only in Phase 1 (triton is Phase 2).
    """
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
    )


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------

def _compute_min_shift_exp(qi_fp, qw_fp, bs, fmt='int8',
                           act_blockify='channel', weight_blockify='channel'):
    """Return layer-wide min(Ew + Ea - 2*mantissa_bias) over all block pairs.

    Ea and Ew are independent (no shared index), so the joint min is simply
    min(Ea) + min(Ew) — holds for both channel and NPE blockings.
    """
    _, mant_bias, _, _ = _int_format_params(fmt)
    if act_blockify == "xblock":
        _, Ea = extract_mxint_xblock(qi_fp, bs, fmt=fmt)
    else:
        _, Ea = extract_mxint(qi_fp, bs, axis=1, fmt=fmt)
    if weight_blockify == "flatten":
        _, Ew = extract_mxint_flatten(qw_fp, bs, fmt=fmt)
    else:
        _, Ew = extract_mxint(qw_fp, bs, axis=1, fmt=fmt)
    return int(Ea.amin().item()) + int(Ew.amin().item()) - 2 * int(mant_bias)


class _CalibrationState:
    """Per-layer running min(Ew + Ea - 2*mant_bias) collector."""

    def __init__(self):
        self.running_min = None

    def update(self, qi_fp, qw_fp, bs, fmt='int8',
               act_blockify='channel', weight_blockify='channel'):
        cand = _compute_min_shift_exp(
            qi_fp, qw_fp, bs, fmt=fmt,
            act_blockify=act_blockify, weight_blockify=weight_blockify,
        )
        self.running_min = cand if self.running_min is None else min(self.running_min, cand)


def report_hw_stats(model, reset=False):
    """Print and return aggregated saturation stats per MXConv2dHW layer.

    Useful as a per-epoch summary when running with verbose=0 or 1. With
    `reset=True`, lifetime counters are zeroed after reporting.
    """
    from mx_layers_blocked import MXConv2dHW
    rows = []
    for m in model.modules():
        if not isinstance(m, MXConv2dHW):
            continue
        name = m._mx_layer_name or f"conv-{id(m):x}"
        sat = int(m._sat_seen_life)
        tot = int(m._sat_total_life)
        fwd = int(m._fwd_count)
        pct = (100.0 * sat / tot) if tot else 0.0
        rows.append((name, fwd, sat, tot, pct))
        if reset:
            m._sat_seen_life = 0
            m._sat_total_life = 0
            m._sat_seen_window = 0
            m._sat_total_window = 0
            m._fwd_count = 0

    rows.sort(key=lambda r: -r[4])   # worst first
    print(f"[MXConv2dHW] saturation report ({len(rows)} layers):")
    for name, fwd, sat, tot, pct in rows[:20]:
        print(f"  {pct:6.3f}%  sat={sat}/{tot}  fwd={fwd}  {name}")
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more layers (sorted by % desc)")
    return rows


def calibrate_e_layer_min(model, data_iter, num_batches=8, forward_fn=None):
    """Populate `e_layer_min` on every `MXConv2dHW` layer by sweeping inputs.

    Parameters
    ----------
    model       : nn.Module holding one or more `MXConv2dHW` instances.
    data_iter   : iterable yielding input tensors or tuples; first element is fed
                  to `forward_fn(model, batch)`.
    num_batches : number of batches to sample.
    forward_fn  : callable `(model, batch) -> any`. Defaults to `model(batch)`.
    """
    from mx_layers_blocked import MXConv2dHW  # local import to avoid cycle

    all_hw = [m for m in model.modules() if isinstance(m, MXConv2dHW)]
    if not all_hw:
        return {}

    # Generic-call friendly: skip layers whose e_layer_min is already pinned.
    layers = [m for m in all_hw if m.e_layer_min is None]
    skipped = len(all_hw) - len(layers)
    if skipped:
        print(f"[calibrate_e_layer_min] {skipped}/{len(all_hw)} layers already "
              f"have e_layer_min set; skipping them.")
    if not layers:
        return {}

    states = {id(m): _CalibrationState() for m in layers}
    for m in layers:
        m._calibration_state = states[id(m)]

    model.eval()
    try:
        with torch.no_grad():
            for i, batch in enumerate(data_iter):
                if i >= num_batches:
                    break
                if forward_fn is not None:
                    forward_fn(model, batch)
                else:
                    if isinstance(batch, (tuple, list)):
                        model(batch[0])
                    else:
                        model(batch)
    finally:
        result = {}
        for m in layers:
            st = getattr(m, "_calibration_state", None)
            if st is not None and st.running_min is not None:
                m.e_layer_min = int(st.running_min)
                result[m] = m.e_layer_min
            if hasattr(m, "_calibration_state"):
                delattr(m, "_calibration_state")
    return result
