"""HW-faithful fixed-point emulation for MX Conv2d.

Real hardware has a single 8b x 8b -> 16b multiplier per BCU (not a block
dot product). Each product is shifted by `Ew + Ea - E_layer_min` and
pushed into a narrow (default 35b) saturating fixed-point accumulator,
product-by-product. Intra-block reduction happens inside the fixed-point
accumulator, not in FP32.

This module provides:
  * `extract_mxint8` - recover int8 values and shared exponent from a
    tensor already quantised by microxcaling's `quantize_mx_op`.
  * `_hw_fxp_conv2d_ref` - pure-torch reference that computes the same
    numerics as the kernel (parallel over B,O,L; serial over K/bs).
  * `HWFxpConv2dFn` - autograd Function with STE backward (treats forward
    as if it were `F.conv2d` for gradient purposes).
  * `hw_fxp_conv2d` - public entry point used by `MXConv2dHW`.
  * `calibrate_e_layer_min` - sweep representative inputs and freeze the
    per-layer static `e_layer_min` attribute.

Only MXINT8 is supported. The FP8/MXFP paths are rejected; the HW flow
we model is integer-multiply + integer-accumulate.
"""

import torch
import torch.nn.functional as F

MANTISSA_BIAS = 6  # MXINT8 step size is 2^(E - 6); int values live in [-127, 127].


def extract_mxint8(x_q_fp32, bs, axis):
    """Recover (int8, shared_exp_int16) from an MXINT8-quantised FP32 tensor.

    `x_q_fp32` is assumed to already lie on the MXINT8 lattice (produced by
    `microxcaling.mx.quantize_mx_op` with elem_format='int8'). The recovery
    is exact on-lattice.

    Returns
    -------
    int8_tensor : same shape as `x_q_fp32`, dtype int8
    shared_exp  : `x_q_fp32` shape with `axis` reduced from C to nb = C/bs,
                  dtype int16
    """
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

    scale_inv = torch.pow(torch.tensor(2.0, device=x.device), MANTISSA_BIAS - E)
    q = torch.round(xb * scale_inv).clamp_(-127, 127).to(torch.int8)

    q_full = q.view(*prefix, C).movedim(-1, axis).contiguous()
    E_full = E.squeeze(-1).to(torch.int16).movedim(-1, axis).contiguous()
    return q_full, E_full


def _sat(x, lo, hi):
    """Clamp to [lo, hi] and return (clamped, over|under mask)."""
    sat = (x < lo) | (x > hi)
    return x.clamp(lo, hi), sat


def _hw_fxp_conv2d_ref(
    qi_i8, qw_i8, Ea, Ew,
    e_layer_min, stride, padding, dilation,
    bs, bits, sat_mode,
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
    two_bias = 2 * MANTISSA_BIAS
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

    out = acc.to(torch.float32) * float(2.0 ** e_min)
    return out, sat


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
    ):
        with torch.no_grad():
            qi_i8, Ea = extract_mxint8(qi_fp, bs, axis=1)
            qw_i8, Ew = extract_mxint8(qw_fp, bs, axis=1)

        use_triton = (backend == "triton" and qi_fp.is_cuda)
        if use_triton:
            from mx_fixed_point_hw_triton import _hw_fxp_conv2d_triton
            out_flat, sat_flat = _hw_fxp_conv2d_triton(
                qi_i8, qw_i8, Ea, Ew,
                int(e_layer_min), stride, padding, dilation,
                bs, int(bits), sat_mode,
            )
        else:
            out_flat, sat_flat = _hw_fxp_conv2d_ref(
                qi_i8, qw_i8, Ea, Ew,
                int(e_layer_min), stride, padding, dilation,
                bs, int(bits), sat_mode,
            )

        B, _, H, W = qi_fp.shape
        O, _, kH, kW = qw_fp.shape
        sH, sW = (stride if isinstance(stride, tuple) else (stride, stride))
        pH, pW = (padding if isinstance(padding, tuple) else (padding, padding))
        dH, dW = (dilation if isinstance(dilation, tuple) else (dilation, dilation))
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        assert H_out * W_out == out_flat.shape[-1]

        out = out_flat.view(B, O, H_out, W_out).to(qi_fp.dtype)
        sat = sat_flat.view(B, O, H_out, W_out)
        if bias_fp is not None:
            out = out + bias_fp.view(1, -1, 1, 1)

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

        # forward took 12 args: qi_fp, qw_fp, bias_fp, e_layer_min, bs, bits,
        # sat_mode, ste_mask, stride, padding, dilation, backend
        return (
            grad_qi, grad_qw, grad_bias,
            None, None, None, None, None,
            None, None, None, None,
        )


def hw_fxp_conv2d(
    qi_fp, qw_fp, bias_fp,
    e_layer_min, bs, bits=35, sat_mode="per_product", ste_mask=False,
    stride=1, padding=0, dilation=1, backend="python",
):
    """Public entry point; see `HWFxpConv2dFn.forward` for semantics."""
    if e_layer_min is None:
        raise RuntimeError(
            "hw_fxp_conv2d requires a static e_layer_min. Run "
            "`calibrate_e_layer_min(model, loader)` before inference."
        )
    return HWFxpConv2dFn.apply(
        qi_fp, qw_fp, bias_fp,
        int(e_layer_min), int(bs), int(bits), str(sat_mode), bool(ste_mask),
        stride, padding, dilation, str(backend),
    )


# ------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------

def _compute_min_shift_exp(qi_fp, qw_fp, bs):
    """Return layer-wide min(Ew + Ea - 2*mantissa_bias) over all block pairs.

    Ea and Ew are independent (no shared index), so the joint min is simply
    min(Ea) + min(Ew).
    """
    _, Ea = extract_mxint8(qi_fp, bs, axis=1)
    _, Ew = extract_mxint8(qw_fp, bs, axis=1)
    return int(Ea.amin().item()) + int(Ew.amin().item()) - 2 * MANTISSA_BIAS


class _CalibrationState:
    """Per-layer running min(Ew + Ea - 12) collector."""

    def __init__(self):
        self.running_min = None

    def update(self, qi_fp, qw_fp, bs):
        cand = _compute_min_shift_exp(qi_fp, qw_fp, bs)
        self.running_min = cand if self.running_min is None else min(self.running_min, cand)


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

    layers = [m for m in model.modules() if isinstance(m, MXConv2dHW)]
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
            delattr(m, "_calibration_state")
    return result
