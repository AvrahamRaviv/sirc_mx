"""Fused Triton kernel path for the emulated fixed-point cross-block accumulator.

Drop-in parallel to FixedPointAccumulator (python loop + int64). Same numerical
semantics: round-to-nearest into int64, per-step saturating add, dequant back to
FP32. Single GPU kernel launch replaces the per-block python loop.

Opt-in via xblock_accum_backend='triton'. No effect on existing modes.
"""

import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def _require_triton_cuda():
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "xblock_accum_backend='triton' requires the `triton` package. "
            "Install it or switch backend to 'python'."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "xblock_accum_backend='triton' requires a CUDA device. "
            "Switch to backend='python' for CPU runs."
        )


if _TRITON_AVAILABLE:

    @triton.jit
    def _sat_accum_kernel(
        partials_ptr,     # *fp32, shape [M, num_blocks], contiguous
        out_ptr,          # *fp32, shape [M]
        sat_mask_ptr,     # *i8,   shape [M]
        M,                # int
        num_blocks,       # int
        scale,            # fp32 (python float -> fp32 scalar)
        inv_scale,        # fp32
        lo,               # i64
        hi,               # i64
        SATURATE: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        acc = tl.zeros([BLOCK_M], dtype=tl.int64)
        sat = tl.zeros([BLOCK_M], dtype=tl.int1)

        for k in range(0, num_blocks):
            ptr = partials_ptr + offs_m * num_blocks + k
            v = tl.load(ptr, mask=mask_m, other=0.0)
            s = tl.extra.cuda.libdevice.llrint(v * scale).to(tl.int64)
            acc = acc + s
            if SATURATE:
                over = acc > hi
                under = acc < lo
                sat = sat | over | under
                acc = tl.where(over, hi, acc)
                acc = tl.where(under, lo, acc)

        out = acc.to(tl.float32) * inv_scale
        tl.store(out_ptr + offs_m, out, mask=mask_m)
        tl.store(sat_mask_ptr + offs_m, sat.to(tl.int8), mask=mask_m)


class FixedPointAccumulatorTriton(torch.autograd.Function):
    """Triton-backed equivalent of FixedPointAccumulator.

    Forward: one kernel launch. Per-step saturating int64 accumulate, dequant to fp32.
    Backward: STE, same as python path.
    """

    @staticmethod
    def forward(ctx, partials, total_bits, scale_exp, saturate, ste_mask):
        _require_triton_cuda()

        from mx_fixed_point import _auto_scale_exp, validate_xblock_accum_bits

        total_bits = int(total_bits)
        validate_xblock_accum_bits(total_bits)
        if scale_exp is None:
            scale_exp = _auto_scale_exp(partials, total_bits)
        scale_exp = int(scale_exp)

        if partials.device.type != "cuda":
            raise RuntimeError(
                f"triton backend requires CUDA tensor, got device={partials.device}"
            )

        lead_shape = partials.shape[:-1]
        num_blocks = partials.shape[-1]
        M = 1
        for d in lead_shape:
            M *= d

        partials_contig = partials.detach().contiguous().to(torch.float32)
        partials_flat = partials_contig.view(M, num_blocks)

        out = torch.empty(M, dtype=torch.float32, device=partials.device)
        sat_mask = torch.zeros(M, dtype=torch.int8, device=partials.device)

        scale = float(2 ** scale_exp)
        inv_scale = float(2 ** -scale_exp)
        lo = -(1 << (total_bits - 1))
        hi = (1 << (total_bits - 1)) - 1

        BLOCK_M = 128
        grid = ((M + BLOCK_M - 1) // BLOCK_M,)

        _sat_accum_kernel[grid](
            partials_flat,
            out,
            sat_mask,
            M,
            num_blocks,
            scale,
            inv_scale,
            lo,
            hi,
            SATURATE=bool(saturate),
            BLOCK_M=BLOCK_M,
        )

        out = out.view(lead_shape).to(partials.dtype)
        saturated_mask = sat_mask.view(lead_shape).to(torch.bool)

        ctx.save_for_backward(saturated_mask)
        ctx.ste_mask = bool(ste_mask)
        ctx.num_blocks = num_blocks
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (saturated_mask,) = ctx.saved_tensors
        grad = grad_out.unsqueeze(-1).expand(*grad_out.shape, ctx.num_blocks)
        if ctx.ste_mask and saturated_mask.any():
            keep = (~saturated_mask).to(grad.dtype).unsqueeze(-1)
            grad = grad * keep
        return grad.contiguous(), None, None, None, None


def fixed_point_accumulate_triton(
    partials,
    total_bits=48,
    scale_exp=None,
    saturate=True,
    ste_mask=False,
):
    return FixedPointAccumulatorTriton.apply(
        partials, total_bits, scale_exp, saturate, ste_mask
    )
