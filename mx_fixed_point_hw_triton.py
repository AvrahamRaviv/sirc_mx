"""Fused Triton kernel for the HW-faithful fixed-point Conv2d path.

Complements `mx_fixed_point_hw._hw_fxp_conv2d_ref`. Same numerics:
per-product int8 x int8 -> int16 multiply, signed shift by
`Ew + Ea - 2*MANTISSA_BIAS - e_layer_min`, saturating int-N add into a
single wide accumulator. Parallel over (M = B*L, O), serial over the
K/bs reduction so the HW pipeline is emulated lane-by-lane.

Importable without Triton (gated on `_TRITON_AVAILABLE`).
"""

import torch
import torch.nn.functional as F

from mx_fixed_point_hw import MANTISSA_BIAS

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def _require_triton_cuda():
    if not _TRITON_AVAILABLE:
        raise RuntimeError(
            "HW fixed-point triton backend requires the `triton` package."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "HW fixed-point triton backend requires a CUDA device."
        )


if _TRITON_AVAILABLE:

    @triton.jit
    def _hw_fxp_conv_kernel(
        qi_ptr,         # *i8,  [M, nb, BS, kK]
        qw_ptr,         # *i8,  [O, nb, BS, kK]
        Ea_ptr,         # *i16, [M, nb, kK]
        Ew_ptr,         # *i16, [O, nb, kK]
        out_ptr,        # *fp32,[M, O]
        sat_ptr,        # *i8,  [M, O]
        M, O, nb, kK,
        e_layer_min,    # i32
        two_bias,       # i32
        lo,             # i64
        hi,             # i64
        inv_scale_ref,  # fp32  = 2^e_layer_min
        BLOCK_M: tl.constexpr,
        BS: tl.constexpr,
        SAT_PER_PRODUCT: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_o = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        acc = tl.zeros([BLOCK_M], dtype=tl.int64)
        sat = tl.zeros([BLOCK_M], dtype=tl.int1)

        stride_qi_m = nb * BS * kK
        stride_qi_kb = BS * kK
        stride_qi_j = kK
        # qw strides mirror qi: [O, nb, BS, kK]
        stride_qw_o = nb * BS * kK
        stride_qw_kb = BS * kK
        stride_qw_j = kK
        # Ea strides: [M, nb, kK]
        stride_Ea_m = nb * kK
        stride_Ea_kb = kK
        # Ew strides: [O, nb, kK]
        stride_Ew_o = nb * kK
        stride_Ew_kb = kK

        for kb in range(0, nb):
            for kk in range(0, kK):
                Ew_val = tl.load(
                    Ew_ptr + pid_o * stride_Ew_o + kb * stride_Ew_kb + kk
                ).to(tl.int64)
                Ea_off = offs_m * stride_Ea_m + kb * stride_Ea_kb + kk
                Ea_val = tl.load(Ea_ptr + Ea_off, mask=mask_m, other=0).to(tl.int64)
                shift = Ew_val + Ea_val - two_bias - e_layer_min
                shift_nonneg = shift >= 0
                pos_shift = tl.where(shift_nonneg, shift, tl.zeros_like(shift))
                neg_shift = tl.where(shift_nonneg, tl.zeros_like(shift), -shift)

                if SAT_PER_PRODUCT:
                    pass
                else:
                    block_acc = tl.zeros([BLOCK_M], dtype=tl.int64)

                for j in tl.static_range(BS):
                    qi_off = offs_m * stride_qi_m + kb * stride_qi_kb + j * stride_qi_j + kk
                    qi_v = tl.load(qi_ptr + qi_off, mask=mask_m, other=0).to(tl.int64)
                    qw_off = pid_o * stride_qw_o + kb * stride_qw_kb + j * stride_qw_j + kk
                    qw_v = tl.load(qw_ptr + qw_off).to(tl.int64)
                    p = qi_v * qw_v
                    s_left = p << pos_shift
                    s_right = p >> neg_shift
                    s = tl.where(shift_nonneg, s_left, s_right)

                    if SAT_PER_PRODUCT:
                        acc = acc + s
                        over = acc > hi
                        under = acc < lo
                        sat = sat | over | under
                        acc = tl.where(over, hi, acc)
                        acc = tl.where(under, lo, acc)
                    else:
                        block_acc = block_acc + s

                if not SAT_PER_PRODUCT:
                    acc = acc + block_acc
                    over = acc > hi
                    under = acc < lo
                    sat = sat | over | under
                    acc = tl.where(over, hi, acc)
                    acc = tl.where(under, lo, acc)

        out = acc.to(tl.float32) * inv_scale_ref
        out_off = offs_m * O + pid_o
        tl.store(out_ptr + out_off, out, mask=mask_m)
        tl.store(sat_ptr + out_off, sat.to(tl.int8), mask=mask_m)


def _prepare_tensors(qi_i8, qw_i8, Ea, Ew, stride, padding, dilation):
    """Layout operands for the kernel.

    Input shapes
    ------------
    qi_i8 : [B, C, H, W]      int8
    qw_i8 : [O, C, kH, kW]    int8
    Ea    : [B, nb, H, W]     int16
    Ew    : [O, nb, kH, kW]   int16

    Output layout
    -------------
    qi_flat  : [M, nb, BS, kK]   int8, M = B * L
    qw_flat  : [O, nb, BS, kK]   int8
    Ea_flat  : [M, nb, kK]       int16
    Ew_flat  : [O, nb, kK]       int16
    (M, L)
    """
    B, C, H, W = qi_i8.shape
    O, _, kH, kW = qw_i8.shape
    nb = Ew.shape[1]
    bs = C // nb
    kK = kH * kW

    # im2col on int8 via float
    qi_unf = F.unfold(
        qi_i8.to(torch.float32), (kH, kW),
        dilation=dilation, padding=padding, stride=stride,
    )  # [B, C*kK, L]
    L = qi_unf.shape[-1]
    qi_unf = qi_unf.view(B, nb, bs, kK, L).permute(0, 4, 1, 2, 3).contiguous().to(torch.int8)
    qi_flat = qi_unf.view(B * L, nb, bs, kK)

    Ea_unf = F.unfold(
        Ea.to(torch.float32), (kH, kW),
        dilation=dilation, padding=padding, stride=stride,
    )  # [B, nb*kK, L]
    Ea_unf = Ea_unf.view(B, nb, kK, L).permute(0, 3, 1, 2).contiguous().to(torch.int16)
    Ea_flat = Ea_unf.view(B * L, nb, kK)

    qw_flat = qw_i8.view(O, nb, bs, kK).contiguous()
    Ew_flat = Ew.view(O, nb, kK).contiguous()
    return qi_flat, qw_flat, Ea_flat, Ew_flat, B * L, L, nb, bs, kK


def _hw_fxp_conv2d_triton(
    qi_i8, qw_i8, Ea, Ew,
    e_layer_min, stride, padding, dilation,
    bs, bits, sat_mode,
):
    _require_triton_cuda()
    if not qi_i8.is_cuda:
        raise RuntimeError("triton backend requires CUDA tensors")

    if not isinstance(stride, tuple):   stride   = (int(stride),   int(stride))
    if not isinstance(padding, tuple):  padding  = (int(padding),  int(padding))
    if not isinstance(dilation, tuple): dilation = (int(dilation), int(dilation))

    qi_flat, qw_flat, Ea_flat, Ew_flat, M, L, nb, bs_chk, kK = _prepare_tensors(
        qi_i8, qw_i8, Ea, Ew, stride, padding, dilation,
    )
    assert bs_chk == bs, f"block_size mismatch: expected {bs}, got {bs_chk}"
    O = qw_flat.shape[0]

    device = qi_i8.device
    out = torch.empty((M, O), dtype=torch.float32, device=device)
    sat = torch.zeros((M, O), dtype=torch.int8, device=device)

    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    two_bias = 2 * MANTISSA_BIAS
    inv_scale_ref = float(2.0 ** int(e_layer_min))
    sat_per_product = 1 if sat_mode == "per_product" else 0

    BLOCK_M = 64
    grid = ((M + BLOCK_M - 1) // BLOCK_M, O)

    _hw_fxp_conv_kernel[grid](
        qi_flat, qw_flat, Ea_flat, Ew_flat, out, sat,
        M, O, nb, kK,
        int(e_layer_min), int(two_bias), int(lo), int(hi), inv_scale_ref,
        BLOCK_M=BLOCK_M, BS=bs, SAT_PER_PRODUCT=sat_per_product,
    )

    # Caller reshapes to [B, O, H_out, W_out]. We return [B, O, L] and sat [B, O, L].
    B = qi_i8.shape[0]
    out_view = out.view(B, L, O).transpose(1, 2).contiguous()   # [B, O, L]
    sat_view = sat.view(B, L, O).transpose(1, 2).contiguous().to(torch.bool)  # [B, O, L]
    return out_view, sat_view
