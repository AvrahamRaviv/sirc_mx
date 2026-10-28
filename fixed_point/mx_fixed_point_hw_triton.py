"""Fused Triton kernel for the HW-faithful fixed-point Conv2d path.

Complements `mx_fixed_point_hw._hw_fxp_conv2d_ref`. Same numerics:
per-product intN x intN multiply, signed shift by
`Ew + Ea - 2*mant_bias - e_layer_min`, saturating int-bits add into a
single wide accumulator. Parallel over (M = B*L, O), serial over the
K/bs reduction so the HW pipeline is emulated lane-by-lane.

Operands are promoted to int32 in the launcher so a single kernel
handles every MXINT format (int8..int16); product math runs in int64.

With APPLY_WIN the static output window (`mx_fixed_point_hw._apply_out_window`)
runs on the int64 word before the fp32 cast; its constants come from
`_window_params`, shared with the torch ref. APPLY_PLA multiplies the word by
the per-channel PLA multiplier first, or with PLA_BINS by its bin's scale plus
bias (`mx_fixed_point_hw._apply_pla`).

Importable without Triton (gated on `_TRITON_AVAILABLE`).
"""

import torch
import torch.nn.functional as F

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
        qi_ptr,         # *i32, [M, nb, BS, kK]   (operands promoted to int32 in launcher)
        qw_ptr,         # *i32, [O, nb, BS, kK]
        Ea_ptr,         # *i16, [M, nb, kK]
        Ew_ptr,         # *i16, [O, nb, kK]
        bias_ptr,       # *i64, [O]   (only read if HAS_BIAS)
        out_ptr,        # *fp32,[M, O]
        sat_ptr,        # *i8,  [M, O]
        M, O, nb, kK,
        e_layer_min,    # i32
        two_bias,       # i32  (= 2 * mant_bias for the chosen MXINT format)
        lo,             # i64
        hi,             # i64
        inv_scale_ref,  # fp32  = 2^e_layer_min
        win_sat_ptr,    # *i8,  [M, O]  (only written if APPLY_WIN)
        win_rshift, win_lshift, win_round_add, win_hi, win_pre_hi,  # i64
        out_scale,      # fp32  = 2^e_qa_out
        pla_pos_ptr,    # *i64, [O]   (only read if APPLY_PLA)
        pla_neg_ptr,    # *i64, [O]
        pla_bias_ptr,   # *i64, [O]   (only read if PLA_POST_BIAS)
        pla_bnd_ptr,    # *i64, [O, PLA_BINS - 1]  (only read if PLA_BINS)
        pla_scl_ptr,    # *i64, [O, PLA_BINS]
        pla_bb_ptr,     # *i64, [O, PLA_BINS]
        BLOCK_M: tl.constexpr,
        BS: tl.constexpr,
        SAT_PER_PRODUCT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        APPLY_WIN: tl.constexpr,
        APPLY_PLA: tl.constexpr,
        PLA_POST_BIAS: tl.constexpr,
        PLA_BINS: tl.constexpr,         # 0 = m_pos/m_neg form, else bins per channel
        PLA_TIE_UPPER: tl.constexpr,
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

        if HAS_BIAS:
            b_int = tl.load(bias_ptr + pid_o).to(tl.int64)
            acc = acc + b_int
            over = acc > hi
            under = acc < lo
            sat = sat | over | under
            acc = tl.where(over, hi, acc)
            acc = tl.where(under, lo, acc)

        if APPLY_PLA:
            if PLA_BINS > 0:
                s_k = tl.zeros_like(acc) + tl.load(pla_scl_ptr + pid_o * PLA_BINS).to(tl.int64)
                b_k = tl.zeros_like(acc) + tl.load(pla_bb_ptr + pid_o * PLA_BINS).to(tl.int64)
                for k in tl.static_range(1, PLA_BINS):
                    t = tl.load(pla_bnd_ptr + pid_o * (PLA_BINS - 1) + (k - 1)).to(tl.int64)
                    if PLA_TIE_UPPER:
                        sel = acc >= t
                    else:
                        sel = acc > t
                    s_k = tl.where(sel, tl.load(pla_scl_ptr + pid_o * PLA_BINS + k).to(tl.int64), s_k)
                    b_k = tl.where(sel, tl.load(pla_bb_ptr + pid_o * PLA_BINS + k).to(tl.int64), b_k)
                acc = acc * s_k + b_k
            else:
                m_pos = tl.load(pla_pos_ptr + pid_o).to(tl.int64)
                m_neg = tl.load(pla_neg_ptr + pid_o).to(tl.int64)
                acc = acc * tl.where(acc < 0, m_neg, m_pos)
                if PLA_POST_BIAS:
                    acc = acc + tl.load(pla_bias_ptr + pid_o).to(tl.int64)

        out_off = offs_m * O + pid_o
        if APPLY_WIN:
            v = (acc + win_round_add) >> win_rshift
            w_over = v > win_pre_hi
            w_under = v < -win_pre_hi
            v = v << win_lshift
            v = tl.where(w_over, win_hi, v)
            v = tl.where(w_under, -win_hi, v)
            out = v.to(tl.float32) * out_scale
            tl.store(win_sat_ptr + out_off, (w_over | w_under).to(tl.int8), mask=mask_m)
        else:
            out = acc.to(tl.float32) * inv_scale_ref
        tl.store(out_ptr + out_off, out, mask=mask_m)
        tl.store(sat_ptr + out_off, sat.to(tl.int8), mask=mask_m)

    @triton.jit
    def _hw_fxp_conv_kernel_npe(
        qi_ptr,         # *i32, [M, nb, BS, kK]
        qw_ptr,         # *i32, [O, nb, BS, kK]
        Ea_ptr,         # *i16, [M, nb, BS, kK]   (per-lane; NPE X-block exp)
        Ew_ptr,         # *i16, [O, nb, BS, kK]   (per-lane; NPE flatten-block exp)
        bias_ptr,       # *i64, [O]   (only read if HAS_BIAS)
        out_ptr,        # *fp32,[M, O]
        sat_ptr,        # *i8,  [M, O]
        M, O, nb, kK,
        e_layer_min,    # i32
        two_bias,       # i32
        lo,             # i64
        hi,             # i64
        inv_scale_ref,  # fp32 = 2^e_layer_min
        win_sat_ptr,    # *i8,  [M, O]  (only written if APPLY_WIN)
        win_rshift, win_lshift, win_round_add, win_hi, win_pre_hi,  # i64
        out_scale,      # fp32  = 2^e_qa_out
        pla_pos_ptr,    # *i64, [O]   (only read if APPLY_PLA)
        pla_neg_ptr,    # *i64, [O]
        pla_bias_ptr,   # *i64, [O]   (only read if PLA_POST_BIAS)
        pla_bnd_ptr,    # *i64, [O, PLA_BINS - 1]  (only read if PLA_BINS)
        pla_scl_ptr,    # *i64, [O, PLA_BINS]
        pla_bb_ptr,     # *i64, [O, PLA_BINS]
        BLOCK_M: tl.constexpr,
        BS: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        APPLY_WIN: tl.constexpr,
        APPLY_PLA: tl.constexpr,
        PLA_POST_BIAS: tl.constexpr,
        PLA_BINS: tl.constexpr,         # 0 = m_pos/m_neg form, else bins per channel
        PLA_TIE_UPPER: tl.constexpr,
    ):
        """NPE kernel: independent per-lane exponents (no lane sharing). Loop
        order kb -> j -> kk (cin = kb*BS + j outer, kk inner) reproduces the
        torch ref's cin-major/kk-inner accumulation order, so per_product
        saturation is bit-identical to the ref and the bs=1 reuse path.
        Ea/Ew share qi/qw's [.,nb,BS,kK] layout, so their element offsets match.
        """
        pid_m = tl.program_id(axis=0)
        pid_o = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        acc = tl.zeros([BLOCK_M], dtype=tl.int64)
        sat = tl.zeros([BLOCK_M], dtype=tl.int1)

        stride_qi_m = nb * BS * kK
        stride_qi_kb = BS * kK
        stride_qi_j = kK
        stride_qw_o = nb * BS * kK
        stride_qw_kb = BS * kK
        stride_qw_j = kK

        for kb in range(0, nb):
            for j in tl.static_range(BS):
                for kk in range(0, kK):
                    qi_off = (offs_m * stride_qi_m + kb * stride_qi_kb
                              + j * stride_qi_j + kk)
                    qw_off = (pid_o * stride_qw_o + kb * stride_qw_kb
                              + j * stride_qw_j + kk)
                    qi_v = tl.load(qi_ptr + qi_off, mask=mask_m, other=0).to(tl.int64)
                    qw_v = tl.load(qw_ptr + qw_off).to(tl.int64)
                    Ea_val = tl.load(Ea_ptr + qi_off, mask=mask_m, other=0).to(tl.int64)
                    Ew_val = tl.load(Ew_ptr + qw_off).to(tl.int64)
                    shift = Ew_val + Ea_val - two_bias - e_layer_min
                    shift_nonneg = shift >= 0
                    pos_shift = tl.where(shift_nonneg, shift, tl.zeros_like(shift))
                    neg_shift = tl.where(shift_nonneg, tl.zeros_like(shift), -shift)
                    p = qi_v * qw_v
                    s_left = p << pos_shift
                    s_right = p >> neg_shift
                    s = tl.where(shift_nonneg, s_left, s_right)
                    acc = acc + s
                    over = acc > hi
                    under = acc < lo
                    sat = sat | over | under
                    acc = tl.where(over, hi, acc)
                    acc = tl.where(under, lo, acc)

        if HAS_BIAS:
            b_int = tl.load(bias_ptr + pid_o).to(tl.int64)
            acc = acc + b_int
            over = acc > hi
            under = acc < lo
            sat = sat | over | under
            acc = tl.where(over, hi, acc)
            acc = tl.where(under, lo, acc)

        if APPLY_PLA:
            if PLA_BINS > 0:
                s_k = tl.zeros_like(acc) + tl.load(pla_scl_ptr + pid_o * PLA_BINS).to(tl.int64)
                b_k = tl.zeros_like(acc) + tl.load(pla_bb_ptr + pid_o * PLA_BINS).to(tl.int64)
                for k in tl.static_range(1, PLA_BINS):
                    t = tl.load(pla_bnd_ptr + pid_o * (PLA_BINS - 1) + (k - 1)).to(tl.int64)
                    if PLA_TIE_UPPER:
                        sel = acc >= t
                    else:
                        sel = acc > t
                    s_k = tl.where(sel, tl.load(pla_scl_ptr + pid_o * PLA_BINS + k).to(tl.int64), s_k)
                    b_k = tl.where(sel, tl.load(pla_bb_ptr + pid_o * PLA_BINS + k).to(tl.int64), b_k)
                acc = acc * s_k + b_k
            else:
                m_pos = tl.load(pla_pos_ptr + pid_o).to(tl.int64)
                m_neg = tl.load(pla_neg_ptr + pid_o).to(tl.int64)
                acc = acc * tl.where(acc < 0, m_neg, m_pos)
                if PLA_POST_BIAS:
                    acc = acc + tl.load(pla_bias_ptr + pid_o).to(tl.int64)

        out_off = offs_m * O + pid_o
        if APPLY_WIN:
            v = (acc + win_round_add) >> win_rshift
            w_over = v > win_pre_hi
            w_under = v < -win_pre_hi
            v = v << win_lshift
            v = tl.where(w_over, win_hi, v)
            v = tl.where(w_under, -win_hi, v)
            out = v.to(tl.float32) * out_scale
            tl.store(win_sat_ptr + out_off, (w_over | w_under).to(tl.int8), mask=mask_m)
        else:
            out = acc.to(tl.float32) * inv_scale_ref
        tl.store(out_ptr + out_off, out, mask=mask_m)
        tl.store(sat_ptr + out_off, sat.to(tl.int8), mask=mask_m)


def _window_launch_args(e_layer_min, e_qa_out, win_bits, out_round, M, O, device):
    """(win_sat buffer, scalar args, APPLY_WIN) for a kernel launch."""
    from .mx_fixed_point_hw import _window_params
    if e_qa_out is None:
        return torch.empty(1, dtype=torch.int8, device=device), (0, 0, 0, 0, 0, 1.0), 0
    rshift, lshift, round_add, win_hi, pre_hi = _window_params(
        e_layer_min, e_qa_out, win_bits, out_round)
    win_sat = torch.zeros((M, O), dtype=torch.int8, device=device)
    return win_sat, (rshift, lshift, round_add, win_hi, pre_hi,
                     float(2.0 ** int(e_qa_out))), 1


def _bias_pla_launch_args(bias_fp, pla, e_layer_min, O, device):
    """(bias_int, HAS_BIAS, e_win, pla tensors, pla constexprs).

    The accumulator bias is round(b * 2^-e_layer_min), half up with PLA; in
    the PLA 'post' order it goes after the multiply instead, on the grid
    e_win = e_layer_min - frac the window then extracts from. 'bins' has no
    accumulator bias: each bin carries its own (`mx_fixed_point_hw.pla_bins`)."""
    from .mx_fixed_point_hw import _bias_to_grid
    dummy = torch.empty(1, dtype=torch.int64, device=device)
    post = pla is not None and pla.bias == "post"
    bins = pla is not None and pla.bias == "bins"
    has_bias = bias_fp is not None and not post and not bins
    bias_int = (_bias_to_grid(bias_fp, e_layer_min, pla is not None, device).contiguous()
                if has_bias else torch.empty(O, dtype=torch.int64, device=device))
    flags = dict(APPLY_PLA=0, PLA_POST_BIAS=0, PLA_BINS=0, PLA_TIE_UPPER=1)
    if pla is None:
        return bias_int, has_bias, int(e_layer_min), (dummy,) * 6, flags
    e_win = int(e_layer_min) - int(pla.frac)
    b_post = (_bias_to_grid(bias_fp, e_win, True, device).contiguous()
              if post and bias_fp is not None else dummy)
    pos = pla.m_pos.to(device=device, dtype=torch.int64).contiguous()
    neg = pla.m_neg.to(device=device, dtype=torch.int64).contiguous()
    bnd = scl = bb = dummy
    if bins:
        bnd, scl, bb = (t.to(device=device, dtype=torch.int64).contiguous()
                        for t in (pla.bounds, pla.scales, pla.biases))
        if bnd.numel() == 0:
            bnd = dummy
    flags.update(APPLY_PLA=1, PLA_POST_BIAS=1 if (post and bias_fp is not None) else 0,
                 PLA_BINS=int(pla.scales.shape[1]) if bins else 0,
                 PLA_TIE_UPPER=1 if pla.tie == "upper" else 0)
    return bias_int, has_bias, e_win, (pos, neg, b_post, bnd, scl, bb), flags


def _fill_win_sink(win_sink, win_sat, B, L, O):
    if win_sink is not None and win_sat.numel() == B * L * O:
        win_sink["win_sat"] = win_sat.view(B, L, O).transpose(1, 2).contiguous().to(torch.bool)


def _prepare_tensors(qi_i8, qw_i8, Ea, Ew, stride, padding, dilation):
    """Layout operands for the kernel.

    Input shapes
    ------------
    qi_i8 : [B, C, H, W]      int8 or int16 (intN, N<=16)
    qw_i8 : [O, C, kH, kW]    int8 or int16
    Ea    : [B, nb, H, W]     int16
    Ew    : [O, nb, kH, kW]   int16

    Output layout (operands promoted to int32 for a single kernel path)
    -------------
    qi_flat  : [M, nb, BS, kK]   int32, M = B * L
    qw_flat  : [O, nb, BS, kK]   int32
    Ea_flat  : [M, nb, kK]       int16
    Ew_flat  : [O, nb, kK]       int16
    (M, L)
    """
    B, C, H, W = qi_i8.shape
    O, _, kH, kW = qw_i8.shape
    nb = Ew.shape[1]
    bs = C // nb
    kK = kH * kW

    # im2col via float (int dtypes not supported by F.unfold), then cast back.
    qi_unf = F.unfold(
        qi_i8.to(torch.float32), (kH, kW),
        dilation=dilation, padding=padding, stride=stride,
    )  # [B, C*kK, L]
    L = qi_unf.shape[-1]
    qi_unf = qi_unf.view(B, nb, bs, kK, L).permute(0, 4, 1, 2, 3).contiguous().to(torch.int32)
    qi_flat = qi_unf.view(B * L, nb, bs, kK)

    Ea_unf = F.unfold(
        Ea.to(torch.float32), (kH, kW),
        dilation=dilation, padding=padding, stride=stride,
    )  # [B, nb*kK, L]
    Ea_unf = Ea_unf.view(B, nb, kK, L).permute(0, 3, 1, 2).contiguous().to(torch.int16)
    Ea_flat = Ea_unf.view(B * L, nb, kK)

    qw_flat = qw_i8.view(O, nb, bs, kK).to(torch.int32).contiguous()
    Ew_flat = Ew.view(O, nb, kK).contiguous()
    return qi_flat, qw_flat, Ea_flat, Ew_flat, B * L, L, nb, bs, kK


def _hw_fxp_conv2d_triton(
    qi_i8, qw_i8, Ea, Ew,
    e_layer_min, stride, padding, dilation,
    bs, bits, sat_mode,
    bias_fp=None,
    mant_bias=6,
    e_qa_out=None, win_bits=None, out_round="floor", win_sink=None,
    pla=None,
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
    two_bias = 2 * int(mant_bias)
    inv_scale_ref = float(2.0 ** int(e_layer_min))
    sat_per_product = 1 if sat_mode == "per_product" else 0

    # Bias onto the accumulator's fixed-point grid: acc represents
    # value * 2^(-e_layer_min), so bias_int = round(b * 2^(-e_layer_min)).
    bias_int, has_bias, e_win, pla_args, pla_flags = \
        _bias_pla_launch_args(bias_fp, pla, e_layer_min, O, device)

    win_sat, win_args, apply_win = _window_launch_args(
        e_win, e_qa_out, win_bits, out_round, M, O, device)

    BLOCK_M = 64
    grid = ((M + BLOCK_M - 1) // BLOCK_M, O)

    _hw_fxp_conv_kernel[grid](
        qi_flat, qw_flat, Ea_flat, Ew_flat, bias_int, out, sat,
        M, O, nb, kK,
        int(e_layer_min), int(two_bias), int(lo), int(hi), inv_scale_ref,
        win_sat, *win_args, *pla_args,
        BLOCK_M=BLOCK_M, BS=bs, SAT_PER_PRODUCT=sat_per_product,
        HAS_BIAS=1 if has_bias else 0, APPLY_WIN=apply_win,
        **pla_flags,
    )

    # Caller reshapes to [B, O, H_out, W_out]. We return [B, O, L] and sat [B, O, L].
    B = qi_i8.shape[0]
    _fill_win_sink(win_sink, win_sat, B, L, O)
    out_view = out.view(B, L, O).transpose(1, 2).contiguous()   # [B, O, L]
    sat_view = sat.view(B, L, O).transpose(1, 2).contiguous().to(torch.bool)  # [B, O, L]
    return out_view, sat_view


def _prepare_tensors_npe(qi_i8, qw_i8, Ea_full, Ew_full,
                         stride, padding, dilation, BS):
    """Layout NPE operands for `_hw_fxp_conv_kernel_npe`.

    Input
    -----
    qi_i8   : [B, C, H, W]     intN
    qw_i8   : [O, C, kH, kW]   intN
    Ea_full : [B, C, H, W]     int16 (per-element X-block exp)
    Ew_full : [O, C, kH, kW]   int16 (per-element flatten-block exp)

    Cin is zero-padded up to a multiple of BS (zero channels -> zero products,
    E=0; no effect on the sum or saturation). Output layout:
      qi_flat/Ea_flat : [M, nb, BS, kK]   (M = B*L)
      qw_flat/Ew_flat : [O, nb, BS, kK]
    """
    B, C, H, W = qi_i8.shape
    O, _, kH, kW = qw_i8.shape
    kK = kH * kW

    Cpad = ((C + BS - 1) // BS) * BS
    pad = Cpad - C
    if pad:
        qi_i8 = F.pad(qi_i8, (0, 0, 0, 0, 0, pad))
        qw_i8 = F.pad(qw_i8, (0, 0, 0, 0, 0, pad))
        Ea_full = F.pad(Ea_full, (0, 0, 0, 0, 0, pad))
        Ew_full = F.pad(Ew_full, (0, 0, 0, 0, 0, pad))
        C = Cpad
    nb = C // BS

    qi_unf = F.unfold(
        qi_i8.to(torch.float32), (kH, kW),
        dilation=dilation, padding=padding, stride=stride,
    )  # [B, C*kK, L]
    L = qi_unf.shape[-1]
    qi_unf = qi_unf.view(B, nb, BS, kK, L).permute(0, 4, 1, 2, 3).contiguous().to(torch.int32)
    qi_flat = qi_unf.view(B * L, nb, BS, kK)

    Ea_unf = F.unfold(
        Ea_full.to(torch.float32), (kH, kW),
        dilation=dilation, padding=padding, stride=stride,
    )  # [B, C*kK, L]
    Ea_unf = Ea_unf.view(B, nb, BS, kK, L).permute(0, 4, 1, 2, 3).contiguous().to(torch.int16)
    Ea_flat = Ea_unf.view(B * L, nb, BS, kK)

    qw_flat = qw_i8.view(O, nb, BS, kK).to(torch.int32).contiguous()
    Ew_flat = Ew_full.view(O, nb, BS, kK).to(torch.int16).contiguous()
    return qi_flat, qw_flat, Ea_flat, Ew_flat, B * L, L, nb, kK


def _hw_fxp_conv2d_triton_npe(
    qi_i8, qw_i8, Ea_full, Ew_full,
    e_layer_min, stride, padding, dilation,
    bits, sat_mode,
    bias_fp=None,
    mant_bias=6,
    BS=32,
    e_qa_out=None, win_bits=None, out_round="floor", win_sink=None,
    pla=None,
):
    """Dedicated NPE triton path (per-lane exponents, BS-tiled). Bit-identical
    to the torch ref and the bs=1 reuse path; here for cross-verification.
    """
    _require_triton_cuda()
    if not qi_i8.is_cuda:
        raise RuntimeError("triton backend requires CUDA tensors")
    if sat_mode == "per_block":
        raise ValueError(
            "sat_mode='per_block' is not supported for NPE blockify; "
            "use sat_mode='per_product'."
        )

    if not isinstance(stride, tuple):   stride   = (int(stride),   int(stride))
    if not isinstance(padding, tuple):  padding  = (int(padding),  int(padding))
    if not isinstance(dilation, tuple): dilation = (int(dilation), int(dilation))

    qi_flat, qw_flat, Ea_flat, Ew_flat, M, L, nb, kK = _prepare_tensors_npe(
        qi_i8, qw_i8, Ea_full, Ew_full, stride, padding, dilation, BS,
    )
    O = qw_flat.shape[0]

    device = qi_i8.device
    out = torch.empty((M, O), dtype=torch.float32, device=device)
    sat = torch.zeros((M, O), dtype=torch.int8, device=device)

    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    two_bias = 2 * int(mant_bias)
    inv_scale_ref = float(2.0 ** int(e_layer_min))

    bias_int, has_bias, e_win, pla_args, pla_flags = \
        _bias_pla_launch_args(bias_fp, pla, e_layer_min, O, device)

    win_sat, win_args, apply_win = _window_launch_args(
        e_win, e_qa_out, win_bits, out_round, M, O, device)

    BLOCK_M = 64
    grid = ((M + BLOCK_M - 1) // BLOCK_M, O)

    _hw_fxp_conv_kernel_npe[grid](
        qi_flat, qw_flat, Ea_flat, Ew_flat, bias_int, out, sat,
        M, O, nb, kK,
        int(e_layer_min), int(two_bias), int(lo), int(hi), inv_scale_ref,
        win_sat, *win_args, *pla_args,
        BLOCK_M=BLOCK_M, BS=BS,
        HAS_BIAS=1 if has_bias else 0, APPLY_WIN=apply_win,
        **pla_flags,
    )

    B = qi_i8.shape[0]
    _fill_win_sink(win_sink, win_sat, B, L, O)
    out_view = out.view(B, L, O).transpose(1, 2).contiguous()
    sat_view = sat.view(B, L, O).transpose(1, 2).contiguous().to(torch.bool)
    return out_view, sat_view
