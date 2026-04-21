import torch

_MIN_BITS = 40
_MAX_BITS = 48
_DEFAULT_BITS = 48


def validate_xblock_accum_bits(bits):
    if not isinstance(bits, int):
        raise TypeError(f"xblock_accum_bits must be int, got {type(bits).__name__}")
    if bits < _MIN_BITS or bits > _MAX_BITS:
        raise ValueError(
            f"xblock_accum_bits={bits} out of range [{_MIN_BITS}, {_MAX_BITS}]"
        )
    return bits


def _auto_scale_exp(partials, total_bits, headroom_bits=2):
    max_abs = partials.detach().abs().amax()
    if max_abs.item() == 0.0:
        return 0
    target = float(1 << (total_bits - 1 - headroom_bits))
    exp = int(torch.floor(torch.log2(target / max_abs)).item())
    return exp


class FixedPointAccumulator(torch.autograd.Function):
    """
    Emulate a signed saturating N-bit fixed-point cross-block accumulator.

    Forward:
        int_partials = round(partials * 2^scale_exp).to(int64)
        acc = 0
        for k in range(num_blocks):
            acc = clamp(acc + int_partials[..., k], -2^(N-1), 2^(N-1)-1)
        out = acc.float() * 2^(-scale_exp)

    Backward: STE — gradient flows to all partials unchanged.
        Optional: zero grad for elements whose final accumulator saturated
        (set via ste_mask flag). Backward cost is ~free.
    """

    @staticmethod
    def forward(ctx, partials, total_bits, scale_exp, saturate, ste_mask):
        total_bits = int(total_bits)
        validate_xblock_accum_bits(total_bits)

        if scale_exp is None:
            scale_exp = _auto_scale_exp(partials, total_bits)
        scale_exp = int(scale_exp)

        lo = -(1 << (total_bits - 1))
        hi = (1 << (total_bits - 1)) - 1

        scale = float(2 ** scale_exp)
        inv_scale = float(2 ** -scale_exp)

        int_partials = torch.round(partials.detach().to(torch.float64) * scale).to(torch.int64)

        num_blocks = int_partials.shape[-1]
        acc = torch.zeros(int_partials.shape[:-1], dtype=torch.int64, device=partials.device)
        saturated_mask = torch.zeros_like(acc, dtype=torch.bool)

        for k in range(num_blocks):
            acc = acc + int_partials[..., k]
            if saturate:
                over = acc > hi
                under = acc < lo
                if over.any() or under.any():
                    saturated_mask = saturated_mask | over | under
                    acc = torch.clamp(acc, min=lo, max=hi)

        out = acc.to(partials.dtype) * inv_scale

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


def fixed_point_accumulate(
    partials,
    total_bits=_DEFAULT_BITS,
    scale_exp=None,
    saturate=True,
    ste_mask=False,
):
    """
    Sum `partials` along the last dim through an emulated N-bit signed saturating
    fixed-point accumulator. FP32 in, FP32 out. STE backward.

    partials : (..., num_blocks) FP32/FP64 tensor
    total_bits : accumulator bitwidth, default 48, valid [40, 48]
    scale_exp  : int. If None, chosen automatically per-call from partials range
    saturate   : if False, behaves as exact int64 sum (for debugging)
    ste_mask   : if True, zero grad on lanes whose accumulator saturated
    """
    return FixedPointAccumulator.apply(partials, total_bits, scale_exp, saturate, ste_mask)


_VALID_MODES = ("fp32", "fixed_point")
_VALID_BACKENDS = ("python", "triton")


def _get_xblock(mx_specs, key, default):
    """Read xblock_accum_* from a specs object.

    Prefers python attribute (bypasses microxcaling apply_mx_specs key check),
    falls back to dict-style __getitem__ for plain-dict specs (used in tests).
    """
    if hasattr(mx_specs, key):
        return getattr(mx_specs, key)
    try:
        return mx_specs[key]
    except (KeyError, TypeError):
        return default


def cross_block_accumulate_from_specs(partials, mx_specs):
    """
    Hook entry: reduce per-block FP32 partials under the layer's mx_specs config.

    Mode selected by mx_specs['xblock_accum_mode']:
      - 'fp32'        : plain torch.sum on last dim (baseline, no emulation)
      - 'fixed_point' : N-bit signed saturating int64-emulated accumulator
    """
    mode = _get_xblock(mx_specs, 'xblock_accum_mode', 'fp32')
    if mode not in _VALID_MODES:
        raise ValueError(f"xblock_accum_mode={mode!r} not in {_VALID_MODES}")

    if mode == "fp32":
        return partials.sum(dim=-1)

    bits = _get_xblock(mx_specs, 'xblock_accum_bits', _DEFAULT_BITS)
    saturate = _get_xblock(mx_specs, 'xblock_accum_saturate', True)
    ste_mask = _get_xblock(mx_specs, 'xblock_accum_ste_mask', False)
    backend = _get_xblock(mx_specs, 'xblock_accum_backend', 'python')

    if backend not in _VALID_BACKENDS:
        raise ValueError(f"xblock_accum_backend={backend!r} not in {_VALID_BACKENDS}")

    if backend == "triton":
        from mx_fixed_point_triton import fixed_point_accumulate_triton
        return fixed_point_accumulate_triton(
            partials, total_bits=bits, scale_exp=None, saturate=saturate, ste_mask=ste_mask
        )

    return fixed_point_accumulate(
        partials, total_bits=bits, scale_exp=None, saturate=saturate, ste_mask=ste_mask
    )
