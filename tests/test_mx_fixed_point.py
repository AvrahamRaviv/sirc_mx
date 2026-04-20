import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mx_fixed_point import (
    FixedPointAccumulator,
    cross_block_accumulate_from_specs,
    fixed_point_accumulate,
    validate_xblock_accum_bits,
)


class _FakeSpecs(dict):
    pass


def _specs(**overrides):
    s = _FakeSpecs()
    s['xblock_accum_mode'] = 'fixed_point'
    s['xblock_accum_bits'] = 48
    s['xblock_accum_saturate'] = True
    s['xblock_accum_ste_mask'] = False
    s.update(overrides)
    return s


def test_validate_bits_range():
    for b in (40, 44, 48):
        assert validate_xblock_accum_bits(b) == b
    for b in (0, 1, 16, 32, 39, 49, 64):
        with pytest.raises(ValueError):
            validate_xblock_accum_bits(b)
    with pytest.raises(TypeError):
        validate_xblock_accum_bits(48.0)


def test_matches_sum_at_48bits_small_partials():
    torch.manual_seed(0)
    partials = torch.randn(8, 16, 32) * 0.1
    ref = partials.sum(-1)
    out = fixed_point_accumulate(partials, total_bits=48, scale_exp=20)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_matches_sum_at_64bit_equivalent_no_saturate():
    torch.manual_seed(1)
    partials = torch.randn(4, 32) * 10.0
    ref = partials.sum(-1)
    out = fixed_point_accumulate(partials, total_bits=48, scale_exp=10, saturate=False)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-4)


def test_saturation_triggers_at_40bits():
    big = 1.0e6
    partials = torch.full((1, 8), big)
    out_nosat = fixed_point_accumulate(partials, total_bits=48, scale_exp=0, saturate=False)
    assert torch.allclose(out_nosat, torch.tensor([8.0 * big]))

    scale_exp = 30
    out_sat = fixed_point_accumulate(partials, total_bits=40, scale_exp=scale_exp, saturate=True)
    hi = ((1 << 39) - 1) * (2 ** -scale_exp)
    assert out_sat.item() == pytest.approx(hi, rel=0, abs=1.0)
    assert out_sat.item() < 8.0 * big


def test_ste_backward_default():
    torch.manual_seed(2)
    partials = torch.randn(3, 5, requires_grad=True)
    out = fixed_point_accumulate(partials, total_bits=48, scale_exp=10)
    out.sum().backward()
    assert partials.grad is not None
    assert torch.allclose(partials.grad, torch.ones_like(partials))


def test_ste_mask_zeros_saturated_grad():
    partials = torch.tensor([[1.0e6, 1.0e6, 1.0e6, 1.0e6]], requires_grad=True)
    out = fixed_point_accumulate(
        partials, total_bits=40, scale_exp=30, saturate=True, ste_mask=True
    )
    out.sum().backward()
    assert torch.all(partials.grad == 0.0)


def test_gradcheck_ste():
    torch.manual_seed(3)
    partials = torch.randn(2, 4, dtype=torch.float64, requires_grad=True) * 0.01

    def fn(p):
        return FixedPointAccumulator.apply(p, 48, 30, True, False)

    assert torch.autograd.gradcheck(fn, (partials,), eps=1e-4, atol=1e-3)


def test_autoscale_no_saturation_for_small_inputs():
    torch.manual_seed(4)
    partials = torch.randn(6, 12) * 0.5
    out_auto = fixed_point_accumulate(partials, total_bits=48, scale_exp=None, saturate=True)
    ref = partials.sum(-1)
    assert torch.allclose(out_auto, ref, atol=1e-4, rtol=1e-4)


def test_hook_fp32_mode_is_plain_sum():
    torch.manual_seed(5)
    partials = torch.randn(4, 7) * 100.0
    out = cross_block_accumulate_from_specs(partials, _specs(xblock_accum_mode='fp32'))
    assert torch.allclose(out, partials.sum(-1))


def test_hook_fixed_point_mode_uses_accumulator():
    torch.manual_seed(6)
    partials = torch.randn(4, 7) * 0.1
    out = cross_block_accumulate_from_specs(partials, _specs())
    assert torch.allclose(out, partials.sum(-1), atol=1e-4, rtol=1e-4)


def test_hook_invalid_mode_raises():
    with pytest.raises(ValueError):
        cross_block_accumulate_from_specs(
            torch.zeros(1, 1), _specs(xblock_accum_mode='float')
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
