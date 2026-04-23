import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mx_fixed_point import (
    FixedPointAccumulator,
    cross_block_accumulate_from_specs,
    fixed_point_accumulate,
)

try:
    from mx_fixed_point_triton import (
        FixedPointAccumulatorTriton,
        _TRITON_AVAILABLE,
        fixed_point_accumulate_triton,
    )
except ImportError:
    _TRITON_AVAILABLE = False


cuda_triton = pytest.mark.skipif(
    not (_TRITON_AVAILABLE and torch.cuda.is_available()),
    reason="requires CUDA + triton",
)


class _FakeSpecs(dict):
    pass


def _specs(**overrides):
    s = _FakeSpecs()
    cfg = {
        'enabled': True,
        'bits': 48,
        'backend': 'triton',
        'scale_exp': None,
        'saturate': True,
        'ste_mask': False,
    }
    cfg.update(overrides)
    s['xblock_accum'] = cfg
    return s


@cuda_triton
def test_triton_parity_no_saturation():
    torch.manual_seed(0)
    partials = (torch.randn(16, 8, 32, device='cuda') * 0.1)
    ref = fixed_point_accumulate(partials.cpu(), total_bits=48, scale_exp=20).cuda()
    out = fixed_point_accumulate_triton(partials, total_bits=48, scale_exp=20)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


@cuda_triton
def test_triton_parity_with_saturation():
    partials = torch.full((1, 8), 1.0e6, device='cuda')
    ref = fixed_point_accumulate(partials.cpu(), total_bits=40, scale_exp=30).cuda()
    out = fixed_point_accumulate_triton(partials, total_bits=40, scale_exp=30)
    assert torch.allclose(out, ref, atol=1.0)


@cuda_triton
def test_triton_hook_dispatch():
    torch.manual_seed(1)
    partials = torch.randn(4, 7, device='cuda') * 0.1
    out = cross_block_accumulate_from_specs(partials, _specs(backend='triton'))
    ref = cross_block_accumulate_from_specs(partials.cpu(), _specs(backend='python')).cuda()
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


@cuda_triton
def test_triton_backward_ste():
    torch.manual_seed(2)
    partials = torch.randn(3, 5, device='cuda', requires_grad=True)
    out = fixed_point_accumulate_triton(partials, total_bits=48, scale_exp=10)
    out.sum().backward()
    assert torch.allclose(partials.grad, torch.ones_like(partials))


@cuda_triton
def test_triton_ste_mask_zeros_saturated_grad():
    partials = torch.full((1, 4), 1.0e6, device='cuda', requires_grad=True)
    out = fixed_point_accumulate_triton(
        partials, total_bits=40, scale_exp=30, saturate=True, ste_mask=True
    )
    out.sum().backward()
    assert torch.all(partials.grad == 0.0)


def test_triton_backend_not_enabled_by_default_and_cpu_fallback_errors_loud():
    """On CPU or without triton, backend='triton' should raise clearly at call time."""
    if _TRITON_AVAILABLE and torch.cuda.is_available():
        pytest.skip("both triton and CUDA present; can't hit the error path")
    partials = torch.randn(2, 4)
    with pytest.raises(RuntimeError):
        cross_block_accumulate_from_specs(partials, _specs(backend='triton'))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
