"""
Tests for per-axis (cube) MX block quantization.

Covers the block_shape extension:
  - microxcaling _quantize_mx accepts a per-axis block_size list
  - list-of-1 is equivalent to a scalar block_size (backward compat)
  - a rectangular cube matches a hand-rolled per-cube MX reference
  - custom_cuda + multi-axis routes through the single-axis kernel via a
    reshape (permute cube dims -> merge -> single-axis quant); the reshape
    route is bit-identical to the direct torch multi-axis path
  - MXConv2d honors block_shape_act / block_shape_wt in forward
  - MXQuantizer wires block_shape* from config end-to-end
"""

import json
import os
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects")
sys.path.insert(0, "/home/avrahamra/PycharmProjects")

from microxcaling.mx.mx_ops import _quantize_mx, _quantize_mx_cube
from microxcaling.mx.elemwise_ops import _quantize_elemwise_core
from microxcaling.mx.formats import _get_format_params
from microxcaling.mx import MxSpecs
from microxcaling.mx.convolution import Conv2d as MXConv2d


def _cpu_specs(**kw):
    s = MxSpecs()
    s['scale_bits'] = 8
    s['w_elem_format'] = 'int8'
    s['a_elem_format'] = 'int8'
    s['block_size'] = 4
    s['shared_exp_method'] = 'max'
    s['custom_cuda'] = False
    for k, v in kw.items():
        s[k] = v
    return s


def test_list_block_size_matches_scalar():
    """A single-axis list block_size must equal the scalar path exactly."""
    torch.manual_seed(0)
    A = torch.randn(2, 8, 4, 4)
    q_scalar = _quantize_mx(A, 8, 'int8', axes=[1], block_size=4, custom_cuda=False)
    q_list = _quantize_mx(A, 8, 'int8', axes=[1], block_size=[4], custom_cuda=False)
    assert torch.equal(q_scalar, q_list)


def test_cube_matches_manual_reference():
    """A 2x4x4 cube over (C,H,W) must match a hand-rolled per-cube MX quant."""
    torch.manual_seed(0)
    A = torch.randn(2, 8, 4, 4)
    axes, bs = [1, 2, 3], [2, 4, 4]  # cube spans full H,W and 2 channels
    q = _quantize_mx(A, 8, 'int8', axes=axes, block_size=bs, custom_cuda=False)

    ebits, mbits, emax, max_norm, _ = _get_format_params('int8')
    scale_emax = 2 ** (8 - 1) - 1
    N, C, H, W = A.shape
    ref = A.clone()
    for n in range(N):
        for ct in range(0, C, 2):
            blk = A[n, ct:ct + 2, :, :]
            se = torch.floor(torch.log2(blk.abs().max())).item() - emax
            se = max(min(se, scale_emax), -scale_emax)
            scaled = blk / (2 ** se)
            eq = _quantize_elemwise_core(
                scaled, mbits, ebits, max_norm, round='nearest',
                allow_denorm=True, saturate_normals=True, custom_cuda=False)
            ref[n, ct:ct + 2, :, :] = eq * (2 ** se)

    assert torch.allclose(q, ref, atol=1e-6)


def test_cube_padding_non_divisible_axis():
    """Cube blocking pads a non-divisible axis and returns the original shape."""
    torch.manual_seed(0)
    A = torch.randn(1, 5, 3, 3)         # C=5, H=W=3 all indivisible by chosen sizes
    q = _quantize_mx(A, 8, 'int8', axes=[1, 2, 3], block_size=[2, 2, 2],
                     custom_cuda=False)
    assert q.shape == A.shape
    assert torch.isfinite(q).all()


@pytest.mark.parametrize("shape,axes,bs", [
    ((2, 8, 4, 4), [1, 2, 3], [2, 4, 4]),
    ((1, 4, 4, 4), [1, 2, 3], [2, 2, 2]),   # arbitrary cube -> ordering
    ((2, 8, 4, 4), [1, 2, 3], [4, 2, 2]),
    ((1, 6, 5, 5), [1, 2, 3], [2, 2, 2]),   # non-divisible -> padding
    ((2, 8, 4, 4), [3, 1, 2], [4, 2, 4]),   # unsorted axes
])
def test_cube_helper_matches_direct(shape, axes, bs):
    """The custom_cuda cube route (permute->merge->single-axis) is bit-identical
    to the direct torch multi-axis path. Exercised with custom_cuda=False so the
    reshape logic runs on CPU; on GPU only the inner reduction swaps to the
    kernel."""
    torch.manual_seed(0)
    A = torch.randn(*shape)
    direct = _quantize_mx(A, 8, 'int8', axes=axes, block_size=bs,
                          custom_cuda=False)
    cube = _quantize_mx_cube(A, 8, 'int8', axes=list(axes),
                             block_size=list(bs), custom_cuda=False)
    assert torch.equal(direct, cube)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="custom_cuda kernels require a CUDA device")
def test_custom_cuda_cube_matches_torch_gpu():
    """On GPU the custom_cuda cube path (single-axis kernel) matches the torch
    multi-axis path within kernel tolerance."""
    torch.manual_seed(0)
    A = torch.randn(2, 16, 8, 8, device='cuda')
    ref = _quantize_mx(A, 8, 'int8', axes=[1, 2, 3], block_size=[4, 4, 4],
                       custom_cuda=False)
    got = _quantize_mx(A, 8, 'int8', axes=[1, 2, 3], block_size=[4, 4, 4],
                       custom_cuda=True)
    assert torch.allclose(ref, got, atol=1e-4)


def test_full_axis_minus_one():
    """block_shape entry -1 spans the whole axis (one block), equal to passing
    that axis's exact length; adapts per layer."""
    torch.manual_seed(0)
    A = torch.randn(2, 32, 6, 8)  # C=32, H=6, W=8
    q_neg = _quantize_mx(A, 8, 'int8', axes=[1, 2, 3], block_size=[16, -1, -1],
                         custom_cuda=False)
    q_exp = _quantize_mx(A, 8, 'int8', axes=[1, 2, 3], block_size=[16, 6, 8],
                         custom_cuda=False)
    assert torch.equal(q_neg, q_exp)
    # all -1 == one block over the full C,H,W cube
    q_all = _quantize_mx(A, 8, 'int8', axes=[1, 2, 3], block_size=[-1, -1, -1],
                         custom_cuda=False)
    q_full = _quantize_mx(A, 8, 'int8', axes=[1, 2, 3], block_size=[32, 6, 8],
                          custom_cuda=False)
    assert torch.equal(q_all, q_full)


def test_bad_negative_block_raises():
    """Only -1 is a valid negative; other negatives error."""
    A = torch.randn(1, 8, 4, 4)
    with pytest.raises(Exception):
        _quantize_mx(A, 8, 'int8', axes=[1, 2, 3], block_size=[4, -2, -1],
                     custom_cuda=False)


def _build_conv(specs, seed=1):
    torch.manual_seed(seed)
    return MXConv2d(8, 4, 3, padding=1, bias=False, mx_specs=specs)


def test_mxconv2d_block_shape_forward():
    """MXConv2d fwd honors block_shape_act/wt: output finite and differs from
    the default channel-blocked output."""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 6, 6)
    y_default = _build_conv(_cpu_specs())(x)
    y_cube = _build_conv(_cpu_specs(
        block_axes_act=[1, 2, 3], block_shape_act=[4, 2, 2],
        block_axes_wt=[1, 2, 3], block_shape_wt=[4, 3, 3]))(x)
    assert y_cube.shape == y_default.shape
    assert torch.isfinite(y_cube).all()
    assert not torch.allclose(y_default, y_cube)


def test_mxconv2d_block_shape_scalar_equiv_default():
    """block_shape equal to scalar block_size on the default axis reproduces
    the plain channel-blocked result."""
    torch.manual_seed(0)
    x = torch.randn(1, 8, 6, 6)
    y_default = _build_conv(_cpu_specs())(x)
    y_shape = _build_conv(_cpu_specs(block_axes_act=[1], block_shape_act=[4],
                                     block_axes_wt=[1], block_shape_wt=[4]))(x)
    assert torch.equal(y_default, y_shape)


def test_quantizer_end_to_end_cube():
    """MXQuantizer plumbs block_shape* from JSON config through to a working
    forward pass."""
    from simple_net import SimpleNet
    from mx_quantizer import MXQuantizer

    tmp = tempfile.mkdtemp()
    cfg = {
        "mx_specs": {
            "w_elem_format": "int8", "a_elem_format": "int8",
            "block_size": 4, "custom_cuda": False,
            "block_axes_act": [1, 2, 3], "block_shape_act": [4, 2, 2],
            "block_axes_wt": [1, 2, 3], "block_shape_wt": [4, 3, 3],
        },
        "layers": ["conv2"], "ptq": False, "measure_error": False,
    }
    with open(os.path.join(tmp, "mx_config.json"), "w") as f:
        json.dump(cfg, f)

    model = MXQuantizer(save_dir=tmp).quant(SimpleNet())
    assert isinstance(model.conv2, MXConv2d)
    y = model(torch.randn(2, 3, 8, 8))
    assert y.shape == (2, 10)
    assert torch.isfinite(y).all()
