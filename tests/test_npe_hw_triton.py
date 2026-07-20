"""GPU parity ladder for NPE fixed-point HW (Phase 2 triton kernels).

Asserts the three NPE implementations agree:
  * torch-cpu ref   (backend='python')
  * triton-reuse    (backend='triton', npe_triton_variant='reuse'; existing kernel, bs=1)
  * triton-opt      (backend='triton', npe_triton_variant='opt'; dedicated per-lane kernel)

Wide bits (no saturation) -> allclose (order-independent).
Narrow bits (real saturation) -> bit-identical (both triton paths reproduce the
ref's cin-major/kk-inner accumulation order).

Requires CUDA + triton; skipped otherwise (runs on the cluster GPU).
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects")
sys.path.insert(0, "/home/avrahamra/PycharmProjects")

from microxcaling.mx import MxSpecs
from microxcaling.mx.mx_ops import quantize_mx_op
from microxcaling.mx.elemwise_ops import quantize_elemwise_op
from microxcaling.mx.convolution import Conv2d as MXConv2d

from fixed_point.mx_fixed_point_hw import _compute_min_shift_exp, hw_fxp_conv2d

_HAS_TRITON = False
try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _HAS_TRITON),
    reason="NPE triton parity requires CUDA + triton (cluster GPU).",
)


def _specs(block_size=32):
    sp = MxSpecs()
    sp["w_elem_format"] = "int8"
    sp["a_elem_format"] = "int8"
    sp["block_size"] = block_size
    sp["scale_bits"] = 8
    sp["shared_exp_method"] = "max"
    sp["custom_cuda"] = False
    return sp


def _npe_operands(x, w, sp, bs):
    """FP32-on-lattice NPE operands: act X-blocked, weight per-filter flattened."""
    bf_in = quantize_elemwise_op(x, mx_specs=sp, round=sp["round_output"])
    qi_fp = quantize_mx_op(bf_in, sp, elem_format="int8", axes=[3], block_size=bs)
    bf_w = quantize_elemwise_op(w, mx_specs=sp, round=sp["round_weight"])
    flat = quantize_mx_op(bf_w.reshape(bf_w.shape[0], -1), sp,
                          elem_format="int8", axes=[1], block_size=bs)
    qw_fp = flat.reshape(bf_w.shape)
    return qi_fp, qw_fp


def _run(qi_fp, qw_fp, e_min, bits, backend, variant, bias=None):
    stats = {}
    out = hw_fxp_conv2d(
        qi_fp, qw_fp, bias, e_layer_min=e_min, bs=32, bits=bits,
        sat_mode="per_product", stride=1, padding=1, dilation=1,
        backend=backend, stats_sink=stats, fmt="int8",
        act_blockify="xblock", weight_blockify="flatten",
        npe_triton_variant=variant,
    )
    return out, stats


def _three_ways(qi_fp, qw_fp, bits, bias=None):
    e_min = _compute_min_shift_exp(qi_fp, qw_fp, 32, fmt="int8",
                                   act_blockify="xblock", weight_blockify="flatten")
    ref, s_ref = _run(qi_fp, qw_fp, e_min, bits, "python", "reuse", bias)
    reuse, s_re = _run(qi_fp, qw_fp, e_min, bits, "triton", "reuse", bias)
    opt, s_op = _run(qi_fp, qw_fp, e_min, bits, "triton", "opt", bias)
    return ref, reuse, opt, (s_ref, s_re, s_op)


def test_parity_wide_bits():
    torch.manual_seed(0)
    dev = "cuda"
    bs = 32
    sp = _specs(bs)
    x = torch.randn(2, 40, 8, 10, device=dev)     # Cin=40, not divisible by 32
    w = torch.randn(6, 40, 3, 3, device=dev)
    qi_fp, qw_fp = _npe_operands(x, w, sp, bs)
    ref, reuse, opt, _ = _three_ways(qi_fp, qw_fp, bits=48)
    assert torch.allclose(ref, reuse, atol=1e-4, rtol=1e-5)
    assert torch.allclose(ref, opt, atol=1e-4, rtol=1e-5)


def test_parity_saturation_bit_identical():
    torch.manual_seed(1)
    dev = "cuda"
    bs = 32
    sp = _specs(bs)
    # Large magnitudes + narrow bits -> real saturation in the accumulator.
    x = (torch.randn(2, 64, 8, 10, device=dev) * 8.0)
    w = (torch.randn(8, 64, 3, 3, device=dev) * 8.0)
    qi_fp, qw_fp = _npe_operands(x, w, sp, bs)
    ref, reuse, opt, stats = _three_ways(qi_fp, qw_fp, bits=16)
    # saturation should actually trigger (else the test is vacuous)
    assert stats[0]["sat_count"] > 0
    assert torch.equal(ref, reuse)
    assert torch.equal(ref, opt)


def test_parity_with_bias():
    torch.manual_seed(2)
    dev = "cuda"
    bs = 32
    sp = _specs(bs)
    x = torch.randn(1, 32, 6, 6, device=dev)
    w = torch.randn(4, 32, 3, 3, device=dev)
    bias = torch.randn(4, device=dev)
    qi_fp, qw_fp = _npe_operands(x, w, sp, bs)
    ref, reuse, opt, _ = _three_ways(qi_fp, qw_fp, bits=48, bias=bias)
    assert torch.allclose(ref, reuse, atol=1e-4, rtol=1e-5)
    assert torch.allclose(ref, opt, atol=1e-4, rtol=1e-5)


def test_parity_vs_float_flatten_wt():
    """NPE triton (both variants) == float flatten_wt MXConv2d at wide bits."""
    torch.manual_seed(3)
    dev = "cuda"
    bs = 32
    sp = _specs(bs)
    x = torch.randn(1, 64, 6, 8, device=dev)
    sp_f = _specs(bs)
    sp_f["flatten_wt"] = True
    sp_f["block_axes_act"] = [3]
    conv_f = MXConv2d(64, 8, 3, padding=1, bias=False, mx_specs=sp_f).to(dev)
    y_float = conv_f(x)

    qi_fp, qw_fp = _npe_operands(x, conv_f.weight.data, sp, bs)
    _, reuse, opt, _ = _three_ways(qi_fp, qw_fp, bits=48)
    assert torch.allclose(reuse, y_float, atol=1e-3, rtol=1e-4)
    assert torch.allclose(opt, y_float, atol=1e-3, rtol=1e-4)
