"""Parity tests for the HW fixed-point Triton kernel.

Skips cleanly when Triton or CUDA isn't available. When both are present,
asserts bit-identical output between `_hw_fxp_conv2d_triton` and the
`_hw_fxp_conv2d_ref` path for both sat modes.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects")
sys.path.insert(0, "/home/avrahamra/PycharmProjects")

cuda_avail = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    triton_avail = True
except ImportError:
    triton_avail = False

pytestmark = pytest.mark.skipif(
    not (cuda_avail and triton_avail),
    reason="requires CUDA and triton",
)

from microxcaling.mx import MxSpecs  # noqa: E402
from microxcaling.mx.mx_ops import quantize_mx_op  # noqa: E402
from microxcaling.mx.elemwise_ops import quantize_elemwise_op  # noqa: E402

from mx_fixed_point_hw import (  # noqa: E402
    MANTISSA_BIAS,
    _hw_fxp_conv2d_ref,
    extract_mxint8,
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


def _prep(x, w, sp):
    bf_x = quantize_elemwise_op(x, mx_specs=sp, round=sp["round_output"])
    bf_w = quantize_elemwise_op(w, mx_specs=sp, round=sp["round_weight"])
    qi = quantize_mx_op(bf_x, sp, elem_format="int8", axes=[1])
    qw = quantize_mx_op(bf_w, sp, elem_format="int8", axes=[1])
    qi_i8, Ea = extract_mxint8(qi, 32, 1)
    qw_i8, Ew = extract_mxint8(qw, 32, 1)
    e_min = int(Ea.amin().item()) + int(Ew.amin().item()) - 2 * MANTISSA_BIAS
    return qi_i8, qw_i8, Ea, Ew, e_min


@pytest.mark.parametrize("kernel,stride,padding", [(1, 1, 0), (3, 1, 1), (3, 2, 1)])
@pytest.mark.parametrize("sat_mode", ["per_product", "per_block"])
def test_triton_matches_ref(kernel, stride, padding, sat_mode):
    from mx_fixed_point_hw_triton import _hw_fxp_conv2d_triton

    torch.manual_seed(3)
    sp = _specs()
    x = torch.randn(2, 32, 8, 8, device="cuda")
    w = torch.randn(8, 32, kernel, kernel, device="cuda")

    qi_i8, qw_i8, Ea, Ew, e_min = _prep(x, w, sp)

    out_ref, sat_ref = _hw_fxp_conv2d_ref(
        qi_i8.cpu(), qw_i8.cpu(), Ea.cpu(), Ew.cpu(), e_min,
        (stride, stride), (padding, padding), (1, 1), 32, 48, sat_mode,
    )
    out_tri, sat_tri = _hw_fxp_conv2d_triton(
        qi_i8, qw_i8, Ea, Ew, e_min,
        (stride, stride), (padding, padding), (1, 1), 32, 48, sat_mode,
    )
    assert torch.equal(out_ref, out_tri.cpu())
    assert torch.equal(sat_ref, sat_tri.cpu())


def test_triton_saturation_trigger():
    from mx_fixed_point_hw_triton import _hw_fxp_conv2d_triton

    torch.manual_seed(7)
    sp = _specs()
    x = torch.randn(1, 32, 6, 6, device="cuda") * 2.0
    w = torch.randn(4, 32, 3, 3, device="cuda") * 2.0
    qi_i8, qw_i8, Ea, Ew, e_min = _prep(x, w, sp)

    _, sat = _hw_fxp_conv2d_triton(
        qi_i8, qw_i8, Ea, Ew, e_min - 25,
        (1, 1), (1, 1), (1, 1), 32, 32, "per_product",
    )
    assert sat.any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
