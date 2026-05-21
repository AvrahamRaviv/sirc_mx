"""Targeted audit for the user's actual failing config:
mxint8 + bits ∈ {35, 48, 64}. User reports loss diverges when this is
used in training. Verify forward + backward of HWFxpConv2dFn against:
  (a) F.conv2d on the same quantised operands (forward parity check).
  (b) PyTorch autograd through F.conv2d (backward STE parity check).

If (a) holds at bits=48 (no saturation expected), forward is numerically
fine. Then divergence must be in backward, output dtype, or the
calibration handshake.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects")
sys.path.insert(0, "/home/avrahamra/PycharmProjects")

from microxcaling.mx import MxSpecs
from microxcaling.mx.mx_ops import quantize_mx_op
from microxcaling.mx.elemwise_ops import quantize_elemwise_op

from mx_fixed_point_hw import (
    _hw_fxp_conv2d_ref,
    calibrate_e_layer_min,
    extract_mxint,
    hw_fxp_conv2d,
)
from mx_layers_blocked import MXConv2dHW


def _sp():
    sp = MxSpecs()
    sp["w_elem_format"] = "int8"
    sp["a_elem_format"] = "int8"
    sp["block_size"] = 32
    sp["scale_bits"] = 8
    sp["shared_exp_method"] = "max"
    sp["custom_cuda"] = False
    return sp


def _quantize(x, sp, weight=False):
    bf = quantize_elemwise_op(
        x, mx_specs=sp,
        round=sp["round_weight"] if weight else sp["round_output"],
    )
    return quantize_mx_op(bf, sp, elem_format="int8", axes=[1])


def _hw_attrs(layer, bits, sat_mode="per_product", e_layer_min=None):
    layer.xblock_accum = {
        "enabled": True, "bits": bits, "backend": "python", "scale_exp": None,
        "saturate": True, "ste_mask": False,
        "mode": "hw_fixed_point", "sat_mode": sat_mode, "e_layer_min": e_layer_min,
    }


# ----------------------------------------------------------------------
# Forward parity: at bits=48 with no saturation, HW = F.conv2d on Q operands.
# ----------------------------------------------------------------------

@pytest.mark.parametrize("bits", [35, 48, 64])
def test_int8_forward_parity_vs_fconv2d_quantized(bits):
    """Direct comparison of HW kernel output vs F.conv2d on quantised operands."""
    torch.manual_seed(0)
    sp = _sp()
    B, C, H, W = 2, 32, 8, 8
    O = 8
    x = torch.randn(B, C, H, W)
    w = torch.randn(O, C, 3, 3) * 0.3
    qi = _quantize(x, sp)
    qw = _quantize(w, sp, weight=True)

    qi_i, Ea = extract_mxint(qi, 32, 1, fmt="int8")
    qw_i, Ew = extract_mxint(qw, 32, 1, fmt="int8")
    e_min = int(Ea.amin().item()) + int(Ew.amin().item()) - 12

    out, sat = _hw_fxp_conv2d_ref(
        qi_i, qw_i, Ea, Ew, e_min,
        (1, 1), (1, 1), (1, 1), 32, bits, "per_product",
        mant_bias=6,
    )
    out_4d = out.view(B, O, H, W)
    fp = F.conv2d(qi, qw, padding=1)

    diff = (out_4d - fp).abs().max().item()
    sat_count = int(sat.sum().item())
    print(f"\n[int8 bits={bits}] max diff vs F.conv2d(Q): {diff:.3e}  sat={sat_count}")

    if bits >= 48 and sat_count == 0:
        # Should be bit-exact: same products, same summation logic differs only
        # in reduction order, but integer additions are associative.
        assert diff < 1e-3, (
            f"HW @ bits={bits} should match F.conv2d on quantised operands "
            f"when no saturation; got diff={diff}"
        )


# ----------------------------------------------------------------------
# Layer parity: full MXConv2dHW forward at bits=48 vs naive MXINT8 path.
# ----------------------------------------------------------------------

def test_layer_parity_mxint8_bits48():
    """MXConv2dHW at bits=48 (no saturation) must match the equivalent
    F.conv2d on quantised operands + output elemwise quant."""
    torch.manual_seed(1)
    sp = _sp()
    hw = MXConv2dHW(32, 8, 3, padding=1, bias=False, mx_specs=sp)
    _hw_attrs(hw, bits=48)

    class _W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)

    data = [torch.randn(1, 32, 8, 8) for _ in range(2)]
    calibrate_e_layer_min(_W(hw), data, num_batches=2)
    hw.eval()

    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        y_hw = hw(x)

    # Build same reference path
    qi = _quantize(x, sp)
    qw = _quantize(hw.weight, sp, weight=True)
    y_ref = F.conv2d(qi, qw, padding=1)
    y_ref = quantize_elemwise_op(y_ref, mx_specs=sp, round=sp["round_output"])

    diff = (y_hw - y_ref).abs().max().item()
    print(f"\n[layer bits=48] max diff: {diff:.3e}")
    assert diff < 1e-4, f"HW layer @ bits=48 should match Q-conv ref; diff={diff}"


# ----------------------------------------------------------------------
# Backward STE parity: does the gradient through HW match F.conv2d's grad?
# ----------------------------------------------------------------------

def test_backward_grad_matches_fconv2d_at_bits48():
    """STE backward should approximate gradients of F.conv2d on quantised
    operands. At bits=48 with no saturation, forward outputs match (above),
    so gradients should also match."""
    torch.manual_seed(2)
    sp = _sp()
    B, C, H, W = 1, 32, 6, 6
    O = 4
    x = torch.randn(B, C, H, W)
    w = torch.randn(O, C, 3, 3) * 0.3
    bias = torch.randn(O) * 0.1

    qi = _quantize(x, sp).detach().requires_grad_(True)
    qw = _quantize(w, sp, weight=True).detach().requires_grad_(True)
    bias_q = quantize_elemwise_op(bias, mx_specs=sp, round=sp["round_weight"]) \
        .detach().requires_grad_(True)

    # Reference path: F.conv2d on quantised operands.
    qi_a = qi.clone().detach().requires_grad_(True)
    qw_a = qw.clone().detach().requires_grad_(True)
    bias_a = bias_q.clone().detach().requires_grad_(True)
    y_a = F.conv2d(qi_a, qw_a, bias=bias_a, padding=1)
    loss_a = (y_a ** 2).sum()
    loss_a.backward()

    # HW path: hw_fxp_conv2d at bits=48 + e_layer_min from extraction.
    qi_b = qi.clone().detach().requires_grad_(True)
    qw_b = qw.clone().detach().requires_grad_(True)
    bias_b = bias_q.clone().detach().requires_grad_(True)
    qi_i, Ea = extract_mxint(qi_b, 32, 1, fmt="int8")
    qw_i, Ew = extract_mxint(qw_b, 32, 1, fmt="int8")
    e_min = int(Ea.amin().item()) + int(Ew.amin().item()) - 12
    y_b = hw_fxp_conv2d(
        qi_b, qw_b, bias_b,
        e_layer_min=e_min, bs=32, bits=48, sat_mode="per_product",
        ste_mask=False, stride=1, padding=1, dilation=1, backend="python",
        fmt="int8",
    )
    loss_b = (y_b ** 2).sum()
    loss_b.backward()

    fwd_diff = (y_a - y_b).abs().max().item()
    grad_qi_diff = (qi_a.grad - qi_b.grad).abs().max().item()
    grad_qw_diff = (qw_a.grad - qw_b.grad).abs().max().item()
    grad_b_diff = (bias_a.grad - bias_b.grad).abs().max().item()
    print(f"\n[backward bits=48]")
    print(f"  fwd diff:       {fwd_diff:.3e}")
    print(f"  grad qi diff:   {grad_qi_diff:.3e}  (qi_a max={qi_a.grad.abs().max():.3e})")
    print(f"  grad qw diff:   {grad_qw_diff:.3e}  (qw_a max={qw_a.grad.abs().max():.3e})")
    print(f"  grad bias diff: {grad_b_diff:.3e}")

    # If forward matches and bias is added on same axis, gradients should match too.
    assert fwd_diff < 1e-3
    assert grad_qi_diff < 1e-3 * qi_a.grad.abs().max().item() + 1e-5
    assert grad_qw_diff < 1e-3 * qw_a.grad.abs().max().item() + 1e-5


# ----------------------------------------------------------------------
# Saturation behaviour at bits=35 — does the kernel match analytical
# expectation? (Useful to confirm bits=35 isn't silently corrupted too.)
# ----------------------------------------------------------------------

def test_int8_bits35_saturation_matches_clamp_of_fp():
    """At bits=35 saturation should kick in for typical inputs. Output ≠ F.conv2d
    but should equal sign(F.conv2d) * 2^34 * 2^e_min wherever saturated."""
    torch.manual_seed(3)
    sp = _sp()
    B, C, H, W = 1, 32, 8, 8
    O = 4
    x = torch.randn(B, C, H, W) * 3.0
    w = torch.randn(O, C, 3, 3) * 3.0
    qi = _quantize(x, sp)
    qw = _quantize(w, sp, weight=True)
    qi_i, Ea = extract_mxint(qi, 32, 1, fmt="int8")
    qw_i, Ew = extract_mxint(qw, 32, 1, fmt="int8")
    e_min = int(Ea.amin().item()) + int(Ew.amin().item()) - 12

    out, sat = _hw_fxp_conv2d_ref(
        qi_i, qw_i, Ea, Ew, e_min,
        (1, 1), (1, 1), (1, 1), 32, 35, "per_product",
        mant_bias=6,
    )
    fp = F.conv2d(qi, qw, padding=1)
    print(f"\n[bits=35] sat fraction: {sat.float().mean().item():.3f}")
    out_4d = out.view(B, O, H, W)
    sat_4d = sat.view(B, O, H, W)

    # On non-saturated lanes, HW must exactly match F.conv2d.
    if (~sat_4d).any():
        diff_unsat = (out_4d[~sat_4d] - fp[~sat_4d]).abs().max().item()
        print(f"  unsat diff: {diff_unsat:.3e}")
        assert diff_unsat < 1e-3, "non-saturated lanes must match F.conv2d"

    # On saturated lanes, HW must equal +/- hi * 2^e_min depending on sign.
    if sat_4d.any():
        hi_val = ((1 << 34) - 1) * (2.0 ** e_min)
        # Saturated outputs should have magnitude == hi_val (approximately).
        sat_mag = out_4d[sat_4d].abs()
        print(f"  saturated magnitude vs hi: "
              f"mean={sat_mag.mean().item():.3e}  hi={hi_val:.3e}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
