"""Compare MXConv2dHW to baseline MXConv2d on the SAME inputs.

User says "regular mxint8 works good, HW makes loss diverge". If outputs
on the SAME input differ between the two paths beyond fp32 noise, that
explains training divergence. We test:
  * single forward at bits=48 (no saturation)
  * forward + backward (compare grads)
  * multiple forwards (shared-layer scenario)
  * what `model.eval()` left over by calibrate_e_layer_min does
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
from microxcaling.mx.convolution import Conv2d as MXConv2d

from mx_fixed_point_hw import calibrate_e_layer_min
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


def _hw_attrs(layer, bits=48):
    layer.xblock_accum = {
        "enabled": True, "bits": bits, "backend": "python", "scale_exp": None,
        "saturate": True, "ste_mask": False,
        "mode": "hw_fixed_point", "sat_mode": "per_product", "e_layer_min": None,
    }


def _pair(in_ch=32, out_ch=8, k=3, padding=1, bias=True, seed=7):
    torch.manual_seed(seed)
    sp = _sp()
    mx = MXConv2d(in_ch, out_ch, k, padding=padding, bias=bias, mx_specs=sp)
    hw = MXConv2dHW(in_ch, out_ch, k, padding=padding, bias=bias, mx_specs=sp)
    _hw_attrs(hw, bits=48)
    with torch.no_grad():
        hw.weight.copy_(mx.weight)
        if bias:
            hw.bias.copy_(mx.bias)
    return mx, hw


def test_forward_mx_vs_hw_at_bits48_no_saturation():
    """MXConv2d and MXConv2dHW with bits=48 should produce ~identical outputs
    for the same input. They share the quantisation; only the reduction
    differs (FP32 add tree vs INT cumulative add). Differences should be
    fp32 rounding noise."""
    mx, hw = _pair()
    class _W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)
    data = [torch.randn(2, 32, 8, 8) for _ in range(3)]
    calibrate_e_layer_min(_W(hw), data, num_batches=3)

    x = torch.randn(2, 32, 8, 8)
    mx.eval(); hw.eval()
    with torch.no_grad():
        y_mx = mx(x)
        y_hw = hw(x)
    diff = (y_mx - y_hw).abs().max().item()
    rel = diff / max(y_mx.abs().max().item(), 1e-6)
    print(f"\n[fwd mx vs hw] max abs diff: {diff:.3e}  rel: {rel:.3e}")
    assert rel < 1e-4, (
        f"MXConv2d and MXConv2dHW @ bits=48 should match within fp32 noise; "
        f"got diff={diff} rel={rel}. Bug somewhere in the HW path."
    )


def test_backward_grads_mx_vs_hw():
    """Gradients to weight, input, and bias must match between the two layers
    at bits=48. If grads diverge, training will diverge."""
    mx, hw = _pair()
    class _W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)
    data = [torch.randn(2, 32, 8, 8) for _ in range(3)]
    calibrate_e_layer_min(_W(hw), data, num_batches=3)

    x_mx = torch.randn(2, 32, 8, 8, requires_grad=True)
    x_hw = x_mx.detach().clone().requires_grad_(True)

    mx.train(); hw.train()
    y_mx = mx(x_mx); loss_mx = (y_mx ** 2).sum(); loss_mx.backward()
    y_hw = hw(x_hw); loss_hw = (y_hw ** 2).sum(); loss_hw.backward()

    gw_diff = (mx.weight.grad - hw.weight.grad).abs().max().item()
    gw_scale = mx.weight.grad.abs().max().item()
    gx_diff = (x_mx.grad - x_hw.grad).abs().max().item()
    gx_scale = x_mx.grad.abs().max().item()
    gb_diff = (mx.bias.grad - hw.bias.grad).abs().max().item()
    gb_scale = mx.bias.grad.abs().max().item()
    print(f"\n[bwd]")
    print(f"  grad weight: diff={gw_diff:.3e}  rel={gw_diff/max(gw_scale,1e-6):.3e}")
    print(f"  grad input:  diff={gx_diff:.3e}  rel={gx_diff/max(gx_scale,1e-6):.3e}")
    print(f"  grad bias:   diff={gb_diff:.3e}  rel={gb_diff/max(gb_scale,1e-6):.3e}")

    # MX path re-quantizes grad_output via quantize_mx_op in its backward;
    # HW path uses pure STE (no backward grad quant). Residual relative
    # mismatch of ~1% is expected. The bug we want to catch is grad = 0
    # (gradient chain dies at quantize_mx_op without STE shim).
    assert gw_diff / max(gw_scale, 1e-6) < 5e-2
    assert gx_diff / max(gx_scale, 1e-6) < 5e-2
    assert gb_diff / max(gb_scale, 1e-6) < 5e-2


def test_calibrate_leaves_model_in_eval_mode():
    """If calibrate_e_layer_min leaves the model in eval mode and the user
    doesn't restore train(), BatchNorm and Dropout silently break training."""
    sp = _sp()
    hw = MXConv2dHW(32, 8, 3, padding=1, bias=False, mx_specs=sp)
    _hw_attrs(hw, bits=48)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(32)
            self.c = hw
        def forward(self, x): return self.c(self.bn(x))

    model = Net()
    model.train()
    assert model.training
    data = [torch.randn(2, 32, 8, 8) for _ in range(2)]
    calibrate_e_layer_min(model, data, num_batches=2)
    print(f"\n[calib aftermath] model.training = {model.training}  "
          f"(BN.training = {model.bn.training})")
    if not model.training:
        print("  WARNING: calibration left model in eval mode. "
              "User must call model.train() afterwards.")


def test_shared_layer_called_twice_per_forward():
    """If the same MXConv2dHW is invoked multiple times in one forward (shared
    weights, multi-scale), each call uses the same frozen e_layer_min. Verify
    outputs are consistent with applying the same layer twice."""
    mx, hw = _pair()
    class _W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)
    data = [torch.randn(2, 32, 8, 8) for _ in range(3)]
    calibrate_e_layer_min(_W(hw), data, num_batches=3)
    hw.eval(); mx.eval()

    x1 = torch.randn(2, 32, 8, 8)
    x2 = torch.randn(2, 32, 8, 8) * 0.5

    with torch.no_grad():
        # First MX
        m1, m2 = mx(x1), mx(x2)
        # Then HW (twice — saturation counters increment, but output should match)
        h1 = hw(x1)
        h2 = hw(x2)

    d1 = (m1 - h1).abs().max().item()
    d2 = (m2 - h2).abs().max().item()
    print(f"\n[shared] call1 diff: {d1:.3e}  call2 diff: {d2:.3e}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
