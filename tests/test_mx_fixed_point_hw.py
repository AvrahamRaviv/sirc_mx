"""Tests for the HW-faithful fixed-point Conv2d emulation.

Covers:
  * `extract_mxint8` roundtrip (quantized FP -> int + exp -> reconstruct).
  * `_hw_fxp_conv2d_ref` parity vs `F.conv2d` on quantized operands at wide bits.
  * `sat_mode` per-product vs per-block equivalence when no saturation occurs.
  * Saturation triggers at narrow bits.
  * Config validation (mode, sat_mode, e_layer_min).
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

from mx_fixed_point import (
    XBLOCK_ACCUM_DEFAULTS,
    normalize_xblock_accum,
    validate_xblock_accum_bits,
)
from mx_fixed_point_hw import (
    MANTISSA_BIAS,
    _hw_fxp_conv2d_ref,
    calibrate_e_layer_min,
    extract_mxint8,
)
from mx_layers_blocked import MXConv2dBlocked, MXConv2dHW


def _specs(block_size=32):
    sp = MxSpecs()
    sp["w_elem_format"] = "int8"
    sp["a_elem_format"] = "int8"
    sp["block_size"] = block_size
    sp["scale_bits"] = 8
    sp["shared_exp_method"] = "max"
    sp["custom_cuda"] = False
    return sp


def _quantize(x, sp, is_weight=False):
    bf = quantize_elemwise_op(
        x, mx_specs=sp,
        round=sp["round_weight"] if is_weight else sp["round_output"],
    )
    return quantize_mx_op(bf, sp, elem_format="int8", axes=[1])


# ----------------------------------------------------------------------
# Config plumbing
# ----------------------------------------------------------------------

def test_defaults_contain_hw_keys():
    for k in ("mode", "sat_mode", "e_layer_min"):
        assert k in XBLOCK_ACCUM_DEFAULTS


def test_bits_range_includes_35():
    validate_xblock_accum_bits(35)


def test_bits_range_rejects_below_32():
    with pytest.raises(ValueError):
        validate_xblock_accum_bits(31)


def test_normalize_rejects_bad_mode():
    with pytest.raises(ValueError):
        normalize_xblock_accum({"enabled": True, "mode": "nope"})


def test_normalize_rejects_bad_sat_mode():
    with pytest.raises(ValueError):
        normalize_xblock_accum({"enabled": True, "sat_mode": "nope"})


def test_normalize_rejects_bad_e_layer_min_type():
    with pytest.raises(TypeError):
        normalize_xblock_accum({"enabled": True, "e_layer_min": 3.5})


def test_normalize_accepts_hw_config():
    cfg = normalize_xblock_accum({
        "enabled": True, "bits": 35,
        "mode": "hw_fixed_point", "sat_mode": "per_block", "e_layer_min": -4,
    })
    assert cfg["mode"] == "hw_fixed_point"
    assert cfg["sat_mode"] == "per_block"
    assert cfg["e_layer_min"] == -4


# ----------------------------------------------------------------------
# extract_mxint8 roundtrip
# ----------------------------------------------------------------------

@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("shape", [(1, 32, 4, 4), (2, 64, 8, 8), (3, 128, 5, 7)])
def test_extract_mxint8_roundtrip(shape, axis):
    torch.manual_seed(0)
    sp = _specs(block_size=32)
    x = torch.randn(*shape)
    q = _quantize(x, sp)
    q_i8, E = extract_mxint8(q, bs=32, axis=axis)
    assert q_i8.dtype == torch.int8
    assert q_i8.shape == q.shape
    assert E.dtype == torch.int16

    # reconstruct: step size is 2^(E - 6); scale expanded over bs along channel axis
    B = q.shape[0]
    C = q.shape[1]
    H, W = q.shape[2], q.shape[3]
    nb = C // 32
    scale = torch.pow(torch.tensor(2.0), E.to(torch.float32) - MANTISSA_BIAS)  # [B, nb, H, W]
    scale = scale.unsqueeze(2).expand(B, nb, 32, H, W).reshape(B, C, H, W)
    recon = q_i8.to(torch.float32) * scale
    assert torch.equal(recon, q)


def test_extract_mxint8_int_range():
    torch.manual_seed(1)
    sp = _specs(block_size=32)
    x = torch.randn(2, 64, 8, 8) * 10.0   # large magnitudes
    q = _quantize(x, sp)
    q_i8, _ = extract_mxint8(q, bs=32, axis=1)
    assert q_i8.abs().max().item() <= 127


# ----------------------------------------------------------------------
# Reference kernel parity
# ----------------------------------------------------------------------

@pytest.mark.parametrize("kernel,stride,padding", [
    (1, 1, 0),
    (3, 1, 1),
    (3, 2, 1),
    (5, 1, 2),
])
def test_ref_matches_fconv2d_at_wide_bits(kernel, stride, padding):
    """At bits=48 and no saturation, the HW fixed-point accumulator is exact,
    so the output must equal F.conv2d(qi, qw) bit-for-bit."""
    torch.manual_seed(4)
    sp = _specs(block_size=32)
    B, C, H, W = 2, 32, 10, 10
    O = 8
    x = torch.randn(B, C, H, W)
    w = torch.randn(O, C, kernel, kernel)
    qi = _quantize(x, sp)
    qw = _quantize(w, sp, is_weight=True)

    fp = F.conv2d(qi, qw, stride=stride, padding=padding)

    qi_i8, Ea = extract_mxint8(qi, 32, 1)
    qw_i8, Ew = extract_mxint8(qw, 32, 1)
    e_min = int(Ea.amin().item()) + int(Ew.amin().item()) - 2 * MANTISSA_BIAS

    out, sat = _hw_fxp_conv2d_ref(
        qi_i8, qw_i8, Ea, Ew, e_min,
        (stride, stride), (padding, padding), (1, 1),
        32, 48, "per_product",
    )
    H_out = fp.shape[2]
    W_out = fp.shape[3]
    out_4d = out.view(B, O, H_out, W_out)
    assert not sat.any()
    assert torch.equal(out_4d, fp)


def test_ref_sat_mode_agree_when_no_saturation():
    """per_product and per_block produce identical results when accumulator never saturates."""
    torch.manual_seed(5)
    sp = _specs(block_size=32)
    x = torch.randn(1, 32, 6, 6)
    w = torch.randn(4, 32, 3, 3)
    qi = _quantize(x, sp)
    qw = _quantize(w, sp, is_weight=True)
    qi_i8, Ea = extract_mxint8(qi, 32, 1)
    qw_i8, Ew = extract_mxint8(qw, 32, 1)
    e_min = int(Ea.amin().item()) + int(Ew.amin().item()) - 2 * MANTISSA_BIAS

    out_p, _ = _hw_fxp_conv2d_ref(qi_i8, qw_i8, Ea, Ew, e_min,
                                  (1, 1), (1, 1), (1, 1), 32, 48, "per_product")
    out_b, _ = _hw_fxp_conv2d_ref(qi_i8, qw_i8, Ea, Ew, e_min,
                                  (1, 1), (1, 1), (1, 1), 32, 48, "per_block")
    assert torch.equal(out_p, out_b)


def test_ref_saturation_triggers_at_narrow_bits():
    """Shrinking the accumulator bit budget must cause saturation for typical inputs."""
    torch.manual_seed(6)
    sp = _specs(block_size=32)
    x = torch.randn(1, 32, 8, 8) * 3.0
    w = torch.randn(16, 32, 3, 3) * 3.0
    qi = _quantize(x, sp)
    qw = _quantize(w, sp, is_weight=True)
    qi_i8, Ea = extract_mxint8(qi, 32, 1)
    qw_i8, Ew = extract_mxint8(qw, 32, 1)
    e_min = int(Ea.amin().item()) + int(Ew.amin().item()) - 2 * MANTISSA_BIAS

    # Shrink e_layer_min aggressively so shifts scale products past 2^31.
    # bits=32 window = ±2^31. With e_min further reduced by 25, shifts are
    # ~25 larger, lifting per-product magnitudes well past the bound.
    _, sat_narrow = _hw_fxp_conv2d_ref(
        qi_i8, qw_i8, Ea, Ew, e_min - 25,
        (1, 1), (1, 1), (1, 1), 32, 32, "per_product",
    )
    assert sat_narrow.any()


# ----------------------------------------------------------------------
# MXConv2dHW layer
# ----------------------------------------------------------------------

def _hw_attrs(layer, bits=48, e_layer_min=None, sat_mode="per_product", backend="python"):
    layer.xblock_accum = {
        "enabled": True,
        "bits": bits,
        "backend": backend,
        "scale_exp": None,
        "saturate": True,
        "ste_mask": False,
        "mode": "hw_fixed_point",
        "sat_mode": sat_mode,
        "e_layer_min": e_layer_min,
    }


def test_mxconv2dhw_requires_e_layer_min():
    sp = _specs()
    hw = MXConv2dHW(32, 8, 3, padding=1, bias=True, mx_specs=sp)
    _hw_attrs(hw, bits=48, e_layer_min=None)
    with pytest.raises(RuntimeError, match="e_layer_min is unset"):
        hw(torch.randn(1, 32, 6, 6))


def test_mxconv2dhw_rejects_non_int8_format():
    sp = _specs()
    sp["a_elem_format"] = "fp8_e4m3"
    hw = MXConv2dHW(32, 8, 3, padding=1, bias=True, mx_specs=sp)
    _hw_attrs(hw, bits=48, e_layer_min=-12)
    hw.e_layer_min = -12
    with pytest.raises(RuntimeError, match="int8"):
        hw(torch.randn(1, 32, 6, 6))


@pytest.mark.parametrize("kernel,stride,padding", [(1, 1, 0), (3, 1, 1), (3, 2, 1)])
def test_mxconv2dhw_matches_fp32_conv_at_wide_bits(kernel, stride, padding):
    """At bits=48 with e_layer_min from extraction, layer output (pre-output-quant)
    should match F.conv2d on the quantized operands."""
    torch.manual_seed(9)
    sp = _specs()
    hw = MXConv2dHW(32, 8, kernel, stride=stride, padding=padding, bias=False, mx_specs=sp)
    _hw_attrs(hw, bits=48)

    # Pre-compute e_layer_min from a dummy forward via calibration.
    data = [torch.randn(1, 32, 8, 8) for _ in range(2)]

    class _W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)

    wrapper = _W(hw)
    calibrate_e_layer_min(wrapper, data, num_batches=2)
    assert hw.e_layer_min is not None

    x = torch.randn(1, 32, 8, 8)
    hw.eval()
    with torch.no_grad():
        y_hw = hw(x)

    # Recompute the reference path manually to avoid output quant drift.
    qi = _quantize(x, sp)
    qw = _quantize(hw.weight, sp, is_weight=True)
    y_ref = F.conv2d(qi, qw, stride=stride, padding=padding)
    y_ref = quantize_elemwise_op(y_ref, mx_specs=sp, round=sp["round_output"])

    assert torch.allclose(y_hw, y_ref, atol=1e-4, rtol=1e-4), \
        f"max diff={(y_hw - y_ref).abs().max().item():.3e}"


def test_calibration_sets_e_layer_min():
    torch.manual_seed(11)
    sp = _specs()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = MXConv2dHW(32, 16, 3, padding=1, bias=True, mx_specs=sp)
            self.c2 = MXConv2dHW(16, 8, 1, bias=False, mx_specs=sp)   # 16 ch won't divide bs=32!

        def forward(self, x):
            return self.c2(self.c1(x))

    # c2 would fail divisibility; skip it by making C=16 layer divisible via bs=16 specs.
    # Simpler: two independent parallel HW layers sharing bs=32.

    class Net2(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = MXConv2dHW(32, 32, 3, padding=1, bias=False, mx_specs=sp)
            self.c2 = MXConv2dHW(32, 16, 3, padding=1, bias=True, mx_specs=sp)

        def forward(self, x):
            return self.c2(self.c1(x))

    m = Net2()
    _hw_attrs(m.c1); _hw_attrs(m.c2)

    data = [torch.randn(2, 32, 6, 6) for _ in range(4)]
    result = calibrate_e_layer_min(m, data, num_batches=4)
    assert m.c1 in result and m.c2 in result
    assert m.c1.e_layer_min is not None
    assert m.c2.e_layer_min is not None
    # After calibration, inference works without further setup.
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(1, 32, 6, 6))
    assert y.shape == (1, 16, 6, 6)
    assert torch.isfinite(y).all()


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------

def test_mxconv2dhw_pad_channels_handles_nondivisible():
    """C=3 (RGB-style first conv) with pad_channels=True must run via HW path."""
    torch.manual_seed(13)
    sp = _specs()
    hw = MXConv2dHW(3, 16, 3, padding=1, bias=True, mx_specs=sp)
    _hw_attrs(hw, bits=48, e_layer_min=-12)
    hw.e_layer_min = -12
    hw.eval()
    with torch.no_grad():
        y = hw(torch.randn(1, 3, 8, 8))
    assert y.shape == (1, 16, 8, 8)
    assert torch.isfinite(y).all()


def test_mxconv2dhw_pad_channels_disabled_raises():
    sp = _specs()
    hw = MXConv2dHW(3, 16, 3, padding=1, bias=False, mx_specs=sp)
    _hw_attrs(hw, bits=48, e_layer_min=-12)
    hw.xblock_accum["pad_channels"] = False
    hw.e_layer_min = -12
    with pytest.raises(AssertionError, match="pad_channels=True"):
        hw(torch.randn(1, 3, 8, 8))


def test_dispatch_hw_fixed_point_uses_mxconv2dhw(tmp_path):
    import json
    from mx_quantizer import MXQuantizer

    cfg = {
        "mx_specs": {
            "w_elem_format": "int8",
            "a_elem_format": "int8",
            "block_size": 32,
            "scale_bits": 8,
            "shared_exp_method": "max",
            "custom_cuda": False,
            "xblock_accum": {
                "enabled": True,
                "bits": 35,
                "backend": "python",
                "mode": "hw_fixed_point",
                "sat_mode": "per_product",
            },
        },
        "layers": [{"name": "conv"}],
        "ptq": False,
        "measure_error": False,
    }
    (tmp_path / "mx_config.json").write_text(json.dumps(cfg))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(32, 16, 3, padding=1)
        def forward(self, x): return self.conv(x)

    q = MXQuantizer(str(tmp_path))
    model = q.quant(Net())
    assert isinstance(model.conv, MXConv2dHW)
    assert not isinstance(model.conv, MXConv2dBlocked)


def test_dispatch_fp32_partial_still_uses_blocked(tmp_path):
    """mode='fp32_partial' (default) must not switch to the HW layer."""
    import json
    from mx_quantizer import MXQuantizer

    cfg = {
        "mx_specs": {
            "w_elem_format": "int8",
            "a_elem_format": "int8",
            "block_size": 32,
            "scale_bits": 8,
            "shared_exp_method": "max",
            "custom_cuda": False,
            "xblock_accum": {"enabled": True, "bits": 48, "backend": "python"},
        },
        "layers": [{"name": "conv"}],
        "ptq": False,
        "measure_error": False,
    }
    (tmp_path / "mx_config.json").write_text(json.dumps(cfg))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(32, 16, 3, padding=1)
        def forward(self, x): return self.conv(x)

    q = MXQuantizer(str(tmp_path))
    model = q.quant(Net())
    assert isinstance(model.conv, MXConv2dBlocked)
    assert not isinstance(model.conv, MXConv2dHW)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
