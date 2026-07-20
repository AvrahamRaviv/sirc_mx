"""Tests for NPE blockify in the fixed-point HW accumulator (Phase 1, torch ref).

NPE = activation blocked along X (width, axis=3) + weight flattened per filter
([Cin,kH,kW] -> 1D) then blocked. Selected via xblock_accum keys
weight_blockify='flatten' + act_blockify='xblock'. Covers:
  * config validation of the new keys + supported-combo guard
  * extract_mxint_flatten / extract_mxint_xblock lattice round-trip
  * NPE HW (wide bits, per_product) == plain conv of the same lattice operands
  * NPE HW == float NPE fake-quant (flatten_wt + block_axes_act=[3]) on MXConv2d
  * e_layer_min compute/calibrate for NPE
  * per_block + NPE raises
  * backward compat: channel-mode default path unchanged
  * end-to-end MXQuantizer with a non-divisible Cin NPE conv
"""

import json
import os
import sys
import tempfile

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
from microxcaling.mx.convolution import Conv2d as MXConv2d

from fixed_point.mx_fixed_point import normalize_xblock_accum
from fixed_point.mx_fixed_point_hw import (
    MANTISSA_BIAS,
    _compute_min_shift_exp,
    _hw_fxp_conv2d_ref,
    extract_mxint,
    extract_mxint_flatten,
    extract_mxint_xblock,
    hw_fxp_conv2d,
)
from mx_layers_blocked import MXConv2dHW


def _specs(block_size=32):
    sp = MxSpecs()
    sp["w_elem_format"] = "int8"
    sp["a_elem_format"] = "int8"
    sp["block_size"] = block_size
    sp["scale_bits"] = 8
    sp["shared_exp_method"] = "max"
    sp["custom_cuda"] = False
    return sp


def _act_xblock_fp(x, sp, bs):
    """FP32-on-lattice activation, blocked along W (axis=3)."""
    bf = quantize_elemwise_op(x, mx_specs=sp, round=sp["round_output"])
    return quantize_mx_op(bf, sp, elem_format="int8", axes=[3], block_size=bs)


def _wt_flatten_fp(w, sp, bs):
    """FP32-on-lattice weight, per-filter flattened then blocked."""
    bf = quantize_elemwise_op(w, mx_specs=sp, round=sp["round_weight"])
    flat = quantize_mx_op(bf.reshape(bf.shape[0], -1), sp,
                          elem_format="int8", axes=[1], block_size=bs)
    return flat.reshape(bf.shape)


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------

def test_defaults_blockify_channel():
    cfg = normalize_xblock_accum(True)
    assert cfg["weight_blockify"] == "channel"
    assert cfg["act_blockify"] == "channel"


def test_npe_combo_valid():
    cfg = normalize_xblock_accum({
        "mode": "hw_fixed_point",
        "weight_blockify": "flatten", "act_blockify": "xblock",
    })
    assert cfg["weight_blockify"] == "flatten"
    assert cfg["act_blockify"] == "xblock"


def test_bad_blockify_value_raises():
    with pytest.raises(ValueError):
        normalize_xblock_accum({"weight_blockify": "bogus"})
    with pytest.raises(ValueError):
        normalize_xblock_accum({"act_blockify": "bogus"})


def test_unsupported_combo_raises():
    # flatten weight without xblock act (and vice versa) is not supported in P1.
    with pytest.raises(ValueError):
        normalize_xblock_accum({"weight_blockify": "flatten"})
    with pytest.raises(ValueError):
        normalize_xblock_accum({"act_blockify": "xblock"})


# ----------------------------------------------------------------------
# Extract lattice round-trip
# ----------------------------------------------------------------------

def test_extract_flatten_roundtrip():
    torch.manual_seed(0)
    sp = _specs(block_size=8)
    w = torch.randn(6, 4, 3, 3)             # Cin*9 = 36, not a multiple of 8
    qw_fp = _wt_flatten_fp(w, sp, bs=8)
    q, E = extract_mxint_flatten(qw_fp, bs=8, fmt="int8")
    recon = q.to(torch.float32) * torch.pow(2.0, E.to(torch.float32) - MANTISSA_BIAS)
    assert q.shape == w.shape and E.shape == w.shape
    assert torch.allclose(recon, qw_fp, atol=1e-6)


def test_extract_xblock_roundtrip():
    torch.manual_seed(0)
    sp = _specs(block_size=8)
    x = torch.randn(2, 5, 4, 20)            # W=20, not a multiple of 8
    qi_fp = _act_xblock_fp(x, sp, bs=8)
    q, E = extract_mxint_xblock(qi_fp, bs=8, fmt="int8")
    recon = q.to(torch.float32) * torch.pow(2.0, E.to(torch.float32) - MANTISSA_BIAS)
    assert q.shape == x.shape and E.shape == x.shape
    assert torch.allclose(recon, qi_fp, atol=1e-6)


# ----------------------------------------------------------------------
# NPE kernel correctness
# ----------------------------------------------------------------------

def test_npe_hw_matches_plain_conv_wide_bits():
    """NPE HW (wide bits, per_product, no bias/sat) == plain conv of the same
    lattice operands: the fixed-point accumulation is exact up to fp32."""
    torch.manual_seed(0)
    bs = 32
    sp = _specs(block_size=bs)
    x = torch.randn(2, 40, 8, 10)           # Cin=40 (not divisible by 32)
    w = torch.randn(6, 40, 3, 3)
    qi_fp = _act_xblock_fp(x, sp, bs)
    qw_fp = _wt_flatten_fp(w, sp, bs)

    e_min = _compute_min_shift_exp(qi_fp, qw_fp, bs, fmt="int8",
                                   act_blockify="xblock", weight_blockify="flatten")
    out = hw_fxp_conv2d(
        qi_fp, qw_fp, None, e_layer_min=e_min, bs=bs, bits=48,
        sat_mode="per_product", stride=1, padding=1, dilation=1,
        backend="python", fmt="int8",
        act_blockify="xblock", weight_blockify="flatten",
    )
    ref = F.conv2d(qi_fp, qw_fp, None, stride=1, padding=1)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-4)


def test_npe_hw_matches_float_flatten_wt_mxconv2d():
    """NPE HW == the float NPE fake-quant (MXConv2d flatten_wt + block_axes_act=[3])
    forward at wide bits (both consume the same NPE-quantized operands)."""
    torch.manual_seed(1)
    bs = 32
    sp = _specs(block_size=bs)
    x = torch.randn(1, 64, 6, 8)
    conv = MXConv2d(64, 8, 3, padding=1, bias=False, mx_specs=sp)
    # float NPE path
    sp_f = _specs(block_size=bs)
    sp_f["flatten_wt"] = True
    sp_f["block_axes_act"] = [3]
    conv_f = MXConv2d(64, 8, 3, padding=1, bias=False, mx_specs=sp_f)
    conv_f.weight.data.copy_(conv.weight.data)
    y_float = conv_f(x)

    # HW NPE path on the same lattice operands
    qi_fp = _act_xblock_fp(x, sp, bs)
    qw_fp = _wt_flatten_fp(conv.weight.data, sp, bs)
    e_min = _compute_min_shift_exp(qi_fp, qw_fp, bs, fmt="int8",
                                   act_blockify="xblock", weight_blockify="flatten")
    y_hw = hw_fxp_conv2d(
        qi_fp, qw_fp, None, e_layer_min=e_min, bs=bs, bits=48,
        sat_mode="per_product", stride=1, padding=1, dilation=1,
        backend="python", fmt="int8",
        act_blockify="xblock", weight_blockify="flatten",
    )
    assert torch.allclose(y_hw, y_float, atol=1e-3, rtol=1e-4)


def test_npe_per_block_raises():
    torch.manual_seed(0)
    bs = 32
    sp = _specs(block_size=bs)
    x = torch.randn(1, 32, 5, 6)
    w = torch.randn(4, 32, 3, 3)
    qi_fp = _act_xblock_fp(x, sp, bs)
    qw_fp = _wt_flatten_fp(w, sp, bs)
    with pytest.raises(ValueError):
        hw_fxp_conv2d(
            qi_fp, qw_fp, None, e_layer_min=-10, bs=bs, bits=48,
            sat_mode="per_block", padding=1, backend="python", fmt="int8",
            act_blockify="xblock", weight_blockify="flatten",
        )


# ----------------------------------------------------------------------
# Backward compat: channel mode default path unchanged
# ----------------------------------------------------------------------

def test_channel_mode_default_unchanged():
    """Default (channel/channel) params reproduce the direct extract+ref path."""
    torch.manual_seed(0)
    bs = 32
    sp = _specs(block_size=bs)
    x = torch.randn(1, 32, 5, 6)
    w = torch.randn(4, 32, 3, 3)
    bf_in = quantize_elemwise_op(x, mx_specs=sp, round=sp["round_output"])
    bf_w = quantize_elemwise_op(w, mx_specs=sp, round=sp["round_weight"])
    qi_fp = quantize_mx_op(bf_in, sp, elem_format="int8", axes=[1])
    qw_fp = quantize_mx_op(bf_w, sp, elem_format="int8", axes=[1])

    qi_i8, Ea = extract_mxint(qi_fp, bs, axis=1, fmt="int8")
    qw_i8, Ew = extract_mxint(qw_fp, bs, axis=1, fmt="int8")
    ref, _ = _hw_fxp_conv2d_ref(qi_i8, qw_i8, Ea, Ew, -10, 1, 1, 1,
                                bs, 48, "per_product")
    B, O = 1, 4
    ref = ref.view(B, O, 5, 6)

    out = hw_fxp_conv2d(
        qi_fp, qw_fp, None, e_layer_min=-10, bs=bs, bits=48,
        sat_mode="per_product", padding=1, backend="python", fmt="int8",
    )  # defaults act/weight blockify = channel
    assert torch.equal(out, ref)


# ----------------------------------------------------------------------
# End-to-end via MXQuantizer (non-divisible Cin accepted for NPE)
# ----------------------------------------------------------------------

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 8, 3, padding=1, bias=False)  # Cin=16 (< bs=32)

    def forward(self, x):
        return self.conv(x)


def test_end_to_end_npe_non_divisible_cin():
    from mx_quantizer import MXQuantizer

    tmp = tempfile.mkdtemp()
    cfg = {
        "mx_specs": {
            "w_elem_format": "int8", "a_elem_format": "int8",
            "block_size": 32, "custom_cuda": False,
            "xblock_accum": {
                "enabled": True, "mode": "hw_fixed_point", "bits": 48,
                "backend": "python", "sat_mode": "per_product",
                "weight_blockify": "flatten", "act_blockify": "xblock",
                "e_layer_min": -20,
            },
        },
        "layers": ["conv"], "ptq": False,
    }
    with open(os.path.join(tmp, "mx_config.json"), "w") as f:
        json.dump(cfg, f)

    model = MXQuantizer(save_dir=tmp).quant(_Net())
    assert isinstance(model.conv, MXConv2dHW)   # NOT fallback despite Cin % 32 != 0
    y = model(torch.randn(2, 16, 8, 10))
    assert y.shape == (2, 8, 8, 10)
    assert torch.isfinite(y).all()
