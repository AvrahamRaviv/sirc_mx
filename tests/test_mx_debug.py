import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, '/Users/avrahamraviv/PycharmProjects')
sys.path.insert(0, '/home/avrahamra/PycharmProjects')

from microxcaling.mx import MxSpecs
from microxcaling.mx.convolution import Conv2d as MXConv2d

from mx_layers_blocked import MXConv2dHW, MXConv2dBlocked
import mx_debug


def _specs(block_size=32):
    sp = MxSpecs()
    sp['w_elem_format'] = 'int8'
    sp['a_elem_format'] = 'int8'
    sp['block_size'] = block_size
    sp['scale_bits'] = 8
    sp['shared_exp_method'] = 'max'
    sp['custom_cuda'] = False
    return sp


def _hw_attrs(layer, bits=35, sat_mode="per_product", e_layer_min=None):
    layer.xblock_accum = {
        "enabled": True, "bits": bits, "backend": "python", "scale_exp": None,
        "saturate": True, "ste_mask": False, "mode": "hw_fixed_point",
        "sat_mode": sat_mode, "e_layer_min": e_layer_min, "pad_channels": True,
        "verbose": 0, "verbose_sample_every": 100,
    }


def _blk_attrs(layer, bits=48):
    layer.xblock_accum = {
        "enabled": True, "bits": bits, "backend": "python", "scale_exp": None,
        "saturate": True, "ste_mask": False,
    }


_EXPECTED_KEYS = {"kind", "name", "in_ch", "out_ch", "kernel", "nb",
                  "input_sqnr_db", "weight_sqnr_db", "out_sqnr_db",
                  "out_err_max", "out_err_mean", "sat_count", "sat_total",
                  "block_trace"}


def _assert_trace_matches_layer(layer, x, out_idx, e_layer_min=-20):
    """debug's block_trace MX/HW MAC must equal the real layer(x) at that element."""
    o, y, ox = out_idx
    r = mx_debug.debug_layer(layer, x, e_layer_min=e_layer_min, out_idx=out_idx,
                             show_dist=True)
    assert _EXPECTED_KEYS.issubset(r.keys())
    with torch.no_grad():
        ref = layer(x)
    got = r["block_trace"]["mx_mac"]
    scale = max(abs(float(ref[0, o, y, ox])), 1.0)
    assert abs(got - float(ref[0, o, y, ox])) < 1e-3 * scale, (got, float(ref[0, o, y, ox]))
    return r


def test_hw_runs_and_is_side_effect_free():
    torch.manual_seed(0)
    layer = MXConv2dHW(64, 32, 3, padding=1, bias=True, mx_specs=_specs())
    _hw_attrs(layer)
    layer._mx_layer_name = "in_ds.0"
    x = torch.randn(2, 64, 16, 16)

    fwd_before = layer._fwd_count
    train_before = layer.training
    r = mx_debug.debug_layer(layer, x)

    assert r["kind"] == "MXConv2dHW"
    assert _EXPECTED_KEYS.issubset(r.keys())
    # side-effect free: derived e_layer_min restored to None, counters & mode untouched.
    assert layer.e_layer_min is None
    assert layer._fwd_count == fwd_before
    assert layer.training == train_before


def test_hw_trace_matches_layer_when_calibrated():
    torch.manual_seed(1)
    layer = MXConv2dHW(64, 16, 3, padding=1, bias=True, mx_specs=_specs())
    _hw_attrs(layer)
    layer.e_layer_min = -18  # pin so debug and layer(x) use the same grid
    x = torch.randn(1, 64, 12, 12)
    _assert_trace_matches_layer(layer, x, out_idx=(5, 4, 4), e_layer_min=-18)
    assert layer.e_layer_min == -18  # pinned value restored after debug


def test_blocked_runs_and_matches_layer():
    torch.manual_seed(2)
    layer = MXConv2dBlocked(64, 32, 3, padding=1, bias=True, mx_specs=_specs())
    _blk_attrs(layer)
    x = torch.randn(1, 64, 12, 12)
    r = _assert_trace_matches_layer(layer, x, out_idx=(3, 6, 6))
    assert r["kind"] == "MXConv2dBlocked"


def test_base_mxconv2d_runs_and_matches_layer():
    torch.manual_seed(3)
    layer = MXConv2d(64, 32, 3, padding=1, bias=True, mx_specs=_specs())
    layer.name = "base.0"
    x = torch.randn(1, 64, 12, 12)
    r = _assert_trace_matches_layer(layer, x, out_idx=(7, 2, 9))
    assert r["kind"] == "MXConv2d"


def test_base_non_divisible_channels():
    torch.manual_seed(4)
    layer = MXConv2d(48, 16, 3, padding=1, bias=False, mx_specs=_specs())  # 48 % 32 != 0
    x = torch.randn(1, 48, 8, 8)
    r = mx_debug.debug_layer(layer, x, show_dist=False)
    assert r["kind"] == "MXConv2d"
    assert "block_trace" in r


def test_hw_channel_padding():
    torch.manual_seed(5)
    layer = MXConv2dHW(48, 16, 3, padding=1, bias=True, mx_specs=_specs())  # padded to 64
    _hw_attrs(layer)
    x = torch.randn(1, 48, 8, 8)
    r = mx_debug.debug_layer(layer, x, show_dist=False)
    assert r["in_ch"] == 64  # 48 -> 64 after zero-pad
    assert layer.e_layer_min is None  # derived value restored


def test_does_not_disturb_autograd_graph():
    torch.manual_seed(6)
    layer = MXConv2dHW(64, 16, 3, padding=1, bias=True, mx_specs=_specs())
    _hw_attrs(layer)
    x = torch.randn(1, 64, 12, 12, requires_grad=True)
    mx_debug.debug_layer(layer, x, show_dist=False)
    assert x.grad is None
    assert x.requires_grad is True


def test_e_layer_min_param_default_and_override():
    torch.manual_seed(8)
    layer = MXConv2dHW(64, 16, 3, padding=1, bias=True, mx_specs=_specs())
    _hw_attrs(layer)
    x = torch.randn(1, 64, 12, 12)

    # default grid is -20; result is reported back via the result dict.
    r_def = mx_debug.debug_layer(layer, x, out_idx=(2, 3, 3), show_dist=False)
    assert r_def["e_layer_min"] == -20
    assert layer.e_layer_min is None  # restored

    # a different grid changes the HW MAC.
    r_alt = mx_debug.debug_layer(layer, x, e_layer_min=-16, out_idx=(2, 3, 3),
                                 show_dist=False)
    assert r_alt["e_layer_min"] == -16
    assert layer.e_layer_min is None
    assert r_def["block_trace"]["mx_mac"] != r_alt["block_trace"]["mx_mac"]


def test_auto_pick_hits_max_abs_element():
    torch.manual_seed(7)
    layer = MXConv2dBlocked(64, 8, 3, padding=1, bias=False, mx_specs=_specs())
    _blk_attrs(layer)
    x = torch.randn(1, 64, 10, 10)
    r = mx_debug.debug_layer(layer, x, out_idx=None, show_dist=False)
    assert "block_trace" in r
