"""Tests for static-scale fixed-point output quantization (Q8.8 flow field)."""

import json
import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, '/Users/avrahamraviv/PycharmProjects')
sys.path.insert(0, '/home/avrahamra/PycharmProjects')

import pytest

from fixed_point.fxp_quant import (
    OUT_QUANT_DEFAULTS, fake_quant_fxp, fxp_clip_stats, fxp_format_str,
    fxp_range, fxp_stats_value, normalize_out_quant,
)
from mx_quantizer import MXQuantizer

STEP = 1.0 / 256
Q88_MAX = 127.99609375
Q88_MIN = -128.0


class FlowNet(nn.Module):
    """Stand-in for DOF: returns a 2-channel flow field."""

    def __init__(self, cin=4):
        super().__init__()
        self.conv = nn.Conv2d(cin, 2, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class Wrapper(nn.Module):
    """Mirrors the real setup, where the net sits under a `model` attribute."""

    def __init__(self):
        super().__init__()
        self.model = FlowNet()

    def forward(self, x):
        return self.model(x)


def _write_cfg(d, cfg):
    with open(os.path.join(d, "mx_config.json"), "w") as f:
        json.dump(cfg, f)


def _out_quant_cfg(**over):
    entry = {"name": "model", "kind": "out_quant"}
    entry.update(over)
    return {"ptq": False, "measure_error": False, "layers": [entry]}


# =============================================================================
# Lattice / range
# =============================================================================

def test_output_lands_on_the_1_over_256_lattice():
    x = torch.randn(4, 2, 8, 8) * 10
    q = fake_quant_fxp(x)
    codes = q / STEP
    assert torch.equal(codes, codes.round()), "values must be exact multiples of 1/256"


def test_saturates_to_the_q88_range():
    x = torch.tensor([-500.0, -128.5, -128.0, 0.0, 127.99609375, 200.0, 1e9])
    q = fake_quant_fxp(x)
    assert q.min().item() >= Q88_MIN
    assert q.max().item() <= Q88_MAX
    assert q[0].item() == Q88_MIN
    assert q[-1].item() == Q88_MAX
    assert q[3].item() == 0.0


def test_saturate_false_keeps_lattice_without_clamping():
    x = torch.tensor([1000.0, -1000.0])
    q = fake_quant_fxp(x, saturate=False)
    assert q[0].item() == 1000.0 and q[1].item() == -1000.0
    # still on-lattice, just unbounded
    assert torch.equal(q / STEP, (q / STEP).round())


def test_unsigned_range():
    x = torch.tensor([-5.0, 0.0, 300.0])
    q = fake_quant_fxp(x, signed=False)
    lo, hi, _ = fxp_range(16, 8, signed=False)
    assert q[0].item() == 0.0                      # negatives clamp to 0
    assert q[2].item() == hi / 256.0               # 255.99609375
    assert lo == 0 and hi == 65535


def test_matches_a_plain_reference_implementation():
    torch.manual_seed(0)
    x = torch.randn(1000) * 30                     # within range, no clipping
    ref = torch.floor(x.abs() * 256 + 0.5).copysign(x) / 256
    assert torch.equal(fake_quant_fxp(x), ref)


# =============================================================================
# Rounding modes
# =============================================================================

def test_round_modes_differ_on_ties():
    # exact ties after *256: odd multiples of 1/512
    x = torch.tensor([1.5, 2.5, -1.5, -2.5]) / 256

    away = fake_quant_fxp(x, round_mode="half_away") * 256
    even = fake_quant_fxp(x, round_mode="half_even") * 256
    trunc = fake_quant_fxp(x, round_mode="trunc") * 256

    assert away.tolist() == [2.0, 3.0, -2.0, -3.0]
    assert even.tolist() == [2.0, 2.0, -2.0, -2.0]
    assert trunc.tolist() == [1.0, 2.0, -1.0, -2.0]
    # the knob is genuinely live, not decorative
    assert not torch.equal(away, even)
    assert not torch.equal(away, trunc)


def test_unknown_round_mode_raises():
    with pytest.raises(ValueError, match="round_mode"):
        fake_quant_fxp(torch.randn(4), round_mode="stochastic")


# =============================================================================
# Gradients (STE)
# =============================================================================

def test_ste_passes_gradient_through():
    x = torch.randn(64, requires_grad=True) * 5
    x.retain_grad()
    fake_quant_fxp(x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x.grad))


def test_clip_grad_zeroes_saturated_elements():
    x = torch.tensor([-500.0, -1.0, 0.5, 900.0], requires_grad=True)
    fake_quant_fxp(x, clip_grad=True).sum().backward()
    assert x.grad.tolist() == [0.0, 1.0, 1.0, 0.0]


def test_forward_value_identical_with_and_without_clip_grad():
    x = torch.randn(128, requires_grad=True) * 200
    a = fake_quant_fxp(x, clip_grad=False)
    b = fake_quant_fxp(x, clip_grad=True)
    assert torch.equal(a, b), "clip_grad must only affect backward"


# =============================================================================
# Clip statistics
# =============================================================================

def test_clip_stats_counts_out_of_range_elements():
    # 3 of 10 outside [-128, 127.996]
    x = torch.tensor([0.0, 1.0, -1.0, 500.0, -500.0, 2.0,
                      3.0, 4.0, 5.0, 1e6])
    n, n_clipped, sum_sq, sum_sq_err = fxp_clip_stats(x)
    assert n == 10
    assert fxp_stats_value(n_clipped) == 3
    assert fxp_stats_value(sum_sq) > 0 and fxp_stats_value(sum_sq_err) > 0


def test_clip_stats_zero_when_in_range():
    x = torch.randn(500) * 10
    n, n_clipped, _, sum_sq_err = fxp_clip_stats(x)
    assert n == 500 and fxp_stats_value(n_clipped) == 0
    assert fxp_stats_value(sum_sq_err) > 0     # rounding error still present


def test_clip_stats_sums_stay_on_device_as_tensors():
    """Accumulating as tensors is what keeps the QAT forward sync-free —
    returning python floats here would cost a host sync every step."""
    _, n_clipped, sum_sq, sum_sq_err = fxp_clip_stats(torch.randn(64))
    for v in (n_clipped, sum_sq, sum_sq_err):
        assert torch.is_tensor(v)


# =============================================================================
# Config normalization
# =============================================================================

def test_normalize_defaults_and_shorthands():
    assert normalize_out_quant(None)["enabled"] is False
    assert normalize_out_quant(False)["enabled"] is False
    cfg = normalize_out_quant(True)
    assert cfg["enabled"] is True
    assert cfg["frac_bits"] == OUT_QUANT_DEFAULTS["frac_bits"] == 8
    assert cfg["total_bits"] == 16 and cfg["round"] == "half_away"


def test_normalize_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown out_quant key"):
        normalize_out_quant({"frac_bits": 8, "fracbits": 9})


def test_normalize_rejects_format_with_no_integer_bits():
    with pytest.raises(ValueError, match="no integer bits"):
        normalize_out_quant({"total_bits": 8, "frac_bits": 8, "signed": True})


def test_format_string():
    cfg = normalize_out_quant({"total_bits": 16, "frac_bits": 8, "signed": True})
    assert fxp_format_str(cfg) == "Q8.8 signed/16b round=half_away"


# =============================================================================
# MXQuantizer wiring
# =============================================================================

def test_quantizer_installs_hook_and_output_is_on_lattice():
    with tempfile.TemporaryDirectory() as d:
        _write_cfg(d, _out_quant_cfg())
        qm = MXQuantizer(save_dir=d).quant(Wrapper())

    assert hasattr(qm.model, "_mx_out_quant")
    out = qm(torch.randn(2, 4, 8, 8))
    codes = out / STEP
    assert torch.equal(codes, codes.round())
    assert qm.model._mx_out_quant["n_calls"] == 1
    assert qm.model._mx_out_quant["n"] == out.numel()


def test_state_dict_keys_unchanged():
    """The reason this is a hook and not a wrapper module: a wrapper would
    insert `.inner` into every parameter name and break checkpoint load."""
    fp32 = Wrapper()
    with tempfile.TemporaryDirectory() as d:
        _write_cfg(d, _out_quant_cfg())
        qm = MXQuantizer(save_dir=d).quant(fp32)

    assert set(qm.state_dict().keys()) == set(fp32.state_dict().keys())
    qm.load_state_dict(fp32.state_dict())      # must not raise


def test_gradient_survives_the_hook():
    with tempfile.TemporaryDirectory() as d:
        _write_cfg(d, _out_quant_cfg())
        qm = MXQuantizer(save_dir=d).quant(Wrapper())

    qm(torch.randn(2, 4, 8, 8)).sum().backward()
    g = qm.model.conv.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_summary_reports_clip_rate_and_sqnr():
    with tempfile.TemporaryDirectory() as d:
        _write_cfg(d, _out_quant_cfg())
        qm = MXQuantizer(save_dir=d).quant(Wrapper())

    # force every element out of range so the clip rate is exactly 100%
    with torch.no_grad():
        qm.model.conv.bias.fill_(1e4)
        qm.model.conv.weight.zero_()
    qm(torch.randn(2, 4, 8, 8))

    st = qm.model._mx_out_quant
    assert st["n"] > 0
    assert fxp_stats_value(st["n_clipped"]) == st["n"]
    summary = MXQuantizer.out_quant_summary(qm.model)
    assert "clip 100.00%" in summary
    assert "Q8.8 signed/16b" in summary


def test_tuple_output_quantizes_every_tensor():
    class TwoHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _TwoOut()

        def forward(self, x):
            return self.model(x)

    with tempfile.TemporaryDirectory() as d:
        _write_cfg(d, _out_quant_cfg())
        qm = MXQuantizer(save_dir=d).quant(TwoHead())

    a, b = qm(torch.randn(2, 4, 8, 8))
    for t in (a, b):
        assert torch.equal(t / STEP, (t / STEP).round())


def test_outputs_index_selection():
    class TwoHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _TwoOut()

        def forward(self, x):
            return self.model(x)

    with tempfile.TemporaryDirectory() as d:
        _write_cfg(d, _out_quant_cfg(outputs=[0]))
        qm = MXQuantizer(save_dir=d).quant(TwoHead())

    a, b = qm(torch.randn(2, 4, 8, 8))
    assert torch.equal(a / STEP, (a / STEP).round())
    assert not torch.equal(b / STEP, (b / STEP).round()), "input 1 must stay FP32"


class _TwoOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 2, 3, padding=1)

    def forward(self, x):
        y = self.conv(x)
        return y, y * 1.000123


def test_out_quant_entry_does_not_leak_into_the_conv_layer_map():
    cfg = _out_quant_cfg()
    cfg["mx_specs"] = {"w_elem_format": "int8", "a_elem_format": "int8",
                       "block_size": 32, "custom_cuda": False}
    cfg["layers"].append("model.conv")

    with tempfile.TemporaryDirectory() as d:
        _write_cfg(d, cfg)
        q = MXQuantizer(save_dir=d)
        layer_map = q._build_layer_map()
        qm = q.quant(Wrapper())

    assert "model" not in layer_map, "out_quant entry must not become a conv layer"
    assert "model.conv" in layer_map
    assert hasattr(qm.model.conv, "mx_specs")       # conv still MX-replaced
    assert hasattr(qm.model, "_mx_out_quant")       # hook still installed


def test_missing_module_name_warns_and_installs_nothing(capsys):
    with tempfile.TemporaryDirectory() as d:
        _write_cfg(d, _out_quant_cfg(name="does.not.exist"))
        qm = MXQuantizer(save_dir=d).quant(Wrapper())

    assert "out_quant layers not found" in capsys.readouterr().out
    assert not hasattr(qm.model, "_mx_out_quant")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
