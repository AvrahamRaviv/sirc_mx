"""
SimpleNet: Test network and tests for MXQuantizer.

Covers:
- Conv2d replacement
- Linear replacement
- Weight/bias preservation after replacement
- Global specs config
- Per-layer specs config
- Group specs config
- Layers not in config are untouched
- Unknown group raises ValueError
- Missing config returns original model
"""

import os
import sys
import json
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('/Users/avrahamraviv/PycharmProjects')

from microxcaling.mx.convolution import Conv2d as MXConv2d
from microxcaling.mx.linear import Linear as MXLinear
from mx_quantizer import MXQuantizer


# =============================================================================
# Network
# =============================================================================

class SimpleNet(nn.Module):
    """Small CNN+MLP covering both Conv2d and Linear layers."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =============================================================================
# Helpers
# =============================================================================

# Minimal specs that work on CPU (no custom CUDA kernels)
_CPU_SPECS = {"w_elem_format": "fp8_e4m3", "a_elem_format": "fp8_e4m3",
              "block_size": 32, "custom_cuda": False}


def _make_quantizer(config_dict, tmp_dir):
    with open(os.path.join(tmp_dir, "mx_config.json"), "w") as f:
        json.dump(config_dict, f)
    return MXQuantizer(save_dir=tmp_dir)


# =============================================================================
# Tests
# =============================================================================

class TestConv2dReplacement(unittest.TestCase):

    def setUp(self):
        self.model = SimpleNet()
        self.tmp = tempfile.mkdtemp()

    def test_single_conv_replaced(self):
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["conv1"]}, self.tmp)
        m = q.quant(self.model)
        self.assertIsInstance(m.conv1, MXConv2d)

    def test_all_convs_replaced(self):
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["conv1", "conv2"]}, self.tmp)
        m = q.quant(self.model)
        self.assertIsInstance(m.conv1, MXConv2d)
        self.assertIsInstance(m.conv2, MXConv2d)

    def test_conv_not_in_config_untouched(self):
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["conv1"]}, self.tmp)
        m = q.quant(self.model)
        # conv2 was not listed — must stay as plain Conv2d
        self.assertNotIsInstance(m.conv2, MXConv2d)
        self.assertIsInstance(m.conv2, nn.Conv2d)

    def test_conv_hyperparams_preserved(self):
        orig = self.model.conv1
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["conv1"]}, self.tmp)
        m = q.quant(self.model)
        new = m.conv1
        self.assertEqual(new.in_channels,  orig.in_channels)
        self.assertEqual(new.out_channels, orig.out_channels)
        self.assertEqual(new.kernel_size,  orig.kernel_size)
        self.assertEqual(new.stride,       orig.stride)
        self.assertEqual(new.padding,      orig.padding)


class TestLinearReplacement(unittest.TestCase):

    def setUp(self):
        self.model = SimpleNet()
        self.tmp = tempfile.mkdtemp()

    def test_single_linear_replaced(self):
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["fc1"]}, self.tmp)
        m = q.quant(self.model)
        self.assertIsInstance(m.fc1, MXLinear)

    def test_all_linears_replaced(self):
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["fc1", "fc2"]}, self.tmp)
        m = q.quant(self.model)
        self.assertIsInstance(m.fc1, MXLinear)
        self.assertIsInstance(m.fc2, MXLinear)

    def test_linear_not_in_config_untouched(self):
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["fc1"]}, self.tmp)
        m = q.quant(self.model)
        self.assertNotIsInstance(m.fc2, MXLinear)
        # MXLinear extends nn.Linear, so check it's the base class
        self.assertIsInstance(m.fc2, nn.Linear)

    def test_linear_hyperparams_preserved(self):
        orig = self.model.fc1
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["fc1"]}, self.tmp)
        m = q.quant(self.model)
        new = m.fc1
        self.assertEqual(new.in_features,  orig.in_features)
        self.assertEqual(new.out_features, orig.out_features)


class TestWeightPreservation(unittest.TestCase):

    def setUp(self):
        self.model = SimpleNet()
        self.tmp = tempfile.mkdtemp()

    def test_conv_weight_preserved(self):
        orig_w = self.model.conv1.weight.data.clone()
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["conv1"]}, self.tmp)
        m = q.quant(self.model)
        self.assertTrue(torch.equal(m.conv1.weight.data, orig_w))

    def test_conv_bias_preserved(self):
        orig_b = self.model.conv1.bias.data.clone()
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["conv1"]}, self.tmp)
        m = q.quant(self.model)
        self.assertTrue(torch.equal(m.conv1.bias.data, orig_b))

    def test_linear_weight_preserved(self):
        orig_w = self.model.fc1.weight.data.clone()
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["fc1"]}, self.tmp)
        m = q.quant(self.model)
        self.assertTrue(torch.equal(m.fc1.weight.data, orig_w))

    def test_linear_bias_preserved(self):
        orig_b = self.model.fc1.bias.data.clone()
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["fc1"]}, self.tmp)
        m = q.quant(self.model)
        self.assertTrue(torch.equal(m.fc1.bias.data, orig_b))

    def test_original_model_not_modified(self):
        """quant() deep-copies, so original must be untouched."""
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["conv1", "fc1"]}, self.tmp)
        q.quant(self.model)
        self.assertNotIsInstance(self.model.conv1, MXConv2d)
        self.assertNotIsInstance(self.model.fc1, MXLinear)


class TestConfigModes(unittest.TestCase):

    def setUp(self):
        self.model = SimpleNet()
        self.tmp = tempfile.mkdtemp()

    # --- global specs ---

    def test_global_specs_string_layers(self):
        q = _make_quantizer({"mx_specs": _CPU_SPECS, "layers": ["conv1", "fc1"]}, self.tmp)
        m = q.quant(self.model)
        self.assertIsInstance(m.conv1, MXConv2d)
        self.assertIsInstance(m.fc1, MXLinear)

    def test_global_specs_dict_layers(self):
        q = _make_quantizer({
            "mx_specs": _CPU_SPECS,
            "layers": [{"name": "conv1"}, {"name": "fc1"}]
        }, self.tmp)
        m = q.quant(self.model)
        self.assertIsInstance(m.conv1, MXConv2d)
        self.assertIsInstance(m.fc1, MXLinear)

    # --- per-layer specs ---

    def test_per_layer_specs_override(self):
        """Each layer can have its own spec dict."""
        q = _make_quantizer({
            "layers": [
                {"name": "conv1", "mx_specs": {**_CPU_SPECS, "w_elem_format": "fp8_e4m3"}},
                {"name": "fc1",   "mx_specs": {**_CPU_SPECS, "w_elem_format": "int8"}},
            ]
        }, self.tmp)
        m = q.quant(self.model)
        self.assertIsInstance(m.conv1, MXConv2d)
        self.assertIsInstance(m.fc1, MXLinear)

    def test_per_layer_overrides_global(self):
        """Per-layer mx_specs take priority over global mx_specs."""
        q = _make_quantizer({
            "mx_specs": {**_CPU_SPECS, "block_size": 32},
            "layers": [
                {"name": "conv1", "mx_specs": {**_CPU_SPECS, "block_size": 16}},
            ]
        }, self.tmp)
        layer_map = q._build_layer_map()
        self.assertEqual(layer_map["conv1"]["block_size"], 16)

    # --- group specs ---

    def test_group_conv(self):
        q = _make_quantizer({
            "groups": {"fp8": _CPU_SPECS},
            "layers": [{"name": "conv1", "group": "fp8"},
                       {"name": "conv2", "group": "fp8"}]
        }, self.tmp)
        m = q.quant(self.model)
        self.assertIsInstance(m.conv1, MXConv2d)
        self.assertIsInstance(m.conv2, MXConv2d)

    def test_group_linear(self):
        q = _make_quantizer({
            "groups": {"fp8": _CPU_SPECS},
            "layers": [{"name": "fc1", "group": "fp8"},
                       {"name": "fc2", "group": "fp8"}]
        }, self.tmp)
        m = q.quant(self.model)
        self.assertIsInstance(m.fc1, MXLinear)
        self.assertIsInstance(m.fc2, MXLinear)

    def test_group_mixed_types(self):
        q = _make_quantizer({
            "groups": {
                "high": {**_CPU_SPECS, "w_elem_format": "fp8_e4m3"},
                "low":  {**_CPU_SPECS, "w_elem_format": "int8"},
            },
            "layers": [
                {"name": "conv1", "group": "high"},
                {"name": "conv2", "group": "low"},
                {"name": "fc1",   "group": "high"},
            ]
        }, self.tmp)
        m = q.quant(self.model)
        self.assertIsInstance(m.conv1, MXConv2d)
        self.assertIsInstance(m.conv2, MXConv2d)
        self.assertIsInstance(m.fc1,   MXLinear)
        # fc2 not in config — untouched
        self.assertNotIsInstance(m.fc2, MXLinear)

    def test_group_specs_in_layer_map(self):
        """Group specs correctly propagate into the layer_map."""
        config = {
            "groups": {"fp8": {**_CPU_SPECS, "block_size": 64}},
            "layers": [{"name": "conv1", "group": "fp8"}]
        }
        q = _make_quantizer(config, self.tmp)
        layer_map = q._build_layer_map()
        self.assertEqual(layer_map["conv1"]["block_size"], 64)

    def test_per_layer_overrides_group(self):
        """Explicit mx_specs on a layer takes priority over its group."""
        config = {
            "groups": {"fp8": {**_CPU_SPECS, "block_size": 32}},
            "layers": [
                {"name": "conv1", "group": "fp8",
                 "mx_specs": {**_CPU_SPECS, "block_size": 8}}
            ]
        }
        q = _make_quantizer(config, self.tmp)
        layer_map = q._build_layer_map()
        self.assertEqual(layer_map["conv1"]["block_size"], 8)

    def test_unknown_group_raises(self):
        q = _make_quantizer({
            "groups": {"fp8": _CPU_SPECS},
            "layers": [{"name": "conv1", "group": "does_not_exist"}]
        }, self.tmp)
        with self.assertRaises(ValueError):
            q.quant(self.model)

    # --- no config ---

    def test_no_config_returns_model_unchanged(self):
        tmp = tempfile.mkdtemp()  # empty dir — no mx_config.json
        q = MXQuantizer(save_dir=tmp)
        m = q.quant(self.model)
        self.assertNotIsInstance(m.conv1, MXConv2d)
        self.assertNotIsInstance(m.fc1, MXLinear)


class TestPTQ(unittest.TestCase):
    """Tests for GPTQ-style PTQ via the 'ptq' config flag and data argument."""

    # Calibration data in three formats to exercise the unpacking logic
    _DATA_TENSOR = [torch.randn(4, 3, 8, 8) for _ in range(3)]
    _DATA_TUPLE  = [(torch.randn(4, 3, 8, 8),) for _ in range(3)]   # (input,) tuple
    _DATA        = _DATA_TENSOR   # default used by most tests

    def setUp(self):
        self.model = SimpleNet()
        self.tmp = tempfile.mkdtemp()

    def _config(self, extra=None):
        cfg = {"mx_specs": _CPU_SPECS, "layers": ["conv1", "conv2", "fc1", "fc2"]}
        if extra:
            cfg.update(extra)
        return cfg

    def test_ptq_changes_weights(self):
        """GPTQ modifies at least some weights compared to plain quantization."""
        q = _make_quantizer(self._config(), self.tmp)
        m_no_ptq = q.quant(self.model)
        m_ptq    = q.quant(self.model, data=self._DATA)
        # At least one quantized layer must have different weights
        changed = any(
            not torch.equal(
                getattr(m_no_ptq, name).weight.data,
                getattr(m_ptq,    name).weight.data,
            )
            for name in ("conv1", "conv2", "fc1", "fc2")
        )
        self.assertTrue(changed, "PTQ should alter weights via GPTQ reconstruction")

    def test_ptq_false_skips_reconstruction(self):
        """'ptq': false must skip PTQ even when data is provided."""
        q_off = _make_quantizer(self._config({"ptq": False}), self.tmp)
        q_on  = _make_quantizer(self._config(),               self.tmp)

        m_off = q_off.quant(self.model, data=self._DATA)
        m_on  = q_on.quant(self.model)               # no data — no PTQ

        # Both should produce the same weights (PTQ was skipped in both cases)
        for name in ("conv1", "fc1"):
            self.assertTrue(
                torch.equal(
                    getattr(m_off, name).weight.data,
                    getattr(m_on,  name).weight.data,
                ),
                f"Layer {name}: weights differ but PTQ should have been skipped",
            )

    def test_no_data_skips_ptq(self):
        """Passing data=None must leave weights identical to the ptq=false case."""
        q = _make_quantizer(self._config(), self.tmp)
        m_no_data  = q.quant(self.model, data=None)
        m_ptq_off  = q.quant(self.model, data=self._DATA)  # with data, no ptq=false

        # m_no_data did NOT run PTQ; make a separate baseline
        m_baseline = q.quant(self.model)
        for name in ("conv1", "fc1"):
            self.assertTrue(
                torch.equal(
                    getattr(m_no_data, name).weight.data,
                    getattr(m_baseline, name).weight.data,
                ),
                f"Layer {name}: data=None should produce same result as no-data call",
            )

    def test_ptq_default_runs_when_key_absent(self):
        """No 'ptq' key in config + data given → PTQ runs by default."""
        q_default  = _make_quantizer(self._config(), self.tmp)
        q_explicit = _make_quantizer(self._config({"ptq": {"enabled": True}}), tempfile.mkdtemp())

        m_default  = q_default.quant(self.model, data=self._DATA)
        m_explicit = q_explicit.quant(self.model, data=self._DATA)

        for name in ("conv1", "fc1"):
            self.assertTrue(
                torch.equal(
                    getattr(m_default,  name).weight.data,
                    getattr(m_explicit, name).weight.data,
                ),
                f"Layer {name}: default and explicit ptq.enabled=true should be identical",
            )

    def test_tuple_batch_uses_first_element(self):
        """Default for tuple batch: model(batch[0]) — same result as plain tensor."""
        q1 = _make_quantizer(self._config(), self.tmp)
        q2 = _make_quantizer(self._config(), tempfile.mkdtemp())
        m_tensor = q1.quant(self.model, data=self._DATA_TENSOR)
        m_tuple  = q2.quant(self.model, data=self._DATA_TUPLE)
        for name in ("conv1", "fc1"):
            self.assertTrue(
                torch.equal(
                    getattr(m_tensor, name).weight.data,
                    getattr(m_tuple,  name).weight.data,
                ),
                f"Layer {name}: tuple batch[0] and plain tensor should give identical PTQ",
            )

    def test_forward_fn_is_called(self):
        """Custom forward_fn must be invoked instead of default batch handling."""
        call_count = [0]
        def fn(m, batch):
            call_count[0] += 1
            m(batch)   # DATA_TENSOR batches are plain tensors

        # disable measure_error so only PTQ calls count
        q = _make_quantizer(self._config({"measure_error": False}), self.tmp)
        q.quant(self.model, data=self._DATA_TENSOR, forward_fn=fn)
        self.assertEqual(call_count[0], len(self._DATA_TENSOR))

    def test_ptq_batches_limits_calibration(self):
        """ptq.batches=N stops after N forward passes during calibration."""
        call_count = [0]
        def counting_fn(m, b):
            call_count[0] += 1
            m(b)

        # disable measure_error so only PTQ calls count
        q = _make_quantizer(self._config({"ptq": {"batches": 2}, "measure_error": False}), self.tmp)
        q.quant(self.model, data=self._DATA_TENSOR, forward_fn=counting_fn)
        self.assertEqual(call_count[0], 2, "ptq.batches=2 should call forward exactly 2 times")

    def test_ptq_batches_zero_uses_all(self):
        """ptq.batches=0 processes every batch (no limit) — same as default when data fits."""
        q_default = _make_quantizer(self._config(),                              self.tmp)
        q_zero    = _make_quantizer(self._config({"ptq": {"batches": 0}}), tempfile.mkdtemp())
        m_default = q_default.quant(self.model, data=self._DATA)
        m_zero    = q_zero.quant(self.model,    data=self._DATA)
        for name in ("conv1", "fc1"):
            self.assertTrue(
                torch.equal(
                    getattr(m_default, name).weight.data,
                    getattr(m_zero,    name).weight.data,
                )
            )

    def test_forward_fn_overrides_default_unpacking(self):
        """forward_fn can pick any subset of a tuple batch (e.g. multi-input models)."""
        # Simulate a batch that's (input, dummy) — forward_fn ignores dummy
        data = [(torch.randn(4, 3, 8, 8), "ignored") for _ in range(3)]
        fn = lambda m, b: m(b[0])

        q = _make_quantizer(self._config(), self.tmp)
        # Should not raise even though batch[1] is a string
        m = q.quant(self.model, data=data, forward_fn=fn)
        self.assertIsInstance(m.conv1, MXConv2d)

    def test_ptq_true_explicit(self):
        """'ptq': true explicitly enables PTQ — same as the default."""
        q = _make_quantizer(self._config({"ptq": True}), self.tmp)
        m = q.quant(self.model, data=self._DATA)
        # Just verify PTQ ran (weights differ from no-data baseline)
        m_base = q.quant(self.model)
        changed = any(
            not torch.equal(
                getattr(m_base, name).weight.data,
                getattr(m,      name).weight.data,
            )
            for name in ("conv1", "fc1")
        )
        self.assertTrue(changed)


class TestMeasureError(unittest.TestCase):
    """Tests for per-layer quantization error measurement (model._quant_errors)."""

    _DATA = [torch.randn(4, 3, 8, 8) for _ in range(3)]

    def setUp(self):
        self.model = SimpleNet()
        self.tmp = tempfile.mkdtemp()

    def _config(self, extra=None):
        cfg = {"mx_specs": _CPU_SPECS, "layers": ["conv1", "conv2", "fc1", "fc2"]}
        if extra:
            cfg.update(extra)
        return cfg

    def test_returns_dict_with_mx_layer_keys(self):
        """_quant_errors has one entry per MX layer with mse/cos_sim/sqnr_db keys."""
        q = _make_quantizer(self._config(), self.tmp)
        m = q.quant(self.model, data=self._DATA)
        self.assertTrue(hasattr(m, "_quant_errors"))
        errors = m._quant_errors
        for name in ("conv1", "conv2", "fc1", "fc2"):
            self.assertIn(name, errors, f"Layer {name} missing from _quant_errors")
            for key in ("mse", "cos_sim", "sqnr_db"):
                self.assertIn(key, errors[name])

    def test_cos_sim_in_valid_range(self):
        """Cosine similarity is in [-1, 1] for every layer."""
        q = _make_quantizer(self._config(), self.tmp)
        m = q.quant(self.model, data=self._DATA)
        for name, metrics in m._quant_errors.items():
            self.assertGreaterEqual(metrics["cos_sim"], -1.0)
            self.assertLessEqual(metrics["cos_sim"], 1.0)

    def test_sqnr_is_positive(self):
        """fp8 quantization should yield SQNR > 0 dB for all layers."""
        q = _make_quantizer(self._config(), self.tmp)
        m = q.quant(self.model, data=self._DATA)
        for name, metrics in m._quant_errors.items():
            self.assertGreater(
                metrics["sqnr_db"], 0.0,
                f"Layer {name}: SQNR={metrics['sqnr_db']:.1f} dB should be > 0",
            )

    def test_no_data_skips_measurement(self):
        """Without calibration data, _quant_errors is not attached."""
        q = _make_quantizer(self._config(), self.tmp)
        m = q.quant(self.model)
        self.assertFalse(hasattr(m, "_quant_errors"))

    def test_measure_error_false_skips(self):
        """measure_error: false in config suppresses measurement even with data."""
        q = _make_quantizer(self._config({"measure_error": False}), self.tmp)
        m = q.quant(self.model, data=self._DATA)
        self.assertFalse(hasattr(m, "_quant_errors"))


class TestAutoMixedPrecision(unittest.TestCase):
    """Tests for automated per-layer precision assignment via 'auto_mixed' config."""

    _DATA = [torch.randn(4, 3, 8, 8) for _ in range(3)]

    _INT4_SPECS = {"w_elem_format": "int4", "a_elem_format": "int4",
                   "block_size": 32, "custom_cuda": False}
    _INT8_SPECS = {"w_elem_format": "int8", "a_elem_format": "int8",
                   "block_size": 32, "custom_cuda": False}

    def setUp(self):
        self.model = SimpleNet()
        self.tmp = tempfile.mkdtemp()

    def _base_config(self, strategy_extra, extra=None):
        cfg = {
            "groups": {"int4": self._INT4_SPECS, "int8": self._INT8_SPECS},
            "auto_mixed": {"base": "int4", "upgrade": "int8", **strategy_extra},
            "layers": ["conv1", "conv2", "fc1", "fc2"],
            "measure_error": False,
            "ptq": False,
        }
        if extra:
            cfg.update(extra)
        return cfg

    def test_auto_mixed_threshold_all_upgrade(self):
        """sqnr_threshold_db=1e9 forces all layers to the upgrade group (int8)."""
        cfg = self._base_config({"strategy": "threshold", "sqnr_threshold_db": 1e9})
        q = _make_quantizer(cfg, self.tmp)
        m = q.quant(self.model, data=self._DATA)
        for name in ("conv1", "conv2"):
            self.assertIsInstance(m.__getattr__(name), MXConv2d)
            self.assertEqual(m.__getattr__(name).mx_specs["w_elem_format"], "int8",
                             f"{name} should be int8 with unreachably high threshold")
        for name in ("fc1", "fc2"):
            self.assertIsInstance(m.__getattr__(name), MXLinear)
            self.assertEqual(m.__getattr__(name).mx_specs["w_elem_format"], "int8",
                             f"{name} should be int8 with unreachably high threshold")

    def test_auto_mixed_threshold_all_base(self):
        """sqnr_threshold_db=-1e9 keeps all layers in the base group (int4)."""
        cfg = self._base_config({"strategy": "threshold", "sqnr_threshold_db": -1e9})
        q = _make_quantizer(cfg, self.tmp)
        m = q.quant(self.model, data=self._DATA)
        for name in ("conv1", "conv2"):
            self.assertEqual(m.__getattr__(name).mx_specs["w_elem_format"], "int4",
                             f"{name} should be int4 with unreachably low threshold")

    def test_auto_mixed_budget_count(self):
        """upgrade_fraction=0.5 upgrades exactly half the layers to int8."""
        cfg = self._base_config({"strategy": "budget", "upgrade_fraction": 0.5})
        q = _make_quantizer(cfg, self.tmp)
        m = q.quant(self.model, data=self._DATA)
        all_mx = [(n, mod) for n, mod in m.named_modules()
                  if isinstance(mod, (MXConv2d, MXLinear))]
        n_int8 = sum(1 for _, mod in all_mx if mod.mx_specs["w_elem_format"] == "int8")
        self.assertEqual(n_int8, 2, "upgrade_fraction=0.5 of 4 layers → exactly 2 int8")

    def test_auto_mixed_without_layers_key(self):
        """auto_mixed without 'layers' key: all Conv2d/Linear discovered automatically."""
        cfg = {
            "groups": {"int4": self._INT4_SPECS, "int8": self._INT8_SPECS},
            "auto_mixed": {"base": "int4", "upgrade": "int8",
                           "strategy": "threshold", "sqnr_threshold_db": 1e9},
            "measure_error": False,
            "ptq": False,
        }
        q = _make_quantizer(cfg, self.tmp)
        m = q.quant(self.model, data=self._DATA)
        # All 4 layers of SimpleNet should have been quantized
        mx_layers = [n for n, mod in m.named_modules()
                     if isinstance(mod, (MXConv2d, MXLinear))]
        self.assertEqual(len(mx_layers), 4, "All 4 Conv2d/Linear should be MX layers")

    def test_auto_mixed_no_data_falls_back_to_base(self):
        """Without calibration data, all layers fall back to the base format."""
        cfg = self._base_config({"strategy": "threshold", "sqnr_threshold_db": 20.0})
        q = _make_quantizer(cfg, self.tmp)
        m = q.quant(self.model)   # no data
        for name in ("conv1", "conv2"):
            self.assertEqual(m.__getattr__(name).mx_specs["w_elem_format"], "int4",
                             f"{name} should default to base (int4) when no data given")

    def test_weight_only_sqnr_covers_unmeasured(self):
        """Layers whose hooks never fire get a weight-only SQNR so they appear in sensitivity."""
        # BypassNet: conv_bypassed is listed in config but never called during forward,
        # so its hook will never fire — triggering the weight-only fallback.
        class BypassNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_active   = nn.Conv2d(3, 8, 3, padding=1)
                self.conv_bypassed = nn.Conv2d(3, 8, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(8, 4)

            def forward(self, x):
                x = F.relu(self.conv_active(x))   # conv_bypassed intentionally skipped
                x = self.pool(x).flatten(1)
                return self.fc(x)

        model = BypassNet()
        cfg = {
            "groups": {"int4": self._INT4_SPECS, "int8": self._INT8_SPECS},
            "auto_mixed": {"base": "int4", "upgrade": "int8",
                           "strategy": "threshold", "sqnr_threshold_db": -1e9},
            "layers": ["conv_active", "conv_bypassed", "fc"],
            "measure_error": False,
            "ptq": False,
        }
        q = _make_quantizer(cfg, self.tmp)
        m = q.quant(model, data=self._DATA)
        # Both conv_active and conv_bypassed must be MX layers (weight-only fills the gap)
        self.assertIsInstance(m.conv_active,   MXConv2d, "conv_active should be MX")
        self.assertIsInstance(m.conv_bypassed, MXConv2d, "conv_bypassed should be MX via weight-only SQNR")


class TestPTQDirectRound(unittest.TestCase):
    """PTQ direct MX rounding for layers whose hooks never fire during calibration."""

    _INT4_SPECS = {"w_elem_format": "int4", "a_elem_format": "int4",
                   "block_size": 32, "custom_cuda": False}
    _DATA = [torch.randn(2, 3, 8, 8) for _ in range(3)]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_ptq_direct_rounds_uncalibrated(self):
        """Layers whose hooks never fire during PTQ get direct MX rounding applied."""
        class BypassNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_active   = nn.Conv2d(3, 8, 3, padding=1)
                self.conv_bypassed = nn.Conv2d(3, 8, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(8, 4)

            def forward(self, x):
                x = F.relu(self.conv_active(x))   # conv_bypassed intentionally skipped
                x = self.pool(x).flatten(1)
                return self.fc(x)

        model = BypassNet()
        # Capture the original bypassed weight before PTQ
        orig_bypassed_w = model.conv_bypassed.weight.data.clone()

        cfg = {
            "mx_specs": self._INT4_SPECS,
            "layers": ["conv_active", "conv_bypassed", "fc"],
            "ptq": {"enabled": True, "batches": 3},
            "measure_error": False,
        }
        q = _make_quantizer(cfg, self.tmp)
        m = q.quant(model, data=self._DATA)

        # conv_bypassed should be an MX layer (replaced)
        self.assertIsInstance(m.conv_bypassed, MXConv2d, "conv_bypassed must be replaced to MXConv2d")

        # Its weights must differ from FP32 original (direct-round changed them)
        self.assertFalse(
            torch.allclose(m.conv_bypassed.weight.data.float(), orig_bypassed_w.float()),
            "conv_bypassed weights should be MX-rounded (differ from original FP32)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
