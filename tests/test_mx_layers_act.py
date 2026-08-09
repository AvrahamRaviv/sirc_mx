"""Tests for MXActQuant: activation-only MX quant of parameter-free ops (e.g. warp)."""

import json
import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, '/Users/avrahamraviv/PycharmProjects')
sys.path.insert(0, '/home/avrahamra/PycharmProjects')

from microxcaling.mx import MxSpecs
from microxcaling.mx.mx_ops import quantize_mx_op
from microxcaling.mx.elemwise_ops import quantize_elemwise_op

from mx_layers_act import MXActQuant
from mx_quantizer import MXQuantizer
import mx_stats


def _specs(fmt='int8', block_size=32):
    sp = MxSpecs()
    sp['w_elem_format'] = fmt
    sp['a_elem_format'] = fmt
    sp['block_size'] = block_size
    sp['scale_bits'] = 8
    sp['shared_exp_method'] = 'max'
    sp['custom_cuda'] = False
    return sp


class Warp(nn.Module):
    """Parameter-free op taking (features, flow); grid_sample-style resample."""

    def forward(self, x, flow):
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(torch.arange(H, dtype=x.dtype),
                                torch.arange(W, dtype=x.dtype), indexing='ij')
        gx = (xx + flow[:, 0]) / max(W - 1, 1) * 2 - 1
        gy = (yy + flow[:, 1]) / max(H - 1, 1) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)
        return F.grid_sample(x, grid, align_corners=True)


class Net(nn.Module):
    """warp used twice (same instance), as in the target model."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 8, 3, padding=1)
        self.warp = Warp()

    def forward(self, x, flow):
        y = self.warp(x, flow)
        y = self.conv(y)
        return self.warp(y, flow)


def _ref_quant(x, sp, axes):
    bf = quantize_elemwise_op(x, mx_specs=sp, round=sp['round_output'])
    return quantize_mx_op(bf, sp, elem_format=sp['a_elem_format'], axes=axes,
                          round=sp['round_mx_output'])


# =============================================================================
# MXActQuant unit behaviour
# =============================================================================

def test_both_inputs_quantized_with_own_specs_and_axes():
    torch.manual_seed(0)
    x = torch.randn(2, 8, 6, 6)
    flow = torch.randn(2, 2, 6, 6)

    sp_x, sp_f = _specs('int8', 32), _specs('int4', 16)
    seen = {}

    class Spy(nn.Module):
        def forward(self, a, b):
            seen['a'], seen['b'] = a, b
            return a

    m = MXActQuant(Spy(), [sp_x, sp_f], [[1], [-1]])
    m(x, flow)

    assert torch.equal(seen['a'], _ref_quant(x, sp_x, [1]))
    assert torch.equal(seen['b'], _ref_quant(flow, sp_f, [-1]))
    # quantization actually changed the operands (int4 flow especially)
    assert not torch.equal(seen['a'], x)
    assert not torch.equal(seen['b'], flow)


def test_none_spec_and_extra_args_pass_through():
    x, flow = torch.randn(1, 4, 4, 4), torch.randn(1, 2, 4, 4)
    seen = {}

    class Spy(nn.Module):
        def forward(self, a, b, c, mode='bilinear'):
            seen.update(a=a, b=b, c=c, mode=mode)
            return a

    m = MXActQuant(Spy(), [_specs('int8'), None], [[1], [1]])
    m(x, flow, 'not-a-tensor', mode='nearest')

    assert not torch.equal(seen['a'], x)          # quantized
    assert torch.equal(seen['b'], flow)           # spec None -> untouched
    assert seen['c'] == 'not-a-tensor'            # non-tensor positional
    assert seen['mode'] == 'nearest'              # kwargs untouched


def test_shared_instance_quantizes_every_call():
    net = Net()
    q = MXActQuant(net.warp, [_specs('int8'), _specs('int8', 16)], [[1], [-1]])
    net.warp = q
    net(torch.randn(1, 8, 8, 8), torch.randn(1, 2, 8, 8))
    assert q.n_calls == 2, "same instance called twice must quantize twice"


def test_gradients_flow_through_ste():
    x = torch.randn(1, 8, 4, 4, requires_grad=True)
    flow = torch.randn(1, 2, 4, 4, requires_grad=True)
    m = MXActQuant(Warp(), [_specs('int8'), _specs('int8', 16)], [[1], [-1]])
    m(x, flow).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert flow.grad is not None


# =============================================================================
# MXQuantizer config wiring
# =============================================================================

def _write_cfg(d, cfg):
    with open(os.path.join(d, "mx_config.json"), "w") as f:
        json.dump(cfg, f)


def test_quantizer_wraps_act_layer_from_config():
    cfg = {
        "mx_specs": {"w_elem_format": "int8", "a_elem_format": "int8",
                     "block_size": 32, "custom_cuda": False},
        "ptq": False,
        "measure_error": False,
        "layers": [
            {"name": "conv"},
            {"name": "warp", "kind": "act_quant",
             "inputs": [
                 {"mx_specs": {"a_elem_format": "int8", "block_size": 32,
                               "custom_cuda": False}, "axes": [1]},
                 {"mx_specs": {"a_elem_format": "int4", "block_size": 16,
                               "custom_cuda": False}, "axes": [-1]},
             ]},
        ],
    }
    with tempfile.TemporaryDirectory() as d:
        _write_cfg(d, cfg)
        qm = MXQuantizer(save_dir=d).quant(Net())

    assert isinstance(qm.warp, MXActQuant)
    assert isinstance(qm.warp.inner, Warp)
    assert qm.warp.specs_per_input[0]['a_elem_format'] == 'int8'
    assert qm.warp.specs_per_input[1]['a_elem_format'] == 'int4'
    assert qm.warp.axes_per_input == [[1], [-1]]
    # conv still replaced normally, act entry did not pollute the layer map
    assert type(qm.conv).__name__ == 'Conv2d' and hasattr(qm.conv, 'mx_specs')

    out = qm(torch.randn(2, 8, 8, 8), torch.randn(2, 2, 8, 8))
    assert out.shape == (2, 8, 8, 8)


def test_act_layer_group_reference_and_default_axes():
    cfg = {
        "groups": {"hi": {"a_elem_format": "int8", "block_size": 32, "custom_cuda": False},
                   "lo": {"a_elem_format": "int4", "block_size": 16, "custom_cuda": False}},
        "ptq": False,
        "measure_error": False,
        "layers": [
            {"name": "warp", "kind": "act_quant",
             "inputs": [{"group": "hi"}, {"group": "lo", "axes": [-1]}]},
        ],
    }
    with tempfile.TemporaryDirectory() as d:
        _write_cfg(d, cfg)
        qm = MXQuantizer(save_dir=d).quant(Net())

    assert qm.warp.axes_per_input == [[1], [-1]]     # input 0 defaults to [1]
    assert qm.warp.specs_per_input[1]['block_size'] == 16


# =============================================================================
# collect_stats integration
# =============================================================================

def test_collect_stats_reports_per_input_sections():
    cfg = {
        "ptq": False,
        "measure_error": False,
        "mx_specs": {"w_elem_format": "int8", "a_elem_format": "int8",
                     "block_size": 32, "custom_cuda": False},
        "layers": [
            {"name": "warp", "kind": "act_quant",
             "inputs": [
                 {"mx_specs": {"a_elem_format": "int8", "block_size": 32,
                               "custom_cuda": False}, "axes": [1]},
                 {"mx_specs": {"a_elem_format": "int4", "block_size": 16,
                               "custom_cuda": False}, "axes": [-1]},
             ]},
        ],
    }
    with tempfile.TemporaryDirectory() as d:
        _write_cfg(d, cfg)
        qm = MXQuantizer(save_dir=d).quant(Net())

    data = [(torch.randn(2, 8, 8, 8), torch.randn(2, 2, 8, 8)) for _ in range(2)]
    stats = mx_stats.collect_stats(
        qm, data=data, forward_fn=lambda m, b: m(b[0], b[1]), max_batches=2)

    e = stats["layers"]["warp"]
    assert e["layer_type"] == "MXActQuant"
    assert e["weight"] is None and e["w_elem_format"] is None
    ins = e["inputs"]
    assert len(ins) == 2
    assert ins[0]["a_elem_format"] == "int8" and ins[0]["axes"] == [1]
    assert ins[1]["a_elem_format"] == "int4" and ins[1]["axes"] == [-1]
    for sec in ins:
        assert sec["n_blocks"] > 0
        assert sec["error"]["sqnr_db"] > 0
    # int4 flow must be noisier than int8 features
    assert ins[1]["error"]["sqnr_db"] < ins[0]["error"]["sqnr_db"]
    # 2 batches x 2 call sites
    assert e["activation"]["n_calls"] == 4


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
