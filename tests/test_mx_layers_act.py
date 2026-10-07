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


# =============================================================================
# Static fixed-point operands and the output stage
#
# NPE's warp carries the reference as int8 but the grid as a 16-bit fixed-point
# word (6 integer + 6 fractional bits), and emits int8. Fixed point is a
# different lattice from MX, not another width of it: one static scale of
# 2^-frac_bits for the whole tensor instead of a shared exponent per block.
# =============================================================================

_GRID_FXP = {"total_bits": 16, "frac_bits": 6, "signed": True,
             "round": "half_away", "saturate": True}


def _fxp_cfg(**kw):
    from fixed_point.fxp_quant import normalize_out_quant
    return normalize_out_quant({**_GRID_FXP, **kw})


def test_fxp_input_lands_on_the_static_lattice():
    """An fxp operand snaps to 2^-frac_bits everywhere, not to a per-block step."""
    flow = (torch.rand(1, 2, 4, 64) * 64 - 32)       # +-32 px
    wrapped = MXActQuant(Warp(), [_specs(), None],
                         axes_per_input=[[-1], [1]],
                         fxp_per_input=[None, _fxp_cfg()])
    q = wrapped.quant_input(flow, 1)

    step = 2.0 ** -6
    codes = q / step
    assert torch.allclose(codes, codes.round()), "not on the 1/64 lattice"
    assert (q - flow).abs().max() <= step / 2 + 1e-6


def test_fxp_grid_is_far_finer_than_mx_int8():
    """The point of the change: MX int8 on a +-32 px grid is ~31x coarser."""
    flow = (torch.rand(1, 2, 4, 64) * 64 - 32)
    mx_err = (MXActQuant(Warp(), [_specs()], axes_per_input=[[-1]])
              .quant_input(flow, 0) - flow).abs().max()
    fxp_err = (MXActQuant(Warp(), [None], fxp_per_input=[_fxp_cfg()])
               .quant_input(flow, 0) - flow).abs().max()
    assert fxp_err < mx_err / 10, f"mx {mx_err:.6f} vs fxp {fxp_err:.6f}"
    assert fxp_err <= 2.0 ** -7 + 1e-6


def test_fxp_saturates_outside_the_representable_range():
    """A static scale can overflow where MX cannot. 6 frac bits of 13 -> +-64 px."""
    flow = torch.tensor([[-500.0, -64.0, 0.0, 63.9, 500.0]])
    wrapped = MXActQuant(Warp(), [None],
                         fxp_per_input=[_fxp_cfg(total_bits=13)])
    q = wrapped.quant_input(flow, 0)
    assert q.min() >= -64.0 and q.max() <= 64.0
    assert q[0, 0] == -64.0 and q[0, -1] > 63.0


def test_output_stage_quantizes_the_result():
    """Without an output spec the result is FP32; with one it is on the lattice."""
    x = torch.randn(1, 8, 4, 4)
    # Fractional flow: grid_sample interpolates, so the result lands off the MX
    # lattice. A zero flow would pass the already-quantized input straight
    # through and the output stage would be a legitimate no-op.
    flow = torch.full((1, 2, 4, 4), 0.5)

    plain = MXActQuant(Warp(), [_specs(), _specs()])
    out_q = MXActQuant(Warp(), [_specs(), _specs()],
                       out_spec=_specs(), out_axes=[1])

    y_plain = plain(x, flow)
    y_quant = out_q(x, flow)
    assert not torch.allclose(y_plain, y_quant), "output stage did nothing"
    assert torch.allclose(y_quant, _ref_quant(y_plain, _specs(), [1]))


def test_output_stage_handles_a_tuple_return():
    class TwoOut(nn.Module):
        def forward(self, a):
            return a * 2, a * 4

    mod = MXActQuant(TwoOut(), [None], out_fxp=_fxp_cfg())
    lo, hi = mod(torch.rand(1, 32) * 10)
    step = 2.0 ** -6
    for t in (lo, hi):
        assert torch.allclose(t / step, (t / step).round())


def test_fxp_and_mx_are_mutually_exclusive_in_config():
    import pytest
    cfg = {"mx_specs": {"a_elem_format": "int8", "block_size": 32},
           "layers": [{"name": "warp", "kind": "act_quant", "inputs": [
               {"fxp": _GRID_FXP, "mx_specs": {"a_elem_format": "int8"}}]}]}
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "mx_config.json"), "w") as f:
            json.dump(cfg, f)
        with pytest.raises(ValueError, match="alternative quantizers"):
            MXQuantizer(save_dir=d).quant(Net())


def test_quantizer_builds_the_npe_warp_shape_from_config():
    """End-to-end: reference MX int8, grid fxp 16/6, output MX int8."""
    cfg = {
        "mx_specs": {"a_elem_format": "int8", "block_size": 32,
                     "scale_bits": 8, "shared_exp_method": "max",
                     "custom_cuda": False},
        "layers": [
            {"name": "warp", "kind": "act_quant",
             "inputs": [
                 {"mx_specs": {"a_elem_format": "int8", "block_size": 32,
                               "scale_bits": 8, "shared_exp_method": "max",
                               "custom_cuda": False}, "axes": [-1]},
                 {"fxp": _GRID_FXP},
             ],
             "output": {"mx_specs": {"a_elem_format": "int8", "block_size": 32,
                                     "scale_bits": 8,
                                     "shared_exp_method": "max",
                                     "custom_cuda": False}, "axes": [-1]}},
        ],
    }
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "mx_config.json"), "w") as f:
            json.dump(cfg, f)
        model = MXQuantizer(save_dir=d).quant(Net())

    w = model.warp
    assert isinstance(w, MXActQuant)
    assert w.fxp_per_input[0] is None and w.fxp_per_input[1] is not None
    assert w.fxp_per_input[1]["frac_bits"] == 6
    assert w.out_spec is not None and w.out_spec["a_elem_format"] == "int8"
    assert "fxp16.6s" in w.extra_repr() and "out=" in w.extra_repr()

    y = model(torch.randn(1, 8, 6, 6), torch.rand(1, 2, 6, 6) * 4 - 2)
    assert y.shape == (1, 8, 6, 6) and torch.isfinite(y).all()


def test_collect_stats_reports_an_fxp_input_section():
    """An fxp operand must not vanish from the stats report."""
    cfg = {
        "mx_specs": {"a_elem_format": "int8", "block_size": 32,
                     "scale_bits": 8, "shared_exp_method": "max",
                     "custom_cuda": False},
        "layers": [{"name": "warp", "kind": "act_quant", "inputs": [
            {"mx_specs": {"a_elem_format": "int8", "block_size": 32,
                          "scale_bits": 8, "shared_exp_method": "max",
                          "custom_cuda": False}, "axes": [-1]},
            {"fxp": {**_GRID_FXP, "total_bits": 13}},
        ]}],
    }
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "mx_config.json"), "w") as f:
            json.dump(cfg, f)
        model = MXQuantizer(save_dir=d).quant(Net())

    # flow well past +-64 px, so the fxp input must report clipping
    data = [(torch.randn(1, 8, 6, 6), torch.full((1, 2, 6, 6), 300.0))]
    stats = mx_stats.collect_stats(
        model, data=data, forward_fn=lambda m, b: m(b[0], b[1]), max_batches=1)

    secs = stats["layers"]["warp"]["inputs"]
    assert secs[0]["quantizer"] == "mx"
    assert secs[0]["a_elem_format"] == "int8"
    assert secs[1]["quantizer"] == "fxp"
    assert secs[1]["format"].startswith("Q7.6")
    assert secs[1]["clipped_frac"] == 1.0, secs[1]
