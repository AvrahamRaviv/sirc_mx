"""Unit + integration tests for mx_stats (per-block quantization statistics)."""

import json
import math
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
from microxcaling.mx.convolution import Conv2d as MXConv2d
from microxcaling.mx.mx_ops import quantize_mx_op

from fixed_point.mx_fixed_point_hw import extract_mxint
from mx_layers_blocked import MXConv2dBlocked
import mx_debug
import mx_stats
from mx_stats import (_blockify, _RunningStat, _ErrAccum, _TensorStats,
                      _tensor_block_stats, collect_stats)


def _specs(fmt='int8', block_size=32):
    sp = MxSpecs()
    sp['w_elem_format'] = fmt
    sp['a_elem_format'] = fmt
    sp['block_size'] = block_size
    sp['scale_bits'] = 8
    sp['shared_exp_method'] = 'max'
    sp['custom_cuda'] = False
    return sp


def _run_block_stats(x, sp, axes=(-1,)):
    """quantize x with sp and accumulate one _TensorStats over it."""
    q = quantize_mx_op(x, sp, elem_format=sp['a_elem_format'], axes=list(axes))
    sink = _TensorStats()
    _tensor_block_stats(x, q, list(axes), sp['block_size'], sink)
    return sink, q


# =============================================================================
# Unit tests: helpers
# =============================================================================

def test_blockify_padding_and_mask():
    x = torch.randn(2, 48)
    blocks, mask = _blockify(x, [-1], 32)
    assert blocks.shape == (4, 32)
    assert mask.shape == (4, 32)
    # rows of 48 -> [32 real | 16 real + 16 pad] per input row
    assert mask.sum(-1).tolist() == [32, 16, 32, 16]
    assert (blocks[mask] != 0).any()
    assert (blocks[~mask] == 0).all()
    # divisible case: no padding, full mask
    blocks, mask = _blockify(torch.randn(2, 64), [-1], 32)
    assert blocks.shape == (4, 32) and mask.all()


def test_block_stats_exact_on_crafted_tensor():
    sp = _specs('int8')
    x = torch.zeros(1, 64)
    x[0, :32] = torch.linspace(-1.0, 1.0, 32)   # block 0
    x[0, 32:] = torch.linspace(0.5, 2.0, 32)    # block 1
    sink, _ = _run_block_stats(x, sp)
    assert sink.n_blocks == 2
    b0, b1 = x[0, :32], x[0, 32:]
    s = sink.block["max_abs"].summary()
    assert abs(s["min"] - b0.abs().max().item()) < 1e-6
    assert abs(s["max"] - b1.abs().max().item()) < 1e-6
    s = sink.block["mean_abs"].summary()
    expected = (b0.abs().mean().item() + b1.abs().mean().item()) / 2
    assert abs(s["mean"] - expected) < 1e-6
    s = sink.block["variance"].summary()
    assert abs(s["min"] - min(b0.var(unbiased=False), b1.var(unbiased=False)).item()) < 1e-5
    s = sink.block["dyn_range"].summary()
    expected_dyn = b1.abs().max().item() / b1.abs().mean().item()
    assert abs(s["min"] - min(expected_dyn,
                              b0.abs().max().item() / b0.abs().mean().item())) < 1e-5


def test_underflow_and_zero_rate():
    sp = _specs('int4')
    # one dominant value per block forces tiny values below the int4 grid
    x = torch.full((1, 64), 1e-4)
    x[0, 0] = 1.0
    x[0, 32] = 1.0
    x[0, 5] = 0.0   # a natural zero: counts in zero_rate, not underflow_rate
    sink, q = _run_block_stats(x, sp)
    assert sink.zero_cnt == int((q == 0).sum())
    assert sink.uf_cnt == int(((q == 0) & (x != 0)).sum())
    assert sink.zero_cnt == sink.uf_cnt + 1        # the single natural zero
    assert sink.uf_cnt == 61                       # all tiny values (64 - 2 ones - 1 zero)
    # int8 has more mantissa bits -> underflow never worse than int4
    torch.manual_seed(0)
    xr = torch.randn(4, 64)
    s4, _ = _run_block_stats(xr, _specs('int4'))
    s8, _ = _run_block_stats(xr, _specs('int8'))
    assert s8.uf_cnt <= s4.uf_cnt


def test_shared_exp_matches_extract_mxint():
    torch.manual_seed(1)
    sp = _specs('int8')
    x = torch.randn(8, 64).clamp(-1.9, 1.9)
    x[:, 0] = 2.0    # pin block maxes to an exact power of 2 (lattice-stable)
    x[:, 32] = 2.0
    q = quantize_mx_op(x, sp, elem_format='int8', axes=[-1])
    sink = _TensorStats()
    _tensor_block_stats(x, q, [-1], 32, sink)
    _, E = extract_mxint(q, 32, -1, 'int8')
    from collections import Counter
    assert sink.exp_counter == Counter(E.flatten().long().tolist())
    # fp8 path: same floor(log2(block max)) formula
    sp8 = _specs('fp8_e4m3')
    sink8, _ = _run_block_stats(x, sp8)
    expected = torch.floor(torch.log2(x.view(-1, 32).abs().amax(-1))).long()
    assert sink8.exp_counter == Counter(expected.tolist())


def test_error_metrics():
    torch.manual_seed(2)
    sp = _specs('int8')
    x = torch.randn(4, 64)
    q = quantize_mx_op(x, sp, elem_format='int8', axes=[-1])
    acc = _ErrAccum()
    acc.update(x, q)
    s = acc.summary()
    assert abs(s["sqnr_db"] - mx_debug._sqnr_db(x, q)) < 1e-6
    assert abs(s["mse"] - (x - q).pow(2).mean().item()) < 1e-9
    assert abs(s["max_abs_err"] - (x - q).abs().max().item()) < 1e-9
    cos = F.cosine_similarity(x.flatten(), q.flatten(), dim=0).item()
    assert abs(s["cos_sim"] - cos) < 1e-5
    assert s["n_elem"] == x.numel()


def test_running_stat_reservoir_cap_and_percentiles():
    torch.manual_seed(3)
    rs = _RunningStat(cap=1000)
    for _ in range(20):
        rs.update(torch.randn(1000))
    assert rs.res.numel() == 1000
    assert rs.n == 20000
    s = rs.summary()
    assert abs(s["mean"]) < 0.05
    assert abs(s["std"] - 1.0) < 0.05
    assert abs(s["p50"]) < 0.2          # median from reservoir sample
    assert s["p1"] < s["p25"] < s["p50"] < s["p75"] < s["p99"]
    h = rs.hist(bins=8)
    assert len(h["counts"]) == 8 and sum(h["counts"]) == 1000


# =============================================================================
# Integration tests: collect_stats
# =============================================================================

class _Net(nn.Module):
    """Two MX layers built directly (no quantizer needed)."""

    def __init__(self, sp):
        super().__init__()
        self.conv = MXConv2d(32, 16, 3, padding=1, mx_specs=sp)
        self.fc = nn.Linear(16, 8)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.fc(x.mean(dim=(2, 3)))


def test_collect_stats_weights_only():
    torch.manual_seed(4)
    m = _Net(_specs('int8'))
    stats = collect_stats(m, data=None, save_path=None)
    assert list(stats["layers"]) == ["conv"]
    e = stats["layers"]["conv"]
    assert e["weight"] is not None
    assert e["weight"]["error"]["sqnr_db"] > 0
    assert e["weight"]["n_blocks"] == 16 * 9      # 32ch/32bs=1 block x kHkW x O
    assert e["activation"] is None
    assert e["output_error"]["isolated"] is None
    assert stats["network"]["mean_w_sqnr_db"] > 0


def test_collect_stats_with_data():
    torch.manual_seed(5)
    m = _Net(_specs('int8'))
    data = [torch.randn(2, 32, 8, 8) for _ in range(3)]
    stats = collect_stats(m, data=data, save_path=None)
    e = stats["layers"]["conv"]
    assert e["activation"] is not None
    assert e["activation"]["n_calls"] == 3
    assert e["activation"]["error"]["sqnr_db"] > 0
    assert e["activation"]["shared_exp"] is not None
    assert e["output_error"]["isolated"]["sqnr_db"] > 0
    assert stats["meta"]["n_batches"] == 3
    assert 0.0 <= e["activation"]["zero_rate"] <= 1.0


def test_output_error_flag_off():
    torch.manual_seed(6)
    m = _Net(_specs('int8'))
    stats = collect_stats(m, data=[torch.randn(1, 32, 8, 8)],
                          output_error=False, save_path=None)
    assert stats["layers"]["conv"]["output_error"]["isolated"] is None


def test_isolated_output_error_matches_manual():
    """Isolated out error == fp32 functional vs quantized forward, computed by hand."""
    torch.manual_seed(7)
    sp = _specs('int8')
    layer = MXConv2d(32, 16, 3, padding=1, mx_specs=sp)
    m = nn.Sequential(layer)
    x = torch.randn(2, 32, 8, 8)
    stats = collect_stats(m, data=[x], save_path=None)
    got = stats["layers"]["0"]["output_error"]["isolated"]["sqnr_db"]
    with torch.no_grad():
        ref = F.conv2d(x, layer.weight, layer.bias, layer.stride,
                       layer.padding, layer.dilation, layer.groups)
        out = layer(x)
    assert abs(got - mx_debug._sqnr_db(ref, out)) < 1e-6


def test_variants_blocked():
    torch.manual_seed(8)
    layer = MXConv2dBlocked(64, 16, 3, padding=1, bias=True, mx_specs=_specs('int8'))
    layer.xblock_accum = {"enabled": True, "bits": 48, "backend": "python",
                          "scale_exp": None, "saturate": True, "ste_mask": False}
    m = nn.Sequential(layer)
    stats = collect_stats(m, data=[torch.randn(1, 64, 8, 8)], save_path=None)
    e = stats["layers"]["0"]
    assert e["layer_type"] == "MXConv2dBlocked"
    assert e["act_axes"] == [1] and e["wt_axes"] == [1]
    assert math.isfinite(e["output_error"]["isolated"]["sqnr_db"])
    assert math.isfinite(e["weight"]["error"]["sqnr_db"])


def test_json_dump_roundtrip():
    torch.manual_seed(9)
    m = _Net(_specs('int8'))
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "quant_stats.json")
    collect_stats(m, data=[torch.randn(1, 32, 8, 8)],
                  histograms=True, save_path=path)
    assert os.path.exists(path)
    with open(path) as f:
        text = f.read()
    assert "NaN" not in text and "Infinity" not in text
    loaded = json.loads(text)
    assert "conv" in loaded["layers"]
    assert loaded["layers"]["conv"]["weight"]["block"]["max_abs"]["hist"] is not None


def test_multi_call_shared_layer():
    torch.manual_seed(10)

    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = MXConv2d(32, 32, 3, padding=1, mx_specs=_specs('int8'))

        def forward(self, x):
            return self.conv(self.conv(x))

    m = Twice()
    data = [torch.randn(1, 32, 8, 8) for _ in range(2)]
    stats = collect_stats(m, data=data, save_path=None)
    e = stats["layers"]["conv"]
    assert e["activation"]["n_calls"] == 4     # 2 calls per forward x 2 batches
    assert math.isfinite(e["output_error"]["isolated"]["sqnr_db"])


def test_detail_keeps_raw_blocks():
    torch.manual_seed(11)
    m = _Net(_specs('int8'))
    stats = collect_stats(m, detail=True, save_path=None)
    raw = stats["layers"]["conv"]["weight"]["raw_blocks"]
    assert len(raw["max_abs"]) == stats["layers"]["conv"]["weight"]["n_blocks"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
