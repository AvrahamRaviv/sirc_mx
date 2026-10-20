"""Static output window of the HW fixed-point conv (the back half).

O = N >> (e_qa_out - e_layer_min), saturated to +-(2^W - 1), output
O * 2^e_qa_out. W = (mbits-1) + (2^out_exp_bits - 1) = 22 for int8 / 4b. Grids from `calibrate_hw_grids`. The worked vector is the one
in two_stage_quant_flow.md section 5.
"""
import json
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects")
sys.path.insert(0, "/home/avrahamra/PycharmProjects")

from microxcaling.mx import MxSpecs

from fixed_point.mx_fixed_point import normalize_xblock_accum
from fixed_point.mx_fixed_point_hw import (
    _apply_out_window, _hw_fxp_conv2d_ref, _hw_fxp_conv2d_ref_npe,
    calibrate_e_layer_min, calibrate_hw_grids, derive_out_window_bits,
    e_qa_out_from_absmax, extract_mxint, hw_fxp_conv2d,
    resolve_out_window_bits,
)
from mx_layers_blocked import MXConv2dHW

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
W = 22                       # int8 + 4b output exponent
WIN_HI = (1 << W) - 1


def _t(v):
    return torch.tensor(v, dtype=torch.int64)


# ---------------------------------------------------------------------------
# Worked vector (guide section 5)
# ---------------------------------------------------------------------------

def _guide_operands():
    """2 blocks (4 + 2 lanes) as one 1x1 conv over 6 channels. Block sizes
    differ, so this goes through the per-element-exponent (NPE) ref."""
    a = [100, 64, -32, 120, 80, -40]
    w = [50, -70, 90, 25, 60, 100]
    ea = [10, 10, 10, 10, 7, 7]
    ew = [8, 8, 8, 8, 9, 9]
    qi = torch.tensor(a, dtype=torch.int8).view(1, 6, 1, 1)
    qw = torch.tensor(w, dtype=torch.int8).view(1, 6, 1, 1)
    Ea = torch.tensor(ea, dtype=torch.int16).view(1, 6, 1, 1)
    Ew = torch.tensor(ew, dtype=torch.int16).view(1, 6, 1, 1)
    return qi, qw, Ea, Ew


def test_guide_worked_vector():
    qi, qw, Ea, Ew = _guide_operands()
    e_acc = (7 - 6) + (8 - 6)                       # E_qa_in + E_qw = 3
    assert e_acc == 3

    out, sat, acc = _hw_fxp_conv2d_ref_npe(
        qi, qw, Ea, Ew, e_acc, 1, 0, 1, bits=48, sat_mode="per_product",
        return_acc=True)
    assert int(acc.item()) == 6720
    assert float(out.item()) == 53760.0              # N * 2^E_acc
    assert not sat.any()

    sink = {}
    out_w, sat_w, acc_w = _hw_fxp_conv2d_ref_npe(
        qi, qw, Ea, Ew, e_acc, 1, 0, 1, bits=48, sat_mode="per_product",
        return_acc=True, e_qa_out=5, win_bits=W, win_sink=sink)
    assert int(acc_w.item()) == 6720                  # window leaves N alone
    assert float(out_w.item()) == 53760.0             # O * 2^5 = 1680 * 32
    assert not sink["win_sat"].any()

    O, win_sat = _apply_out_window(acc_w, e_acc, 5, W)
    assert int(O.item()) == 1680                      # shift_out = 5 - 1 - 2 = 2
    assert not win_sat.any()


# ---------------------------------------------------------------------------
# Window arithmetic
# ---------------------------------------------------------------------------

def test_window_saturates_both_signs():
    acc = _t([1 << 30, -(1 << 30), (1 << 24) - 1, 1 << 24, -(1 << 24)])
    O, sat = _apply_out_window(acc, 0, 2, W)
    assert O.tolist() == [WIN_HI, -WIN_HI, WIN_HI, WIN_HI, -WIN_HI]
    # (2^24 - 1) >> 2 = 2^22 - 1 fits exactly; 2^24 >> 2 = 2^22 does not.
    assert sat.tolist() == [True, True, False, True, True]


def test_window_rounding_floor_vs_nearest():
    acc = _t([6723, -6723, 6722, -6722])
    O_floor, _ = _apply_out_window(acc, 0, 2, W, out_round="floor")
    O_near, _ = _apply_out_window(acc, 0, 2, W, out_round="nearest")
    assert O_floor.tolist() == [1680, -1681, 1680, -1681]     # arithmetic shift
    assert O_near.tolist() == [1681, -1681, 1681, -1680]      # round half up


def test_window_left_shift_regime():
    # e_qa_out < e_acc: the accumulator holds fewer than 23 significant bits.
    O, sat = _apply_out_window(_t([100, -100]), 5, 3, W)
    assert O.tolist() == [400, -400] and not sat.any()
    edge = WIN_HI >> 2
    O, sat = _apply_out_window(_t([edge, edge + 1, -edge, -edge - 1]), 5, 3, W)
    assert O.tolist() == [edge << 2, WIN_HI, -(edge << 2), -WIN_HI]
    assert sat.tolist() == [False, True, False, True]


def test_window_zero_shift_is_clamp_only():
    O, sat = _apply_out_window(_t([123, WIN_HI + 1]), 4, 4, W)
    assert O.tolist() == [123, WIN_HI]
    assert sat.tolist() == [False, True]


def test_e_qa_out_rule():
    # Owner decision 2026-10-20: floor(log2(AbsMax)) - (W-1) + margin_bits,
    # W = 22 for int8 / 4b -> floor(log2(AbsMax)) - 21.
    assert e_qa_out_from_absmax(2.0 ** 28, W) == 7
    assert e_qa_out_from_absmax(2.0 ** 28 - 1, W) == 6
    assert e_qa_out_from_absmax(3 * 2.0 ** 27, W) == 7
    assert e_qa_out_from_absmax(2.0 ** 28, W, margin_bits=1) == 8
    assert e_qa_out_from_absmax(0.75, W) == -22
    assert e_qa_out_from_absmax(0.0, W) is None
    assert e_qa_out_from_absmax(2.0 ** 28, 14) == 15
    # The calibrated max itself always fits the window unsaturated.
    for wb in [W, 14, 30]:
        for absmax in [1.0, 2.0 ** 28, 2.0 ** 28 - 1, 3 * 2.0 ** 27, 0.3, 1e6]:
            e = e_qa_out_from_absmax(absmax, wb)
            assert absmax / 2.0 ** e < 2 ** wb


@pytest.mark.parametrize("fmt,exp_bits,want", [
    ("int8", 4, 22), ("int8", 3, 14), ("int8", 5, 38),
    ("int4", 4, 18), ("int16", 4, 30), ("int16", 5, 46),
])
def test_derive_out_window_bits(fmt, exp_bits, want):
    assert derive_out_window_bits(fmt, exp_bits) == want


def test_derived_width_matches_mx_block_span():
    """W is exactly the bit span an int8 block with a 4b exponent addresses:
    lowest LSB at Emin - mant_bias, highest MSB at Emax = Emin + 15."""
    e_min_blk, mant_bias, max_m = 0, 6, 127
    lsb = e_min_blk - mant_bias
    top = max_m * 2.0 ** (e_min_blk + 15 - mant_bias)       # largest value
    assert top / 2.0 ** lsb < 2 ** W                         # fits W bits
    assert top / 2.0 ** lsb >= 2 ** (W - 1)                  # and needs all W
    # Both E_q forms agree at W: floor(log2 max) - (W-1) = Emin - mant_bias.
    assert e_qa_out_from_absmax(top, W) == lsb


def test_resolve_out_window_bits():
    assert resolve_out_window_bits({}, "int8") == 22
    assert resolve_out_window_bits({"out_exp_bits": 3}, "int8") == 14
    assert resolve_out_window_bits({"out_window_bits": 23}, "int8") == 23
    assert resolve_out_window_bits({"out_window_bits": None,
                                    "out_exp_bits": 4}, "int4") == 18
    with pytest.raises(ValueError):
        resolve_out_window_bits({"out_window_bits": 47}, "int8")


# ---------------------------------------------------------------------------
# Through hw_fxp_conv2d
# ---------------------------------------------------------------------------

def _mx_operands(seed=0, C=32, O=8, bs=32):
    from microxcaling.mx.mx_ops import quantize_mx_op
    sp = _sp()
    torch.manual_seed(seed)
    x = torch.randn(2, C, 6, 6) * 4
    w = torch.randn(O, C, 3, 3) * 0.1
    qi = quantize_mx_op(x, sp, elem_format="int8", axes=[1], block_size=bs)
    qw = quantize_mx_op(w, sp, elem_format="int8", axes=[1], block_size=bs)
    return qi, qw


@pytest.mark.parametrize("out_round", ["floor", "nearest"])
def test_hw_fxp_conv2d_window_matches_ref(out_round):
    qi, qw = _mx_operands()
    bias = torch.randn(8) * 0.5
    e_min = -20
    plain_sink = {"want_acc": True}
    out_plain = hw_fxp_conv2d(qi, qw, bias, e_min, 32, bits=48, padding=1,
                              stats_sink=plain_sink)
    acc = plain_sink["acc"]
    assert "win" not in plain_sink and "win_sat_count" not in plain_sink

    e_qa_out = e_qa_out_from_absmax(float(out_plain.abs().max()), W)
    sink = {"want_acc": True}
    # win_bits left None: derived from fmt=int8 + default 4b exponent = W.
    out = hw_fxp_conv2d(qi, qw, bias, e_min, 32, bits=48, padding=1,
                        stats_sink=sink, e_qa_out=e_qa_out, out_round=out_round)
    O, win_sat = _apply_out_window(acc, e_min, e_qa_out, W, out_round=out_round)
    assert torch.equal(sink["acc"], acc)
    assert torch.equal(sink["win"], O)
    assert torch.equal(out, O.to(torch.float32) * 2.0 ** e_qa_out)
    assert sink["win_sat_count"] == int(win_sat.sum()) == 0
    # Window error is below one output LSB (floor) / half an LSB (nearest).
    lsb = 2.0 ** e_qa_out
    err = (out.double() - acc.double() * 2.0 ** e_min).abs().max().item()
    assert err < (lsb if out_round == "floor" else lsb / 2 + 1e-12)


def test_hw_fxp_conv2d_window_off_unchanged():
    qi, qw = _mx_operands(seed=3)
    out = hw_fxp_conv2d(qi, qw, None, -20, 32, bits=48, padding=1)
    qi8, Ea = extract_mxint(qi, 32, axis=1)
    qw8, Ew = extract_mxint(qw, 32, axis=1)
    ref, _ = _hw_fxp_conv2d_ref(qi8, qw8, Ea, Ew, -20, 1, 1, 1, 32, 48, "per_product")
    assert torch.equal(out, ref.view_as(out))


# ---------------------------------------------------------------------------
# Layer: calibration + infer-only flag
# ---------------------------------------------------------------------------

def _sp():
    sp = MxSpecs()
    sp["w_elem_format"] = "int8"
    sp["a_elem_format"] = "int8"
    sp["block_size"] = 32
    sp["scale_bits"] = 8
    sp["shared_exp_method"] = "max"
    sp["custom_cuda"] = False
    return sp


def _layer(out_window="off", e_layer_min=None, seed=7, npe=False, **extra):
    torch.manual_seed(seed)
    hw = MXConv2dHW(32, 8, 3, padding=1, bias=True, mx_specs=_sp())
    cfg = {
        "enabled": True, "bits": 48, "backend": "python",
        "mode": "hw_fixed_point", "sat_mode": "per_product",
        "e_layer_min": e_layer_min, "out_window": out_window, "verbose": 0,
    }
    if npe:
        cfg.update(act_blockify="xblock", weight_blockify="flatten")
    cfg.update(extra)
    hw.xblock_accum = normalize_xblock_accum(cfg)
    hw.e_layer_min = e_layer_min
    hw._mx_layer_name = "c"
    return hw


class _Net(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.bn = nn.BatchNorm2d(32)
        self.c = conv

    def forward(self, x):
        return self.c(self.bn(x))


def _data(n=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, 32, 8, 8, generator=g) * 3 for _ in range(n)]


@pytest.mark.parametrize("npe", [False, True])
def test_calibrate_hw_grids_sets_both_and_restores_mode(npe):
    model = _Net(_layer(npe=npe))
    model.train()
    res = calibrate_hw_grids(model, _data(), num_batches=3, verbose=0)
    hw = model.c
    assert model.training and model.bn.training
    assert hw.e_layer_min is not None and hw.e_qa_out is not None
    r = res["c"]
    assert r["e_layer_min"] == r["e_qa_in"] + r["e_qw"] == hw.e_layer_min
    assert r["win_bits"] == W
    # Typically negative: an MXINT8 tensor spans fewer than 23 bits above its
    # min block exp, so the window pads (left shift) and loses nothing.
    assert r["shift_out"] == hw.e_qa_out - hw.e_layer_min
    # The calibration data itself fits the window.
    model.eval()
    hw.xblock_accum["out_window"] = "infer"
    hw._tap_acc = True
    with torch.no_grad():
        for x in _data():
            model(x)
            assert not hw._last_win_sat.any()


def test_calibrate_respects_preset_e_layer_min_unless_force():
    model = _Net(_layer(e_layer_min=-20))
    calibrate_hw_grids(model, _data(), num_batches=2, verbose=0)
    assert model.c.e_layer_min == -20 and model.c.e_qa_out is not None
    e_qa_out = model.c.e_qa_out

    # Both grids set: a second call does not sweep ...
    r = calibrate_hw_grids(model, _data(), num_batches=2, verbose=0)["c"]
    assert r["e_qa_in"] is None and r["e_layer_min"] == -20
    # ... unless forced, which also replaces the pinned -20.
    calibrate_hw_grids(model, _data(), num_batches=2, force=True, verbose=0)
    assert model.c.e_layer_min != -20
    assert model.c.e_qa_out == e_qa_out

    m2 = _Net(_layer(e_layer_min=-20))
    assert calibrate_e_layer_min(m2, _data(), num_batches=2) == {}
    assert m2.c.e_layer_min == -20
    calibrate_e_layer_min(m2, _data(), num_batches=2, force=True)
    assert m2.c.e_layer_min == model.c.e_layer_min


def test_margin_bits():
    a = _Net(_layer())
    b = _Net(_layer())
    calibrate_hw_grids(a, _data(), num_batches=2, verbose=0)
    calibrate_hw_grids(b, _data(), num_batches=2, margin_bits=2, verbose=0)
    assert b.c.e_qa_out == a.c.e_qa_out + 2


def test_width_from_config():
    """out_exp_bits=3 -> W=14: same max, grid 8 bits coarser. An explicit
    out_window_bits overrides the derivation."""
    a = _Net(_layer())
    b = _Net(_layer(out_exp_bits=3))
    c = _Net(_layer(out_window_bits=23))
    ra = calibrate_hw_grids(a, _data(), num_batches=2, verbose=0)["c"]
    rb = calibrate_hw_grids(b, _data(), num_batches=2, verbose=0)["c"]
    rc = calibrate_hw_grids(c, _data(), num_batches=2, verbose=0)["c"]
    assert (ra["win_bits"], rb["win_bits"], rc["win_bits"]) == (22, 14, 23)
    assert b.c.e_qa_out == a.c.e_qa_out + 8
    assert c.c.e_qa_out == a.c.e_qa_out - 1

    hw = b.c
    hw.eval()
    hw.xblock_accum["out_window"] = "infer"
    hw._tap_acc = True
    with torch.no_grad():
        hw(_data(1, seed=4)[0])
    O, sat = _apply_out_window(hw._last_acc, hw.e_layer_min, hw.e_qa_out, 14)
    assert torch.equal(hw._last_win, O)
    assert hw._last_win.abs().max() <= (1 << 14) - 1


def _fwd_bwd(hw, x):
    hw.zero_grad()
    x = x.clone().requires_grad_(True)
    out = hw(x)
    out.backward(torch.ones_like(out))
    return out.detach(), x.grad.clone(), hw.weight.grad.clone(), hw.bias.grad.clone()


def test_infer_only_window():
    off = _layer("off")
    win = _layer("infer")
    win.load_state_dict(off.state_dict())
    calibrate_hw_grids(_Net(off), _data(), num_batches=2, verbose=0)
    win.e_layer_min, win.e_qa_out = off.e_layer_min, off.e_qa_out
    x = _data(1, seed=5)[0]

    # Train mode: identical to the window-off path, forward and backward.
    off.train(); win.train()
    o_off, *g_off = _fwd_bwd(off, x)
    o_win, *g_win = _fwd_bwd(win, x)
    assert torch.equal(o_off, o_win)
    for a, b in zip(g_off, g_win):
        assert torch.equal(a, b)

    # Eval mode: forward windowed, backward still the same STE.
    off.eval(); win.eval()
    off._tap_acc = win._tap_acc = True
    o_off, *g_off = _fwd_bwd(off, x)
    o_win, *g_win = _fwd_bwd(win, x)
    assert off._last_win is None
    O, _ = _apply_out_window(off._last_acc, off.e_layer_min, off.e_qa_out, W)
    assert torch.equal(win._last_win, O)
    assert torch.equal(win._last_acc, off._last_acc)
    assert win._last_qi is not None and win._last_qw is not None
    for a, b in zip(g_off, g_win):
        assert torch.equal(a, b)
    if win.e_qa_out <= win.e_layer_min:
        assert torch.equal(o_off, o_win)          # left shift: exact padding

    # Window forced 6 bits above the accumulator grid: drops sub-LSB bits.
    win.e_qa_out = win.e_layer_min + 6
    o_win, *g_win = _fwd_bwd(win, x)
    O, sat = _apply_out_window(off._last_acc, off.e_layer_min, win.e_qa_out, W)
    assert torch.equal(win._last_win, O) and not sat.any()
    assert not torch.equal(o_off, o_win)
    assert (o_off - o_win).abs().max() < 2.0 ** win.e_qa_out
    for a, b in zip(g_off, g_win):
        assert torch.equal(a, b)


def test_always_window_in_train():
    hw = _layer("always")
    calibrate_hw_grids(_Net(hw), _data(), num_batches=2, verbose=0)
    hw.train()
    hw._tap_acc = True
    hw(_data(1)[0])
    assert hw._last_win is not None


def test_window_without_e_qa_out_raises():
    hw = _layer("infer", e_layer_min=-20)
    hw.eval()
    with pytest.raises(RuntimeError, match="calibrate_hw_grids"):
        hw(_data(1)[0])
    hw.train()
    hw(_data(1)[0])                               # train mode: window not used


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_defaults_and_validation():
    cfg = normalize_xblock_accum({"mode": "hw_fixed_point"})
    assert cfg["out_window"] == "off"
    assert cfg["out_window_bits"] is None          # derive
    assert cfg["out_exp_bits"] == 4
    assert cfg["out_round"] == "floor"
    assert cfg["out_margin_bits"] == 0
    assert normalize_xblock_accum({"out_window_bits": 23})["out_window_bits"] == 23
    for bad in [{"out_window": "eval"}, {"out_round": "trunc"},
                {"out_window_bits": 0}, {"out_margin_bits": -1},
                {"out_exp_bits": 0}, {"out_exp_bits": 6}]:
        with pytest.raises(ValueError):
            normalize_xblock_accum(bad)


@pytest.mark.parametrize("name", ["mx_config_int8_hw_win.json",
                                  "mx_config_dof_run_win.json"])
def test_opt_in_configs(name):
    with open(os.path.join(REPO, "configs", name)) as f:
        full = json.load(f)
    assert full["calibrate_hw_grids"]["enabled"] is True
    cfg = normalize_xblock_accum(full["mx_specs"]["xblock_accum"])
    assert cfg["e_layer_min"] is None
    assert cfg["out_window"] == "infer"


def test_existing_configs_untouched():
    for name in ["mx_config_int8_hw.json", "mx_config_dof_run.json"]:
        with open(os.path.join(REPO, "configs", name)) as f:
            spec = json.load(f)["mx_specs"]["xblock_accum"]
        assert spec["e_layer_min"] == -20
        assert normalize_xblock_accum(spec)["out_window"] == "off"


# ---------------------------------------------------------------------------
# Triton parity (CUDA only)
# ---------------------------------------------------------------------------

try:
    import triton  # noqa: F401
    _HAS_TRITON = torch.cuda.is_available()
except ImportError:
    _HAS_TRITON = False


@pytest.mark.skipif(not _HAS_TRITON, reason="triton + CUDA required")
@pytest.mark.parametrize("out_round", ["floor", "nearest"])
@pytest.mark.parametrize("kind", ["channel", "npe_reuse", "npe_opt"])
@pytest.mark.parametrize("regime", ["fit", "sat", "left"])
def test_triton_window_bit_identical(kind, out_round, regime):
    qi, qw = _mx_operands(seed=11)
    qi, qw = qi.cuda(), qw.cuda()
    bias = (torch.randn(8) * 0.5).cuda()
    e_min = -20
    blk = {} if kind == "channel" else dict(
        act_blockify="xblock", weight_blockify="flatten",
        npe_triton_variant="opt" if kind == "npe_opt" else "reuse")
    probe = hw_fxp_conv2d(qi, qw, bias, e_min, 32, bits=48, padding=1, **blk)
    e_fit = e_qa_out_from_absmax(float(probe.abs().max()), W)
    e_qa_out = {"fit": max(e_fit, e_min + 3),       # right shift, no sat
                "sat": e_min + 1,                   # right shift 1, clips
                "left": e_min - 2}[regime]          # left shift
    outs, sinks = {}, {}
    for backend in ["python", "triton"]:
        sinks[backend] = {}
        outs[backend] = hw_fxp_conv2d(
            qi, qw, bias, e_min, 32, bits=48, padding=1, backend=backend,
            stats_sink=sinks[backend], e_qa_out=e_qa_out, out_round=out_round,
            **blk)
    assert torch.equal(outs["python"], outs["triton"])
    assert sinks["python"]["win_sat_count"] == sinks["triton"]["win_sat_count"]
