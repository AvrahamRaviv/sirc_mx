"""HW-exact switches M1-M8 (HW answers Q1-Q6, Q10): ALG matches the NPE
emulator where the owner decided "match theirs". Every switch is off by
default; each test has a hand vector, and where the emulator has the same
primitive (configs/hw/npe_mx_emu) the test checks against it directly.

M1 mant_round 'truncate'   M2/M3 overflow 'raise'   M4 out_round 'nearest'
M5 bias_round 'half_up'    M6 in_exp_anchor 'min'   M7 static_shift_bits
M8 in_mode 'int'           M3 window + overflow_saturate ['out_window']: saturates (HW)
"""
import glob
import json
import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects")
sys.path.insert(0, "/home/avrahamra/PycharmProjects")

from fixed_point.hw_grids import check_hw_grid, load_hw_grids
from fixed_point.mx_fixed_point import normalize_xblock_accum
from fixed_point.mx_fixed_point_hw import (_apply_out_window, _bias_to_grid, _window_params,
                                           out_window_sat_word, window_sat_word,
                                           block_exp_range, calibrate_hw_grids,
                                           check_static_shift, clamp_block_exp,
                                           export_hw_grids, in_int_scale_from_absmax,
                                           static_shift_of)
from mx_layers_blocked import quantize_mx_op
from mx_quantizer import MXQuantizer


def _hw_pipe():
    hw = os.path.join(REPO, "configs", "hw")
    if hw not in sys.path:
        sys.path.insert(0, hw)
    try:
        from npe_mx_emu import hw_pipe, calib
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"HW emulator not importable: {e!r}")
    return hw_pipe, calib


class _Net(nn.Module):
    def __init__(self, cin=32, cout=2, k=1, bias=True):
        super().__init__()
        torch.manual_seed(3)
        self.c1 = nn.Conv2d(cin, cout, k, padding=k // 2, bias=bias)

    def forward(self, x):
        return self.c1(x)


def _model(net=None, hw_grids=None, top=None, data=None, **xblock):
    tmp = tempfile.mkdtemp()
    xb = {"enabled": True, "mode": "hw_fixed_point", "bits": 48,
          "backend": "python", "sat_mode": "per_product", "e_layer_min": None,
          "verbose": 0, **xblock}
    cfg = {"mx_specs": {"w_elem_format": "int8", "a_elem_format": "int8",
                        "block_size": 32, "custom_cuda": False,
                        "xblock_accum": xb},
           "layers": ["c1"], "ptq": False, "measure_error": False, **(top or {})}
    if hw_grids is not None:
        cfg["hw_grids"] = hw_grids
    with open(os.path.join(tmp, "mx_config.json"), "w") as f:
        json.dump(cfg, f)
    return MXQuantizer(save_dir=tmp).quant(net or _Net(), data=data), tmp


def _set_w(model, w, b=None):
    with torch.no_grad():
        model.c1.weight.copy_(w)
        if b is not None:
            model.c1.bias.copy_(b)
        elif model.c1.bias is not None:
            model.c1.bias.zero_()


def _run(model, x, train=False):
    model.train(train)
    with torch.no_grad():
        return model(x)


# ---------------------------------------------------------------------------
# M1 mantissa truncation (Q1)
# ---------------------------------------------------------------------------


def _m1_input():
    """One block (32 channels) at E = 0: step 2^-6. Channel 1 = 32.75 steps,
    channel 2 = -32.75 steps."""
    x = torch.zeros(1, 32, 1, 1)
    x[0, 0] = 1.0
    x[0, 1] = (32 + 0.75) / 64
    x[0, 2] = -(32 + 0.75) / 64
    return x


def _m1_model(**kw):
    model, _ = _model(e_layer_min=-40, **kw)
    w = torch.zeros(2, 32, 1, 1)
    w[0, 1] = 1.0                    # out0 = x1
    w[1, 2] = 1.0                    # out1 = x2
    _set_w(model, w)
    return model


def test_m1_truncate_hand_vector():
    x = _m1_input()
    near = _run(_m1_model(), x).flatten().tolist()
    trunc = _run(_m1_model(mant_round="truncate"), x).flatten().tolist()
    assert near == [33 / 64, -33 / 64]
    assert trunc == [32 / 64, -32 / 64]        # toward zero, both signs


def test_m1_truncate_matches_emulator_int():
    """HW base_mx_quantizer: data = (x / 2^(E-6)).int() (truncate toward 0)."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 32, 3, 5, generator=g)
    sp = _m1_model().c1.mx_specs
    q = quantize_mx_op(x, sp, elem_format="int8", axes=[1], round="floor")
    E = torch.floor(torch.log2(x.abs().amax(1, keepdim=True)))
    hw = (x / 2.0 ** (E - 6)).int().float() * 2.0 ** (E - 6)
    assert torch.equal(q, hw)


def test_m1_clamp_reround_truncates():
    # 100 steps at E = -20 onto base -15 (step 2^-21): 3.125 -> 3, both modes
    # differ only on > .5: 3.75 -> nearest 4, truncate 3.
    q = torch.tensor([120 * 2.0 ** -26, -120 * 2.0 ** -26])
    assert clamp_block_exp(q, -15, 4)[0].tolist() == [4 * 2.0 ** -21, -4 * 2.0 ** -21]
    assert clamp_block_exp(q, -15, 4, round="truncate")[0].tolist() == \
        [3 * 2.0 ** -21, -3 * 2.0 ** -21]


def test_m1_default_ignores_round():
    """mant_round 'nearest' passes no round= (bit-identical to v1.81)."""
    model = _m1_model()
    assert model.c1._mant_round() == {}
    assert _m1_model(mant_round="truncate").c1._mant_round() == {"round": "floor"}


# ---------------------------------------------------------------------------
# M2 accumulator overflow raises in eval (Q2, Q6)
# ---------------------------------------------------------------------------


def _m2_model(**kw):
    # x = w = 1 -> mantissas 64 * 64 = 2^12, shift 0 + 0 - 12 + 60 = 48:
    # each product 2^60 > 2^47, the 48b accumulator overflows.
    model, _ = _model(e_layer_min=-60, **kw)
    _set_w(model, torch.ones(2, 32, 1, 1))
    return model


def test_m2_overflow_raises_in_eval():
    x = torch.ones(1, 32, 1, 1)
    with pytest.raises(RuntimeError, match="overflowed the 48b accumulator"):
        _run(_m2_model(overflow="raise"), x)


def test_m2_train_and_default_clamp():
    x = torch.ones(1, 32, 1, 1)
    hi = float(torch.tensor(2 ** 47 - 1).float()) * 2.0 ** -60   # fp32 cast of the word
    assert _run(_m2_model(overflow="raise"), x, train=True).flatten().tolist() == [hi, hi]
    assert _run(_m2_model(), x).flatten().tolist() == [hi, hi]


def test_m2_in_range_no_raise():
    model, _ = _model(e_layer_min=-20, overflow="raise")
    _set_w(model, torch.ones(2, 32, 1, 1))
    assert _run(model, torch.ones(1, 32, 1, 1)).flatten().tolist() == [32.0, 32.0]


# ---------------------------------------------------------------------------
# M3 window: |value| >= 2^22 impossible -> raise (Q2)
# ---------------------------------------------------------------------------


def _m3_model(e_qa_out, **kw):
    model, _ = _model(e_layer_min=-20, out_window="infer", overflow="raise",
                      hw_grids={"c1": {"e_layer_min": -20, "e_qa_out": e_qa_out}}, **kw)
    _set_w(model, torch.ones(2, 32, 1, 1))
    return model


def test_m3_window_boundary_hand_vector():
    # The window word: 2^22 - 1 fits, 2^22 saturates (22 bits + sign).
    acc = torch.tensor([2 ** 22 - 1, 2 ** 22, -(2 ** 22 - 1), -(2 ** 22)])
    v, sat = _apply_out_window(acc, 0, 0, 22)
    assert sat.tolist() == [False, True, False, True]
    assert v.tolist() == [2 ** 22 - 1, 2 ** 22 - 1, -(2 ** 22 - 1), -(2 ** 22 - 1)]


def test_m3_window_overflow_raises():
    x = torch.ones(1, 32, 1, 1)                 # out = 32 = 2^5
    assert _run(_m3_model(-16), x).flatten().tolist() == [32.0, 32.0]  # O = 2^21
    with pytest.raises(RuntimeError, match="22b output window"):
        _run(_m3_model(-17), x)                 # O = 2^22
    m = _m3_model(-17)
    assert _run(m, x, train=True).flatten().tolist() == [32.0, 32.0]   # train: no window


def test_m3_window_overflow_clamps_by_default():
    model, _ = _model(e_layer_min=-20, out_window="infer",
                      hw_grids={"c1": {"e_layer_min": -20, "e_qa_out": -17}})
    _set_w(model, torch.ones(2, 32, 1, 1))
    out = _run(model, torch.ones(1, 32, 1, 1)).flatten().tolist()
    assert out == [(2 ** 22 - 1) * 2.0 ** -17] * 2


# ---------------------------------------------------------------------------
# M3 window with overflow_saturate ['out_window']: saturates to the HW value
# (HW answer 2026-11-01: mantissa all ones at SE = 15, 127 * 2^15 on the
# output grid), while the accumulator, the exponent register and the int
# input keep raising
# ---------------------------------------------------------------------------

SAT = ["out_window"]
HW_MAX = 127 * 2 ** 15            # int8 mantissa all ones at the top 4b code


def test_window_sat_word_hand_vector():
    assert window_sat_word("int8", 22) == HW_MAX
    assert window_sat_word("int8", 23) == 127 * 2 ** 16
    assert window_sat_word("int4", 22) == 7 * 2 ** 19
    with pytest.raises(ValueError, match="narrower"):
        window_sat_word("int8", 6)
    assert out_window_sat_word(normalize_xblock_accum({}), "int8") is None
    on = normalize_xblock_accum({"overflow": "raise", "overflow_saturate": SAT})
    assert out_window_sat_word(on, "int8") == HW_MAX


def test_window_params_sat_word():
    # Detection stays at 2^22 - 1 (pre_hi); only the clamp value moves.
    assert _window_params(0, 3, 22, "floor") == (3, 0, 0, 2 ** 22 - 1, 2 ** 22 - 1)
    assert _window_params(0, 3, 22, "floor", HW_MAX) == (3, 0, 0, HW_MAX, 2 ** 22 - 1)
    assert _window_params(5, 3, 22, "floor", HW_MAX)[3:] == (HW_MAX, (2 ** 22 - 1) >> 2)
    for bad in (0, 2 ** 22):
        with pytest.raises(ValueError, match="saturation word"):
            _window_params(0, 0, 22, "floor", bad)
    from fixed_point.mx_fixed_point_hw_triton import _window_launch_args
    _, args, on = _window_launch_args(0, 3, 22, "nearest", 1, 1, "cpu", HW_MAX)
    assert on == 1 and args[:5] == (3, 0, 4, HW_MAX, 2 ** 22 - 1)


def test_apply_out_window_sat_word():
    acc = torch.tensor([2 ** 22, 2 ** 22 - 1, HW_MAX + 1, -(2 ** 22), -(2 ** 22 - 1), 10 ** 9])
    v, sat = _apply_out_window(acc, 0, 0, 22, sat_word=HW_MAX)
    assert v.tolist() == [HW_MAX, 2 ** 22 - 1, HW_MAX + 1, -HW_MAX, -(2 ** 22 - 1), HW_MAX]
    assert sat.tolist() == [True, False, False, True, False, True]
    # Left shift 2: pre_hi = (2^22 - 1) >> 2 catches it before the shift.
    edge = (2 ** 22 - 1) >> 2
    v, sat = _apply_out_window(torch.tensor([edge, edge + 1, -edge - 1]), 5, 3, 22,
                               sat_word=HW_MAX)
    assert v.tolist() == [edge << 2, HW_MAX, -HW_MAX]
    # out_round nearest, rshift 1: 2^23 - 1 rounds up to 2^22, past the edge.
    v, sat = _apply_out_window(torch.tensor([2 ** 23 - 3, 2 ** 23 - 1, -(2 ** 23 + 1)]),
                               0, 1, 22, out_round="nearest", sat_word=HW_MAX)
    assert v.tolist() == [2 ** 22 - 1, HW_MAX, -HW_MAX]
    assert sat.tolist() == [False, True, True]


def test_m3_saturate_does_not_raise_and_counts():
    x = torch.ones(1, 32, 1, 1)                 # O = 2^22: one past the window
    m = _m3_model(-17, overflow_saturate=SAT)
    assert _run(m, x).flatten().tolist() == [HW_MAX * 2.0 ** -17] * 2
    assert m.c1._win_sat_seen_life == 2                       # still counted
    assert _run(m, x, train=True).flatten().tolist() == [32.0, 32.0]


def test_m3_saturate_left_shift_regime():
    # e_qa_out = e_layer_min - 2: acc = 32 * 2^20 = 2^25, shifted left by 2.
    x = torch.ones(1, 32, 1, 1)
    with pytest.raises(RuntimeError, match="22b output window"):
        _run(_m3_model(-22), x)
    out = _run(_m3_model(-22, overflow_saturate=SAT), x).flatten().tolist()
    assert out == [HW_MAX * 2.0 ** -22] * 2


def test_m3_saturate_keeps_acc_raise():
    m, _ = _model(e_layer_min=-60, out_window="infer", overflow="raise",
                  overflow_saturate=SAT,
                  hw_grids={"c1": {"e_layer_min": -60, "e_qa_out": -30}})
    _set_w(m, torch.ones(2, 32, 1, 1))          # 48b accumulator overflows
    with pytest.raises(RuntimeError, match="overflowed the 48b accumulator"):
        _run(m, torch.ones(1, 32, 1, 1))


def test_m3_saturate_keeps_register_and_int_input_raise():
    m = _m6_model(in_exp_anchor="min", overflow="raise", overflow_saturate=SAT)
    calibrate_hw_grids(m, _m6_data(), num_batches=1, verbose=0)   # top 2^12
    hi = torch.zeros(1, 32, 1, 2)
    hi[0, :, 0, 0], hi[0, :, 0, 1] = 2.0 ** -3, 2.0 ** 13
    with pytest.raises(RuntimeError, match="exponent register"):
        _run(m, hi)
    x = torch.tensor([12.86, 0.0, 0.0]).view(1, 3, 1, 1)
    with pytest.raises(RuntimeError, match="outside int8"):
        _run(_m8_model(0.1, overflow="raise", overflow_saturate=SAT), x)


# Accumulator words at the edges (on the output grid), the window word each
# must give, and what the next layer's MX requant must read, by hand: words
# past 2^22 - 1 saturate to 127 * 2^15; in-range words keep their value, block
# exponent 21, step 2^15: nearest rounds then clamps the mantissa to 127,
# truncate drops the bits. 2^22 - 1 itself is left out: microxcaling's fp32
# floor(log2) reads it as 2^22 at large |e_qa_out| (upstream, open question).
_EDGE = [2 ** 22 + 5, 2 ** 22, HW_MAX + 1, HW_MAX, HW_MAX - 1]
_EDGE = _EDGE + [-w for w in _EDGE]
_WORD = [HW_MAX, HW_MAX, HW_MAX + 1, HW_MAX, HW_MAX - 1]
_WANT = {"nearest": [HW_MAX] * 5, "truncate": [HW_MAX] * 4 + [126 * 2 ** 15]}


class _Two(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.c1 = nn.Conv2d(32, n, 1)
        self.c2 = nn.Conv2d(n, 1, 1, bias=False)

    def forward(self, x):
        return self.c2(self.c1(x))


@pytest.mark.parametrize("e_out", [-40, -30, -17, -13, 0])
@pytest.mark.parametrize("mant_round", ["nearest", "truncate"])
def test_m3_saturated_word_reaches_next_layer_as_hw_max(mant_round, e_out):
    """Two real layers, NPE blockify, overflow 'raise' with the window
    saturating: c1's window word (one channel = one W-block of c2) is read by
    c2 as +-127 * 2^15 on c1's output grid for every word past the window,
    at every grid (the fp32 log2 of the old +-(2^22 - 1) clamp rounds up
    outside e_qa_out -29..-14)."""
    n = len(_EDGE)
    net = _Two(n)
    tmp = tempfile.mkdtemp()
    xb = {"enabled": True, "mode": "hw_fixed_point", "bits": 48, "backend": "python",
          "sat_mode": "per_product", "verbose": 0, "out_window": "infer",
          "out_round": "nearest", "weight_blockify": "flatten", "act_blockify": "xblock",
          "mant_round": mant_round, "overflow": "raise", "overflow_saturate": SAT}
    cfg = {"mx_specs": {"w_elem_format": "int8", "a_elem_format": "int8",
                        "block_size": 32, "custom_cuda": False, "xblock_accum": xb},
           "layers": ["c1", "c2"], "ptq": False, "measure_error": False,
           "hw_grids": {"c1": {"e_layer_min": e_out - 3, "e_qa_out": e_out},
                        "c2": {"e_layer_min": e_out - 3, "e_qa_out": e_out + 3}}}
    with open(os.path.join(tmp, "mx_config.json"), "w") as f:
        json.dump(cfg, f)
    m = MXQuantizer(save_dir=tmp).quant(net)
    # c1: word_o = (sign_o * 31.5 * 2^17 + bias_o * 2^-e_out) = _EDGE[o], exact:
    # x = 31.5 * 2^(e_out + 17) is on the input's MX grid, the bias a multiple
    # of 2^e_out on the accumulator grid 2^(e_out - 3).
    sign = [1.0 if v > 0 else -1.0 for v in _EDGE]
    w = torch.zeros(n, 32, 1, 1)
    w[:, 0, 0, 0] = torch.tensor(sign)
    b = torch.tensor([(v - s * 31.5 * 2 ** 17) * 2.0 ** e_out for v, s in zip(_EDGE, sign)],
                     dtype=torch.float64)
    with torch.no_grad():
        m.c1.weight.copy_(w)
        m.c1.bias.copy_(b)
        m.c2.weight.fill_(1.0)
    assert torch.equal(m.c1.bias.double(), b)                 # fp32 holds it exactly
    m.c1._tap_acc = m.c2._tap_acc = True
    x = torch.zeros(1, 32, 1, 1)
    x[0, 0] = 31.5 * 2.0 ** (e_out + 17)
    _run(m, x)
    assert m.c1._last_win.flatten().tolist() == _WORD + [-v for v in _WORD]
    assert m.c1._win_sat_seen_life == 4                       # +-2^22, +-(2^22 + 5)
    got = (m.c2._last_qi.flatten().double() * 2.0 ** -e_out).tolist()
    want = _WANT[mant_round]
    assert got == want + [-v for v in want]                   # symmetric


# ---------------------------------------------------------------------------
# M4 output rounding: round half up, ties toward +inf (Q4)
# ---------------------------------------------------------------------------


def test_m4_nearest_hand_vector():
    # rshift 1: 2.5 -> 3, -1.5 -> -1, 1.5 -> 2, -2.5 -> -2, -0.5 -> 0
    acc = torch.tensor([5, -3, 3, -5, -1])
    v, _ = _apply_out_window(acc, 0, 1, 22, out_round="nearest")
    assert v.tolist() == [3, -1, 2, -2, 0]
    v, _ = _apply_out_window(acc, 0, 1, 22, out_round="floor")
    assert v.tolist() == [2, -2, 1, -3, -1]


def test_m4_nearest_equals_emulator_pcu_div():
    hw_pipe, _ = _hw_pipe()
    g = torch.Generator().manual_seed(0)
    acc = torch.randint(-2 ** 30, 2 ** 30, (4000,), generator=g, dtype=torch.int64)
    acc[:8] = torch.tensor([5, -3, 3, -5, 6, -6, 7, -7])
    for k in (1, 2, 5, 13):
        ours, _ = _apply_out_window(acc, 0, k, 46, out_round="nearest")
        theirs = hw_pipe.pcu_div(acc, shift_right=-k, round_type="round")
        assert torch.equal(ours, theirs), k


# ---------------------------------------------------------------------------
# M5 bias rounding floor(v + 0.5) (Q4)
# ---------------------------------------------------------------------------


def test_m5_bias_helper_hand_vector():
    b = torch.tensor([2.5, -1.5, -2.5, 0.5, -0.5, 1.25])
    assert _bias_to_grid(b, 0, False, "cpu").tolist() == [2, -2, -2, 0, 0, 1]
    assert _bias_to_grid(b, 0, True, "cpu").tolist() == [3, -1, -2, 1, 0, 1]


def test_m5_bias_matches_emulator_c_struct():
    _, calib = _hw_pipe()
    b = torch.tensor([2.5, -1.5, -2.5, 0.37, -7.75])
    qp = calib.LayerMXQParams(x_cal=1, w_cal=2)          # bias units 2^-3
    hw = torch.stack([calib.build_c_struct(qp, v).bias for v in b])
    assert torch.equal(_bias_to_grid(b, -3, True, "cpu"), hw.flatten())


def _m5_model(**kw):
    model, _ = _model(e_layer_min=0, **kw)
    _set_w(model, torch.zeros(2, 32, 1, 1), torch.tensor([2.5, -1.5]))
    return model


def test_m5_bias_round_in_layer():
    x = torch.ones(1, 32, 1, 1)
    assert _run(_m5_model(), x).flatten().tolist() == [2.0, -2.0]
    assert _run(_m5_model(bias_round="half_up"), x).flatten().tolist() == [3.0, -1.0]


# ---------------------------------------------------------------------------
# M6 exponent register anchored at the min non-zero block (Q2, Q6)
# ---------------------------------------------------------------------------


def _m6_data():
    """Two pixels with block max 2^-3 and 2^4, the rest zero blocks."""
    x = torch.zeros(1, 32, 1, 3)
    x[0, :, 0, 0] = 2.0 ** -3
    x[0, :, 0, 1] = 2.0 ** 4
    return [x]


def _m6_model(**kw):
    kw = {"in_exp_mode": "infer", **kw}
    model, _ = _model(in_exp_bits=4, e_layer_min=-20, **kw)
    _set_w(model, torch.full((2, 32, 1, 1), 0.5))
    return model


def test_block_exp_range_skips_zero_blocks():
    assert block_exp_range(_m6_data()[0], 32, "channel") == (-3, 4)
    assert block_exp_range(torch.zeros(1, 32, 1, 1), 32, "channel") == (None, None)


def test_m6_min_anchor_calibration():
    a = _m6_model(in_exp_anchor="min", overflow="raise")
    calibrate_hw_grids(a, _m6_data(), num_batches=1, verbose=0)
    assert a.c1.e_base_a == -3                         # HW: min non-zero block
    assert a.c1.weight_exp_base() == -1                # 0.5 -> E = -1
    b = _m6_model()
    calibrate_hw_grids(b, _m6_data(), num_batches=1, verbose=0)
    assert b.c1.e_base_a == 4 - 15                     # ALG: max - 15


def test_m6_eval_raises_outside_register():
    m = _m6_model(in_exp_anchor="min", overflow="raise")
    calibrate_hw_grids(m, _m6_data(), num_batches=1, verbose=0)   # base -3, top 12
    ok = torch.zeros(1, 32, 1, 2)
    ok[0, :, 0, 0], ok[0, :, 0, 1] = 2.0 ** -3, 2.0 ** 12
    _run(m, ok)
    hi = ok.clone()
    hi[0, :, 0, 1] = 2.0 ** 13
    with pytest.raises(RuntimeError, match="exponent register"):
        _run(m, hi)
    lo = ok.clone()
    lo[0, :, 0, 0] = 2.0 ** -4
    with pytest.raises(RuntimeError, match="exponent register"):
        _run(m, lo)
    _run(m, hi, train=True)                            # train: graceful (infer only)


def test_m6_train_stays_max_anchored():
    m = _m6_model(in_exp_anchor="min", in_exp_mode="always")
    calibrate_hw_grids(m, _m6_data(), num_batches=1, verbose=0)
    x = torch.zeros(1, 32, 1, 2)
    x[0, :, 0, 0], x[0, :, 0, 1] = 65 * 2.0 ** -18, 2.0 ** 4   # E = -12 and 4
    m.train()
    with torch.no_grad():
        y = m(x)
    # Live max anchor 4 - 15 = -11 (step 2^-17): 32.5 steps -> 33 (half away).
    # The calibrated min anchor (-3, step 2^-9) would have zeroed it.
    assert m.c1._in_exp_cut_life[0] == 32
    assert y[0, 0, 0, 0].item() == 16 * 33 * 2.0 ** -17


def test_m6_calibration_contract_raise():
    x = torch.zeros(1, 32, 1, 2)
    x[0, :, 0, 0], x[0, :, 0, 1] = 2.0 ** -20, 2.0 ** 4      # 25 codes > 16
    m = _m6_model(in_exp_anchor="min", overflow="raise")
    with pytest.raises(ValueError, match="span more than the exponent register"):
        calibrate_hw_grids(m, [x], num_batches=1, verbose=0)
    m = _m6_model(in_exp_anchor="min")                         # clamp: no raise
    calibrate_hw_grids(m, [x], num_batches=1, verbose=0)


# ---------------------------------------------------------------------------
# M7 PCU static shift (Q3)
# ---------------------------------------------------------------------------


def test_m7_static_shift_hand_vector():
    assert static_shift_of(-20) == 8          # 2^-20 = 2^-12 grid shifted left 8
    assert static_shift_of(-12) == 0
    assert static_shift_of(-4) == -8
    assert check_static_shift(-75, 7) == 63 and check_static_shift(52, 7) == -64
    for bad in (-76, 53):
        with pytest.raises(ValueError, match="7b signed"):
            check_static_shift(bad, 7)


def test_m7_static_shift_matches_emulator_pcu():
    """pcu_div(acc, shift_right = s + 2*mant_bias + E_acc) (emulator pcu with
    e_min_layer_x + e_min_layer_w = 12 + E_acc) equals our window word."""
    hw_pipe, _ = _hw_pipe()
    g = torch.Generator().manual_seed(1)
    acc = torch.randint(-2 ** 40, 2 ** 40, (2000,), generator=g, dtype=torch.int64)
    for e_acc, e_out in ((-30, -20), (-40, -18), (-25, -27)):
        ours, _ = _apply_out_window(acc, e_acc, e_out, 46, out_round="nearest")
        s = static_shift_of(e_out)
        theirs = hw_pipe.pcu_div(acc, shift_right=s + 12 + e_acc, round_type="round")
        assert torch.equal(ours, theirs), (e_acc, e_out)


def test_m7_export_and_load():
    m, tmp = _model(out_window="infer", static_shift_bits=7)
    g = torch.Generator().manual_seed(0)
    data = [torch.randn(1, 32, 2, 2, generator=g)]
    calibrate_hw_grids(m, data, num_batches=1, verbose=0)
    entry = export_hw_grids(m)["c1"]
    assert entry["static_shift"] == -(entry["e_qa_out"] + 12)
    bad = dict(entry, static_shift=entry["static_shift"] + 1)
    with pytest.raises(ValueError, match="static_shift"):
        _model(out_window="infer", hw_grids={"c1": bad})
    _model(out_window="infer", hw_grids={"c1": entry})       # round trip loads


def test_m7_forward_range_check():
    m, _ = _model(out_window="infer", static_shift_bits=7,
                  hw_grids={"c1": {"e_layer_min": -90, "e_qa_out": -80}})
    with pytest.raises(ValueError, match="static shift 68"):
        _run(m, torch.ones(1, 32, 1, 1))


# ---------------------------------------------------------------------------
# M8 first layer: per-tensor int8 input (Q5)
# ---------------------------------------------------------------------------


def test_m8_scale_hand_vector():
    assert in_int_scale_from_absmax(2.54) == pytest.approx(0.02)
    assert in_int_scale_from_absmax(2.54, "pot") == 2.0 ** -5     # 2.54 / 2^-5 = 81
    assert in_int_scale_from_absmax(1.0, "pot") == 2.0 ** -6      # 64 <= 127
    assert in_int_scale_from_absmax(0.0) == 1.0


def _m8_model(scale, **kw):
    net = _Net(cin=3, cout=2, k=1)
    m, _ = _model(net=net, e_layer_min=-40, in_mode="int",
                  hw_grids={"c1": {"e_layer_min": -40, "e_qa_out": -30,
                                   "in_scale": scale}}, **kw)
    w = torch.zeros(2, 3, 1, 1)
    w[0, 0], w[1, 1], w[1, 2] = 1.0, 0.5, -0.25
    _set_w(m, w)
    return m


def test_m8_int_input_hand_vector():
    # scale 0.1: 0.25 -> round(2.5) = 2 (torch.round, as the emulator), 0.36
    # -> 4, -0.04 -> 0; out0 = 2 * 0.1, out1 = 0.5 * 0.4 - 0.25 * 0 = 0.2
    x = torch.tensor([0.25, 0.36, -0.04]).view(1, 3, 1, 1)
    # The scale is folded into the weights before MX: W * 0.1 -> 0.1 is 102
    # steps of 2^-10, 0.05 is 102 of 2^-11 (-0.025 is -51 of 2^-11).
    y = _run(_m8_model(0.1), x).flatten()
    assert y.tolist() == [2 * 102 * 2.0 ** -10, 4 * 102 * 2.0 ** -11]
    pot = _run(_m8_model(2.0 ** -3), x).flatten()      # 2, 2.88 -> 3, -0.32 -> 0
    assert pot.tolist() == [2 * 2.0 ** -3, 0.5 * 3 * 2.0 ** -3]


def test_m8_out_of_range_raises_in_eval():
    x = torch.tensor([12.86, 0.0, 0.0]).view(1, 3, 1, 1)  # 129 at scale 0.1
    with pytest.raises(RuntimeError, match="outside int8"):
        _run(_m8_model(0.1, overflow="raise"), x)
    y = _run(_m8_model(0.1), x).flatten()                  # default clamps to 127
    assert y[0].item() == 127 * 102 * 2.0 ** -10


def test_m8_no_block_exponent_on_input():
    """With the 4b register on, the int input is not clamped (HW has no input
    block exponent): a 2^-10 and a 2^6 value in one tensor stay exact."""
    m = _m8_model(1.0, in_exp_bits=4, in_exp_mode="infer")
    m.c1.e_base_a = 100                                    # would cut everything
    x = torch.tensor([3.0, 100.0, 1.0]).view(1, 3, 1, 1)
    y = _run(m, x).flatten().tolist()
    assert y == [3.0, 50.0 - 0.25] and m.c1._in_exp_cut_life[0] == 0


def test_m8_calibrate_export_round_trip():
    net = _Net(cin=3, cout=4, k=3)
    g = torch.Generator().manual_seed(2)
    data = [torch.rand(2, 3, 6, 6, generator=g) * 2.54 for _ in range(2)]
    m, tmp = _model(net=net, out_window="infer", in_mode="int", data=data,
                    top={"calibrate_hw_grids": {"batches": 2, "verbose": 0}})
    amax = max(float(d.abs().max()) for d in data)
    assert m.c1.in_scale == pytest.approx(amax / 127)
    path = os.path.join(tmp, "mx_config_grids.json")
    entry = load_hw_grids(path)["c1"]
    assert entry["in_scale"] == m.c1.in_scale
    x = data[0]
    y = _run(m, x)
    m2 = MXQuantizer(save_dir=tmp, config_path=path).quant(_Net(cin=3, cout=4, k=3))
    assert torch.equal(_run(m2, x), y)
    # FP32 reference of the int input: small error only
    ref = F.conv2d(torch.round(x / m.c1.in_scale) * m.c1.in_scale,
                   _Net(cin=3, cout=4, k=3).c1.weight, _Net(cin=3, cout=4, k=3).c1.bias,
                   padding=1)
    assert F.cosine_similarity(y.flatten(), ref.flatten(), dim=0) > 0.999


def test_m8_grads_flow():
    m = _m8_model(0.1)
    x = torch.tensor([0.25, 0.36, -0.04]).view(1, 3, 1, 1).requires_grad_(True)
    m.train()
    m(x).sum().backward()
    # STE through q = round(x / s) and the folded weight MX(W * s): dL/dx =
    # sum_o MX(W * s)[o, c] / s
    want = torch.tensor([102 * 2.0 ** -10, 102 * 2.0 ** -11, -51 * 2.0 ** -11]) / 0.1
    assert torch.allclose(x.grad.flatten(), want)
    assert m.c1.weight.grad is not None and m.c1.weight.grad.abs().sum() > 0


def test_m8_needs_scale():
    net = _Net(cin=3, cout=2, k=1)
    m, _ = _model(net=net, e_layer_min=-40, in_mode="int")
    with pytest.raises(RuntimeError, match="in_scale"):
        _run(m, torch.ones(1, 3, 1, 1))


def test_hw_grid_in_scale_validation():
    check_hw_grid("c1", {"e_layer_min": -30, "e_qa_out": -19, "in_scale": 0.02,
                         "static_shift": 7})
    for bad in (0.0, -1.0, True, "0.1"):
        with pytest.raises(TypeError):
            check_hw_grid("c1", {"e_layer_min": -30, "e_qa_out": -19, "in_scale": bad})
    with pytest.raises(TypeError):
        check_hw_grid("c1", {"e_layer_min": -30, "e_qa_out": -19, "static_shift": 1.0})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_defaults_and_validation():
    d = normalize_xblock_accum({})
    assert (d["mant_round"], d["overflow"], d["bias_round"], d["in_exp_anchor"],
            d["static_shift_bits"], d["in_mode"], d["in_int_scale"]) == \
        ("nearest", "clamp", "even", "max", None, "mx", "free")
    assert d["overflow_saturate"] == ()
    assert normalize_xblock_accum({"overflow": "raise", "overflow_saturate": ["out_window"]})[
        "overflow_saturate"] == ("out_window",)
    for bad in [{"mant_round": "floor"}, {"overflow": "wrap"}, {"bias_round": "up"},
                {"in_exp_anchor": "mid"}, {"static_shift_bits": 1},
                {"static_shift_bits": True}, {"in_mode": "fp"},
                {"in_int_scale": "pow2"},
                {"overflow_saturate": ["out_window"]}]:      # needs overflow 'raise'
        with pytest.raises(ValueError):
            normalize_xblock_accum(bad)
    for bad in ["out_window", ["acc"], None, ["out_window", "out_window"]]:
        with pytest.raises(ValueError, match="overflow_saturate must"):
            normalize_xblock_accum({"overflow": "raise", "overflow_saturate": bad})


def _config_specs():
    """{config file: [every xblock_accum dict in it (mx_specs + groups)]}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(REPO, "configs", "*.json"))):
        with open(path) as f:
            cfg = json.load(f)
        specs = [cfg.get("mx_specs", {})] + list(cfg.get("groups", {}).values())
        out[os.path.basename(path)] = [s["xblock_accum"] for s in specs
                                       if isinstance(s.get("xblock_accum"), dict)]
    return out


HW_EXACT_CONFIGS = ["mx_config_dof_run_hw_exact.json", "mx_config_npe_hw_exact.json",
                    "mx_config_npe_hw_exact_in.json", "mx_config_npe_hw_exact_nearest.json"]


def test_window_saturates_only_in_hw_exact_configs():
    """The four HW-exact configs raise on overflow except the window, which
    saturates as HW; every other config keeps the default (no key)."""
    specs = _config_specs()
    assert len(specs) > 30 and all(n in specs for n in HW_EXACT_CONFIGS)
    for name, xbs in specs.items():
        for xb in xbs:
            n = normalize_xblock_accum(xb)
            if name in HW_EXACT_CONFIGS:
                assert (n["overflow"], n["overflow_saturate"]) == \
                    ("raise", ("out_window",)), name
            else:
                assert "overflow_saturate" not in xb, name
    assert all(specs[n] for n in HW_EXACT_CONFIGS)


def test_hw_exact_config():
    with open(os.path.join(REPO, "configs", "mx_config_dof_run_hw_exact.json")) as f:
        cfg = json.load(f)
    xb = normalize_xblock_accum(cfg["mx_specs"]["xblock_accum"])
    assert (xb["mant_round"], xb["overflow"], xb["bias_round"], xb["out_round"],
            xb["in_exp_anchor"], xb["static_shift_bits"], xb["pla"],
            xb["pla_bias"]) == ("truncate", "raise", "half_up", "nearest", "min", 7,
                                "infer", "bins")
    assert xb["in_exp_bits"] == 4 and xb["in_exp_mode"] == "infer"
    first = cfg["groups"]["input_layer"]["xblock_accum"]
    assert normalize_xblock_accum(first)["in_mode"] == "int"


def test_wide_register_does_not_overflow_fp32():
    q = torch.tensor([3.0, -1.0])
    y, n = clamp_block_exp(q, -100, 8)          # top 2^155 > fp32 max
    assert torch.equal(y, q) and n == 0
