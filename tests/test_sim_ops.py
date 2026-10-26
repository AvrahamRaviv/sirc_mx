"""Single-op simulator (configs/): every case runs on the ALG side, and its
integer taps explain its own output - conv acc from a_int/w_int, warp out_code
from ref_code/grid_code."""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(_ROOT, "configs"), _ROOT, os.path.dirname(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cases as C           # noqa: E402
import main_ALG as ALG      # noqa: E402
import main_HW as HW        # noqa: E402
import sim_config as SC     # noqa: E402

E4 = SC.load_config("mx_config_npe_e4.json")   # NPE exponent width (4 bits)


def _expand(code, axis, n, bs=C.BS):
    """Per-block exponent codes -> per-element real exponent along `axis`."""
    return (code.double() - 127).repeat_interleave(bs, dim=axis).narrow(axis, 0, n)


@pytest.mark.parametrize("name", C.select("conv"))
def test_conv_acc_from_taps(name):
    case = C.build(name)
    t = ALG.run(case)
    a = t["a_int"].double() * 2.0 ** (_expand(t["a_exp"], 3, t["a_int"].shape[3]) - 6)
    w_int = t["w_int"]
    O = w_int.shape[0]
    Ew = _expand(t["w_exp"], 1, w_int[0].numel()).reshape(w_int.shape)
    w = w_int.double() * 2.0 ** (Ew - 6)
    b = None
    if case["b"] is not None:   # bias lands on the accumulator grid
        b = torch.round(case["b"].double() * 2.0 ** 20) * 2.0 ** -20
    p = case["params"]
    want = F.conv2d(a, w, b, p["stride"], p["padding"], p["dilation"])
    assert int(t["sat"].sum()) == 0
    assert torch.equal(t["acc"].double(), want * 2.0 ** 20)


@pytest.mark.parametrize("name", C.select("warp"))
def test_warp_out_from_taps(name):
    case = C.build(name)
    t = ALG.run(case)
    rs, os_ = case["warp"]["ref_scale"], case["warp"]["out_scale"]
    pre = C.Warp()(t["ref_code"].double() * rs, t["grid_code"].double() / 256)
    v = pre / os_
    want = (torch.sign(v) * torch.floor(v.abs() + 0.5)).clamp(-128, 127)
    assert torch.equal(t["out_code"].double(), want)


@pytest.mark.parametrize("name", C.select("convt"))
def test_convt_runs(name):
    t = ALG.run(C.build(name))
    assert t["out"].shape[2:] == (10, 32)


@pytest.mark.parametrize("name", ["conv1x1", "conv3x3_p0"])
def test_hw_conv_runs(name):
    """HW team's npe_mx_emu conv, wired in: runs, shapes line up with ALG."""
    t = HW.run(C.build(name), E4)
    assert t["out"].shape == ALG.run(C.build(name))["out"].shape
    assert t["hw_acc"].shape == t["out_raw"].shape


@pytest.mark.parametrize("name", ["conv3x3", "conv3x3_d2"])
def test_hw_conv_unsupported_raises(name):
    with pytest.raises(NotImplementedError):
        HW.run(C.build(name), E4)


@pytest.mark.parametrize("name", ["convt_k3_s2", "warp_int_shift"])
def test_hw_placeholder(name):
    with pytest.raises(NotImplementedError):
        HW.run(C.build(name), E4)


def test_int_input_is_on_grid_and_hw_preround_is_noop():
    """--input int: x * 2^x_frac is an exact integer on both sides, so
    x * 2^x_cal is already integer while x_cal >= x_frac (HW rounds only an
    input layer now; either way nothing would move)."""
    case = C.build("conv3x3_p0", x_frac=4)
    v = case["x"].double() * 16
    assert bool((v == v.round()).all())
    t = HW.run(case, E4)
    assert int(t["x_cal"]) >= 4
    assert int(t["pre_round_moved"]) == 0


def test_exponent_width_comes_from_config():
    """scale_bits drives both sides: ALG codes are E + 2^(n-1)-1; HW picks its
    mx_type (and so its wrap mask) from it, and refuses a type it lacks."""
    c = C.build("conv3x3_p0")
    dof = SC.load_config("mx_config_dof.json")
    e8 = ALG.run(c, dof)
    e4 = ALG.run(c, E4)
    assert int(e8["a_exp"].min()) > 100          # 8-bit: E + 127
    assert int(e4["a_exp"].max()) <= 14          # 4-bit: E + 7, in [0, 14]
    assert bool((e8["a_exp"].long() - 127 == e4["a_exp"].long() - 7).all())
    assert HW.hw_format(E4) == ("mx-e4bm8", 32)
    assert HW.hw_format(dof) == ("mx-e8bm8", 32)
    e5 = {"mx_specs": dict(E4["mx_specs"], scale_bits=5)}
    with pytest.raises(NotImplementedError, match="not in npe_mx_emu"):
        HW.hw_format(e5)


def test_hw_acc_width_comes_from_config():
    """xblock_accum.bits sets their accumulator width; unsupported widths are
    refused."""
    assert HW.hw_acc_bits(E4) == 48
    t = HW.run(C.build("conv1x1"), E4)
    assert int(t["acc_bits"]) == 48
    xa = dict(E4["mx_specs"]["xblock_accum"], bits=35)
    with pytest.raises(NotImplementedError, match="supports"):
        HW.hw_acc_bits({"mx_specs": dict(E4["mx_specs"], xblock_accum=xa)})
