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
    t = HW.run(C.build(name))
    assert t["out"].shape == ALG.run(C.build(name))["out"].shape
    assert t["hw_acc"].shape == t["out_raw"].shape


@pytest.mark.parametrize("name", ["conv3x3", "conv3x3_d2"])
def test_hw_conv_unsupported_raises(name):
    with pytest.raises(NotImplementedError):
        HW.run(C.build(name))


@pytest.mark.parametrize("name", ["convt_k3_s2", "warp_int_shift"])
def test_hw_placeholder(name):
    with pytest.raises(NotImplementedError):
        HW.run(C.build(name))
