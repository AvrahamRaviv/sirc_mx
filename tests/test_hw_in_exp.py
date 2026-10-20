"""Accumulator range on wide-range inputs: the e_layer_min headroom floor
(acc_headroom_bits) and the HW a/w shared-exponent register (in_exp_bits).

The DOF run hit this: an input whose blocks span ~37 exponent codes
calibrated e_layer_min so low that the 48b accumulator saturated on every
output. Both options raise e_layer_min until the largest values fit."""
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
from fixed_point.mx_fixed_point_hw import (calibrate_hw_grids, clamp_block_exp,
                                           floor_log2)
from mx_quantizer import MXQuantizer


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.c1 = nn.Conv2d(32, 8, 3, padding=1)

    def forward(self, x):
        return self.c1(x)


def _data(n=3, seed=1):
    """Inputs with a few tiny rows: their blocks sit ~40 codes below the rest."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(2, 32, 8, 8, generator=g) * 3
        x[..., :3, :] *= 2.0 ** -40
        out.append(x)
    return out


def _model(tmp=None, hw_grids=None, top=None, **xblock):
    tmp = _write_cfg(tmp, hw_grids, top, **xblock)
    return MXQuantizer(save_dir=tmp).quant(_Net()), tmp


def _write_cfg(tmp=None, hw_grids=None, top=None, **xblock):
    tmp = tmp or tempfile.mkdtemp()
    xb = {"enabled": True, "mode": "hw_fixed_point", "bits": 48,
          "backend": "python", "sat_mode": "per_product", "e_layer_min": None,
          "out_window": "infer", "verbose": 0, **xblock}
    cfg = {"mx_specs": {"w_elem_format": "int8", "a_elem_format": "int8",
                        "block_size": 32, "custom_cuda": False,
                        "xblock_accum": xb},
           "layers": ["c1"], "ptq": False, "measure_error": False, **(top or {})}
    if hw_grids is not None:
        cfg["hw_grids"] = hw_grids
    with open(os.path.join(tmp, "mx_config.json"), "w") as f:
        json.dump(cfg, f)
    return tmp


def _cos(model, x):
    model.eval()
    with torch.no_grad():
        y, ref = model(x), _Net()(x)
    return float(F.cosine_similarity(y.flatten(), ref.flatten(), dim=0))


# ---------------------------------------------------------------------------
# clamp_block_exp
# ---------------------------------------------------------------------------

def test_clamp_rerounds_block_below_base():
    # q = 100 at E = -20 (value 100 * 2^-26); base -15 -> step 2^-21 -> 3.
    q = torch.tensor([100 * 2.0 ** -26, -100 * 2.0 ** -26])
    y, n = clamp_block_exp(q, -15, 4)
    assert y.tolist() == [3 * 2.0 ** -21, -3 * 2.0 ** -21] and n == 2


def test_clamp_saturates_block_above_top():
    # base 0, 4 bits -> top 15; a block at E = 20 clamps to 127 * 2^(15-6).
    # A small element of that block (3 * 2^14 = 96 top steps) fits as is.
    q = torch.tensor([127 * 2.0 ** 14, 3 * 2.0 ** 14])
    y, n = clamp_block_exp(q, 0, 4)
    assert y.tolist() == [127 * 2.0 ** 9, 3 * 2.0 ** 14] and n == 1


def test_clamp_leaves_in_range_blocks():
    g = torch.Generator().manual_seed(0)
    q = torch.randint(-127, 128, (64,), generator=g).float()
    E = torch.randint(-15, 1, (64,), generator=g).float()   # within [-15, 0]
    q = q * 2.0 ** (E - 6)
    y, n = clamp_block_exp(q, -15, 4)
    assert torch.equal(y, q) and n == 0


def test_floor_log2():
    assert floor_log2(1.0) == 0 and floor_log2(1.99) == 0
    assert floor_log2(2 ** -20) == -20 and floor_log2(0.0) is None


# ---------------------------------------------------------------------------
# The DOF failure and both fixes
# ---------------------------------------------------------------------------

def test_wide_input_without_floor_saturates():
    model, _ = _model()
    calibrate_hw_grids(model, _data(), num_batches=3, verbose=0)
    assert model.c1.e_qa_out - model.c1.e_layer_min > 26     # output needs > 48b
    assert _cos(model, _data(1, seed=9)[0]) < 0.5
    assert model.c1._sat_seen_life > 0


def test_headroom_floor_fixes_saturation():
    model, _ = _model(acc_headroom_bits=2)
    res = calibrate_hw_grids(model, _data(), num_batches=3, verbose=0)["c1"]
    assert res["e_layer_min"] > res["e_qa_in"] + res["e_qw"]   # floored
    assert _cos(model, _data(1, seed=9)[0]) > 0.999
    assert model.c1._sat_seen_life == 0


def test_headroom_floor_bound():
    """e_layer_min = top - (bits-1) + headroom, top = Ea_max + Ew_max + 2."""
    model, _ = _model(acc_headroom_bits=3)
    calibrate_hw_grids(model, _data(), num_batches=3, verbose=0)
    xs = _data()
    ea = max(floor_log2(float(x.abs().max())) for x in xs)
    ew = floor_log2(float(model.c1.weight.detach().abs().max()))
    assert model.c1.e_layer_min >= ea + ew + 2 - 47 + 3 - 1   # MX rounding slack


def test_headroom_noop_on_narrow_input():
    g = torch.Generator().manual_seed(1)
    narrow = [torch.randn(2, 32, 8, 8, generator=g) for _ in range(3)]
    a, _ = _model()
    b, _ = _model(acc_headroom_bits=2)
    ra = calibrate_hw_grids(a, narrow, num_batches=3, verbose=0)["c1"]
    rb = calibrate_hw_grids(b, narrow, num_batches=3, verbose=0)["c1"]
    assert ra["e_layer_min"] == rb["e_layer_min"]


def test_in_exp_fixes_saturation_and_cuts_tiny_blocks():
    model, _ = _model(in_exp_bits=4, in_exp_mode="infer")
    res = calibrate_hw_grids(model, _data(), num_batches=3, verbose=0)["c1"]
    m = model.c1
    assert res["a_codes"] > 16                                  # needed > 4b
    assert m.e_base_a == floor_log2(max(float(x.abs().max()) for x in _data())) - 15
    assert m.e_layer_min >= m.e_base_a + res["e_base_w"] - 12
    x = _data(1, seed=9)[0]
    assert _cos(model, x) > 0.999
    assert m._sat_seen_life == 0
    assert m._in_exp_cut_life[0] > 0                            # tiny rows cut


def test_in_exp_infer_skips_train_and_keeps_grads():
    model, _ = _model(in_exp_bits=4, in_exp_mode="infer")
    calibrate_hw_grids(model, _data(), num_batches=3, verbose=0)
    model.train()
    x = _data(1)[0].requires_grad_(True)
    model(x).sum().backward()
    assert model.c1._in_exp_cut_life == [0, 0]
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_in_exp_always_is_ste():
    model, _ = _model(in_exp_bits=4, in_exp_mode="always")
    calibrate_hw_grids(model, _data(), num_batches=3, verbose=0)
    model.train()
    x = _data(1)[0].requires_grad_(True)
    model(x).sum().backward()
    assert model.c1._in_exp_cut_life[0] > 0
    assert x.grad[..., :3, :].abs().sum() > 0     # cut rows still get grads


def test_in_exp_needs_base():
    model, _ = _model(in_exp_bits=4, in_exp_mode="always",
                      hw_grids={"c1": {"e_layer_min": -40, "e_qa_out": -18}})
    with pytest.raises(RuntimeError, match="e_base_a"):
        _cos(model, _data(1)[0])


def test_in_exp_grids_round_trip():
    tmp = _write_cfg(in_exp_bits=4, in_exp_mode="infer",
                     top={"calibrate_hw_grids": {"batches": 3, "verbose": 0}})
    model = MXQuantizer(save_dir=tmp).quant(_Net(), data=_data())
    path = os.path.join(tmp, "mx_config_grids.json")
    entry = load_hw_grids(path)["c1"]
    assert entry["e_base_a"] == model.c1.e_base_a
    assert entry["e_base_w"] == model.c1.weight_exp_base()
    x = _data(1, seed=9)[0]
    model.eval()
    with torch.no_grad():
        y = model(x)
    model2 = MXQuantizer(save_dir=tmp, config_path=path).quant(_Net())
    assert model2.c1.e_base_w == entry["e_base_w"]
    model2.eval()
    with torch.no_grad():
        assert torch.equal(model2(x), y)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_validation():
    cfg = normalize_xblock_accum({"in_exp_bits": 4, "in_exp_mode": "infer",
                                  "acc_headroom_bits": 2})
    assert cfg["in_exp_bits"] == 4
    d = normalize_xblock_accum({})
    assert (d["in_exp_bits"], d["in_exp_mode"], d["acc_headroom_bits"]) == (None, "off", None)
    for bad in [{"in_exp_bits": 0}, {"in_exp_bits": 9}, {"in_exp_mode": "eval"},
                {"in_exp_mode": "infer"}, {"acc_headroom_bits": -1},
                {"acc_headroom_bits": True}]:
        with pytest.raises(ValueError):
            normalize_xblock_accum(bad)


def test_hw_grid_accepts_bases():
    check_hw_grid("c1", {"e_layer_min": -30, "e_qa_out": -19,
                         "e_base_a": -12, "e_base_w": -20})
    with pytest.raises(TypeError):
        check_hw_grid("c1", {"e_layer_min": -30, "e_qa_out": -19, "e_base_a": 1.0})


@pytest.mark.parametrize("name,headroom,exp_bits", [
    ("mx_config_dof_run_win.json", 2, None),
    ("mx_config_int8_hw_win.json", 2, None),
    ("mx_config_dof_run_win_se4.json", 2, 4),
])
def test_win_configs(name, headroom, exp_bits):
    with open(os.path.join(REPO, "configs", name)) as f:
        cfg = normalize_xblock_accum(json.load(f)["mx_specs"]["xblock_accum"])
    assert cfg["acc_headroom_bits"] == headroom
    assert cfg["in_exp_bits"] == exp_bits
