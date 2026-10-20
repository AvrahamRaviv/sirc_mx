"""Static HW grids in the config ("hw_grids"): load, priority over
calibration, export / reload, and the checks that keep inference from
silently using a wrong grid."""
import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects")
sys.path.insert(0, "/home/avrahamra/PycharmProjects")

from fixed_point.hw_grids import check_hw_grid, hw_grid_for, load_hw_grids
from fixed_point.mx_fixed_point import normalize_xblock_accum
from fixed_point.mx_fixed_point_hw import calibrate_hw_grids, export_hw_grids
from mx_layers_blocked import MXConv2dHW
from mx_quantizer import MXQuantizer


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.c1 = nn.Conv2d(32, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 8, 3, padding=1)

    def forward(self, x):
        return self.c2(torch.relu(self.c1(x)))


class _TwiceNet(_Net):
    def forward(self, x):
        return self.c2(torch.relu(self.c1(torch.relu(self.c1(x)))))


def _write_cfg(tmp, hw_grids=None, top=None, **xblock):
    xb = {"enabled": True, "mode": "hw_fixed_point", "bits": 48,
          "backend": "python", "sat_mode": "per_product",
          "e_layer_min": None, "out_window": "infer", "verbose": 0}
    xb.update(xblock)
    cfg = {"mx_specs": {"w_elem_format": "int8", "a_elem_format": "int8",
                        "block_size": 32, "custom_cuda": False,
                        "xblock_accum": xb},
           "layers": ["c1", "c2"], "ptq": False, "measure_error": False}
    if hw_grids is not None:
        cfg["hw_grids"] = hw_grids
    cfg.update(top or {})
    with open(os.path.join(tmp, "mx_config.json"), "w") as f:
        json.dump(cfg, f)
    return cfg


def _data(n=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, 32, 8, 8, generator=g) * 3 for _ in range(n)]


def _quant(tmp, net=None, **kw):
    return MXQuantizer(save_dir=tmp, **kw).quant(net or _Net())


def _eval(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)


GRID_C1 = {"e_layer_min": -30, "e_qa_out": -19, "win_bits": 22, "shift_out": 11}


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_export_reload_round_trip_bit_identical():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp)
    q = MXQuantizer(save_dir=tmp)
    model = q.quant(_Net())
    calibrate_hw_grids(model, _data(), num_batches=3, verbose=0)
    x = _data(1, seed=9)[0]
    y = _eval(model, x)

    path = q.save_hw_grids(model)
    assert path == os.path.join(tmp, "mx_config_grids.json")
    grids = load_hw_grids(path)
    assert set(grids) == {"c1", "c2"}
    for name, m in [("c1", model.c1), ("c2", model.c2)]:
        assert grids[name] == {"e_layer_min": m.e_layer_min, "e_qa_out": m.e_qa_out,
                               "win_bits": 22,
                               "shift_out": m.e_qa_out - m.e_layer_min}

    # Inference: read the grids file, no calibration.
    model2 = _quant(tmp, config_path=path)
    for a, b in [(model.c1, model2.c1), (model.c2, model2.c2)]:
        assert (b.e_layer_min, b.e_qa_out) == (a.e_layer_min, a.e_qa_out)
        assert b._grid_src == {"e_layer_min": "config", "e_qa_out": "config"}
    res = calibrate_hw_grids(model2, _data(), verbose=0)
    assert all(r["src"] == {"e_layer_min": "config", "e_qa_out": "config"}
               and r["e_qa_in"] is None for r in res.values())   # not swept
    assert torch.equal(_eval(model2, x), y)


def test_save_does_not_overwrite():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp)
    q = MXQuantizer(save_dir=tmp)
    model = q.quant(_Net())
    calibrate_hw_grids(model, _data(), num_batches=2, verbose=0)
    path = q.save_hw_grids(model)
    with pytest.raises(FileExistsError):
        q.save_hw_grids(model)
    assert q.save_hw_grids(model, overwrite=True) == path
    other = os.path.join(tmp, "x.json")
    assert q.save_hw_grids(model, path=other) == other
    # The source config is never touched.
    with open(os.path.join(tmp, "mx_config.json")) as f:
        assert "hw_grids" not in json.load(f)


def test_export_without_grids_raises():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp)
    model = _quant(tmp)
    with pytest.raises(ValueError, match="calibrate_hw_grids"):
        export_hw_grids(model)


# ---------------------------------------------------------------------------
# Priority: hw_grids > xblock_accum > calibration
# ---------------------------------------------------------------------------

def test_config_beats_calibration():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, hw_grids={"c1": GRID_C1})
    model = _quant(tmp)
    assert (model.c1.e_layer_min, model.c1.e_qa_out) == (-30, -19)
    assert model.c2.e_layer_min is None and model.c2.e_qa_out is None

    res = calibrate_hw_grids(model, _data(), num_batches=2, verbose=0)
    assert (model.c1.e_layer_min, model.c1.e_qa_out) == (-30, -19)
    assert res["c1"]["src"] == {"e_layer_min": "config", "e_qa_out": "config"}
    assert res["c2"]["src"] == {"e_layer_min": "calib", "e_qa_out": "calib"}

    # Offline force recalibrates everything.
    res = calibrate_hw_grids(model, _data(), num_batches=2, force=True, verbose=0)
    assert model.c1.e_layer_min != -30
    assert res["c1"]["src"] == {"e_layer_min": "calib", "e_qa_out": "calib"}


def test_hw_grids_beats_xblock_accum():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, hw_grids={"c1": GRID_C1}, e_layer_min=-20, e_qa_out=-3)
    model = _quant(tmp)
    assert (model.c1.e_layer_min, model.c1.e_qa_out) == (-30, -19)
    assert (model.c2.e_layer_min, model.c2.e_qa_out) == (-20, -3)
    assert model.c2._grid_src == {"e_layer_min": "config", "e_qa_out": "config"}


def test_xblock_e_layer_min_kept_e_qa_out_calibrated():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, e_layer_min=-20)
    model = _quant(tmp)
    res = calibrate_hw_grids(model, _data(), num_batches=2, verbose=0)
    assert model.c1.e_layer_min == -20 and model.c1.e_qa_out is not None
    assert res["c1"]["src"] == {"e_layer_min": "config", "e_qa_out": "calib"}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def test_unknown_name_raises():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, hw_grids={"c1": GRID_C1, "c3": GRID_C1})
    with pytest.raises(ValueError, match="c3"):
        _quant(tmp)


@pytest.mark.parametrize("bad,match", [
    ({**GRID_C1, "shift_out": 10}, "shift_out"),
    ({**GRID_C1, "win_bits": 23}, "win_bits"),
    ({"e_layer_min": -30}, "missing"),
    ({**GRID_C1, "e_cut": 1}, "unknown keys"),
])
def test_inconsistent_entry_raises(bad, match):
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, hw_grids={"c1": bad})
    with pytest.raises(ValueError, match=match):
        _quant(tmp)


def test_win_bits_follows_config_width():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, hw_grids={"c1": {**GRID_C1, "win_bits": 23}},
               out_window_bits=23)
    assert _quant(tmp).c1.e_qa_out == -19
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, hw_grids={"c1": {**GRID_C1, "win_bits": 14}}, out_exp_bits=3)
    assert _quant(tmp).c1.e_qa_out == -19


def test_non_int_entry_raises():
    with pytest.raises(TypeError):
        check_hw_grid("c1", {"e_layer_min": -3.0, "e_qa_out": 1})
    with pytest.raises(TypeError):
        normalize_xblock_accum({"e_qa_out": 1.5})


def test_double_call_with_window_raises_in_eval_only():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp)
    model = _quant(tmp, net=_TwiceNet())
    calibrate_hw_grids(model, _data(), num_batches=2, verbose=0)
    x = _data(1)[0]
    with pytest.raises(RuntimeError, match="c1: called 2 times"):
        _eval(model, x)
    model.train()
    model(x)                                     # train: not checked
    # Window off: one grid is not used for extraction, no check.
    model.c1.xblock_accum["out_window"] = "off"
    _eval(model, x)


def test_single_call_passes_every_forward():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, hw_grids={"c1": GRID_C1, "c2": {**GRID_C1}})
    model = _quant(tmp)
    for x in _data(3):
        _eval(model, x)                          # counter resets per forward


# ---------------------------------------------------------------------------
# quant() calibration step ("calibrate_hw_grids" config key)
# ---------------------------------------------------------------------------

CAL = {"calibrate_hw_grids": {"enabled": True, "batches": 2, "verbose": 0}}


def test_quant_calibrates_and_saves():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, top=CAL)
    model = MXQuantizer(save_dir=tmp).quant(_Net(), data=_data(),
                                            forward_fn=lambda m, b: m(b))
    for m in (model.c1, model.c2):
        assert m.e_layer_min is not None and m.e_qa_out is not None
        assert m._grid_src == {"e_layer_min": "calib", "e_qa_out": "calib"}
    path = os.path.join(tmp, "mx_config_grids.json")
    assert set(load_hw_grids(path)) == {"c1", "c2"}
    x = _data(1, seed=9)[0]
    y = _eval(model, x)
    # Inference from the saved file: no data, no calibration, same output.
    model2 = _quant(tmp, config_path=path)
    assert model2.c1._grid_src == {"e_layer_min": "config", "e_qa_out": "config"}
    assert torch.equal(_eval(model2, x), y)


def test_quant_calibration_keeps_existing_file():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, top=CAL)
    path = os.path.join(tmp, "mx_config_grids.json")
    with open(path, "w") as f:
        f.write("{}")
    MXQuantizer(save_dir=tmp).quant(_Net(), data=_data())
    with open(path) as f:
        assert f.read() == "{}"


def test_quant_calibration_needs_data():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, top=CAL)
    with pytest.raises(ValueError, match="no data"):
        _quant(tmp)


def test_quant_calibration_noop_when_config_sets_all():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, hw_grids={"c1": GRID_C1, "c2": GRID_C1}, top=CAL)
    model = _quant(tmp)                          # no data needed
    assert model.c2.e_qa_out == -19
    assert not os.path.exists(os.path.join(tmp, "mx_config_grids.json"))


def test_quant_calibration_off_by_default_and_bad_key():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp)
    assert _quant(tmp).c1.e_layer_min is None    # no key -> no calibration
    _write_cfg(tmp, top={"calibrate_hw_grids": {"batch": 2}})
    with pytest.raises(ValueError, match="unknown keys"):
        _quant(tmp)


# ---------------------------------------------------------------------------
# Sim reader
# ---------------------------------------------------------------------------

def test_hw_grid_for_reads_json_without_torch():
    tmp = tempfile.mkdtemp()
    cfg = _write_cfg(tmp, hw_grids={"c1": {"e_layer_min": -30, "e_qa_out": -19}})
    path = os.path.join(tmp, "mx_config.json")
    assert hw_grid_for(path, "c1") == {"e_layer_min": -30, "e_qa_out": -19,
                                       "shift_out": 11}
    assert hw_grid_for(cfg, "c1")["shift_out"] == 11
    with pytest.raises(KeyError, match="c2"):
        hw_grid_for(path, "c2")
    code = ("import sys; sys.path.insert(0, sys.argv[1]);"
            "from fixed_point.hw_grids import hw_grid_for;"
            "print(hw_grid_for(sys.argv[2], 'c1')['shift_out']);"
            "print(any(m.split('.')[0] in ('torch', 'microxcaling') "
            "for m in sys.modules))")
    out = subprocess.run([sys.executable, "-I", "-c", code, REPO, path],
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == ["11", "False"]
