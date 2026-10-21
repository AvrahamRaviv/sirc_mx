"""quant() steps for finding where a net loses accuracy:

- "calibrate_act_scales": static act_quant scales (the DOF warp) frozen per
  call site inside quant(), saved next to the HW grids, reloaded without data.
- [ActQuant] clip % / SQNR per fxp operand and call site.
- "measure_error": {"isolated": true}: each layer fed its FP32 twin's input.
"""
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

from mx_layers_act import MXActQuant
from mx_quantizer import MXQuantizer
from fixed_point.mx_fixed_point import normalize_xblock_accum


class Warp(nn.Module):
    def forward(self, x, flow):
        B, C, H, W = x.shape
        yy, xx = torch.meshgrid(torch.arange(H, dtype=x.dtype),
                                torch.arange(W, dtype=x.dtype), indexing="ij")
        gx = (xx + flow[:, 0]) / max(W - 1, 1) * 2 - 1
        gy = (yy + flow[:, 1]) / max(H - 1, 1) * 2 - 1
        return F.grid_sample(x, torch.stack([gx, gy], dim=-1), align_corners=True)


class Net(nn.Module):
    """warp -> HW conv -> warp (call sites 0 and 1, 4x apart in scale)."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.warp = Warp()
        self.c1 = nn.Conv2d(32, 32, 3, padding=1)

    def forward(self, x, flow):
        y = self.c1(self.warp(x, flow))
        return self.warp(y * 4.0, flow)


def _fwd(m, b):
    return m(b[0], b[1])


def _data(n=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randn(2, 32, 8, 8, generator=g) * 3,
             torch.randn(2, 2, 8, 8, generator=g)) for _ in range(n)]


_FEAT = {"total_bits": 8, "signed": True, "calibrate": True,
         "round": "half_away", "saturate": True}
_FLOW = {"total_bits": 16, "frac_bits": 8, "signed": True,
         "round": "half_away", "saturate": True}


def _cfg(feat=None, top=None, hw=True, **xblock):
    xb = {"enabled": hw, "mode": "hw_fixed_point", "bits": 48,
          "backend": "python", "sat_mode": "per_product", "e_layer_min": None,
          "out_window": "infer", "acc_headroom_bits": 2, "verbose": 0, **xblock}
    return {"mx_specs": {"w_elem_format": "int8", "a_elem_format": "int8",
                         "block_size": 32, "custom_cuda": False,
                         "xblock_accum": xb},
            "layers": ["c1",
                       {"name": "warp", "kind": "act_quant", "call_sites": 2,
                        "inputs": [{"fxp": feat or _FEAT}, {"fxp": _FLOW}]}],
            "ptq": False, "measure_error": False, **(top or {})}


def _write(cfg, tmp=None):
    tmp = tmp or tempfile.mkdtemp()
    with open(os.path.join(tmp, "mx_config.json"), "w") as f:
        json.dump(cfg, f)
    return tmp


_STEPS = {"calibrate_act_scales": {"batches": 3, "verbose": 0},
          "calibrate_hw_grids": {"batches": 3, "verbose": 0}}


# ---------------------------------------------------------------------------
# calibrate_act_scales step
# ---------------------------------------------------------------------------

def test_quant_calibrates_act_scales_per_call_site():
    tmp = _write(_cfg(top=_STEPS))
    model = MXQuantizer(save_dir=tmp).quant(Net(), data=_data(), forward_fn=_fwd)
    s0, s1 = (c["scale"] for c in model.warp.fxp_per_input[0])
    xs = _data()
    assert s0 == pytest.approx(max(float(x.abs().max()) for x, _ in xs) / 127)
    assert s1 > s0                              # site 1 sees the 4x conv output
    assert model.c1.e_layer_min is not None and model.c1.e_qa_out is not None


def test_act_scales_run_before_hw_grids():
    """The two pending steps need each other's values: act scales first, with
    the ungridded HW conv on its FP32 path; the grids then see the quantized
    warp output, and the conv is back on the HW path afterwards."""
    tmp = _write(_cfg(top=_STEPS))
    model = MXQuantizer(save_dir=tmp).quant(Net(), data=_data(), forward_fn=_fwd)
    assert getattr(model.c1, "_calibration_state", None) is None
    model.eval()
    with torch.no_grad():
        x, flow = _data(1, seed=9)[0]
        model.warp.reset_calls()
        y = model(x, flow)
    assert torch.isfinite(y).all() and model.c1._sat_seen_life == 0


def test_act_step_needs_data():
    tmp = _write(_cfg(top={"calibrate_act_scales": True}, hw=False))
    with pytest.raises(ValueError, match="calibrate_act_scales.*no data"):
        MXQuantizer(save_dir=tmp).quant(Net())


def test_act_step_noop_when_scales_given():
    feat = [{**_FEAT, "calibrate": False, "scale": 0.1},
            {**_FEAT, "calibrate": False, "scale": 0.4}]
    tmp = _write(_cfg(feat=feat, top={"calibrate_act_scales": True}, hw=False))
    model = MXQuantizer(save_dir=tmp).quant(Net())      # no data needed
    assert [c["scale"] for c in model.warp.fxp_per_input[0]] == [0.1, 0.4]
    assert not os.path.exists(os.path.join(tmp, "mx_config_grids.json"))


def test_act_step_rejects_unknown_key():
    tmp = _write(_cfg(top={"calibrate_act_scales": {"batchs": 3}}, hw=False))
    with pytest.raises(ValueError, match="unknown keys"):
        MXQuantizer(save_dir=tmp).quant(Net(), data=_data(), forward_fn=_fwd)


def test_saved_config_reloads_without_data_bit_exact():
    tmp = _write(_cfg(top=_STEPS))
    model = MXQuantizer(save_dir=tmp).quant(Net(), data=_data(), forward_fn=_fwd)
    path = os.path.join(tmp, "mx_config_grids.json")
    with open(path) as f:
        saved = json.load(f)
    warp = next(l for l in saved["layers"] if isinstance(l, dict)
                and l.get("name") == "warp")
    sites = warp["inputs"][0]["fxp"]
    assert isinstance(sites, list) and len(sites) == 2
    assert all("calibrate" not in s and s["round"] == "half_away" for s in sites)
    assert [s["scale"] for s in sites] == [c["scale"] for c in
                                           model.warp.fxp_per_input[0]]
    assert warp["inputs"][1]["fxp"] == _FLOW        # given spec untouched
    assert "c1" in saved["hw_grids"]

    model2 = MXQuantizer(save_dir=tmp, config_path=path).quant(Net())
    x, flow = _data(1, seed=9)[0]
    outs = []
    for m in (model, model2):
        m.eval()
        m.warp.reset_calls()
        with torch.no_grad():
            outs.append(m(x, flow))
    assert torch.equal(outs[0], outs[1])


def test_act_only_step_saves_scales():
    tmp = _write(_cfg(top={"calibrate_act_scales": {"batches": 2, "verbose": 0}},
                      hw=False))
    MXQuantizer(save_dir=tmp).quant(Net(), data=_data(), forward_fn=_fwd)
    with open(os.path.join(tmp, "mx_config_grids.json")) as f:
        saved = json.load(f)
    warp = next(l for l in saved["layers"] if isinstance(l, dict))
    assert all(s["scale"] > 0 for s in warp["inputs"][0]["fxp"])


# ---------------------------------------------------------------------------
# [ActQuant] stats
# ---------------------------------------------------------------------------

def test_fxp_summary_reports_clip_and_sqnr_per_site():
    w = MXActQuant(Warp(), [None, None], fxp_per_input=[
        [{**_FEAT, "calibrate": False, "scale": 1.0 / 127, "frac_bits": 8,
          "clip_grad": False},
         {**_FEAT, "calibrate": False, "scale": 100.0 / 127, "frac_bits": 8,
          "clip_grad": False}], None], call_sites=2)
    x = torch.linspace(-4, 4, 1000).view(1, 1, 10, 100)
    flow = torch.zeros(1, 2, 10, 100)
    w(x, flow)
    w(x, flow)
    lines = w.fxp_summary()
    assert len(lines) == 2 and lines[0].startswith("in0@0:")
    clip0 = float(lines[0].split("clip ")[1].split("%")[0])
    clip1 = float(lines[1].split("clip ")[1].split("%")[0])
    assert clip0 == pytest.approx(75.0, abs=0.5) and clip1 == 0.0
    assert "max|x| 4" in lines[0] and "range 1" in lines[0]
    w.reset_fxp_stats()
    assert w.fxp_summary() == []


def test_print_stat_logs_act_quant_lines():
    tmp = _write(_cfg(top={**_STEPS, "measure_error": True}))
    logs = []

    class _L:
        def info(self, msg):
            logs.append(msg)

    MXQuantizer(save_dir=tmp).quant(Net(), data=_data(), forward_fn=_fwd,
                                    log=_L())
    act = [l for l in logs if l.startswith("[ActQuant] warp in")]
    assert [l.split()[2] for l in act] == ["in0@0:", "in0@1:", "in1@0:", "in1@1:"]
    assert all("clip" in l and "SQNR" in l for l in act)


# ---------------------------------------------------------------------------
# isolated error
# ---------------------------------------------------------------------------

def test_isolated_error_is_the_layers_own():
    """Isolated error of c1 = its own error on the FP32 warp output, which is
    smaller than the propagated one (that also carries the warp's error)."""
    feat = [{**_FEAT, "calibrate": False, "scale": 0.5},
            {**_FEAT, "calibrate": False, "scale": 2.0}]
    tmp = _write(_cfg(feat=feat, hw=False,
                      top={"measure_error": {"enabled": True, "isolated": True}}))
    data = _data()
    fp32 = Net()
    model = MXQuantizer(save_dir=tmp).quant(fp32, data=data, forward_fn=_fwd)
    iso, prop = model._quant_errors_isolated, model._quant_errors
    assert {"c1", "warp"} <= set(iso)
    assert iso["c1"]["sqnr_db"] > prop["c1"]["sqnr_db"] + 3

    # By hand: c1 quantized on the exact FP32 input.
    sig = noise = 0.0
    with torch.no_grad():
        for x, flow in data:
            a = fp32.warp(x, flow)
            f, q = fp32.c1(a), model.c1(a)
            sig += f.pow(2).sum().item()
            noise += (f - q).pow(2).sum().item()
    import math
    assert iso["c1"]["sqnr_db"] == pytest.approx(10 * math.log10(sig / noise),
                                                 abs=1e-3)


def test_isolated_runs_with_hw_layers_and_warp_sites():
    tmp = _write(_cfg(top={**_STEPS, "measure_error": {"isolated": True}}))
    model = MXQuantizer(save_dir=tmp).quant(Net(), data=_data(), forward_fn=_fwd)
    iso = model._quant_errors_isolated
    assert iso["c1"]["sqnr_db"] > 20 and iso["warp"]["sqnr_db"] > 20
    # Site counter still aligned for the propagated pass that ran after.
    assert model.warp.n_calls % model.warp.call_sites == 0


def test_measure_error_rejects_unknown_key():
    tmp = _write(_cfg(hw=False, top={"measure_error": {"isolate": True}},
                      feat={**_FEAT, "calibrate": False, "scale": 0.1}))
    with pytest.raises(ValueError, match="measure_error: unknown keys"):
        MXQuantizer(save_dir=tmp).quant(Net(), data=_data(), forward_fn=_fwd)


# ---------------------------------------------------------------------------
# call-site guard
# ---------------------------------------------------------------------------

def _fixed_sites(n):
    return [{**_FEAT, "calibrate": False, "scale": 0.1 * (i + 1)} for i in range(n)]


def test_wrong_call_sites_warns_and_does_not_rotate(capsys):
    cfg = _cfg(feat=_fixed_sites(3), hw=False)
    cfg["layers"][1]["call_sites"] = 3                  # Net calls warp twice
    model = MXQuantizer(save_dir=_write(cfg)).quant(Net())
    model.eval()
    seen = []
    orig = model.warp.quant_input

    def spy(x, idx, site=None):
        if idx == 0:
            seen.append(site)
        return orig(x, idx, site)

    model.warp.quant_input = spy
    x, flow = _data(1)[0]
    with torch.no_grad():
        for _ in range(3):
            model(x, flow)
    assert seen == [0, 1] * 3                           # no drift across forwards
    out = capsys.readouterr().out
    assert out.count("call_sites=3 but was called 2 times") == 1


def test_right_call_sites_is_silent(capsys):
    model = MXQuantizer(save_dir=_write(_cfg(feat=_fixed_sites(2), hw=False))
                        ).quant(Net())
    model.eval()
    x, flow = _data(1)[0]
    with torch.no_grad():
        for _ in range(3):
            model(x, flow)
    assert model.warp.n_calls == 2
    assert "WARNING" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# DOF config
# ---------------------------------------------------------------------------

def test_dof_cal_config():
    with open(os.path.join(REPO, "configs", "mx_config_dof_run_win_cal.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(REPO, "configs", "mx_config_dof_run_win.json")) as f:
        base = json.load(f)
    assert cfg["mx_specs"] == base["mx_specs"]
    normalize_xblock_accum(cfg["mx_specs"]["xblock_accum"])
    assert cfg["calibrate_act_scales"]["enabled"] is True
    assert cfg["measure_error"] == {"enabled": True, "isolated": True}
    warp = next(l for l in cfg["layers"] if isinstance(l, dict)
                and l.get("name") == "model.warp")
    assert warp["call_sites"] == 2           # DOF run: warp called twice
    assert warp["inputs"][0]["fxp"]["calibrate"] is True
    assert warp["inputs"][0]["fxp"]["total_bits"] == 8
    convs = [l for l in cfg["layers"] if l != warp]
    assert convs == [l for l in base["layers"] if not (
        isinstance(l, dict) and l.get("name") == "model.warp")]
