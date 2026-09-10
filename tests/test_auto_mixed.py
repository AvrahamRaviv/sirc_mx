"""Automated mixed-precision: module construction, scoring and assignment.

Covers the pieces that let one sensitivity score per layer drive an N-rung
precision ladder. See mx_quantizer._build_mx_module / mx_sensitivity /
mx_assign.
"""
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, '/Users/avrahamraviv/PycharmProjects')
sys.path.insert(0, '/home/avrahamra/PycharmProjects')

from microxcaling.mx.convolution import Conv2d as MXConv2d
from microxcaling.mx.transpose_convolution import ConvTranspose2d as MXConvTranspose2d
from microxcaling.mx.linear import Linear as MXLinear

from mx_layers_blocked import MXConv2dBlocked, MXConv2dHW, MXLinearBlocked
from mx_quantizer import MXQuantizer


_INT8 = {"w_elem_format": "int8", "a_elem_format": "int8",
         "block_size": 32, "scale_bits": 8, "shared_exp_method": "max",
         "custom_cuda": False}

_XBLOCK_HW = dict(_INT8, xblock_accum={
    "enabled": True, "mode": "hw_fixed_point", "bits": 48,
    "backend": "python", "sat_mode": "per_product",
    "weight_blockify": "flatten", "act_blockify": "xblock",
})

_XBLOCK_FP = dict(_INT8, xblock_accum={
    "enabled": True, "mode": "fp32_partial", "bits": 48, "backend": "python",
})


def _quantizer(tmp_path, config):
    import json
    (tmp_path / "mx_config.json").write_text(json.dumps(config))
    return MXQuantizer(save_dir=str(tmp_path))


# =============================================================================
# _build_mx_module — one class-selection rule for deployment and measurement
# =============================================================================

@pytest.mark.parametrize("spec,expected", [
    (_INT8, MXConv2d),
    (_XBLOCK_FP, MXConv2dBlocked),
    (_XBLOCK_HW, MXConv2dHW),
])
def test_build_mx_module_selects_deployed_conv_class(tmp_path, spec, expected):
    """The measurement path must get the same class deployment installs.

    Probing a plain MXConv2d for a layer that deploys as MXConv2dHW would
    measure the wrong noise — on the NPE path the fixed-point accumulator is
    often the dominant error term.
    """
    q = _quantizer(tmp_path, {"layers": []})
    conv = nn.Conv2d(32, 16, 3, padding=1)
    new = q._build_mx_module(conv, q._build_mx_specs(spec), name="c", verbose=0)
    assert type(new) is expected


def test_build_mx_module_linear_and_convT(tmp_path):
    """ConvTranspose2d is plain-MX only; Linear follows the blocked flag."""
    q = _quantizer(tmp_path, {"layers": []})
    hw = q._build_mx_specs(_XBLOCK_HW)

    convT = q._build_mx_module(nn.ConvTranspose2d(32, 16, 2, stride=2), hw,
                               name="t", verbose=0)
    assert type(convT) is MXConvTranspose2d

    lin = q._build_mx_module(nn.Linear(64, 32), hw, name="l", verbose=0)
    assert type(lin) is MXLinearBlocked
    plain = q._build_mx_module(nn.Linear(64, 32), q._build_mx_specs(_INT8),
                               name="l", verbose=0)
    assert type(plain) is MXLinear


def test_build_mx_module_shares_weights(tmp_path):
    """Weight/bias are shared, not copied — same Parameter object."""
    q = _quantizer(tmp_path, {"layers": []})
    conv = nn.Conv2d(32, 16, 3)
    new = q._build_mx_module(conv, q._build_mx_specs(_INT8), name="c", verbose=0)
    assert new.weight is conv.weight
    assert new.bias is conv.bias


def test_build_mx_module_rejects_unsupported_type(tmp_path):
    """A non-conv/linear name in the config must fail loudly, not silently
    fall through to MXLinear and raise AttributeError deep inside."""
    q = _quantizer(tmp_path, {"layers": []})
    with pytest.raises(TypeError, match="bn1"):
        q._build_mx_module(nn.BatchNorm2d(8), q._build_mx_specs(_INT8), name="bn1")


def test_build_mx_module_falls_back_and_tallies(tmp_path):
    """Grouped conv cannot use the blocked path; the reason is recorded."""
    q = _quantizer(tmp_path, {"layers": []})
    summary = {k: [] for k in ('hw', 'blocked', 'mx_default_conv',
                               'mx_default_convT', 'mx_default_linear',
                               'fallback_conv', 'fallback_linear')}
    conv = nn.Conv2d(32, 32, 3, groups=4)
    new = q._build_mx_module(conv, q._build_mx_specs(_XBLOCK_HW), name="g",
                             verbose=0, summary=summary)
    assert type(new) is MXConv2d
    assert summary['fallback_conv'] == [("g", "groups=4")]
    assert summary['mx_default_conv'] == ["g"]


# =============================================================================
# Name handling and shared modules
# =============================================================================

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return self.fc(x.mean((2, 3)))


class _SharedNet(nn.Module):
    """One conv object bound under two names, applied twice."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 3, padding=1)
        self.body = nn.Conv2d(8, 8, 3, padding=1)
        self.body_again = self.body

    def forward(self, x):
        x = self.stem(x)
        return self.body_again(self.body(x))


def test_dataparallel_model_is_quantized(tmp_path):
    """Auto-discovery under DataParallel must actually replace layers.

    named_modules() yields `module.conv1` while _replace_layers looks up the
    stripped `conv1`, so before the clean_name fix this replaced nothing at all
    and reported success.
    """
    q = _quantizer(tmp_path, {"groups": {"int8": _INT8},
                              "auto_mixed": {"base": "int8", "upgrade": "int8"},
                              "ptq": False, "measure_error": False})
    model = nn.DataParallel(_Net())
    names = q._get_candidate_layers(model)
    assert names == ["conv1", "conv2", "fc"]

    qm = q.quant(model)
    assert isinstance(qm.module.conv1, MXConv2d)
    assert isinstance(qm.module.conv2, MXConv2d)
    assert isinstance(qm.module.fc, MXLinear)


def test_shared_module_all_bindings_replaced(tmp_path):
    """A module reachable under two names must be replaced at both bindings,
    with one shared MX layer — not left half-quantized."""
    q = _quantizer(tmp_path, {"mx_specs": _INT8,
                              "layers": ["stem", "body", "body_again"],
                              "ptq": False, "measure_error": False})
    qm = q.quant(_SharedNet())
    assert isinstance(qm.body, MXConv2d)
    assert isinstance(qm.body_again, MXConv2d)
    assert qm.body is qm.body_again


def test_unknown_layer_name_warns(tmp_path, capsys):
    """A typo'd layer name quantizes nothing; say so instead of passing."""
    q = _quantizer(tmp_path, {"mx_specs": _INT8,
                              "layers": ["conv1", "conv_typo"],
                              "ptq": False, "measure_error": False})
    q.quant(_Net())
    out = capsys.readouterr().out
    assert "matched no" in out and "conv_typo" in out


# =============================================================================
# OAT scoring
# =============================================================================

import mx_sensitivity as mxs


class _CanaryNet(nn.Module):
    """Three convs; conv2's weights are blown up so MX loses the most on it."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 8, 3, padding=1)

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


class _BypassNet(nn.Module):
    """`unused` is a real Conv2d that forward() never calls."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.unused = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


def _oat(model, names, batches, probe_spec, tmp_path, ref_spec=None):
    """Score `names` by OAT: reference at ref_spec (FP32 if None), probe demoted."""
    q = _quantizer(tmp_path, {"layers": []})
    probe = q._build_mx_specs(probe_spec)
    ref_model = model
    if ref_spec is not None:
        ref_model = copy.deepcopy(model)
        q._replace_layers(ref_model,
                          layer_map={n: q._build_mx_specs(ref_spec) for n in names})
    types = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
    entries = mxs.resolve_bindings(ref_model, names, types)
    return ref_model, entries, mxs.score_oat(
        ref_model, entries, batches,
        build_probe=lambda n, m: q._build_mx_module(m, probe, name=n, verbose=0))


import copy


def test_oat_ranks_the_planted_layer_first(tmp_path):
    """Canary: the layer whose weights span a huge dynamic range must rank
    most sensitive. If it does not, the probe is not measuring what we think."""
    torch.manual_seed(0)
    model = _CanaryNet().eval()
    with torch.no_grad():
        model.conv2.weight[0] *= 1000.0     # one filter forces a huge block scale
    batches = [torch.randn(2, 3, 16, 16) for _ in range(2)]

    _, _, res = _oat(model, ["conv1", "conv2", "conv3"], batches,
                     dict(_INT8, w_elem_format="int4", a_elem_format="int4"),
                     tmp_path)

    scored = {k: v for k, v in res.items() if k != "__meta__"}
    assert all(v["status"] == "ok" for v in scored.values())
    worst = max(scored, key=lambda k: scored[k]["sensitivity"])
    assert worst == "conv2"


def test_oat_marks_unreached_layer(tmp_path):
    """A layer the forward pass never calls is 'unreached', not 'insensitive'.

    Its output is identical either way, so an output-identity check would call
    it harmless; only a call counter tells the truth.
    """
    torch.manual_seed(0)
    model = _BypassNet().eval()
    batches = [torch.randn(2, 3, 16, 16)]
    _, _, res = _oat(model, ["conv", "unused"], batches,
                     dict(_INT8, w_elem_format="int4", a_elem_format="int4"),
                     tmp_path)
    assert res["unused"]["status"] == "unreached"
    assert res["unused"]["sensitivity"] is None
    assert res["conv"]["status"] == "ok"


def test_oat_restores_the_model(tmp_path):
    """Scoring must leave the reference network byte-identical."""
    torch.manual_seed(0)
    model = _CanaryNet().eval()
    batches = [torch.randn(2, 3, 16, 16)]
    before = {n: m for n, m in model.named_modules()}

    ref_model, _, _ = _oat(model, ["conv1", "conv2", "conv3"], batches,
                           dict(_INT8, w_elem_format="int4"), tmp_path)

    assert ref_model is model
    after = {n: m for n, m in model.named_modules()}
    assert before.keys() == after.keys()
    assert all(before[n] is after[n] for n in before)
    with torch.no_grad():
        assert torch.equal(model(batches[0]), model(batches[0]))


def test_oat_scores_shared_module_through_all_bindings(tmp_path):
    """A tied layer is probed at every binding, so its score reflects both uses."""
    torch.manual_seed(0)
    model = _SharedNet().eval()
    batches = [torch.randn(2, 3, 16, 16)]
    _, entries, res = _oat(model, ["stem", "body"], batches,
                           dict(_INT8, w_elem_format="int4", a_elem_format="int4"),
                           tmp_path)
    assert len(entries["body"]["sites"]) == 2
    assert res["body"]["n_calls"] == 2          # fired twice per batch
    assert res["body"]["status"] == "ok"


def test_oat_rejects_nondeterministic_model(tmp_path):
    """A forward that mutates state makes every score noise — refuse to rank.

    Scoring forces .eval(), so dropout is not the hazard; a running buffer
    updated inside forward() is (temporal models keep exactly that kind of
    state). Two identical runs then disagree and every OAT difference is
    contaminated.
    """
    torch.manual_seed(0)

    class _Stateful(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.register_buffer("memory", torch.zeros(1))

        def forward(self, x):
            self.memory += 1.0
            return self.conv(x) * self.memory

    model = _Stateful().eval()
    batches = [torch.randn(2, 3, 16, 16)]
    with pytest.raises(RuntimeError, match="deterministic"):
        _oat(model, ["conv"], batches, dict(_INT8, w_elem_format="int4"), tmp_path)


def test_resolve_bindings_follows_aliases(tmp_path):
    """Asking for `body` must also find its `body_again` binding."""
    model = _SharedNet().eval()
    entries = mxs.resolve_bindings(model, ["body"], (nn.Conv2d,))
    assert sorted(entries["body"]["paths"]) == ["body", "body_again"]


def test_resolve_bindings_dedupes_two_names_for_one_module(tmp_path):
    """Both names for one module resolve to a single entry, probed once."""
    model = _SharedNet().eval()
    entries = mxs.resolve_bindings(model, ["body", "body_again"], (nn.Conv2d,))
    assert len(entries) == 1
    (name, entry), = entries.items()
    assert name == "body" and entry["aliases"] == ["body_again"]
    assert len(entry["sites"]) == 2


# =============================================================================
# Assignment — a pure function of the artifact
# =============================================================================

import mx_assign as mxa


_LADDER = {"int4": dict(_INT8, w_elem_format="int4", a_elem_format="int4"),
           "int6": dict(_INT8, w_elem_format="int6", a_elem_format="int6"),
           "int8": dict(_INT8)}

_QUANTILE_CFG = {"ladder": ["int4", "int6", "int8"], "strategy": "quantile",
                 "quantile": {"int4": 0.10, "int6": 0.20, "int8": 0.70}}


def _rows(n, start=0.0):
    """n layers with strictly increasing sensitivity."""
    return [{"name": f"L{i:03d}", "sensitivity": start + i, "status": "ok",
             "cost": {"macs": 1000}} for i in range(n)]


@pytest.mark.parametrize("n", [47, 10, 3, 100])
def test_apportion_sums_to_n(n):
    """Largest remainder must place every layer exactly once.

    Truncation loses layers — 10/20/70 of 47 truncates to 4+9+32 = 45.
    """
    counts = mxa.apportion(n, [("int4", 0.10), ("int6", 0.20), ("int8", 0.70)])
    assert sum(counts.values()) == n


def test_quantile_ladder_splits_10_20_70():
    """The requested 10/20/70 over 47 layers, least sensitive going lowest."""
    rows = _rows(47)
    assignments, _ = mxa.assign(rows, _QUANTILE_CFG, _LADDER)
    counts = {}
    for g in assignments.values():
        counts[g] = counts.get(g, 0) + 1
    assert counts == {"int4": 5, "int6": 9, "int8": 33}
    assert sum(counts.values()) == 47
    # least sensitive at the bottom rung, most sensitive at the top
    assert assignments["L000"] == "int4"
    assert assignments["L046"] == "int8"


def test_unmeasured_layers_go_top_and_leave_fractions_alone():
    """Unmeasured is not 'insensitive': top rung, and out of the denominator."""
    rows = _rows(10)
    rows.append({"name": "never_ran", "sensitivity": None, "status": "unreached",
                 "cost": {"macs": 0}})
    assignments, notes = mxa.assign(rows, _QUANTILE_CFG, _LADDER)
    assert assignments["never_ran"] == "int8"
    assert notes["unmeasured"] == ["never_ran"]
    ranked = {k: v for k, v in assignments.items() if k != "never_ran"}
    counts = {}
    for g in ranked.values():
        counts[g] = counts.get(g, 0) + 1
    assert sum(counts.values()) == 10       # denominator untouched by the outlier


def test_pins_win_and_leave_the_pool():
    rows = _rows(10)
    cfg = dict(_QUANTILE_CFG, pins={"L000": "int8"})
    assignments, notes = mxa.assign(rows, cfg, _LADDER)
    assert assignments["L000"] == "int8"    # despite being least sensitive
    assert notes["pinned"] == ["L000"]


def test_ties_are_deterministic_and_reported():
    rows = [{"name": f"L{i}", "sensitivity": 5.0, "status": "ok",
             "cost": {"macs": 1}} for i in range(4)]
    a1, notes = mxa.assign(rows, _QUANTILE_CFG, _LADDER)
    a2, _ = mxa.assign(list(reversed(rows)), _QUANTILE_CFG, _LADDER)
    assert a1 == a2
    assert notes["tied"]["5.0"] == ["L0", "L1", "L2", "L3"]


def test_threshold_strategy():
    rows = [{"name": "low", "sensitivity": 1.0, "status": "ok", "cost": {}},
            {"name": "mid", "sensitivity": 12.0, "status": "ok", "cost": {}},
            {"name": "high", "sensitivity": 40.0, "status": "ok", "cost": {}}]
    cfg = {"ladder": ["int4", "int6", "int8"], "strategy": "threshold",
           "threshold": {"int4": 8.0, "int6": 20.0}}
    assignments, _ = mxa.assign(rows, cfg, _LADDER)
    assert assignments == {"low": "int4", "mid": "int6", "high": "int8"}


# --- validation ---------------------------------------------------------------

def test_validate_rejects_unknown_group():
    with pytest.raises(mxa.AssignError, match="undefined group"):
        mxa.validate({"ladder": ["int4", "nope"]}, _LADDER)


def test_validate_rejects_bad_fractions():
    cfg = dict(_QUANTILE_CFG, quantile={"int4": 0.1, "int6": 0.2, "int8": 0.5})
    with pytest.raises(mxa.AssignError, match="sum to"):
        mxa.validate(cfg, _LADDER)


def test_validate_rejects_xblock_on_a_rung():
    """A rung carrying xblock_accum would silently change the layer class."""
    groups = dict(_LADDER, int4=dict(_LADDER["int4"], xblock_accum={"enabled": True}))
    with pytest.raises(mxa.AssignError, match="xblock_accum"):
        mxa.validate(_QUANTILE_CFG, groups)


def test_validate_rejects_unsupported_format():
    groups = dict(_LADDER, int4=dict(_LADDER["int4"], w_elem_format="int3"))
    with pytest.raises(mxa.AssignError, match="does not support"):
        mxa.validate(_QUANTILE_CFG, groups)


def test_validate_rejects_shared_key_conflict():
    groups = dict(_LADDER, int4=dict(_LADDER["int4"], scale_bits=4))
    with pytest.raises(mxa.AssignError, match="scale_bits"):
        mxa.validate(_QUANTILE_CFG, groups)


# --- w/a merge ----------------------------------------------------------------

def test_merge_wa_takes_each_format_from_its_own_rung():
    name, spec = mxa.merge_wa(_LADDER, "int4", "int8")
    assert spec["w_elem_format"] == "int4"
    assert spec["a_elem_format"] == "int8"
    assert name == "w4a8"


def test_merge_wa_keeps_deployment_accumulator():
    """xblock_accum comes from the deployment group, never from a rung."""
    groups = dict(_LADDER, deploy=dict(_INT8, xblock_accum={"enabled": True,
                                                            "mode": "hw_fixed_point"}))
    _, spec = mxa.merge_wa(groups, "int4", "int8", deploy_group="deploy")
    assert spec["xblock_accum"]["mode"] == "hw_fixed_point"
    assert spec["w_elem_format"] == "int4" and spec["a_elem_format"] == "int8"


def test_merge_wa_rejects_incompatible_block_size_under_xblock():
    groups = dict(_LADDER,
                  int4=dict(_LADDER["int4"], block_size=16),
                  deploy=dict(_INT8, xblock_accum={"enabled": True}))
    with pytest.raises(mxa.AssignError, match="block_size_wt"):
        mxa.merge_wa(groups, "int4", "int8", deploy_group="deploy")


def test_resolved_config_is_a_plain_replay(tmp_path):
    """The emitted config has no auto_mixed and re-runs deterministically."""
    rows = _rows(10)
    assignments, _ = mxa.assign(rows, _QUANTILE_CFG, _LADDER)
    flat, extra = mxa.resolve_groups(assignments, _LADDER)
    resolved = mxa.build_resolved_config({"ptq": False}, flat, _LADDER, extra)
    assert "auto_mixed" not in resolved
    assert set(resolved["groups"]) <= {"int4", "int6", "int8"}
    assert len(resolved["layers"]) == 10
    assert all(set(e) == {"name", "group"} for e in resolved["layers"])


# =============================================================================
# End to end: plan -> artifacts -> replay
# =============================================================================

import json


def _auto_config(**over):
    cfg = {
        "mx_specs": _INT8,
        "groups": {"int4": {"w_elem_format": "int4", "a_elem_format": "int4"},
                   "int6": {"w_elem_format": "int6", "a_elem_format": "int6"},
                   "int8": {"w_elem_format": "int8", "a_elem_format": "int8"}},
        "auto_mixed": {"ladder": ["int4", "int6", "int8"],
                       "scorer": "oat_output", "batches": 2,
                       "strategy": "quantile",
                       "quantile": {"int4": 0.34, "int6": 0.33, "int8": 0.33}},
        "ptq": False, "measure_error": False,
    }
    cfg["auto_mixed"].update(over)
    return cfg


def test_plan_writes_both_artifacts_and_replays(tmp_path):
    """The whole two-step workflow: plan, review, replay."""
    torch.manual_seed(0)
    model = _CanaryNet().eval()
    data = [torch.randn(2, 3, 16, 16) for _ in range(2)]

    q = _quantizer(tmp_path, _auto_config())
    plan = q.plan_mixed_precision(model, data=data)

    assert set(plan["assignments"]) == {"conv1", "conv2", "conv3"}
    assert sorted(plan["assignments"].values()) == ["int4", "int6", "int8"]

    sens = json.loads((tmp_path / "sensitivity.json").read_text())
    assert sens["schema_version"] == 1
    assert [r["name"] for r in sens["layers"]]          # sorted worst first
    assert sens["layers"][0]["sensitivity"] >= sens["layers"][-1]["sensitivity"]
    assert all(r["cost"]["macs"] > 0 for r in sens["layers"])
    assert sens["summary"]["macs_weighted_avg_bits"] > 0

    resolved = json.loads((tmp_path / "mx_config_resolved.json").read_text())
    assert "auto_mixed" not in resolved
    assert resolved["_provenance"]["scorer"] == "oat_output"

    # The resolved config is a plain replay: no scoring, no data needed.
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    (replay_dir / "mx_config.json").write_text(json.dumps(resolved))
    qm = MXQuantizer(save_dir=str(replay_dir)).quant(_CanaryNet())
    installed = {n: m.mx_specs["w_elem_format"]
                 for n, m in qm.named_modules() if isinstance(m, MXConv2d)}
    assert installed == {n: plan["assignments"][n].replace("int", "int")
                         for n in plan["assignments"]}


def test_plan_leaves_the_caller_model_untouched(tmp_path):
    torch.manual_seed(0)
    model = _CanaryNet().eval()
    q = _quantizer(tmp_path, _auto_config())
    q.plan_mixed_precision(model, data=[torch.randn(2, 3, 16, 16)], write=False)
    assert all(not isinstance(m, MXConv2d) for m in model.modules())


def test_quant_runs_the_ladder_end_to_end(tmp_path):
    """auto_mixed with a ladder installs mixed precision in one call."""
    torch.manual_seed(0)
    q = _quantizer(tmp_path, _auto_config())
    qm = q.quant(_CanaryNet().eval(), data=[torch.randn(2, 3, 16, 16)] * 2)
    fmts = sorted(m.mx_specs["w_elem_format"]
                  for m in qm.modules() if isinstance(m, MXConv2d))
    assert fmts == ["int4", "int6", "int8"]


def test_plan_uses_deployment_accumulator_for_probes(tmp_path):
    """Rungs set only the format; block geometry and xblock_accum come from
    mx_specs, so probes run the deployed arithmetic."""
    cfg = _auto_config()
    cfg["mx_specs"] = _XBLOCK_HW
    q = _quantizer(tmp_path, cfg)
    groups = cfg["groups"]
    spec = q._rung_spec("int4", groups, q._deploy_spec(cfg["auto_mixed"], groups))
    assert spec["w_elem_format"] == "int4"
    assert spec["xblock_accum"]["mode"] == "hw_fixed_point"
    built = q._build_mx_module(nn.Conv2d(32, 16, 3), q._build_mx_specs(spec),
                               name="c", verbose=0)
    assert type(built) is MXConv2dHW


def test_plan_rejects_empty_layer_list(tmp_path):
    cfg = _auto_config()
    cfg["layers"] = []
    q = _quantizer(tmp_path, cfg)
    with pytest.raises(ValueError, match="no candidate layers"):
        q.plan_mixed_precision(_CanaryNet(), data=[torch.randn(2, 3, 16, 16)])


def test_from_stats_scorer_needs_no_forward_passes(tmp_path):
    """Zero-compute path: reuse the out-SQNR collect_stats already measured."""
    stats = {"layers": {
        "module.conv1": {"output_error": {"isolated": {"sqnr_db": 40.0}},
                         "weight": {"underflow_rate": 0.01}, "activation": None},
        "module.conv2": {"output_error": {"isolated": {"sqnr_db": 12.0}},
                         "weight": {"underflow_rate": 0.30}, "activation": None},
        "module.conv3": {"output_error": {"isolated": {"sqnr_db": 25.0}},
                         "weight": None, "activation": None},
    }}
    stats_path = tmp_path / "quant_stats.json"
    stats_path.write_text(json.dumps(stats))

    q = _quantizer(tmp_path, _auto_config(scorer="from_stats",
                                          scorer_options={"path": str(stats_path)}))
    plan = q.plan_mixed_precision(_CanaryNet().eval(),
                                  data=[torch.randn(2, 3, 16, 16)])
    # worst out-SQNR -> most sensitive -> highest rung
    assert plan["assignments"]["conv2"] == "int8"
    assert plan["assignments"]["conv1"] == "int4"
    worst = plan["rows"][0]
    assert worst["name"] == "conv2" and worst["why"]["w_underflow"] == 0.30


# =============================================================================
# Absolute reference and w/a refinement
# =============================================================================

def test_absolute_reference_adds_a_second_column(tmp_path):
    """Both references are reported, with a rank delta and a correlation."""
    torch.manual_seed(0)
    q = _quantizer(tmp_path, _auto_config(reference="both"))
    plan = q.plan_mixed_precision(_CanaryNet().eval(),
                                  data=[torch.randn(2, 3, 16, 16)] * 2, write=False)
    for row in plan["rows"]:
        assert "marginal" in row["scores"] and "absolute" in row["scores"]
        assert "rank_delta" in row
    assert plan["meta"]["marginal_vs_absolute_spearman"] is not None


def test_reference_marginal_only_skips_the_second_pass(tmp_path):
    torch.manual_seed(0)
    q = _quantizer(tmp_path, _auto_config(reference="marginal"))
    plan = q.plan_mixed_precision(_CanaryNet().eval(),
                                  data=[torch.randn(2, 3, 16, 16)], write=False)
    assert "absolute" not in plan["rows"][0]["scores"]


def test_wa_refinement_records_both_operands(tmp_path):
    torch.manual_seed(0)
    q = _quantizer(tmp_path, _auto_config(
        reference="marginal", separable_wa={"enabled": True, "refine_top": 3}))
    plan = q.plan_mixed_precision(_CanaryNet().eval(),
                                  data=[torch.randn(2, 3, 16, 16)] * 2, write=False)
    top = plan["rows"][0]
    assert top["scores"]["w"] is not None
    assert top["scores"]["a"] is not None
    assert "interaction" in top["scores"]


def test_wa_split_demotes_the_quiet_operand():
    """The operand that is not causing the loss drops a rung; the other holds."""
    ladder = ["int4", "int6", "int8"]
    rows = [{"name": "wdriven", "sensitivity": 30.0,
             "scores": {"w": 29.0, "a": 2.0}},
            {"name": "adriven", "sensitivity": 30.0,
             "scores": {"w": 2.0, "a": 29.0}},
            {"name": "balanced", "sensitivity": 30.0,
             "scores": {"w": 20.0, "a": 20.5}}]
    assignments = {"wdriven": "int8", "adriven": "int8", "balanced": "int8"}
    out = MXQuantizer._apply_wa_split(assignments, rows, ladder,
                                      {"enabled": True, "margin_db": 3.0})
    assert out["wdriven"] == {"w": "int8", "a": "int6"}
    assert out["adriven"] == {"w": "int6", "a": "int8"}
    assert out["balanced"] == "int8"        # no clear culprit, left alone


def test_wa_split_produces_a_merged_group(tmp_path):
    """A split assignment resolves to a real, named, buildable group."""
    groups = {"int4": dict(_INT8, w_elem_format="int4", a_elem_format="int4"),
              "int8": dict(_INT8)}
    flat, extra = mxa.resolve_groups({"L": {"w": "int8", "a": "int4"}}, groups)
    assert flat["L"] == "w8a4"
    assert extra["w8a4"]["w_elem_format"] == "int8"
    assert extra["w8a4"]["a_elem_format"] == "int4"


# =============================================================================
# Per-rung curve, cost budget, verification pass
# =============================================================================

def test_per_rung_probing_builds_a_curve(tmp_path):
    torch.manual_seed(0)
    q = _quantizer(tmp_path, _auto_config(probe="all_rungs", reference="marginal"))
    plan = q.plan_mixed_precision(_CanaryNet().eval(),
                                  data=[torch.randn(2, 3, 16, 16)], write=False)
    for row in plan["rows"]:
        assert set(row["per_rung"]) == {"int4", "int6", "int8"}
        assert row["per_rung"]["int8"] == 0.0        # the reference itself
        # lower precision must not hurt less than higher precision
        assert row["per_rung"]["int4"] >= row["per_rung"]["int6"]


def test_cost_budget_refuses_a_rank_only_artifact():
    """Trading dB against bits needs a curve, not a ranking."""
    rows = _rows(5)
    cfg = {"ladder": ["int4", "int6", "int8"], "strategy": "cost_budget",
           "cost_budget": {"target_avg_bits": 5.0}}
    with pytest.raises(mxa.AssignError, match="per-rung"):
        mxa.assign(rows, cfg, _LADDER)


def test_cost_budget_hits_the_target_and_spares_the_costly_layers():
    """Demote by bits-saved-per-dB-lost, so a big sensitive layer is kept."""
    rows = [
        # huge and sensitive: demoting it saves a lot but costs a lot of dB
        {"name": "big_sensitive", "sensitivity": 40.0, "status": "ok",
         "cost": {"macs": 1_000_000},
         "per_rung": {"int4": 40.0, "int6": 20.0, "int8": 0.0}},
        # huge and flat: demoting it is nearly free
        {"name": "big_flat", "sensitivity": 1.0, "status": "ok",
         "cost": {"macs": 1_000_000},
         "per_rung": {"int4": 1.0, "int6": 0.5, "int8": 0.0}},
        {"name": "small", "sensitivity": 5.0, "status": "ok",
         "cost": {"macs": 1_000},
         "per_rung": {"int4": 5.0, "int6": 2.0, "int8": 0.0}},
    ]
    cfg = {"ladder": ["int4", "int6", "int8"], "strategy": "cost_budget",
           "cost_budget": {"target_avg_bits": 6.5}}
    assignments, _ = mxa.assign(rows, cfg, _LADDER)
    assert assignments["big_flat"] == "int4"
    assert assignments["big_sensitive"] == "int8"


def test_verification_pass_reports_the_finished_mix(tmp_path):
    """One extra pass measures the assigned network, not a per-layer proxy."""
    torch.manual_seed(0)
    q = _quantizer(tmp_path, _auto_config(reference="marginal"))
    plan = q.plan_mixed_precision(_CanaryNet().eval(),
                                  data=[torch.randn(2, 3, 16, 16)], write=False)
    verified = plan["meta"]["assigned_vs_reference"]
    assert verified["status"] == "ok"
    assert verified["sqnr_db"] is not None


# =============================================================================
# Hand-written layer entries survive the ladder
# =============================================================================

class _ConvTNet(nn.Module):
    """Two convs and a ConvTranspose2d, as in the DOF decoder blocks."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.up = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.conv2 = nn.Conv2d(32, 8, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.up(self.conv1(x)))


def _convT_config(**over):
    """A ladder over an xblock_accum deployment spec, with the convT pinned.

    The shape of configs/mx_config_dof_npe_auto_ladder.json: the accumulator
    model lives on mx_specs, the rungs carry formats only, and the transpose
    convolution is pinned to a group that has no accumulator model at all.
    """
    cfg = _auto_config(**over)
    cfg["mx_specs"] = _XBLOCK_FP
    cfg["groups"]["convT_plain"] = dict(_INT8)
    cfg["layers"] = ["conv1", "conv2", {"name": "up", "group": "convT_plain"}]
    return cfg


def test_explicit_group_in_layers_pins_the_layer(tmp_path):
    """A layer that already names a group is not re-assigned by the ladder."""
    torch.manual_seed(0)
    q = _quantizer(tmp_path, _convT_config(reference="marginal"))
    plan = q.plan_mixed_precision(_ConvTNet().eval(),
                                  data=[torch.randn(2, 3, 16, 16)], write=False)
    assert plan["assignments"]["up"] == "convT_plain"
    assert plan["assignments"]["conv1"] in ("int4", "int6", "int8")
    assert "up" in plan["notes"]["pinned"]


def test_pinned_group_keeps_its_own_spec(tmp_path):
    """The pin is used raw — it does not inherit the deployment accumulator."""
    torch.manual_seed(0)
    q = _quantizer(tmp_path, _convT_config(reference="marginal"))
    plan = q.plan_mixed_precision(_ConvTNet().eval(),
                                  data=[torch.randn(2, 3, 16, 16)], write=False)
    groups = plan["config"]["groups"]
    assert "xblock_accum" not in groups["convT_plain"]
    for rung in ("int4", "int6", "int8"):
        if rung in groups:
            assert groups[rung]["xblock_accum"]["enabled"] is True


def test_ladder_inherits_xblock_accum_from_mx_specs(tmp_path):
    """Rungs may not *carry* xblock_accum, but they must inherit it.

    The rejection is about what the user wrote on a rung; the merged rung spec
    always has the deployment accumulator, and validating that would reject
    every accumulator-model config.
    """
    torch.manual_seed(0)
    q = _quantizer(tmp_path, _convT_config(reference="marginal"))
    plan = q.plan_mixed_precision(_ConvTNet().eval(),
                                  data=[torch.randn(2, 3, 16, 16)], write=False)
    assert plan["meta"]["n_candidates"] == 3


def test_inline_layer_mx_specs_becomes_a_pinned_group(tmp_path):
    """An inline per-layer spec is honoured, not silently dropped."""
    torch.manual_seed(0)
    cfg = _auto_config(reference="marginal")
    cfg["layers"] = ["conv1", "conv3",
                     {"name": "conv2", "mx_specs": dict(_INT8, w_elem_format="int2")}]
    q = _quantizer(tmp_path, cfg)
    plan = q.plan_mixed_precision(_CanaryNet().eval(),
                                  data=[torch.randn(2, 3, 16, 16)], write=False)
    group = plan["assignments"]["conv2"]
    assert group == "fixed_conv2"
    assert plan["config"]["groups"][group]["w_elem_format"] == "int2"


def test_auto_mixed_pins_may_name_a_non_ladder_group(tmp_path):
    """auto_mixed.pins is not restricted to the ladder either."""
    torch.manual_seed(0)
    cfg = _convT_config(reference="marginal",
                        pins={"conv1": "convT_plain"})
    q = _quantizer(tmp_path, cfg)
    plan = q.plan_mixed_precision(_ConvTNet().eval(),
                                  data=[torch.randn(2, 3, 16, 16)], write=False)
    assert plan["assignments"]["conv1"] == "convT_plain"


def test_auto_mixed_pin_beats_the_layers_entry(tmp_path):
    """The more specific statement of intent wins."""
    torch.manual_seed(0)
    cfg = _convT_config(reference="marginal", pins={"up": "int8"})
    q = _quantizer(tmp_path, cfg)
    plan = q.plan_mixed_precision(_ConvTNet().eval(),
                                  data=[torch.randn(2, 3, 16, 16)], write=False)
    assert plan["assignments"]["up"] == "int8"


def test_unknown_group_in_layers_is_an_error(tmp_path):
    """A typo'd group name fails before any scoring runs."""
    cfg = _auto_config()
    cfg["layers"] = ["conv1", {"name": "conv2", "group": "nope"}]
    q = _quantizer(tmp_path, cfg)
    with pytest.raises(ValueError, match="nope"):
        q.plan_mixed_precision(_CanaryNet().eval(), data=[torch.randn(2, 3, 16, 16)],
                               write=False)


# =============================================================================
# Calibration batches meet the model's device
# =============================================================================

def test_to_device_walks_nested_batches():
    """Dicts, lists, tuples and namedtuples all get their tensors moved."""
    import mx_sensitivity as mxs
    batch = {"img1": torch.zeros(2), "meta": ["a", torch.ones(2)],
             "pair": (torch.ones(1), 3)}
    out, n = mxs.to_device(batch, torch.device("cpu"))
    assert n == 0                      # already there: nothing moved, no copies
    assert out["meta"][0] == "a"
    assert out["pair"][1] == 3
    assert torch.is_tensor(out["img1"])


def test_to_device_leaves_non_tensors_alone():
    """Strings, ints and None survive the walk untouched."""
    import mx_sensitivity as mxs
    out, n = mxs.to_device({"name": "x", "k": None, "n": 7}, torch.device("cpu"))
    assert out == {"name": "x", "k": None, "n": 7}
    assert n == 0


def test_model_device_reports_parameter_device():
    import mx_sensitivity as mxs
    assert mxs.model_device(_CanaryNet()).type == "cpu"
    assert mxs.model_device(nn.Module()) is None
