import re
import math
import os
import json
import copy
import sys

sys.path.append('/Users/avrahamraviv/PycharmProjects')
sys.path.append('/home/avrahamra/PycharmProjects')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from microxcaling.mx.convolution import Conv2d as MXConv2d
from microxcaling.mx.transpose_convolution import ConvTranspose2d as MXConvTranspose2d
from microxcaling.mx.linear import Linear as MXLinear
from microxcaling.mx.mx_ops import quantize_mx_op
from microxcaling.mx import MxSpecs

from fixed_point.mx_fixed_point import normalize_xblock_accum
from fixed_point.fxp_quant import (
    fake_quant_fxp, fxp_clip_stats, fxp_format_str, fxp_stats_value,
    normalize_out_quant,
)
from mx_layers_blocked import MXConv2dBlocked, MXLinearBlocked, MXConv2dHW
from mx_layers_act import MXActQuant
from mx_stats import collect_stats as _collect_stats_impl
import mx_assign as _mxa
import mx_sensitivity as _mxs
from mx_sensitivity import clean_name as _clean_name


def _out_quant_tensor(t, state):
    """Quantize one output tensor and fold its error into the running stats."""
    cfg = state["cfg"]
    if not torch.is_tensor(t) or not t.is_floating_point():
        return t

    q = fake_quant_fxp(
        t,
        frac_bits=cfg["frac_bits"], total_bits=cfg["total_bits"],
        signed=cfg["signed"], round_mode=cfg["round"],
        saturate=cfg["saturate"], clip_grad=cfg["clip_grad"],
    )
    n, n_clip, ssq, sse = fxp_clip_stats(
        t, q, frac_bits=cfg["frac_bits"], total_bits=cfg["total_bits"],
        signed=cfg["signed"], round_mode=cfg["round"],
    )
    state["n"] += n
    state["n_clipped"] += n_clip
    state["sum_sq"] += ssq
    state["sum_sq_err"] += sse
    return q


def _out_quant_hook(mod, inp, out):
    """Forward hook: replace a module's output with its fixed-point quantization.

    Module-level (not a closure over the module) so that deepcopying a
    quantized model keeps each copy's hook bound to its own state.
    """
    state = mod._mx_out_quant
    state["n_calls"] += 1
    sel = state["cfg"]["outputs"]

    if torch.is_tensor(out):
        return _out_quant_tensor(out, state)
    if isinstance(out, (tuple, list)):
        keep = range(len(out)) if sel == "all" else sel
        new = [_out_quant_tensor(o, state) if i in keep else o
               for i, o in enumerate(out)]
        return type(out)(new) if isinstance(out, list) else tuple(new)
    return out          # dict / dataclass outputs: left alone


class MXQuantizer:
    """
    MXQuantizer: Replace selected Conv2d and Linear layers with MX-quantized equivalents,
    with optional PTQ via GPTQ-style weight reconstruction.

    Features:
    - Supports three config formats:
        1. Global mx_specs + list of layer names
        2. Per-layer mx_specs
        3. Named groups of specs, assigned per layer
    - Preserves original weights and bias (pre-PTQ)
    - PTQ (GPTQ-style): forward-only, no backward required
    - Debug printing of replaced and missed layers

    Usage:
    >>> quantizer = MXQuantizer(save_dir="/path/to/save")
    >>> quantized_model = quantizer.quant(model)            # no PTQ
    >>> quantized_model = quantizer.quant(model, data=cal)  # with PTQ

    PTQ behaviour (controlled by optional "ptq" key in config):
    - key absent                    → PTQ runs if data given, 128 batches
    - "ptq": {"enabled": true/false, "batches": N}  → full control
    - "ptq": false                  → shorthand to disable PTQ

    Config (`mx_config.json`) examples:

    1. Shared/global mx_specs:
    {
        "mx_specs": {
            "w_elem_format": "fp8_e4m3",
            "a_elem_format": "fp8_e4m3",
            "block_size": 32,
            "scale_bits": 8,
            "shared_exp_method": "max",
            "custom_cuda": true
        },
        "layers": [
            {"name": "layer1.conv1"},
            {"name": "layer2.fc1"}
        ]
    }

    2. Per-layer mx_specs:
    {
        "layers": [
            {
                "name": "layer1.conv1",
                "mx_specs": {"w_elem_format": "fp8_e4m3"}
            },
            {
                "name": "layer2.fc1",
                "mx_specs": {"w_elem_format": "int8"}
            }
        ]
    }

    3. Named groups:
    {
        "groups": {
            "high_precision": {"w_elem_format": "fp8_e4m3", "block_size": 32},
            "low_precision":  {"w_elem_format": "int4",     "block_size": 16}
        },
        "layers": [
            {"name": "layer1.conv1", "group": "high_precision"},
            {"name": "layer2.fc1",   "group": "low_precision"}
        ]
    }

    4. PTQ control:
    {
        "ptq": {"enabled": true, "batches": 128},
        "layers": [...]
    }
    Shorthand to disable: "ptq": false

    5. Quantization statistics (per-block stats + quant error, off by default):
    {
        "collect_stats": {"enabled": true, "batches": 32, "histograms": false,
                          "output_error": true, "detail": false, "save_json": true},
        "layers": [...]
    }
    Shorthand to enable with defaults: "collect_stats": true
    Also callable standalone: quantizer.collect_stats(quant_model, data=cal)

    6. Fixed-point output quantization (static scale, not MX):
    {
        "layers": [
            {"name": "model", "kind": "out_quant",
             "total_bits": 16, "frac_bits": 8, "signed": true,
             "round": "half_away", "saturate": true, "clip_grad": false}
        ]
    }
    The named module's output is snapped to a lattice of step 2^-frac_bits and
    clamped to the word's range — e.g. Q8.8 signed = step 1/256, range
    [-128, +127.996]. Attached as a forward hook, so state_dict keys are
    unchanged. Reported by _print_stat as clip rate + SQNR; clip rate is the
    number that matters, since a static scale (unlike MX) can overflow.
    "outputs" (default "all") selects indices when the module returns a tuple.

    Priority (highest to lowest): per-layer mx_specs > group > global mx_specs > defaults
    Note: scale_bits is shared between weights and activations (library limitation).
    """

    # GPTQ damping factor: fraction of mean diagonal to add for numerical stability
    _GPTQ_DAMPING = 0.01

    def __init__(self, save_dir, log=None):
        self.save_dir = save_dir
        self.config_path = os.path.join(save_dir, "mx_config.json")
        self.config = self._load_config()
        self._last_plan = None        # set by plan_mixed_precision, read by quant
        if log is not None:
            log.info(f"Load MX configuration from: {self.save_dir}")

    # =========================
    # Public API
    # =========================
    def quant(self, model, data=None, forward_fn=None, log=None):
        """
        Replace configured Conv2d and Linear layers with MX-quantized equivalents,
        then optionally run GPTQ-style PTQ if calibration data is provided.

        Args:
            model (nn.Module): Original PyTorch model.
            data (iterable, optional): Calibration batches passed to forward_fn.
            forward_fn (callable, optional): How to run one calibration batch.
                Signature: forward_fn(model, batch). If None, defaults to
                model(batch[0]) for tuple/list batches, model(**batch) for dicts,
                and model(batch) for plain tensors. Use this to handle models
                whose forward() requires more than a single image tensor.

                Example — model that takes (images, memory, anchors):
                    forward_fn=lambda m, b: m(b[0], b[2])

        Returns:
            nn.Module: Quantized model with MXConv2d / MXLinear layers.
        """
        if self.config is None:
            print("No MX configuration found. Skipping quantization.")
            return model

        fp32_model = model                   # keep FP32 reference before deep-copy
        model = copy.deepcopy(model)

        # Parse PTQ config early (needed for auto_mixed calibration batch count)
        ptq_cfg = self.config.get("ptq", {})
        if isinstance(ptq_cfg, bool):        # allow shorthand "ptq": false
            ptq_enabled, ptq_batches = ptq_cfg, 128
        else:
            ptq_enabled = ptq_cfg.get("enabled", True)
            ptq_batches = ptq_cfg.get("batches", 128)

        # Auto mixed-precision: measure isolated sensitivity, assign precision per layer
        auto_mixed = self.config.get("auto_mixed")
        groups = self.config.get("groups", {})

        if auto_mixed and "ladder" in auto_mixed:
            # New N-rung path: score, assign, install.
            # A plan already computed on this quantizer is reused rather than
            # re-scored: scoring costs real forward passes, and with a shuffled
            # loader a second pass would quietly produce a different assignment
            # than the one written to disk and reviewed. `replan: true` forces
            # a fresh one.
            plan = self._last_plan
            if plan is not None and not auto_mixed.get("replan", False):
                self._log(log, "auto_mixed | reusing the plan from "
                               "plan_mixed_precision() (set auto_mixed.replan "
                               "to re-score)")
            else:
                plan = self.plan_mixed_precision(
                    fp32_model, data=data, forward_fn=forward_fn, log=log,
                    write=auto_mixed.get("write_artifacts", True))
            all_groups = dict(plan["config"]["groups"])
            layer_map = {name: self._build_mx_specs(all_groups[grp])
                         for name, grp in plan["assignments"].items()}
            self._replace_layers(model, layer_map=layer_map)
        elif auto_mixed:
            # Legacy two-rung path (base/upgrade), kept bit-for-bit as it was.
            candidates = self._get_candidate_layers(fp32_model)
            base_specs = self._build_mx_specs(groups[auto_mixed["base"]])
            if data is not None:
                sens_batches = auto_mixed.get("batches", ptq_batches)
                sensitivity = self._measure_isolated_sensitivity(
                    fp32_model, candidates, base_specs, data, forward_fn, sens_batches, log)
                assignments = self._auto_assign_precisions(sensitivity, auto_mixed, log)
                # Candidates with no sensitivity data (hooks never fired) → upgrade as safe fallback
                unmeasured = [n for n in candidates if n not in sensitivity]
                if unmeasured:
                    self._log(log, f"auto_mixed | {len(unmeasured)} layers had no sensitivity data "
                                   f"→ defaulting to {auto_mixed['upgrade']}:")
                    for n in unmeasured:
                        self._log(log, f"auto_mixed |   - {n}")
                        assignments[n] = auto_mixed["upgrade"]
            else:
                self._log(log, "auto_mixed | No calibration data — using base format for all layers.")
                assignments = {name: auto_mixed["base"] for name in candidates}
            layer_map = {name: self._build_mx_specs(groups[grp])
                         for name, grp in assignments.items()}
            self._replace_layers(model, layer_map=layer_map)
        else:
            self._replace_layers(model)

        if data is not None and ptq_enabled:
            self._ptq(model, data, forward_fn, log, ptq_batches)

        measure_cfg = self.config.get("measure_error", True)
        if data is not None and measure_cfg:
            errors = self._measure_error(fp32_model, model, data, forward_fn, log, ptq_batches)
            model._quant_errors = errors

        stats_cfg = self.config.get("collect_stats", False)
        if stats_cfg is True or (isinstance(stats_cfg, dict) and stats_cfg.get("enabled", True)):
            self.collect_stats(model, data, forward_fn, log, fp32_model=fp32_model)

        self._print_stat(model, log)
        return model

    def collect_stats(self, model, data=None, forward_fn=None, log=None,
                      fp32_model=None, **overrides):
        """
        Collect per-block / per-layer quantization statistics on an
        already-quantized model (see mx_stats.collect_stats for details).

        Per MX layer, for weights and (with data) activations:
          - per-block max_abs / variance / mean_abs / dynamic range,
            underflow rate, shared-exponent distribution
          - quantization error err = x - Q(x): SQNR dB, MSE, max abs err, cos sim
          - isolated layer-output error (fp32 functional vs quantized forward) —
            the level at which plain/Blocked/HW variants actually differ
          - propagated output error (merged from _measure_error) when
            fp32_model and data are both given

        Options come from the optional "collect_stats" config key
        ({"enabled", "batches", "histograms", "output_error", "detail",
        "save_json"}) and can be overridden via **overrides.

        Returns the stats dict; also attached as model._quant_stats and,
        unless save_json is false, dumped to save_dir/quant_stats.json.
        """
        cfg = self.config.get("collect_stats", {}) if self.config else {}
        if isinstance(cfg, bool):
            cfg = {}
        opts = dict(max_batches=cfg.get("batches", 32),
                    histograms=cfg.get("histograms", False),
                    detail=cfg.get("detail", False),
                    output_error=cfg.get("output_error", True))
        opts.update(overrides)
        save_path = (os.path.join(self.save_dir, "quant_stats.json")
                     if cfg.get("save_json", True) else None)
        stats = _collect_stats_impl(model, data, forward_fn, log=log,
                                    save_path=save_path, **opts)
        if stats and fp32_model is not None and data is not None:
            propagated = self._measure_error(fp32_model, model, data,
                                             forward_fn, log, opts["max_batches"])
            for name, metrics in propagated.items():
                if name in stats["layers"]:
                    stats["layers"][name]["output_error"]["propagated"] = metrics
        model._quant_stats = stats
        return stats

    # =========================
    # Automated mixed precision
    # =========================
    def plan_mixed_precision(self, model, data=None, forward_fn=None, log=None,
                             output_fn=None, write=True):
        """Score every candidate layer, assign it a precision, write the plan.

        Deliberately separate from `quant()`. Scoring costs real forward passes,
        and the result — which layer runs at which format — is a decision worth
        reviewing before a training run depends on it. So this produces two
        artifacts in `save_dir` and stops:

          sensitivity.json        every score, why, and what it earned
          mx_config_resolved.json a plain groups+layers config, no auto_mixed

        The resolved config is what training points at: re-running it is a
        deterministic replay with no scoring and nothing left to re-derive, and
        it can be hand-edited where you disagree with an assignment.

        Args:
            model: FP32 model. Never mutated — everything runs on a deep copy.
            data: calibration batches; a one-shot iterator is fine, the first
                `auto_mixed.batches` of them are materialized once.
            forward_fn: forward_fn(model, batch), as elsewhere in this class.
            output_fn: reduces the model's return value to a list of float
                tensors for comparison. Defaults to a recursive flatten, which
                handles tuple / list / dict / dataclass returns.
            write: set False to compute the plan without touching the filesystem.

        Returns:
            dict: {"scores", "assignments", "rows", "summary", "config", "meta"}.
        """
        cfg = (self.config or {}).get("auto_mixed")
        if not cfg or "ladder" not in cfg:
            raise ValueError(
                "plan_mixed_precision needs an 'auto_mixed' config block with a "
                "'ladder' (lowest precision first), e.g. "
                '"ladder": ["int4", "int6", "int8"]')

        groups = self.config.get("groups", {})
        deploy_spec = self._deploy_spec(cfg, groups)
        rungs = {g: self._rung_spec(g, groups, deploy_spec) for g in cfg["ladder"]}

        # A layer that already names a group (or its own mx_specs) in "layers"
        # has had its decision made by hand; the ladder does not get to overrule
        # it. Those specs are used raw, exactly as the explicit path uses them,
        # so a group that deliberately omits xblock_accum keeps omitting it.
        cfg, group_table = self._freeze_explicit_layers(cfg, groups, rungs)
        _mxa.validate(cfg, rungs, quant_probe=self._trial_quantize,
                      all_groups=group_table, raw_groups=groups)

        ladder = list(cfg["ladder"])
        top, bottom = ladder[-1], ladder[0]
        probe_group = cfg.get("probe_group", bottom)
        method = cfg.get("scorer", "oat_output")

        # Not every deployment path can run a different format for weights and
        # activations. Checked here, before any forward pass, because the probe
        # that violates it does not fail until the middle of the refinement
        # stage — with the whole scoring run already spent.
        wa_cfg = dict(cfg.get("separable_wa") or {})
        wa_ok, wa_why = _mxa.split_wa_supported(deploy_spec)
        wa_note = None
        if wa_cfg.get("enabled") and not wa_ok:
            wa_cfg["enabled"] = False
            wa_note = f"separable_wa disabled: {wa_why}"
            self._log(log, f"auto_mixed | WARNING: {wa_note}")

        base_model, was_dp = _mxs.unwrap_parallel(model)
        candidates = self._get_candidate_layers(base_model)
        if not candidates:
            raise ValueError(
                "auto_mixed found no candidate layers. An empty \"layers\": [] "
                "list means 'quantize nothing' — remove the key to auto-discover "
                "every Conv2d / ConvTranspose2d / Linear.")

        batches = _mxs.materialize(data, int(cfg.get("batches", 8)))

        # A DataLoader hands out CPU tensors and the training loop moves them to
        # the GPU itself; nothing here would have. Mixed devices are not always
        # a loud failure — the HW fixed-point conv selects its Triton kernel on
        # `qi.is_cuda`, so CPU activations would quietly fall back to the torch
        # reference and score arithmetic the deployment never runs.
        if batches and cfg.get("move_batches", True):
            device = _mxs.model_device(base_model)
            if device is not None:
                batches, n_moved = _mxs.to_device(batches, device)
                if n_moved:
                    self._log(log, f"auto_mixed | moved {n_moved} batch tensor(s) "
                                   f"to {device}")

        # The reference network: every candidate at the top rung, built through
        # the normal replacement path so xblock_accum, act_quant wrapping and
        # out_quant hooks are all in place — the scorer must see the deployed
        # arithmetic, not an approximation of it.
        ref_model = copy.deepcopy(base_model)
        ref_map = self._reference_map(candidates, cfg, group_table, top)
        self._replace_layers(ref_model, layer_map=ref_map, verbose=0)

        types = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
        entries = _mxs.resolve_bindings(ref_model, candidates, types)
        costs = self._layer_costs(ref_model, entries, batches, forward_fn)

        probe_specs = self._build_mx_specs(rungs[probe_group])

        def build_probe(name, module):
            return self._build_mx_module(module, probe_specs, name=name, verbose=0)

        ctx = _mxs.ScoreContext(
            ref_model=ref_model, entries=entries, batches=batches,
            build_probe=build_probe, forward_fn=forward_fn, output_fn=output_fn,
            options=self._scorer_options(cfg),
            log=(lambda m: self._log(log, m)))

        n_probes = len(entries)
        self._log(log, f"auto_mixed | scorer={method} probe={probe_group} "
                       f"reference={top} layers={n_probes} "
                       f"batches={len(batches) if batches else 0}")
        self._log_eta(ref_model, batches, forward_fn, n_probes, cfg, log)

        scores = _mxs.get_scorer(method)(ctx)
        meta = scores.pop("__meta__", {})

        rows = self._sensitivity_rows(scores, entries, costs)

        # Absolute reference: the same probe measured against plain FP32 instead
        # of the quantized baseline. The marginal score answers "what does
        # demoting this layer cost in the net we ship"; the absolute one answers
        # "how much does this layer dislike low precision at all". They usually
        # agree; where they do not, the rank delta says which layers are only
        # sensitive because of what surrounds them.
        want_abs = str(cfg.get("reference", "both")).lower() in ("both", "absolute")
        if want_abs and batches and method == "oat_output":
            self._absolute_scores(base_model, candidates, batches, forward_fn,
                                  output_fn, probe_specs, rows, meta, log)

        # w/a refinement: the joint probe ranks layers; solo probes over the most
        # sensitive ones say WHICH operand is responsible. Two layers can share
        # an output SQNR while one is weight-underflow driven and the other
        # activation-underflow driven, and spending bits on the wrong one buys
        # nothing.
        # A per-rung curve, not just a ranking: what does this layer cost at
        # int6, at int4? Required by cost_budget, which trades dB against bits
        # and cannot do that from a rank alone.
        if (str(cfg.get("probe", "bottom")) == "all_rungs"
                and batches and method == "oat_output"):
            self._per_rung_scores(ref_model, entries, batches, forward_fn,
                                  output_fn, rungs, ladder, rows, log)

        if wa_cfg.get("enabled") and batches and method == "oat_output":
            self._refine_wa(ref_model, entries, batches, forward_fn, output_fn,
                            rungs, top, probe_group, rows, wa_cfg, log)
        assignments, notes = _mxa.assign(rows, cfg, group_table,
                                         log=lambda m: self._log(log, m))
        assignments = self._apply_wa_split(assignments, rows, ladder, wa_cfg, log)
        flat, extra = _mxa.resolve_groups(assignments, group_table, deploy_group=None)
        summary = _mxa.cost_summary(flat, group_table, extra, costs)

        for row in rows:
            row["assigned"] = flat.get(row["name"])
            row["pinned"] = row["name"] in notes["pinned"]

        meta.update({"scorer": method, "probe_group": probe_group,
                     "reference_group": top, "ladder": ladder,
                     "strategy": cfg.get("strategy", "quantile"),
                     "dataparallel_unwrapped": was_dp,
                     "n_candidates": len(candidates),
                     "pinned": sorted(cfg.get("pins") or {}),
                     "separable_wa": wa_note or bool(wa_cfg.get("enabled"))})

        # OAT scores every layer in the quietest possible context — everything
        # else at the top rung. The mixed net is noisier than that, so scores are
        # optimistic. One extra pass measures what was actually bought.
        if batches and cfg.get("verify", True):
            baseline, assigned, relative = self._verify_assignment(
                base_model, ref_map, flat, extra, group_table, top, batches,
                forward_fn, output_fn, log)
            meta["baseline_vs_fp32"] = baseline
            meta["assigned_vs_fp32"] = assigned
            meta["assigned_vs_reference"] = relative

        resolved = _mxa.build_resolved_config(
            self.config, flat, group_table, extra,
            provenance={"generated_by": "MXQuantizer.plan_mixed_precision",
                        "scorer": method, "probe_group": probe_group,
                        "reference_group": top})

        self._print_sensitivity_table(rows, summary, meta, log)

        if write:
            _mxa.write_sensitivity(os.path.join(self.save_dir, "sensitivity.json"),
                                   meta, rows, summary)
            _mxa.write_resolved_config(
                os.path.join(self.save_dir, "mx_config_resolved.json"), resolved)
            self._log(log, f"auto_mixed | wrote sensitivity.json and "
                           f"mx_config_resolved.json to {self.save_dir}")

        self._last_plan = {"scores": scores, "assignments": flat, "rows": rows,
                           "summary": summary, "config": resolved, "meta": meta,
                           "notes": notes}
        return self._last_plan

    def _per_rung_scores(self, ref_model, entries, batches, forward_fn, output_fn,
                         rungs, ladder, rows, log=None):
        """Score every layer at every rung below the top, not just the bottom.

        Turns a ranking into a degradation curve, which is what a cost-aware
        assignment needs: demoting a layer one rung has to be priced in dB
        before it can be traded against the bits it saves.
        """
        by_name = {r["name"]: r for r in rows}
        for rung in ladder[:-1]:
            spec = self._build_mx_specs(rungs[rung])
            self._log(log, f"auto_mixed | per-rung pass at {rung} ...")
            res = _mxs.score_oat(
                ref_model, entries, batches, verbose=False,
                build_probe=lambda n, mod, _s=spec: self._build_mx_module(
                    mod, _s, name=n, verbose=0),
                forward_fn=forward_fn, output_fn=output_fn,
                min_floor_db=self._min_floor())
            res.pop("__meta__", None)
            for name, r in res.items():
                by_name[name].setdefault("per_rung", {})[rung] = r.get("sensitivity")
        for row in rows:
            row.setdefault("per_rung", {})[ladder[-1]] = 0.0   # the reference itself

    def _verify_assignment(self, base_model, ref_map, assignments, extra,
                           group_table, top, batches, forward_fn, output_fn,
                           log=None):
        """Measure the finished networks against FP32 and against each other.

        Three numbers from one loop, because on their own each is misleading:

          baseline_vs_fp32       the all-top-rung network vs FP32 — the budget
          assigned_vs_fp32       the mixed network vs FP32 — what you ship
          assigned_vs_reference  mixed vs baseline — what the ladder cost

        The last one is what the OAT scores predict, but it cannot be read
        alone: "the mix is 30 dB below the int8 net" means something very
        different depending on whether the int8 net was 31 dB or 45 dB below
        FP32. And unlike any per-layer score, these have every demotion in
        place at once, so the interactions one-at-a-time probing cannot see are
        included.
        """
        all_groups = dict(group_table)
        all_groups.update(extra)
        output_fn = output_fn or _mxs.flatten_outputs

        ref = copy.deepcopy(base_model)
        self._replace_layers(ref, layer_map=ref_map, verbose=0)

        mixed = copy.deepcopy(base_model)
        self._replace_layers(mixed, layer_map={
            n: self._build_mx_specs(all_groups[g]) for n, g in assignments.items()},
            verbose=0)

        fp32 = base_model
        was_training = fp32.training
        fp32.eval()
        ref.eval()
        mixed.eval()
        base_acc, mix_acc, rel_acc = _mxs.ErrAcc(), _mxs.ErrAcc(), _mxs.ErrAcc()
        try:
            with torch.no_grad():
                for batch in batches:
                    f = [t.detach().float().cpu() for t in
                         output_fn(_mxs.default_forward(fp32, batch, forward_fn))]
                    a = [t.detach().float().cpu() for t in
                         output_fn(_mxs.default_forward(ref, batch, forward_fn))]
                    b = [t.detach().float().cpu() for t in
                         output_fn(_mxs.default_forward(mixed, batch, forward_fn))]
                    for tf, ta, tb in zip(f, a, b):
                        base_acc.update(tf, ta)
                        mix_acc.update(tf, tb)
                        rel_acc.update(ta, tb)
        finally:
            if was_training:
                fp32.train()

        def _pack(acc):
            status, metrics = acc.summary()
            return {"status": status, **metrics}

        baseline = _pack(base_acc)
        assigned = _pack(mix_acc)
        relative = _pack(rel_acc)

        b_db, m_db, r_db = (baseline.get("sqnr_db"), assigned.get("sqnr_db"),
                            relative.get("sqnr_db"))
        if b_db is not None and m_db is not None:
            self._log(log, f"auto_mixed | vs FP32: {top} baseline "
                           f"{b_db:.2f} dB -> assigned mix {m_db:.2f} dB "
                           f"(ladder cost {b_db - m_db:.2f} dB)")
        if r_db is not None:
            self._log(log, f"auto_mixed | assigned mix vs {top} reference: "
                           f"{r_db:.2f} dB SQNR")
        return baseline, assigned, relative

    def _log_eta(self, ref_model, batches, forward_fn, n_probes, cfg, log=None):
        """Time one batch and print what the scoring run will cost.

        OAT is (1 + probes) x batches forward passes. Saying so up front, in
        minutes, is the difference between an informed wait and a killed job.
        """
        if not batches:
            return
        import time
        t0 = time.time()
        with torch.no_grad():
            _mxs.default_forward(ref_model, batches[0], forward_fn)
        per_batch = time.time() - t0

        n_ref = 2                                   # reference + determinism check
        if str(cfg.get("reference", "both")).lower() in ("both", "absolute"):
            n_probes *= 2
            n_ref += 1
        n_fwd = (n_probes + n_ref) * len(batches)
        est = n_fwd * per_batch
        self._log(log, f"auto_mixed | {n_fwd} forward passes "
                       f"({per_batch:.2f}s/batch) -> est {est / 60:.1f} min")

    def _absolute_scores(self, base_model, candidates, batches, forward_fn,
                         output_fn, probe_specs, rows, meta, log=None):
        """Score every candidate against an FP32 reference, and compare rankings."""
        fp32_ref = copy.deepcopy(base_model)
        types = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)
        entries = _mxs.resolve_bindings(fp32_ref, candidates, types)

        def build_probe(name, module):
            return self._build_mx_module(module, probe_specs, name=name, verbose=0)

        self._log(log, "auto_mixed | absolute reference (vs FP32) ...")
        abs_scores = _mxs.score_oat(fp32_ref, entries, batches,
                                    build_probe=build_probe, forward_fn=forward_fn,
                                    output_fn=output_fn, verbose=False,
                                    min_floor_db=self._min_floor())
        abs_scores.pop("__meta__", None)

        order_marginal = [r["name"] for r in rows]
        by_abs = sorted((n for n in abs_scores),
                        key=lambda n: -(abs_scores[n].get("sensitivity") or -math.inf))
        abs_rank = {n: i for i, n in enumerate(by_abs)}
        for i, row in enumerate(rows):
            res = abs_scores.get(row["name"], {})
            row["scores"]["marginal"] = row["sensitivity"]
            row["scores"]["absolute"] = res.get("sensitivity")
            row["rank_delta"] = abs_rank.get(row["name"], i) - i

        meta["marginal_vs_absolute_spearman"] = _mxs.spearman(
            [r["sensitivity"] for r in rows],
            [r["scores"].get("absolute") for r in rows])
        rho = meta["marginal_vs_absolute_spearman"]
        if rho is not None:
            self._log(log, f"auto_mixed | marginal vs absolute rank correlation: "
                           f"{rho:.2f}")

    def _refine_wa(self, ref_model, entries, batches, forward_fn, output_fn,
                   rungs, top, probe_group, rows, wa_cfg, log=None):
        """Probe weights alone and activations alone, on the top-M layers only.

        Solo probes over every layer would triple an already expensive scheme,
        and buy nothing for layers headed to the bottom rung on both axes
        regardless. So refine only where the answer can change the assignment.
        """
        m = int(wa_cfg.get("refine_top", 16))
        ranked = [r["name"] for r in rows if r["sensitivity"] is not None][:m]
        if not ranked:
            return
        subset = {n: entries[n] for n in ranked if n in entries}
        self._log(log, f"auto_mixed | refining w/a on the top {len(subset)} layers")

        w_spec = self._build_mx_specs(
            dict(rungs[top], w_elem_format=rungs[probe_group]["w_elem_format"]))
        a_spec = self._build_mx_specs(
            dict(rungs[top], a_elem_format=rungs[probe_group]["a_elem_format"]))

        by_name = {r["name"]: r for r in rows}
        for label, spec in (("w", w_spec), ("a", a_spec)):
            res = _mxs.score_oat(
                ref_model, subset, batches, verbose=False,
                build_probe=lambda n, mod, _s=spec: self._build_mx_module(
                    mod, _s, name=n, verbose=0),
                forward_fn=forward_fn, output_fn=output_fn,
                min_floor_db=self._min_floor())
            res.pop("__meta__", None)
            for name, r in res.items():
                by_name[name]["scores"][label] = r.get("sensitivity")

        # Diagnostic only, never an assignment input: the operand errors are
        # correlated through the same activations and the shared block exponent,
        # so joint is measured, not inferred.
        for name in subset:
            row = by_name[name]
            s_w, s_a = row["scores"].get("w"), row["scores"].get("a")
            if s_w is not None and s_a is not None and row["sensitivity"] is not None:
                row["scores"]["interaction"] = row["sensitivity"] - max(s_w, s_a)

    @staticmethod
    def _apply_wa_split(assignments, rows, ladder, wa_cfg, log=None):
        """Demote the operand that is clearly not the problem, by one rung.

        A heuristic, and deliberately a small one: when a layer's weight and
        activation probes disagree by more than `margin_db`, the quiet operand
        drops one rung while the loud one keeps the assigned format. Both scores
        stay in the artifact, so a reviewer can overrule it in the resolved
        config.
        """
        if not wa_cfg.get("enabled"):
            return assignments
        margin = float(wa_cfg.get("margin_db", 3.0))
        by_name = {r["name"]: r for r in rows}
        split = 0
        for name, group in list(assignments.items()):
            if not isinstance(group, str) or group not in ladder:
                continue
            idx = ladder.index(group)
            if idx == 0:
                continue                      # already at the bottom rung
            scores = (by_name.get(name) or {}).get("scores") or {}
            s_w, s_a = scores.get("w"), scores.get("a")
            if s_w is None or s_a is None or abs(s_w - s_a) < margin:
                continue
            lower = ladder[idx - 1]
            if s_w > s_a:
                assignments[name] = {"w": group, "a": lower}
            else:
                assignments[name] = {"w": lower, "a": group}
            split += 1
        if split and log is not None:
            print(f"[MXQuantizer] auto_mixed | {split} layer(s) got a split w/a format")
        return assignments

    def _reference_map(self, candidates, cfg, group_table, top):
        """The baseline every score is measured against.

        Every candidate at the top rung, except the pinned ones, which sit at
        the format they were pinned to. A pinned layer is not going to move, so
        putting it anywhere else would measure the other layers against a
        network that will never exist.
        """
        pins = cfg.get("pins") or {}
        out = {}
        for name in candidates:
            pin = pins.get(name)
            group = pin if isinstance(pin, str) else top
            out[name] = self._build_mx_specs(group_table[group])
        return out

    def _freeze_explicit_layers(self, cfg, groups, rungs):
        """Turn hand-written per-layer decisions into pins.

        Under `auto_mixed` the ladder assigns a group to every candidate, which
        would quietly overwrite a `{"name": ..., "group": ...}` entry the user
        wrote on purpose. Those entries are the escape hatch for layers the
        ladder cannot describe — a ConvTranspose2d that must stay plain MX while
        the ladder inherits `xblock_accum`, say — so they become pins instead.

        Pinned groups are taken **raw** from `groups`, not merged onto the
        deployment spec: the point of pinning a layer to `convT_plain` is that
        it does *not* get the accumulator model, and a merge would put it back.

        An explicit `auto_mixed.pins` entry wins over the `layers` entry for the
        same layer, since it is the more specific statement of intent.

        Returns:
            (cfg, group_table): `cfg` with the merged pins, and the group table
            to assign from — the rungs plus every group a pin names.
        """
        frozen, table = {}, dict(rungs)
        for entry in self.config.get("layers", []):
            if not isinstance(entry, dict):
                continue
            if entry.get("kind") in ("act_quant", "out_quant"):
                continue
            name = entry.get("name")
            if not name:
                continue
            if "mx_specs" in entry:
                # An inline spec has no name to pin to, so give it one. Without
                # this the spec would be silently discarded by the ladder.
                gname = "fixed_" + re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
                table[gname] = dict(entry["mx_specs"])
                frozen[name] = gname
            elif "group" in entry:
                gname = entry["group"]
                if gname not in groups:
                    raise ValueError(
                        f"layer '{name}' references group '{gname}', which is "
                        f"not defined in 'groups'")
                table.setdefault(gname, dict(groups[gname]))
                frozen[name] = gname

        if not frozen:
            return cfg, table

        pins = dict(frozen)
        pins.update(cfg.get("pins") or {})
        cfg = dict(cfg, pins=pins)
        return cfg, table

    def _deploy_spec(self, cfg, groups):
        """The spec that carries block geometry and the accumulator model.

        Ladder rungs say only which number format to use; everything structural
        — block_size, block_axes, xblock_accum — comes from here, so demoting a
        layer changes its precision and nothing else about how it runs.
        """
        name = cfg.get("deploy_group")
        if name is not None:
            if name not in groups:
                raise ValueError(f"auto_mixed.deploy_group '{name}' is not in 'groups'")
            return dict(groups[name])
        return dict(self.config.get("mx_specs") or {})

    @staticmethod
    def _rung_spec(group, groups, deploy_spec):
        """Deployment spec overridden by one rung's formats."""
        if group not in groups:
            raise ValueError(f"auto_mixed.ladder references undefined group '{group}'")
        return dict(deploy_spec, **groups[group])

    def _trial_quantize(self, spec_dict):
        """Quantize a tiny tensor to prove a spec is actually usable.

        Cheap insurance: an element format the library does not implement should
        surface here, not deep inside the first probe after a long scoring run.
        Run on CPU with custom_cuda off — this validates the *format*, and should
        not fail on a machine that merely lacks a CUDA toolchain.
        """
        specs = self._build_mx_specs(dict(spec_dict, custom_cuda=False))
        for key in ('w_elem_format', 'a_elem_format'):
            quantize_mx_op(torch.zeros(1, 32), specs, elem_format=specs[key],
                           axes=[-1], round=specs.get('round_mx_output', 'nearest'))

    def _min_floor(self):
        """Minimum acceptable run-to-run SQNR, in dB.

        Every OAT score is a difference of two forward passes, so it is only
        readable down to the level at which the model reproduces itself. Kept
        in one place because the marginal, absolute, per-rung and w/a passes
        all have to apply the same bar.
        """
        cfg = (self.config or {}).get("auto_mixed") or {}
        return float(cfg.get("min_noise_floor_db", 60.0))

    @staticmethod
    def _scorer_options(cfg):
        """Scorer-specific options, passed through to the registry entry."""
        opts = dict(cfg.get("scorer_options") or {})
        for key in ("path", "target", "capture_chunk", "min_noise_floor_db"):
            if key in cfg:
                opts.setdefault(key, cfg[key])
        return opts

    def _layer_costs(self, ref_model, entries, batches, forward_fn):
        """MACs and parameter counts per layer, from one shape-probe forward.

        Written into the artifact so a later reassignment — a different ladder,
        a different budget — is a pure offline recomputation with no model and
        no forward passes.
        """
        costs = {n: {"macs": 0, "params": 0, "type": type(e["module"]).__name__}
                 for n, e in entries.items()}
        for name, entry in entries.items():
            mod = entry["module"]
            costs[name]["params"] = int(sum(p.numel() for p in mod.parameters()))

        if not batches:
            return costs

        shapes = {}
        handles = []
        for name, entry in entries.items():
            def make_hook(n):
                def hook(mod, inp, out):
                    if torch.is_tensor(out):
                        shapes.setdefault(n, tuple(out.shape))
                return hook
            handles.append(entry["module"].register_forward_hook(make_hook(name)))
        try:
            with torch.no_grad():
                _mxs.default_forward(ref_model, batches[0], forward_fn)
        finally:
            for h in handles:
                h.remove()

        for name, shape in shapes.items():
            costs[name]["macs"] = _mxs.conv_macs(entries[name]["module"], shape)
        return costs

    @staticmethod
    def _sensitivity_rows(scores, entries, costs):
        """Flatten the scorer output into artifact rows, worst first."""
        rows = []
        for name, res in scores.items():
            entry = entries.get(name, {})
            rows.append({
                "name": name,
                "type": type(entry.get("module")).__name__ if entry else None,
                "status": res.get("status", _mxs.STATUS_OK),
                "sensitivity": res.get("sensitivity"),
                "scores": res.get("scores", {}),
                "per_rung": res.get("per_rung", {}),
                "metrics": res.get("metrics", {}),
                "why": res.get("why", {}),
                "cost": costs.get(name, {}),
                "n_calls": res.get("n_calls"),
                "aliases": entry.get("aliases", []),
                "shared_paths": entry.get("paths", []) if len(
                    entry.get("paths", [])) > 1 else [],
            })
        rows.sort(key=lambda r: (r["sensitivity"] is None,
                                 -(r["sensitivity"] or 0.0), r["name"]))
        return rows

    def _print_sensitivity_table(self, rows, summary, meta, log=None):
        """Ranked table, in the same shape as the collect_stats report."""
        self._log(log, f"Sensitivity ({meta.get('scorer')}, probe="
                       f"{meta.get('probe_group')}, worst first; "
                       f"Sens = -SQNR, higher = needs more bits):")
        self._log(log, f"  {'#':>3} {'Layer':<45} {'Type':<20} {'MACs':>12} "
                       f"{'SQNR(dB)':>9} {'Sens':>8} {'W':>8} {'A':>8} "
                       f"{'Status':>10}  {'->group':<10}")
        for i, row in enumerate(rows, 1):
            def _f(v, width=8):
                return f"{'N/A':>{width}}" if v is None else f"{v:>{width}.2f}"
            sqnr = (row.get("metrics") or {}).get("sqnr_db")
            scores = row.get("scores") or {}
            macs = (row.get("cost") or {}).get("macs") or 0
            self._log(log, f"  {i:>3} {row['name']:<45} {str(row['type']):<20} "
                           f"{macs:>12,} {_f(sqnr, 9)} {_f(row['sensitivity'])} "
                           f"{_f(scores.get('w'))} {_f(scores.get('a'))} "
                           f"{row['status']:>10}  "
                           f"{str(row.get('assigned') or ''):<10}")
        counts = ", ".join(f"{g}: {n}" for g, n in summary["per_rung_counts"].items())
        mw = summary.get("macs_weighted_avg_bits")
        self._log(log, f"  Assigned: {counts}")
        self._log(log, f"  Avg bits: {summary['avg_bits']:.2f} | "
                       f"MAC-weighted: {mw:.2f}" if mw is not None
                  else f"  Avg bits: {summary['avg_bits']:.2f}")
        rho = meta.get("half_split_spearman")
        if rho is not None:
            note = "" if rho >= 0.9 else "  <-- low: increase auto_mixed.batches"
            self._log(log, f"  Half-split rank correlation: {rho:.2f}{note}")
        floor = meta.get("noise_floor_db")
        if floor is not None and math.isfinite(floor):
            near = meta.get("layers_at_noise_floor") or 0
            note = (f"  <-- {near} layer(s) within 10 dB of it, order not "
                    f"reproducible") if near else ""
            self._log(log, f"  Run-to-run noise floor: {floor:.0f} dB{note}")

    # =========================
    # Config
    # =========================
    def _load_config(self):
        if not os.path.exists(self.config_path):
            print(f"mx_config.json not found in {self.save_dir}")
            return None

        with open(self.config_path, "r") as f:
            config = json.load(f)

        if "layers" not in config and "auto_mixed" not in config:
            raise ValueError("Config must contain 'layers' or 'auto_mixed'")

        return config

    # =========================
    # MX specs
    # =========================
    def _build_mx_specs(self, spec_dict=None):
        """
        Construct MxSpecs object with defaults and optional overrides.
        """
        mx_specs = MxSpecs()

        # defaults
        mx_specs['scale_bits'] = 8
        mx_specs['w_elem_format'] = 'fp8_e4m3'
        mx_specs['a_elem_format'] = 'fp8_e4m3'
        mx_specs['block_size'] = 32
        mx_specs['shared_exp_method'] = 'max'
        mx_specs['custom_cuda'] = True

        # xblock_accum is a single nested config (bool or dict), modelled on ptq.
        # Stored as python attribute so microxcaling.apply_mx_specs does not reject it.
        xblock_raw = None
        if spec_dict is not None:
            spec_dict = dict(spec_dict)
            if 'xblock_accum' in spec_dict:
                xblock_raw = spec_dict.pop('xblock_accum')
            for k, v in spec_dict.items():
                mx_specs[k] = v

        setattr(mx_specs, 'xblock_accum', normalize_xblock_accum(xblock_raw))
        return mx_specs

    # =========================
    # MX module construction
    # =========================
    def _build_mx_module(self, module, mx_specs, name="", verbose=1, summary=None):
        """
        Build the MX replacement for one Conv2d / ConvTranspose2d / Linear.

        Single source of truth for class selection — plain MX vs Blocked vs HW,
        decided from `mx_specs.xblock_accum`. Every caller goes through here, so
        a module built for measurement (sensitivity scoring) runs the same
        arithmetic as the one `_replace_layers` installs for deployment. That
        matters: on the NPE path the fixed-point accumulator is often the
        dominant error term, so probing a plain MXConv2d for a layer that
        deploys as MXConv2dHW measures the wrong noise.

        Weight and bias are shared (not copied) with `module`, as in
        `_replace_layers` — safe there because it operates on a deepcopy.

        Args:
            module: original nn.Conv2d / nn.ConvTranspose2d / nn.Linear.
            mx_specs: MxSpecs for this layer (carries xblock_accum as a py attr).
            name: layer name, used only in log messages.
            verbose: 0 silent, 1 warnings, 2 per-layer detail.
            summary: optional dict of lists to tally into; keys as built by
                `_replace_layers` ('hw', 'blocked', 'mx_default_conv',
                'mx_default_convT', 'mx_default_linear', 'fallback_conv',
                'fallback_linear'). Fallback buckets get (name, reason) tuples.

        Returns:
            nn.Module: the new MX layer.

        Raises:
            TypeError: for any other module type.
        """
        is_convT = isinstance(module, nn.ConvTranspose2d)
        # nn.ConvTranspose2d is NOT a subclass of nn.Conv2d — keep is_conv exclusive.
        is_conv = isinstance(module, nn.Conv2d) and not is_convT
        is_linear = isinstance(module, nn.Linear)
        if not (is_conv or is_convT or is_linear):
            raise TypeError(
                f"_build_mx_module: cannot quantize '{name}' of type "
                f"{type(module).__name__} — expected Conv2d, ConvTranspose2d or Linear.")

        def _tally(bucket, value):
            if summary is not None:
                summary[bucket].append(value)

        xblock_cfg = getattr(mx_specs, 'xblock_accum', None) or {}
        want_blocked = bool(xblock_cfg.get('enabled', False))
        mode = xblock_cfg.get('mode', 'fp32_partial')
        want_hw = want_blocked and mode == 'hw_fixed_point'
        bs = mx_specs.get('block_size', 0) if hasattr(mx_specs, 'get') else mx_specs['block_size']

        if is_convT:
            # ConvTranspose2d: plain MX only (no HW / blocked variants).
            if want_blocked and verbose >= 1:
                print(f"[MXQuantizer] xblock_accum ignored for convT "
                      f"'{name}'; ConvTranspose2d only supports plain MX.")
            new = MXConvTranspose2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                output_padding=module.output_padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                mx_specs=mx_specs,
            )
            _tally('mx_default_convT', name)
        elif is_conv:
            reason = None
            hw_pad = want_hw and bool(xblock_cfg.get('pad_channels', True))
            # NPE (weight flatten + act X-block) needs no channel divisibility:
            # the weight flattens per filter and the activation blocks along W.
            is_npe = (want_hw
                      and xblock_cfg.get('weight_blockify') == 'flatten'
                      and xblock_cfg.get('act_blockify') == 'xblock')
            if want_blocked and module.groups != 1:
                reason = f"groups={module.groups}"
            elif (want_blocked and bs > 0 and module.in_channels % bs != 0
                    and not hw_pad and not is_npe):
                reason = f"in_channels={module.in_channels} not divisible by block_size={bs}"
            use_blocked = want_blocked and reason is None
            if want_blocked and not use_blocked:
                if verbose >= 2:
                    print(f"[MXQuantizer] xblock_accum blocked path skipped for "
                          f"conv '{name}' ({reason}); using original MXConv2d.")
                _tally('fallback_conv', (name, reason))
            if use_blocked and want_hw:
                conv_cls = MXConv2dHW
                if verbose >= 2:
                    print(f"[MXQuantizer] conv '{name}' -> MXConv2dHW "
                          f"(hw_fixed_point; bits={xblock_cfg.get('bits')}, "
                          f"sat_mode={xblock_cfg.get('sat_mode')}, "
                          f"e_layer_min={xblock_cfg.get('e_layer_min')})")
                _tally('hw', name)
            elif use_blocked:
                conv_cls = MXConv2dBlocked
                _tally('blocked', name)
            else:
                conv_cls = MXConv2d
                _tally('mx_default_conv', name)
            _axes_keys = {k: mx_specs.get(k) for k in
                          ('block_axes', 'block_axes_act', 'block_axes_wt',
                           'block_shape', 'block_shape_act', 'block_shape_wt',
                           'block_size_wt', 'flatten_wt')
                          if mx_specs.get(k)}
            if _axes_keys and conv_cls is not MXConv2d and verbose >= 1:
                print(f"[MXQuantizer] WARNING: conv '{name}' uses "
                      f"{conv_cls.__name__}; {_axes_keys} is ignored "
                      f"(only MXConv2d fwd reads block_axes*/block_shape*).")
            new = conv_cls(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                mx_specs=mx_specs
            )
        else:
            reason = None
            if want_blocked and bs > 0 and module.in_features % bs != 0:
                reason = f"in_features={module.in_features} not divisible by block_size={bs}"
            use_blocked = want_blocked and reason is None
            if want_blocked and not use_blocked:
                if verbose >= 2:
                    print(f"[MXQuantizer] xblock_accum blocked path skipped for "
                          f"linear '{name}' ({reason}); using original MXLinear.")
                _tally('fallback_linear', (name, reason))
            linear_cls = MXLinearBlocked if use_blocked else MXLinear
            if use_blocked:
                _tally('blocked', name)
            else:
                _tally('mx_default_linear', name)
            new = linear_cls(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                mx_specs=mx_specs
            )

        # preserve weights/bias
        new.weight = module.weight
        new.bias = module.bias

        # Propagate xblock_accum config onto the new layer; microxcaling
        # re-builds its internal mx_specs and drops python attrs.
        if hasattr(mx_specs, 'xblock_accum'):
            setattr(new, 'xblock_accum', getattr(mx_specs, 'xblock_accum'))

        if isinstance(new, MXConv2dHW):
            new._mx_layer_name = name
            cfg_e = getattr(mx_specs, 'xblock_accum', {}).get('e_layer_min')
            if cfg_e is not None:
                new.e_layer_min = int(cfg_e)

        return new

    # =========================
    # Replacement logic
    # =========================
    def _replace_layers(self, model, layer_map=None, verbose=None):
        """
        Replace Conv2d and Linear layers based on config (or a pre-built layer_map).

        `verbose` overrides the level derived from the specs — used by the
        sensitivity planner, which builds several throwaway models and should
        not print a replacement summary for each one.
        """
        if layer_map is None:
            layer_map = self._build_layer_map()

        # Determine verbose level once from the first xblock-enabled mx_specs.
        if verbose is None:
            verbose = 1
            for spec in layer_map.values():
                xc = getattr(spec, 'xblock_accum', None)
                if xc and xc.get('enabled'):
                    verbose = int(xc.get('verbose', 1))
                    break

        replace_summary = {
            'hw': [], 'blocked': [], 'mx_default_conv': [], 'mx_default_convT': [],
            'mx_default_linear': [], 'fallback_conv': [], 'fallback_linear': [],
        }

        # Snapshot the module list before mutating the tree, and keep duplicate
        # bindings: a module reachable under two names is deduplicated by
        # named_modules() by default, so only one of its parents would be
        # rebound and the other call site would keep running FP32.
        all_modules = list(model.named_modules(remove_duplicate=False))

        built = {}          # id(original module) -> the MX module built for it
        bindings = {}       # id(original module) -> [full_name, ...]
        matched = set()     # layer_map keys that hit a real module

        for full_name, module in all_modules:
            is_convT = (isinstance(module, nn.ConvTranspose2d)
                        and not isinstance(module, MXConvTranspose2d))
            # nn.ConvTranspose2d is NOT a subclass of nn.Conv2d — keep is_conv exclusive.
            is_conv = (isinstance(module, nn.Conv2d) and not isinstance(module, MXConv2d)
                       and not is_convT)
            is_linear = isinstance(module, nn.Linear) and not isinstance(module, MXLinear)
            if not (is_conv or is_convT or is_linear):
                continue

            name = _clean_name(full_name)
            bindings.setdefault(id(module), []).append(name)
            if name not in layer_map:
                continue
            matched.add(name)

            mx_specs = layer_map[name]

            parent, leaf = self._get_parent(model, full_name)
            if parent is None:
                print(f"[MXQuantizer] WARNING: layer '{name}' is in the config but "
                      f"its parent module could not be resolved — not quantized.")
                continue

            # One MX module per original module, reused across every binding, so
            # tied layers stay tied instead of becoming independent copies.
            if id(module) in built:
                new = built[id(module)]
            else:
                new = self._build_mx_module(module, mx_specs, name=name,
                                            verbose=verbose, summary=replace_summary)
                built[id(module)] = new
            setattr(parent, leaf, new)

        self._warn_unreplaced(layer_map, matched, bindings, verbose)

        # Param-free ops (kind == 'act_quant') are wrapped after the conv/linear
        # pass: wrapping inserts an `.inner` level in the module path, which
        # would break name lookup for any conv/linear nested under them.
        n_act = self._wrap_act_layers(model, verbose=verbose)

        # Output fixed-point quant (kind == 'out_quant') goes last. It attaches a
        # forward hook rather than wrapping, so module paths and state_dict keys
        # are untouched — required, since this usually targets the whole model.
        n_out = self._install_out_quant(model, verbose=verbose)

        if verbose >= 1:
            n_hw = len(replace_summary['hw'])
            n_blk = len(replace_summary['blocked'])
            n_def_c = len(replace_summary['mx_default_conv'])
            n_def_ct = len(replace_summary['mx_default_convT'])
            n_def_l = len(replace_summary['mx_default_linear'])
            n_fb_c = len(replace_summary['fallback_conv'])
            n_fb_l = len(replace_summary['fallback_linear'])
            print(
                f"[MXQuantizer] replace summary: "
                f"hw={n_hw} blocked={n_blk} "
                f"mxconv={n_def_c} mxconvT={n_def_ct} mxlinear={n_def_l} "
                f"act_quant={n_act} out_quant={n_out} "
                f"fallback_conv={n_fb_c} fallback_linear={n_fb_l}"
            )
            if n_fb_c or n_fb_l:
                reasons = {}
                for _, r in replace_summary['fallback_conv'] + replace_summary['fallback_linear']:
                    reasons[r] = reasons.get(r, 0) + 1
                tally = ", ".join(f"{r}: {n}" for r, n in reasons.items())
                print(f"[MXQuantizer] fallback reasons: {tally}")

    @staticmethod
    def _warn_unreplaced(layer_map, matched, bindings, verbose=1):
        """Report config entries that quantized nothing, and partial tied layers.

        Both cases used to pass silently: a typo'd or stale layer name simply
        never matched a module, and a module bound under two names was only
        half-listed. Either way the layer stays FP32 while the config claims
        otherwise.
        """
        if verbose < 1:
            return

        missing = sorted(set(layer_map) - set(matched))
        if missing:
            print(f"[MXQuantizer] WARNING: {len(missing)} configured layer(s) matched no "
                  f"Conv2d/ConvTranspose2d/Linear in the model — check for typos or a "
                  f"stale config:")
            for n in missing[:20]:
                print(f"[MXQuantizer]   - {n}")
            if len(missing) > 20:
                print(f"[MXQuantizer]   ... and {len(missing) - 20} more")

        for names in bindings.values():
            if len(names) < 2:
                continue
            listed = [n for n in names if n in layer_map]
            if listed and len(listed) != len(names):
                skipped = [n for n in names if n not in layer_map]
                print(f"[MXQuantizer] WARNING: shared module quantized via {listed} "
                      f"but also reachable as {skipped}; all bindings now use the same "
                      f"MX layer.")

    def _build_layer_map(self):
        """
        Returns:
            dict: layer_name -> mx_specs

        Supports:
            1. layers as list[str] with global mx_specs
            2. layers as list[dict] with optional per-layer mx_specs
            3. layers as list[dict] with group reference

        Priority: per-layer mx_specs > group > global mx_specs
        """
        layer_map = {}

        global_specs = self.config.get("mx_specs", None)
        groups = self.config.get("groups", {})

        for layer in self.config["layers"]:
            if isinstance(layer, str):
                name = layer
                spec_dict = global_specs
            else:
                name = layer["name"]
                if layer.get("kind") in ("act_quant", "out_quant"):
                    continue          # handled by _build_act_map / _build_out_map
                if "mx_specs" in layer:
                    spec_dict = layer["mx_specs"]
                elif "group" in layer:
                    group_name = layer["group"]
                    if group_name not in groups:
                        raise ValueError(f"Group '{group_name}' not defined in config 'groups'")
                    spec_dict = groups[group_name]
                else:
                    spec_dict = global_specs

            layer_map[name] = self._build_mx_specs(spec_dict)

        return layer_map

    def _build_act_map(self):
        """Parse `kind: "act_quant"` config entries.

        Returns:
            dict: layer_name -> (specs_per_input, axes_per_input)

        Entry format (one per parameter-free module to wrap):
            {"name": "warp", "kind": "act_quant",
             "inputs": [
               {"mx_specs": {...}, "axes": [1]},     # input 0
               {"group": "low_precision", "axes": [-1]},  # input 1
               null                                   # input 2: not quantized
             ]}

        Per-input spec priority mirrors layers: "mx_specs" > "group" > global.
        `axes` defaults to [1] (channel axis for NCHW activations).
        """
        if not self.config or "layers" not in self.config:
            return {}

        global_specs = self.config.get("mx_specs", None)
        groups = self.config.get("groups", {})
        act_map = {}

        for layer in self.config["layers"]:
            if not isinstance(layer, dict) or layer.get("kind") != "act_quant":
                continue
            name = layer["name"]
            inputs = layer.get("inputs")
            if not inputs:
                raise ValueError(
                    f"act_quant layer '{name}' must define a non-empty 'inputs' list")

            specs_per_input, axes_per_input = [], []
            for i, ent in enumerate(inputs):
                if ent is None:
                    specs_per_input.append(None)
                    axes_per_input.append([1])
                    continue
                if "mx_specs" in ent:
                    spec_dict = ent["mx_specs"]
                elif "group" in ent:
                    group_name = ent["group"]
                    if group_name not in groups:
                        raise ValueError(
                            f"Group '{group_name}' (act_quant '{name}' input {i}) "
                            f"not defined in config 'groups'")
                    spec_dict = groups[group_name]
                else:
                    spec_dict = global_specs
                specs_per_input.append(self._build_mx_specs(spec_dict))
                axes = ent.get("axes", [1])
                if len(axes) != 1:
                    raise ValueError(
                        f"act_quant '{name}' input {i}: exactly one quant axis "
                        f"supported, got {axes}")
                axes_per_input.append(axes)

            act_map[name] = (specs_per_input, axes_per_input)

        return act_map

    def _wrap_act_layers(self, model, verbose=1):
        """Wrap configured parameter-free modules in MXActQuant. Returns count."""
        act_map = self._build_act_map()
        if not act_map:
            return 0

        wrapped = set()
        for full_name, module in list(model.named_modules()):
            clean_name = (full_name[len("module."):]
                          if full_name.startswith("module.") else full_name)
            if clean_name not in act_map or isinstance(module, MXActQuant):
                continue
            parent, leaf = self._get_parent(model, full_name)
            if parent is None:
                continue
            specs_per_input, axes_per_input = act_map[clean_name]
            new = MXActQuant(module, specs_per_input, axes_per_input)
            new._mx_layer_name = clean_name
            setattr(parent, leaf, new)
            wrapped.add(clean_name)
            if verbose >= 2:
                print(f"[MXQuantizer] act_quant '{clean_name}' -> "
                      f"MXActQuant({new.extra_repr()})")

        missing = sorted(set(act_map) - wrapped)
        if missing and verbose >= 1:
            print(f"[MXQuantizer] WARNING: act_quant layers not found in model: "
                  f"{missing}")
        return len(wrapped)

    def _build_out_map(self):
        """Parse `kind: "out_quant"` config entries.

        Returns:
            dict: layer_name -> normalized out_quant config

        Entry format (one per module whose output leaves in fixed point):
            {"name": "model", "kind": "out_quant",
             "total_bits": 16, "frac_bits": 8, "signed": true,
             "round": "half_away", "saturate": true, "clip_grad": false}

        Unlike MX, the scale here is static (2^-frac_bits), so there is no
        mx_specs / group indirection — the format is spelled out on the entry.
        """
        if not self.config or "layers" not in self.config:
            return {}

        out_map = {}
        for layer in self.config["layers"]:
            if not isinstance(layer, dict) or layer.get("kind") != "out_quant":
                continue
            name = layer["name"]
            cfg_dict = {k: v for k, v in layer.items()
                        if k not in ("name", "kind")}
            out_map[name] = normalize_out_quant(cfg_dict)

        return out_map

    def _install_out_quant(self, model, verbose=1):
        """Attach fixed-point output quant hooks. Returns count.

        A forward hook is used rather than a wrapper module: this normally
        targets the top-level model, and wrapping that would insert an `.inner`
        level into every parameter name, breaking checkpoint load/save. A hook
        that returns a value replaces the output, keeps the autograd graph, and
        leaves module paths and state_dict keys identical.
        """
        out_map = self._build_out_map()
        if not out_map:
            return 0

        installed = set()
        for full_name, module in list(model.named_modules()):
            clean_name = (full_name[len("module."):]
                          if full_name.startswith("module.") else full_name)
            if clean_name not in out_map or hasattr(module, "_mx_out_quant"):
                continue
            cfg = out_map[clean_name]
            if not cfg.get("enabled", True):
                continue
            module._mx_out_quant = {
                "cfg": cfg, "name": clean_name,
                "n": 0, "n_clipped": 0, "sum_sq": 0.0, "sum_sq_err": 0.0,
                "n_calls": 0,
            }
            module._mx_out_quant_handle = module.register_forward_hook(
                _out_quant_hook)
            installed.add(clean_name)
            if verbose >= 2:
                print(f"[MXQuantizer] out_quant '{clean_name}' -> "
                      f"{fxp_format_str(cfg)}")

        missing = sorted(set(out_map) - installed)
        if missing and verbose >= 1:
            print(f"[MXQuantizer] WARNING: out_quant layers not found in model: "
                  f"{missing}")
        return len(installed)

    @staticmethod
    def out_quant_summary(module):
        """Format the accumulated out-quant stats for a hooked module."""
        state = getattr(module, "_mx_out_quant", None)
        if state is None:
            return None
        fmt = fxp_format_str(state["cfg"])
        if not state["n"]:
            return f"{fmt} | no data yet"

        # Counters are accumulated as device tensors to keep the forward
        # sync-free; this is the only place they are read back to the host.
        n_clipped = fxp_stats_value(state["n_clipped"])
        sum_sq = fxp_stats_value(state["sum_sq"])
        sum_sq_err = fxp_stats_value(state["sum_sq_err"])

        clip_pct = 100.0 * n_clipped / state["n"]
        if sum_sq_err > 0:
            sqnr_s = f"{10.0 * math.log10(sum_sq / sum_sq_err):.1f} dB"
        else:
            sqnr_s = "inf"
        return f"{fmt} | clip {clip_pct:.2f}% | SQNR {sqnr_s}"

    def _create_mx_module(self, orig_module, mx_specs):
        """
        Deprecated alias for `_build_mx_module`, kept for external callers.

        Note the behaviour change: this used to build plain MXConv2d /
        MXLinear only, ignoring `xblock_accum`. It now returns the same class
        `_replace_layers` would install, so measurements match deployment.
        """
        return self._build_mx_module(orig_module, mx_specs, verbose=0)

    def _get_candidate_layers(self, model):
        """
        Return the list of layer names that are candidates for quantization.

        If the config has a 'layers' key, those names are used.
        Otherwise (auto_mixed without explicit layers), discovers all nn.Conv2d and
        nn.Linear modules in the model automatically.
        """
        if "layers" in self.config:
            return [l if isinstance(l, str) else l["name"]
                    for l in self.config["layers"]
                    if isinstance(l, str)
                    or l.get("kind") not in ("act_quant", "out_quant")]
        # auto-discover all Conv2d / ConvTranspose2d / Linear (excluding already-MX
        # layers). Names are stripped of the DataParallel `module.` prefix so they
        # match what _replace_layers looks up — without this, auto-discovery on a
        # DataParallel model produced keys that matched nothing and quantized none
        # of the network.
        seen = []
        for n, m in model.named_modules(remove_duplicate=False):
            if not isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                continue
            if isinstance(m, (MXConv2d, MXConvTranspose2d, MXLinear)):
                continue
            name = _clean_name(n)
            if name not in seen:
                seen.append(name)
        return seen

    def _get_parent(self, model, full_name):
        parts = full_name.split(".")
        parent = model

        for p in parts[:-1]:
            if not hasattr(parent, p):
                return None, None
            parent = getattr(parent, p)

        return parent, parts[-1]

    # =========================
    # Auto mixed-precision
    # =========================
    def _measure_isolated_sensitivity(self, fp32_model, candidates, base_specs,
                                      data, forward_fn=None, max_batches=128, log=None):
        """
        Measure each layer's own quantization error in isolation (no upstream propagation).

        Phase A — one FP32 forward pass: capture per-layer input and output tensors.
        Phase B — per-layer: create a temporary MX copy (base_specs), feed the captured
                  FP32 inputs, compute SQNR against the FP32 outputs.

        Uses a FIFO queue to handle layers that fire multiple times per forward pass
        (e.g. shared modules in multi-scale detection models).

        Returns:
            dict: {layer_name: sqnr_db}, sorted worst-first in the log table.
        """
        fp32_inputs  = {}  # name -> [list of tensors, FIFO order]
        fp32_outputs = {}  # name -> [list of tensors, FIFO order]

        candidate_set = set(candidates)
        fp32_mods = {n: m for n, m in fp32_model.named_modules() if n in candidate_set}

        handles = []
        for name, mod in fp32_mods.items():
            def make_hook(n):
                def h(m, inp, out):
                    # Offload to CPU immediately to avoid holding all batches on GPU
                    fp32_inputs.setdefault(n, []).append(inp[0].detach().cpu())
                    fp32_outputs.setdefault(n, []).append(out.detach().cpu())
                return h
            handles.append(mod.register_forward_hook(make_hook(name)))

        def _fwd(m, batch):
            if forward_fn is not None:
                forward_fn(m, batch)
            elif isinstance(batch, (list, tuple)):
                m(batch[0])
            elif isinstance(batch, dict):
                m(**batch)
            else:
                m(batch)

        with torch.no_grad():
            for i, batch in enumerate(data):
                if max_batches > 0 and i >= max_batches:
                    break
                _fwd(fp32_model, batch)

        for h in handles:
            h.remove()

        # Phase B: compute isolated SQNR per layer.
        # Tensors are on CPU; move to device for the temp MX forward, then delete immediately.
        sensitivity = {}
        for name in candidates:
            mod = fp32_mods.get(name)
            if mod is None or name not in fp32_inputs:
                continue

            device = next(iter(mod.parameters()), torch.tensor(0.0)).device
            temp_mx = self._create_mx_module(mod, base_specs)
            temp_mx = temp_mx.to(device)

            stats = dict(signal_sq=0.0, noise_sq=0.0, n_elem=0)
            with torch.no_grad():
                for fp32_in_cpu, fp32_out_cpu in zip(fp32_inputs[name], fp32_outputs[name]):
                    fp32_in  = fp32_in_cpu.to(device)
                    fp32_out = fp32_out_cpu.float()        # stays on CPU for accumulation
                    quant_out = temp_mx(fp32_in).float().cpu()
                    stats["signal_sq"] += fp32_out.pow(2).sum().item()
                    stats["noise_sq"]  += (fp32_out - quant_out).pow(2).sum().item()
                    stats["n_elem"]    += fp32_out.numel()
                    del fp32_in, quant_out                 # free GPU tensors after each sample

            # free CPU buffers for this layer before moving to the next
            del fp32_inputs[name], fp32_outputs[name]

            if stats["n_elem"] == 0:
                continue

            if stats["signal_sq"] == 0:
                sqnr_db = float("nan")
            elif stats["noise_sq"] == 0:
                sqnr_db = float("inf")
            else:
                sqnr_db = 10 * math.log10(stats["signal_sq"] / stats["noise_sq"])

            sensitivity[name] = sqnr_db

        # Weight-only fallback for layers whose hooks never fired (e.g. 1×1 depthwise)
        weight_only_names = set()
        for name in candidates:
            if name in sensitivity:
                continue
            mod = fp32_mods.get(name)
            if mod is None or not hasattr(mod, 'weight') or mod.weight is None:
                continue
            sqnr = self._weight_only_sqnr(mod, base_specs)
            sensitivity[name] = sqnr
            weight_only_names.add(name)
        if weight_only_names:
            self._log(log, f"  ({len(weight_only_names)} layers had no activation data "
                           f"— weight-only SQNR used, marked (w))")

        # Log sorted worst-first
        sorted_sens = sorted(
            sensitivity.items(),
            key=lambda x: x[1] if not math.isnan(x[1]) else float("inf")
        )
        self._log(log, f"Isolated sensitivity (base={base_specs['w_elem_format']}, worst first):")
        self._log(log, f"  {'Layer':<45} {'SQNR (dB)':>10}")
        for n, sqnr in sorted_sens:
            if math.isnan(sqnr):
                sqnr_str = f"{'N/A':>9}"
            elif sqnr == float("inf"):
                sqnr_str = f"{'inf':>9}"
            else:
                sqnr_str = f"{sqnr:>9.1f}"
            marker = " (w)" if n in weight_only_names else ""
            self._log(log, f"  {n:<45} {sqnr_str}{marker}")

        return sensitivity

    def _auto_assign_precisions(self, sensitivity, auto_mixed_cfg, log=None):
        """
        Map per-layer SQNR scores to group names (e.g. 'int4' or 'int8').

        Strategies:
          'threshold': layers with SQNR < sqnr_threshold_db → upgrade group
          'budget':    worst upgrade_fraction of layers → upgrade group

        Returns:
            dict: {layer_name: group_name}
        """
        base    = auto_mixed_cfg["base"]
        upgrade = auto_mixed_cfg["upgrade"]
        strategy = auto_mixed_cfg.get("strategy", "threshold")

        assignments = {}

        if strategy == "threshold":
            threshold = auto_mixed_cfg["sqnr_threshold_db"]
            for name, sqnr in sensitivity.items():
                assignments[name] = upgrade if (math.isnan(sqnr) or sqnr < threshold) else base

        elif strategy == "budget":
            fraction = auto_mixed_cfg["upgrade_fraction"]
            n_upgrade = int(len(sensitivity) * fraction)
            sorted_layers = sorted(
                sensitivity.items(),
                key=lambda x: x[1] if not math.isnan(x[1]) else -float("inf")
            )
            for i, (name, _) in enumerate(sorted_layers):
                assignments[name] = upgrade if i < n_upgrade else base

        else:
            raise ValueError(
                f"Unknown auto_mixed strategy: {strategy!r}. Use 'threshold' or 'budget'."
            )

        n_base_count    = sum(1 for g in assignments.values() if g == base)
        n_upgrade_count = sum(1 for g in assignments.values() if g == upgrade)
        self._log(log, f"Auto mixed-precision: {n_base_count} layers → {base}, "
                       f"{n_upgrade_count} layers → {upgrade}")

        return assignments

    def _weight_only_sqnr(self, mod, base_specs):
        """
        Estimate quantization sensitivity from the weight tensor alone.
        Used as fallback when a layer's forward hook never fires during calibration.
        """
        if not hasattr(mod, 'weight') or mod.weight is None:
            return float('nan')
        w = mod.weight.data.float()
        w_fmt = base_specs['w_elem_format']
        rnd = base_specs.get('round_mx_output', 'nearest')
        w_q = quantize_mx_op(w, base_specs, elem_format=w_fmt, axes=[-1], round=rnd)
        signal_sq = w.pow(2).sum().item()
        noise_sq = (w - w_q).pow(2).sum().item()
        if signal_sq == 0:
            return float('nan')
        return float('inf') if noise_sq == 0 else 10 * math.log10(signal_sq / noise_sq)

    # =========================
    # PTQ — GPTQ-style reconstruction
    # =========================
    def _ptq(self, model, data, forward_fn=None, log=None, max_batches=128):
        """
        Two-phase PTQ:
          Phase 1: accumulate per-layer Hessian online via forward hooks.
          Phase 2: GPTQ-style block-wise weight reconstruction, layer by layer.

        No backward pass required.
        Memory: O(in_dim²) per layer — activations are never stored in full.

        Note: MXConvTranspose2d layers are intentionally excluded — the GPTQ
        Hessian uses conv im2col (F.unfold), which does not model the transposed
        conv. Their weights still get MX-quantized on-the-fly in forward.
        """
        self._log(log, f"PTQ | Phase 1: collecting Hessians ({max_batches} batches) ...")
        hessians = self._collect_activations(model, data, forward_fn, log, max_batches)

        mx_layers = [(n, m) for n, m in model.named_modules()
                     if isinstance(m, (MXConv2d, MXLinear))]
        total    = len(mx_layers)
        no_calib = [n for n, _ in mx_layers if n not in hessians]
        if no_calib:
            self._log(log, f"PTQ | WARNING: {len(no_calib)} layers had no calibration data "
                           f"(hooks never fired) — skipping GPTQ for these:")
            for n in no_calib:
                self._log(log, f"PTQ |   - {n}")

        calib_layers = [(n, m) for n, m in mx_layers if n in hessians]
        n_calib = len(calib_layers)
        self._log(log, f"PTQ | Phase 2: GPTQ reconstruction on {n_calib}/{total} layers ...")

        for done, (full_name, module) in enumerate(calib_layers, 1):
            H, n = hessians[full_name]
            self._gptq_layer(module, H, n)
            pct = done / n_calib * 100
            if int(pct) // 5 > int((done - 1) / n_calib * 100) // 5 or done == n_calib:
                self._log(log, f"PTQ | Phase 2: {done}/{n_calib} layers done ({pct:.0f}%)")

        # Phase 2b: direct MX rounding for layers with no calibration data
        direct_round = [(n, m) for n, m in mx_layers if n not in hessians]
        if direct_round:
            self._log(log, f"PTQ | Phase 2b: direct MX rounding for {len(direct_round)} uncalibrated layers ...")
            for name, module in direct_round:
                w_fmt = module.mx_specs.get('w_elem_format')
                if w_fmt is None:
                    continue
                W = module.weight.data.float()
                rnd = module.mx_specs.get('round_mx_output', 'nearest')
                W_q = quantize_mx_op(W, module.mx_specs, elem_format=w_fmt, axes=[-1], round=rnd)
                module.weight.data = W_q.to(module.weight.dtype)
                self._log(log, f"PTQ |   [direct-round] {name}")

    def _collect_activations(self, model, data, forward_fn=None, log=None, max_batches=128):
        """
        Register forward hooks on all MXConv2d and MXLinear layers and run
        calibration batches, accumulating the Hessian H = X^T X online.
        Activations are never stored in full — memory is O(in_dim²) per layer.

        Returns:
            dict: {layer_full_name: (H, n)} where
                  H is [in_dim, in_dim] accumulated on CPU,
                  n is the total number of samples seen.
        """
        store   = {}  # name -> [H_accum, n_accum]
        handles = []

        def make_hook(name):
            def hook(mod, inputs, output):
                x = inputs[0].detach().float()          # stay on original device
                if isinstance(mod, MXConv2d):
                    x = F.unfold(
                        x,
                        kernel_size=mod.kernel_size,
                        dilation=mod.dilation,
                        padding=mod.padding,
                        stride=mod.stride,
                    )                                    # [N, C_in*kH*kW, patches]
                    x = x.permute(0, 2, 1).reshape(-1, x.shape[1])
                else:
                    x = x.reshape(-1, x.shape[-1])      # [N*seq, in_features]

                if name not in store:
                    store[name] = [torch.zeros(x.shape[1], x.shape[1]), 0]
                # compute on GPU, accumulate on CPU (H is small: [in_dim, in_dim])
                store[name][0] += (x.T @ x).cpu()
                store[name][1] += x.shape[0]
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (MXConv2d, MXLinear)):
                handles.append(module.register_forward_hook(make_hook(name)))

        total_batches = max_batches if max_batches > 0 else len(data) if hasattr(data, '__len__') else None
        report_every = max(1, total_batches // 20) if total_batches else 10  # ~5% steps

        with torch.no_grad():
            for i, batch in enumerate(data):
                if max_batches > 0 and i >= max_batches:
                    break
                if forward_fn is not None:
                    forward_fn(model, batch)
                elif isinstance(batch, (list, tuple)):
                    model(batch[0])
                elif isinstance(batch, dict):
                    model(**batch)
                else:
                    model(batch)

                if (i + 1) % report_every == 0 or (total_batches and i + 1 == total_batches):
                    pct = f"{(i+1)/total_batches*100:.0f}%" if total_batches else f"batch {i+1}"
                    self._log(log, f"PTQ | Phase 1: {i+1} batches done ({pct})")

        for h in handles:
            h.remove()

        return {name: (vals[0], vals[1]) for name, vals in store.items()}

    @staticmethod
    def _log(log, msg):
        if log is not None:
            log.info(msg)
        else:
            print(msg)

    def _gptq_layer(self, module, H, n):
        """
        GPTQ weight reconstruction for a single MXConv2d or MXLinear layer.

        Algorithm (Frantar et al. 2022):
          1. H = H_accum / n  (normalise pre-accumulated Hessian)
          2. Cholesky decompose H^{-1}
          3. Process W in column-groups of block_size:
               - MX-quantize the block
               - propagate error to remaining columns via Cholesky

        Args:
            module: MXConv2d or MXLinear layer (modified in-place).
            H:      accumulated X^T X  [in_dim, in_dim]  on CPU.
            n:      total number of samples accumulated.
        """
        w_fmt = module.mx_specs['w_elem_format']
        if w_fmt is None:
            return  # weight quantization disabled for this layer

        block_size = module.mx_specs['block_size']

        is_conv = isinstance(module, MXConv2d)
        W = module.weight.data.float()
        orig_shape = W.shape
        if is_conv:
            W = W.view(W.shape[0], -1)   # [C_out, C_in*kH*kW]

        rows, cols = W.shape
        device = W.device

        if block_size <= 0:
            block_size = cols

        # ---- Normalise Hessian ----
        H = H.to(device) / n
        damp = self._GPTQ_DAMPING * H.diagonal().mean()
        H.diagonal().add_(damp)

        # ---- Phase 2: Cholesky of H^{-1} ----
        try:
            H_inv = torch.linalg.inv(H)
            H_inv = (H_inv + H_inv.T) / 2          # enforce symmetry
            H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)
        except torch.linalg.LinAlgError:
            # Numerically degenerate — skip this layer
            return

        # ---- Phase 3: block-wise reconstruction ----
        W = W.clone()
        rnd = module.mx_specs['round_mx_output']

        for q in range(0, cols, block_size):
            b = min(block_size, cols - q)

            w_blk = W[:, q:q+b]                    # [rows, b]

            w_blk_q = quantize_mx_op(
                w_blk,
                module.mx_specs,
                elem_format=w_fmt,
                axes=[-1],
                round=rnd,
            )

            err = w_blk - w_blk_q                  # [rows, b]

            # Scale error by Cholesky: solve  chol_blk @ E^T = err^T
            chol_blk = H_inv_chol[q:q+b, q:q+b]   # [b, b] upper triangular
            E = torch.linalg.solve_triangular(
                chol_blk, err.T, upper=True
            ).T                                     # [rows, b]

            if q + b < cols:
                W[:, q+b:] -= E @ H_inv_chol[q:q+b, q+b:]

            W[:, q:q+b] = w_blk_q

        module.weight.data = W.view(orig_shape).to(module.weight.dtype)

    # =========================
    # Error measurement
    # =========================
    def _measure_error(self, fp32_model, quant_model, data,
                       forward_fn=None, log=None, max_batches=128):
        """
        Compare per-layer outputs of fp32_model vs quant_model on calibration data.

        Registers output hooks on matching layers in both models, runs each batch
        through FP32 then quant, and accumulates statistics online (no full tensor
        storage — memory is O(1) per layer).

        Returns:
            dict: {layer_name: {"mse": float, "cos_sim": float, "sqnr_db": float}}
                  sorted worst-first by SQNR.  Also attached to quant_model as
                  quant_model._quant_errors by the caller.
        """
        quant_names = {n for n, m in quant_model.named_modules()
                       if isinstance(m, (MXConv2d, MXConvTranspose2d, MXLinear))}
        fp32_layer_map = {n: m for n, m in fp32_model.named_modules()
                          if n in quant_names}

        if not fp32_layer_map:
            self._log(log, "measure_error | WARNING: no matching layers found, skipping.")
            return {}

        accum = {name: dict(signal_sq=0.0, noise_sq=0.0, dot=0.0,
                            quant_sq=0.0, n_elem=0)
                 for name in quant_names}
        # Use a FIFO queue per layer to handle layers called multiple times
        # per forward pass (e.g. multi-scale / shared modules in detection models).
        fp32_store = {}   # name -> list of outputs (in call order)
        handles = []

        def make_fp32_hook(name):
            def hook(mod, inp, out):
                fp32_store.setdefault(name, []).append(out.detach().float())
            return hook

        def make_quant_hook(name):
            def hook(mod, inp, out):
                queue = fp32_store.get(name)
                if not queue:
                    return
                fp32_out = queue.pop(0)   # consume in call order
                q_out = out.detach().float()
                s = accum[name]
                s["signal_sq"] += fp32_out.pow(2).sum().item()
                s["noise_sq"]  += (fp32_out - q_out).pow(2).sum().item()
                s["dot"]       += (fp32_out * q_out).sum().item()
                s["quant_sq"]  += q_out.pow(2).sum().item()
                s["n_elem"]    += fp32_out.numel()
            return hook

        for name, module in fp32_layer_map.items():
            handles.append(module.register_forward_hook(make_fp32_hook(name)))
        for name, module in quant_model.named_modules():
            if name in quant_names:
                handles.append(module.register_forward_hook(make_quant_hook(name)))

        def _fwd(m, batch):
            if forward_fn is not None:
                forward_fn(m, batch)
            elif isinstance(batch, (list, tuple)):
                m(batch[0])
            elif isinstance(batch, dict):
                m(**batch)
            else:
                m(batch)

        with torch.no_grad():
            for i, batch in enumerate(data):
                if max_batches > 0 and i >= max_batches:
                    break
                _fwd(fp32_model, batch)
                _fwd(quant_model, batch)
                fp32_store.clear()

        for h in handles:
            h.remove()

        # Compute final metrics
        results = {}
        for name, s in accum.items():
            if s["n_elem"] == 0:
                continue
            mse = s["noise_sq"] / s["n_elem"]
            zero_output = s["signal_sq"] == 0 and s["quant_sq"] == 0
            denom = (s["signal_sq"] * s["quant_sq"]) ** 0.5
            if zero_output:
                cos_sim, sqnr_db = float("nan"), float("nan")
            else:
                cos_sim = max(-1.0, min(1.0, s["dot"] / denom)) if denom > 0 else 0.0
                sqnr_db = (10 * math.log10(s["signal_sq"] / s["noise_sq"])
                           if s["noise_sq"] > 0 else float("inf"))
            results[name] = {"mse": mse, "cos_sim": cos_sim, "sqnr_db": sqnr_db,
                             "zero_output": zero_output}

        # Sort worst first by SQNR (nan / zero-output layers go last)
        results = dict(sorted(
            results.items(),
            key=lambda x: x[1]["sqnr_db"] if not math.isnan(x[1]["sqnr_db"]) else float("inf")
        ))

        # Log table
        self._log(log, "Quantization error per layer (worst first):")
        self._log(log, f"  {'Layer':<45} {'MSE':>10} {'Cos Sim':>10} {'SQNR (dB)':>10}")
        for name, m in results.items():
            if m["zero_output"]:
                self._log(log, f"  {name:<45} {'0.00e+00':>10} {'N/A':>10} {'N/A (zero output)':>18}")
                continue
            sqnr_str = f"{m['sqnr_db']:>9.1f}" if m["sqnr_db"] != float("inf") else "      inf"
            self._log(log, f"  {name:<45} {m['mse']:>10.3e} {m['cos_sim']:>10.4f} {sqnr_str}")
        finite = [m["sqnr_db"] for m in results.values()
                  if not m["zero_output"] and m["sqnr_db"] != float("inf")]
        if finite:
            self._log(log, f"  Overall (mean SQNR): {sum(finite)/len(finite):.1f} dB")

        return results

    # =========================
    # Debug printing
    # =========================
    def _print_stat(self, model, log=None):
        """
        Prints replaced and missed Conv2d / ConvTranspose2d / Linear layers.
        """
        num_mx_conv, num_mx_linear, num_mx_convT = 0, 0, 0
        num_fp_conv, num_fp_linear, num_fp_convT = 0, 0, 0

        num_mx_act, num_out_quant = 0, 0

        for name, module in model.named_modules():
            if hasattr(module, "_mx_out_quant"):
                self._log(log, f"[OutQuant] {name or '<root>'}: "
                               f"{self.out_quant_summary(module)}")
                num_out_quant += 1

            if isinstance(module, MXActQuant):
                self._log(log, f"[ActQuant->MX] {name}: {module.extra_repr()}")
                num_mx_act += 1
            elif isinstance(module, MXConvTranspose2d):
                self._log(log, f"[ConvTranspose2d->MX] {name}: {module}")
                num_mx_convT += 1
            elif isinstance(module, MXConv2d):
                self._log(log, f"[Conv2d->MX] {name}: {module}")
                num_mx_conv += 1
            elif isinstance(module, MXLinear):
                self._log(log, f"[Linear->MX] {name}: {module}")
                num_mx_linear += 1
            elif isinstance(module, nn.ConvTranspose2d):
                self._log(log, f"[MISSED] {name}: still nn.ConvTranspose2d!")
                num_fp_convT += 1
            elif isinstance(module, nn.Conv2d):
                self._log(log, f"[MISSED] {name}: still nn.Conv2d!")
                num_fp_conv += 1
            elif isinstance(module, nn.Linear):
                self._log(log, f"[MISSED] {name}: still nn.Linear!")
                num_fp_linear += 1

        self._log(log, f"MX convs: {num_mx_conv}, regular convs: {num_fp_conv}, "
                       f"MX convTs: {num_mx_convT}, regular convTs: {num_fp_convT}, "
                       f"MX linears: {num_mx_linear}, regular linears: {num_fp_linear}, "
                       f"MX act-quant wrappers: {num_mx_act}, "
                       f"fxp out-quant hooks: {num_out_quant}.")
