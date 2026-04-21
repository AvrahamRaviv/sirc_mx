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
from microxcaling.mx.linear import Linear as MXLinear
from microxcaling.mx.mx_ops import quantize_mx_op
from microxcaling.mx import MxSpecs

from mx_fixed_point import (
    fixed_point_accumulate,
    validate_xblock_accum_bits,
    _DEFAULT_BITS as _XBLOCK_ACCUM_DEFAULT_BITS,
)
from mx_layers_blocked import MXConv2dBlocked, MXLinearBlocked


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

    Priority (highest to lowest): per-layer mx_specs > group > global mx_specs > defaults
    Note: scale_bits is shared between weights and activations (library limitation).
    """

    # GPTQ damping factor: fraction of mean diagonal to add for numerical stability
    _GPTQ_DAMPING = 0.01

    def __init__(self, save_dir, log=None):
        self.save_dir = save_dir
        self.config_path = os.path.join(save_dir, "mx_config.json")
        self.config = self._load_config()
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

        if auto_mixed:
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

        self._print_stat(model, log)
        return model

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

        # Extract xblock_accum_* keys — stored as attributes (NOT dict items) so
        # microxcaling.apply_mx_specs does not reject them as unknown keys.
        xblock_cfg = {
            'xblock_accum_mode': 'fp32',
            'xblock_accum_bits': _XBLOCK_ACCUM_DEFAULT_BITS,
            'xblock_accum_saturate': True,
            'xblock_accum_ste_mask': False,
            'xblock_accum_backend': 'python',
        }
        if spec_dict is not None:
            spec_dict = dict(spec_dict)
            for k in list(xblock_cfg.keys()):
                if k in spec_dict:
                    xblock_cfg[k] = spec_dict.pop(k)
            for k, v in spec_dict.items():
                mx_specs[k] = v

        if xblock_cfg['xblock_accum_mode'] not in ('fp32', 'fixed_point'):
            raise ValueError(
                f"xblock_accum_mode must be 'fp32' or 'fixed_point', "
                f"got {xblock_cfg['xblock_accum_mode']!r}"
            )
        if xblock_cfg['xblock_accum_mode'] == 'fixed_point':
            validate_xblock_accum_bits(xblock_cfg['xblock_accum_bits'])

        if xblock_cfg['xblock_accum_backend'] not in ('python', 'triton'):
            raise ValueError(
                f"xblock_accum_backend must be 'python' or 'triton', "
                f"got {xblock_cfg['xblock_accum_backend']!r}"
            )

        for k, v in xblock_cfg.items():
            setattr(mx_specs, k, v)

        return mx_specs

    # =========================
    # Replacement logic
    # =========================
    def _replace_layers(self, model, layer_map=None):
        """
        Replace Conv2d and Linear layers based on config (or a pre-built layer_map).
        """
        if layer_map is None:
            layer_map = self._build_layer_map()

        for full_name, module in model.named_modules():
            is_conv = isinstance(module, nn.Conv2d) and not isinstance(module, MXConv2d)
            is_linear = isinstance(module, nn.Linear) and not isinstance(module, MXLinear)
            if not (is_conv or is_linear):
                continue

            clean_name = full_name[len("module."):] if full_name.startswith("module.") else full_name
            if clean_name not in layer_map:
                continue

            mx_specs = layer_map[clean_name]

            parent, leaf = self._get_parent(model, full_name)
            if parent is None:
                continue

            if is_conv:
                use_blocked = (
                    getattr(mx_specs, 'xblock_accum_mode', 'fp32') == 'fixed_point'
                    and module.groups == 1
                )
                conv_cls = MXConv2dBlocked if use_blocked else MXConv2d
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
                linear_cls = (
                    MXLinearBlocked
                    if getattr(mx_specs, 'xblock_accum_mode', 'fp32') == 'fixed_point'
                    else MXLinear
                )
                new = linear_cls(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    mx_specs=mx_specs
                )

            # preserve weights/bias
            new.weight = module.weight
            new.bias = module.bias

            # Propagate xblock_accum_* attrs onto the new layer; microxcaling
            # re-builds its internal mx_specs and drops python attrs.
            for k in ('xblock_accum_mode', 'xblock_accum_bits',
                      'xblock_accum_saturate', 'xblock_accum_ste_mask',
                      'xblock_accum_backend'):
                if hasattr(mx_specs, k):
                    setattr(new, k, getattr(mx_specs, k))

            setattr(parent, leaf, new)

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

    def _create_mx_module(self, orig_module, mx_specs):
        """
        Build an MXConv2d or MXLinear from an existing nn.Conv2d / nn.Linear,
        sharing the original weight and bias tensors.
        Used for temporary isolated-sensitivity measurement and for _replace_layers.
        """
        if isinstance(orig_module, nn.Conv2d):
            new = MXConv2d(
                orig_module.in_channels,
                orig_module.out_channels,
                orig_module.kernel_size,
                stride=orig_module.stride,
                padding=orig_module.padding,
                dilation=orig_module.dilation,
                groups=orig_module.groups,
                bias=orig_module.bias is not None,
                mx_specs=mx_specs,
            )
        else:
            new = MXLinear(
                orig_module.in_features,
                orig_module.out_features,
                bias=orig_module.bias is not None,
                mx_specs=mx_specs,
            )
        new.weight = orig_module.weight
        new.bias = orig_module.bias
        return new

    def _get_candidate_layers(self, model):
        """
        Return the list of layer names that are candidates for quantization.

        If the config has a 'layers' key, those names are used.
        Otherwise (auto_mixed without explicit layers), discovers all nn.Conv2d and
        nn.Linear modules in the model automatically.
        """
        if "layers" in self.config:
            return [l if isinstance(l, str) else l["name"]
                    for l in self.config["layers"]]
        # auto-discover all Conv2d / Linear (excluding already-MX layers)
        return [n for n, m in model.named_modules()
                if isinstance(m, (nn.Conv2d, nn.Linear))
                and not isinstance(m, (MXConv2d, MXLinear))]

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
                       if isinstance(m, (MXConv2d, MXLinear))}
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
        Prints replaced and missed Conv2d / Linear layers.
        """
        num_mx_conv, num_mx_linear = 0, 0
        num_fp_conv, num_fp_linear = 0, 0

        for name, module in model.named_modules():
            if isinstance(module, MXConv2d):
                self._log(log, f"[Conv2d->MX] {name}: {module}")
                num_mx_conv += 1
            elif isinstance(module, MXLinear):
                self._log(log, f"[Linear->MX] {name}: {module}")
                num_mx_linear += 1
            elif isinstance(module, nn.Conv2d):
                self._log(log, f"[MISSED] {name}: still nn.Conv2d!")
                num_fp_conv += 1
            elif isinstance(module, nn.Linear):
                self._log(log, f"[MISSED] {name}: still nn.Linear!")
                num_fp_linear += 1

        self._log(log, f"MX convs: {num_mx_conv}, regular convs: {num_fp_conv}, "
                       f"MX linears: {num_mx_linear}, regular linears: {num_fp_linear}.")
