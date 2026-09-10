# MX Quantizer

A PyTorch quantization wrapper that replaces selected `Conv2d` and `Linear` layers with **MX-quantized** equivalents (`MXConv2d` / `MXLinear`). Supports config-driven quantization, optional PTQ, automated mixed-precision selection, and per-layer error measurement.

> **For now, we are using format: `int8` (MXINT8)**



---

## Features

- Replace `Conv2d` and `Linear` layers based on a **config file**
- Three config modes: global specs, per-layer specs, named groups
- **Automated mixed-precision** (`auto_mixed`): per-layer sensitivity analysis, threshold or budget strategy
- **PTQ** (GPTQ-style weight reconstruction, implemented but **ineffective** — see PTQ section)
- **Per-layer error measurement**: MSE, cosine similarity, SQNR per layer vs FP32 baseline
- Debug printing of replaced / missed layers

---

## Installation

Ensure `microxcaling` is installed (local repo or package):

```bash
pip install microxcaling
```

---

## Usage

```python
from mx_quantizer import MXQuantizer

model = ...           # PyTorch model
cal_data = [...]      # iterable of calibration batches

quantizer = MXQuantizer(save_dir="/path/to/config_dir")

# Replace layers only (no calibration)
quantized_model = quantizer.quant(model)

# With calibration data — enables PTQ, sensitivity measurement, and error measurement
quantized_model = quantizer.quant(
    model,
    data=cal_data,
    forward_fn=lambda m, batch: m(batch[0], batch[2]),  # optional, required for complex forward passes
)
```

### `forward_fn`

If your model's `forward()` requires more than a single tensor (e.g. anchors, memory, image sizes), provide `forward_fn`. It is used consistently across all phases (sensitivity measurement, PTQ, error measurement). ODT example:

```python
def my_forward(model, batch):
    images, anchors, memory = batch
    return model(images, anchors, memory=memory)

quantized_model = quantizer.quant(model, data=cal_data, forward_fn=my_forward)
```

Without `forward_fn`, the default fallback is `model(batch[0])` for list/tuple batches, `model(**batch)` for dicts, and `model(batch)` otherwise.

---

## Config (`mx_config.json`)

### Mode 1 — Global MX specs + layer list

All listed layers share the same quantization spec.

```json
{
  "mx_specs": {
    "w_elem_format": "int8",
    "a_elem_format": "int8",
    "block_size": 32,
    "scale_bits": 8,
    "shared_exp_method": "max",
    "custom_cuda": true
  },
  "layers": [
    {"name": "backbone.conv1"},
    {"name": "head.fc"}
  ]
}
```

### Mode 2 — Per-layer MX specs

Each layer can carry its own spec dict. Useful for one-off overrides.

```json
{
  "layers": [
    {
      "name": "backbone.conv1",
      "mx_specs": {"w_elem_format": "int8", "a_elem_format": "int8", "block_size": 32, "custom_cuda": true}
    },
    {
      "name": "head.fc",
      "mx_specs": {"w_elem_format": "int4", "a_elem_format": "int4", "block_size": 32, "custom_cuda": true}
    }
  ]
}
```

### Mode 3 — Named groups (mixed-precision)

Define reusable precision groups and assign each layer to a group. This is the preferred format for mixed-precision configurations.

```json
{
  "groups": {
    "int8": {
      "w_elem_format": "int8",
      "a_elem_format": "int8",
      "block_size": 32,
      "scale_bits": 8,
      "shared_exp_method": "max",
      "custom_cuda": true
    },
    "int4": {
      "w_elem_format": "int4",
      "a_elem_format": "int4",
      "block_size": 32,
      "scale_bits": 8,
      "shared_exp_method": "max",
      "custom_cuda": true
    }
  },
  "layers": [
    {"name": "backbone.conv1",  "group": "int8"},
    {"name": "backbone.conv2",  "group": "int4"},
    {"name": "head.fc",         "group": "int8"}
  ]
}
```

---

## PTQ

PTQ runs GPTQ-style block-wise weight reconstruction (forward-only, no gradients). It requires calibration `data`.

```json
{
  "ptq": {"enabled": true, "batches": 128}
}
```

Shorthand to disable: `"ptq": false`.  
Key absent: PTQ runs automatically if `data` is provided (128 batches default).

**Limitation — PTQ has no effect with the Microsoft MX library:**
`MXConv2d` / `MXLinear` store weights in FP32 and re-quantize them on every forward pass at runtime. GPTQ writes corrected weights back to `module.weight.data`, but the MX layer immediately re-quantizes those weights again on the next forward call — discarding the correction. PTQ is therefore a no-op in the current setup and is kept only for potential future use if the underlying library changes. For layers whose forward hooks never fire during calibration, a direct MX round-to-nearest pass is applied as a fallback (same limitation applies).

---

## Automated Mixed-Precision (`auto_mixed`)

Scores every candidate layer, then assigns each one a precision from a ladder —
e.g. the least sensitive 10% to MXINT4, the next 20% to MXINT6, the rest to
MXINT8. Training-free: no gradients, no labels, no fine-tuning.

### Two-step workflow

Scoring costs real forward passes and the result decides what a training run
does, so it is worth reviewing before depending on it:

```python
quantizer = MXQuantizer(save_dir="/path/to/config_dir")
plan = quantizer.plan_mixed_precision(model, data=cal_data, forward_fn=my_forward)
```

writes two files into `save_dir`:

| File | What it is |
|---|---|
| `sensitivity.json` | every score, why, and what it earned — sorted worst first |
| `mx_config_resolved.json` | a plain `groups` + `layers` config, no `auto_mixed` |

The resolved config is what training points at. Re-running it is a deterministic
replay: no scoring, nothing left to re-derive, and you can hand-edit any
assignment you disagree with.

`quant()` also runs the whole thing in one call when `auto_mixed` has a `ladder`.

### plan() then quant()

`quant()` reads the same `auto_mixed` block, so calling both would score twice.
A plan already computed on that quantizer is reused:

```python
quantizer = MXQuantizer(save_dir=save_root, log=log)
quantizer.plan_mixed_precision(model, data=train_loader, forward_fn=fwd)
model = quantizer.quant(model, data=None, log=log)      # installs that plan
```

Set `auto_mixed.replan` to force a fresh scoring run. Skipping `plan()` entirely
and calling `quant(model, data=train_loader, forward_fn=fwd)` also works — it
plans inline — but then nothing is reviewed before training starts. The plan on
disk is the point: `mx_config_resolved.json` has no `auto_mixed` key, so a
quantizer pointed at it is a deterministic replay with nothing left to re-derive.

### Config

```json
{
  "mx_specs": {
    "block_size": 32, "scale_bits": 8, "shared_exp_method": "max",
    "custom_cuda": true,
    "xblock_accum": {"enabled": true, "mode": "hw_fixed_point", "bits": 48}
  },
  "groups": {
    "int4": {"w_elem_format": "int4", "a_elem_format": "int4"},
    "int6": {"w_elem_format": "int6", "a_elem_format": "int6"},
    "int8": {"w_elem_format": "int8", "a_elem_format": "int8"}
  },
  "auto_mixed": {
    "ladder": ["int4", "int6", "int8"],
    "scorer": "oat_output",
    "probe": "bottom",
    "reference": "both",
    "batches": 8,
    "strategy": "quantile",
    "quantile": {"int4": 0.10, "int6": 0.20, "int8": 0.70},
    "separable_wa": {"enabled": true, "refine_top": 16},
    "pins": {"model.head.cls": "int8"}
  }
}
```

`ladder` is ordered **lowest precision first**. Rungs set only the number format:
block geometry and `xblock_accum` come from `mx_specs` (or `deploy_group`), so
demoting a layer changes its precision and nothing else about how it runs — a
probe measures the arithmetic that will actually be deployed, HW accumulator
included.

### Which layers the ladder covers

No `layers` key: every `Conv2d` / `ConvTranspose2d` / `Linear` in the model is a
candidate. With a `layers` list, only those names are — that is how a layer is
kept in FP32 entirely. (`"layers": []` means "quantize nothing" and is an error
under `auto_mixed`.)

A layer entry that already names a `group` — or carries its own `mx_specs` — is
a decision made by hand, so the ladder does not overrule it: it becomes a **pin**
and keeps that spec **raw**, without inheriting the deployment `mx_specs`. That
is the escape hatch for layers the ladder cannot describe. The usual case is a
`ConvTranspose2d` on the NPE path: it has no HW/blocked variant, so it is pinned
to a plain-MX group while the ladder rungs keep the accumulator model.

```json
"groups": {
  "int4": {"w_elem_format": "int4", "a_elem_format": "int4"},
  "convT_plain": {"w_elem_format": "int8", "a_elem_format": "int8",
                  "block_size": 32, "scale_bits": 8,
                  "shared_exp_method": "max", "custom_cuda": true}
},
"layers": [
  "model.block2.conv0.0.0",
  {"name": "model.block2.convtranspose_2", "group": "convT_plain"}
]
```

`auto_mixed.pins` does the same thing and wins over the `layers` entry for the
same layer. Pinned layers are still scored and still appear in the table — you
see what the pin cost — they are just not re-assigned, and they sit at their
pinned format in the reference network too, since that is the network that will
exist. `act_quant` / `out_quant` entries are never candidates and are carried
into the resolved config untouched.

`configs/mx_config_dof_npe_auto_ladder.json` is a complete worked example: the
DOF net on the NPE Triton path, 30 convs on an int4/int6/int8 ladder and 5
transpose convs pinned to plain MX.

### Scorers

| `scorer` | Cost | What it measures |
|---|---|---|
| `oat_output` *(default)* | `(1+N)×batches` forwards | demote one layer, measure the **network output**. Sees error propagation. |
| `isolated_sqnr` | 1 pass + N replays | each layer on its own captured inputs. Cheap, ignores propagation. |
| `weight_only` | ~0 | weight round-trip SQNR only. |
| `from_stats` | **0** | reuses the out-SQNR `collect_stats` already wrote to `quant_stats.json`. |
| `from_file` | 0 | any external JSON of per-layer scores. |
| `callable` | — | `"package.module:function"` — plug in any algorithm. |

Sensitivity is normalized so **higher always means "needs more bits"**, whatever
the scorer. New algorithms register with `@mx_sensitivity.register("name")` and
need not touch `MXQuantizer`.

### References

`reference` picks what the degradation is measured against:

- `marginal` — the whole net at the top rung, one layer demoted. What it costs
  to demote this layer *in the net you ship*. This drives the assignment.
- `absolute` — plain FP32, one layer quantized. How much the layer dislikes low
  precision at all.
- `both` *(default)* — reports both plus a rank delta and their Spearman
  correlation. Where they disagree, a layer is only sensitive because of what
  surrounds it.

### Strategies

| Strategy | Keys | Behaviour |
|---|---|---|
| `quantile` | `quantile: {group: fraction}` | fractions per rung, apportioned by largest remainder so the counts sum exactly |
| `threshold` | `threshold: {group: max_sensitivity}` | absolute cutoffs, stable across models |
| `cost_budget` | `cost_budget: {target_avg_bits}` | greedy by bits-saved-per-dB-lost; needs `probe: "all_rungs"` |

Layers that could not be measured (`unreached`, `zero_output`, `no_change`) go to
the top rung and are **excluded from the denominator** — they are not
insensitive layers, they are layers we failed to measure, and counting them
would quietly shift every requested fraction. Ties are broken by name and
reported in the artifact.

### Weight/activation split

Two layers can post the same output SQNR while one is weight-underflow driven
and the other activation-underflow driven; spending bits on the wrong operand
buys nothing. With `separable_wa.enabled`, the most sensitive `refine_top`
layers get solo weight-only and activation-only probes, and when the two
disagree by more than `margin_db` (default 3 dB) the quiet operand drops one
rung. That produces a merged group named for what it is — `w8a4` — written into
the resolved config where you can overrule it.

Only the format keys split. Block geometry is shared by both operands and the
accumulator model belongs to the deployment spec, so both are taken whole; a
rung that tries to carry `xblock_accum` is rejected at validation, because it
would silently change the layer class.

### Guards

Everything checkable is checked before the first forward pass: group names,
element formats the library actually implements (a trial quantization, not just
a name lookup), fractions that sum to 1, shared keys that agree across rungs,
and a non-empty candidate list. During scoring:

- the forward pass must be **reproducible** — the reference runs twice and the
  SQNR between the two runs is the **noise floor**. Bit-equality is not the
  test: cuDNN autotunes its convolution algorithm and reduces in
  nondeterministic order, so a real GPU model is never bit-identical twice
  while still reproducing to ~120 dB. Below `min_noise_floor_db` (default 60)
  scoring aborts, and layers landing within 10 dB of the floor are reported as
  unranked — their order is noise;
- a **half-split rank correlation** over the calibration batches is printed —
  below ~0.9 means `batches` is too low. It costs no extra forwards;
- an **ETA** is printed before probing starts;
- after assignment, one extra pass measures the **finished mix** against the
  reference — the number no per-layer score can give you.

### Legacy two-rung config

The older `base` / `upgrade` form still works unchanged and follows the original
code path exactly:

```json
"auto_mixed": {"base": "int4", "upgrade": "int8",
               "strategy": "budget", "upgrade_fraction": 0.8, "batches": 32}
```

| Strategy | Key | Behaviour |
|----------|-----|-----------|
| `threshold` | `sqnr_threshold_db` | Layers with SQNR below threshold → `upgrade` group |
| `budget` | `upgrade_fraction` | Worst N% of layers by SQNR → `upgrade` group |

Layers whose hooks never fire receive a weight-only SQNR estimate and are marked
`(w)` in the sensitivity log.

---

## Error Measurement

Enabled by default when `data` is provided. Measures per-layer output error (MSE, cosine similarity, SQNR) between FP32 and quantized model. Results are attached to the returned model as `model._quant_errors`.

```json
{
  "measure_error": true
}
```

Set to `false` to disable.

---

## Notes

- `scale_bits` is shared between weights and activations (library constraint).
- Spec priority (highest to lowest): per-layer `mx_specs` > group > global `mx_specs`.
