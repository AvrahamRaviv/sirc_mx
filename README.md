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

### Reusing a score set (`reuse`)

Sensitivity is a property of the **weights, the candidate layer set, the probe
rung and the reference rung** — and of nothing else. Ladder fractions,
thresholds, budgets, pins and strategy are applied afterwards, in `assign()`,
and cost nothing. So tuning a ladder does not need a second scoring run.

`plan_mixed_precision` therefore replays `save_dir/sensitivity.json` when it
finds one, and re-scores when it does not:

```json
"auto_mixed": {"reuse": "auto"}
```

| value | behaviour |
|---|---|
| `"auto"` *(default)* | replay when every fingerprint matches, otherwise re-score and overwrite |
| `"force"` | replay whatever is on disk, stale or not |
| `false` | always re-score |

Nothing about this is silent. Every run logs which half happened, and a re-score
logs why:

```
auto_mixed | REUSED sensitivity.json (30/35 layers measured, scorer=oat_output,
             probed at int4, scored 2026-09-16T11:04:22) — no scoring passes this run
auto_mixed | re-scoring, cached sensitivity is stale: model weights changed
```

**The guard is the feature.** A stale score set does not crash — it produces a
confident, plausible, wrong precision assignment that nobody questions again. So
`meta.fingerprint` records a hash of the parameter *values* of every candidate
layer (a fine-tune keeps every shape identical), the candidate name set, and the
resolved probe and reference specs (adding `block_size: 16` to the int4 rung
leaves its *name* `int4` and changes the arithmetic every probe ran under). Any
mismatch re-scores. The calibration input shape is checked too, when both sides
know it — cached `cost.macs` comes from shape hooks on a real forward.

Two further refusals, both because a replay cannot invent data it never measured:
`strategy: "cost_budget"` against an artifact probed at `"bottom"` (a rank is not
a price — re-score with `probe: "all_rungs"`), and `separable_wa` against an
artifact with no w/a columns, which logs a warning and falls back to joint
assignment rather than splitting blind.

The `verify` pass still runs on a replay. It is two forward passes, not N, and it
is the only thing that can catch a cache which passed every fingerprint and is
still wrong.

`reuse` and `replan` are different caches and compose: `replan` controls whether
`quant()` re-calls `plan_mixed_precision` at all, `reuse` controls whether that
call probes or replays.

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
    "reuse": "auto",
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
`ConvTranspose2d` on the NPE path kept on plain MX (the default), pinned to a
plain-MX group while the ladder rungs keep the accumulator model.

`xblock_accum.convT: "hw"` (default `"plain"`) puts a `ConvTranspose2d` with
`groups == 1` on the HW path instead: `MXConvTranspose2dHW`
(`mx_layers_convT.py`), a stride-1 conv over the zero-inserted input with the
weight transposed and flipped. The input is MX-quantized on the real tensor by
the normal `MXConv2dHW` path (any act blockify), the kernel inserts the zeros
after extracting the ints and exponents, and the window, `hw_grids`,
calibration and stats are the `MXConv2dHW` ones under the layer's name. The
weight parameter stays in convT layout, so state_dict keys and shapes do not
change. Grouped (depthwise) convT stays on plain MX; GPTQ skips the layer.
`configs/mx_config_dof_run_win_cal_convT.json` is the DOF run config with the
three dense `conv1` layers on this path.

`xblock_accum.pla: "infer"` (default `"off"`, needs `out_window` on) folds the
`PReLU` / `LeakyReLU` / `ReLU` that directly follows an HW conv (or HW convT) in
an `nn.Sequential` into the PCU multiply in front of the output window: per
output channel two signed `pla_mult_bits` (8) multipliers, `m_pos = 2^frac`
and `m_neg = floor(alpha * 2^frac + 0.5)`, are picked by the sign of the
accumulator word (with the bias), and the window extracts from
`N * m` (48 + 8 = 56 bits) on the grid `2^(e_layer_min - frac)`. `frac`
(`pla_frac_bits`, null = largest that fits: 6 for 8b) and the slopes come from
the module offline; `e_qa_out` is calibrated on the activated output by the
same `calibrate_hw_grids`. The activation module stays in the model (state_dict
unchanged) and passes a PLA output through; train mode runs it as before.
`pla_bias: "bins"` (the default with PLA on) is the HW form: per channel the
bias-free accumulator is split into bins by boundaries, each bin with its own
int8 scale `S_k` and bias `B_k`, `P = S_k * N + B_k`. PReLU is two bins:
boundary `-floor(b * 2^-e_layer_min + 0.5)`, `S = m_neg | m_pos`,
`B_k = floor(S_k * b * 2^-e_layer_min + 0.5)` on the grid of `P`, offline. A word
equal to a boundary goes to the bin above (`pla_bin_tie: "upper"`, or
`"lower"`). It picks the same segment as `"pre"` and differs from it only by the
bias rounding (`S_k * b_acc` vs `q(S_k * b)`, at most `S_k / 2 + 1/2` on that
grid). `pla_bias: "pre"` adds the bias into the accumulator before one
multiply. `pla_bias: "post"` is the emulator's single-bin `pcu_pla` order (bias
after the multiply, segment by the bias-free word), kept for the HW comparison
and wrong for PReLU. The grids file carries `pla_frac`, `pla_m_pos`,
`pla_m_neg` per layer, plus `pla_bin_bounds` / `pla_bin_scales` /
`pla_bin_biases` (per channel lists) with `"bins"`, and the sim/inference use
them as given; a grids file without them still loads.
`configs/mx_config_dof_run_win_cal_pla.json` is the convT run config with PLA on.

`xblock_accum.train_path: "fast"` (default `"hw"`) trains the HW layers
(`MXConv2dHW`, `MXConvTranspose2dHW`) without the bit-exact kernel, a
per-product int64 loop with no tensor cores (a DOF-shaped training step: 85x
the float one on CPU). In train mode only, the same MX operands (blockify,
block exponents, `in_exp` clamp, int input, convT zero-insert) go through
`F.conv2d` in fp32 with autocast off (under AMP a bf16 sum would round to 8
bits; on GPU cuDNN still uses the tensor cores through TF32, exact for MXINT8
to MXINT12 operands, not MXINT16), and what the kernel does after its sum is
applied as fake-quant: the bias on the accumulator grid, the `bits`
accumulator range, and with `out_window: "always"` the window itself
(`_apply_out_window`: same grid, rounding and saturation word). The backward
is the kernel's STE, `ste_mask` included, bit for bit on an fp32 model while
no partial sum saturates (the kernel masks a lane that saturated at any
product or at the bias add, fast only on the final sum; on a model cast to
bf16 the input and weight gradients differ by bf16 rounding, about 1 bf16
ulp: fast computes them in fp32 and rounds once, the kernel's STE runs them
in bf16; bias gradients stay equal). Eval (`model.eval()`) and the sim
always run the kernel, whatever `train_path` says.

It equals the kernel while every product sits on the accumulator grid and the
partial sums fit fp32's 24 bits (measured on CPU; on GPU cuDNN may pick an FFT
or Winograd algorithm, under `cudnn.benchmark` in particular, which does not
sum exactly, so this is unproven there). A wider sum (a wide-range input)
rounds. Measured on CPU, one seed, on a 256-channel 3x3 layer whose input has
its top 2 rows 2^-20 below the rest, at 16x20 to 128x224: with the window, up
to 2 output LSB (2 only at 16x20 with `out_round: "nearest"`, 1 elsewhere) on
0.7-11.4% of outputs; without it, up to 5 fp32 ulps of the layer's largest
output, on about half of them. These are measurements, not a bound. It never
wraps, checks the range on the final sum only, and
keeps a product below the grid (an input finer than the calibrated one, or
`e_layer_min` pinned high) that the kernel floors: there the two differ by up
to one accumulator step per such product. The step reads nothing back from the
device, except with `in_exp_mode: "always"` (the live weight exponent base
when there is no `e_base_w`, and the live input max with `in_exp_anchor:
"min"`) and for the saturation log: at `verbose` 1 (the default) once per
`verbose_sample_every` (100) forwards per layer, at `verbose` 2 every step;
`verbose` 0 reads nothing. Both paths return the input's type (a bf16 model
stays bf16), after the window. With `train_path: "fast"`, calibration's float
reference (`calibrate_hw_grids`) also sums in fp32 with autocast off, so the
grids it picks are those of the sum it trains on, AMP or not; with `"hw"` it
is unchanged (the input's type under the caller's autocast: a bf16 model sums
in bf16).

```json
"xblock_accum": {"out_window": "always", "train_path": "fast"}
```

HW-exact switches (all off by default, match the NPE emulator):
`mant_round: "truncate"`, `overflow: "raise"`, `bias_round: "half_up"`,
`out_round: "nearest"`, `in_exp_anchor: "min"`, `static_shift_bits: 7`
(hw_grids `static_shift = -(e_qa_out + 12)`), `in_mode: "int"` (first layer,
hw_grids `in_scale`). `configs/mx_config_dof_run_hw_exact.json` turns them all
on. `overflow: "raise"` raises in eval on an accumulator, window, exponent
register or int input overflow. `overflow_saturate: ["out_window"]` (needs
`overflow: "raise"`; on in the HW-exact configs) saturates the window instead,
to the HW value: mantissa all ones at the top 4b code, `+-127 * 2^15` on the
output grid, at any `e_qa_out`. The default clamp `+-(2^22 - 1)` reads the
same after the next MX requant only for `e_qa_out` in -29..-14 (fp32 log2 in
microxcaling). The accumulator keeps raising (HW wraps it).
`mant_round: "truncate"` is what HW silicon does today (Q1); it costs ~10 dB
on a 4-layer DOF-shaped net and is under discussion with HW, so the default
stays `"nearest"`.

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

**Not every path can split.** `MXConv2dHW` shifts both operands onto one shared
fixed-point accumulator grid and requires `a_elem_format == w_elem_format`. With
`xblock_accum.mode: "hw_fixed_point"`, `separable_wa` is therefore disabled
before scoring starts, with a warning and a note in the artifact — the ladder
still assigns, it just assigns one format per layer.

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
- after assignment, one extra pass measures the **finished networks**, every
  demotion in place at once — the numbers no per-layer score can give you:

  ```
  auto_mixed | vs FP32: int8 baseline 44.10 dB -> assigned mix 30.53 dB (ladder cost 13.57 dB)
  auto_mixed | assigned mix vs int8 reference: 30.53 dB SQNR
  ```

  `assigned_vs_reference` alone is not readable: "30 dB below the int8 net"
  means something very different depending on whether that net was 31 dB or
  45 dB below FP32. All three land in `sensitivity.json` under `meta`.

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

## DOF edge KPI (`tools/dof_edge_kpi.py`)

Quality of DOF under quantization, by the DOF owner's method: diff L is the luma
difference between the reference and the other frame warped onto it, written by
DOF infer as `diff_img = (full_res_y[0] - rgb_to_grayscale(im0_recon)).abs().clamp(max=255)`,
i.e. |Y(A) - gray(warp of B onto A)|, already abs, in [0, 255], no offset (the
script takes it as given and never recomputes it). Errors near edges are the most
motion-sensitive and suffer most from quantization, so the KPI finds the edge
pixels of the reference frame A, widens them into a band, and sums |diff L| over it.

```bash
pip install opencv-python-headless        # or: pip install '.[eval]'
python tools/dof_edge_kpi.py --images FRAMES_DIR --diffs DIFF_DIR --out kpi.csv
```

Without OpenCV, `tests/test_dof_edge_kpi.py` is skipped with the reason "OpenCV
missing: pip install '.[eval]' to run the DOF edge KPI tests" (`pytest -rs` shows
it); the rest of the suite does not need it.

- `--images` holds every input frame, both A and B; edges are found on Y of the
  reference A. Preferred: the network's Y8 inputs (e.g. `Y8_LME_576x1024_PNG/`,
  one channel, already luma).
- Pairing. When every diff is named `<A>_<B>_diff_L.<ext>` (DOF infer's
  `common_name`, e.g. `0115_IMX874_Front_142_0115_IMX874_Front_141_diff_L.jpg` or
  `0015_GN3_001_0015_GN3_000_diff_L.png`), each diff pairs with its reference.
  Reference = first stem of the diff name (DOF infer's im0: confirmed from its save
  code, and measured 2026-11-10 on the DOF infer output); `--ref-frame second` to
  flip, as an override only. Each run prints which. Stems hold `_`, so the split
  used is the one where both halves are stems in `--images`; none or several is an
  error naming the file. Otherwise pairs by file stem (`img_001.png` with
  `img_001.npy`). `--pair-from-diff-name` / `--pair-same-stem` force a mode.
  An unpaired file is listed and is an error unless `--allow-unpaired`; in
  diff-name pairing that is a diff whose A or B frame is missing, while frames no
  diff names (a dataset folder always holds some) are only a note: their count and
  first names, in the run summary. An empty folder is an error.
- Images: png/bmp/jpg/tif, 8 or 16 bit. One channel is used as is; color is
  reduced to BT.601 luma (the same weights as torchvision `rgb_to_grayscale`).
- diff L: `.npy`, 8/16-bit png/tif, or 8-bit jpg. A 3-channel diff whose channels
  agree within `--diff-channel-tol` (default 2) is read as gray (channel 0);
  otherwise it is an error naming the largest channel difference (a colormap
  needs another rule), never a silent luma. JPEG is lossy: the diff carries
  compression noise of a few levels everywhere. The KPI compares runs on the same
  data, so it stays fair, but png is cleaner when the DOF owner can give it.
  The DOF tool's files measured 2026-11-10: jpg, uint8, 3 equal channels, min 0,
  median 7: plain |diff|, no offset. `--diff-offset` is subtracted
  from png/tif/jpg values before abs, for a signed diff stored with an offset (never
  from a `.npy`, already signed). To tell which a png is, run once
  and read the printed `diff L as stored: min, max, median` line: a plain |diff| has
  min 0 and a median near 0; a signed diff with an offset has its median near the
  offset (128 for 8 bit, 32768 for 16 bit) with values on both sides: rerun with
  `--diff-offset 128` (or `32768`). `--diff-offset` and `--diff-offset-float` take
  the same values: a finite number, or `none` (no offset; `--diff-offset`'s
  default). Text, `nan` or `inf` is refused with a named error (exit 2): a nan or
  inf offset would make every diff value nan or inf.
- Edges: `--detector canny` (default; Gaussian blur `--sigma` 1.0, thresholds 0.66 /
  1.33 x the median per image, or fixed `--canny-low/--canny-high`), or `sobel`
  (gradient magnitude above `--sobel-percentile`, default 90) as a cross-check.
  `--dilate N` (default 1) widens each edge by N px. `--save-masks DIR` writes the
  masks as png to check by eye.
- CSV: one row per pair (`name, H, W, edge_pixels, edge_fraction,
  sum_abs_diff_on_edges, mean_abs_diff_on_edges, sum_abs_diff_all,
  ratio_edge_to_all`), then a TOTAL row from the summed counts (means recomputed,
  never averaged). A frame with no edge pixels has mean `nan` and still counts.
- Float vs quant, opt-in: `--diffs-float FLOAT_DIR` adds the float model's diff L
  folder (same frames); `--diffs` stays the run under test (quant). Without it, the
  CSV and the printed lines are exactly as before (a test pins them byte for byte).
  ```bash
  python tools/dof_edge_kpi.py --images FRAMES_DIR --diffs QUANT_DIR --diffs-float FLOAT_DIR --out kpi.csv
  ```
  Both folders go through the same loader and pairing, and must hold the same pairs:
  a pair in one run only is listed and is an error, or with `--allow-unpaired` it
  is skipped in both and counted in a note. Edges are found once per pair on the
  reference frame, and the same band is used for both runs. Each row adds
  `float_*` (the float run's KPI on that band), `delta_*` (quant - float of each
  measure: what quantization adds over float) and `qerr_*` (the KPI of
  |dq - df|), then `diff_kind`. The TOTAL row builds every column from the summed
  counts. diff L = warped - reference and the reference is the same in both runs,
  so for signed diffs (`.npy`, or png/tif/jpg with an offset) dq - df =
  warped_quant - warped_float: the quantization error alone, with the float model's
  own error cancelled. For a plain |diff| file (the DOF tool's files today),
  `diff_kind` is `abs` and the run warns: `qerr_*` is then not that difference,
  and only `delta_*` means something (TOTAL: `signed`, `abs` or `mixed`).
  `--diff-offset-float` is the float run's offset (default: `--diff-offset`), or
  `none` for a plain |diff| float run against a signed quant one; a `.npy` never
  gets one. Formats may differ per folder (png vs npy); a pair whose
  two files differ in shape, or in stored dtype when both are images or both
  `.npy` (8 vs 16 bit is another scale), is an error naming the pair.
  The run prints a `float diff L as stored` line and three more TOTAL lines:
  `TOTAL float`, `TOTAL delta quant - float` and `TOTAL |dq - df| (<diff_kind>)`.
- Decision: OpenCV (`opencv-python-headless`, optional `eval` extra) for the edges
  and the image IO: the mainstream library, one dependency for both. Rejected:
  scikit-image, a second large dependency for the same job. The library does not
  import it.
- Decision (2026-11-11, owner: compare two output folders, backward compatible):
  float vs quant is a second folder in the same run, measured on one band, with
  both the delta and |dq - df|. Rejected: running the tool twice and subtracting
  the CSVs (two bands if the settings differ, and no |dq - df|); delta only (a
  sign flip or a moved error of the same size cancels in it); |dq - df| only (it
  is meaningless for the plain |diff| files the DOF tool writes today).

---

## Single-op simulator (`configs/`)

Each NPE op (conv, convT, warp) runs alone through the MX lib (ALG) and the
HW team's emulator (`configs/hw/npe_mx_emu`, verbatim), and every stage is
compared. Conv and warp run on their code; their emulator has no convT yet.
How it works, the warp included: `configs/README.md`.

```bash
cd configs
python main.py -m warp --config mx_config_npe_e4.json   # warp, ALG vs HW, per stage
python hw_vs_alg_check.py --only warp                    # one vector per HW question
```

---

## Notes

- `scale_bits` is shared between weights and activations (library constraint).
- The block shared exponent is `floor(log2(amax))` computed in float32 (microxcaling, and the HW
  team's emulator `mx_format` the same). For a block max just below a power of two (e.g. `2^22 - 1`)
  its float32 log2 rounds up to the integer, so the exponent comes out one too high. Measured: about 3e-7
  of random float blocks, 9e-7 of random integer blocks. ALG and the emulator agree; the silicon
  may differ (question sent to the HW team).
- Spec priority (highest to lowest): per-layer `mx_specs` > group > global `mx_specs`.
