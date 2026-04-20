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

Measures each layer's isolated quantization sensitivity (SQNR of its own output with clean FP32 inputs, no upstream error propagation), then assigns each layer to a precision group automatically.

```json
{
  "groups": {
    "int4": {"w_elem_format": "int4", "a_elem_format": "int4", "block_size": 32, "custom_cuda": true},
    "int8": {"w_elem_format": "int8", "a_elem_format": "int8", "block_size": 32, "custom_cuda": true}
  },
  "layers": ["backbone.conv1", "backbone.conv2", "head.fc"],
  "auto_mixed": {
    "base":     "int4",
    "upgrade":  "int8",
    "strategy": "budget",
    "upgrade_fraction": 0.8,
    "batches":  32
  }
}
```

### Strategies

| Strategy | Key | Behaviour |
|----------|-----|-----------|
| `threshold` | `sqnr_threshold_db` | Layers with SQNR below threshold → `upgrade` group |
| `budget` | `upgrade_fraction` | Worst N% of layers by SQNR → `upgrade` group |

`upgrade_fraction: 0.8` upgrades 80% of layers to `int8` (the worst 80% by sensitivity).

Layers whose hooks never fire (no activation data) receive a **weight-only SQNR** estimate as fallback and are marked `(w)` in the sensitivity log.

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
