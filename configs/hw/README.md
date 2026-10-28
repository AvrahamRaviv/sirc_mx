# Single-op MX Simulator

## Goal

Run each HW op alone, on tiny tensors, through two independent implementations,
the MX lib (ALG) and the HW implementation, and compare every intermediate
value. The scope is the three ops DOF needs: **conv**, **convT** (zero-insert +
conv), and **warp**.

## Background: MX format on the NPE

The NPE computes one conv line at a time: each input line is broadcast over many
BCUs (dot-product units) that multiply-accumulate one element per cycle. The
software model (`npe_mx_emu/`) reproduces only the input ordering and the
accumulation order, not the cycle-level broadcast.

MX pipeline per layer:

1. the block-shared exponent is broadcast per value (block = 32 elements along W)
2. mantissa multiply, then shift by the **sum** of activation + weight exponents
3. exponent shifts are **unsigned** (shift-left only; format-sized register)
4. the accumulator drains into the **PCU** (output window 24 bits = 23
   magnitude + sign): bias-add (bias pre-scaled into accumulator units
   `2**(x_cal + w_cal)`), shift by the **sum of the operands'
   `e_min_layer`** (+ static shift), round, clamp to the activation int range
5. the output feeds the next MX layer through a (re)quantizer; HW also supports
   static quantization (future flow)

Scale balance across the network (one integer-ization, unsigned BCU):

0. **scale-to-integer** the network input (first layer ONLY) and all the
   weights (offline): the BCU shifters are unsigned / shift-left only, so
   every operand must sit at a scale whose exponents are all `>= 0` — the
   product of a first-layer activation with any weight then never needs a
   right shift, by construction.
1. **BCU**: integer multiply-accumulate, left shifts only.
2. **PCU pla**: add the bias in accumulator units.
3. **PCU shift back down** (`pcu_div` `shift_right`; the PCU shifter is
   bidirectional): realigns the scale between two layers via the operands'
   `e_min_layer` sum, keeping the output activation in the
   non-negative-exponent regime — without it the accumulated `2**(x_cal +
   w_cal)` scale would explode layer over layer.
4. **MX-quantize** the output activation: the quantizer's own per-block
   divide + truncate *is* the integer-ization — there is **no** separate
   round-to-integer at layer boundaries. `x * 2^cal -> round` happens exactly
   once, at the first layer.

Constraints:

1. **No negative exponents.** The BCU can only shift left, so every operand is
   pre-scaled such that every *block's* exponent is `>= 0`. Calibration is
   **per-tensor only** (NPE structure), and a global max is NOT sufficient:
   `MXObserver` (in `npe_mx_emu/observers.py`) anchors the *smallest non-zero
   block amax* at exponent 0, `calibrate_offset = mantissa_offset - floor(log2(a_min))`
   (extraction is `floor(log2)` per block -- the block-max mantissa fills `[64, 127]`,
   so the integer bit of the `[-2, 2)` mantissa is a real precision bit).
   If the tensor's dynamic range exceeds the exponent window it raises loudly —
   a silent wrap annihilates whole blocks and the output collapses to bias-only.
   Resolution in that case: use a wider-exponent format, or rescale upstream layers.
   Note the exponent register is **wrap-only** (no saturation) and calibration
   is **static per-tensor**, fixed at compile time from offline PTQ: the
   calibration data must therefore *bound* the deployment inputs — leave
   margin. Minmax over the calibration set stays the default observer
   reduction; EMA on activations is a calculated risk (a deployment batch
   exceeding the EMA range can wrap a block exponent with no warning).
2. **First (sensor) layer is integer-only**, with a zero shared exponent; the
   scale rule is `calibrate = mantissa_offset - ceil(log2(a_max))` (e.g. `2**7`
   for data in `[-0.5, 0.5)`), modeled by `input_mx_quantizer`.  This is also
   the only place `x * 2^cal` is rounded to integer (see the scale-balance
   scheme above).
3. **`e_min_layer` = `mantissa_offset - calibrate_offset`** per operand; the PCU
   shifts the accumulator by their sum, realigning layers whose calibrated
   scales differ (input layer vs. intermediate MX layers).
4. **Base output scale = `2**(2 * bias_exponent)`**, where
   `bias_exponent = mantissa_bits - sign_bit - 1` (= 6 for all `*bm8` formats,
   so `2**12` in the default case). This fixed, format-defined output scale is
   what makes **static quantization** tractable: downstream layers always know
   the incoming tensor's scale without runtime metadata.

Supported formats are named `e{#exponent_bits}b{#mantissa_bits}` (the `b`
separates it from floating-point scale types): `e4bm8` (default), `e6bm8`,
`e8bm8`; mantissa is 8-bit for all current configs.

Worked two-layer example with stage-by-stage dumps (quantize -> TRU -> BCU ->
PCU): `EL_nb/mx_conv_npe_emu_example.ipynb`, built on `npe_mx_emu/` (repo root).

## Files

| file | what |
|---|---|
| `cases.py` | the case table: one op per case, tiny shapes, fixed seed. Also `Warp`, the float spec of the warp |
| `taps.py` | `tap(taps, name, t)` logs a value at any point; `show(taps)` prints them in order |
| `main_ALG.py` | ALG side: MX lib only, one function per op (`conv`, `convt`, `warp`) |
| `main_HW.py` | HW side: **placeholders**, filled by HW. Docstrings give the steps and the taps to emit |
| `main.py` | runs both sides, compares the shared taps in order, names the first that differs |
| `npe_mx_emu/` | standalone NPE MX emulator package: quantizers (`mx_format`), TRU/BCU/PCU pipe (`hw_pipe`), calibration observers (`observers`) + glue (`calib`), torch-tensor conv wrapper (`layers`) |
| `EL_nb/` | demo/experiment notebooks, incl. the `npe_mx_emu` walkthrough |

## Taps

| op | taps (ALG order) |
|---|---|
| conv | `x w a_int a_exp w_int w_exp acc sat out` |
| convt | `x w a_int a_exp w_int w_exp out` (HW adds `x_dil`) |
| warp | `x flow ref_code grid_code out_pre out_code out` |

- **conv**:
  - Integer form: `value = mant * 2^(E-6)` and `code = E + 127`.
  - NPE blocking of 32: activations along W; weights per filter, flattened C-major.
  - `acc` is the int64 word of the 48-bit saturating accumulator, with LSB `2^-20`.
- **convT**:
  - The lib has no HW path. It quantizes plain MX blocked along Cin and accumulates in fp32, so only `out` is comparable, as float.
  - Open question: does HW quantize before or after the zero insert?
- **warp**:
  - `ref` is int8 with a static scale.
  - `grid` is Q8.8: a displacement in px, channel 0 = x.
  - Bilinear with 8-bit weights; samples outside the image count as 0.
  - `out` is int8 with a static scale.
  - Every step rounds half away from zero and saturates.

## Usage

```bash
python cases.py                      # list cases
python main_ALG.py -m conv3x3        # one case, ALG taps (mx_config_dof.json)
python main_ALG.py -m conv3x3 --config mx_config_dof_npe_int6.json
python main_ALG.py -m convt --layer model.block2.conv1   # use that entry's group
python main_ALG.py -m conv3x3 --backend python           # step into the python ref
python main_ALG.py -m warp           # every warp case
python main.py -m all                # both sides compared (exit 2 until HW is filled)
pytest ../tests/test_sim_ops.py      # ALG taps explain ALG output
```

## Config

The ALG side reads a real framework `mx_config.json` (`--config`, default
`mx_config_dof.json`) and calls `MXQuantizer(save_dir).quant(model)` the same
way the training/eval code does. The file's `mx_specs` and `groups` are used
unchanged; `layers` is reduced to the one layer under test:

- `--layer NAME` copies that `layers` entry (its group, kind, inputs).
  Defaults: conv/convT use plain global `mx_specs`; warp uses the file's first
  `act_quant` entry.
- `--backend` overrides `xblock_accum.backend` and nothing else.
- With no CUDA, `custom_cuda` is set to False; triton already falls back to
  python on a CPU tensor.
- Warp: the case's static ref/out scales replace the file's `calibrate`.

To debug, start in `main_ALG.py`, set `-m <case>`, and break in the op function.
