# Single-op MX Simulator

## Goal

Run each HW op alone, on tiny tensors, through two independent implementations,
the MX lib (ALG) and the HW implementation, and compare every intermediate
value. The scope is the three ops DOF needs: **conv**, **convT** (zero-insert +
conv), and **warp**.

## Files

| file | what |
|---|---|
| `cases.py` | the case table: one op per case, tiny shapes, fixed seed. Also `Warp`, the float spec of the warp |
| `taps.py` | `tap(taps, name, t)` logs a value at any point; `show(taps)` prints them in order |
| `main_ALG.py` | ALG side: MX lib only, one function per op (`conv`, `convt`, `warp`) |
| `main_HW.py` | HW side: wires in the HW team's `hw/npe_mx_emu` (conv). convT and warp are still placeholders |
| `hw_vs_alg_check.py` | one test vector per known ALG-vs-HW arithmetic difference; prints numbers. Rerun on each HW drop |
| `hw/` | HW team code, **verbatim, never edited here**: `npe_mx_emu/` package, their confluence guide, their example notebook (as flattened text) |
| `main.py` | runs both sides, compares the shared taps in order, names the first that differs |

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
python main.py -m all                # both sides compared
python main.py -m conv3x3_p0         # HW conv runs on unpadded cases
pytest ../tests/test_sim_ops.py      # ALG taps explain ALG output
```

## HW side (npe_mx_emu)

`main_HW.conv` runs their flow as in their notebook:
1. `MXObserver(mode="mx")` on x and on w gives `x_cal` and `w_cal`.
2. `LayerMXQParams` → `CalibratedMXConv2d` → `calibrate()` → `forward(x)`.
3. `out = out_raw * 2^-output_scale_log2`.

Their intermediates are tapped without touching their code: a forward pre-hook
catches the MX input, and a pass-through wrapper on `bcu` keeps each `p_sum`.
Their format is not ALG's (e4bm8: per-tensor calibrate offset, ceil log2,
unsigned 4-bit exponent with wrap, mantissa truncate, 46-bit acc, PCU round,
24-bit clamp), so their integer taps are `hw_*` and only `out` is compared.

Their code does not support padding (it raises) or dilation (it has no such
parameter, so the wiring refuses it). Use `conv1x1` or `conv3x3_p0`.

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
