# DOF inference experiments: configs to make HW grids from

One config per experiment. Each makes its static HW grids once, offline; inference then reads them from
the config and never calibrates. Every change reuses the keys of an existing config, named below.

| file | what it changes, from which config | grids written to `mx_config_grids.json` |
|---|---|---|
| `base.json` | the owner's DOF infer config, as is: the 30 convs on the HW path, output window `always` | `hw_grids` for the 30 convs |
| `se4.json` | `base.json` + the 4-bit shared-exponent register: `xblock_accum.in_exp_bits: 4`, `in_exp_mode: "always"` (from `mx_config_dof_run_win_se4.json`) | as base, each entry also holds `e_base_a` / `e_base_w` |
| `se4_warp.json` | `se4.json` + the `model.warp` `act_quant` entry (int8 ref, `calibrate: true`; Q8.8 grid; `call_sites: 2`) and `calibrate_act_scales` (both from `mx_config_dof_run_win_cal_convT.json`) | as se4, plus the warp ref scale per call site, written into the warp entry. The warp output is not quantized (no `output` fxp), though arch says it is plain INT8 on the NPE (`mx_config_npe_e4.json` quantizes it) |
| `se4_convT.json` | `se4.json` + the five ConvTranspose2d layers as in `mx_config_dof_run_win_cal_convT.json`: the dense `model.block{2,3,4}.conv1` on the HW path (`convT: "hw"`, already in base), `model.block{2,3}.convtranspose_2` on plain MX (group `convT_plain`) | as se4, plus `hw_grids` for the 3 `conv1` (plain MX needs none) |
| `se4_warp_convT.json` | `se4.json` + both changes above | both of the above |
| `se4_warp_convT_hw.json` | `se4_warp_convT.json` + every HW behaviour the lib has today (below, "Gaps to HW"): from `mx_config_dof_run_hw_exact.json` the `xblock_accum` keys `pla: "infer"`, `pla_bias: "bins"`, `bias_round: "half_up"`, `out_round: "nearest"`, `out_margin_bits: 1`, `in_exp_anchor: "min"`, `static_shift_bits: 7`, the `input_layer` group on `model.feature_pyramid_extractor.convs.0.0.0` (`in_mode: "int"`, `in_int_scale: "free"`, its other keys this config's global ones) and the model `out_quant` entry (Q8.8); from `mx_config_npe_e4.json` the warp `output` fxp (int8, `calibrate: true`, `pow2: false`); fxp `round: "half_up"` on the warp reference, grid and output and the model `out_quant` (2026-11-17) | as se4_warp_convT, each entry also holds `static_shift`; the PLA entries `pla_frac`, `pla_m_pos`, `pla_m_neg`, `pla_bin_bounds`, `pla_bin_scales`, `pla_bin_biases` on each folded layer; `in_scale` on the first layer; the warp output scale per call site, in the warp entry |

Base and the 4-bit exponent:
- In `base.json` no ConvTranspose2d is in `layers`, so they all run FP32. Its `convT: "hw"` and its
  `convT_plain` group only take effect once those layers are listed, as in the two `convT` configs.
- 4-bit shared exponent = `in_exp_bits: 4`, the shift in [0, 15] above the calibrated `e_base` (the owner's
  meaning): each tensor's block exponents are held in 16 codes above it (`e_base_a` / `e_base_w`, written
  to the grids). The MX scale code width, `scale_bits`, stays 8. `mx_config_npe_e4.json`, a sim config,
  uses the other meaning, `scale_bits: 4`, which adds nothing to the grids.

Not made: 5 w_int8, 6 se4_w_int8: deferred by the owner 2026-11-10; weights stay MX int8 with the 4-bit
shared exponent, as in the se4 configs.

## Results (DOF infer, owner's runs 2026-11-18)

Each quant run against the float reference `infer_front_float_reproduce`, 374 pairs, KPI by
`tools/dof_edge_kpi.py` (owner's output 2026-11-18). motion = motion error / float motion, on the edge
band; > 1 px = pixels with motion error > 1 px; image = image error added over float.

| run | motion | > 1 px | image |
|---|---|---|---|
| float rerun vs float reference (noise floor) | 0.01 % | 0.00 % | +0.00 % |
| `se4_warp_convT` + PLA | 2.43 % | 0.29 % | +0.13 % |
| `se4_warp_convT` + HW rounding | 2.43 % | 0.26 % | +0.12 % |
| `se4_warp_convT` + warp output int8 | 2.59 % | 0.30 % | +0.17 % |
| `se4_warp_convT` + `out_quant` Q8.8 | 2.43 % | 0.29 % | +0.13 % |
| `se4_warp_convT` + first layer int8 | 2.65 % | 0.34 % | +0.13 % |
| `se4_warp_convT_hw`, `in_exp_anchor: "max"` (all HW keys) | 2.82 % | 0.36 % | +0.18 % |

- Each bisect row is `se4_warp_convT.json` plus that one HW change.
- The full HW row runs `in_exp_anchor: "max"`: `"min"` gives NaN (below, row "near-zero weight block").
- The numbers need the nndof motion-save fix (below, row "DOF mv save"). Before it, every run with
  `out_quant` read 75 % motion error: the `out_quant` run's `mv/*.dof` were exactly 0.25x the others'
  (correlation 1.000).

## Gaps to HW (`se4_warp_convT_hw.json`)

What the closest config does against the NPE HW, owner decisions 2026-11-12. `Qn` = question n of the HW
team's answers, `configs/hw/warp_hw_questions_and_answers.md` (2026-11-15).

| item | this config | HW | state |
|---|---|---|---|
| mantissa round | `nearest` (lib default; no `mant_round` key) | nearest (HW agreed 2026-11-02); the HW emulator still truncates (`configs/README.md`), so a bit-exact run against it differs here | already matched |
| exponent base | static `e_base_a` / `e_base_w` from the grids, 4-bit register (`in_exp_bits: 4`) | static, 4 bits | already matched |
| exponent register anchor | `in_exp_anchor: "min"`; the DOF runs need `"max"` until the near-zero row below is solved | min non-zero block exponent | blocked by the near-zero row |
| near-zero weight block, `min` anchor | `block_exp_range` counts every block with max > 0, so a dead filter (max 4.9e-39, DOF `convs.0.0.0` out-channel 0) sets `e_base_w` -128 and the run gives NaN (owner's run 2026-11-16; no NaN with anchor `max`) | not known | waiting for HW |
| block exponent near 2^k | `floor(log2(amax))` in float32 (microxcaling): one too high just below a power of two (README, "Notes") | not answered (Q12) | waiting for HW |
| PLA | `pla: "infer"`, `pla_bias: "bins"`: the activation folded into the PCU multiply, per-bin scale and bias | PCU PLA, bin form | fixed here |
| bias / output round + static shift | `bias_round: "half_up"`, `out_round: "nearest"`, `out_margin_bits: 1`, `static_shift_bits: 7` | half up, nearest, 7-bit signed static shift | fixed here |
| fixed-point round (warp ref, grid, warp output, model output) | fxp `round: "half_up"`, ties toward +inf (-37.5 -> -37), on the warp reference, grid and output and the model `out_quant` (owner and co-manager 2026-11-17); the lib default stays `half_away` | PCU nearest, ties toward +inf (-37.5 -> -37, Q7); the reference and grid words are rounded by the layer that makes them (Q5); the float-to-code tie of the reference is not answered (Q5 open) | fixed here (reference tie: waiting for HW) |
| first-layer input | `in_mode: "int"`: per-tensor int8, scale `in_scale` in the grids | int8 input, no block exponent | fixed here |
| first-layer bias | the bias rounds on the lib's first-layer accumulator grid, finer than HW's: bit-exact only without bias; with a bias a per-channel offset up to 2^-16 is left (`configs/README.md`, "HW-exact (M1-M8)") | bias on HW's accumulator grid | left open |
| warp reference | signed int8, calibrated scale per call site | signed int8 `[-128, 127]` by default, uint8 when selected (Q6); the reference step not answered (Q6 open), it reaches HW only through the PCU multiplier (row "warp output multiplier") | already matched (format) |
| warp grid | relative motion, Q8.8 on both axes | relative motion, int16 Qm.k with 0 <= k <= 8, X and Y may differ, the format fixed by the producer, not per frame (Q3, Q4) | already matched (Q8.8 is legal; the other formats are unused) |
| warp border | the DOF model's own warp (in nndof, outside this repo) samples the float frame: `grid_sample`, zero outside, `align_corners=True` (Q1); the sim's float spec `cases.Warp` now clamps the whole coordinate, as HW and their emulator do | the whole coordinate clamped to the edge, its excess fraction dropped: edge replication (Q1); negative words and all four sides not confirmed | left open (the DOF model's warp, in nndof) |
| motion limit | only the Q8.8 word range (saturate); no separate X / Y limit | separate X and Y maximum motion, saturated to it (320.23 -> 319, Q3); how the limits are set is not specified | left open |
| warp full weight | float bilinear, no weight field | 16-bit unsigned weight field; the full weight 2^16 at 8 + 8 fraction bits does not fit, its encoding is open (Q4) | waiting for HW |
| warp output | int8, calibrated arbitrary-real scale per call site | int8 `[-128, 127]`, own scale through the PCU rescale (Q7) | fixed here |
| warp output multiplier | the scale is any real (`pow2: false`) | PCU multiplier format, and its link to the Q10 scaling unit, not given (Q7) | waiting for HW |
| warp batch | the model's batch, reference and grid as the model gives them | one frame and a same-size grid per call (Q8, Q10) | left open: check that DOF infer runs batch 1 with a same-size grid |
| model output | `out_quant` Q8.8 | Q8.8 | fixed here |
| DOF mv save (nndof, outside this repo) | nndof `infer_list.py` saved `flow_pyr[0]`. In float runs it aliases the model flow and gets the post-processing's in-place x4 (flow px -> net_in px); with `out_quant` the alias breaks, so those runs saved 1/4 units, and `flow_pyr[0]` is never quantized by `out_quant`. Fix (owner, local nndof edit 2026-11-18): save the model flow, cloned before post-processing, times net_in / flow size (4); float files unchanged (float rerun vs reference 0.01 %) | n/a (a DOF infer save, not an HW op) | fixed locally, to push into nndof |
| `model.block{2,3}.convtranspose_2` (grouped convT) | plain MX (group `convT_plain`), no HW path | grouped convT on the NPE | waiting for HW |
| overflow | clamp (lib default; no `overflow` / `overflow_saturate` keys) | accumulator wraps, window saturates | left open |
| concat / eltwise sum | float | NPE ops | later |
| full-network bit-exact check vs the emulator | not done | the HW team fixed the emulator bugs agreed in the answers (commit `257a52e`): border clamp (Q1), last row and column (Q2), relative grid with 0..8 fraction bits per axis (Q3, Q4), signed int8 reference (Q6), PCU on the warp output with the int8 output clamp (`calib.py` `relux_bits` 24 -> 8, Q7), the fraction-sum cap 14 -> 16 (Q4), int64 product and accumulator (Q9), no width padding (Q10), one frame with a same-size grid (Q8, Q10). The sim's warp op is bit-exact against it on every warp case HW runs (`configs/README.md`, "HW warp") | not done |

### Open gaps

Every row above whose state is not "already matched" or "fixed here":

- exponent register anchor: blocked by the near-zero weight block.
- near-zero weight block, `min` anchor: NaN; waiting for HW.
- block exponent near 2^k: waiting for HW (Q12).
- fixed-point round: the reference's float-to-code tie, waiting for HW (Q5).
- first-layer bias: lib's accumulator grid finer than HW's; left open.
- warp border: the DOF model's warp, in nndof; left open.
- motion limit: no separate X / Y limit; left open.
- warp full weight: encoding open; waiting for HW (Q4).
- warp output multiplier: format not given; waiting for HW (Q7).
- warp batch: check batch 1 with a same-size grid; left open.
- DOF mv save: fixed locally, to push into nndof.
- grouped convT `model.block{2,3}.convtranspose_2`: no HW path; waiting for HW.
- overflow: clamp, HW wraps / saturates; left open.
- concat / eltwise sum: float; later.
- full-network bit-exact check vs the emulator: not done.

### Notes

- Fixed-point round: the conv outputs already round as the PCU does (`out_round: "nearest"` adds half, then
  floors; `bias_round: "half_up"`); the fxp quantizers (warp, `out_quant`) round `half_up` in this config only,
  every other config keeps the default `half_away`.
- Overflow stays the default clamp (owner: "left overflow issue for now unsolved"): `overflow: "raise"`
  would stop an inference run at the first out-of-range word.
- `out_window` stays `always` as in se4 (PLA needs the window on; `always` is on, and with PLA `infer` train
  mode runs the float activation). `in_exp_mode` stays `always`: in eval both modes read the calibrated base.
- PLA folds only a `PReLU` / `LeakyReLU` / `ReLU` that directly follows an HW conv in an `nn.Sequential`.
  On the test stand-in that is the 30 convs; the 3 dense `conv1` convT have no activation after them, so
  no `pla_*`. The real DOF printed "PLA: folded the activation into 30/33 HW layers" (owner's log,
  2026-11-01): check that line at quant time.
- `static_shift_bits: 7` checks at calibration that `-(e_qa_out + 12)` fits 7 signed bits, and stops naming
  the layer if not.

## Make the grids (GPU)

Per config: copy it into the run folder as `mx_config.json`, then run the trainer's quant step with data:

```python
model = MXQuantizer(save_dir).quant(model, data=cal_data, forward_fn=forward_fn)
```

`calibrate_hw_grids` (and, with the warp, `calibrate_act_scales`) runs on 8 batches and writes
`<save_dir>/mx_config_grids.json`; an existing file is kept, so start from an empty folder. Copy that file
over `mx_config.json` for inference.

`call_sites: 2` is measured: on the real DOF, `calibrate_act_scales` refused 3 with "called 2 times" (commit
`cb581b8`). If the model changes, it stops again and names the count: set `call_sites` to it.

Checked by `tests/test_infer_dof_configs.py`, on a CPU stand-in with DOF's layer names and kinds.
