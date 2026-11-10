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
| `main_HW.py` | HW side: wires in the HW team's `hw/npe_mx_emu`: conv (incl. PLA via a labelled bins stand-in) and warp (`CalibratedMXWarp`) run on their code. convT is a placeholder: their emulator has no convT |
| `hw_vs_alg_check.py` | one test vector per known ALG-vs-HW arithmetic difference; prints numbers. Rerun on each HW drop. `--only warp`: the warp vectors (T12) alone |
| `warp_vectors.py` | the T12 warp vectors: label, printed text, and each side's values |
| `hw/` | HW team code, **verbatim, never edited here**: `npe_mx_emu/` package, their `README.md` and confluence guide (with their answers to the review table), `QA.md` (their answers to our later questions), `PLAN.md` (their PT2E plan), `EL_nb/` their notebooks |
| `sim_config.py` | loads the one framework config both sides read (pure json, no MX lib; per-call-site fxp lists by the lib's pure rule, `fixed_point/fxp_sites.py`) |
| `sim_cli.py` | what the CLIs share: the exit codes, and the many-case loop that reports each refusal and goes on |
| `mx_config_npe_e4.json` | `mx_config_dof.json` with `scale_bits: 4`, the NPE exponent width |
| `mx_config_npe_e4_pla.json` | npe_e4 + output window + PLA in HW bin form, `hw_grids` for `conv_prelu`, `conv_relu`, `conv_prelu_negb` |
| `mx_config_npe_hw_exact.json` (+ `_nearest`, `_in`) | the lib's HW-exact switches M1-M8 (from `mx_config_dof_run_hw_exact.json`) with `hw_grids` for the conv cases; `_nearest`: mantissa round to nearest (the truncate share); `_in`: first-layer integer input (`in_mode int`, POT scale) |
| `gen_hw_grids.py` | writes a config's `hw_grids` from an OFFLINE lib calibration on seeds 1-8 (never the test seed). The sim itself never calibrates |
| `main.py` | runs both sides, compares the shared taps in order, names the first that differs |
| `infer_dof/` | not the sim: one framework config per DOF inference experiment, to make its HW grids from (its `README.md`) |

## Taps

| op | taps (ALG order) |
|---|---|
| conv | `x w a_int a_exp w_int w_exp acc sat out` |
| convt | `x w a_int a_exp w_int w_exp out` (HW adds `x_dil`) |
| warp | `x flow ref_code grid_code out_pre out_code out` (HW: see HW warp) |

- **conv**:
  - Integer form: `value = mant * 2^(E-6)` and `code = E + 127`.
  - NPE blocking of 32: activations along W; weights per filter, flattened C-major.
  - `acc` is the int64 word of the 48-bit saturating accumulator, with LSB `2^-20`.
- **convT**:
  - `xblock_accum.convT = "hw"`: the lib's `MXConvTranspose2dHW` quantizes the compact input (blocks along W), zero-inserts the extracted operand and runs the HW conv. Taps as conv; `w_int` / `w_exp` are its equivalent conv weight `[O, Cin, kH, kW]`. HW confirmed this order (quantize compact, zeros never in a block).
  - Otherwise plain MX blocked along Cin with fp32 accumulation: only `out` compares, as float.
  - HW side: their emulator has no convT yet (output interleaving still open).
- **warp**:
  - `ref` is int8 with a static scale.
  - `grid` is Q8.8: a displacement in px, channel 0 = x.
  - Bilinear with 8-bit weights; samples outside the image count as 0.
  - `out` is int8 with a static scale, when the warp entry has an `output` fxp. With none (the
    `hw_exact` and `dof_run` configs), or one the lib reads as off (`false`, `null`,
    `{"enabled": false}`), the lib leaves the warp result unrounded: ALG `out` is
    `out_pre`, there is no `out_code`, and the run prints a line saying so (`main_ALG.warp_out_note`).
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
python main.py -m warp --config mx_config_npe_e4.json   # warp: ALG vs their emulator, per stage
python hw_vs_alg_check.py --only warp                    # warp vectors, one per HW question
python main.py -m conv3x3_p0 --config mx_config_npe_e4.json   # HW conv runs at 4-bit exp
python main.py -m conv3x3_p0 --config mx_config_npe_e4.json --input both   # float vs integer input
pytest ../tests/test_sim_ops.py      # ALG taps explain ALG output
```

Over many cases (`-m all`, an op), `main.py`, `main_HW.py` and `main_ALG.py`
run every case (`main_HW` and `main_ALG` through one loop, `sim_cli.run_each`;
`main.py` loops over both sides itself and reports each case by the same
`sim_cli` rule, `not_run`). A case a side refuses (`sim_config.REFUSED`: their code cannot run it, the
config lacks what the sim reads, such as static grids, or a per-site fxp list
does not hold one spec per call site, the lib's error,
`sim_config.SiteListError`) prints `refused: <why>`
and the run goes on; any other error prints `FAILED` with its traceback and
the run goes on. The last line names the refused and failed cases. A config
the sim refuses at load prints `refused: <why>`. These missing config entries
are refusals that name the entry and why (`sim_config.require`), not a
`KeyError`: the warp entry's inputs and their fxp (`sim_config.warp_sites`),
HW's ref/grid `total_bits`, and the PLA keys HW's bins read. A grid the lib
itself rejects (`fixed_point/hw_grids.check_hw_grid`) is still the lib's
`ValueError`, a failure.

Exit codes, one meaning in `main.py`, `main_HW.py`, `main_ALG.py` and
`hw_vs_alg_check.py` (`sim_cli.py`):

| code | meaning |
|---|---|
| 0 | every case (section) ran clean |
| 1 | a failure (an error that is not a refusal) or a mismatch |
| 2 | something was refused or skipped (a case, a section, the config), nothing failed (an argparse usage error also exits 2) |
| 3 | `main.py` only: everything ran, only float taps differ |

A refusal is never a pass: it never exits 0.

## HW side (npe_mx_emu)

`main_HW.conv` runs their flow as in their notebook:
1. `MXObserver(mode="mx")` on x and on w (`per_row_flatten=True`: blocks
   per flattened kernel, as their quantizer) gives `x_cal` and `w_cal`.
2. `LayerMXQParams` → `CalibratedMXConv2d` → `calibrate()` → `forward(x)`.
3. `out = out_raw * 2^-output_scale_log2`.

Their intermediates are tapped without touching their code: a forward pre-hook
catches the MX input, and a pass-through wrapper on `bcu` keeps each `p_sum`.
Their format is not ALG's (e4bm8: per-tensor calibrate offset, floor log2,
unsigned exponent with wrap, mantissa truncate, int64 acc with two's-complement
wrap, PCU round half up, 24-bit clamp, output grid 2^-12), so their integer
taps are `hw_*` and only `out` is compared.

Input mode: `--input float` feeds the float tensor (what the MX lib sees in
training). `--input int --x-frac F` first snaps x to the grid 2^-F, giving an
integer tensor as the NPE receives it; both sides get the same tensor.
`--input both` prints HW-ALG for each. Their forward rounds `x * 2^x_cal` only
for an input layer (`is_input_layer`); the sim's layer is not one, so
`pre_round_moved` only counts elements off that integer grid.

Exponent width: `mx_specs.scale_bits` drives both sides. ALG decodes the
exponent code as `E + 2^(n-1)-1`. HW maps the config to their `mx_type`
`mx-e{scale_bits}bm{N}` (e4bm8, e6bm8, e8bm8; the wrap mask comes with the
type) and `block_size`, and refuses a type they don't have. Conv runs int8
mantissas only: their `mx-e4bm16` is the warp's 16-bit lane (grid and
weights), and their conv on it gives wrong numbers without an error, so
`hw_format` refuses it; another width is refused as a type they lack. The NPE
width is 4 bits: `--config mx_config_npe_e4.json`.
Accumulator width: `xblock_accum.bits`, one of their `SUPPORTED_ACC_WIDTHS`
(46, 48, 54, 56), else refused. Fixed inside their code (their API has no
parameter): PCU clamp 24 bits (`main_HW.HW_FIXED`).

Their code does not support padding (it raises) or dilation (it has no such
parameter, so the wiring refuses it). Use `conv1x1` or `conv3x3_p0`.

## HW warp (npe_mx_emu)

`main_HW.warp` runs their `CalibratedMXWarp` as their example notebook
calls it; nothing of theirs is reimplemented or edited:
1. Inputs from the case: `reference = x / ref_scale` (code units; their
   packer rounds half to even and clamps to uint8 0..255, no scale) and
   `coords` = absolute positions, mesh + flow, `[B, H, W, 2]`, float64 as in
   their notebooks (their weights and bcu product take the coordinates' dtype).
2. Config: grid word width (their `grid_bit_width`) and fraction bits from the
   warp entry's grid fxp at call site 0 (a per-site list must hold one spec
   per call site, by the lib's rule in `fixed_point/fxp_sites.py`, else
   refused; the sim runs one forward, so both sides use site 0; `call_sites`
   is tapped);
   accumulator width from `xblock_accum.bits` (the BCU conv uses). Refused
   unless the reference is 8-bit and the grid word at most 16 bits.
3. Setup: `LayerMXQParams(x_cal=0, w_cal=0, out_cal=0)` (output in code
   units), then `calibrate()` with its default fraction bits (7, 7). It only
   fills the PCU c_structs, which `forward()` requires and never reads (their
   PCU is bypassed). Nothing comes from data.
4. Taps, by pass-through wrappers on `pack_warp_reference`, `pack_warp_grid`,
   `trc` and `bcu`: `hw_ref_code` (W padded to 32), `hw_grid_code`,
   `hw_frac_bits`, `hw_weights`, `hw_corners`, `hw_acc` (bcu's int64 p_sum,
   taken inside their loop), `out_raw` (their float32 output as is: acc /
   2^(frx+fry), rounded by float32 once |acc| > 2^24) and
   `out = out_raw * ref_scale`.

Their fraction bits come from the data (per frame, from max |position|); the
config's are fixed. `frac_cfg`, `hw_frac_bits` and `frac_match` show both, and
a `!!` line is printed when they differ; their code runs on its own.
`frac_over`: frx + fry above 14, their weight lane (their `trc_warp` and
`build_warp_c_struct_arr` refuse it, the path that runs does not).

Refused, because their code would give wrong numbers without an error:
batch > 1 with different flows (they read batch 0's grid), a reference their
packer would clamp (negative or above 255, so the randn warp cases), a
non-finite reference, H < 2, and an axis whose fraction bits come out infinite
or negative. The `warp_u8_*` cases (codes
0..127, the range both sides hold the same) run on both sides.

main.py adds, per stage and labelled with the numbers of the questions sent
to the HW team: ref code, grid position (px), acc inside the frame and on the
border (code units), and their float32 output against their own int64 acc.
The generic compare covers `out`. ALG's int8 `out_code` has no HW counterpart.
`hw_vs_alg_check.py` T12 has one vector per difference, labelled the same way
(built in `warp_vectors.py`, which keeps each vector's ALG and HW values for
the tests). On a config the warp cannot run on (no `act_quant` entry, a
warp entry missing an input or its fxp, a per-site list of the wrong length;
any refusal, `sim_config.REFUSED`) T12 prints
`T12 skipped on <config>: <why>` (`warp_vectors.probe`). The same
holds for every section: T1-T10 run on fixed conv cases (`conv3x3_p0`, `conv1x1`, `conv3x3_p0_bias`), so a
config that holds grids only for its own cases (`npe_e4_pla`,
`npe_hw_exact_in`) skips them with the reason, and the rest runs. The last
line names the skipped sections; exit 0 when every section ran, 2 when one
was skipped (the exit codes above). Any other error, in T12 or any section, is
not a skip: it propagates with its traceback, exit 1.

## Static grids and PLA

The sim never calibrates: per-layer grids come from the config's `hw_grids`
(`e_layer_min`, `e_qa_out`, `pla_frac`, `pla_bin_bounds` / `pla_bin_scales` /
`pla_bin_biases`), looked up by `--layer` or else the case name. main_ALG hands
that entry to the lib (re-keyed to the sim's layer name), so the lib loads it
as in the framework. With the output window on and no grids the run stops
with a clear message. Make the values offline with `gen_hw_grids.py`.

`conv_prelu` / `conv_relu` / `conv_prelu_negb`: conv + PReLU (per-channel
slope) in an `nn.Sequential`, so the lib folds the activation into the PCU
(PLA, `pla_bias: "bins"`). ALG taps add `e_layer_min e_qa_out pla_frac
pla_m_pos pla_m_neg pla_bin_* win win_sat`.

HW PLA (their answer): per channel, bins split by boundaries on the
accumulator word, each with an int8 scale S_k and bias B_k: `S_k * acc + B_k`
→ `pcu_div` (round half up) → `relux`. PReLU = 2 bins at `acc = -b`. Their
emulator is still one bin (one `Cstruct` scale and bias). main_HW:
- `out` / `out_raw`: their PCU as it is, one bin (the upper one).
- `out_bins` / `hw_out_raw_bins`: **our assumption until HW emulator has
  bins** (`main_HW.bins_pcu`): pick the bin by acc vs boundary (config tie
  rule), then their `pcu` with that bin's S_k, B_k.

main.py prints HW-ALG for both, split by bin, and the PCU alone: the lib's
bins + window (`main_ALG.pcu_on`, the layer's saturation word with
`overflow_saturate`) on HW's own accumulator words vs the stand-in. `hw_vs_alg_check.py` T11 does the same on 20000 random words, the
tie case `acc == boundary` and both bias signs.

## HW-exact (M1-M8)

`--config mx_config_npe_hw_exact.json` turns on the lib's HW-exact switches
(mantissa truncate, overflow raise with the output window saturating to the
HW value `+-127 * 2^15` on the output grid (`overflow_saturate: ["out_window"]`),
out_round nearest, bias half up, 4-bit register anchored at the min block,
static shift, PLA bins). Both sides then
run from the config's grids only:
- HW `mx_type` uses the NPE register width (`in_exp_bits`), not `scale_bits`.
- HW `x_cal` / `w_cal` = `6 - e_base_a` / `6 - e_base_w` (their observer
  anchors the smallest block at shift 0; the config's bases are that block's
  exponent). Without the bases in `hw_grids`, their observers run on the input
  (tap `cal_from_config` says which).
- HW `Cstruct.shift` = `hw_grids` `static_shift`; the output decodes on
  `2^-(12 + static_shift)` = ALG's `2^e_qa_out`.
- First layer (`_in` cases): their `is_input_layer` with `cal = -log2(in_scale)`
  (POT only); weight cal from the lib's folded weight base.

Result: conv cases bit-exact (float and integer input), PLA cases bit-exact
against the bins stand-in, first layer bit-exact without bias. With a bias a
per-channel constant <= 2^-16 is left: the lib's first-layer accumulator
grid is finer than HW's, so the bias rounds on another grid.

## Config

The ALG side reads a real framework `mx_config.json` (`--config`, default
`mx_config_dof.json`) and calls `MXQuantizer(save_dir).quant(model)` the same
way the training/eval code does. The file's `mx_specs` and `groups` are used
unchanged; `layers` is reduced to the one layer under test:

- `--layer NAME` copies that `layers` entry (its group, kind, inputs).
  Defaults: conv/convT use plain global `mx_specs`; warp uses the file's first
  `act_quant` entry.
- `--backend` overrides `xblock_accum.backend` and nothing else.
- Refused, with the reason, instead of run on the wrong thing: a file with
  no global `mx_specs` or no element formats in it (per-layer groups or
  auto assignment), and, for conv (and the convT HW path), a config that
  does not give the lib's HW conv on NPE blocks (`xblock_accum` mode
  `hw_fixed_point`, `act_blockify: xblock`, `weight_blockify: flatten`):
  the taps decode those blocks. Also a warp entry without its two inputs
  (ref, grid), each with an `fxp` (and, for HW, its `total_bits`), and a
  grid missing a PLA key HW's bins read: each refusal names the entry. A
  grid the lib rejects (`check_hw_grid`) stays the lib's error, a failure.
- With no CUDA, `custom_cuda` is set to False; triton already falls back to
  python on a CPU tensor.
- Warp: the case's static ref/out scales replace the file's `calibrate` (or
  frozen scale) at every call site; a per-site fxp list is read by the lib's
  rule (`fixed_point/fxp_sites.py`). One forward of a fresh module runs call
  site 0. An entry with no `output` fxp is run as the lib runs it: unrounded.

To debug, start in `main_ALG.py`, set `-m <case>`, and break in the op function.
