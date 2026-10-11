# Small Model MX Simulator

## Goal

A minimal HW implementation of MX, runnable side by side with the MX lib emulation on the
same small model, same weights, same test vectors — so we can check bit-exactness, or
measure exactly where and by how much the two diverge.

Debugging on a toy model instead of a full network: one layer, a handful of blocks, every
intermediate value printable.

## Content

- `models.py` — benchmark of small models: single conv, conv-relu, conv-relu-conv, dw conv,
  concatenation, etc. Each model also comes in variants that hit the corner cases: tail
  blocks, stride / dilation / non-square kernels, spatial padding, bias, batch > 1, blocks
  that underflow or saturate the accumulator, MX rounding ties and mantissa overflow, zero
  blocks, exponent-field clamping, other element formats and accumulator widths. A case can
  override `mx_specs` (`mx=`) and `xblock_accum` (`accum=`) for itself.
- Some cases also carry a **static fixed-point output stage** (`out_quant`): the network
  output leaves in a fixed `Qm.n` word rather than in MX. It is a different quantizer —
  MX takes its scale from each block's max and so cannot overflow, while a static scale is
  fixed up front and overflows silently. Tagged `fxp`.
- `main_HW.py` — the HW side, and the contract: tap names, shapes, config keys, the open
  spec questions with the case that settles each.
- `main_ALG.py` — same interface, using the MX lib (`MXConv2dHW`, the fixed-point
  accumulator datapath).
- `tap_check.py` — the ALG side's own check, written from the spec with no MX lib import:
  rebuilds every HW layer's accumulator from its integer taps. `main_ALG.run` refuses to
  return taps that fail it.
- `main.py` — runs both and compares their taps in pipeline order.
- `export_vectors.py` — freezes cases into `bundle/` (`.npy` + `.txt` + manifest) for a
  side that has neither torch nor the MX lib.

## DOF first (tag `dof`)

DOF is conv + ConvTranspose + warp. Covered here:

- **warp** (tag `warp`, 11 cases) - `models.Warp` is the spec: int8 ref, Q8.8
  grid (a displacement in px, channel 0 = x), bilinear with 8-bit weights,
  zero padding, int8 out; static arbitrary-real scales per call site.
  `tap_check.ref_warp` rebuilds every `out_code` from the integer taps. One
  case per open convention (grid meaning, coordinate split, out of bounds,
  three tie rules, both saturations), plus `warp_subpix` (DOF as configured),
  `warp_levels` (3 call sites) and `conv_warp` (MX conv -> warp handoff).
- **DOF formats** - `fmt_int4`, `fmt_int6`, `fmt_int4_ties` (the ladder rungs).
- **DOF-shaped convs** (tag `dof_conv`) - generated from `dof_shapes.json`,
  one case per distinct Conv2d geometry, real W. The file comes from the real
  net: run `dump_layer_shapes.dump_layer_shapes(model, inputs, path)` where
  DOF lives and drop the JSON here. Until then the tag is empty.
- ConvTranspose2d - not yet (no integer path in the lib).

## Geometry

NPE by default, as confirmed by arch: activations blocked along W, weights flattened per
filter (C-major) and blocked along that stream. `--blockify channel` switches both the
config and the pattern layout to Cin blocking. The two have to agree: patterns such as
"one dominant element per block" are laid out along the block axis.

## Tap contract (per HW layer)

| tap | dtype | shape (NPE) |
|---|---|---|
| `a_int` | int8 / int16 | `[B, C, H, W]` |
| `a_exp` | int16, unsigned code `E + emax` | `[B, C, H, ceil(W/bs)]` |
| `w_int` | int8 / int16 | `[O, Cin, kH, kW]` |
| `w_exp` | int16, code | `[O, ceil(Cin·kH·kW/bs)]` |
| `acc` | int64 | `[B, O, Ho, Wo]` — the bit-exact layer output |
| `sat` | uint8 | `[B, O, Ho, Wo]` |
| `out` | fp32 | `acc · 2^e_layer_min` (lossy above 24 bits) |

`emax = 2^(scale_bits-1) - 1`; an all-zero block has code 0.

## Usage

```bash
python main_ALG.py -m all                     # every case, taps + tap_check summary
python main_ALG.py -m mxq --blockify channel  # a tag, other geometry
python main.py -m baseline                    # both sides, compared (exit 2 until HW exists)
python main_ALG.py -m dof                     # DOF-first cases
python export_vectors.py -m corner --clean    # freeze cases for arch
pytest ../tests/test_sim_cases.py             # every case rebuilds + triggers its target
```
