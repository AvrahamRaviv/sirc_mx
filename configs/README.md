# Small Model MX Simulator

## Goal

A minimal HW implementation of MX, runnable side by side with the MX lib emulation on the
same small model, same weights, same test vectors — so we can check bit-exactness, or
measure exactly where and by how much the two diverge.

Debugging on a toy model instead of a full network: one layer, a handful of blocks, every
intermediate value printable.

## Content

- `models.py` — benchmark of small models: single conv, conv-relu, conv-relu-conv, dw conv,
  concatenation, etc. Each model also comes in variants that hit the corner cases:
  channels not a multiple of `block_size` (tail block), stride / dilation, spatial padding,
  bias, blocks that fully underflow, blocks that saturate the accumulator.
- Some cases also carry a **static fixed-point output stage** (`out_quant`): the network
  output leaves in a fixed `Qm.n` word rather than in MX. It is a different quantizer —
  MX takes its scale from each block's max and so cannot overflow, while a static scale is
  fixed up front and overflows silently. Tagged `fxp`.
- `main_HW.py` — takes `model` (from `models.py`), `ckpt`, `tvs`, and a `config` holding the
  HW-specific settings (accumulator bitwidth, scale size, blockification, saturation mode,
  `e_layer_min`). Replaces every op with its HW implementation, returns output per tv.
- `main_ALG.py` — same interface, using the MX lib as the replacement.
- `main.py` — runs both and compares their outputs.

## Backlog

- [ ] Add prints at each stage of `main_HW` / `main_ALG`, to compare intermediate computation
      states: W int + shared exp, A int + shared exp, accumulator before and after shift,
      accumulator after saturation, output requant. First stage that differs is the spec line
      we disagree on.
- [ ] Report, per stage: bit-exact yes/no, flat index of the first mismatch, histogram of the
      integer deltas.
