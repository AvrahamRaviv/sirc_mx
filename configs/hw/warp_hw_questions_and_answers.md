# Warp hardware questions and answers

This document preserves the original numbered questions and the answers supplied during review. Questions below describe the delivered emulator before corrections. Answers distinguish confirmed behavior from details that remain open.

All source references are relative to `configs/hw/npe_mx_emu/` in the delivered copy. ALG denotes our library. Reference values in examples are integer codes; bilinear outputs are in code units, `acc / 2^(frx+fry)`.

## 1. Border rule

**Original question.** `MXWarp_Loops` uses `trc → get_ref_frame`, clamping the corner index to `[0, W_padded−2] × [0, H−2]` while retaining the raw fraction (`hw_pipe.py:765–777, 756–762`). This is neither zero padding nor replication. The unused `trc_warp` and notebook references `infer_warp` and `golden_int` replicate edges. ALG uses `torch.grid_sample` with zero padding and `align_corners=True`.

For a 4×6 reference with `ref[y,x] = 10(x+1)+y`:

| Sample | Zero outside | Replicate | Delivered emulator |
| --- | ---: | ---: | ---: |
| `x = −0.25, y = 1` | 8.25 | 11.00 | 18.50 |
| `x = 2, y = H−1+0.25` | 24.75 | 33.00 | 32.25 |

Does silicon use zero, replication, or index clamping with the raw fraction? Is it the same on all four sides?

**Answer.** Sampling saturates at the boundary: the complete coordinate is clamped, without retaining an excess fraction. With maximum coordinate 319, `320.19 → 319`, not `319.19`. At a current point of 319, the resulting motion is zero. This corresponds to edge replication at that boundary, rather than the delivered emulator’s raw-fraction behavior.

The grid uses relative motion. The HW explanation uses a top-left coordinate convention under which motion is positive. Exact handling of negative input words and symmetry across all four sides were not separately established.

## 2. Identity flow: last row and column

**Original question.** With zero motion, the emulator returns row `H−2` instead of `H−1`, and column `W−2` instead of `W−1` when width is a multiple of 32. For example, a last row `[24,27,30,33]` is replaced by the preceding row `[12,15,18,21]`. Is this an emulator bug, with silicon preserving the reference under zero motion?

**Answer.** Yes. The emulator runs to `W−2, H−2` but should include `W−1, H−1`. The last column and row should be copied twice at the boundary to supply interpolation corners. Zero motion must preserve the reference.

## 3. Relative grid, fraction bits, motion limits, and offsets

**Original question.** The emulator quantizes absolute positions with per-call, data-dependent fraction counts:

`frac = 14 − floor(log2(ceil(max|pos|)))`

QA Q9 and `infer_warp` instead use relative motion, fraction counts derived from `node.warp.limit` and capped at 8, motion clamping, and `reference_offset`. ALG uses relative Q8.8. Measurements showed that `infer_warp` supplied with ALG’s Q8.8 words at limit 127 matched ALG inside the frame, while adaptive absolute quantization produced different precision. `generate_absolut_grid` also encoded a 3-pixel motion with 13 fraction bits while `infer_warp` decoded with 8.

Does the grid contain motion or position? Are fraction counts fixed or data-dependent? What is the legal range? Are motion clamping and `reference_offset` hardware features? Is Q8.8 legal?

**Answer.** The grid contains relative motion, encoded as int16 fixed point by the preceding layer. It can use any Qm.k format with `0 ≤ k ≤ 8`; Q8.8 is one supported format, not the only format. X and Y may use different fraction counts. The warp consumes the supplied format rather than deriving precision from each frame’s data.

Hardware has independent maximum motion limits for X and Y and saturates to them. For example, a limit of 319 maps 320.23 to 319.

**Open.** The mechanism supplying the configured fraction counts and motion limits, and the meaning/support of `reference_offset`, were not specified. Relative Q8.8 encoding is supported; the original question’s particular limit-127 configuration was not separately confirmed.

## 4. Combined fraction count above 14

**Original question.** `trc_warp` (`hw_pipe.py:257–261`) and `build_warp_c_struct_arr` (`calib.py:116–120`) reject `frx+fry > 14`, while the active emulator does not. Adaptive quantization reaches sums from 15 to 24 in several frame sizes. Is 14 a hard hardware limit, and does 8+8 fit?

**Answer.** The 14-bit restriction was added because of a signedness assumption. The fraction/weight field is unsigned; the confirmed combined fraction count is up to 16, supporting 8 fraction bits on each axis.

**Open.** The exact full-weight endpoint requires clarification: `2^16 = 65536` does not fit directly in a plain unsigned 16-bit field. Its hardware encoding or special handling was not established in this review.

## 5. Rounding ties in the grid and reference

**Original question.** The emulator uses `torch.round` (ties to even) for grid conversion (`layers.py:346`) and float references (`layers.py:318`). ALG rounds ties away from zero. For example, at 8 fraction bits, `+0.5/256` maps to zero versus `+1/256`; reference 2.5 maps to 2 versus 3. Which rule builds the grid word, and is the reference already integer?

**Answer.** The warp receives fixed-point grid words from the preceding layer. Fraction bits are part of that input representation. Rounding belongs to the producing layer; when the producer passes through the PCU, PCU rounding applies. Standard PCU rounding was confirmed in question 7: nearest, with ties toward positive infinity.

The pipeline reference is integer-coded. Converting a float example into reference codes is a software input-preparation operation; its tie rule was not separately settled.

## 6. Reference format

**Original question.** `pack_warp_reference` (`layers.py:314–318`) clamps all references to `[0,255]`, discarding negative values. ALG uses int8 with a free real step; DOF features can be signed. Does silicon support signed int8, unsigned uint8, or both per layer?

**Answer.** The unconditional unsigned clamp is a mistake. Clamp according to the selected type:

- int8: `[-128,127]`.
- uint8: `[0,255]`.
- Other supplied types: default to int8 unless explicitly specified otherwise.

The MX container preserves the integer interface format; it does not imply shared-exponent quantization for the warp inputs.

**Open.** The original reference-step question was not separately answered; output rescaling is addressed below.

## 7. Output scale, clamp, and rounding

**Original question.** `MXWarp_Loops.forward` bypasses the PCU (`o_line = p_sum`, `hw_pipe.py:878–881`), and `CalibratedMXWarp.forward` returns a float divided by `2^(frx+fry)` (`layers.py:513`). The intended calibration describes PCU shifting and rounding with a 24-bit clamp. QA Q9 says output step equals reference step; ALG instead uses an independently chosen output step, int8 saturation, and ties away from zero.

Must output step equal reference step, or can scaling support an independent step? Is the clamp int8 or 24-bit? Does standard PCU rounding apply, or does another scaling unit process warp output?

**Answer.** The PCU bypass and equal-scale behavior are simplifications of the single-layer example. ALG should select output scaling through PCU rescaling. Equal input/output steps are not a general warp requirement.

Output defaults to int8 with `[-128,127]` saturation unless explicitly configured otherwise. Warp uses standard PCU rounding: nearest, with ties toward positive infinity. Thus `37.5 → 38` and `−37.5 → −37`.

**Open.** The precise programmable multiplier format and its relationship to the separately mentioned Q10 scaling unit were not detailed.

## 8. Batch

**Original question.** `pack_warp_grid` reads `coords[0]` and `trc` reads `grid[0]`, so multiple reference batch elements would share batch 0’s grid. Should silicon take one grid per batch element through a compiler loop?

**Answer.** In `grid[0,0,...]` and `grid[0,1,...]`, the second index selects X/Y and the first selects the frame. Intended usage is one reference frame and one grid producing one output frame. Native multi-batch support is not established. Multiple frames could be supported through separate invocations; that extension is not required now.

The observed batch behavior is therefore outside the intended single-frame interface rather than evidence of a required native-batch feature.

## 9. Float precision effects in the emulator

**Original question.** The output allocation defaults to float32 (`hw_pipe.py:846`), losing integer accumulator precision above `2^24`. Weight construction also uses floating-point fraction-count tensors, and BCU casts only after multiplication (`hw_pipe.py:349`), allowing product rounding. Are these emulator stand-ins, with silicon using integer arithmetic?

**Answer/disposition.** These are emulator implementation issues to fix. Although the values represent integers, operations can still execute in floating point. Keep fraction counts, weight construction, products, and accumulator storage in sufficiently wide integer types through the PCU. Casting after a floating-point product does not recover lost precision.

No additional hardware accumulator width or overflow behavior was specified during this exchange.

## 10. Grid size and width padding

**Original question.** The emulator allows different grid/reference sizes, such as reference `240×320` and grid `80×106`. It pads reference width to a multiple of 32, causing right-edge sampling to read zero padding for some widths. For constant 100 at `x = W−1+0.5`, the delivered emulator gives 50 for width 40 and 100 for width 32. Does padding affect hardware results, and are different dimensions supported?

**Answer.** Warp has no padding. The internal padding introduced by the emulator must not alter sampling behavior. Boundary handling follows the actual reference dimensions.

Different reference/grid dimensions are not important for the current use case. Do not add support if it requires effort; use matching spatial dimensions for now.

## 11. Authoritative hardware reference

**Original question.** The delivered emulator, notebook `golden_int`, and `infer_warp` differ in borders, coordinates, precision, and output processing. Which is the hardware specification?

**Answer.** The emulator using the **BCU → PCU structure**, after correcting border handling, should be the reference. The agreed emulator mistakes identified in this review should also be corrected. `warp_infer` is a float version and does not mimic hardware.

## 12. Separate topic: shared-exponent float32 slip

**Original question (not warp).** `mx_format` and ALG’s microxcaling library compute shared exponent using `floor(torch.log2(amax))` in float32. Just below a power of two, rounding can select an exponent one too high. For `amax = 2^22−1`, the calculation can return 22 rather than the exact 21. Does silicon use an exact leading-one calculation?

**Answer/disposition.** This was not answered as a general MX-format question. It does not apply to the warp’s fixed-point interface: grid words are int16, references default to int8, and output defaults to int8. MX is used to preserve the container format here. The general shared-exponent question remains separate and open.

## Agreed execution scope

Apply the emulator fixes and run the current example. Do not add verification tests or run broader verification. Do not extend native batching or support different reference/grid dimensions as part of this work.
