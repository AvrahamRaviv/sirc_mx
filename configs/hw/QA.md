# NPE MX Emulator — Q&A

Answers to the single-op simulator team's questions after they integrated the updated
`npe_mx_emu` (/home/eliyahul/DL_ARCH/mx-emo). Collected in a 12-round Q&A session.

Legend: **Answer** = what to tell the sim team; **Emulator impact** = what changes (or
stays) in the code.

---

## Part 1 — New questions (their latest review)

### Q1 — Mantissa truncation cost

**Question:** Truncating doubles the quantization error per element and makes the 3×3
conv error against float 2.6× larger (0.025 → 0.065). Is truncation final in silicon, or
could the quantizer round to nearest (round half up, like the PCU)?

**Answer:** **Final for this silicon.** The narrowing happens in **PCU block mode** (not
the BCU): mantissas are aligned to the block-max exponent by a **right shift**, and the
shifted-out bits are simply dropped — truncation toward zero. Rounding is not forced by
the shift itself (add ½ LSB pre-shift — the PCU shift path already does round half up),
but the block-mode emit path does not include that logic.

**Emulator impact:** Keep `data.int()` at `npe_mx_emu/mx_format.py:105` (and `:161` for
the first layer). The remaining error vs. float is inherent.

### Q2 — Top bit of the PCU output window

**Question:** The window is 23 magnitude bits + sign. Values above 2²² need block shift
16, but a 4-bit exponent holds 0..15, so shift 16 wraps to 0. Can the top magnitude bit
ever be used with a 4-bit exponent? If not, should the clamp be ±(2²² − 1)?

**Answer:**
1. 4-bit, wrap-only, no saturation — **confirmed**.
2. Values ≥ 2²² are **impossible by construction** — PTQ bounds the accumulator. There is
   **no clamp** in the PCU for this; do not add one. (Max encodable is 127·2¹⁵ = 2²² − 2¹⁵.)

**Emulator impact:** Treat ≥ 2²² as unreachable; **raise loudly** if observed (mirrors Q6 policy).

### Q3 — Static shift

**Question:** The output grid is fixed at 2⁻¹²; the guide mentions an extra "static shift"
in the future flow. One value per layer? Set at calibration? How many bits? Which direction?

**Answer:**
1. **Per-tensor** only, currently.
2. Set at **calibration**; the calibration flow must support **both standard (affine) and
   POT quantization**.
3. Width = log₂(accumulator bits + 8) → with the 48-bit accumulator: **7-bit signed**
   (6-bit magnitude up to 56 + sign). Will be tightened in a future revision.
4. **Bidirectional** — left and right (matches the bidirectional PCU shifter).
5. **Exists in the current flow** — not future-only.

**Emulator impact:** Already structurally present — it is the `c_struct.shift` argument of
`pcu_div` (`npe_mx_emu/hw_pipe.py:234`). Can be exercised now.

### Q4 — PCU rounding ties

**Question:** Round-half-up was noted with a caveat; away-from-zero would flip negative
ties (−1.5 → −2 instead of −1). Can you confirm silicon rounds ties toward +∞?

**Answer:** **Confirmed — ties toward +∞** (add ½, then arithmetic right shift; two's
complement gives ties toward +∞ on negatives for free: (−3+1)>>1 = −1). Applies to both
signs: 2.5 → 3, −1.5 → −1. The "nearest-absolute round" naming in the docs is loose.

**Emulator impact:** `pcu_div(round_type="round")` (`hw_pipe.py:209`) is already correct.

### Q5 — First-layer format

**Question:** The input layer is integer-only, zero shared exponent, cal = 6 − ⌈log₂amax⌉;
small values lose precision (at cal 7: 0.01 → 0.0078, −0.003 → 0). Is per-tensor int8 the
final plan, or can the first layer use block exponents?

**Answer:** **Question dissolved by reframing.** The sensor input **arrives as integers** -
that is the native data; the float domain exists only training-side as a SW mimic. So
per-tensor integer is final — block exponents are neither needed nor planned at layer 1.
The input scale is a **free SW calibration choice, explicitly not restricted to powers of
two**; the cal-7 / 2⁷ case was just the simplest toy scale. A real calibration maps the
sensor's integer range onto the int window, so the observed crushing does not occur in the
deployed flow.

**Emulator impact:** `input_mx_quantizer`'s POT path is a valid special case; no rework.

### Q6 — Calibration margin vs. exponent wrap

**Question:** The exponent register wraps instead of saturating, so an input outside the
calibration range corrupts a whole block. Does the compiler add a margin over the
calibration min/max? How many bits?

**Answer:**
1. Yes — **PTQ must guarantee** values stay inside the representable window.
2. **No explicit margin bits** are added.
3. Not a fixed number: the guarantee is delivered by the **choice of exponent width**
   (e4/e6/e8) plus activation/weight calibration — a restriction on the exponent.

**Emulator impact:** Keep the raise-loudly behavior on out-of-window ranges; faithful wrap
emulation is unnecessary because exceeding the window is a PTQ contract violation.
*(Emulator-side inference — flag if it should instead emulate the wrap.)*

---

## Part 2 — Carried-over questions

### Q7 — Transposed conv

**Questions:** How is convT lowered? How are the weights blocked? Is the input quantized
before or after the zero insert? How is the output interleaved?

**Answers:**
1. **Lowering: confirmed.** Compact input in memory; zeros fed on the fly inside the pipe
   (input-side dilation); then an ordinary conv, with zero-skip skipping wasted cycles.
2. **Weight blocking:** per-filter, flattened **C-major**, exactly **as HW receives them**.
   (Zero-padding the filter in the emulator is a simplification of zero-skip.)
3. **Quantize the compact tensor first.** The inserted zeros are pipe-generated, never
   quantized — they must not be part of any block. (Their ALG side currently inserts zeros
   *before* quantizing → wrong block packing → different shared exponents. **Fix:** quantize
   compact, dilate after.)
4. **Output interleaving: OPEN.** The question, with example, is at silicon: with input
   `[a, b]`, kernel `[k0 k1 k2]`, 2× upsampling, does memory get
   **(A) raster** `[a·k0, a·k1, a·k2+b·k0, b·k1, b·k2]` or
   **(B) phase-chunked** `[a·k0, a·k2+b·k0, b·k2 | a·k1, b·k1]`?
   (2D: 4 phases — 4 concatenated small images vs. one interleaved image.)

### Q8 — Depthwise conv

**Questions:** MAC array or dedicated unit? Which data type? How are filters shorter than
a block (16 values vs 32) blocked?

**Answers:**
1. **Same MAC grid.** Each channel is processed as its own kernel.
2. **Same data type** — MX e4bm8 integer path.
3. **"Pad" option for now:** one block (32) per filter, zero-filled; a later revision may
   pack multiple channel filters per block.

**Emulator impact:** **DONE** — `CalibratedMXDepthwiseConv2d` (`npe_mx_emu/layers.py`) +
`MXDepthwise2D_Loops` + dedicated TRU `tru_a_dw` (`npe_mx_emu/hw_pipe.py`), exported
from the package. Verified against float depthwise (`F.conv2d(groups=C)`): max abs err
≤ 0.045 (within the e4bm8 budget — sim team's K3 conv benchmark is 0.065); channel
independence exact; one padded 32-block per filter confirmed (weight shape (C, 1, 32)).

### Q9 — Warp

**Questions:** How is the static scale chosen? Is the int8 signed?

**Answer:** **OPEN — the layer is still being defined.** Working hypothesis
(`warp_m/README.md`): the reference int8 step is a **free calibration pick (non-POT
allowed)**; the warp output step is **forced = ref step** so post-warp requant is
**shift-only** (2^(flx+fly), no multiplier); the **grid** is int16 **signed** with POT step
(`frac_bits = min(8, 16 − (2 + ⌊log₂ pixel_limit⌋))` per axis). ref/out int8 signedness TBD.

**Emulator impact:** Defer all warp modeling decisions until the layer definition lands.

### Q10 — PCU scale width and activation

**Questions:** What is the bit width of `scale`? Is ReLU only the clamp?

**Answers:**
1. **PCU scale = int8** (signed −128..127 assumed; binary point placed by its companion
   shift field). The 16-bit scaler mentioned in the warp flow is a **different unit's
   special case** — to be discussed when warp work resumes.
2. **Activations are decomposed into PLA stages — not limited to ReLU/ReLU6.**
   The PCU's PLA divides the accumulator into **bins** (per-bin **boundary**
   comparisons); each bin applies its own **scale** (the int8 S from item 1) and
   **bias**; then the **ReLUX step clamps/truncates** to the output format
   (negatives → 0 and/or clamp at the per-bin truncate value, per configuration).
   ReLU and ReLU6 are just degenerate configurations (one slope + clamps).
   **PReLU / leaky ReLU, and even sigmoid, are realized the same way** — sigmoid as
   a multi-bin piecewise-linear approximation, with the clamp step holding the two
   saturated ends (≈0 and ≈1).

**Emulator impact:** `relux()` (`hw_pipe.py:217`) covers exactly the final
clamp/truncate step — already exact. `pcu_pla` currently models the degenerate
single-bin case (scale=1); when a non-ReLU-family activation is needed, extend
`Cstruct`/`pcu_pla` with per-bin **boundaries + int8 scales + biases** (sigmoid =
N-segment PLA approximation). Exercise `Cstruct.scale` (currently hard-set to 1 at
`npe_mx_emu/calib.py:76`) as int8.

### Q11 — Padding

**Question:** Are the zeros inserted before or after MX quantization?

**Answers:**
1. **After, in the pipe.** Border zeros are fed on the fly (like convT dilation zeros) —
   never stored, never quantized.
2. **Zero-mode only** for now — mirror/edge-extend are too HW-costly.

**Emulator impact:** **TODO** — implement border handling inside `tru_a` and remove the
`NotImplementedError("Padding not yet supported")` at `hw_pipe.py:291-292`.

### Q12 — Emulator timeline (warp, convT, skip connection)

**Question:** When will warp, transposed conv and skip connection be added to the emulator?

**Answer:** **Dateless, dependency-ordered** queue:

| Order | Item | Effort | Blocked on |
|---|---|---|---|
| 1 | Zero-padding in TRU | Small | Nothing (Q11 answered) |
| 2 | Depthwise conv | Medium | Nothing (Q8 answered) |
| 3 | ConvT | Medium | Q7-4 silicon answer |
| 4 | Warp | Large | Q9 layer definition |
| 5 | Static-shift usage + int8 PCU scale modeling | Small | Nothing (Q3, Q10 answered) |

**Premise correction:** the skip-connection **add is already implemented** —
`CalibratedMXAdd` (`npe_mx_emu/layers.py`) + `MXAdd_Loops` (`npe_mx_emu/hw_pipe.py`),
incl. the per-channel scale hook.

---

## Open items (waiting on HW/silicon)

- **Q7-4** — convT output memory layout: raster vs. phase-chunked (example delivered).
- **Q9** — warp layer definition, incl. ref/out int8 signedness.
- Minor: Q10 int8 scale assumed **signed** (flag if unsigned).

## Implementation queue (post-Plan-mode)

1. ~~`CalibratedMXDepthwiseConv2d` + `MXDepthwise2D_Loops`~~ **DONE** — channel-as-kernel,
   pad-blocks. Verification: `/tmp/opencode/test_dw_emu.py` (torch venv:
   `/tmp/opencode/venv`) — abs err ≤ 0.045, channel independence exact, block layout OK.
2. Zero padding in `tru_a`; drop `NotImplementedError` at `hw_pipe.py:291-292`.
   Schedule: `plan/SCHEDULE_item2_zero_padding.md`.
   **Note:** the same guard now also exists in `MXDepthwise2D_Loops.forward` — drop both.
3. int8 `Cstruct.scale` + static-shift exercise in calibration.
4. ConvT (compact-quantize + pipe-fed zeros; interleave after Q7-4).
5. Sim-side fix reminder for the algo team: convT must quantize the compact tensor
   *before* zero-insert, not after.
