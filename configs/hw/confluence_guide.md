1. confluence guide
Notice:

you can find a jupyter notebook example here:

http://gitlab-srv/dl-arch/mx-emo/-/blob/main/EL_nb/mx_conv_npe_emu_example.ipynb?ref_type=heads

Introduction




This document explain how to emulates the MX (microscaling) number format as implemented on the NPE (Neural Processing Engine). The goal is a bit-accurate / close to bit-accurate Python model of the hardware pipeline so that quantized networks can be evaluated and calibrated before deployment.

Hardware pipeline




The base compute unit is the BCU (Basic Calculate Unit, dot-product unit):

The input is delivered as lines; each line is broadcast over many BCUs, each multiply-accumulating one element per cycle.
In the Python model we don't emulate cycle-by-cycle broadcast; we reorder the input into the same computation order (activation patches aligned with flattened kernels) and preserve only the accumulation order, which is what the precision behavior depends on.




MX format support is layered on top:

The shared exponent is broadcast per block (every mantissa in a block carries the same exponent).
After the mantissa multiply, the product is shifted by the sum of the two exponents (activation + weight).
Exponent shifts are unsigned (shift-left only; 4-bit register on the e4bm8 format).
The accumulator drains into the PCU, which (1) adds the bias (pre-scaled into accumulator units 2**(x_cal + w_cal)), (2) shifts by e_min_layer (= sum of operand e_min_layers) plus an optional static shift, (3) rounds, and (4) clamps to the activation integer range — producing the default-scaled 2**(2·bias_exponent) output that either feeds the next MX layer or the static-quant path.
The PCU output feeds the next layer directly, passing through a (re)quantizer per layer.
The hardware already supports static quantization; enabling the full static-quant flow is future work.

Structural constraints and how the software model handles them
No negative exponents. The BCU can only shift left, so a negative exponent would be invalid. All inputs/weights must be pre-scaled so that every block's exponent is ≥ 0. The NPE supports per-tensor calibration only — one calibrate offset for the whole tensor — so the global max is not sufficient: MXObserver anchors the smallest non-zero block amax at exponent 0 (calibrate_offset = mantissa_offset − ceil(log2(a_min))). If the tensor's dynamic range exceeds the exponent window, the observer raises loudly instead of silently wrapping (the alternative is invisible: entire blocks annihilate to zero, output collapses to bias-only). Supported resolutions at that point: pick a wider-exponent format or re-scale upstream layers.
The first (sensor) layer is special. Input data is integer-only with a zero shared exponent, and by default is scaled by mantissa_bits − 1 reserve for the sign (observer mode="input" uses calibrate = mantissa_offset − ceil(log2(a_max)), e.g. 2**7 for data in [-0.5, 0.5)). A dedicated input_mx_quantizer models this path.
Inter-layer alignment via e_min_layer and the PCU shift. The mantissa spans [-2, 2), i.e. it carries mantissa_bits − sign_bit − 1 = mantissa_offset fractional bits (the value we also call the bias exponent of the format). Each operand carries e_min_layer = mantissa_offset − calibrate_offset, and the PCU shifts the accumulator by the sum of the two operands' e_min_layer before rounding/clamping. This is what realigns layers whose calibrated scales differ (e.g. input layer vs. intermediate MX layers).
Default output scale = 2**(2 · bias_exponent). Because both the activation and weight mantissas carry bias_exponent fractional bits, the accumulator arrives scaled by 2**(2·bias_exponent), and the PCU's e_min_layer shift converts exactly between the operands' calibrated scale and this base scale. In the default/simple case bias_exponent = 6, so the layer's raw integer output is in units of 2**12 (i.e. divide by 4096 to get float). This fixed, format-defined output scale is what makes static quantization tractable: downstream layers (and their observers) always know the scale of the incoming tensor without runtime metadata.
# TODO- handling of 16-bit input
Multi-format 

Naming: e{#exponent_bits}b{#mantissa_bits} — e4bm8, e8bm8, e6bm8. Mantissa is always 8-bit for now (the int8 sign-masking path raises otherwise, unchanged).

#TODO One real code change required — in base_mx_quantizer the wrap mask is hardcoded:

preserve_lsb = 0xf          # today: 4-bit only

preserve_lsb = 2**exp_bits - 1   # generalized: e4->0xF, e8->0xFF, e6->0x3F







Example:

lets take a case of 2 elements per block:

[0.4299, 0.1382]






Path ONE — v as input layer activation (input_mx_quantizer, observer mode="input")

(data=[55, 17], shared_exp=0, e_min_layer=-1, calibrate_offset=7)

stage	rule	value
observe	c = 6 − ceil(log2(a_max)) = 6 − ceil(log2(0.4299))	c = 7
scale (fixed, chosen)	x_int = (v · 2^7).round()	[55.027, 17.690] → [55, 18]`
block stage	none — no log2, no amax, no divisor	—
shared exponent	structural, always	0
store	int() + int8 sign restore	data = [55, 18]
e_min_layer	6 − 7 (for PCU alignment only)	−1
dequant check	data · 2^(0 − 7)	[0.4296875, **0.140625**]




Path TWO — v as intermediate activation (base_mx_quantizer, observer mode="mx")




stage	rule	value
observe	c = 6 − ceil(log2(a_min)); one block → a_min = 0.4299	c = 7
scale (pre-factor only)	v · 2^7, kept float (no rounding here)	[55.0272, 17.6896]
per-element log2 (ceil)	ceil(log2(55.03)), ceil(log2(17.69))	[6, 5]
shared exponent	E = amax(-1) → 6; divisor = 2^(6−6) — this is the scale	E = 6, divisor 2^0 = 1
store mantissas	int(x_scaled / divisor) → truncate	data = [55, 17]
stored exp	E − mantissa_offset	0 (register-legal ✔)
e_min_layer	6 − 7	−1
dequant check	data · 2^(exp − 7)	[0.4296875, **0.1328125**]




The motivation for using observer:

Here's v = [0.04299, 0.01382] (intermediate path) without observer guidance — three realistic "no observer" ways to pick c, versus the observer:

scenario	c	block E (log2 amax of scaled)	raw exp	& 0xF wrap	divisor	mantissas	result
default, no scaling	0	−4	−10	6	2^6 = 64	[0, 0]	zero
old hardcoded default	6	2	−4	12	2^12 = 4096	[0, 0]	zero
one bit short of anchor	9	5	−1	15	2^15 = 32768	[0, 0]	zero
observer (a_min anchor)	10	6	0	0	1	[44, 14]	✔ [0.04297, 0.01367]




What happens, stage by stage (e.g. the c=6 row)




scale: v · 2^6 = [2.7514, 0.8845]
per-element log2 ceil: [2, 0] → E = 2
stored exp: 2 − 6 = −4 → unsigned register wraps: −4 & 0xF = 12
divisor: 2^12 = 4096
mantissas: int([2.7514, 0.8845] / 4096) = [0, 0] — the whole block is annihilated




Why this failure is catastrophic, not just inaccurate




Cliff, not slope. The moment the exponent needs to go negative, you don't get "a bit less precision" — the divisor becomes 2^(wrapped) and every value in the block floors to 0. c=9 was 100× more calibrated than c=0 and dies exactly the same way. The only safe region is raw exp ≥ 0.
Silent. shared_exp = 6/12/15 are perfectly legal register values; the MXTensor looks valid. No exception, no NaN — the data is just gone.
Cascade. Zero mantissas → p_sum = 0 in the BCU → PCU output = bias only → the entire channel emits one constant. That's the 424-plane you pasted from your run, and the reason the whole debugging session existed.
Bonus corruption downstream. e_min_layer is garbage too (6 instead of −4 for the real tensor), so even the bias/scale bookkeeping in the PCU misaligns the output scale.




The observer exists precisely to make the first constraint a guarantee checked at calibration time (exp ≥ 0 for the smallest block + window check ≤ 15 for the largest), rather than a coin flip per layer.




Implemented layers:
layer	status
conv2d	v
warp	x
transpose conv	x
skip connection	x
