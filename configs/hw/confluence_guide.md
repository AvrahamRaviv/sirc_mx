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


#	Topic	Algorithm model	Emulator	What the test showed	Question	HW Answer	


	1	Block exponent	floor(log2 max)	ceil(log2 max)	Block with max 0.43: floor stores 110 (step 1/256), ceil stores 55 (step 1/128). With ceil the mantissa never exceeds 64, so half of the int8 range is never used: one bit of precision lost on every block	Your format defines the mantissa as [−2, 2) (1 integer bit + 6 fraction bits, `mantissa_offset = 6`). Floor fills that range; ceil never uses the integer bit. Is ceil intended? (The code comment says "I think it should be ceil but check that".)	FIXED - floor(log₂ amax) (HW). Block-max mantissa fills [64,127] — integer bit is real;

	2	Mantissa rounding	round to nearest	truncate toward zero	Values shrink toward zero on average (mean error −0.019 vs −0.0002)	Truncate or round to nearest? If round, are ties rounded away from zero or to even?	Truncate toward zero — magnitude dropped, sign preserved; sub-LSB negatives → 0, never −1 (HW)

	3	Input scaling	quantized once	every layer first rounds `x · 2^cal` to an integer, then MX-quantizes it	Values are rounded twice: in a block with step 4, 7.4 → 4 but 7.6 → 8, a cutoff of 7.5 (neither truncation nor rounding). The smallest block gets round-to-nearest; larger blocks get this mix. When both sides are given the same integer input, the emulator–model gap drops by about a third	The NPE only receives integers, so some rounding step is real. Does it happen here, on every layer, with the grid picked by this layer's observer? Or only on the sensor input and in the previous layer's requantizer? If the latter, the `round()` in `CalibratedMXConv2d.forward` should apply only when `is_input_layer=True`	Once, at the first layer only (x·2^cal → round). Intermediate layers: no pre-round — PCU drains integers, shift_right (e_min sum) realigns scale; MX quantizer's truncate is the integer-ization. Motivation: BCU shifters unsigned/left-only; PCU shifter bidirectional (HW architect, this session)

	4	Exponent register	signed exponent in a fixed range around 0 (8 bits today; −7..7 with 4 bits)	4-bit unsigned right shift (0..15) on the integer tensor `x · 2^cal`; the per-tensor `cal` moves the range; out-of-range values wrap (`& 0xF`)	(a) Low end: a block [20, −7, 3] next to a block with max 1000 needs a negative shift. The wrap turns it into shift 15 and stores [0, 0, 0]. Clamping to 0 would store it exactly, because small integers already fit in int8. (b) High end: a block max above 2^21 needs shift 16, which wraps to 0, so 2,097,157 reads back as 5. (c) A 24-bit signed word needs shifts 0..16 (23 magnitude bits − 7 mantissa bits): 17 values, while 4 bits give 16	(a) When a block would need a negative shift, does the silicon clamp it to 0 (lossless for integer input), or wrap? (b) Is the top bit of the 24-bit PCU window ever used? If yes, a 4-bit exponent can't reach it: it needs 5 bits, or a 23-bit window. (c) Is `cal` one value per tensor, fixed at PTQ?	(a) wrap-only on negative (unsigned 4-bit, left-only) — calibration must prevent them; (b) 24 bits = 23 magnitude + sign; (c) static per-tensor, fixed from offline PTQ

	5	Accumulator	Fixed point	float32	--	As discussed, the accumulator is fixed point. (Does float32 give exactly the same result when the inputs are integers?) Either way, please switch the emulator to a fixed-point accumulator.	FIXED - int64 fixed-point

	6	Saturation	48 bits, clamp after each add and continue	46 bits; once the limit is hit, later products are ignored	+100, +100, −100 in an 8-bit range gives 27 vs 127	Which width, and which behavior: freeze, or clamp and continue?	FIXED - Two's-complement wrap-and-continue at acc_bit_width ∈ {46,48,54,56}, default 48 (replaces old freeze+clamp)

	7	Output rounding (PCU)	—	set to "round", but the code floors before rounding, so the result is a floor	2.75 → 2	Floor, round half up, or round half to even?	FIXED - Round half up (ties toward +∞, incl. −1.5→−1). Caveat: if silicon means away-from-zero, negative ties flip

	8	Bias	round(b · 2^20)	floor(b · 2^(x_cal + w_cal))	Small error (6e-6 vs 3e-7)	Floor or round when converting the bias to accumulator units?	Offline; SW choice round half up floor(v+0.5), .long() into units 2^(x_cal+w_cal)

	9	Weight calibration	—	the observer splits the weights into blocks differently from the quantizer	w_cal is 11 instead of 9; 2 bits of exponent range are wasted. Harmless for now	Should the observer use the same flattened blocks as the quantizer?	FIXED  - Observer must block weights per flattened kernel (per OC, C·Kh·Kw), mirroring HW consumption order

	10	`mx_dequantizer`	—	result is off by 2^-6 from the values the pipeline actually uses	The helper only; the pipeline itself is consistent	Is a fix planned?	shift = shared_exp − calibrate_offset (old code subtracted m_off twice → ×2⁻⁶); ndim-keyed trim fixes conv-weight reshape crash; conv weights stack [OC,blocks,BS], linear [1,OC,blocks,BS] — both block per OC row

Implemented layers:
layer	status
conv2d	v
warp	x
transpose conv	x
skip connection	x
