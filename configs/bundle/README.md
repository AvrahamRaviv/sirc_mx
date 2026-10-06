# MX single-layer bundle

Two frozen single-conv cases. Each directory is self-contained: the weights, the
input, our MX output, and the integer intermediates we got on the way. Nothing
here needs PyTorch or the MX lib to read — every tensor is both a `.npy` and a
flat `.txt` (one value per line, C order, shape in the header comment).

The ask: implement the same conv in your own environment (torch / C / Excel),
run it on `input` + `weight`, and compare against our integer taps. Matching
`alg_a_int` / `alg_a_exp` / `alg_w_int` / `alg_w_exp` means we agree on the MX
quantizer; `alg_out` then follows from them.

## The two cases

| case | layer | why |
|---|---|---|
| `conv1x1_ramp` | 1×1, Cin=Cout=32, no bias | Weights and input are the integers 1..32, so one block covers exactly 32 values and every output is Σ i² = **11440**, exactly. Nothing rounds, nothing underflows. Start here — if this does not match, the mismatch is in the quantizer, not in the arithmetic. |
| `conv3x3_randn` | 3×3, Cin=Cout=32, no padding, no bias | Random data, so real rounding and real underflow. The realistic check once the first one matches. |

## Format

MXINT8, block size 32, shared exponent from the block max:

```
value      = mantissa * 2**(shared_exp - 6)        # mant_bias = 6
shared_exp = floor(log2(max(|x|) over the block))
mantissa   = round(value * 2**(6 - shared_exp)),  clamped to [-127, 127]
```

Blocks run along dim 1 (channels), 32 elements each. Both cases have Cin=32, so
there is exactly one block per position and no tail — tail handling is a
separate question and a separate case (`tail_block_c33`), deliberately not in
this first bundle.

`manifest.json` in each directory repeats all of this, plus the layer geometry
and the full config.

## Files

| file | what |
|---|---|
| `weight` | the checkpoint, `[Cout, Cin, kH, kW]` |
| `input` | the test vector, `[N, C, H, W]` |
| `ref_fp32` | unquantised conv output. Context only — **neither** side should match this; it is what quantization moves away from. |
| `alg_out` | our MX output. The float comparison. |
| `alg_conv_a_int`, `alg_conv_a_exp` | input after MX quantization: int8 mantissas, and the shared exponent per block |
| `alg_conv_w_int`, `alg_conv_w_exp` | same for the weights |
| `alg_conv_out` | the layer output (same as `alg_out` on a single-layer case) |

`a_exp` / `w_exp` have the blocked axis reduced to the block count: for
`[1, 32, 2, 2]` with block size 32 that is `[1, 1, 2, 2]`, one exponent per
spatial position.

## Reproducing our side

```bash
git clone <repo> && cd sirc_mx/configs
python main_ALG.py -m conv1x1_ramp        # prints the taps and the SQNR
python export_vectors.py -m conv1x1_ramp  # rewrites this bundle
```

## Comparing

Integer taps are compared **exactly** — one differing element is a failure.
`alg_out` is compared by max abs delta and SQNR. If you produce the taps in the
same layout, `main.py` does the comparison and reports the first stage that
diverges; otherwise a text diff on the `.txt` files is enough to start.
