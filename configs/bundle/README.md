# MX single-layer bundle

Two frozen single-conv cases in the **NPE geometry** (activations blocked along
W, weights flattened per filter). Each directory is self-contained: the
parameters, the input, our output, and the integer intermediates we got on the
way. Nothing here needs PyTorch or the MX lib to read — every tensor is both a
`.npy` and a flat `.txt` (one value per line, C order, shape in the header
comment).

The ask: implement the same conv in your own environment (torch / C / Excel),
run it on `input` + `param_conv_weight`, and compare against our integer taps
in pipeline order. Matching `a_int` / `a_exp` / `w_int` / `w_exp` means we agree
on the MX quantizer; matching `acc` / `sat` means we agree on the accumulator.

Every tap here was checked before export: an independent integer reference
(`tap_check.py`) rebuilds `acc` from the four operand taps and must agree with
our accumulator bit for bit.

## The two cases

| case | layer | why |
|---|---|---|
| `conv1x1_ramp` | 1×1, Cin=Cout=32, no bias, input `[1,32,2,2]` | Weights are 1..32 along each filter, the input is 1..32 along channels (constant along W), so every output is Σ i² = **11440** exactly: `acc = 11440 · 2^20`. Nothing rounds, nothing underflows. Start here. |
| `conv3x3_randn` | 3×3, Cin=Cout=32, no padding, no bias, input `[1,32,5,5]` | Random data, so real rounding and real underflow. The realistic check once the first one matches. |

## Format

MXINT8, block size 32, 8-bit unsigned exponent field:

```
E          = floor(log2(max |x| over the block)), clamped to [-127, 127]
code       = E + 127                              # what a_exp / w_exp hold
mantissa   = round(x * 2**(6 - E)), ties away from zero, clamped to [-127, 127]
value      = mantissa * 2**(code - 127 - 6)       # mant_bias = 6
zero block = code 0
```

Blocking (NPE):

- **activations**: 32 along W, per (n, c, h). A row shorter than 32 is one
  short block. `a_exp` is `[N, C, H, ceil(W/32)]` — here `[1, 32, 2, 1]` and
  `[1, 32, 5, 1]`. Conv zero padding is applied *after* quantization; pad zeros
  join no block.
- **weights**: each filter `[Cin, kH, kW]` flattened C-major (c, kh, kw) and
  cut into blocks of 32 along that stream. `w_exp` is `[O, ceil(Cin·kH·kW/32)]`
  — here `[32, 1]` and `[32, 9]`. `w_int` stays `[O, Cin, kH, kW]`;
  `w_int.reshape(O, -1)` is the stream.

## Accumulator

48 bits, `e_layer_min = -20`, `sat_mode = per_product`:

```
for c in Cin, for (kh, kw) in kernel:          # the flattened filter order
    p     = a_int * w_int                      # exact
    shift = Ea + Ew - 12 - e_layer_min         # Ea, Ew = code - 127
    acc  += p << shift   (shift >= 0)
            p >> -shift  (shift < 0, arithmetic: rounds toward -inf)
    acc   = clamp(acc, -2^47, 2^47 - 1)        # sat = 1 where it fired
out = acc * 2^-20
```

`manifest.json` repeats all of this per layer, plus the geometry, the full
config and the patterns.

## Files

| file | what |
|---|---|
| `param_conv_weight` | the checkpoint, `[Cout, Cin, kH, kW]` |
| `input` | the test vector, `[N, C, H, W]` |
| `ref_fp32` | unquantised conv output. Context only — **neither** side should match this. |
| `alg_conv_a_int`, `alg_conv_a_exp` | input after MX quantization: int8 mantissas and exponent codes |
| `alg_conv_w_int`, `alg_conv_w_exp` | same for the weights |
| `alg_conv_acc` | int64 accumulator word — **the bit-exact layer output** |
| `alg_conv_sat` | 1 where the accumulator clamped |
| `alg_conv_out`, `alg_out` | `acc · 2^-20` as fp32. Loses bits once `acc` is wider than 24 bits, so compare `acc`, not these. |

## Reproducing our side

```bash
git clone <repo> && cd sirc_mx/configs
python main_ALG.py -m conv1x1_ramp                          # prints the taps
python export_vectors.py -m conv1x1_ramp,conv3x3_randn --clean   # rewrites this bundle
python export_vectors.py -m all --out more_vectors          # every case
```

## Comparing

Integer taps are compared **exactly** — one differing element is a failure.
Compare in pipeline order (a_int, a_exp, w_int, w_exp, acc, sat): the first
mismatch is the root cause, the rest follow from it. If you produce the taps in
the same layout, `main.py` does the comparison and names the first diverging
stage with its index; otherwise a text diff on the `.txt` files is enough to
start.
