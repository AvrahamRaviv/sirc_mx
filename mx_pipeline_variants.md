# MX Integer Pipeline Variants

---

## 1. Emulation (current library)

Scale is baked into FP32 values before MAC. Everything stays FP32.

```
   a: FP32          w: FP32
      |                 |
      v                 v
 [MX quantize]    [MX quantize]       ← divide by 2^se, clip mantissa, multiply back
      |                 |
   FP32 (low prec)   FP32 (low prec)
      |                 |
      +--------+--------+
               |
               v
          [FP32 × FP32]
               |
             FP32
               |
               v
          [FP32 accum]                ← all blocks accumulated together, no scale separation
               |
             FP32
               |
               v
        [elemwise quant]
               |
           FP32 / BF16
```

---

## 2. HW Variant A — Scale before MAC (dequantize first)

Scale applied before multiply. Target format depends on hardware MAC unit — not necessarily FP32.
Examples: MXINT4/INT6 → dequantize to INT8 → INT8×INT8 MAC; MXINT8 → INT16×INT16; MXFP4 → FP16×FP16.

```
   a: MXINTN + S_a     w: MXINTN + S_w
      |                    |
      v                    v
  [× 2^S_a]            [× 2^S_w]     ← dequantize to wider integer or float
      |                    |
  INT8/INT16/FP16      INT8/INT16/FP16
      |                    |
      +---------+----------+
                |
                v
      [INT8×INT8 / INT16×INT16 / FP16×FP16]
                |
          INT16/INT32/FP32
                |
                v
          [accum in wider format]
                |
             FP32 output
```

---

## 3. HW Variant B — Scale after multiply, before accumulate

Each product is scaled individually before adding to accumulator.

```
   a: INT8 + S_a     w: INT8 + S_w
      |                 |
      +--------+--------+
               |
               v
          [INT8 × INT8]
               |
             INT16
               |
               v
       [× (S_a × S_w)]               ← one scale per multiply result
               |
             FP32
               |
               v
          [FP32 accum]                ← accumulate scaled products across all blocks
               |
           FP32 output
```

---

## 4. HW Variant C — Scale after block accumulator (MX spec)

Each block accumulates fully in INT32, then one scale per block. Cross-block sum in FP32.

```
  Block 0                Block 1                Block 2
  a0:INT8  w0:INT8       a1:INT8  w1:INT8       a2:INT8  w2:INT8
     |         |            |         |            |         |
     +----+----+            +----+----+            +----+----+
          |                      |                      |
          v                      v                      v
    [INT8 × INT8 MAC]      [INT8 × INT8 MAC]      [INT8 × INT8 MAC]
          |                      |                      |
        INT32                  INT32                  INT32
          |                      |                      |
          v                      v                      v
   [× (S_a0 × S_w0)]     [× (S_a1 × S_w1)]     [× (S_a2 × S_w2)]
          |                      |                      |
        FP32                   FP32                   FP32
          |                      |                      |
          +----------+-----------+----------+-----------+
                     |
                     v
               [FP32 accum]                             ← cross-block sum in FP32
                     |
                 FP32 output
```

**This is the correct MX hardware model.**
One INT32 accumulator per block → scale once → sum block results in FP32.
