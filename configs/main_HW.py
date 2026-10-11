"""
main_HW.py - run a benchmark case through the HW implementation (the HW side).

NOT IMPLEMENTED YET. This file is the contract: the structure, the interface
`main.py` calls, and comments describing what each step has to do. Arch fills in
the bodies (here in torch, or in C / Excel with this as the spec).

Same interface as `main_ALG.py`:

    from main_HW import run
    taps = run(case, config)        # {tap_name: tensor}, same keys as ALG

Why it exists separately: `main_ALG` goes through the MX lib, which we already
trust. This side is an independent implementation of the same arithmetic,
written from the HW spec and deliberately NOT importing the MX lib - if both
sides shared the quantizer, agreeing would prove nothing. Any difference
`main.py` reports is then a real disagreement about the spec, on a model small
enough to read every intermediate value.

Rules for whoever implements this:

  1. Do not import microxcaling / mx_quantizer / mx_layers_*. Integer ops,
     shifts and a saturating accumulator only.
  2. Emit the same tap names as `main_ALG` (see TAP NAMES below), so `main.py`
     can line them up key by key.
  3. Emit the integer taps, not just the final output. A mismatch on
     `model/out` alone says nothing about which stage is wrong.
  4. Keep it deterministic: no RNG anywhere in this file. All randomness lives
     in `models.py` and is already baked into the case.
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import models as M


# =============================================================================
# TAP NAMES
# =============================================================================
#
# Must match `main_ALG` exactly - name, shape and dtype. Per layer, in pipeline
# order (main.py compares in this order and blames the first mismatch):
#
#   <layer>/a_int    intN activation mantissas, [B, C, H, W]
#   <layer>/a_exp    activation exponent CODES, one per block, unsigned:
#                      code = E + emax,  emax = 2^(scale_bits-1) - 1 (127)
#                      all-zero block -> code 0
#                    NPE shape [B, C, H, ceil(W/bs)]
#   <layer>/w_int    intN weight mantissas, [O, Cin, kH, kW]
#   <layer>/w_exp    weight exponent codes, one per block of the filter
#                    flattened C-major (c, kh, kw): [O, ceil(Cin*kH*kW/bs)]
#   <layer>/acc      int64 accumulator word after the last product, the bias
#                    and the final clamp. THE bit-exact layer output.
#   <layer>/sat      uint8 [B, O, Ho, Wo]: 1 where the accumulator clamped at
#                    any step of that output's reduction
#   <layer>/out      acc * 2^e_layer_min (float; ALG holds it in fp32, which
#                    drops bits once acc is wider than 24 - compare acc)
#   model/out        network output
#
# (`--blockify channel` cases use [B, ceil(C/bs), H, W] / [O, ceil(Cin/bs),
# kH, kW] exponent shapes instead; NPE is the target.)
#
# Optional HW-only taps - stages the MX lib does not expose. `main.py` lists
# them as HW-only rather than as a mismatch:
#
#   <layer>/acc_raw  accumulator contents before any shift
#   <layer>/acc_shf  after the per-product shift
#   <layer>/acc_sat  after saturation, per step
#   <layer>/out_int  requantised integer output + <layer>/out_exp
#
# And, for cases carrying a static fixed-point output stage (case["out_quant"]
# is not None - see `models.py`, tag `fxp`):
#
#   model/out_pre    the output before the fixed-point stage
#   model/out_code   the integer word: round(value * 2^frac_bits), clamped
#   model/out_clip   1 where saturation touched the element, else 0
#
# Warp cases (tag `warp`; case["flow"] is the second network input), per call
# of the warp - prefix "warp", or "warp@<site>" when one instance runs once per
# pyramid level (`warp_levels`):
#
#   warp/ref_scale   float64 scalar: the int8 reference word's step
#   warp/ref_code    int8 [B, C, H, W]: round(ref / ref_scale), clamped
#   warp/grid_scale  float64 scalar: the grid step, 2^-8
#   warp/grid_code   int16 [B, 2, H, W]: the Q8.8 flow word, channel 0 = x
#   warp/out_scale   float64 scalar: the int8 result word's step
#   warp/out_pre     float64: the warped value before the out quantizer
#   warp/out_code    int8 [B, C, H, W]: THE bit-exact warp output
#   warp/out_clip    uint8: 1 where the out word saturated
#   warp/out         out_code * out_scale
#
# <layer> is the module name from `model.named_modules()`, e.g. "conv",
# "conv1", "dw", "head". `models.py` prints them with --list.


# =============================================================================
# Config keys this side has to honour
# =============================================================================
#
# `main.py` passes `main_ALG.effective_config(case, config)`: the base config
# with the case's own overrides (case["meta"]["mx"] / ["accum"]) already
# applied, so element format, scale_bits, bits and e_layer_min can differ per
# case. Read them from the config, never hard-code int8 / 48 / -20.
#
#   mx_specs.block_size        block length (32)
#   mx_specs.w_elem_format     'int8' -> 8-bit mantissa, mant_bias = 6
#   mx_specs.a_elem_format     same for activations (int10 / int16 cases too)
#   mx_specs.scale_bits        width of the unsigned exponent field (8)
#   mx_specs.shared_exp_method 'max' - exponent from the block max
#   xblock_accum.bits          accumulator width (48; 35 in accum35_*)
#   xblock_accum.sat_mode      'per_product' (NPE supports nothing else)
#   xblock_accum.e_layer_min   accumulator LSB exponent; per layer in
#                              case["meta"]["e_layer_min"] after ALG ran
#   xblock_accum.weight_blockify  'flatten' (NPE) | 'channel'
#   xblock_accum.act_blockify     'xblock'  (NPE) | 'channel'
#
# Settled with arch:
#   - geometry is NPE: activations block along W, weights flatten per filter.
#   - the exponent field is unsigned, scale_bits wide (default 8), biased by
#     2^(scale_bits-1) - 1. The real exponent still goes negative; below the
#     floor it clamps (`exp_underflow_clamp`, `scale_bits_5`).
#
# Still open - each has a case whose vectors differ between the candidates, so
# the answer can be read off a comparison rather than argued:
#   - MX mantissa tie rule. The lib rounds half AWAY from zero. Half-even or
#     truncation changes a_int / w_int in `mx_round_ties`.
#   - mantissa overflow: a block max just under 2^(E+1) rounds to 128. The lib
#     keeps E and clamps to +-127. Bumping E, or allowing -128, changes
#     `mx_mant_overflow`.
#   - zero-block exponent code. The contract pins 0; `zero_blocks`.
#   - do conv pad zeros join an X-block? The lib quantizes before padding, so
#     they do not. `xblock_w32` / `padded_edges` differ if HW blocks the
#     padded row.
#   - right shift on underflow: arithmetic (floor, the lib: -8064 >> 20 == -1)
#     or truncate toward zero (0)? `flush_to_zero` (all-or-nothing) and
#     `e_layer_min_m8` (a few LSBs per output) separate them.
#   - bias projection round(b * 2^-e_layer_min): the lib uses torch.round,
#     ties to EVEN - unlike its own MX quantizer. `bias_ties`.
#   - accumulation order. It only matters once per_product saturates
#     (`accum_sat_transient`); the lib walks the flattened filter stream
#     C-major: for c, for (kh, kw).
#   - groups != 1 has no HW path (MXConv2dHW asserts groups == 1) and there is
#     no HW Linear, so `depthwise`, `grouped4`, `linear` and the dw half of
#     `dw_pw` fall back to the FP32 path on the ALG side. Does the real HW run
#     depthwise convs on this datapath?
#
# The static fixed-point output stage is configured per case rather than in
# mx_specs, because it is a different quantizer (see `out_quant_fxp` below):
#   out_quant.total_bits / frac_bits / signed / round / saturate
#
# The warp is configured per case too (`main_ALG.warp_config(case)`, recorded
# in case["meta"]["warp_config"]): ref / grid / out words, call_sites. Settled
# with arch: ref and out are int8 with a static arbitrary-real scale, one per
# pyramid level; the grid is Q8.8 (16 bits, 1/256 px). Still open, each with
# the case that settles it:
#   - grid meaning: a displacement added to the output pixel's coordinate (the
#     spec), or an absolute source coordinate? Which channel is x?
#     `warp_int_shift` - a pure shift under the spec.
#   - coordinate split: x0 = X >> 8 (arithmetic, floor) and fx = X & 255, or a
#     split toward zero? Only differs for negative coordinates:
#     `warp_neg_frac`.
#   - out of bounds: a neighbour outside the image contributes 0 (spec), or
#     the index clamps to the border? `warp_oob`.
#   - tie rules of the three quantizers (lib: half away from zero everywhere):
#     `warp_ref_ties`, `warp_grid_ties`, `warp_half_pixel` (the out word).
#   - saturation: the grid clamps to [-32768, 32767] (`warp_grid_sat`); the
#     out word to [-128, 127], i.e. -128 is reachable (`warp_out_sat`).
#   - the rescale acc * ref_scale / out_scale: in what precision does HW
#     multiply by an arbitrary real (fp32? a fixed-point multiplier + shift)?
#     The lib uses float64, so a coarser multiplier can move an out_code by
#     one near a tie. `warp_subpix` / `warp_levels` use calibrated scales.
#   - DOF's production warp is grid_sample (fp32, coordinates normalised to
#     [-1, 1] and back), not this exact form: its round trip is not exact, so
#     the trained model may sit up to one out LSB away from the integer warp
#     near ties.


def quantize_mx(t, bs, mbits, scale_bits=8, axis=None, blockify="xblock"):
    """FP tensor -> (intN mantissas, per-block exponent codes).

    What it has to do:
      * split `t` into blocks of `bs` along the blockified axis. `xblock`
        blocks activations along W; `flatten` flattens [Cin,kH,kW] per output
        filter (C-major) and blocks along that stream; `channel` blocks along
        dim 1. A tail shorter than `bs` is its own block.
      * E = floor(log2(max|x| in the block))  -- shared_exp_method 'max'
      * clamp E to [-emax, emax], emax = 2^(scale_bits-1) - 1
      * mantissa = round(x * 2^(mant_bias - E)), clamped to +-(2^(mbits-1) - 1)
      * mant_bias = mbits - 2  (6 for int8)
      * code = E + emax; an all-zero block gets code 0
    """
    raise NotImplementedError("quantize_mx")


def out_quant_fxp(t, cfg):
    """Static fixed-point output word. NOT MX - a different quantizer.

    MX derives its scale from each block's max, so it cannot overflow. This
    stage has a scale fixed up front - step = 2^-frac_bits, whatever the data -
    so it CAN overflow, silently. That is the one failure mode to get right
    here, and `out_fxp_saturate` / `out_fxp_unsigned` in the registry are the
    cases that show it.

    What it has to do:
      1. code = round(value * 2^frac_bits), with the configured tie rule:
         'half_away' (ties away from zero), 'half_even', or 'trunc' (toward
         zero). The three differ on real data - `out_fxp_round_modes` is tuned
         so ties are common.
      2. clamp code to the word: signed -> [-2^(tb-1), 2^(tb-1) - 1];
         unsigned -> [0, 2^tb - 1]. Both directions matter; an unsigned word
         has to clamp negatives to 0.
      3. emit model/out_code (the integer word - this is the bit-exact
         comparable), model/out_clip (where the clamp fired) and model/out
         (code * 2^-frac_bits, for the float comparison).

    For DOF the format is Q8.8 signed/16b: step 1/256, range
    [-128, +127.99609375].
    """
    raise NotImplementedError("out_quant_fxp")


def conv2d_hw(a_int, a_exp, w_int, w_exp, bias, geom, cfg):
    """One conv, the way the HW does it. The core of this file.

    Not a block dot product in FP: HW has a single Nb x Nb -> 2Nb multiplier per
    BCU, and the reduction happens inside a narrow fixed-point accumulator.

    Per output element, loop over the reduction (k*k*Cin) one product at a time:

      1. p = a_int * w_int                      exact, 2*Nb bits, no rounding
      2. shift = a_exp + w_exp - 2*mant_bias - e_layer_min
         p <<= shift                            signed shift; right shift drops
                                                bits, which is a real loss
         (a_exp / w_exp here are the real exponents, code - emax)
      3. acc += p, then clamp acc to the accumulator width - every product
         (sat_mode 'per_product'). Walk the products in the flattened filter
         order: for c in Cin, for (kh, kw) in the kernel.
      4. mark the output in <layer>/sat if any step clamped

    Then the output side:
      5. bias: round(b * 2^-e_layer_min) into the accumulator after the last
         product, clamp once more. `with_bias` / `bias_ties` show it.
      6. emit <layer>/acc (int64) and <layer>/out = acc * 2^e_layer_min.

    `geom` carries stride / padding / dilation / groups from the nn.Conv2d the
    case built; read them off the module rather than re-deriving.
    """
    raise NotImplementedError("conv2d_hw")


def warp_hw(ref, flow, wcfg, site=0):
    """The DOF warp, the way the HW does it. Integer only once quantized.

      1. ref_code  = clamp(round_half_away(ref / ref_scale), -128, 127)
         grid_code = clamp(round_half_away(flow * 256), -32768, 32767)
         (ref_scale / out_scale per call site: case taps warp*/ref_scale, or
         calibrate max|x| / 127 the way `calibrate_act_scales` does)
      2. per output pixel (x, y), per channel:
           X  = (x << 8) + grid_code[:, 0]      Y = (y << 8) + grid_code[:, 1]
           x0 = X >> 8, fx = X & 255           y0 = Y >> 8, fy = Y & 255
           acc = sum of ref_code[y0 + dy, x0 + dx] * wx * wy over dx, dy in
                 {0, 1}, wx = 256 - fx | fx, wy = 256 - fy | fy; a neighbour
                 outside the image contributes 0
      3. out_code = clamp(round_half_away(acc * ref_scale / out_scale / 65536),
                          -128, 127)
      4. emit the warp taps listed under TAP NAMES.

    `tap_check.ref_warp` is the ALG side's version of exactly this (the
    options it takes are the alternatives listed as open above).
    """
    raise NotImplementedError("warp_hw")


def run(case, config=None, verbose=False):
    """Run one case through the HW implementation.

    Args:
        case: dict from `models.build_case` - has "model" (nn.Module, weights
              already filled), "x" (the tv), "y_fp32" (FP reference), "meta".
        config: the same dict `main_ALG.run` gets.

    Returns:
        dict: tap name -> tensor, including "model/out".

    How to walk the model: the toy models are small and explicit
    (`SingleConv`, `ConvReLU`, `ConvReLUConv`, `DwPw`, `ConvConcat`, `ConvAdd`,
    `SingleLinear`), so the simplest correct thing is to walk
    `case["model"].named_modules()` in order, dispatch Conv2d / Linear to
    `conv2d_hw`, and keep ReLU / concat / add in FP between layers - mirroring
    what the MX lib does, where only conv and linear are quantized.

    What must be true before anything else: layer N+1's activations are layer
    N's *requantised* output, not an FP32 one. That handoff is `conv_relu_conv`
    in the registry, and getting it wrong is invisible on single-layer cases.

    Warp cases take two inputs: `models.inputs_of(case)` is (x, flow). Run
    every `models.Warp` module through `warp_hw`, in call order.

    Finally, if `case["out_quant"]` is set, push the network output through
    `out_quant_fxp` and emit the model/out_pre, model/out_code and
    model/out_clip taps.
    """
    raise NotImplementedError(
        "main_HW.run is not implemented yet - this file is the spec skeleton. "
        "See the comments in conv2d_hw / quantize_mx."
    )


def main():
    parser = M.add_model_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--config", default=None)
    parser.add_argument("--dump", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    for case in M.cases_from_args(args):
        taps = run(case, config=None, verbose=args.verbose)
        print(case["meta"]["case"], {k: tuple(v.shape) for k, v in taps.items()})


if __name__ == "__main__":
    main()
