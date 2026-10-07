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
# Must match `main_ALG` exactly. Per layer, in pipeline order:
#
#   <layer>/a_int    intN activation mantissas, same shape as the activation
#   <layer>/a_exp    per-block activation shared exponent
#   <layer>/w_int    intN weight mantissas, same shape as the weight
#   <layer>/w_exp    per-block weight shared exponent
#   <layer>/out      layer output
#   model/out        network output
#
# And the HW-only taps - the stages the MX lib does not have, which is exactly
# where we expect the disagreement to be. `main.py` reports them as HW-only
# rather than as a mismatch:
#
#   <layer>/acc_raw  accumulator contents before any shift, integer
#   <layer>/acc_shf  after the per-product shift by Ew + Ea - 2*mant_bias
#                    - e_layer_min
#   <layer>/acc_sat  after saturation into the narrow accumulator
#   <layer>/sat_cnt  how many products saturated (per output element)
#   <layer>/out_int  requantised integer output + <layer>/out_exp
#
# And, for cases carrying a static fixed-point output stage (case["out_quant"]
# is not None - see `models.py`, tag `fxp`):
#
#   model/out_pre    the output before the fixed-point stage
#   model/out_code   the integer word: round(value * 2^frac_bits), clamped
#   model/out_clip   1 where saturation touched the element, else 0
#
# <layer> is the module name from `model.named_modules()`, e.g. "conv",
# "conv1", "dw", "head". `models.py` prints them with --list.


# =============================================================================
# Config keys this side has to honour
# =============================================================================
#
# Read from the same config dict `main_ALG` gets, so one file drives both sides:
#
#   mx_specs.block_size        block length (32)
#   mx_specs.w_elem_format     'int8' -> 8-bit mantissa, mant_bias = 6
#   mx_specs.a_elem_format     same for activations
#   mx_specs.scale_bits        width of the shared exponent field
#   mx_specs.shared_exp_method 'max' - exponent from the block max
#   xblock_accum.bits          accumulator width (35 / 48)
#   xblock_accum.sat_mode      'per_product' | 'per_block' - where the clamp is
#   xblock_accum.e_layer_min   static per-layer exponent floor
#   xblock_accum.weight_blockify  'flatten' | 'channel'
#   xblock_accum.act_blockify     'xblock'  | 'channel'
#
# The open questions from the spec thread belong here, as explicit config, not
# as an assumption buried in the code:
#   - is the shared exponent / shift non-negative only (0..15), or can it go
#     negative?
#   - does a partial tail block get its own shared exponent, or is it padded
#     to a full block?
#   - do conv pad zeros join a block (shifting its alignment) or are they
#     skipped?
#   - is the bias added into the fixed-point accumulator, or after the requant?
#   - when a product underflows the accumulator grid (shift < 0), does the
#     right shift round toward -inf or truncate toward zero? MXConv2dHW uses an
#     arithmetic shift, so -8064 >> 20 == -1, not 0: a block of underflowing
#     products leaves -1 LSB per negative product and 0 per positive one, i.e.
#     a systematic negative bias rather than a clean flush to zero. Truncating
#     toward zero gives exactly 0 instead. The `flush_to_zero` case separates
#     the two - it lands on [-2.1e-05 .. -1.0e-05] under arithmetic shift and
#     on 0 under truncation.
#   - per_product or per_block saturation? `accum_sat_transient` settles it:
#     each block's partial sum leaves the accumulator range and comes back, so
#     per_product clamps at the peak, loses the excess and ends pinned at the
#     negative bound (-1.342e+08), while per_block sums first and returns
#     exactly 0. No other case distinguishes them - accum_saturate pins the
#     clamp bound but both modes agree there.
#   - groups != 1 has no HW path at all (MXConv2dHW asserts groups == 1) and
#     there is no HW Linear, so `depthwise`, `grouped4`, `linear` and half of
#     `dw_pw` fall back to the FP32 path on the ALG side. Does the real HW run
#     depthwise convs on this datapath, and if so with what blockify?
#
# The static fixed-point output stage is configured per case rather than in
# mx_specs, because it is a different quantizer (see `out_quant_fxp` below):
#   out_quant.total_bits / frac_bits / signed / round / saturate


def quantize_mx(t, bs, mbits, axis=None, blockify="channel"):
    """FP tensor -> (intN mantissas, per-block shared exponent).

    What it has to do:
      * split `t` into blocks of `bs` along the blockified axis. `channel`
        blocks along dim 1; `flatten` flattens [Cin,kH,kW] per output filter
        and blocks along that stream; `xblock` blocks along W.
      * E = floor(log2(max|x| in the block))  -- shared_exp_method 'max'
      * mantissa = round(x * 2^(mant_bias - E)), clamped to +-(2^(mbits-1) - 1)
      * mant_bias = mbits - 2  (6 for int8)
      * decide the tail block: own exponent, or zero-pad to full. See the open
        questions above.
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
      3. acc += p
         if sat_mode == 'per_product': clamp acc to the accumulator width now,
         every product. If 'per_block': only at the end of a block. These two
         differ only once something saturates - that is what `accum_saturate`
         in the registry is for.
      4. record how many products hit the clamp -> <layer>/sat_cnt

    Then the output side:
      5. bias: into the accumulator at the e_layer_min scale, or after the
         requant? Config decides, and `with_bias` is the case that shows it.
      6. scale back by 2^e_layer_min and requantise to the output format,
         emitting <layer>/out_int + <layer>/out_exp alongside the float
         <layer>/out.

    `geom` carries stride / padding / dilation / groups from the nn.Conv2d the
    case built; read them off the module rather than re-deriving.
    """
    raise NotImplementedError("conv2d_hw")


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
