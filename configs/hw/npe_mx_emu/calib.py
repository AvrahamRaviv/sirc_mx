"""Calibration glue: LayerMXQParams and Cstruct construction.

Scale bookkeeping of the NPE MX flow lives here and in observers.py:

    calibrate_offset   derived by MXObserver from collected block-amax stats
    e_min_layer        = mantissa_offset - calibrate_offset   (this module is the
                        only code path that may contain this formula, via
                        observers.MXObserver.e_min_layer / e_min_layer())
    accumulator units  = 2 ** (x_cal + w_cal)                 (bias must live here)
    output scale       = 2 ** (2 * mantissa_offset)           (HW-default PCU output)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from npe_mx_emu.hw_pipe import Cstruct, frac_bits_pair
from npe_mx_emu.mx_format import mantissa_offset


def e_min_layer(mx_type: str, calibrate_offset: int) -> int:
    return mantissa_offset(mx_type) - calibrate_offset


@dataclass
class LayerMXQParams:
    """Per-layer MX quantization parameters (derived from observers).

    x_cal  : calibrate offset of the layer input
    w_cal  : calibrate offset of the weights (per-tensor for now)
    out_cal: scale -log2 of the raw integer output.  If None, defaults to
             2 * mantissa_offset (the HW-default PCU output scale).
    is_input_layer: select the integer-only input quantizer (zero exponent).
    """

    x_cal: int
    w_cal: int
    out_cal: "int | None" = None
    block_size: int = 32
    mx_type: str = "mx-e4bm8"
    is_input_layer: bool = False

    def __post_init__(self):
        if self.out_cal is None:
            self.out_cal = 2 * mantissa_offset(self.mx_type)

    @property
    def b_cal(self) -> int:
        """Bias scale -log2: bias must be in accumulator units 2**(x_cal + w_cal)."""
        return self.x_cal + self.w_cal

    @property
    def output_scale_log2(self) -> int:
        assert self.out_cal is not None
        return self.out_cal


def build_c_struct(
    qparams: LayerMXQParams,
    bias_oc: torch.Tensor,
    acc_bits: int = 46,
    relux_bits: int = 24,
) -> Cstruct:
    """Derive the PCU c_struct for one output channel from layer qparams.

    Only place where the bias is rescaled into accumulator units.  This
    conversion is done OFFLINE (HW answer, Q8): the rounding is a SW design
    choice -- round half up (ties toward +inf, matching the PCU convention
    in pcu_div), error <= 0.5 ULP of accumulator units.  ``.long()`` keeps
    the whole PCU path integer (pcu_pla -> pcu_div) end to end.
    """
    return Cstruct(
        bias=(bias_oc.float() * 2.0 ** qparams.b_cal + 0.5).floor().long(),
        scale=1,
        shift=0,
        round_type="round",
        min_val=-2.0 ** (relux_bits - 1),
        max_val=2.0 ** (relux_bits - 1) - 1,
    )


def build_warp_c_struct_arr(
    qparams: LayerMXQParams,
    bias: torch.Tensor,
    out_channels: int,
    frac_bits=(7, 7),
    relux_bits: int = 8,
) -> dict:
    """Derive the PCU c_structs for a warp layer.

    All warp operands are plain INTs with zero exponents ("NOT MX INT BUT
    INT"): the reference is uint8 (x scale **1) and the TRC weights are exact
    integers over ``2 ** (frx + fry)`` (per-axis fraction bits from the grid
    quantizer), so the accumulator units are ``2 ** (frx + fry)`` and the
    bias is placed on those units.  The PCU shift removes the weight fraction
    scale; the runtime ``e_min_layer`` of the reference (``mantissa_offset``,
    calibrate_offset 0) covers the ``mantissa_offset`` part, hence::

        c_struct.shift      = out_cal - mantissa_offset - (frx + fry)
        effective pcu shift = shift + e_min_layer = out_cal - (frx + fry)

    With ``out_cal = 0`` the output is the warped value rounded to integer,
    then reluxed to the int range (default the 24-bit relux, like
    ``build_c_struct`` -- the warp demo data spans 0..255, so an int8 clamp
    is available only by passing ``relux_bits=8`` explicitly).

    Args:
        frac_bits: scalar or (frx, fry) pair used to quantize the grid.
    """
    if bias.shape[0] != out_channels:
        raise ValueError(f"bias has {bias.shape[0]} channels, layer expects {out_channels}")
    frx, fry = frac_bits_pair(frac_bits)
    fp = frx + fry
    if not (0 <= frx <= 8 and 0 <= fry <= 8):
        raise ValueError(
            f"weight scale 2**{fp} overflows the 16-bit weight mantissa (max 2**16); "
            f"frac_bits={(frx, fry)}"
        )
    mo = mantissa_offset(qparams.mx_type)
    shift = qparams.out_cal - mo - fp
    relux_max = 2.0 ** (relux_bits - 1) - 1
    relux_min = -2.0 ** (relux_bits - 1)
    return {
        oc: Cstruct(
            bias=(bias[oc].float() * 2.0 ** fp + 0.5).floor().long(),
            scale=1,
            shift=shift,
            round_type="round",
            min_val=relux_min,
            max_val=relux_max,
        )
        for oc in range(out_channels)
    }


def build_c_struct_arr(
    qparams: LayerMXQParams,
    bias: torch.Tensor,
    out_channels: int,
    acc_bits: int = 46,
    relux_bits: int = 24,
) -> dict:
    if bias.shape[0] != out_channels:
        raise ValueError(f"bias has {bias.shape[0]} channels, layer expects {out_channels}")
    return {oc: build_c_struct(qparams, bias[oc], acc_bits=acc_bits, relux_bits=relux_bits)
            for oc in range(out_channels)}
