"""MX (microscaling) format primitives for the NPE emulator.

Verbatim transcription of the validated notebook quantizers — keep in sync
with the notebook cell-for-cell.  Quirks are intentional HW emulation:
  * unsigned 4-bit exponent wrap-around (``& 0xF``)
  * ``input_mx_quantizer``'s uninitialized ``log2_vals`` (all-zero shared exp)
  * int storage with sign restored in the high bits
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

SIGN_BIT = 1
GREATER_THAN_ONE_BIT = 1


@dataclass
class MXTensorConfig:
    mantissa_bits: int
    exponent_bits: int


@dataclass
class MXTensor:
    shared_exp: torch.Tensor
    data: torch.Tensor
    original_shape: tuple
    exponent_bits: torch.Tensor
    mantissa_bits: torch.Tensor
    e_min_layer: torch.Tensor
    calibrate_offset: torch.Tensor


mx_config_factory = dict()
mx_config_factory['mx-e4bm8'] = MXTensorConfig(mantissa_bits=torch.tensor(8), exponent_bits=torch.tensor(4))
mx_config_factory['mx-e8bm8'] = MXTensorConfig(mantissa_bits=torch.tensor(8), exponent_bits=torch.tensor(8))


def mantissa_offset(mx_type: str = "mx-e4bm8") -> int:
    """Number of fractional bits of the stored mantissa (value in [-2, 2))."""
    return int(mx_config_factory[mx_type].mantissa_bits - SIGN_BIT - GREATER_THAN_ONE_BIT)


def base_mx_quantizer(x: torch.Tensor, block_size: int = 32, mx_type : str = "mx-e4bm8", calibrate_offset : int = 0) -> MXTensor:

    x_scaled = x # offset to enssure unsigned exponenet (exponent shifted left)
    # 1. Pad to a multiple of block_size
    width = x_scaled.shape[-1]
    original_shape = tuple(x_scaled.shape)
    num_blocks = math.ceil(width / block_size)

    width = x_scaled.shape[-1]
    num_blocks = math.ceil(width / block_size)
    pad_amount = num_blocks * block_size - width

    if pad_amount > 0:
        x_pad = F.pad(x_scaled, (0, pad_amount), mode='constant', value=0)
    else:
        x_pad = x_scaled

    blocks = x_pad.view([*x_pad.shape[:-1], num_blocks, block_size])

    abs_blocks = blocks.abs()
    mask = abs_blocks > 0
    log2_vals = torch.zeros_like(abs_blocks)
    log2_vals[mask] = torch.log2(abs_blocks[mask]).ceil()  # i think it should be ceil but check that
    log2_scale = log2_vals.amax(dim=-1)                                  # [batch, num_blocks]

    mantissa_offset = mx_config_factory[mx_type].mantissa_bits - SIGN_BIT - GREATER_THAN_ONE_BIT
    e_min_layer = mantissa_offset - calibrate_offset

    # exponent saturation
    sat_val = 2**(mx_config_factory[mx_type].exponent_bits)-1

    shift_sat = log2_scale - mantissa_offset

    # warp-around handling
    preserve_lsb = 0xf
    shift_sat = shift_sat.int() & preserve_lsb  # keep only unsigned 4-bits

    divisor = torch.exp2(shift_sat.unsqueeze(-1).float())
    data = blocks / divisor
    if mx_config_factory[mx_type].mantissa_bits == 8:
        preserve_lsb = 0xff
        preserve_sign = 0xffff_ff00
    else:
        raise NotImplementedError("mx-int16 is not supported right now")  # handle both 8 (and 16 bit scenario (0xffff) in the future TODO)
    data = data.int()

    data_sign = data < 0
    data = data & preserve_lsb  # keep only the 8-lsbs

    data = torch.where(data_sign, data | preserve_sign, data)  # handle negative values (restore the sign for 8 bit)

    return MXTensor(
        shared_exp=shift_sat,
        data=data,
        original_shape=original_shape,
        exponent_bits=mx_config_factory[mx_type].exponent_bits,
        mantissa_bits=mx_config_factory[mx_type].mantissa_bits,
        e_min_layer=e_min_layer,
        calibrate_offset=calibrate_offset
    )


def input_mx_quantizer(x: torch.Tensor, block_size: int = 32, mx_type : str = "mx-e4bm8", input_type : int = 8, calibrate_offset : int = 6) -> MXTensor:

    x_scaled = x  # offset to enssure unsigned exponenet (exponent shifted left)
    # 1. Pad to a multiple of block_size
    width = x_scaled.shape[-1]
    original_shape = tuple(x_scaled.shape)
    num_blocks = math.ceil(width / block_size)

    width = x_scaled.shape[-1]
    num_blocks = math.ceil(width / block_size)
    pad_amount = num_blocks * block_size - width

    if pad_amount > 0:
        x_pad = F.pad(x_scaled, (0, pad_amount), mode='constant', value=0)
    else:
        x_pad = x_scaled

    blocks = x_pad.view([*x_pad.shape[:-1], num_blocks, block_size])

    abs_blocks = blocks.abs()
    mask = abs_blocks > 0
    log2_vals = torch.zeros_like(abs_blocks)
    log2_scale = log2_vals.amax(dim=-1)                                  # [batch, num_blocks]

    mantissa_offset = mx_config_factory[mx_type].mantissa_bits - SIGN_BIT - GREATER_THAN_ONE_BIT
    e_min_layer = mantissa_offset - calibrate_offset

    shift_sat = log2_scale

    # overflow handling
    if input_type:
        preserve_lsb = 0xff
        preserve_sign = 0xffff_ff00
    else:
        raise NotImplementedError("Missing 16-bit input implementation")  # handle both 8 (and 16 bit scenario (0xffff) in the future TODO)

    data = blocks.int()
    data_sign = data < 0
    data = data & preserve_lsb  # keep only the 8-lsbs
    data = torch.where(data_sign, data | preserve_sign, data)  # handle negative values (restore the sign for 8 bit)

    return MXTensor(
        shared_exp=shift_sat,
        data=data,
        original_shape=original_shape,
        exponent_bits=mx_config_factory[mx_type].exponent_bits,
        mantissa_bits=mx_config_factory[mx_type].mantissa_bits,
        e_min_layer=e_min_layer,
        calibrate_offset=calibrate_offset
    )


def mx_weight_quantizer(
        weight: torch.Tensor,
        block_size: int = 32,
        mx_type: str = "mx-e4bm8",
        skip_exp: bool = False,
        calibrate_offset: int = 6,
    ):
    w_data_list_t = []
    w_exp_list_t  = []

    if len(weight.shape) > 2:
        out_channels = weight.shape[0]

        for oc in range(out_channels):
            w_flat = weight[oc].data.view(-1)
            w_flat = w_flat * 2.0 ** calibrate_offset
            mx_w = base_mx_quantizer(w_flat, block_size=block_size, mx_type=mx_type, calibrate_offset=calibrate_offset)
            w_data_list_t.append(mx_w.data)
            w_exp_list_t.append(mx_w.shared_exp)
    else:
        shallow_inp = weight * 2.0 ** calibrate_offset
        mx_w = base_mx_quantizer(shallow_inp, block_size=block_size, mx_type=mx_type, calibrate_offset=calibrate_offset)
        w_data_list_t.append(mx_w.data)
        w_exp_list_t.append(mx_w.shared_exp)

    mantissa_offset = mx_config_factory[mx_type].mantissa_bits - SIGN_BIT - GREATER_THAN_ONE_BIT
    e_min_layer = mantissa_offset - calibrate_offset

    return MXTensor(
        shared_exp=torch.stack(w_exp_list_t),
        data=torch.stack(w_data_list_t),
        original_shape=tuple(weight.shape),
        exponent_bits=mx_config_factory[mx_type].exponent_bits,
        mantissa_bits=mx_config_factory[mx_type].mantissa_bits,
        e_min_layer=e_min_layer,
        calibrate_offset=calibrate_offset
    )


def mx_dequantizer(mx: MXTensor, mx_type: str = "mx-e4bm8") -> torch.Tensor:
    # real value = mantissa * 2^(shared_exp - mantissa_offset - calibrate_offset)
    shift = mx.shared_exp - (mx_config_factory[mx_type].mantissa_bits - SIGN_BIT - GREATER_THAN_ONE_BIT) - mx.calibrate_offset
    deq = mx.data * torch.exp2(shift.unsqueeze(-1).float())
    flat = deq.reshape(*deq.shape[:-2], -1)
    return flat[..., : mx.original_shape[-1]].reshape(mx.original_shape)
