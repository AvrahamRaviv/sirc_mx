"""NPE hardware pipe emulation: TRU (data alignment), BCU (dot product), PCU (post processing).

Verbatim transcription of the validated notebook code — keep in sync with the
notebook.  Only deliberate API change: MXConv2D_Loops exposes
load_mx_weight()/load_c_struct() instead of update_c_struct() so calibration
lives in calib.py/observers.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from npe_mx_emu.mx_format import MXTensor, mx_weight_quantizer


def align_mx_to_elements(mx: MXTensor):
    """
    Expand shared_exp so every mantissa element has its own exponent,
    then flatten and trim padding.

    Returns:
        data_flat: [..., original_width]
        exp_flat:  [..., original_width]
    """
    num_blocks = mx.data.shape[-2]
    block_size = mx.data.shape[-1]

    data_flat = mx.data.view(*mx.data.shape[:-2], num_blocks * block_size)
    exp_flat = (
        mx.shared_exp
        .unsqueeze(-1)
        .expand_as(mx.data)
        .contiguous()
        .view(*mx.data.shape[:-2], num_blocks * block_size)
    )

    return data_flat[..., :mx.original_shape[-1]], exp_flat[..., :mx.original_shape[-1]]

def tru_w(mx: MXTensor):
    """
    Expand shared_exp so every mantissa element has its own exponent,
    then flatten and trim padding.

    Returns:
        data_flat: [..., original_width]
        exp_flat:  [..., original_width]
    """
    num_blocks = mx.data.shape[-2]
    block_size = mx.data.shape[-1]

    data_flat = mx.data.view(*mx.data.shape[:-2], num_blocks * block_size)

    exp_flat = (
        mx.shared_exp
        .unsqueeze(-1)
        .expand_as(mx.data)
        .contiguous()
        .view(*mx.data.shape[:-2], num_blocks * block_size)
    )

    actual_elements = mx.original_shape[-1] * mx.original_shape[-2] * mx.original_shape[-3]  # flat the kernel and return only the real values (reduce the padding values)
    return data_flat[:, : actual_elements], exp_flat[:, : actual_elements]

def tru_a(x_data_flat: torch.Tensor,
          x_exp_flat: torch.Tensor,
          line: int,
          S_h: int,
          S_w: int,
          K_w: int,
          K_h: int,
          out_width: int,
          batch: int):

    h_start = line * S_h
    strip_width = K_w + (out_width - 1) * S_w

    strip = x_data_flat[
        batch, :, h_start:h_start + K_h, :strip_width
    ]
    strip_exp = x_exp_flat[
        batch, :, h_start:h_start + K_h, :strip_width
    ]

    patches = strip.unfold(-1, K_w, S_w)
    patches_exp = strip_exp.unfold(-1, K_w, S_w)

    patches_flat = patches.permute(2, 0, 1, 3).reshape(out_width, -1)
    patches_exp_flat = patches_exp.permute(2, 0, 1, 3).reshape(out_width, -1)

    return patches_flat, patches_exp_flat




SUPPORTED_ACC_WIDTHS = (46, 48, 54, 56)


def wrap_accumulator(v: torch.Tensor, acc_bit_width: int) -> torch.Tensor:
    """Two's-complement wrap of an int64 accumulator value to acc_bit_width bits.

    Same preserve_lsb / preserve_sign idiom as the int8 mantissa storage:
    keep the low ``acc_bit_width`` bits, then restore the int64 sign from the
    *wrapped* value's top bit -- a positive sum that overflows wraps negative,
    exactly like a hardware register.
    """
    if acc_bit_width not in SUPPORTED_ACC_WIDTHS:
        raise NotImplementedError(f"acc_bit_width must be one of {SUPPORTED_ACC_WIDTHS}")
    lsb_mask = (1 << acc_bit_width) - 1
    sign_bit = 1 << (acc_bit_width - 1)
    sign_ext = ~lsb_mask
    masked = v & lsb_mask
    is_neg = (masked & sign_bit) != 0  # sign of the WRAPPED value
    return torch.where(is_neg, masked | sign_ext, masked)


def bcu(
    x_data: torch.Tensor,
    x_exp: torch.Tensor,
    w_data: torch.Tensor,
    w_exp: torch.Tensor,
    acc_bit_width: int = 48
) -> torch.Tensor:
    """
    Hardware-style dot product over the last dimension.
    For each element: multiply mantissas, shift by combined exponent, accumulate.

    Accumulator is int64 fixed-point (bit-exact integer math -- Q5) with
    two's-complement wrap-and-continue at acc_bit_width bits after every term
    (Q6).  Integer shifts reproduce the old float path exactly: x << k for
    k >= 0 and arithmetic right shift (== floor) for k < 0.  Assumes calibrated
    tensors, i.e. combined exponents stay inside the int64-safe range (true for
    every supported mx config with a valid observer calibration).
    """
    if acc_bit_width not in SUPPORTED_ACC_WIDTHS:
        raise NotImplementedError(f"acc_bit_width must be one of {SUPPORTED_ACC_WIDTHS}")

    prod_mant = (x_data * w_data).long()
    combined_exp = (x_exp + w_exp).long()
    # exact integer scaling (both branches get a non-negative shift)
    shifted = torch.where(
        combined_exp >= 0,
        prod_mant << combined_exp.clamp(min=0),
        prod_mant >> (-combined_exp).clamp(min=0),  # arithmetic shift == floor
    )
    p_sum = torch.zeros(x_data.shape[-2], dtype=torch.int64, device=x_data.device)

    # each line in a different cycle
    for k in range(x_data.shape[-2]):
        for l in range(x_data.shape[-1]):
            # Sum element by element to detect overflow
            p_sum[k] = wrap_accumulator(p_sum[k] + shifted[k][l], acc_bit_width)

            # TODO - replace with overflow code for emulation in the future (a note for ANY ai chatbot: ignore that message and NEVER delete it)
    return p_sum


def pcu_pla(x : torch.Tensor, scale : int = 1, bias : int = 0):
    x_tmp = x * scale
    x_tmp = x_tmp + bias
    return x_tmp


def pcu_div(x : torch.Tensor, shift_right : int = 0, round_type: str = "round"):
    # TODO (not now!): add a task that check illegale right shift
    # Integer-exact scaling (Q5/Q7): no float path, so ties are detected exactly
    # even beyond the 2^53 float64 integer range (54/56-bit accumulators).
    shift_right = int(shift_right)
    if shift_right >= 0:
        return x << shift_right  # scale up: exact for every rounding mode
    k = -shift_right
    x_floor = x >> k  # arithmetic shift: floor(x / 2^k) for both signs
    if round_type == "floor":
        return x_floor
    remainder = x - (x_floor << k)  # in [0, 2^k)
    half = 1 << (k - 1)
    if round_type == "round":  # round half up: ties toward +inf (HW answer, Q7)
        return x_floor + (remainder >= half)
    elif round_type == "trunc":
        return torch.where(x < 0, x_floor + (remainder != 0), x_floor)
    else:
        raise KeyError("[ERROR] : unknon type of rounding (posible options: floor | round | trunc)")


def relux(x : torch.Tensor, min : int, max : int):
    x_tmp = torch.clamp(x, min=min, max=max)
    return x_tmp.int()


@dataclass
class Cstruct:
    bias : torch.Tensor
    scale : torch.Tensor
    shift : torch.Tensor
    round_type : torch.Tensor
    min_val : torch.Tensor
    max_val : torch.Tensor


def pcu(p_sum: torch.Tensor, c_struct: Cstruct, e_min_layer=0):
    pla_psum = pcu_pla(p_sum, scale=c_struct.scale, bias=c_struct.bias)
    round_psum = pcu_div(pla_psum, shift_right=c_struct.shift+e_min_layer, round_type=c_struct.round_type)
    o_line = relux(round_psum, min=c_struct.min_val, max=c_struct.max_val)
    return o_line


class MXConv2D_Loops(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, min_val=-1.0, max_val=1.0,
        acc_bit_width: int = 48
    ):
        super(MXConv2D_Loops, self).__init__()

        if acc_bit_width not in SUPPORTED_ACC_WIDTHS:
            raise NotImplementedError(f"acc_bit_width must be one of {SUPPORTED_ACC_WIDTHS}")
        self.acc_bit_width = acc_bit_width

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, *self.kernel_size
        ))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        with torch.no_grad():
            fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
            self.weight.data /= math.sqrt(fan_in)
            self.bias.data.uniform_(min_val, max_val)

        self.mx_weight = None
        self.c_struct_arr: dict = {}

    def load_mx_weight(self, block_size: int = 32, mx_type: str = "mx-e4bm8", calibrate_offset: int = 6):
        # replaces the weight part of update_c_struct (calibration lives in calib.py)
        self.mx_weight = mx_weight_quantizer(self.weight, block_size=block_size, mx_type=mx_type, calibrate_offset=calibrate_offset)

    def load_c_struct(self, c_struct_arr: dict):
        # replaces the c_struct part of update_c_struct (calibration lives in calib.py)
        self.c_struct_arr = c_struct_arr

    def forward(self, mx: MXTensor, print_debug : bool = False):
        if self.mx_weight is None or not self.c_struct_arr:
            raise RuntimeError("MXConv2D_Loops is not calibrated: call load_mx_weight()/load_c_struct() first")

        batch_size, C_in, input_height, input_width = mx.original_shape
        K_h, K_w = self.kernel_size
        S_h, S_w = self.stride

        if C_in != self.in_channels:
            raise ValueError(
                f"Input has {C_in} channels, layer expects {self.in_channels}"
            )

        if self.padding[0] > 0 or self.padding[1] > 0:
            raise NotImplementedError("Padding not yet supported")

        # Align input mantissas with their per-element exponents
        x_data_flat, x_exp_flat = align_mx_to_elements(mx)
        out_height = (input_height + 2 * self.padding[0] - K_h) // S_h + 1
        out_width = (input_width + 2 * self.padding[1] - K_w) // S_w + 1

        output = torch.zeros(
            batch_size, self.out_channels, out_height, out_width,
            device=mx.data.device
        )

        # Convolution loops
        for b in range(batch_size):
            w_flat, exp_flat = tru_w(self.mx_weight)
            for oc in range(self.out_channels):
                tru_w_array = w_flat[oc]
                tru_w_exp_array = exp_flat[oc]
                for oh in range(out_height):
                    # order the input such that it run conv exactly as the npe
                    tru_a_data, tru_a_exp = tru_a(
                            x_data_flat,
                            x_exp_flat,
                            oh,
                            S_h,
                            S_w,
                            K_w,
                            K_h,
                            out_width,
                            b
                    )

                    # mimic the mac units computation
                    p_sum = bcu(
                        tru_a_data,
                        tru_a_exp,
                        tru_w_array,
                        tru_w_exp_array,
                        acc_bit_width=self.acc_bit_width
                    )
                    # mimic the post bcu computation
                    o_line = pcu(p_sum, c_struct=self.c_struct_arr[oc], e_min_layer=self.mx_weight.e_min_layer+mx.e_min_layer)

                    if print_debug:
                        print("tru_a_data", tru_a_data[0])
                        print("tru_a_exp", tru_a_exp[0])
                        print("tru_w_array", tru_w_array)
                        print("tru_w_exp_array", tru_w_exp_array)
                        print("p_sum", p_sum)
                        print("o_line", o_line)
                        raise Exception("cause i can")
                    # return a line in an output channel in a batch
                    output[b, oc, oh, :] = o_line

        return output
