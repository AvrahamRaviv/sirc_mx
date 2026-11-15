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

from npe_mx_emu.mx_format import MXTensor, mx_config_factory, mx_weight_quantizer


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


def tru_a_add(xa_data_flat: torch.Tensor,
              xa_exp_flat: torch.Tensor,
              xb_data_flat: torch.Tensor,
              xb_exp_flat: torch.Tensor,
              channel: int,
              line: int,
              batch: int):
    """TRU-A stage of the elementwise-add op.

    Both terms enter through the activation port, one element at a time, and
    land in the same accumulator: term 0 is input A, term 1 is input B.
    Stacks row ``line`` of ``channel``/``batch`` from each aligned input
    into per-position pairs.

    Returns:
        data_pairs: [W, 2]
        exp_pairs:  [W, 2]
    """
    a_data = xa_data_flat[batch, channel, line, :]
    a_exp = xa_exp_flat[batch, channel, line, :]
    b_data = xb_data_flat[batch, channel, line, :]
    b_exp = xb_exp_flat[batch, channel, line, :]

    data_pairs = torch.stack([a_data, b_data], dim=-1)
    exp_pairs = torch.stack([a_exp, b_exp], dim=-1)

    return data_pairs, exp_pairs


def tru_a_dw(x_data_flat: torch.Tensor,
             x_exp_flat: torch.Tensor,
             channel: int,
             line: int,
             S_h: int,
             S_w: int,
             K_w: int,
             K_h: int,
             out_width: int,
             batch: int):
    """TRU-A stage of the depthwise op.

    Each channel feeds its own private kernel (channel-as-kernel), so the
    A-line of one output row covers only ``channel``'s strip.  Element order
    per pixel is (kh, kw) -- matching the per-channel flattened weight
    (kernel flatten of [1, K_h, K_w]).

    Returns:
        patches:     [out_width, K_h*K_w]
        patches_exp: [out_width, K_h*K_w]
    """
    h_start = line * S_h
    strip_width = K_w + (out_width - 1) * S_w

    strip = x_data_flat[batch, channel, h_start:h_start + K_h, :strip_width]
    strip_exp = x_exp_flat[batch, channel, h_start:h_start + K_h, :strip_width]

    patches = strip.unfold(-1, K_w, S_w)
    patches_exp = strip_exp.unfold(-1, K_w, S_w)

    patches_flat = patches.permute(1, 0, 2).reshape(out_width, -1)
    patches_exp_flat = patches_exp.permute(1, 0, 2).reshape(out_width, -1)

    return patches_flat, patches_exp_flat



def split_msb_lsb(q: torch.Tensor, frac_bits):
    """Extract the msb/lsb split of a fixed-point integer word.

        idx  = q >> frac_bits          # msb: reference index (floor, signed)
        frac = q & (2**frac_bits - 1)  # lsb: fraction, always in [0, 2**frac_bits)

    Guarantees ``q == idx * 2**frac_bits + frac`` also for negative ``q``:
    the shift is arithmetic (floor) and the mask is two's-complement, so the
    fraction never goes negative.  This is the HW_WARP.md exponent/mantissa
    split: the index is the reference address, the fraction is the mantissa.

    ``frac_bits`` may be a scalar or broadcastable to ``q`` (e.g. per-axis
    ``[1, 2, 1, 1]``).  Returns ``(idx, frac)`` as long tensors.
    """
    if q.dtype not in (torch.int16, torch.int32, torch.int64):
        raise TypeError(f"expected an integer grid, got {q.dtype}")
    fr = torch.as_tensor(frac_bits).long()
    idx = q.long() >> fr
    frac = q.long() & ((torch.ones_like(fr) << fr) - 1)
    return idx, frac


def frac_bits_pair(frac_bits) -> tuple:
    """Normalize a per-axis fraction-bit spec (int, pair or tensor) to (frx, fry)."""
    if isinstance(frac_bits, torch.Tensor):
        vals = [int(v) for v in frac_bits.reshape(-1)]
    elif isinstance(frac_bits, (tuple, list)):
        vals = [int(v) for v in frac_bits]
    else:
        vals = [int(frac_bits)]
    if len(vals) == 1:
        vals = [vals[0], vals[0]]
    if len(vals) != 2:
        raise ValueError(f"frac_bits must be an int or a (frx, fry) pair, got {frac_bits!r}")
    return vals[0], vals[1]


def trc_warp(q_x: torch.Tensor,
             q_y: torch.Tensor,
             frac_bits,
             H_in: int,
             W_in: int):
    """TRC (Tensor Run Control) stage of the warp: address + weight generation.

    Consumes the int16 fixed-point grid ``q = round(coord * 2**frac)`` per axis
    (plain INT on an MX container whose exponent field is zero -- ``mx-e4bm16``,
    "NOT MX INT BUT INT"), performs the msb/lsb split itself:

        idx  = q >> frac_bits          (reference index -- HW_WARP.md "exponent")
        frac = q & (2**frac_bits - 1)  (fraction -- HW_WARP.md "mantissa")

    and produces, per output pixel:

      * ``idx_x_c``, ``idx_y_c``: corner read addresses clipped to
        ``[0, dim-2]`` so the ``+1`` neighbor always exists;
      * ``mx_weights``: the four bilinear weights as an ``mx-e4bm16`` MXTensor
        with a zero exponent field (the data lives in the 16-bit mantissa,
        one exponent per weight -> block_size 1).  Weight order matches the
        TRU-A corner order: (w00, w01, w10, w11) for
        (iy, ix), (iy, ix+1), (iy+1, ix), (iy+1, ix+1).

    Border semantics match ``np_warp`` (replicate): the fraction is recovered
    from the *unclipped* integer coordinate, so at the borders the 1-D weights
    saturate to (S, 0) instead of wrapping.

    Args:
        q_x, q_y:  [H_out, W_out] int16 fixed-point coordinates
        frac_bits: scalar or (frx, fry) pair
        H_in, W_in: reference frame height/width
    """
    frx, fry = frac_bits_pair(frac_bits)
    if frx < 0 or fry < 0:
        raise ValueError(f"frac_bits must be non-negative, got {(frx, fry)}")
    if not (0 <= frx <= 8 and 0 <= fry <= 8):
        raise ValueError(
            f"weight scale 2**{frx + fry} overflows the 16-bit weight mantissa "
            f"(mx-e4bm16 holds up to 2**14); frac_bits={(frx, fry)}"
        )
    Sx, Sy = 1 << frx, 1 << fry

    idx_x, f_x = split_msb_lsb(q_x, frx)
    idx_y, f_y = split_msb_lsb(q_y, fry)

    idx_x_c = torch.clamp(idx_x, 0, W_in - 2)
    idx_y_c = torch.clamp(idx_y, 0, H_in - 2)

    # effective fraction after the clip (replicate semantics at the borders)
    wx1 = torch.clamp((idx_x - idx_x_c) * Sx + f_x, 0, Sx)
    wy1 = torch.clamp((idx_y - idx_y_c) * Sy + f_y, 0, Sy)
    wx0 = Sx - wx1
    wy0 = Sy - wy1

    w00 = wx0 * wy0
    w01 = wx1 * wy0
    w10 = wx0 * wy1
    w11 = wx1 * wy1
    weights = torch.stack([w00, w01, w10, w11], dim=-1)   # [H_out, W_out, 4]

    cfg = mx_config_factory["mx-e4bm16"]
    mx_weights = MXTensor(
        shared_exp=torch.zeros_like(weights.long()),       # one zero exponent per weight
        data=weights.long(),
        original_shape=tuple(weights.shape),
        exponent_bits=cfg.exponent_bits,
        mantissa_bits=cfg.mantissa_bits,
        e_min_layer=torch.tensor(0),
        calibrate_offset=torch.tensor(0),
    )
    return idx_x_c, idx_y_c, mx_weights


def tru_a_warp(x_data_flat: torch.Tensor,
               x_exp_flat: torch.Tensor,
               iy_c: torch.Tensor,
               ix_c: torch.Tensor,
               channel: int,
               line: int,
               batch: int):
    """TRU-A stage of the warp: fetch the four reference corners of one line.

    Two reference lines (``iy``, ``iy + 1``) are read; per output pixel the
    corners are (iy, ix), (iy, ix+1), (iy+1, ix), (iy+1, ix+1) -- matching the
    TRC weight order.  Returns data/exp lines shaped ``[W_out, 4]``.
    """
    iy0 = iy_c[line]
    ix0 = ix_c[line]

    tru_a_data = torch.stack([
        x_data_flat[batch, channel, iy0, ix0],
        x_data_flat[batch, channel, iy0, ix0 + 1],
        x_data_flat[batch, channel, iy0 + 1, ix0],
        x_data_flat[batch, channel, iy0 + 1, ix0 + 1],
    ], dim=-1)

    tru_a_exp = torch.stack([
        x_exp_flat[batch, channel, iy0, ix0],
        x_exp_flat[batch, channel, iy0, ix0 + 1],
        x_exp_flat[batch, channel, iy0 + 1, ix0],
        x_exp_flat[batch, channel, iy0 + 1, ix0 + 1],
    ], dim=-1)

    return tru_a_data, tru_a_exp


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

    prod_mant = x_data.long() * w_data.long()
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


class MXDepthwise2D_Loops(nn.Module):
    """Depthwise conv through the NPE pipe: TRU -> BCU -> PCU.

    HW answers (algo table Q8): runs on the same MAC grid and the same MX
    integer data path as dense conv; each channel is treated as a separate
    kernel (one [1, K_h, K_w] filter per channel, no cross-channel mixing).
    Weight blocking is the "pad" option: each channel filter flattens to
    K_h*K_w real values and base_mx_quantizer zero-pads the tail of its
    single 32-wide block.
    """

    def __init__(self, channels, kernel_size,
                 stride=1, padding=0, min_val=-1.0, max_val=1.0,
                 acc_bit_width: int = 48):
        super(MXDepthwise2D_Loops, self).__init__()

        if acc_bit_width not in SUPPORTED_ACC_WIDTHS:
            raise NotImplementedError(f"acc_bit_width must be one of {SUPPORTED_ACC_WIDTHS}")
        self.acc_bit_width = acc_bit_width

        self.channels = channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding

        self.weight = nn.Parameter(torch.randn(
            channels, 1, *self.kernel_size
        ))
        self.bias = nn.Parameter(torch.zeros(channels))

        with torch.no_grad():
            fan_in = self.kernel_size[0] * self.kernel_size[1]
            self.weight.data /= math.sqrt(fan_in)
            self.bias.data.uniform_(min_val, max_val)

        self.mx_weight = None
        self.c_struct_arr: dict = {}

    def load_mx_weight(self, block_size: int = 32, mx_type: str = "mx-e4bm8", calibrate_offset: int = 6):
        # same weight path as conv; each channel filter flattens to K_h*K_w
        # real values (pad option -- a single zero-filled block per filter)
        self.mx_weight = mx_weight_quantizer(self.weight, block_size=block_size, mx_type=mx_type, calibrate_offset=calibrate_offset)

    def load_c_struct(self, c_struct_arr: dict):
        self.c_struct_arr = c_struct_arr

    def forward(self, mx: MXTensor, print_debug : bool = False):
        if self.mx_weight is None or not self.c_struct_arr:
            raise RuntimeError("MXDepthwise2D_Loops is not calibrated: call load_mx_weight()/load_c_struct() first")

        batch_size, C_in, input_height, input_width = mx.original_shape
        K_h, K_w = self.kernel_size
        S_h, S_w = self.stride

        if C_in != self.channels:
            raise ValueError(
                f"Input has {C_in} channels, layer expects {self.channels}"
            )

        if self.padding[0] > 0 or self.padding[1] > 0:
            raise NotImplementedError("Padding not yet supported")

        # Align input mantissas with their per-element exponents
        x_data_flat, x_exp_flat = align_mx_to_elements(mx)
        out_height = (input_height + 2 * self.padding[0] - K_h) // S_h + 1
        out_width = (input_width + 2 * self.padding[1] - K_w) // S_w + 1

        output = torch.zeros(
            batch_size, self.channels, out_height, out_width,
            device=mx.data.device
        )

        # Depthwise loops -- same stage order as conv: tru(_a/_w) -> bcu -> pcu,
        # but each channel carries its own private kernel (channel loop plays
        # the role of the conv's output-channel loop).
        for b in range(batch_size):
            w_flat, exp_flat = tru_w(self.mx_weight)
            for c in range(self.channels):
                tru_w_array = w_flat[c]
                tru_w_exp_array = exp_flat[c]
                for oh in range(out_height):
                    # order the input of channel c exactly as the npe feeds it
                    tru_a_data, tru_a_exp = tru_a_dw(
                            x_data_flat,
                            x_exp_flat,
                            c,
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
                    o_line = pcu(p_sum, c_struct=self.c_struct_arr[c], e_min_layer=self.mx_weight.e_min_layer+mx.e_min_layer)

                    if print_debug:
                        print("tru_a_data", tru_a_data[0])
                        print("tru_a_exp", tru_a_exp[0])
                        print("tru_w_array", tru_w_array)
                        print("tru_w_exp_array", tru_w_exp_array)
                        print("p_sum", p_sum)
                        print("o_line", o_line)
                        raise Exception("cause i can")
                    # return a line in a channel in a batch
                    output[b, c, oh, :] = o_line

        return output


class MXAdd_Loops(nn.Module):
    """Elementwise add (A + B) through the NPE pipe: TRU -> BCU -> PCU.

    The add runs on the BCU accumulator as a 2-term dot product per output
    element, so the multiplier is skipped: its B operand is the constant
    ones vector.  ``weight`` is shaped [C, 1, 1, 2] so the existing weight
    quantizer / tru_w treat each channel row as a flattened 2-element
    "kernel" ([s_c, s_c]); replacing the ones with an optional per-channel
    float scale gives out_c = scale_c * (A_c + B_c).

    Both inputs arrive on the same (shared) calibration scale -- x_cal --
    exactly as two residual streams would on HW.
    """

    def __init__(
        self, channels,
        min_val=-1.0, max_val=1.0,
        acc_bit_width: int = 48
    ):
        super(MXAdd_Loops, self).__init__()

        if acc_bit_width not in SUPPORTED_ACC_WIDTHS:
            raise NotImplementedError(f"acc_bit_width must be one of {SUPPORTED_ACC_WIDTHS}")
        self.acc_bit_width = acc_bit_width

        self.channels = channels

        self.weight = nn.Parameter(torch.ones(channels, 1, 1, 2))
        self.bias = nn.Parameter(torch.zeros(channels))

        self.mx_weight = None
        self.c_struct_arr: dict = {}

    def load_mx_weight(self, block_size: int = 32, mx_type: str = "mx-e4bm8", calibrate_offset: int = 6):
        # same weight path as conv; for ones + calibrate_offset == mantissa_offset
        # the vector encodes as mantissa 64 / shared_exp 0 (weight scale stays zero)
        self.mx_weight = mx_weight_quantizer(self.weight, block_size=block_size, mx_type=mx_type, calibrate_offset=calibrate_offset)

    def load_c_struct(self, c_struct_arr: dict):
        self.c_struct_arr = c_struct_arr

    def forward(self, mx_a: MXTensor, mx_b: MXTensor, print_debug: bool = False):
        if self.mx_weight is None or not self.c_struct_arr:
            raise RuntimeError("MXAdd_Loops is not calibrated: call load_mx_weight()/load_c_struct() first")

        if mx_a.original_shape != mx_b.original_shape:
            raise ValueError(
                f"Elementwise add requires equal input shapes, got {mx_a.original_shape} and {mx_b.original_shape}"
            )

        batch_size, C_in, input_height, input_width = mx_a.original_shape

        if C_in != self.channels:
            raise ValueError(
                f"Input has {C_in} channels, layer expects {self.channels}"
            )

        # TODO OPENCODE - FIX THIS SECTION 
        # Align both input mantissas with their per-element exponents
        xa_data_flat, xa_exp_flat = align_mx_to_elements(mx_a)
        xb_data_flat, xb_exp_flat = align_mx_to_elements(mx_b)

        output = torch.zeros(
            batch_size, self.channels, input_height, input_width,
            device=mx_a.data.device
        )

        # Elementwise-add loops -- same stage order as conv: tru(_a/_w) -> bcu -> pcu
        for b in range(batch_size):
            w_flat, exp_flat = tru_w(self.mx_weight)
            for c in range(self.channels):
                tru_w_array = w_flat[c]
                tru_w_exp_array = exp_flat[c]
                for ih in range(input_height):
                    # stream the two inputs term by term, exactly as the npe feeds them
                    tru_a_data, tru_a_exp = tru_a_add(
                        xa_data_flat,
                        xa_exp_flat,
                        xb_data_flat,
                        xb_exp_flat,
                        c,
                        ih,
                        b
                    )

                    # mimic the mac units computation (2 terms per position: A then B)
                    p_sum = bcu(
                        tru_a_data,
                        tru_a_exp,
                        tru_w_array,
                        tru_w_exp_array,
                        acc_bit_width=self.acc_bit_width
                    )

                    # mimic the post bcu computation
                    o_line = pcu(p_sum, c_struct=self.c_struct_arr[c], e_min_layer=self.mx_weight.e_min_layer+mx_a.e_min_layer)

                    if print_debug:
                        print("tru_a_data", tru_a_data[0])
                        print("tru_a_exp", tru_a_exp[0])
                        print("tru_w_array", tru_w_array)
                        print("tru_w_exp_array", tru_w_exp_array)
                        print("p_sum", p_sum)
                        print("o_line", o_line)
                        raise Exception("cause i can")
                    # return a line in a channel in a batch
                    output[b, c, ih, :] = o_line

        return output

def gen_warp_weights(x_frac, y_frac, frac_bits):
    w00 = (2 ** frac_bits[0][0] - x_frac) * (2 ** frac_bits[0][1] - y_frac)
    w10 = (2 ** frac_bits[0][0] - x_frac) * (y_frac)
    w01 = (x_frac) * (2 ** frac_bits[0][1] - y_frac)
    w11 = (x_frac) * (y_frac)

    return w00, w01, w10, w11


def get_ref_frame(src, x_offset, y_offset):
    # this is a relative grid and to be able to handle the last element in the line it copied
    src_height, src_width = src.shape[2:]

    xs0 = torch.clamp(torch.floor(x_offset), 0, src_width - 1).long()
    ys0 = torch.clamp(torch.floor(y_offset), 0, src_height - 1).long()

    ref00 = src[..., ys0, xs0]
    ref01 = src[..., ys0, (xs0 + 1).clamp(max=src_width - 1)]
    ref10 = src[..., (ys0 + 1).clamp(max=src_height - 1), xs0]
    ref11 = src[..., (ys0 + 1).clamp(max=src_height - 1), (xs0 + 1).clamp(max=src_width - 1)]

    return ref00, ref01, ref10, ref11


def trc(quantized_grid, src, frac_bits):
    frx, fry = frac_bits_pair(frac_bits)
    h, w = src.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(h, device=src.device),
                            torch.arange(w, device=src.device), indexing="ij")
    qx = (quantized_grid[0, 0].long() + (xx << frx)).clamp(0, (w - 1) << frx)
    qy = (quantized_grid[0, 1].long() + (yy << fry)).clamp(0, (h - 1) << fry)
    x_offset, x_frac = split_msb_lsb(qx, frx)
    y_offset, y_frac = split_msb_lsb(qy, fry)
    # interpulate_weights = torch.stack(gen_warp_weights(x_frac, y_frac))
    # frame_coords = torch.stack(get_ref_frame(src, ys0, xs0))

    tru_a = gen_warp_weights(x_frac, y_frac, frac_bits)
    tru_w = get_ref_frame(src, x_offset, y_offset)

    tru_a = torch.stack(tru_a).permute(1,2,0)
    tru_w = torch.stack(tru_w).permute(1,2,3,4,0)
    
    return tru_a, tru_w


class MXWarp_Loops(nn.Module):
    """Warp through the NPE pipe: TRC -> TRU-A / TRU-W -> BCU -> PCU.

    The TRC (``trc_warp``) is the new stage: it turns the int16 fixed-point
    grid into (a) the four corner read addresses and (b) the four bilinear
    weights -- what the pipe then consumes exactly like a conv's TRU-A/TRU-W
    operands.  Everything is plain INT on MX containers with a zero exponent
    field ("NOT MX INT BUT INT"):

      * reference ("tru_a"): uint8 values, ``mx-e4bm8``, exp = 0;
      * grid (input):        int16 ``q = round(coord * 2**fr)`` per axis,
                             ``mx-e4bm16``, exp = 0;
      * weights ("tru_w"):   generated by the TRC from the grid, integer in
                             ``[0, 2**(frx+fry)]``, ``mx-e4bm16``, exp = 0 --
                             the 16-bit mantissa is required because the
                             weight spans up to 2**14 and does not fit mxint8.

    BCU MACs the 4 pixel/weight terms per output pixel per channel (same
    ``bcu``), PCU divides by ``2 ** (frx + fry)`` with the same ``pcu``
    (round half up + relux).
    """

    def __init__(self, channels, acc_bit_width: int = 48):
        super(MXWarp_Loops, self).__init__()

        if acc_bit_width not in SUPPORTED_ACC_WIDTHS:
            raise NotImplementedError(f"acc_bit_width must be one of {SUPPORTED_ACC_WIDTHS}")
        self.acc_bit_width = acc_bit_width

        self.channels = channels

        self.bias = nn.Parameter(torch.zeros(channels))
        self.c_struct_arr: dict = {}

    def load_c_struct(self, c_struct_arr: dict):
        self.c_struct_arr = c_struct_arr

    def forward(self, mx_ref: MXTensor, mx_grid: MXTensor, frac_bits,
                print_debug: bool = False):
        if not self.c_struct_arr:
            raise RuntimeError("MXWarp_Loops is not calibrated: call load_c_struct() first")

        batch_size, C_in, input_height, input_width = mx_ref.original_shape

        if C_in != self.channels:
            raise ValueError(
                f"Reference has {C_in} channels, layer expects {self.channels}"
            )

        out_height, out_width = mx_grid.data.shape[2], mx_grid.data.shape[3]

        output = torch.zeros(
            batch_size, self.channels, out_height, out_width,
            device=mx_ref.data.device, dtype=torch.int64
        )

        tru_a_elem_data, tru_w_elem_data = trc(mx_grid.data, mx_ref.data, frac_bits)
        tru_a_elem_exp, tru_w_elem_exp = torch.zeros_like(tru_a_elem_data), torch.zeros_like(tru_w_elem_data)
        
        for b in range(batch_size):

            for c in range(self.channels):
                for oh in range(out_height):
                    tru_a_inp = tru_a_elem_data[oh]
                    tru_a_exp = tru_a_elem_exp[oh]

                    tru_w_inp = tru_w_elem_data[b][c][oh]
                    tru_w_exp = tru_w_elem_exp[b][c][oh]
                    
                    # BCU: 4 terms per output pixel
                    p_sum = bcu(
                        tru_a_inp,
                        tru_a_exp,
                        tru_w_inp,
                        tru_w_exp,
                        acc_bit_width=self.acc_bit_width
                    )

                    # PCU: same post-BCU path as conv
                    
                    # Warning: this is a workaround 
                    # TODO in a network use the scale from frac_bits, obtained by pack_warp_grid to compute the actual o_line
                    # this code is only to simplify this
                    # o_line = pcu(p_sum, c_struct=self.c_struct_arr[c], e_min_layer=mx_ref.e_min_layer)
                    o_line = pcu(p_sum, c_struct=self.c_struct_arr[c], e_min_layer=mx_ref.e_min_layer) 
                    # return a line in a channel in a batch
                    output[b, c, oh, :] = o_line


        return output
