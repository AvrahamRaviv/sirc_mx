"""Torch-tensor-in / raw-int-out conv layer wrapping the NPE HW emulation pipe."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from npe_mx_emu.calib import LayerMXQParams, build_c_struct_arr, build_warp_c_struct_arr
from npe_mx_emu.hw_pipe import MXAdd_Loops, MXConv2D_Loops, MXConvTranspose2D_Loops, MXDepthwise2D_Loops, MXWarp_Loops, \
    frac_bits_pair
from npe_mx_emu.mx_format import MXTensor, base_mx_quantizer, input_mx_quantizer, mantissa_offset, \
    mx_config_factory


class CalibratedMXConv2d(nn.Module):
    """MX conv layer driven by observer-derived LayerMXQParams.

    forward() takes a plain float torch.Tensor (like nn.Conv2d) and returns
    the raw integer accumulator output; the float value is
    ``output * 2**-output_scale_log2``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        min_val: float = -1.0,
        max_val: float = 1.0,
        qparams: "LayerMXQParams | None" = None,
        dilation=1,
    ):
        super(CalibratedMXConv2d, self).__init__()
        if qparams is None:
            raise ValueError("qparams (LayerMXQParams) is required")
        self.qparams = qparams

        self.mx_conv = MXConv2D_Loops(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            min_val=min_val,
            max_val=max_val,
        )

    @classmethod
    def from_torch_conv(cls, conv: nn.Conv2d, qparams: LayerMXQParams) -> "CalibratedMXConv2d":
        kernel_size = conv.kernel_size[0] if conv.kernel_size[0] == conv.kernel_size[1] else conv.kernel_size
        stride = conv.stride[0] if conv.stride[0] == conv.stride[1] else conv.stride
        padding = conv.padding[0] if conv.padding[0] == conv.padding[1] else conv.padding
        dilation = conv.dilation[0] if conv.dilation[0] == conv.dilation[1] else conv.dilation
        layer = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            qparams=qparams,
        )
        with torch.no_grad():
            layer.mx_conv.weight.copy_(conv.weight)
            layer.mx_conv.bias.copy_(conv.bias)
        return layer

    @property
    def weight(self):
        return self.mx_conv.weight

    @property
    def bias(self):
        return self.mx_conv.bias

    @property
    def output_scale_log2(self) -> int:
        return self.qparams.output_scale_log2

    def calibrate(self):
        """(Re)quantize weights and rebuild the PCU c_structs. Idempotent."""
        qp = self.qparams
        self.mx_conv.load_mx_weight(
            block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.w_cal
        )
        self.mx_conv.load_c_struct(
            build_c_struct_arr(qp, self.mx_conv.bias.data, self.mx_conv.out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mx_conv.mx_weight is None:
            self.calibrate()

        qp = self.qparams
        x_scaled = x * 2.0 ** qp.x_cal
        if qp.is_input_layer:
            # The round-to-integer of x*2^cal happens exactly ONCE in the
            # network, at the first layer (HW-confirmed, algo table Q3): the
            # BCU shifters are unsigned/left-only, so operands must sit at a
            # scale whose exponents are all non-negative -- i.e. integers.
            mx = input_mx_quantizer(x_scaled.round(), block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.x_cal)
        else:
            # Intermediate layers: NO pre-round -- the activation arrives in
            # integer domain from the PCU (its shift_right realigns the scale
            # between layers), and the MX quantizer's own per-block truncate
            # is the only integer-ization.  A float boundary tensor therefore
            # goes straight into the quantizer.
            mx = base_mx_quantizer(x_scaled, block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.x_cal)
        out_raw = self.mx_conv(mx)
        return out_raw


class CalibratedMXDepthwiseConv2d(nn.Module):
    """MX depthwise conv layer driven by observer-derived LayerMXQParams.

    Depthwise on the NPE (HW answer, algo Q8): each channel is a separate
    kernel on the same MAC grid and the same MX integer data path --
    out_c = in_c (*) w_c, no cross-channel terms.  One padded weight block
    per channel filter.  forward() takes a plain float torch.Tensor and
    returns the raw integer accumulator output; the float value is
    ``output * 2**-output_scale_log2``.
    """

    def __init__(
        self,
        channels: int,
        kernel_size,
        stride=1,
        padding=0,
        min_val: float = -1.0,
        max_val: float = 1.0,
        qparams: "LayerMXQParams | None" = None,
        dilation=1,
    ):
        super(CalibratedMXDepthwiseConv2d, self).__init__()
        if qparams is None:
            raise ValueError("qparams (LayerMXQParams) is required")
        self.qparams = qparams

        self.mx_dw = MXDepthwise2D_Loops(
            channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            min_val=min_val,
            max_val=max_val,
        )

    @classmethod
    def from_torch_conv(cls, conv: nn.Conv2d, qparams: LayerMXQParams) -> "CalibratedMXDepthwiseConv2d":
        if conv.groups != conv.in_channels or conv.out_channels != conv.in_channels:
            raise ValueError(
                "from_torch_conv expects a depthwise nn.Conv2d "
                "(groups == in_channels == out_channels)"
            )
        kernel_size = conv.kernel_size[0] if conv.kernel_size[0] == conv.kernel_size[1] else conv.kernel_size
        stride = conv.stride[0] if conv.stride[0] == conv.stride[1] else conv.stride
        padding = conv.padding[0] if conv.padding[0] == conv.padding[1] else conv.padding
        dilation = conv.dilation[0] if conv.dilation[0] == conv.dilation[1] else conv.dilation
        layer = cls(
            channels=conv.in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            qparams=qparams,
        )
        with torch.no_grad():
            layer.mx_dw.weight.copy_(conv.weight)
            layer.mx_dw.bias.copy_(conv.bias)
        return layer

    @property
    def weight(self):
        return self.mx_dw.weight

    @property
    def bias(self):
        return self.mx_dw.bias

    @property
    def output_scale_log2(self) -> int:
        return self.qparams.output_scale_log2

    def calibrate(self):
        """(Re)quantize weights and rebuild the PCU c_structs. Idempotent."""
        qp = self.qparams
        self.mx_dw.load_mx_weight(
            block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.w_cal
        )
        self.mx_dw.load_c_struct(
            build_c_struct_arr(qp, self.mx_dw.bias.data, self.mx_dw.channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mx_dw.mx_weight is None:
            self.calibrate()

        qp = self.qparams
        x_scaled = x * 2.0 ** qp.x_cal
        if qp.is_input_layer:
            # The round-to-integer of x*2^cal happens exactly ONCE in the
            # network, at the first layer (HW-confirmed, algo table Q3): the
            # BCU shifters are unsigned/left-only, so operands must sit at a
            # scale whose exponents are all non-negative -- i.e. integers.
            mx = input_mx_quantizer(x_scaled.round(), block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.x_cal)
        else:
            # Intermediate layers: NO pre-round -- the activation arrives in
            # integer domain from the PCU (its shift_right realigns the scale
            # between layers), and the MX quantizer's own per-block truncate
            # is the only integer-ization.  A float boundary tensor therefore
            # goes straight into the quantizer.
            mx = base_mx_quantizer(x_scaled, block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.x_cal)
        out_raw = self.mx_dw(mx)
        return out_raw


class CalibratedMXConvTranspose2d(nn.Module):
    """MX transposed conv layer driven by observer-derived LayerMXQParams.

    Geometry: stride-s zero-insert of the activation + stride-1 dense conv
    (nn.ConvTranspose2d semantics for groups=1, output_padding=0).
    `dilation` is torch's KERNEL dilation (weight-side tap spacing).

    Public weight stays in torch ConvTranspose2d layout (in, out, K, K); the
    transpose+flip conversion happens inside ``load_mx_weight`` BEFORE weight
    quantization.  IMPORTANT: observe/calibrate weights against the CONVERTED
    tensor ``w.transpose(0, 1).flip(-1, -2)`` -- the conversion changes which
    elements group per OC, hence the block statistics.

    forward() takes a plain float torch.Tensor (like nn.ConvTranspose2d) and
    returns the raw integer accumulator output; the float value is
    ``output * 2**-output_scale_log2``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        min_val: float = -1.0,
        max_val: float = 1.0,
        qparams: "LayerMXQParams | None" = None,
        dilation=1,
    ):
        super(CalibratedMXConvTranspose2d, self).__init__()
        if qparams is None:
            raise ValueError("qparams (LayerMXQParams) is required")
        self.qparams = qparams

        self.mx_convt = MXConvTranspose2D_Loops(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
            min_val=min_val,
            max_val=max_val,
        )

    @classmethod
    def from_torch_conv(cls, convT: "nn.ConvTranspose2d", qparams: LayerMXQParams) -> "CalibratedMXConvTranspose2d":
        # groups != 1 and output_padding != 0 are rejected by the engine ctor
        kernel_size = convT.kernel_size[0] if convT.kernel_size[0] == convT.kernel_size[1] else convT.kernel_size
        stride = convT.stride[0] if convT.stride[0] == convT.stride[1] else convT.stride
        padding = convT.padding[0] if convT.padding[0] == convT.padding[1] else convT.padding
        dilation = convT.dilation[0] if convT.dilation[0] == convT.dilation[1] else convT.dilation
        layer = cls(
            in_channels=convT.in_channels,
            out_channels=convT.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=convT.output_padding,
            groups=convT.groups,
            dilation=dilation,
            qparams=qparams,
        )
        with torch.no_grad():
            layer.mx_convt.weight.copy_(convT.weight)
            layer.mx_convt.bias.copy_(convT.bias)
        return layer

    @property
    def weight(self):
        return self.mx_convt.weight

    @property
    def bias(self):
        return self.mx_convt.bias

    @property
    def output_scale_log2(self) -> int:
        return self.qparams.output_scale_log2

    def calibrate(self):
        """(Re)quantize the (converted) weights and rebuild the PCU c_structs. Idempotent."""
        qp = self.qparams
        self.mx_convt.load_mx_weight(
            block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.w_cal
        )
        self.mx_convt.load_c_struct(
            build_c_struct_arr(qp, self.mx_convt.bias.data, self.mx_convt.out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mx_convt.mx_weight is None:
            self.calibrate()

        qp = self.qparams
        x_scaled = x * 2.0 ** qp.x_cal
        if qp.is_input_layer:
            # round-to-integer of x*2^cal happens ONLY at the first layer (Q3)
            mx = input_mx_quantizer(x_scaled.round(), block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.x_cal)
        else:
            # intermediate layers: no pre-round; quantizer truncate is the only
            # integer-ization (Q3)
            mx = base_mx_quantizer(x_scaled, block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.x_cal)
        out_raw = self.mx_convt(mx)
        return out_raw


class CalibratedMXAdd(nn.Module):
    """MX elementwise-add layer driven by observer-derived LayerMXQParams.

    forward() takes two plain float torch.Tensors (like torch.add) and returns
    the raw integer accumulator output; the float value is
    ``output * 2**-output_scale_log2``.  Both inputs share the layer qparams
    (residual scales must be pre-aligned before the add, as on HW).
    """

    def __init__(
        self,
        channels: int,
        min_val: float = -1.0,
        max_val: float = 1.0,
        qparams: "LayerMXQParams | None" = None,
    ):
        super(CalibratedMXAdd, self).__init__()
        if qparams is None:
            raise ValueError("qparams (LayerMXQParams) is required")
        self.qparams = qparams

        self.mx_add = MXAdd_Loops(
            channels=channels,
            min_val=min_val,
            max_val=max_val,
        )

    @property
    def weight(self):
        return self.mx_add.weight

    @property
    def bias(self):
        return self.mx_add.bias

    @property
    def output_scale_log2(self) -> int:
        return self.qparams.output_scale_log2

    def set_scale(self, scale: torch.Tensor):
        """Optional per-channel float scale, replacing the default ones vector:
        out_c = scale_c * (A_c + B_c).  Takes effect on the next calibrate().
        Calibrate its w_cal with MXObserver(mode="mx", per_row_flatten=True)
        over ``layer.weight`` after setting -- same rule as conv weights (no
        negative block exponents, so scales below 1 need a larger w_cal).
        """
        scale = torch.as_tensor(scale, dtype=self.mx_add.weight.dtype, device=self.mx_add.weight.device)
        if scale.numel() != self.mx_add.channels:
            raise ValueError(f"scale has {scale.numel()} channels, layer expects {self.mx_add.channels}")
        with torch.no_grad():
            self.mx_add.weight.data[:, 0, 0, :] = scale.reshape(-1, 1)

    def calibrate(self):
        """(Re)quantize the ones/scale vector and rebuild the PCU c_structs. Idempotent."""
        qp = self.qparams
        self.mx_add.load_mx_weight(
            block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.w_cal
        )
        self.mx_add.load_c_struct(
            build_c_struct_arr(qp, self.mx_add.bias.data, self.mx_add.channels)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.mx_add.mx_weight is None:
            self.calibrate()

        qp = self.qparams
        x_scaled = x * 2.0 ** qp.x_cal
        y_scaled = y * 2.0 ** qp.x_cal
        if qp.is_input_layer:
            # Same single round-to-integer rule as CalibratedMXConv2d (algo Q3),
            # applied to each of the two input streams.
            mx_a = input_mx_quantizer(x_scaled.round(), block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.x_cal)
            mx_b = input_mx_quantizer(y_scaled.round(), block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.x_cal)
        else:
            mx_a = base_mx_quantizer(x_scaled, block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.x_cal)
            mx_b = base_mx_quantizer(y_scaled, block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.x_cal)
        out_raw = self.mx_add(mx_a, mx_b)
        return out_raw


def pack_warp_reference(reference: torch.Tensor, mx_type: str = "mx-e4bm8", block_size: int = 32) -> MXTensor:
    """Pack the warp reference as integer codes with zero exponents (plain INT).

    The reference is NOT quantized to MX: just ``round + clamp`` into the
    selected integer range and store with an all-zero exponent field (the first-layer
    integer-input rule).  ``e_min_layer = mantissa_offset`` (calibrate_offset
    0) so the PCU's effective shift becomes ``out_cal - (frx + fry)``.

    Padding/blocking along W mirrors the activation quantizers
    (``[B, C, H, num_blocks, block_size]``), which is what
    ``align_mx_to_elements`` consumes.

    Returns:
        MXTensor with ``data`` = int values in [0, 255], zero ``shared_exp``.
    """
    cfg = mx_config_factory[mx_type]
    if reference.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        # integer reference (sensor format): only clamp, rounding is a no-op
        q = reference.clamp(0, 255).long() if reference.dtype == torch.uint8 else reference.clamp(-128, 127).long()
    else:
        q = torch.round(reference).clamp(-128, 127).long()
    W = q.shape[-1]
    num_blocks = math.ceil(W / block_size)
    pad = num_blocks * block_size - W
    # blocks = q.view(*q.shape[:-1], num_blocks, block_size)
    blocks = q
    return MXTensor(
        shared_exp=torch.zeros(blocks.shape[:-1], dtype=torch.long),
        data=blocks,
        original_shape=tuple(reference.shape),
        exponent_bits=cfg.exponent_bits,
        mantissa_bits=cfg.mantissa_bits,
        e_min_layer=torch.tensor(mantissa_offset(mx_type)),
        calibrate_offset=torch.tensor(0),
    )


def grid_quantization(
    grid,
    grid_bit_width: int
):

    delta_grid = grid
    max_motion = torch.ceil(torch.abs(delta_grid.reshape((-1, 2))).max(dim=0).values)
    frac_bits = grid_bit_width - torch.floor(torch.log2(max_motion)) - 2
    # TODO add assertions to avoid 0 frac bits and more than 8 frac bits
    delta_grid = torch.round(delta_grid * frac_bits.exp2()).to(torch.int16)
    delta_grid = delta_grid.permute(2, 0, 1)[None, ...]
    
    return delta_grid, max_motion, frac_bits.reshape(1, 2, 1, 1)


def pack_warp_grid(coords: torch.Tensor, grid_bit_width: int = 16,
                   mx_type: str = "mx-e4bm16", frac_bits=(8, 8)) -> tuple:
    """Pack an encoded int16 relative grid with configured per-axis fraction bits."""
    if coords.dtype != torch.int16:
        raise TypeError("warp grid must be encoded int16 relative motion")
    frx, fry = frac_bits_pair(frac_bits)
    if not (0 <= frx <= 8 and 0 <= fry <= 8):
        raise ValueError("per-axis fraction bits must be in 0..8")
    frac_bits = torch.tensor([frx, fry], device=coords.device, dtype=torch.long).reshape(1, 2, 1, 1)
    quantized_grid = coords

    cfg = mx_config_factory[mx_type]

    # per-axis adaptive fraction bits (same rule as the notebook grid_quantization)

    mx_grid = MXTensor(
        shared_exp=torch.zeros_like(quantized_grid),                 # plain INT: zero exponent
        data=quantized_grid,
        original_shape=tuple(quantized_grid.shape),
        exponent_bits=cfg.exponent_bits,
        mantissa_bits=cfg.mantissa_bits,
        e_min_layer=torch.tensor(0),
        calibrate_offset=torch.tensor(0),
    )
    return mx_grid, frac_bits

def grid_sample(tensor: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid.

    Args:
        tensor: (N, C, H_in, W_in) tensor
        grid: (N, H_out, W_out, 2) tensor in the range of [-1, 1]

    Returns:
        (N, C, H_out, W_out) tensor

    See `torch.nn.functional.grid_sample`.
    """
    b, c, h, w = tensor.shape
    b_, h_out, w_out, w_ = grid.shape
    assert b == b_ and w_ == 2
    out = []
    for (t, g) in zip(tensor, grid):
        x_ = 0.5 * (w - 1) * (g[..., 0].reshape(-1) + 1)
        y_ = 0.5 * (h - 1) * (g[..., 1].reshape(-1) + 1)
        ix = torch.floor(x_).clamp(0, w - 2).long()
        iy = torch.floor(y_).clamp(0, h - 2).long()
        dx = x_ - ix
        dy = y_ - iy
        out.append((1 - dx) * (1 - dy) * t[..., iy, ix]
                   + dx * (1 - dy) * t[..., iy, ix + 1]
                   + (1 - dx) * dy * t[..., iy + 1, ix]
                   + dx * dy * t[..., iy + 1, ix + 1])
    return torch.cat(out, dim=0).reshape(b, c, h_out, w_out)


def sw_warp(ref, grid):
    new_h, new_w = grid.shape[-3:-1]
    orig_h, orig_w = ref.shape[-2:]
    grid_norm_scale = 2 / torch.tensor([orig_w - 1, orig_h - 1], dtype=torch.float64).reshape((1, 1, 1, -1))
    grid_normed = grid_norm_scale * grid - 1
    expected = grid_sample(ref, grid_normed)

    return expected

class CalibratedMXWarp(nn.Module):
    """Warp layer driven by LayerMXQParams through the TRC-based NPE pipe.

    HW scheme (HW_WARP.md): the TRC stage generates the four reference read
    addresses and the four bilinear weights from the int16 fixed-point grid
    (``q = round(coord * 2**frac)``, ``mx-e4bm16``, zero exponent); TRU-A
    fetches the four corners; TRU-W carries the TRC weights (integers over
    ``2**(frx+fry)`` -- the mxint16 mantissa is required because the weight
    spans up to 2**14, which does not fit mxint8); the BCU MACs the four
    pixel/weight terms; the PCU divides by ``2**(frx+fry)``, rounds and
    reluxes.

    forward() takes plain float tensors: ``reference`` [B, C, H_in, W_in] in
    uint8 units and ``coords`` [B, 2, H_out, W_out] ABSOLUTE sample positions
    in pixels (channel 0 = x, 1 = y); returns the raw integer output (the
    warped value when ``qparams.out_cal == 0``); the float value is
    ``output * 2**-output_scale_log2``.
    """

    def __init__(
        self,
        channels: int,
        grid_bit_width: int = 16,
        acc_bit_width: int = 48,
        qparams: "LayerMXQParams | None" = None,
    ):
        super(CalibratedMXWarp, self).__init__()
        if qparams is None:
            raise ValueError("qparams (LayerMXQParams) is required")
        self.qparams = qparams
        self.grid_bit_width = grid_bit_width

        self.mx_warp = MXWarp_Loops(channels=channels, acc_bit_width=acc_bit_width)

        # PCU configuration follows the supplied grid format.
        self._frac_key = None
        self.is_calibrate = True

    @property
    def bias(self):
        return self.mx_warp.bias

    @property
    def output_scale_log2(self) -> int:
        return self.qparams.output_scale_log2

    def calibrate(self, frac_bits=(7, 7)):
        """(Re)build the warp PCU c_structs for these grid fraction bits. Idempotent.

        Fraction bits are supplied by the preceding layer.
        """
        self.mx_warp.load_c_struct(
            build_warp_c_struct_arr(
                self.qparams,
                self.mx_warp.bias.data,
                self.mx_warp.channels,
                frac_bits=frac_bits,
            )
        )
        self._frac_key = tuple(int(v) for v in frac_bits_pair(frac_bits))

    def forward(self, reference: torch.Tensor, coords: torch.Tensor, frac_bits=(8, 8)) -> torch.Tensor:
        if reference.shape[0] != 1 or coords.shape[0] != 1 or reference.shape[-2:] != coords.shape[-2:]:
            raise ValueError("warp requires one frame and a same-size grid")
        mx_ref = pack_warp_reference(reference, mx_type=self.qparams.mx_type)
        mx_grid, fr = pack_warp_grid(coords, grid_bit_width=self.grid_bit_width, frac_bits=frac_bits)
        self.calibrate(fr)
        return self.mx_warp(mx_ref, mx_grid, fr)
