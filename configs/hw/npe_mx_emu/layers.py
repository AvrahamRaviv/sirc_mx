"""Torch-tensor-in / raw-int-out conv layer wrapping the NPE HW emulation pipe."""

from __future__ import annotations

import torch
import torch.nn as nn

from npe_mx_emu.calib import LayerMXQParams, build_c_struct_arr
from npe_mx_emu.hw_pipe import MXConv2D_Loops
from npe_mx_emu.mx_format import base_mx_quantizer, input_mx_quantizer


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
            min_val=min_val,
            max_val=max_val,
        )

    @classmethod
    def from_torch_conv(cls, conv: nn.Conv2d, qparams: LayerMXQParams) -> "CalibratedMXConv2d":
        kernel_size = conv.kernel_size[0] if conv.kernel_size[0] == conv.kernel_size[1] else conv.kernel_size
        stride = conv.stride[0] if conv.stride[0] == conv.stride[1] else conv.stride
        padding = conv.padding[0] if conv.padding[0] == conv.padding[1] else conv.padding
        layer = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
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
        x_int = (x * 2.0 ** qp.x_cal).round()
        if qp.is_input_layer:
            mx = input_mx_quantizer(x_int, block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.x_cal)
        else:
            mx = base_mx_quantizer(x_int, block_size=qp.block_size, mx_type=qp.mx_type, calibrate_offset=qp.x_cal)
        out_raw = self.mx_conv(mx)
        return out_raw
