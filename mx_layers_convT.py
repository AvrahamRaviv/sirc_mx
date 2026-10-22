"""HW path for ConvTranspose2d: a zero-insert conv on the MXConv2dHW path.

ConvTranspose2d(k, s, p, output_padding, dilation d) equals a stride-1 conv
over the input with s-1 zeros inserted between neighbours (plus
output_padding zeros at the bottom/right), padding d(k-1)-p, and the weight
transposed to [Cout, Cin, k, k] and flipped in both spatial dims.

`MXConvTranspose2dHW` is MXConv2dHW with two overrides and nothing else:
  * `_hw_weight`: that weight re-layout (the parameter stays in convT layout
    [Cin, Cout, k, k], so state_dict keys/shapes match nn.ConvTranspose2d and
    QAT gradients reach it);
  * `_hw_geom`: that conv geometry, with the zero-insert as in_dil/in_extra.

The input is MX-quantized on the real (undilated) tensor by the normal
MXConv2dHW act path, whatever the act blockify; the kernel inserts the zeros
after extracting the ints and exponents (`dilate_input`). Window, hw_grids,
calibration, stats and taps are the MXConv2dHW ones, keyed by the layer name.

Depthwise / grouped convT (groups > 1) is not supported here: the quantizer
keeps those on plain MXConvTranspose2d.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mx_layers_blocked import MXConv2dHW


class MXConvTranspose2dHW(MXConv2dHW):
    """HW-faithful fixed-point ConvTranspose2d (zero-insert lowering)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1,
                 mx_specs=None, name=None):
        if groups != 1:
            raise NotImplementedError(
                f"MXConvTranspose2dHW: groups={groups} is not supported; "
                f"grouped convT stays on plain MX.")
        super().__init__(in_channels, out_channels, kernel_size, stride=stride,
                         padding=padding, dilation=dilation, bias=bias,
                         mx_specs=mx_specs, name=name)
        self.output_padding = _pair(output_padding)
        for k, d, p in zip(self.kernel_size, self.dilation, self.padding):
            if d * (k - 1) < p:
                raise NotImplementedError(
                    f"MXConvTranspose2dHW: padding {self.padding} larger than "
                    f"dilation*(kernel-1) needs an output crop; not supported.")
        # convT layout, initialized as nn.ConvTranspose2d does.
        self.weight = nn.Parameter(
            self.weight.new_empty(in_channels, out_channels, *self.kernel_size))
        self.reset_parameters()

    def _hw_weight(self):
        return self.weight.transpose(0, 1).flip(-1, -2)

    def _hw_geom(self):
        return dict(
            stride=(1, 1),
            padding=tuple(d * (k - 1) - p for k, d, p in
                          zip(self.kernel_size, self.dilation, self.padding)),
            dilation=self.dilation,
            in_dil=self.stride,
            in_extra=self.output_padding,
        )

    def forward(self, x):
        if self.mx_none:
            return F.conv_transpose2d(x, self.weight, self.bias, self.stride,
                                      self.padding, self.output_padding,
                                      self.groups, self.dilation)
        return super().forward(x)
