"""Activation-only MX quantization for parameter-free modules.

Some ops in a network carry no learned weights but still consume MX-quantized
operands in hardware — e.g. a `warp` layer that takes (features, flow) and
resamples. `MXQuantizer._replace_layers` only rewrites Conv2d / ConvTranspose2d
/ Linear, so such ops stay FP32 unless wrapped.

`MXActQuant` wraps the original module and MX-quantizes each positional tensor
input before delegating. Each input gets its own `mx_specs` and its own quant
axis, because operands of the same op often need different block axes: a
feature map blocks along channels (`axes=[1]`), while a 2-channel flow field is
better blocked along width (`axes=[-1]`) since C=2 would leave a mostly-padded
block.

Config (mx_config.json), one entry per wrapped module:

    {"name": "warp", "kind": "act_quant",
     "inputs": [
       {"mx_specs": {"a_elem_format": "fp8_e4m3", "block_size": 32}, "axes": [1]},
       {"mx_specs": {"a_elem_format": "fp8_e4m3", "block_size": 16}, "axes": [-1]}
     ]}

An input entry of `null` (or a spec with `a_elem_format: null`) leaves that
operand untouched. Inputs beyond the listed ones pass through unquantized.

A single wrapper instance called several times per forward quantizes every
call with the same specs; per-call-site specs require distinct module
instances.
"""

import torch
import torch.nn as nn

from microxcaling.mx.elemwise_ops import quantize_elemwise_op
from mx_layers_blocked import quantize_mx_op  # STE-wrapped


class MXActQuant(nn.Module):
    """MX-quantize the tensor inputs of a parameter-free module.

    Args:
        inner: the wrapped module (kept as a submodule, so its own
            parameters/buffers, if any, still move with `.to()` / `state_dict`).
        specs_per_input: list of MxSpecs (or None) aligned with the wrapped
            module's positional args.
        axes_per_input: list of quant axes lists, same alignment. Defaults to
            `[1]` (channel axis) for every input.

    Only positional args are quantized; keyword args pass through untouched.
    """

    def __init__(self, inner, specs_per_input, axes_per_input=None):
        super().__init__()
        self.inner = inner
        self.specs_per_input = list(specs_per_input)
        if axes_per_input is None:
            axes_per_input = [[1]] * len(self.specs_per_input)
        self.axes_per_input = [list(a) for a in axes_per_input]
        assert len(self.axes_per_input) == len(self.specs_per_input), \
            "axes_per_input and specs_per_input must have the same length"
        self.n_calls = 0

    @property
    def mx_specs(self):
        """First non-None spec — lets generic MX tooling introspect this layer."""
        for sp in self.specs_per_input:
            if sp is not None:
                return sp
        return None

    def quant_input(self, x, idx):
        """Quantize positional input `idx`; pass through if not configured."""
        if idx >= len(self.specs_per_input):
            return x
        sp = self.specs_per_input[idx]
        if sp is None or sp['a_elem_format'] is None:
            return x
        if not torch.is_tensor(x) or not x.is_floating_point():
            return x
        bf = quantize_elemwise_op(x, mx_specs=sp, round=sp['round_output'])
        return quantize_mx_op(
            bf, sp,
            elem_format=sp['a_elem_format'],
            axes=self.axes_per_input[idx],
            round=sp['round_mx_output'],
        )

    def forward(self, *args, **kwargs):
        qargs = [self.quant_input(a, i) for i, a in enumerate(args)]
        self.n_calls += 1
        return self.inner(*qargs, **kwargs)

    def extra_repr(self):
        parts = []
        for i, sp in enumerate(self.specs_per_input):
            if sp is None or sp['a_elem_format'] is None:
                parts.append(f"in{i}=off")
            else:
                parts.append(f"in{i}={sp['a_elem_format']}/bs{sp['block_size']}"
                             f"/axes{self.axes_per_input[i]}")
        return ", ".join(parts)
