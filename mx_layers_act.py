"""Activation-only MX quantization for parameter-free modules.

Some ops in a network carry no learned weights but still consume MX-quantized
operands in hardware — e.g. a `warp` layer that takes (features, flow) and
resamples. `MXQuantizer._replace_layers` only rewrites Conv2d / ConvTranspose2d
/ Linear, so such ops stay FP32 unless wrapped.

`MXActQuant` wraps the original module and quantizes each positional tensor
input before delegating, then optionally quantizes the result. Each input gets
its own format and its own quant axis, because operands of the same op often
need different ones: a feature map blocks along channels (`axes=[1]`), while a
2-channel flow field is better blocked along width (`axes=[-1]`) since C=2
would leave a mostly-padded block.

An operand can use either quantizer:

  * `mx_specs` / `group` - MX block floating point, one shared exponent per
    block of `block_size` along `axes`.
  * `fxp` - static fixed point (`fixed_point/fxp_quant.py`), a single scale of
    2^-frac_bits for the whole tensor and no shared exponent. This is what the
    accelerator uses for operands it carries as plain Qm.n words, e.g. the DOF
    warp grid: 16-bit signed with 6 fractional bits, so the subpixel step is
    1/64. Needs no `axes` - there are no blocks.

The two are different lattices, not two widths of the same thing, so matching
hardware means matching the family as well as the bit count.

Config (mx_config.json), one entry per wrapped module:

    {"name": "warp", "kind": "act_quant",
     "inputs": [
       {"mx_specs": {"a_elem_format": "int8", "block_size": 32}, "axes": [-1]},
       {"fxp": {"total_bits": 16, "frac_bits": 6, "signed": true,
                "round": "half_away", "saturate": true}}
     ],
     "output": {"mx_specs": {"a_elem_format": "int8", "block_size": 32},
                "axes": [-1]}}

An input entry of `null` (or a spec with `a_elem_format: null`) leaves that
operand untouched. Inputs beyond the listed ones pass through unquantized.
`output` is optional; without it the wrapped module's result is returned as-is.

A single wrapper instance called several times per forward quantizes every
call with the same specs; per-call-site specs require distinct module
instances.
"""

import torch
import torch.nn as nn

from microxcaling.mx.elemwise_ops import quantize_elemwise_op
from mx_layers_blocked import quantize_mx_op  # STE-wrapped
from fixed_point.fxp_quant import fake_quant_fxp  # STE-wrapped


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

    def __init__(self, inner, specs_per_input, axes_per_input=None,
                 fxp_per_input=None, out_spec=None, out_axes=None,
                 out_fxp=None):
        super().__init__()
        self.inner = inner
        self.specs_per_input = list(specs_per_input)
        if axes_per_input is None:
            axes_per_input = [[1]] * len(self.specs_per_input)
        self.axes_per_input = [list(a) for a in axes_per_input]
        assert len(self.axes_per_input) == len(self.specs_per_input), \
            "axes_per_input and specs_per_input must have the same length"
        # Static fixed-point spec per input, aligned with specs_per_input. An
        # entry here wins over the MX spec at the same index: the two are
        # alternative quantizers, not a pipeline.
        if fxp_per_input is None:
            fxp_per_input = [None] * len(self.specs_per_input)
        self.fxp_per_input = list(fxp_per_input)
        assert len(self.fxp_per_input) == len(self.specs_per_input), \
            "fxp_per_input and specs_per_input must have the same length"
        self.out_spec = out_spec
        self.out_axes = list(out_axes) if out_axes is not None else [1]
        self.out_fxp = out_fxp
        self.n_calls = 0

    @property
    def mx_specs(self):
        """First non-None spec — lets generic MX tooling introspect this layer."""
        for sp in self.specs_per_input:
            if sp is not None:
                return sp
        return None

    @staticmethod
    def _quant(x, spec, axes, fxp):
        """Apply one operand's quantizer. Non-float / unconfigured pass through.

        `fxp` takes priority over `spec`: an operand the hardware carries as a
        plain Qm.n word has no shared exponent, so the MX path would be the
        wrong lattice even at a matching bit width.
        """
        if not torch.is_tensor(x) or not x.is_floating_point():
            return x
        if fxp is not None and fxp.get('enabled', True):
            return fake_quant_fxp(
                x,
                frac_bits=fxp['frac_bits'], total_bits=fxp['total_bits'],
                signed=fxp['signed'], round_mode=fxp['round'],
                saturate=fxp['saturate'], clip_grad=fxp['clip_grad'],
            )
        if spec is None or spec['a_elem_format'] is None:
            return x
        bf = quantize_elemwise_op(x, mx_specs=spec, round=spec['round_output'])
        return quantize_mx_op(
            bf, spec,
            elem_format=spec['a_elem_format'],
            axes=axes,
            round=spec['round_mx_output'],
        )

    def quant_input(self, x, idx):
        """Quantize positional input `idx`; pass through if not configured."""
        if idx >= len(self.specs_per_input):
            return x
        return self._quant(x, self.specs_per_input[idx],
                           self.axes_per_input[idx], self.fxp_per_input[idx])

    def quant_output(self, y):
        """Quantize the wrapped module's result; pass through if not configured.

        Hardware emits the warped result as an intN word, so whatever consumes
        this module downstream must see the rounded value, not the FP32 one.
        A tuple/list return has every float tensor in it quantized.
        """
        if self.out_spec is None and self.out_fxp is None:
            return y
        if isinstance(y, (tuple, list)):
            q = [self._quant(t, self.out_spec, self.out_axes, self.out_fxp)
                 for t in y]
            return type(y)(q)
        return self._quant(y, self.out_spec, self.out_axes, self.out_fxp)

    def forward(self, *args, **kwargs):
        qargs = [self.quant_input(a, i) for i, a in enumerate(args)]
        self.n_calls += 1
        return self.quant_output(self.inner(*qargs, **kwargs))

    @staticmethod
    def _fmt(spec, axes, fxp):
        """One operand's format, as it appears in the module repr."""
        if fxp is not None and fxp.get('enabled', True):
            return (f"fxp{fxp['total_bits']}.{fxp['frac_bits']}"
                    f"{'s' if fxp['signed'] else 'u'}")
        if spec is None or spec['a_elem_format'] is None:
            return "off"
        return f"{spec['a_elem_format']}/bs{spec['block_size']}/axes{axes}"

    def extra_repr(self):
        parts = [
            f"in{i}=" + self._fmt(sp, self.axes_per_input[i],
                                  self.fxp_per_input[i])
            for i, sp in enumerate(self.specs_per_input)
        ]
        if self.out_spec is not None or self.out_fxp is not None:
            parts.append("out=" + self._fmt(self.out_spec, self.out_axes,
                                            self.out_fxp))
        return ", ".join(parts)
