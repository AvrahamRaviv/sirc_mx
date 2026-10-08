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
  * `fxp` - static fixed point (`fixed_point/fxp_quant.py`), a single scale for
    the whole tensor and no shared exponent. This is what the accelerator uses
    for operands it carries as plain intN words: the DOF warp grid (16-bit
    signed, Q8.8, subpixel step 1/256) and the warp reference and result
    (int8 with a scale frozen by static quantization, not a power of two).
    Needs no `axes` - there are no blocks.

The two are different lattices, not two widths of the same thing, so matching
hardware means matching the family as well as the bit count.

Config (mx_config.json), one entry per wrapped module:

    {"name": "warp", "kind": "act_quant", "call_sites": 3,
     "inputs": [
       {"fxp": {"total_bits": 8, "signed": true, "calibrate": true}},
       {"fxp": {"total_bits": 16, "frac_bits": 8, "signed": true,
                "round": "half_away", "saturate": true}}
     ],
     "output": {"fxp": {"total_bits": 8, "signed": true, "calibrate": true}}}

An input entry of `null` (or a spec with `a_elem_format: null`) leaves that
operand untouched. Inputs beyond the listed ones pass through unquantized.
`output` is optional; without it the wrapped module's result is returned as-is.

## Per-call-site scales

MX needs no help here: its shared exponent is derived per block, so one spec
covers every call of a shared module. A *static* scale cannot - it is frozen up
front, and the tensors differ per call. The DOF warp is one module instance
called once per pyramid level, and the feature magnitude grows coarse-to-fine,
so a single scale is badly wrong at some levels (measured: up to 40 dB worse
than a per-level scale).

`call_sites: N` says the wrapper is called N times per forward, in a fixed
order. Call `i % N` then uses its own scale. An `fxp` operand marked
`calibrate: true` gets an independent scale per call site; one with an explicit
`scale` / `frac_bits` shares that single spec across all of them, since it was
given, not learned. A per-site list in the config works too, for hand-set
scales that differ per level.

`calibrate: true` operands have no scale until `calibrate_act_scales` (in
`fixed_point/mx_fixed_point_hw.py`) has run; quantizing before that raises.
"""

import copy

import torch
import torch.nn as nn

from microxcaling.mx.elemwise_ops import quantize_elemwise_op
from mx_layers_blocked import quantize_mx_op  # STE-wrapped
from fixed_point.fxp_quant import (  # STE-wrapped fake_quant_fxp
    fake_quant_fxp,
    fxp_scale_for_max_abs,
)

class MXActQuant(nn.Module):
    """MX-quantize the tensor inputs of a parameter-free module.

    Args:
        inner: the wrapped module (kept as a submodule, so its own
            parameters/buffers, if any, still move with `.to()` / `state_dict`).
        specs_per_input: list of MxSpecs (or None) aligned with the wrapped
            module's positional args.
        axes_per_input: list of quant axes lists, same alignment. Defaults to
            `[1]` (channel axis) for every input.
        fxp_per_input: list aligned the same way; each entry is None, one
            normalized fxp config, or a list of them (one per call site).
        out_spec / out_axes / out_fxp: the same three for the result.
        call_sites: how many times this instance is called per forward pass.
            Only matters for static (`fxp`) operands - see the module docstring.

    Only positional args are quantized; keyword args pass through untouched.
    """

    def __init__(self, inner, specs_per_input, axes_per_input=None,
                 fxp_per_input=None, out_spec=None, out_axes=None,
                 out_fxp=None, call_sites=1):
        super().__init__()
        self.inner = inner
        self.specs_per_input = list(specs_per_input)
        if axes_per_input is None:
            axes_per_input = [[1]] * len(self.specs_per_input)
        self.axes_per_input = [list(a) for a in axes_per_input]
        assert len(self.axes_per_input) == len(self.specs_per_input), \
            "axes_per_input and specs_per_input must have the same length"
        if call_sites < 1:
            raise ValueError(f"call_sites must be >= 1, got {call_sites}")
        self.call_sites = int(call_sites)

        if fxp_per_input is None:
            fxp_per_input = [None] * len(self.specs_per_input)
        assert len(fxp_per_input) == len(self.specs_per_input), \
            "fxp_per_input and specs_per_input must have the same length"
        self.fxp_per_input = [
            self._expand_sites(f, f"input {i}")
            for i, f in enumerate(fxp_per_input)
        ]

        self.out_spec = out_spec
        self.out_axes = list(out_axes) if out_axes is not None else [1]
        self.out_fxp = self._expand_sites(out_fxp, "output")
        self.n_calls = 0
        self.calibrating = False
        self._obs = {}

    # -- construction ------------------------------------------------------

    def _expand_sites(self, fxp, where):
        """Normalize one operand's fxp config into a per-call-site list.

        A list is taken as given. A single config is shared across call sites
        when its scale is already known, and deep-copied per site when the
        scale is to be calibrated - a learned scale is a property of the tensor
        a given call sees, not of the module.
        """
        if fxp is None:
            return None
        if isinstance(fxp, (list, tuple)):
            if len(fxp) != self.call_sites:
                raise ValueError(
                    f"{where}: {len(fxp)} per-call-site fxp configs for "
                    f"call_sites={self.call_sites}")
            return [copy.deepcopy(c) for c in fxp]
        if fxp.get("calibrate") and self.call_sites > 1:
            return [copy.deepcopy(fxp) for _ in range(self.call_sites)]
        return [fxp] * self.call_sites

    @property
    def mx_specs(self):
        """First non-None spec — lets generic MX tooling introspect this layer."""
        for sp in self.specs_per_input:
            if sp is not None:
                return sp
        return None

    @property
    def site(self):
        """Which call site the next forward is, in `range(call_sites)`."""
        return self.n_calls % self.call_sites

    @property
    def last_site(self):
        """The call site of the most recent forward (for post-forward hooks)."""
        return (self.n_calls - 1) % self.call_sites if self.n_calls else 0

    def fxp_cfg(self, idx, site=None):
        """Input `idx`'s fxp config at one call site, or None if it has none."""
        fxp = self.fxp_per_input[idx]
        if fxp is None:
            return None
        return fxp[self.last_site if site is None else site]

    def out_fxp_cfg(self, site=None):
        """The output's fxp config at one call site, or None if it has none."""
        if self.out_fxp is None:
            return None
        return self.out_fxp[self.last_site if site is None else site]

    def reset_calls(self):
        """Put the call-site counter back to 0 (start of a forward pass)."""
        self.n_calls = 0

    # -- quantization ------------------------------------------------------

    def _quant(self, x, spec, axes, fxp, obs_key):
        """Apply one operand's quantizer. Non-float / unconfigured pass through.

        `fxp` takes priority over `spec`: an operand the hardware carries as a
        plain intN word has no shared exponent, so the MX path would be the
        wrong lattice even at a matching bit width.
        """
        if not torch.is_tensor(x) or not x.is_floating_point():
            return x
        if fxp is not None and fxp.get('enabled', True):
            if fxp.get('calibrate') and fxp.get('scale') is None:
                if self.calibrating:
                    self._observe(obs_key, x)
                    return x
                raise RuntimeError(
                    f"{obs_key}: fxp operand is marked calibrate=true but has "
                    f"no scale yet. Run calibrate_act_scales(model, loader) "
                    f"before inference.")
            return fake_quant_fxp(
                x,
                frac_bits=fxp['frac_bits'], total_bits=fxp['total_bits'],
                signed=fxp['signed'], round_mode=fxp['round'],
                saturate=fxp['saturate'], clip_grad=fxp['clip_grad'],
                scale=fxp.get('scale'),
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

    def quant_input(self, x, idx, site=None):
        """Quantize positional input `idx`; pass through if not configured."""
        if idx >= len(self.specs_per_input):
            return x
        site = self.site if site is None else site
        fxp = self.fxp_per_input[idx]
        return self._quant(x, self.specs_per_input[idx],
                           self.axes_per_input[idx],
                           None if fxp is None else fxp[site],
                           ("in", idx, site))

    def quant_output(self, y, site=None):
        """Quantize the wrapped module's result; pass through if not configured.

        Hardware emits the warped result as an intN word, so whatever consumes
        this module downstream must see the rounded value, not the FP32 one.
        A tuple/list return has every float tensor in it quantized.
        """
        if self.out_spec is None and self.out_fxp is None:
            return y
        site = self.site if site is None else site
        fxp = None if self.out_fxp is None else self.out_fxp[site]
        key = ("out", site)
        if isinstance(y, (tuple, list)):
            q = [self._quant(t, self.out_spec, self.out_axes, fxp, key)
                 for t in y]
            return type(y)(q)
        return self._quant(y, self.out_spec, self.out_axes, fxp, key)

    def forward(self, *args, **kwargs):
        site = self.site
        qargs = [self.quant_input(a, i, site) for i, a in enumerate(args)]
        self.n_calls += 1
        return self.quant_output(self.inner(*qargs, **kwargs), site)

    # -- calibration -------------------------------------------------------

    def _observe(self, key, x):
        m = float(x.detach().abs().max())
        self._obs[key] = max(self._obs.get(key, 0.0), m)

    def start_calibration(self):
        """Observe magnitudes instead of quantizing `calibrate` fxp operands."""
        self.calibrating = True
        self._obs = {}
        self.reset_calls()

    def finish_calibration(self):
        """Freeze a static scale per observed operand and per call site.

        The scale covers the largest magnitude seen, so nothing clips on the
        calibration set: step = max|x| / (2^(bits-1) - 1) for a signed word.
        An operand with `pow2: true` rounds that step up to a power of two,
        which is what a shift-only datapath can represent; the default keeps
        the arbitrary real step, which is what a float-multiplier datapath does
        and is 2-5 dB better because it uses the full code range.

        Returns:
            dict: the frozen scales, keyed `"in<i>@<site>"` / `"out@<site>"`.
        """
        self.calibrating = False
        frozen = {}
        for key, max_abs in sorted(self._obs.items(), key=lambda kv: str(kv[0])):
            cfg = self._cfg_for(key)
            if cfg is None or cfg.get("scale") is not None:
                continue
            cfg["scale"] = fxp_scale_for_max_abs(
                max_abs, total_bits=cfg["total_bits"], signed=cfg["signed"],
                pow2=cfg.get("pow2", False))
            frozen[self._key_str(key)] = cfg["scale"]
        self.reset_calls()
        return frozen

    def _cfg_for(self, key):
        """The fxp config dict an observation key refers to."""
        if key[0] == "in":
            fxp = self.fxp_per_input[key[1]]
            return None if fxp is None else fxp[key[2]]
        return None if self.out_fxp is None else self.out_fxp[key[1]]

    @staticmethod
    def _key_str(key):
        return f"in{key[1]}@{key[2]}" if key[0] == "in" else f"out@{key[1]}"

    def export_scales(self):
        """Frozen static scales, as a plain dict (scales are config, not state).

        `state_dict` does not carry them - they live in the fxp config dicts -
        so a calibrated model must either be recalibrated or have these written
        back with `load_scales`.
        """
        out = {}
        for i, fxp in enumerate(self.fxp_per_input):
            if fxp is None:
                continue
            for site, cfg in enumerate(fxp):
                if cfg.get("scale") is not None:
                    out[f"in{i}@{site}"] = cfg["scale"]
        if self.out_fxp is not None:
            for site, cfg in enumerate(self.out_fxp):
                if cfg.get("scale") is not None:
                    out[f"out@{site}"] = cfg["scale"]
        return out

    def load_scales(self, scales):
        """Write scales produced by `export_scales` back into the fxp configs."""
        for name, val in scales.items():
            where, _, site_s = name.partition("@")
            site = int(site_s)
            if where == "out":
                key = ("out", site)
            else:
                key = ("in", int(where[2:]), site)
            cfg = self._cfg_for(key)
            if cfg is None:
                raise KeyError(f"no fxp operand for scale key {name!r}")
            cfg["scale"] = float(val)

    # -- repr --------------------------------------------------------------

    @staticmethod
    def _fmt_one(spec, axes, fxp):
        """One operand at one call site, as it appears in the module repr."""
        if fxp is not None and fxp.get('enabled', True):
            sign = 's' if fxp['signed'] else 'u'
            if fxp.get('scale') is not None:
                return f"fxp{fxp['total_bits']}{sign}/step{fxp['scale']:.4g}"
            if fxp.get('calibrate'):
                return f"fxp{fxp['total_bits']}{sign}/uncal"
            return f"fxp{fxp['total_bits']}.{fxp['frac_bits']}{sign}"
        if spec is None or spec['a_elem_format'] is None:
            return "off"
        return f"{spec['a_elem_format']}/bs{spec['block_size']}/axes{axes}"

    def _fmt(self, spec, axes, fxp):
        """One operand across call sites; collapsed when every site agrees."""
        if fxp is None:
            return self._fmt_one(spec, axes, None)
        per_site = [self._fmt_one(spec, axes, c) for c in fxp]
        if len(set(per_site)) == 1:
            return per_site[0]
        return "[" + "|".join(per_site) + "]"

    def extra_repr(self):
        parts = [
            f"in{i}=" + self._fmt(sp, self.axes_per_input[i],
                                  self.fxp_per_input[i])
            for i, sp in enumerate(self.specs_per_input)
        ]
        if self.out_spec is not None or self.out_fxp is not None:
            parts.append("out=" + self._fmt(self.out_spec, self.out_axes,
                                            self.out_fxp))
        if self.call_sites > 1:
            parts.append(f"call_sites={self.call_sites}")
        return ", ".join(parts)
