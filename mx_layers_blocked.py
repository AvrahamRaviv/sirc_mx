"""Blocked MX layer variants that expose per-block FP32 partials to the
cross-block accumulator hook.

Only instantiated when `xblock_accum_mode == 'fixed_point'`. With the default
`fp32` mode, `mx_quantizer._replace_layers` keeps the original MXLinear /
MXConv2d, so these classes have zero effect on existing runs.

Scope (Phase A): `MXLinearBlocked` only. Conv2d is Phase B.

Matches `microxcaling.mx.linear.LinearFunction.forward` quantization sequence
(elemwise + MX-quant of operands, elemwise on output and post-bias). Replaces
the flat `F.linear(qis_input, qis_weight)` with a blocked einsum that produces
per-block partials `[..., O, nb]`, then reduces via
`cross_block_accumulate_from_specs(partials, self)`.

Backward: handled by PyTorch autograd through the einsum + reshape + our STE
in the fixed-point accumulator. microxcaling's custom `LinearFunction.backward`
(which re-quantizes grads through MX ops) is NOT reproduced in this first slice.
Documented as a known limitation — Phase C can wrap this in a custom
`torch.autograd.Function`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from microxcaling.mx.linear import Linear as MXLinear
from microxcaling.mx.convolution import Conv2d as MXConv2d
from microxcaling.mx.mx_ops import quantize_mx_op as _quantize_mx_op_raw
from microxcaling.mx.elemwise_ops import quantize_elemwise_op

from fixed_point.mx_fixed_point import cross_block_accumulate_from_specs, _get_xblock_cfg
from fixed_point.mx_fixed_point_hw import (hw_fxp_conv2d, resolve_out_window_bits,
                                           clamp_block_exp, floor_log2, in_exp_on,
                                           dilate_input, PLAParams, pla_bins, pla_multipliers,
                                           block_exp_range, check_block_exp,
                                           check_static_shift, int_input)

# Activations PLA can fold into the PCU multiply: two segments split at 0.
PLA_ACTS = (nn.PReLU, nn.LeakyReLU, nn.ReLU)


def quantize_mx_op(x, *args, **kwargs):
    """STE wrapper around microxcaling's quantize_mx_op.

    The underlying op does not propagate gradient (Microsoft's MXLinear /
    MXConv2d compensate by wrapping the whole forward in a custom
    autograd.Function with an explicit backward). Our blocked / HW layers
    call quantize_mx_op directly, so without an STE shim the gradient
    chain dies at this op and weight/input grads come out as 0, which
    silently breaks training.

    Forward = the quantized value. Backward = identity (STE).
    """
    with torch.no_grad():
        x_q = _quantize_mx_op_raw(x, *args, **kwargs)
    if not x.requires_grad:
        return x_q
    return x + (x_q - x).detach()


class MXLinearBlocked(MXLinear):
    """MXLinear variant that exposes per-block partials to the cross-block accumulator.

    Requires `in_features % block_size == 0`. Cross-block accumulator config is
    read from attributes on `self` (propagated by `mx_quantizer._replace_layers`):
      - `xblock_accum_mode`       (must be 'fixed_point' to take this path)
      - `xblock_accum_bits`       int in [40, 48]
      - `xblock_accum_saturate`   bool
      - `xblock_accum_ste_mask`   bool
      - `xblock_accum_backend`    'python' | 'triton'
    """

    def forward(self, x):
        if self.mx_none:
            return super().forward(x)
        if self.prequantized_weights:
            assert not self.training, \
                "Cannot use prequantized weights when training!"

        sp = self.mx_specs
        bs = sp['block_size']
        in_feat = x.shape[-1]
        assert in_feat % bs == 0, (
            f"MXLinearBlocked requires in_features ({in_feat}) divisible by "
            f"block_size ({bs})"
        )

        bf_in = quantize_elemwise_op(x, mx_specs=sp, round=sp['round_output'])
        bf_w = quantize_elemwise_op(self.weight, mx_specs=sp, round=sp['round_weight'])
        if self.bias is not None:
            bf_bias = quantize_elemwise_op(self.bias, mx_specs=sp, round=sp['round_weight'])

        qi = quantize_mx_op(
            bf_in, sp,
            elem_format=sp['a_elem_format'],
            axes=[-1],
            round=sp['round_mx_output'],
        )
        qw = quantize_mx_op(
            bf_w, sp,
            elem_format=sp['w_elem_format'],
            axes=[-1],
            round=sp['round_mx_output'],
        )

        if qw.dtype == torch.bfloat16 and qi.dtype != torch.bfloat16:
            qw = qw.to(qi.dtype)

        nb = in_feat // bs
        qi_b = qi.view(*qi.shape[:-1], nb, bs)       # [..., nb, bs]
        qw_b = qw.view(qw.shape[0], nb, bs)          # [O, nb, bs]

        # Per-block partials: same FLOPs as F.linear, just reshaped reduction.
        partials = torch.einsum('...nk,onk->...no', qi_b, qw_b)  # [..., nb, O]
        partials = partials.transpose(-1, -2).contiguous()        # [..., O, nb]

        out = cross_block_accumulate_from_specs(partials, self)   # [..., O]
        out = quantize_elemwise_op(out, mx_specs=sp, round=sp['round_output'])

        if self.bias is not None:
            out = out + bf_bias
            out = quantize_elemwise_op(out, mx_specs=sp, round=sp['round_output'])

        return out


class MXConv2dBlocked(MXConv2d):
    """MXConv2d variant that exposes per-block partials to the cross-block accumulator.

    Requires `in_channels % block_size == 0` and `groups == 1`. Block axis is
    input channels (matches microxcaling's `axes=[1]` scale-sharing).

    Path: quantize operands (elemwise + MX), `F.unfold` the activations to
    `[B, C, kK, L]`, split the C axis into `[nb, bs]` on both operand and
    weight, einsum produces `[B, O, nb, L]` partials, reduce via the
    cross-block accumulator hook, reshape back to `[B, O, H_out, W_out]`.

    Cross-block accumulator config read from `self.xblock_accum_*` attrs
    (propagated by `mx_quantizer._replace_layers`).
    """

    def forward(self, x):
        if self.mx_none:
            return super()._conv_forward(x, self.weight, self.bias)

        assert self.groups == 1, "MXConv2dBlocked: groups != 1 not supported"

        sp = self.mx_specs
        bs = sp['block_size']
        B, C, H, W = x.shape
        assert C % bs == 0, (
            f"MXConv2dBlocked requires in_channels ({C}) divisible by "
            f"block_size ({bs})"
        )

        bf_in = quantize_elemwise_op(x, mx_specs=sp, round=sp['round_output'])
        bf_w = quantize_elemwise_op(self.weight, mx_specs=sp, round=sp['round_weight'])
        if self.bias is not None:
            bf_bias = quantize_elemwise_op(self.bias, mx_specs=sp, round=sp['round_weight'])

        qi = quantize_mx_op(bf_in, sp, elem_format=sp['a_elem_format'], axes=[1])
        qw = quantize_mx_op(bf_w, sp, elem_format=sp['w_elem_format'], axes=[1])

        if qw.dtype == torch.bfloat16 and qi.dtype != torch.bfloat16:
            qw = qw.to(qi.dtype)

        O = qw.shape[0]
        kH, kW = qw.shape[2], qw.shape[3]

        # im2col: [B, C*kH*kW, L]
        x_unf = F.unfold(qi, (kH, kW),
                         dilation=self.dilation,
                         padding=self.padding,
                         stride=self.stride)
        L = x_unf.shape[-1]
        nb = C // bs
        kK = kH * kW

        # F.unfold orders the inner dim as [c * kH*kW + kh*kW + kw], so C is outer.
        x_b = x_unf.view(B, nb, bs, kK, L)    # [B, nb, bs, kK, L]
        w_b = qw.view(O, nb, bs, kK)          # [O, nb, bs, kK]

        # Per-block partials: sum over block-inner (bs) and kernel (kK) dims.
        partials = torch.einsum('bnkpl,onkp->bonl', x_b, w_b)   # [B, O, nb, L]
        partials = partials.permute(0, 1, 3, 2).contiguous()    # [B, O, L, nb]

        out_flat = cross_block_accumulate_from_specs(partials, self)  # [B, O, L]

        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        assert H_out * W_out == L

        out = out_flat.view(B, O, H_out, W_out)
        if self.bias is not None:
            out = out + bf_bias.view(1, -1, 1, 1)
        out = quantize_elemwise_op(out, mx_specs=sp, round=sp['round_output'])
        return out


def _pla_bypass(act, args, out):
    """Forward hook on an activation folded into the MXConv2dHW before it:
    the conv's PLA already applied it when its output is marked. A module
    function (not a closure) so it survives deepcopy."""
    x = args[0] if args else None
    return x if getattr(x, "_pla_folded", False) else None


def _reset_hw_call_counts(root, args):
    """Forward pre-hook on the quantized root: a new forward starts, every
    MXConv2dHW in it is uncalled. A module function (not a closure) so the
    hook survives deepcopy / DataParallel replication of the root."""
    for m in root.modules():
        if isinstance(m, MXConv2dHW):
            m._win_calls = 0


def install_hw_call_guard(model):
    """Make an eval-mode forward fail loudly when an MXConv2dHW with the
    output window on is called twice: one hw_grids entry holds one grid, and
    a second call site would silently reuse it. Idempotent. Train mode is not
    checked, so gradient checkpointing (which re-runs forwards) is safe."""
    if getattr(model, "_hw_call_guard", None) is None:
        model._hw_call_guard = model.register_forward_pre_hook(_reset_hw_call_counts)
    return model


class MXConv2dHW(MXConv2d):
    """HW-faithful fixed-point MXConv2d emulation.

    Replaces the FP32 block dot-product with a per-product int8 x int8 multiply,
    signed shift by `Ew + Ea - 2*MANTISSA_BIAS - e_layer_min`, and saturating
    accumulation in a narrow (default 35b) fixed-point accumulator. See
    `mx_fixed_point_hw.hw_fxp_conv2d` for the kernel semantics.

    Requirements:
      - `in_channels % block_size == 0` and `groups == 1` (dispatcher falls back).
      - `e_layer_min` must be populated before inference. Run
        `mx_fixed_point_hw.calibrate_e_layer_min(model, loader)` once, then
        freeze. Alternatively, set via `xblock_accum.e_layer_min` in config.
      - `a_elem_format` / `w_elem_format` must be 'int8' (the HW pipeline is
        integer-only).
      - `xblock_accum.out_window` ('off' | 'infer' | 'always') applies the
        static output window at `e_qa_out` ('infer' = eval mode only). Needs
        `e_qa_out`: run `mx_fixed_point_hw.calibrate_hw_grids(model, loader)`.
      - `xblock_accum.in_exp_bits` + `in_exp_mode` model the HW a/w
        shared-exponent register: block exponents limited to
        [e_base, e_base + 2^in_exp_bits - 1] per tensor (see
        `clamp_block_exp`). Input base `e_base_a` is calibrated; weight base
        `e_base_w` comes from the current weights unless the config sets it.
      - `xblock_accum.pla` ('off' | 'infer') folds the activation that follows
        in an nn.Sequential (`fold_pla`) into the PCU multiply before the
        window, in eval mode. The activation module stays in place (state_dict
        unchanged) and passes a PLA output through; train mode runs it as
        before. Multipliers: hw_grids pla_* if set, else from its slope.
      - HW-exact switches, all off by default (`xblock_accum`):
        `mant_round` 'truncate' (a/w mantissas toward zero), `overflow`
        'raise' (eval: accumulator / window / exponent register / int input
        out of range raise instead of clamping), `bias_round` 'half_up',
        `in_exp_anchor` 'min' (register base at the min non-zero block, eval
        only; train stays max-anchored), `static_shift_bits` (range check of
        the PCU static shift), `in_mode` 'int' (per-tensor int8 input, the
        first layer: q = round(x / in_scale), the scale folded into the
        weights, conv(q, W * s) == conv(q * s, W)).
    """

    def weight_exp_base(self):
        """Weight SE base: config value, else max block exp of the weights
        minus (2^in_exp_bits - 1). None when in_exp_bits is unset."""
        if self.e_base_w is not None:
            return int(self.e_base_w)
        if self._last_e_base_w is not None:
            return int(self._last_e_base_w)
        cfg = _get_xblock_cfg(self)
        bits = cfg.get("in_exp_bits")
        if bits is not None and cfg.get("in_exp_anchor", "max") == "min":
            w = self._hw_weight() * self._in_weight_scale()
            return block_exp_range(w, self.mx_specs['block_size'],
                                   cfg.get("weight_blockify", "channel"))[0]
        e = floor_log2(float(self.weight.detach().abs().max()))
        return None if bits is None or e is None else e - ((1 << int(bits)) - 1)

    def _in_weight_scale(self):
        """in_mode 'int' folds the input scale into the weights; else 1."""
        if (_get_xblock_cfg(self).get("in_mode", "mx") == "int"
                and self.in_scale is not None):
            return float(self.in_scale)
        return 1.0

    def _mant_round(self):
        """quantize_mx_op round= kwargs for the a/w mantissas: {} keeps the
        microxcaling default (nearest); 'truncate' is its 'floor' (toward zero)."""
        if _get_xblock_cfg(self).get("mant_round", "nearest") == "truncate":
            return {"round": "floor"}
        return {}

    def _hw_weight(self):
        """Weight in conv layout [O, Cin, kH, kW] as the HW path consumes it.
        A subclass whose parameter is stored in another layout (convT)
        returns its re-layout here; the weight quant path is the same."""
        return self.weight

    def _hw_geom(self):
        """Geometry of the conv the kernel runs: stride, padding, dilation,
        plus the input zero-insert (in_dil, in_extra) applied after the
        input is MX-quantized. Plain conv: no zero-insert."""
        return dict(stride=self.stride, padding=self.padding,
                    dilation=self.dilation, in_dil=(1, 1), in_extra=(0, 0))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e_layer_min = None
        self.e_qa_out = None            # output window grid (calibrate_hw_grids)
        self.e_base_a = None            # input SE base (in_exp_bits; calibrated)
        self.e_base_w = None            # weight SE base; None = from the weights
        self.in_scale = None            # in_mode 'int': input scale (calibrated)
        self._last_e_base_w = None
        self._in_exp_cut_life = [0, 0]  # a, w elements changed by the SE register
        # Where each grid came from: 'config' (hw_grids / xblock_accum) or
        # 'calib'. A config value is never replaced unless calibration is forced.
        self._grid_src = {"e_layer_min": None, "e_qa_out": None}
        self._win_calls = None          # set per root forward by install_hw_call_guard
        self.hw_calls = 1               # window calls allowed per forward (hw_grids "calls")
        self._mx_layer_name = None
        self._hw_logged = False
        self._pad_logged = False
        self._fwd_count = 0
        # Lifetime totals (used by report_hw_stats).
        self._sat_total_life = 0
        self._sat_seen_life = 0
        # Sampling-window counters (reset every verbose_sample_every forwards).
        self._sat_total_window = 0
        self._sat_seen_window = 0
        self._win_sat_seen_life = 0     # outputs clipped by the output window
        # Opt-in taps for the single-layer simulator: when set, the forward
        # keeps the MX-quantized operands it fed the accumulator (qi, qw: any
        # backend) and the int64 accumulator word + per-output saturation map
        # (python backend only) of the last call. With the output window on,
        # also the int64 window word O (any backend) + its saturation map.
        self._tap_acc = False
        self._last_qi = None
        self._last_qw = None
        self._last_acc = None
        self._last_sat = None
        self._last_win = None
        self._last_win_sat = None
        # PLA: the folded activation, kept in a tuple so it is not registered
        # as a submodule (it stays where it is in the model), and hw_grids
        # multipliers (frac, m_pos, m_neg) when the config gives them.
        self._pla_act = None
        self._pla_grid = None
        self._pla_bins_grid = None      # hw_grids pla_bin_* (bounds, scales, biases)
        self._pla_last = False          # the last forward applied PLA

    def fold_pla(self, act):
        """Fold `act` (applied to this layer's output) into the PLA stage."""
        if not isinstance(act, PLA_ACTS):
            raise TypeError(f"PLA cannot fold {type(act).__name__}")
        O = self.out_channels
        if isinstance(act, nn.PReLU) and act.num_parameters not in (1, O):
            raise ValueError(f"PLA: PReLU has {act.num_parameters} slopes, "
                             f"{self._mx_layer_name} has {O} channels")
        if getattr(act, "inplace", False):
            act.inplace = False         # the bypass hands its input on
        if not getattr(act, "_pla_hooked", False):
            act.register_forward_hook(_pla_bypass)
            act._pla_hooked = True
        self._pla_act = (act,)

    def pla_alpha(self):
        """Negative-side slope per output channel of the folded activation."""
        act, O = self._pla_act[0], self.out_channels
        if isinstance(act, nn.PReLU):
            a = act.weight.detach().to(torch.float64).cpu().flatten()
        elif isinstance(act, nn.LeakyReLU):
            a = torch.tensor([float(act.negative_slope)], dtype=torch.float64)
        else:
            a = torch.zeros(1, dtype=torch.float64)
        return a.expand(O).clone() if a.numel() == 1 else a

    def pla_on(self):
        """PLA configured and an activation folded (the mode decides when)."""
        return (self._pla_act is not None
                and _get_xblock_cfg(self).get("pla", "off") != "off")

    def _hw_bias(self):
        """The bias as the HW path consumes it (elementwise-quantized)."""
        if self.bias is None:
            return None
        sp = self.mx_specs
        return quantize_elemwise_op(self.bias, mx_specs=sp, round=sp['round_weight'])

    def pla_params(self, e_acc=None):
        """`PLAParams` of this layer: hw_grids multipliers, else quantized from
        the folded activation's slopes. With pla_bias 'bins' also the bins:
        hw_grids pla_bin_* if set, else `pla_bins` of the bias on the
        accumulator grid `e_acc` (default e_layer_min)."""
        cfg = _get_xblock_cfg(self)
        if self._pla_grid is not None:
            frac, m_pos, m_neg = self._pla_grid
            if len(m_pos) != self.out_channels:
                raise ValueError(
                    f"hw_grids[{self._mx_layer_name!r}]: {len(m_pos)} PLA "
                    f"multipliers for {self.out_channels} channels")
            m_pos = torch.tensor(m_pos, dtype=torch.int64)
            m_neg = torch.tensor(m_neg, dtype=torch.int64)
        else:
            m_pos, m_neg, frac = pla_multipliers(
                self.pla_alpha(), int(cfg.get("pla_mult_bits", 8)),
                cfg.get("pla_frac_bits"))
        mode = cfg.get("pla_bias", "bins")
        if mode != "bins":
            return PLAParams(m_pos, m_neg, int(frac), mode)
        if self._pla_bins_grid is not None:
            lists = self._pla_bins_grid
            if len({len(v) for v in lists[1]}) != 1:
                raise ValueError(f"hw_grids[{self._mx_layer_name!r}]: PLA bins "
                                 f"must have one bin count per layer")
            bounds, scales, biases = (torch.tensor(v, dtype=torch.int64).view(len(v), -1)
                                      for v in lists)
            if scales.shape[0] != self.out_channels:
                raise ValueError(
                    f"hw_grids[{self._mx_layer_name!r}]: {scales.shape[0]} PLA "
                    f"bins for {self.out_channels} channels")
        else:
            e = self.e_layer_min if e_acc is None else e_acc
            bounds, scales, biases = pla_bins(m_pos, m_neg, self._hw_bias(), int(e), int(frac))
        return PLAParams(m_pos, m_neg, int(frac), mode, bounds, scales, biases,
                         cfg.get("pla_bin_tie", "upper"))

    def _pla_float(self, z):
        """The folded activation in float (calibration / error references)."""
        a = self.pla_alpha().to(device=z.device, dtype=z.dtype).view(1, -1, 1, 1)
        return torch.where(z >= 0, z, z * a)

    def pla_ref(self, y):
        """`y` (an FP32 reference of this conv) through the folded activation
        when the last forward applied PLA, so the two compare like for like."""
        return self._pla_float(y) if self._pla_last else y

    def forward(self, x):
        if self.mx_none:
            return super()._conv_forward(x, self.weight, self.bias)

        assert self.groups == 1, "MXConv2dHW: groups != 1 not supported"

        sp = self.mx_specs
        bs = sp['block_size']
        B, C, H, W = x.shape
        a_fmt = sp['a_elem_format']
        w_fmt = sp['w_elem_format']
        if a_fmt != w_fmt:
            raise RuntimeError(
                f"MXConv2dHW requires a_elem_format == w_elem_format "
                f"(got a={a_fmt!r}, w={w_fmt!r})."
            )
        if not (isinstance(a_fmt, str) and a_fmt.startswith('int')):
            raise RuntimeError(
                f"MXConv2dHW models integer HW; set a_elem_format and "
                f"w_elem_format to an MXINT format (int8, int10, int12, "
                f"int16, ...). Got {a_fmt!r}."
            )

        cfg_early = _get_xblock_cfg(self)
        name = self._mx_layer_name or f"conv-{id(self):x}"
        hw_eval = not self.training
        raise_over = cfg_early.get('overflow', 'clamp') == 'raise' and hw_eval
        int_in = cfg_early.get('in_mode', 'mx') == 'int'
        if int_in:
            obs = getattr(self, '_in_scale_obs', None)
            if obs is not None:
                # in_scale pre-sweep: observe the raw input, run a float conv.
                obs[0] = max(obs[0], float(x.detach().abs().max()))
                g = self._hw_geom()
                return F.conv2d(dilate_input(x, g['in_dil'], g['in_extra']),
                                self._hw_weight(), self.bias, g['stride'],
                                g['padding'], g['dilation'])
            if self.in_scale is not None:
                x = int_input(x, self.in_scale, check=raise_over, what=f"MXConv2dHW {name}")
            elif getattr(self, '_calibration_state', None) is None:
                raise RuntimeError(
                    f"MXConv2dHW {name}: in_mode 'int' needs in_scale. Run "
                    f"`calibrate_hw_grids` or set hw_grids in_scale.")
        w_scale = self._in_weight_scale()
        pad_channels = bool(cfg_early.get('pad_channels', True))
        verbose = int(cfg_early.get('verbose', 1))
        # NPE blockify: weight per-filter flatten + activation X-block. Needs no
        # channel divisibility/padding (flatten handles any Cin; X-block is on W).
        act_blockify = cfg_early.get('act_blockify', 'channel')
        weight_blockify = cfg_early.get('weight_blockify', 'channel')
        npe = (act_blockify == "xblock" and weight_blockify == "flatten")
        if not npe and C % bs != 0:
            if not pad_channels:
                raise AssertionError(
                    f"MXConv2dHW requires in_channels ({C}) divisible by "
                    f"block_size ({bs}); set xblock_accum.pad_channels=True "
                    f"to allow zero-padding."
                )
            pad = bs - (C % bs)
            x = F.pad(x, (0, 0, 0, 0, 0, pad))           # pad C with zeros
            weight = F.pad(self._hw_weight(), (0, 0, 0, 0, 0, pad))
            if w_scale != 1.0:
                weight = weight * w_scale
            B, C, H, W = x.shape
            if verbose >= 2 and not self._pad_logged:
                print(
                    f"[MXConv2dHW {self._mx_layer_name or '?'}] zero-padding "
                    f"in_channels {C - pad} -> {C} to satisfy block_size={bs}"
                )
                self._pad_logged = True
        else:
            weight = self._hw_weight()
            if w_scale != 1.0:
                weight = weight * w_scale

        bf_in = quantize_elemwise_op(x, mx_specs=sp, round=sp['round_output'])
        bf_w = quantize_elemwise_op(weight, mx_specs=sp, round=sp['round_weight'])
        if self.bias is not None:
            bf_bias = quantize_elemwise_op(self.bias, mx_specs=sp, round=sp['round_weight'])
        else:
            bf_bias = None

        mr = self._mant_round()   # an int input is on the int8 lattice: exact either way
        if npe:
            # Fake-quant with the NPE blockings so extract_mxint_{xblock,flatten}
            # recover a consistent lattice (and STE grads flow). Activation blocks
            # along W (axis=3); weight flattens each filter [Cin,kH,kW]->1D.
            qi = quantize_mx_op(bf_in, sp, elem_format=sp['a_elem_format'],
                                axes=[3], block_size=bs, **mr)
            flat = bf_w.reshape(bf_w.shape[0], -1)
            flat = quantize_mx_op(flat, sp, elem_format=sp['w_elem_format'],
                                  axes=[1], block_size=bs, **mr)
            qw = flat.reshape(bf_w.shape)
        else:
            qi = quantize_mx_op(bf_in, sp, elem_format=sp['a_elem_format'], axes=[1], **mr)
            qw = quantize_mx_op(bf_w, sp, elem_format=sp['w_elem_format'], axes=[1], **mr)
        if qw.dtype == torch.bfloat16 and qi.dtype != torch.bfloat16:
            qw = qw.to(qi.dtype)
        if self._tap_acc:
            self._last_qi, self._last_qw = qi.detach(), qw.detach()

        # Calibration short-circuit: observe Ea/Ew, fall back to FP32 blocked
        # path so downstream layers see sensible activations.
        cal = getattr(self, '_calibration_state', None)
        if cal is not None:
            cal.update(qi, qw, bs, fmt=a_fmt,
                       act_blockify=act_blockify, weight_blockify=weight_blockify)
            out = self._fp32_blocked_forward(qi, qw, bf_bias, H, W)
            # With PLA the window sees the activated output.
            cal.update_out(self._pla_float(out) if self.pla_on() else out)
            return out

        cfg = _get_xblock_cfg(self)
        exp_mode = cfg.get('in_exp_mode', 'off')
        if in_exp_on(cfg) and (exp_mode == 'always'
                               or (exp_mode == 'infer' and not self.training)):
            if self.e_base_a is None:
                raise RuntimeError(
                    f"MXConv2dHW {self._mx_layer_name or '?'}: in_exp_mode="
                    f"{exp_mode!r} needs e_base_a. Run `calibrate_hw_grids` "
                    f"or set hw_grids e_base_a.")
            exp_bits = int(cfg['in_exp_bits'])
            span = (1 << exp_bits) - 1
            # 'min' (HW) anchors at the min non-zero block in eval; train and
            # 'max' keep the graceful max-anchored register.
            hw_anchor = cfg.get('in_exp_anchor', 'max') == 'min' and hw_eval
            w_blk = weight_blockify if npe else 'channel'
            a_blk = act_blockify if npe else 'channel'
            if self.e_base_w is None:
                if hw_anchor:
                    self._last_e_base_w = block_exp_range(qw, bs, w_blk)[0]
                else:
                    e_w = floor_log2(float(qw.detach().abs().max()))
                    self._last_e_base_w = (None if e_w is None else e_w - span)
            base_w = self.e_base_w if self.e_base_w is not None else self._last_e_base_w
            if hw_anchor:
                base_a = self.e_base_a
            elif cfg.get('in_exp_anchor', 'max') == 'min':
                e_a = floor_log2(float(qi.detach().abs().max()))   # train: live max anchor
                base_a = self.e_base_a if e_a is None else e_a - span
            else:
                base_a = self.e_base_a
            rnd = 'truncate' if mr else 'nearest'
            if hw_anchor and raise_over:
                if not int_in:
                    check_block_exp(qi, bs, a_blk, base_a, exp_bits,
                                    f"MXConv2dHW {name} input")
                if base_w is not None:
                    check_block_exp(qw, bs, w_blk, base_w, exp_bits,
                                    f"MXConv2dHW {name} weight")
            if int_in:
                na = 0                               # HW int input: no block exponent
            else:
                ca, na = clamp_block_exp(qi.detach(), base_a, exp_bits, a_fmt, rnd)
                qi = qi + (ca - qi).detach()         # STE: grads unchanged
            if base_w is not None:
                cw, nw = clamp_block_exp(qw.detach(), base_w, exp_bits, w_fmt, rnd)
                qw = qw + (cw - qw).detach()
            else:
                nw = 0
            self._in_exp_cut_life[0] += na
            self._in_exp_cut_life[1] += nw
            if self._tap_acc:
                self._last_qi, self._last_qw = qi.detach(), qw.detach()

        e_min = self.e_layer_min if self.e_layer_min is not None else cfg.get('e_layer_min')
        if e_min is None:
            raise RuntimeError(
                f"MXConv2dHW: e_layer_min is unset. Run "
                f"`calibrate_e_layer_min(model, loader)` or set "
                f"`xblock_accum.e_layer_min` in config."
            )

        name = self._mx_layer_name or f"conv-{id(self):x}"
        win_mode = cfg.get('out_window', 'off')
        apply_win = (win_mode == 'always'
                     or (win_mode == 'infer' and not self.training))
        if apply_win and self.e_qa_out is None:
            raise RuntimeError(
                f"MXConv2dHW {name}: out_window={win_mode!r} needs e_qa_out. "
                f"Run `calibrate_hw_grids(model, loader)` first."
            )
        e_qa_out = int(self.e_qa_out) if apply_win else None
        if apply_win and cfg.get('static_shift_bits') is not None:
            check_static_shift(e_qa_out, cfg['static_shift_bits'], a_fmt, f"MXConv2dHW {name}")
        apply_pla = (self.pla_on() and cfg.get('pla') == 'infer'
                     and not self.training)
        if apply_pla and not apply_win:
            raise RuntimeError(f"MXConv2dHW {name}: pla needs the output window on")
        self._pla_last = apply_pla
        if apply_win and not self.training and self._win_calls is not None:
            self._win_calls += 1
            if self._win_calls > self.hw_calls:
                raise RuntimeError(
                    f"MXConv2dHW {name}: called {self._win_calls} times in one "
                    f"forward with the output window on, but its grid is for "
                    f"{self.hw_calls}. hw_grids holds one grid per layer. If "
                    f"every call sees the same kind of input (e.g. a feature "
                    f"extractor run on both images), calibrate the grid over "
                    f"all calls (calibrate_hw_grids does) or set "
                    f"\"calls\": {self._win_calls} in its hw_grids entry."
                )
        if verbose >= 2 and not self._hw_logged:
            print(
                f"[MXConv2dHW {name}] HW path active | "
                f"in={C} out={qw.shape[0]} k={tuple(self.kernel_size)} "
                f"stride={tuple(self.stride)} pad={tuple(self.padding)} | "
                f"bits={cfg['bits']} sat_mode={cfg['sat_mode']} "
                f"e_layer_min={int(e_min)} backend={cfg['backend']} "
                f"out_window={win_mode} e_qa_out={self.e_qa_out}"
            )
            self._hw_logged = True

        stats = {"want_acc": True} if self._tap_acc else {}
        geom = self._hw_geom()
        out = hw_fxp_conv2d(
            qi, qw, bf_bias,
            e_layer_min=int(e_min),
            bs=bs,
            bits=cfg['bits'],
            sat_mode=cfg['sat_mode'],
            ste_mask=cfg['ste_mask'],
            stride=geom['stride'],
            padding=geom['padding'],
            dilation=geom['dilation'],
            in_dil=geom['in_dil'],
            in_extra=geom['in_extra'],
            backend=cfg['backend'],
            stats_sink=stats,
            fmt=a_fmt,
            act_blockify=act_blockify,
            weight_blockify=weight_blockify,
            npe_triton_variant=cfg.get('npe_triton_variant', 'reuse'),
            e_qa_out=e_qa_out,
            win_bits=resolve_out_window_bits(cfg, a_fmt) if apply_win else None,
            out_round=cfg.get('out_round', 'floor'),
            pla=self.pla_params(int(e_min)) if apply_pla else None,
            bias_round=cfg.get('bias_round', 'even'),
        )
        if self._tap_acc:
            self._last_acc = stats.get('acc')
            self._last_sat = stats.get('sat_map')
            self._last_win = stats.get('win')
            self._last_win_sat = stats.get('win_sat_map')
        self._win_sat_seen_life += int(stats.get('win_sat_count', 0))
        sat_count = int(stats.get('sat_count', 0))
        total = int(stats.get('total', 0))
        if raise_over and sat_count:
            raise RuntimeError(
                f"MXConv2dHW {name}: {sat_count}/{total} outputs overflowed the "
                f"{cfg['bits']}b accumulator (e_layer_min={int(e_min)}). HW wraps "
                f"(PTQ contract violation, xblock_accum.overflow 'raise').")
        if raise_over and stats.get('win_sat_count', 0):
            raise RuntimeError(
                f"MXConv2dHW {name}: {int(stats['win_sat_count'])}/{total} outputs "
                f"outside the {resolve_out_window_bits(cfg, a_fmt)}b output window "
                f"(e_qa_out={e_qa_out}): |value| >= 2^win_bits is impossible in HW "
                f"(PTQ contract violation, xblock_accum.overflow 'raise'). "
                f"Recalibrate or raise out_margin_bits.")
        self._fwd_count += 1
        self._sat_total_life += total
        self._sat_seen_life += sat_count
        self._sat_total_window += total
        self._sat_seen_window += sat_count

        if verbose >= 2 and sat_count > 0:
            pct = 100.0 * sat_count / max(total, 1)
            print(
                f"[MXConv2dHW {name}] saturated {sat_count}/{total} "
                f"outputs ({pct:.2f}%) this forward"
            )
        elif verbose >= 1:
            sample_every = int(cfg_early.get('verbose_sample_every', 100))
            if self._fwd_count % sample_every == 0 and self._sat_seen_window > 0:
                pct = 100.0 * self._sat_seen_window / max(self._sat_total_window, 1)
                print(
                    f"[MXConv2dHW {name}] sat sample over last {sample_every} "
                    f"forwards: {self._sat_seen_window}/{self._sat_total_window} "
                    f"outputs ({pct:.3f}%)"
                )
                self._sat_seen_window = 0
                self._sat_total_window = 0

        out = quantize_elemwise_op(out, mx_specs=sp, round=sp['round_output'])
        if apply_pla:
            out._pla_folded = True      # the folded activation passes it on
        return out

    def _fp32_blocked_forward(self, qi, qw, bf_bias, H, W):
        """FP32 reference conv used only while calibrating `e_layer_min`."""
        sp = self.mx_specs
        bs = sp['block_size']
        geom = self._hw_geom()
        qi = dilate_input(qi, geom['in_dil'], geom['in_extra'])   # exact in fp32
        H, W = qi.shape[-2:]
        B = qi.shape[0]
        C = qi.shape[1]
        O = qw.shape[0]
        kH, kW = qw.shape[2], qw.shape[3]
        kK = kH * kW

        # NPE (or any non-divisible Cin): the channel-blocked reshape below does
        # not apply; a plain conv is an equivalent FP32 reference for producing
        # downstream activations during calibration.
        if C % bs != 0:
            out = F.conv2d(qi, qw, bf_bias, geom['stride'], geom['padding'],
                           geom['dilation'], self.groups)
            return quantize_elemwise_op(out, mx_specs=sp, round=sp['round_output'])
        nb = C // bs

        x_unf = F.unfold(qi, (kH, kW),
                         dilation=geom['dilation'],
                         padding=geom['padding'],
                         stride=geom['stride'])
        L = x_unf.shape[-1]
        x_b = x_unf.view(B, nb, bs, kK, L)
        w_b = qw.view(O, nb, bs, kK)
        out = torch.einsum('bnkpl,onkp->bol', x_b, w_b)

        sH, sW = geom['stride']
        pH, pW = geom['padding']
        dH, dW = geom['dilation']
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        out = out.view(B, O, H_out, W_out)
        if bf_bias is not None:
            out = out + bf_bias.view(1, -1, 1, 1)
        out = quantize_elemwise_op(out, mx_specs=sp, round=sp['round_output'])
        return out
