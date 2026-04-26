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
import torch.nn.functional as F

from microxcaling.mx.linear import Linear as MXLinear
from microxcaling.mx.convolution import Conv2d as MXConv2d
from microxcaling.mx.mx_ops import quantize_mx_op
from microxcaling.mx.elemwise_ops import quantize_elemwise_op

from mx_fixed_point import cross_block_accumulate_from_specs, _get_xblock_cfg
from mx_fixed_point_hw import hw_fxp_conv2d


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
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.e_layer_min = None
        self._mx_layer_name = None
        self._hw_logged = False
        self._sat_total = 0
        self._sat_seen = 0

    def forward(self, x):
        if self.mx_none:
            return super()._conv_forward(x, self.weight, self.bias)

        assert self.groups == 1, "MXConv2dHW: groups != 1 not supported"

        sp = self.mx_specs
        bs = sp['block_size']
        B, C, H, W = x.shape
        if sp['a_elem_format'] != 'int8' or sp['w_elem_format'] != 'int8':
            raise RuntimeError(
                "MXConv2dHW models integer HW; set a_elem_format and "
                "w_elem_format to 'int8'."
            )

        cfg_early = _get_xblock_cfg(self)
        pad_channels = bool(cfg_early.get('pad_channels', True))
        if C % bs != 0:
            if not pad_channels:
                raise AssertionError(
                    f"MXConv2dHW requires in_channels ({C}) divisible by "
                    f"block_size ({bs}); set xblock_accum.pad_channels=True "
                    f"to allow zero-padding."
                )
            pad = bs - (C % bs)
            x = F.pad(x, (0, 0, 0, 0, 0, pad))           # pad C with zeros
            weight = F.pad(self.weight, (0, 0, 0, 0, 0, pad))
            B, C, H, W = x.shape
            if not getattr(self, '_pad_logged', False):
                print(
                    f"[MXConv2dHW {self._mx_layer_name or '?'}] zero-padding "
                    f"in_channels {C - pad} -> {C} to satisfy block_size={bs}"
                )
                self._pad_logged = True
        else:
            weight = self.weight

        bf_in = quantize_elemwise_op(x, mx_specs=sp, round=sp['round_output'])
        bf_w = quantize_elemwise_op(weight, mx_specs=sp, round=sp['round_weight'])
        if self.bias is not None:
            bf_bias = quantize_elemwise_op(self.bias, mx_specs=sp, round=sp['round_weight'])
        else:
            bf_bias = None

        qi = quantize_mx_op(bf_in, sp, elem_format=sp['a_elem_format'], axes=[1])
        qw = quantize_mx_op(bf_w, sp, elem_format=sp['w_elem_format'], axes=[1])
        if qw.dtype == torch.bfloat16 and qi.dtype != torch.bfloat16:
            qw = qw.to(qi.dtype)

        # Calibration short-circuit: observe Ea/Ew, fall back to FP32 blocked
        # path so downstream layers see sensible activations.
        cal = getattr(self, '_calibration_state', None)
        if cal is not None:
            cal.update(qi, qw, bs)
            return self._fp32_blocked_forward(qi, qw, bf_bias, H, W)

        cfg = _get_xblock_cfg(self)
        e_min = self.e_layer_min if self.e_layer_min is not None else cfg.get('e_layer_min')
        if e_min is None:
            raise RuntimeError(
                f"MXConv2dHW: e_layer_min is unset. Run "
                f"`calibrate_e_layer_min(model, loader)` or set "
                f"`xblock_accum.e_layer_min` in config."
            )

        name = self._mx_layer_name or f"conv-{id(self):x}"
        if not self._hw_logged:
            print(
                f"[MXConv2dHW {name}] HW path active | "
                f"in={C} out={qw.shape[0]} k={tuple(self.kernel_size)} "
                f"stride={tuple(self.stride)} pad={tuple(self.padding)} | "
                f"bits={cfg['bits']} sat_mode={cfg['sat_mode']} "
                f"e_layer_min={int(e_min)} backend={cfg['backend']}"
            )
            self._hw_logged = True

        stats = {}
        out = hw_fxp_conv2d(
            qi, qw, bf_bias,
            e_layer_min=int(e_min),
            bs=bs,
            bits=cfg['bits'],
            sat_mode=cfg['sat_mode'],
            ste_mask=cfg['ste_mask'],
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            backend=cfg['backend'],
            stats_sink=stats,
        )
        sat_count = int(stats.get('sat_count', 0))
        total = int(stats.get('total', 0))
        self._sat_total += total
        self._sat_seen += sat_count
        if sat_count > 0:
            pct = 100.0 * sat_count / max(total, 1)
            print(
                f"[MXConv2dHW {name}] saturated {sat_count}/{total} "
                f"outputs ({pct:.2f}%) this forward"
            )

        out = quantize_elemwise_op(out, mx_specs=sp, round=sp['round_output'])
        return out

    def _fp32_blocked_forward(self, qi, qw, bf_bias, H, W):
        """FP32 reference conv used only while calibrating `e_layer_min`."""
        sp = self.mx_specs
        bs = sp['block_size']
        B = qi.shape[0]
        C = qi.shape[1]
        O = qw.shape[0]
        kH, kW = qw.shape[2], qw.shape[3]
        kK = kH * kW
        nb = C // bs

        x_unf = F.unfold(qi, (kH, kW),
                         dilation=self.dilation,
                         padding=self.padding,
                         stride=self.stride)
        L = x_unf.shape[-1]
        x_b = x_unf.view(B, nb, bs, kK, L)
        w_b = qw.view(O, nb, bs, kK)
        out = torch.einsum('bnkpl,onkp->bol', x_b, w_b)

        sH, sW = self.stride
        pH, pW = self.padding
        dH, dW = self.dilation
        H_out = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
        W_out = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
        out = out.view(B, O, H_out, W_out)
        if bf_bias is not None:
            out = out + bf_bias.view(1, -1, 1, 1)
        out = quantize_elemwise_op(out, mx_specs=sp, round=sp['round_output'])
        return out
