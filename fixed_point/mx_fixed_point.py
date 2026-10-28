import torch

_MIN_BITS = 32
_MAX_BITS = 64
_DEFAULT_BITS = 48
_VALID_MODES = ("fp32_partial", "hw_fixed_point")
_VALID_SAT_MODES = ("per_product", "per_block")
_VALID_WEIGHT_BLOCKIFY = ("channel", "flatten")
_VALID_ACT_BLOCKIFY = ("channel", "xblock")
_VALID_NPE_TRITON_VARIANTS = ("reuse", "opt")
_VALID_OUT_WINDOW = ("off", "infer", "always")
_VALID_OUT_ROUND = ("floor", "nearest")
_VALID_IN_EXP_MODE = ("off", "infer", "always")
_VALID_CONVT = ("plain", "hw")
_VALID_PLA = ("off", "infer")
_VALID_PLA_BIAS = ("pre", "post")


def validate_xblock_accum_bits(bits):
    if not isinstance(bits, int):
        raise TypeError(f"xblock_accum_bits must be int, got {type(bits).__name__}")
    if bits < _MIN_BITS or bits > _MAX_BITS:
        raise ValueError(
            f"xblock_accum_bits={bits} out of range [{_MIN_BITS}, {_MAX_BITS}]"
        )
    return bits


def _auto_scale_exp(partials, total_bits, headroom_bits=2):
    """Pick scale_exp so the *accumulated* sum fits in the target window.

    Worst-case cumulative sum is `num_blocks * max(|partial|)`, so the
    per-partial max must be bounded by `2^(total_bits-1-headroom) / num_blocks`
    to guarantee no saturation.
    """
    max_abs = partials.detach().abs().amax()
    if max_abs.item() == 0.0:
        return 0
    num_blocks = max(1, partials.shape[-1])
    target = float(1 << (total_bits - 1 - headroom_bits)) / num_blocks
    exp = int(torch.floor(torch.log2(target / max_abs)).item())
    return exp


class FixedPointAccumulator(torch.autograd.Function):
    """
    Emulate a signed saturating N-bit fixed-point cross-block accumulator.

    Forward:
        int_partials = round(partials * 2^scale_exp).to(int64)
        acc = 0
        for k in range(num_blocks):
            acc = clamp(acc + int_partials[..., k], -2^(N-1), 2^(N-1)-1)
        out = acc.float() * 2^(-scale_exp)

    Backward: STE — gradient flows to all partials unchanged.
        Optional: zero grad for elements whose final accumulator saturated
        (set via ste_mask flag). Backward cost is ~free.
    """

    @staticmethod
    def forward(ctx, partials, total_bits, scale_exp, saturate, ste_mask):
        total_bits = int(total_bits)
        validate_xblock_accum_bits(total_bits)

        if scale_exp is None:
            scale_exp = _auto_scale_exp(partials, total_bits)
        scale_exp = int(scale_exp)

        lo = -(1 << (total_bits - 1))
        hi = (1 << (total_bits - 1)) - 1

        scale = float(2 ** scale_exp)
        inv_scale = float(2 ** -scale_exp)

        int_partials = torch.round(partials.detach().to(torch.float64) * scale).to(torch.int64)

        num_blocks = int_partials.shape[-1]
        acc = torch.zeros(int_partials.shape[:-1], dtype=torch.int64, device=partials.device)
        saturated_mask = torch.zeros_like(acc, dtype=torch.bool)

        for k in range(num_blocks):
            acc = acc + int_partials[..., k]
            if saturate:
                over = acc > hi
                under = acc < lo
                if over.any() or under.any():
                    saturated_mask = saturated_mask | over | under
                    acc = torch.clamp(acc, min=lo, max=hi)

        out = acc.to(partials.dtype) * inv_scale

        ctx.save_for_backward(saturated_mask)
        ctx.ste_mask = bool(ste_mask)
        ctx.num_blocks = num_blocks
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (saturated_mask,) = ctx.saved_tensors
        grad = grad_out.unsqueeze(-1).expand(*grad_out.shape, ctx.num_blocks)
        if ctx.ste_mask and saturated_mask.any():
            keep = (~saturated_mask).to(grad.dtype).unsqueeze(-1)
            grad = grad * keep
        return grad.contiguous(), None, None, None, None


def fixed_point_accumulate(
    partials,
    total_bits=_DEFAULT_BITS,
    scale_exp=None,
    saturate=True,
    ste_mask=False,
):
    """
    Sum `partials` along the last dim through an emulated N-bit signed saturating
    fixed-point accumulator. FP32 in, FP32 out. STE backward.

    partials : (..., num_blocks) FP32/FP64 tensor
    total_bits : accumulator bitwidth, default 48, valid [40, 48]
    scale_exp  : int. If None, chosen automatically per-call from partials range
    saturate   : if False, behaves as exact int64 sum (for debugging)
    ste_mask   : if True, zero grad on lanes whose accumulator saturated
    """
    return FixedPointAccumulator.apply(partials, total_bits, scale_exp, saturate, ste_mask)


_VALID_BACKENDS = ("python", "triton")


XBLOCK_ACCUM_DEFAULTS = {
    "enabled": True,
    "bits": _DEFAULT_BITS,
    "backend": "python",
    "scale_exp": None,   # None = auto-pick per forward from partial range
    "saturate": True,
    "ste_mask": False,
    # HW-faithful emulation knobs. Only consulted when mode == 'hw_fixed_point'.
    "mode": "fp32_partial",       # 'fp32_partial' (legacy einsum+xblock_accum) or 'hw_fixed_point'
    "sat_mode": "per_product",    # 'per_product' (HW-faithful) or 'per_block' (sum-then-sat; fast)
    # NPE blockify (hw_fixed_point only). Defaults 'channel'/'channel' = block both
    # operands along Cin (current behavior). NPE = weight 'flatten' + act 'xblock'.
    "weight_blockify": "channel", # 'channel' (block Cin) or 'flatten' (per-filter [Cin,kH,kW]->1D)
    "act_blockify": "channel",    # 'channel' (block Cin) or 'xblock' (block along W/width)
    "npe_triton_variant": "reuse",# NPE+triton only: 'reuse' (existing kernel, bs=1) or
                                  # 'opt' (dedicated per-lane kernel). Bit-identical; cross-check.
    "e_layer_min": None,          # int; required for 'hw_fixed_point' inference (set via calibration)
    "pad_channels": True,         # if True, MXConv2dHW pads C up to a multiple of bs (zeros).
                                  # Disable to force fallback to MXConv2d for non-divisible layers.
    "verbose": 1,                 # 0 = silent, 1 = quant-time summary + sampled per-layer log,
                                  # 2 = full per-forward / per-layer logging.
    "verbose_sample_every": 100,  # at verbose=1, emit one saturation summary per N forwards
                                  # per layer (only if non-zero saturation observed).
    # Static output window (hw_fixed_point only): O = N >> (e_qa_out - e_layer_min),
    # saturated to +-(2^win_bits - 1). e_qa_out comes from calibrate_hw_grids.
    "out_window": "off",          # 'off' (fp32 cast of the acc), 'infer' (window in eval mode
                                  # only), 'always' (train too; backward stays STE either way)
    "out_window_bits": None,      # magnitude bits kept by the window. None = derive what
                                  # the output MX block can address: (mbits-1) +
                                  # (2^out_exp_bits - 1); int8 + 4b exp -> 22
    "out_exp_bits": 4,            # HW output shared-exponent width (all 2^bits codes usable)
    "out_round": "floor",         # bits dropped below the window: 'floor' (arithmetic shift)
                                  # or 'nearest' (round half up)
    "out_margin_bits": 0,         # headroom added to e_qa_out at calibration
    "e_qa_out": None,             # int; output window grid for every layer of this spec.
                                  # Per-layer values go in the config's "hw_grids".
    # Accumulator headroom (calibration only): floor e_layer_min so the largest
    # product and the output fit in `bits` with this many spare bits. None =
    # pure min block exponent (a wide-range input can then overflow the acc).
    "acc_headroom_bits": None,
    # HW shared-exponent register for inputs/weights: per tensor only
    # 2^in_exp_bits block exponents, [e_base, e_base + 2^in_exp_bits - 1].
    # Blocks below re-round to the e_base step, above saturate. e_base =
    # max block exp - (2^in_exp_bits - 1) (anchored at the max); inputs
    # calibrated (hw_grids e_base_a), weights from the weights (e_base_w).
    "in_exp_bits": None,          # None = unlimited (microxcaling scale_bits only)
    "in_exp_mode": "off",         # 'off', 'infer' (eval only) or 'always' (train too;
                                  # backward stays STE)
    # ConvTranspose2d under hw_fixed_point: 'plain' keeps plain MXConvTranspose2d,
    # 'hw' builds MXConvTranspose2dHW (zero-insert conv on the MXConv2dHW path).
    # Grouped convT stays plain either way.
    "convT": "plain",
    # PLA (hw_fixed_point + out_window only): fold the PReLU / LeakyReLU / ReLU
    # that directly follows an MXConv2dHW in an nn.Sequential into the PCU
    # multiply before the window (see mx_fixed_point_hw.PLAParams). 'infer' =
    # eval mode only; train mode runs the float activation module as before.
    "pla": "off",
    "pla_mult_bits": 8,           # signed width of the combined multiplier
    "pla_frac_bits": None,        # its fraction bits; None = largest that fits
    "pla_bias": "pre",            # 'pre' (bias in acc, PReLU order) or 'post'
                                  # (emulator pcu_pla order: after the multiply)
}


def normalize_xblock_accum(value):
    """Normalize user-supplied xblock_accum config into a canonical dict.

    Accepts:
      - None / False → disabled (fall back to plain FP32 sum).
      - True         → enabled with defaults.
      - dict         → merged over defaults; unknown keys raise.

    Validates bits range, backend, and scale_exp type.
    """
    if value is None or value is False:
        return {**XBLOCK_ACCUM_DEFAULTS, "enabled": False}
    if value is True:
        return dict(XBLOCK_ACCUM_DEFAULTS)
    if not isinstance(value, dict):
        raise TypeError(
            f"xblock_accum must be bool or dict, got {type(value).__name__}"
        )

    cfg = dict(XBLOCK_ACCUM_DEFAULTS)
    for k, v in value.items():
        if k not in cfg:
            raise ValueError(
                f"unknown xblock_accum key: {k!r}; "
                f"valid keys: {sorted(cfg.keys())}"
            )
        cfg[k] = v

    if cfg["enabled"]:
        validate_xblock_accum_bits(cfg["bits"])
        if cfg["backend"] not in _VALID_BACKENDS:
            raise ValueError(
                f"xblock_accum.backend must be in {_VALID_BACKENDS}, "
                f"got {cfg['backend']!r}"
            )
        if cfg["scale_exp"] is not None and not isinstance(cfg["scale_exp"], int):
            raise TypeError(
                f"xblock_accum.scale_exp must be None or int, "
                f"got {type(cfg['scale_exp']).__name__}"
            )
        if cfg["mode"] not in _VALID_MODES:
            raise ValueError(
                f"xblock_accum.mode must be in {_VALID_MODES}, got {cfg['mode']!r}"
            )
        if cfg["sat_mode"] not in _VALID_SAT_MODES:
            raise ValueError(
                f"xblock_accum.sat_mode must be in {_VALID_SAT_MODES}, "
                f"got {cfg['sat_mode']!r}"
            )
        if cfg["e_layer_min"] is not None and not isinstance(cfg["e_layer_min"], int):
            raise TypeError(
                f"xblock_accum.e_layer_min must be None or int, "
                f"got {type(cfg['e_layer_min']).__name__}"
            )
        if not isinstance(cfg["pad_channels"], bool):
            raise TypeError(
                f"xblock_accum.pad_channels must be bool, "
                f"got {type(cfg['pad_channels']).__name__}"
            )
        if not isinstance(cfg["verbose"], int) or cfg["verbose"] < 0 or cfg["verbose"] > 2:
            raise ValueError(
                f"xblock_accum.verbose must be int in [0, 2], "
                f"got {cfg['verbose']!r}"
            )
        if not isinstance(cfg["verbose_sample_every"], int) or cfg["verbose_sample_every"] < 1:
            raise ValueError(
                f"xblock_accum.verbose_sample_every must be a positive int, "
                f"got {cfg['verbose_sample_every']!r}"
            )
        if cfg["weight_blockify"] not in _VALID_WEIGHT_BLOCKIFY:
            raise ValueError(
                f"xblock_accum.weight_blockify must be in {_VALID_WEIGHT_BLOCKIFY}, "
                f"got {cfg['weight_blockify']!r}"
            )
        if cfg["act_blockify"] not in _VALID_ACT_BLOCKIFY:
            raise ValueError(
                f"xblock_accum.act_blockify must be in {_VALID_ACT_BLOCKIFY}, "
                f"got {cfg['act_blockify']!r}"
            )
        if cfg["npe_triton_variant"] not in _VALID_NPE_TRITON_VARIANTS:
            raise ValueError(
                f"xblock_accum.npe_triton_variant must be in "
                f"{_VALID_NPE_TRITON_VARIANTS}, got {cfg['npe_triton_variant']!r}"
            )
        if cfg["e_qa_out"] is not None and (
                isinstance(cfg["e_qa_out"], bool) or not isinstance(cfg["e_qa_out"], int)):
            raise TypeError(
                f"xblock_accum.e_qa_out must be None or int, "
                f"got {type(cfg['e_qa_out']).__name__}"
            )
        if cfg["out_window"] not in _VALID_OUT_WINDOW:
            raise ValueError(
                f"xblock_accum.out_window must be in {_VALID_OUT_WINDOW}, "
                f"got {cfg['out_window']!r}"
            )
        if cfg["out_round"] not in _VALID_OUT_ROUND:
            raise ValueError(
                f"xblock_accum.out_round must be in {_VALID_OUT_ROUND}, "
                f"got {cfg['out_round']!r}"
            )
        if cfg["out_window_bits"] is not None and (
                not isinstance(cfg["out_window_bits"], int)
                or not 1 <= cfg["out_window_bits"] <= 46):
            raise ValueError(
                f"xblock_accum.out_window_bits must be None (derive) or int in "
                f"[1, 46], got {cfg['out_window_bits']!r}"
            )
        if not isinstance(cfg["out_exp_bits"], int) or not 1 <= cfg["out_exp_bits"] <= 5:
            raise ValueError(
                f"xblock_accum.out_exp_bits must be int in [1, 5], "
                f"got {cfg['out_exp_bits']!r}"
            )
        if cfg["acc_headroom_bits"] is not None and (
                isinstance(cfg["acc_headroom_bits"], bool)
                or not isinstance(cfg["acc_headroom_bits"], int)
                or cfg["acc_headroom_bits"] < 0):
            raise ValueError(
                f"xblock_accum.acc_headroom_bits must be None or a non-negative "
                f"int, got {cfg['acc_headroom_bits']!r}"
            )
        if cfg["in_exp_bits"] is not None and (
                isinstance(cfg["in_exp_bits"], bool)
                or not isinstance(cfg["in_exp_bits"], int)
                or not 1 <= cfg["in_exp_bits"] <= 8):
            raise ValueError(
                f"xblock_accum.in_exp_bits must be None or int in [1, 8], "
                f"got {cfg['in_exp_bits']!r}"
            )
        if cfg["in_exp_mode"] not in _VALID_IN_EXP_MODE:
            raise ValueError(
                f"xblock_accum.in_exp_mode must be in {_VALID_IN_EXP_MODE}, "
                f"got {cfg['in_exp_mode']!r}"
            )
        if cfg["convT"] not in _VALID_CONVT:
            raise ValueError(
                f"xblock_accum.convT must be in {_VALID_CONVT}, got {cfg['convT']!r}"
            )
        if cfg["in_exp_mode"] != "off" and cfg["in_exp_bits"] is None:
            raise ValueError(
                "xblock_accum.in_exp_mode is on but in_exp_bits is None")
        if not isinstance(cfg["out_margin_bits"], int) or cfg["out_margin_bits"] < 0:
            raise ValueError(
                f"xblock_accum.out_margin_bits must be a non-negative int, "
                f"got {cfg['out_margin_bits']!r}"
            )
        if cfg["pla"] not in _VALID_PLA:
            raise ValueError(
                f"xblock_accum.pla must be in {_VALID_PLA}, got {cfg['pla']!r}")
        if cfg["pla_bias"] not in _VALID_PLA_BIAS:
            raise ValueError(
                f"xblock_accum.pla_bias must be in {_VALID_PLA_BIAS}, "
                f"got {cfg['pla_bias']!r}")
        mb = cfg["pla_mult_bits"]
        if isinstance(mb, bool) or not isinstance(mb, int) or not 2 <= mb <= 16:
            raise ValueError(
                f"xblock_accum.pla_mult_bits must be int in [2, 16], got {mb!r}")
        fb = cfg["pla_frac_bits"]
        if fb is not None and (isinstance(fb, bool) or not isinstance(fb, int)
                               or not 0 <= fb <= mb - 2):
            raise ValueError(
                f"xblock_accum.pla_frac_bits must be None or int in "
                f"[0, pla_mult_bits - 2], got {fb!r}")
        if cfg["pla"] != "off":
            if cfg["out_window"] == "off":
                raise ValueError("xblock_accum.pla needs out_window 'infer' or "
                                 "'always' (PLA sits in front of the window)")
            if cfg["bits"] + mb > 62:
                raise ValueError(
                    f"xblock_accum: bits + pla_mult_bits = {cfg['bits'] + mb} "
                    f"does not fit the int64 emulation (max 62)")
        # Phase 1 supports only the two documented combinations.
        _combo = (cfg["act_blockify"], cfg["weight_blockify"])
        if _combo not in (("channel", "channel"), ("xblock", "flatten")):
            raise ValueError(
                f"unsupported blockify combination act={cfg['act_blockify']!r}, "
                f"weight={cfg['weight_blockify']!r}; supported: "
                f"('channel','channel') [default] or ('xblock','flatten') [NPE]"
            )
    return cfg


def _get_xblock_cfg(mx_specs):
    """Read the xblock_accum config dict from a specs object.

    Prefers python attribute (bypasses microxcaling apply_mx_specs key check),
    falls back to dict-style __getitem__ for plain-dict specs (used in tests).
    Returns {'enabled': False, ...} when unset.
    """
    if hasattr(mx_specs, 'xblock_accum'):
        val = getattr(mx_specs, 'xblock_accum')
    else:
        try:
            val = mx_specs['xblock_accum']
        except (KeyError, TypeError):
            val = None
    if isinstance(val, dict) and 'enabled' in val:
        return val
    return normalize_xblock_accum(val)


def cross_block_accumulate_from_specs(partials, mx_specs):
    """Reduce per-block FP32 partials under the layer's xblock_accum config.

    Disabled (or unset) → plain `partials.sum(-1)`.
    Enabled → N-bit signed saturating int64-emulated accumulator (python or triton).
    """
    cfg = _get_xblock_cfg(mx_specs)
    if not cfg.get("enabled", False):
        return partials.sum(dim=-1)

    bits = cfg["bits"]
    saturate = cfg["saturate"]
    ste_mask = cfg["ste_mask"]
    scale_exp = cfg["scale_exp"]
    backend = cfg["backend"]

    if backend == "triton":
        from .mx_fixed_point_triton import fixed_point_accumulate_triton
        return fixed_point_accumulate_triton(
            partials, total_bits=bits, scale_exp=scale_exp,
            saturate=saturate, ste_mask=ste_mask,
        )

    return fixed_point_accumulate(
        partials, total_bits=bits, scale_exp=scale_exp,
        saturate=saturate, ste_mask=ste_mask,
    )
