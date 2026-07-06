"""Standalone, eyes-on debugger for a single MX conv layer.

`debug_layer(layer, x)` (alias `debug`) prints a structured, copy-pastable trace of
exactly what one conv layer does to one input: the quantization snapshot, the FP32 vs
MX/HW output gap, a per-block element-level MAC walk-through, and ASCII distributions.

Designed to be dropped at a live pdb breakpoint mid-forward:

    >>> import mx_debug; mx_debug.debug(self, x)   # self is the layer, x its input

It is side-effect free: `x` is detached, all compute runs under `torch.no_grad()`, and the
layer's training flag, HW saturation counters and `e_layer_min` are saved and restored. For
`MXConv2dHW` with an uncalibrated `e_layer_min`, one is derived from this input (and then
restored to its prior value) so the call works with zero prerequisites.

Supported layer types:
  * MXConv2dHW       - int8 shift/saturate HW accumulator trace.
  * MXConv2dBlocked  - FP32 per-block partials + fixed-point cross-block accumulator.
  * base MXConv2d    - plain MX path (MX-quantize operands, then F.conv2d).
"""

import math

import torch
import torch.nn.functional as F

from microxcaling.mx.convolution import Conv2d as MXConv2d
from microxcaling.mx.elemwise_ops import quantize_elemwise_op

from mx_layers_blocked import (
    MXConv2dHW, MXConv2dBlocked, quantize_mx_op,
)
from fixed_point.mx_fixed_point_hw import extract_mxint, _int_format_params
from fixed_point.mx_fixed_point import _get_xblock_cfg


_SPARK = "▁▂▃▄▅▆▇█"


# ----------------------------------------------------------------------------
# small formatting / numeric helpers
# ----------------------------------------------------------------------------

def _f(t):
    """Tensor-or-number -> python float."""
    return float(t.item()) if torch.is_tensor(t) else float(t)


def _row(vals, n, fmt="{: .3g}"):
    """First `n` entries of a 1-D sequence as a compact bracketed string."""
    vals = list(vals)
    shown = ", ".join(fmt.format(_f(v)) for v in vals[:n])
    tail = " ..." if len(vals) > n else ""
    return f"[{shown}{tail}]"


def _irow(vals, n):
    vals = list(vals)
    shown = ", ".join(f"{int(_f(v)):d}" for v in vals[:n])
    tail = " ..." if len(vals) > n else ""
    return f"[{shown}{tail}]"


def _sqnr_db(x, xq):
    """Signal-to-quantization-noise ratio in dB: 10*log10(sum(x^2)/sum((x-xq)^2))."""
    sig = (x.double() ** 2).sum().item()
    noise = ((x.double() - xq.double()) ** 2).sum().item()
    if noise <= 0:
        return float("inf")
    if sig <= 0:
        return float("nan")
    return 10.0 * math.log10(sig / noise)


def _is_int_fmt(fmt):
    return isinstance(fmt, str) and fmt.startswith("int")


def _conv_out(n, k, s, p, d):
    return (n + 2 * p - d * (k - 1) - 1) // s + 1


def _pair(v):
    return v if isinstance(v, tuple) else (v, v)


def _ascii_hist(values, bins=12, width=40, label=""):
    """Counts-per-bin ASCII histogram of a 1-D float tensor."""
    v = values.detach().flatten().float().cpu()
    if v.numel() == 0:
        return f"  {label}: (empty)"
    lo = _f(v.min())
    hi = _f(v.max())
    if hi <= lo:
        return f"  {label}: all == {lo:.4g}  (n={v.numel()})"
    edges = torch.linspace(lo, hi, bins + 1)
    idx = torch.bucketize(v, edges[1:-1].contiguous())
    counts = torch.bincount(idx, minlength=bins)
    cmax = int(counts.max().item()) or 1
    lines = [f"  {label}  (min {lo:.4g}, max {hi:.4g}, n={v.numel()})"]
    for b in range(bins):
        c = int(counts[b].item())
        bar = "█" * round(width * c / cmax)
        lines.append(f"    [{edges[b]:+8.3g}, {edges[b + 1]:+8.3g}) {c:7d} {bar}")
    return "\n".join(lines)


def _int_hist(int_values, label=""):
    """Histogram over discrete integer values (e.g. shared exponents)."""
    v = int_values.detach().flatten().long().cpu()
    if v.numel() == 0:
        return f"  {label}: (empty)"
    lo = int(v.min().item())
    hi = int(v.max().item())
    counts = torch.bincount((v - lo), minlength=(hi - lo + 1))
    cmax = int(counts.max().item()) or 1
    lines = [f"  {label}  (n={v.numel()})"]
    for e in range(lo, hi + 1):
        c = int(counts[e - lo].item())
        if c == 0:
            continue
        bar = "█" * round(40 * c / cmax)
        lines.append(f"    e={e:+4d} {c:7d} {bar}")
    return "\n".join(lines)


def _sparkline(values):
    v = values.detach().flatten().float().cpu()
    if v.numel() == 0:
        return ""
    lo = _f(v.min())
    hi = _f(v.max())
    if hi <= lo:
        return _SPARK[0] * v.numel()
    idx = ((v - lo) / (hi - lo) * (len(_SPARK) - 1)).round().long()
    return "".join(_SPARK[int(i)] for i in idx)


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------

def debug_layer(layer, x, *, e_layer_min=-20, out_idx=None, n_blocks=2, n_lanes=8,
                show_dist=True):
    """Print an interpretable trace of `layer(x)` and return the computed numbers.

    Parameters
    ----------
    layer       : MXConv2dHW | MXConv2dBlocked | microxcaling Conv2d.
    x           : input activation tensor [B, C, H, W] (detached internally).
    e_layer_min : HW accumulator exponent grid (MXConv2dHW only). Default -20; pass a
                  different value if the layer was calibrated to one. Temporarily applied
                  to the layer for the faithful forward, then restored. Ignored for
                  MXConv2dBlocked / base MXConv2d.
    out_idx     : (o, y, x) output element to deep-dive; None -> argmax |fp32 out|.
    n_blocks    : channel-blocks to print in the element deep-dive (default 2).
    n_lanes     : lanes printed per block (default 8).
    show_dist   : print the ASCII distribution section.

    Returns
    -------
    dict of computed numbers (also useful for tests / programmatic inspection).
    """
    if layer.mx_none:
        print("[debug_layer] layer.mx_none is True -> MX disabled; nothing to trace.")
        return {"mx_none": True}

    is_hw = isinstance(layer, MXConv2dHW)
    is_blk = (not is_hw) and isinstance(layer, MXConv2dBlocked)
    kind = "MXConv2dHW" if is_hw else "MXConv2dBlocked" if is_blk else "MXConv2d"

    x = x.detach()
    was_training = layer.training
    layer.eval()

    sp = layer.mx_specs
    bs = int(sp["block_size"])
    a_fmt = sp["a_elem_format"]
    w_fmt = sp["w_elem_format"]
    name = (getattr(layer, "_mx_layer_name", None) or getattr(layer, "name", None)
            or f"{kind.lower()}-{id(layer):x}")

    stride = _pair(layer.stride)
    padding = _pair(layer.padding)
    dilation = _pair(layer.dilation)

    result = {"kind": kind, "name": name}

    try:
        with torch.no_grad():
            _run(layer, x, sp, bs, a_fmt, w_fmt, name, kind, is_hw, is_blk,
                 stride, padding, dilation, int(e_layer_min), out_idx, n_blocks,
                 n_lanes, show_dist, result)
    finally:
        if was_training:
            layer.train()

    return result


debug = debug_layer  # short alias: debug(layer, input)


def _run(layer, x, sp, bs, a_fmt, w_fmt, name, kind, is_hw, is_blk,
         stride, padding, dilation, e_layer_min, out_idx, n_blocks, n_lanes,
         show_dist, result):

    weight = layer.weight.detach()
    bias = layer.bias.detach() if layer.bias is not None else None
    x_orig = x  # unpadded input; the layer does its own channel-padding in forward()
    B, C, H, W = x.shape
    O = weight.shape[0]
    kH, kW = weight.shape[2], weight.shape[3]

    # --- HW: zero-pad channels to a block multiple, mirroring MXConv2dHW.forward.
    e_min_saved = None
    pad = 0
    if is_hw and C % bs != 0:
        cfg0 = _get_xblock_cfg(layer)
        if not bool(cfg0.get("pad_channels", True)):
            print(f"[debug_layer] in_channels {C} not divisible by block_size {bs} "
                  f"and pad_channels=False; cannot trace.")
            return
        pad = bs - (C % bs)
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        weight = F.pad(weight, (0, 0, 0, 0, 0, pad))
        B, C, H, W = x.shape

    nb = C // bs if C % bs == 0 else 1
    eff_bs = bs if C % bs == 0 else C
    kK = kH * kW

    # --- quantize operands (same primitives the layers use).
    bf_in = quantize_elemwise_op(x, mx_specs=sp, round=sp["round_output"])
    bf_w = quantize_elemwise_op(weight, mx_specs=sp, round=sp["round_weight"])
    bf_bias = (quantize_elemwise_op(bias, mx_specs=sp, round=sp["round_weight"])
               if bias is not None else None)
    qi = quantize_mx_op(bf_in, sp, elem_format=a_fmt, axes=[1])
    qw = quantize_mx_op(bf_w, sp, elem_format=w_fmt, axes=[1])
    if qw.dtype == torch.bfloat16 and qi.dtype != torch.bfloat16:
        qw = qw.to(qi.dtype)

    # --- HW: use the supplied e_layer_min as the accumulator grid. Apply it to the
    # layer for the faithful forward, restoring the prior value afterwards.
    cfg = _get_xblock_cfg(layer) if (is_hw or is_blk) else None
    e_min = None
    if is_hw:
        e_min = int(e_layer_min)
        e_min_saved = layer.e_layer_min
        layer.e_layer_min = e_min

    # --- header
    H_out = _conv_out(H, kH, stride[0], padding[0], dilation[0])
    W_out = _conv_out(W, kW, stride[1], padding[1], dilation[1])
    print("=" * 78)
    head = (f"Debug {kind} '{name}' | in={C}{'(+%d pad)' % pad if pad else ''} out={O} "
            f"k=({kH},{kW}) s={stride} p={padding} bs={bs} fmt={a_fmt}")
    print(head)
    if is_hw:
        note = " (default)" if e_min == -20 else " (passed)"
        print(f"  mode={cfg.get('mode')} bits={cfg.get('bits')} "
              f"sat_mode={cfg.get('sat_mode')} backend={cfg.get('backend')} "
              f"e_layer_min={e_min}{note}")
    elif is_blk:
        print(f"  mode={cfg.get('mode')} bits={cfg.get('bits')} "
              f"saturate={cfg.get('saturate')} scale_exp={cfg.get('scale_exp')}")
    print(f"  x: {list(x.shape)}  weight: {list(weight.shape)}  "
          f"out: [{B},{O},{H_out},{W_out}]")
    result.update(in_ch=C, out_ch=O, kernel=(kH, kW), nb=nb, e_layer_min=e_min)

    # --- [1] quantization snapshot
    print("\n[1] Quantization snapshot (orig -> MX)")
    for tag, orig, q in (("input ", x, qi), ("weight", weight, qw)):
        sq = _sqnr_db(orig, q)
        err = (orig - q).abs()
        print(f"  {tag} | max {_f(orig.abs().max()):.4g} mean {_f(orig.abs().mean()):.4g}"
              f" | MXerr max {_f(err.max()):.3g} mean {_f(err.mean()):.3g}"
              f" | SQNR {sq:.1f} dB")
        result[f"{tag.strip()}_sqnr_db"] = sq

    # --- [2] output: FP32 reference vs MX/HW (faithful layer(x), counters restored)
    out_fp32 = F.conv2d(x, weight, bias, stride, padding, dilation, 1)
    out_mx, sat_count, total = _faithful_forward(layer, x_orig, is_hw, e_min_saved)
    err = (out_fp32 - out_mx).abs()
    rel = err / (out_fp32.abs() + 1e-8)
    print("\n[2] Output: FP32 reference vs MX/HW")
    print(f"  fp32 max {_f(out_fp32.abs().max()):.4g} | mx max {_f(out_mx.abs().max()):.4g}"
          f" | err max {_f(err.max()):.4g} mean {_f(err.mean()):.4g}"
          f" rel_mean {_f(rel.mean()):.4g} | SQNR {_sqnr_db(out_fp32, out_mx):.1f} dB")
    if is_hw and total:
        print(f"  saturation: {sat_count} / {total} outputs "
              f"({100.0 * sat_count / total:.3f}%)")
    result.update(out_sqnr_db=_sqnr_db(out_fp32, out_mx),
                  out_err_max=_f(err.max()), out_err_mean=_f(err.mean()),
                  sat_count=sat_count, sat_total=total)

    # --- [3] element deep-dive
    if out_idx is None:
        flat = out_fp32[0].abs().reshape(-1).argmax().item()
        o = flat // (H_out * W_out)
        rem = flat % (H_out * W_out)
        oy, ox = rem // W_out, rem % W_out
        pick_note = "  [auto: max |fp32|]"
    else:
        o, oy, ox = out_idx
        pick_note = "  [user out_idx]"
    print(f"\n[3] Element deep-dive @ out(o={o}, y={oy}, x={ox}){pick_note}")
    l = oy * W_out + ox

    if is_hw:
        trace = _deep_dive_hw(qi, qw, x, weight, o, l, bs, nb, kK, a_fmt, e_min,
                              cfg, stride, padding, dilation, bf_bias,
                              n_blocks, n_lanes)
    else:
        trace = _deep_dive_fp(qi, qw, x, weight, o, l, eff_bs, nb, kK,
                              stride, padding, dilation, n_blocks, n_lanes,
                              a_fmt, bs, bias, bf_bias)
    fp32_mac = trace["fp32_mac"]
    mx_mac = trace["mx_mac"]
    abse = abs(fp32_mac - mx_mac)
    relp = 100.0 * abse / (abs(fp32_mac) + 1e-12)
    print(f"  TOTAL  FP32 MAC = {fp32_mac:.5g}  |  "
          f"{'HW' if is_hw else 'MX'} MAC = {mx_mac:.5g}  |  "
          f"abs err {abse:.4g} ({relp:.2f}%)")
    print(f"         layer out[{o},{oy},{ox}] = {_f(out_mx[0, o, oy, ox]):.5g}")
    result["block_trace"] = trace

    # --- [4] distributions
    if show_dist:
        print("\n[4] Distributions (ASCII)")
        if _is_int_fmt(a_fmt) and C % bs == 0:
            _, Ea = extract_mxint(qi, bs, axis=1, fmt=a_fmt)
            _, Ew = extract_mxint(qw, bs, axis=1, fmt=w_fmt)
            print(_int_hist(Ea, "Ea shared-exp"))
            print(_int_hist(Ew, "Ew shared-exp"))
        print(_ascii_hist(err, label="output abs-err"))
        per_oc = (out_fp32[0] - out_mx[0]).abs().flatten(1).mean(dim=1)
        print(f"  per-out-channel mean err: {_sparkline(per_oc)}")
        print(f"    (min {_f(per_oc.min()):.3g} @ch{int(per_oc.argmin())}, "
              f"max {_f(per_oc.max()):.3g} @ch{int(per_oc.argmax())})")
    print("=" * 78)


def _faithful_forward(layer, x, is_hw, e_min_saved):
    """Run layer(x) for the true MX/HW output; restore HW counters and the prior
    e_layer_min afterwards so the debug call leaves no trace.
    Returns (out, sat_count, total)."""
    if not is_hw:
        return layer(x), 0, 0

    saved = {k: getattr(layer, k) for k in
             ("_fwd_count", "_sat_total_life", "_sat_seen_life",
              "_sat_total_window", "_sat_seen_window")}
    try:
        out = layer(x)
        sat_count = int(layer._sat_seen_life - saved["_sat_seen_life"])
        total = int(layer._sat_total_life - saved["_sat_total_life"])
    finally:
        for k, v in saved.items():
            setattr(layer, k, v)
        layer.e_layer_min = e_min_saved  # restore prior value (often None)
    return out, sat_count, total


# ----------------------------------------------------------------------------
# element deep-dives
# ----------------------------------------------------------------------------

def _col(t, l, kH, kW, stride, padding, dilation):
    """Unfold `t` and return column `l`: shape [C*kH*kW] (c outer, kk inner)."""
    unf = F.unfold(t.float(), (kH, kW), dilation=dilation,
                   padding=padding, stride=stride)
    return unf[0, :, l]


def _deep_dive_hw(qi, qw, x, weight, o, l, bs, nb, kK, fmt, e_min, cfg,
                  stride, padding, dilation, bf_bias, n_blocks, n_lanes):
    """Reproduce the HW int8 shift/saturate MAC for one output element."""
    _, mant_bias, _, _ = _int_format_params(fmt)
    bits = int(cfg.get("bits"))
    sat_mode = cfg.get("sat_mode")
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    two_bias = 2 * int(mant_bias)
    kH, kW = weight.shape[2], weight.shape[3]

    qi_i8, Ea = extract_mxint(qi, bs, axis=1, fmt=fmt)
    qw_i8, Ew = extract_mxint(qw, bs, axis=1, fmt=fmt)

    # columns for this output element (lists of python ints / floats).
    ci = _col(qi_i8, l, kH, kW, stride, padding, dilation).view(nb, bs, kK).long()
    cEa = _col(Ea, l, kH, kW, stride, padding, dilation).view(nb, kK).long()
    cw = qw_i8[o].view(nb, bs, kK).long()
    cEw = Ew[o].view(nb, kK).long()
    # FP32 reference patch (full precision) and MX-fp values for display.
    cx = _col(x, l, kH, kW, stride, padding, dilation).view(nb, bs, kK)
    wf = weight[o].view(nb, bs, kK)
    cqi = _col(qi, l, kH, kW, stride, padding, dilation).view(nb, bs, kK)
    cqw = qw[o].view(nb, bs, kK)

    acc = 0
    fp32_mac = 0.0
    blocks = []
    for kb in range(nb):
        fp_block = 0.0
        for kk in range(kK):
            ew = int(cEw[kb, kk]); ea = int(cEa[kb, kk])
            shift = ew + ea - two_bias - e_min
            for j in range(bs):
                p = int(ci[kb, j, kk]) * int(cw[kb, j, kk])
                s = p << shift if shift >= 0 else p >> (-shift)
                acc += s
                if sat_mode == "per_product":
                    acc = max(lo, min(hi, acc))
                fp_block += _f(cx[kb, j, kk]) * _f(wf[kb, j, kk])
            if sat_mode != "per_product":
                acc = max(lo, min(hi, acc))
        fp32_mac += fp_block
        if kb < n_blocks:
            kk0 = 0
            shift0 = int(cEw[kb, kk0]) + int(cEa[kb, kk0]) - two_bias - e_min
            prods = [int(ci[kb, j, kk0]) * int(cw[kb, j, kk0]) for j in range(bs)]
            print(f"  --- block {kb} (ch {kb * bs}-{kb * bs + bs - 1}) ---")
            print(f"    Ew_shared={int(cEw[kb, kk0])}  Ea_shared(tap0)={int(cEa[kb, kk0])}"
                  f"  shift(tap0)=Ew+Ea-2*{mant_bias}-({e_min})={shift0:+d}")
            print(f"    in  int[:{n_lanes}] = {_irow(ci[kb, :, kk0].tolist(), n_lanes)}"
                  f"   (MX fp = {_row(cqi[kb, :, kk0].tolist(), n_lanes)})")
            print(f"    w   int[:{n_lanes}] = {_irow(cw[kb, :, kk0].tolist(), n_lanes)}"
                  f"   (MX fp = {_row(cqw[kb, :, kk0].tolist(), n_lanes)})")
            print(f"    prod[:{n_lanes}]    = {_irow(prods, n_lanes)}  "
                  f"{'<<' if shift0 >= 0 else '>>'} {abs(shift0)}")
            print(f"    acc after block {kb} = {acc}   "
                  f"(FP32 block partial = {fp_block:.4g} ; HW deq = {acc * 2.0 ** e_min:.4g})")
        blocks.append({"kb": kb, "fp_partial": fp_block, "acc": acc})

    if bf_bias is not None:
        bint = round(_f(bf_bias[o]) * 2.0 ** (-e_min))
        acc = max(lo, min(hi, acc + bint))
        fp32_mac += _f(bf_bias[o])
    mx_mac = acc * 2.0 ** e_min
    return {"fp32_mac": fp32_mac, "mx_mac": mx_mac, "blocks": blocks,
            "n_blocks_printed": min(n_blocks, nb)}


def _deep_dive_fp(qi, qw, x, weight, o, l, bs, nb, kK, stride, padding, dilation,
                  n_blocks, n_lanes, fmt, real_bs, bias=None, bf_bias=None):
    """FP32 per-block partial walk for MXConv2dBlocked / base MXConv2d.

    `bs`/`nb` are the display block geometry (whole-channel fallback when C is not
    divisible by block_size). Int columns are shown only for intN formats.
    """
    kH, kW = weight.shape[2], weight.shape[3]
    cqi = _col(qi, l, kH, kW, stride, padding, dilation).view(nb, bs, kK)
    cqw = qw[o].view(nb, bs, kK)
    cx = _col(x, l, kH, kW, stride, padding, dilation).view(nb, bs, kK)
    wf = weight[o].view(nb, bs, kK)

    show_int = _is_int_fmt(fmt) and (real_bs == bs)
    if show_int:
        qi_i8, _ = extract_mxint(qi, real_bs, axis=1, fmt=fmt)
        qw_i8, _ = extract_mxint(qw, real_bs, axis=1, fmt=fmt)
        ci = _col(qi_i8, l, kH, kW, stride, padding, dilation).view(nb, bs, kK).long()
        cw = qw_i8[o].view(nb, bs, kK).long()

    fp32_mac = 0.0
    mx_mac = 0.0
    running = 0.0
    blocks = []
    for kb in range(nb):
        mx_block = float((cqi[kb] * cqw[kb]).sum())
        fp_block = float((cx[kb] * wf[kb]).sum())
        mx_mac += mx_block
        fp32_mac += fp_block
        running += mx_block
        if kb < n_blocks:
            print(f"  --- block {kb} (ch {kb * bs}-{kb * bs + bs - 1}) ---")
            if show_int:
                print(f"    in  int[:{n_lanes}] = {_irow(ci[kb, :, 0].tolist(), n_lanes)}")
                print(f"    w   int[:{n_lanes}] = {_irow(cw[kb, :, 0].tolist(), n_lanes)}")
            print(f"    MX in[:{n_lanes}] = {_row(cqi[kb, :, 0].tolist(), n_lanes)}")
            print(f"    MX w [:{n_lanes}] = {_row(cqw[kb, :, 0].tolist(), n_lanes)}")
            print(f"    MX block partial = {mx_block:.4g}  (FP32 partial = {fp_block:.4g})"
                  f"  running MX sum = {running:.4g}")
        blocks.append({"kb": kb, "mx_partial": mx_block, "fp_partial": fp_block})
    if bf_bias is not None:
        mx_mac += _f(bf_bias[o])
    if bias is not None:
        fp32_mac += _f(bias[o])
    return {"fp32_mac": fp32_mac, "mx_mac": mx_mac, "blocks": blocks,
            "n_blocks_printed": min(n_blocks, nb)}
