"""Per-block quantization statistics and error measurement for MX-quantized models.

Collects, per MX layer (weights and activations):
  * Per-block stats: max_abs, mean_abs, variance, dynamic range
    (max_abs / mean_abs), underflow rate, shared-exponent distribution.
  * Quantization error err = x - Q(x) where Q is the fake-quant
    `quantize_mx_op` output: SQNR (dB), MSE, max abs err, cosine similarity —
    at tensor level and per block.
  * Isolated layer-output error: FP32 functional (conv2d/linear/conv_transpose2d
    on the layer's own stored weights) vs the layer's actual quantized output.
    This is the level at which the MX variants (plain / Blocked / HW) differ —
    they share operand quantization and differ only in accumulation.

Design notes:
  * All accumulation is online (running moments + capped reservoir sample);
    no activations are ever stored — memory is O(reservoir) per layer.
  * Shared exponents are computed as floor(log2(block max_abs)) on the
    pre-quant operand — identical to mx_ops._shared_exponents(method="max"),
    which is what `quantize_mx_op` actually uses.
  * After PTQ the stored weights already lie on the MX lattice, so weight
    quant error is ~0 by construction. That is correct (it reflects the
    operand actually entering forward) and is flagged via "ptq_note".
"""

import json
import math
import os
import sys
from collections import Counter

sys.path.append('/Users/avrahamraviv/PycharmProjects')
sys.path.append('/home/avrahamra/PycharmProjects')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from microxcaling.mx.convolution import Conv2d as MXConv2d, _fwd_block_axes
from microxcaling.mx.transpose_convolution import ConvTranspose2d as MXConvTranspose2d
from microxcaling.mx.linear import Linear as MXLinear
from microxcaling.mx.mx_ops import quantize_mx_op
from microxcaling.mx.elemwise_ops import quantize_elemwise_op

from mx_layers_blocked import MXConv2dBlocked, MXLinearBlocked, MXConv2dHW
from mx_debug import _ascii_hist

_RESERVOIR_CAP = 65536
_MX_TYPES = (MXConv2dHW, MXConv2dBlocked, MXConvTranspose2d, MXConv2d,
             MXLinearBlocked, MXLinear)


def _log(log, msg):
    if log is not None:
        log.info(msg)
    else:
        print(msg)


def _layer_type_name(module):
    """Most-derived readable class name (library classes are named Conv2d/Linear)."""
    for cls, name in ((MXConv2dHW, "MXConv2dHW"),
                      (MXConv2dBlocked, "MXConv2dBlocked"),
                      (MXLinearBlocked, "MXLinearBlocked"),
                      (MXConvTranspose2d, "MXConvTranspose2d"),
                      (MXConv2d, "MXConv2d"),
                      (MXLinear, "MXLinear")):
        if isinstance(module, cls):
            return name
    return type(module).__name__


def _layer_quant_axes(module):
    """Return (act_axes, wt_axes) matching each layer's forward quantization."""
    sp = module.mx_specs
    if isinstance(module, (MXConv2dHW, MXConv2dBlocked)):
        return [1], [1]
    if isinstance(module, MXConvTranspose2d):
        return [1], [0]
    if isinstance(module, MXConv2d):
        return (_fwd_block_axes(sp, key="block_axes_act", default=(1,)),
                _fwd_block_axes(sp, key="block_axes_wt", default=(1,)))
    # MXLinear / MXLinearBlocked
    return [-1], [-1]


def _blockify(t, axes, block_size):
    """Tile `t` along a single quant axis into [n_blocks, block_size].

    Zero-pads the axis to a multiple of block_size (same rule as
    mx_ops._reshape_to_blocks). Returns (blocks, valid_mask) where the mask
    excludes pad elements from element-wise statistics. Zero pads do not
    perturb the block shared exponent (max-abs based), so padded blocks match
    the blocks `quantize_mx_op` actually forms.
    """
    assert len(axes) == 1, f"single quant axis expected, got {axes}"
    axis = axes[0] % t.dim()
    x = t.detach().movedim(axis, -1).contiguous()
    C = x.shape[-1]
    x = x.reshape(-1, C)
    pad = (-C) % block_size
    mask = torch.ones_like(x, dtype=torch.bool)
    if pad:
        x = F.pad(x, (0, pad))
        mask = F.pad(mask, (0, pad))
    return x.view(-1, block_size), mask.view(-1, block_size)


class _RunningStat:
    """Online moments + capped reservoir sample (percentiles / histograms)."""

    def __init__(self, cap=_RESERVOIR_CAP):
        self.n = 0
        self.s = 0.0
        self.s2 = 0.0
        self.mn = float("inf")
        self.mx = float("-inf")
        self.cap = cap
        self.res = torch.empty(0)

    def update(self, v):
        v = v.detach().flatten().float().cpu()
        v = v[torch.isfinite(v)]
        m = v.numel()
        if m == 0:
            return
        self.s += v.sum().item()
        self.s2 += (v * v).sum().item()
        self.mn = min(self.mn, v.min().item())
        self.mx = max(self.mx, v.max().item())
        if self.res.numel() < self.cap:
            take = min(self.cap - self.res.numel(), m)
            self.res = torch.cat([self.res, v[:take]])
            v = v[take:]
        if v.numel():
            # vectorized reservoir replacement (Algorithm R, batched)
            idx = torch.randint(0, self.n + m, (v.numel(),))
            keep = idx < self.cap
            self.res[idx[keep]] = v[keep]
        self.n += m

    def summary(self):
        if self.n == 0:
            return None
        mean = self.s / self.n
        var = max(0.0, self.s2 / self.n - mean * mean)
        q = torch.quantile(self.res, torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99]))
        return {"n": self.n, "mean": mean, "std": math.sqrt(var),
                "min": self.mn, "max": self.mx,
                "p1": q[0].item(), "p25": q[1].item(), "p50": q[2].item(),
                "p75": q[3].item(), "p99": q[4].item()}

    def hist(self, bins=12):
        if self.res.numel() == 0:
            return None
        counts, edges = torch.histogram(self.res, bins=bins)
        return {"edges": edges.tolist(), "counts": counts.long().tolist()}


class _ErrAccum:
    """Online tensor-error accumulator; same conventions as MXQuantizer._measure_error."""

    def __init__(self):
        self.signal_sq = 0.0
        self.noise_sq = 0.0
        self.dot = 0.0
        self.quant_sq = 0.0
        self.max_abs_err = 0.0
        self.n_elem = 0

    def update(self, ref, q):
        ref = ref.detach().float()
        q = q.detach().float()
        diff = ref - q
        self.signal_sq += ref.pow(2).sum().item()
        self.noise_sq += diff.pow(2).sum().item()
        self.dot += (ref * q).sum().item()
        self.quant_sq += q.pow(2).sum().item()
        self.max_abs_err = max(self.max_abs_err, diff.abs().max().item())
        self.n_elem += ref.numel()

    def summary(self):
        if self.n_elem == 0:
            return None
        mse = self.noise_sq / self.n_elem
        zero_output = self.signal_sq == 0 and self.quant_sq == 0
        denom = (self.signal_sq * self.quant_sq) ** 0.5
        if zero_output:
            cos_sim, sqnr_db = float("nan"), float("nan")
        else:
            cos_sim = max(-1.0, min(1.0, self.dot / denom)) if denom > 0 else 0.0
            sqnr_db = (10 * math.log10(self.signal_sq / self.noise_sq)
                       if self.noise_sq > 0 else float("inf"))
        return {"sqnr_db": sqnr_db, "mse": mse, "max_abs_err": self.max_abs_err,
                "cos_sim": cos_sim, "n_elem": self.n_elem}


class _TensorStats:
    """All per-tensor accumulators for one operand (weight or activation)."""

    def __init__(self):
        self.block = {k: _RunningStat() for k in
                      ("max_abs", "mean_abs", "variance", "dyn_range")}
        self.exp_counter = Counter()
        self.err = _ErrAccum()
        self.block_sqnr = _RunningStat()
        self.zero_cnt = 0
        self.total_cnt = 0
        self.uf_cnt = 0
        self.nz_cnt = 0
        self.n_blocks = 0
        self.raw_blocks = None  # filled only with detail=True (weights)

    def finalize(self, histograms=False):
        out = {"n_blocks": self.n_blocks, "block": {}}
        for k, rs in self.block.items():
            s = rs.summary()
            if s is not None and histograms:
                s = dict(s, hist=rs.hist())
            out["block"][k] = s
        out["zero_rate"] = self.zero_cnt / self.total_cnt if self.total_cnt else None
        out["underflow_rate"] = self.uf_cnt / self.nz_cnt if self.nz_cnt else None
        if self.exp_counter:
            exps = sorted(self.exp_counter)
            mode = max(self.exp_counter.items(), key=lambda kv: kv[1])[0]
            out["shared_exp"] = {"counts": {str(e): c for e, c in
                                            sorted(self.exp_counter.items())},
                                 "min": exps[0], "max": exps[-1], "mode": mode}
        else:
            out["shared_exp"] = None
        out["error"] = self.err.summary()
        bs = self.block_sqnr.summary()
        out["block_sqnr_db"] = (dict(bs, hist=self.block_sqnr.hist())
                                if bs is not None and histograms else bs)
        if self.raw_blocks is not None:
            out["raw_blocks"] = {k: v.tolist() for k, v in self.raw_blocks.items()}
        return out


def _tensor_block_stats(x_fp, q, axes, block_size, sink, detail=False):
    """Update `sink` (_TensorStats) with per-block stats of one (x_fp, q) pair.

    x_fp: pre-quant fp operand (post elemwise round — mirrors the forward).
    q:    fake-quant output of quantize_mx_op on x_fp.
    """
    xb, mask = _blockify(x_fp, axes, block_size)
    qb, _ = _blockify(q, axes, block_size)
    cnt = mask.sum(-1)
    valid = cnt > 0
    if not valid.all():
        xb, qb, mask, cnt = xb[valid], qb[valid], mask[valid], cnt[valid]
    cnt = cnt.float()

    absx = xb.abs()
    max_abs = absx.amax(-1)
    mean_abs = absx.sum(-1) / cnt
    mean = xb.sum(-1) / cnt
    var = ((xb * xb).sum(-1) / cnt - mean * mean).clamp_min(0)
    nonzero_blk = max_abs > 0
    dyn = max_abs[nonzero_blk] / mean_abs[nonzero_blk]

    sink.n_blocks += xb.shape[0]
    sink.block["max_abs"].update(max_abs)
    sink.block["mean_abs"].update(mean_abs)
    sink.block["variance"].update(var)
    sink.block["dyn_range"].update(dyn)

    # zero / underflow rates (pad elements excluded via mask)
    zq = (qb == 0) & mask
    nz = (xb != 0) & mask
    sink.zero_cnt += int(zq.sum())
    sink.total_cnt += int(mask.sum())
    sink.uf_cnt += int((zq & nz).sum())
    sink.nz_cnt += int(nz.sum())

    # shared exponents: floor(log2(block max_abs)), identical to
    # mx_ops._shared_exponents(method="max") which quantize_mx_op uses.
    E = torch.floor(torch.log2(max_abs[nonzero_blk])).long()
    sink.exp_counter.update(E.cpu().tolist())

    # tensor-level + per-block error
    sink.err.update(x_fp, q)
    diff = xb - qb
    sig_b = (xb * xb * mask).sum(-1)
    noise_b = (diff * diff * mask).sum(-1)
    ok = (sig_b > 0) & (noise_b > 0)
    if ok.any():
        sink.block_sqnr.update(10.0 * torch.log10(sig_b[ok] / noise_b[ok]))

    if detail and sink.raw_blocks is None:
        sink.raw_blocks = {"max_abs": max_abs.cpu(), "mean_abs": mean_abs.cpu(),
                           "variance": var.cpu(),
                           "block_sqnr_db": (10.0 * torch.log10(
                               sig_b.clamp_min(1e-45) / noise_b.clamp_min(1e-45))).cpu()}


def _quant_operand(x, sp, elem_format, axes, round_key):
    """Mirror the forward operand pipeline: elemwise round then MX block quant.

    Returns (bf, q): bf is the fp reference (so metrics isolate MX block
    quantization), q is the fake-quant FP32 tensor on the MX lattice.
    """
    bf = quantize_elemwise_op(x, mx_specs=sp, round=sp[round_key])
    q = quantize_mx_op(bf, sp, elem_format=elem_format, axes=axes,
                       round=sp.get("round_mx_output", "nearest"))
    return bf, q


def _weight_stats(module, detail=False):
    sp = module.mx_specs
    w_fmt = sp["w_elem_format"]
    if w_fmt is None:
        return None
    _, wt_axes = _layer_quant_axes(module)
    sink = _TensorStats()
    with torch.no_grad():
        w = module.weight.data.float()
        bf_w, q_w = _quant_operand(w, sp, w_fmt, wt_axes, "round_weight")
        _tensor_block_stats(bf_w, q_w, wt_axes, sp["block_size"], sink,
                            detail=detail)
    return sink


def _fp32_functional(mod, x):
    """FP32 reference output of `mod` on input x, using its stored weights."""
    w = mod.weight.to(x.dtype)
    b = mod.bias.to(x.dtype) if mod.bias is not None else None
    if isinstance(mod, MXConvTranspose2d):
        return F.conv_transpose2d(x, w, b, mod.stride, mod.padding,
                                  mod.output_padding, mod.groups, mod.dilation)
    if isinstance(mod, MXConv2d):
        return F.conv2d(x, w, b, mod.stride, mod.padding,
                        mod.dilation, mod.groups)
    if isinstance(mod, MXLinear):
        return F.linear(x, w, b)
    return None


def _make_stats_hook(name, state, output_error=True):
    def hook(mod, inp, out):
        with torch.no_grad():
            x = inp[0].detach().float()
            st = state[name]
            sp = mod.mx_specs
            a_fmt = sp["a_elem_format"]
            if a_fmt is not None:
                act_axes, _ = _layer_quant_axes(mod)
                bf, q = _quant_operand(x, sp, a_fmt, act_axes, "round_output")
                _tensor_block_stats(bf, q, act_axes, sp["block_size"], st["act"])
            if output_error:
                ref = _fp32_functional(mod, x)
                if ref is not None:
                    st["out_err"].update(ref, out.detach().float())
            st["n_calls"] += 1
    return hook


def _sort_key(entry):
    """Worst-first ordering: out-SQNR, fallback act, then weight; nan/missing last."""
    for section in ("output_error", "activation", "weight"):
        d = entry.get(section)
        if section == "output_error":
            d = (d or {}).get("isolated")
        else:
            d = (d or {}).get("error")
        if d and d.get("sqnr_db") is not None and not math.isnan(d["sqnr_db"]):
            return d["sqnr_db"]
    return float("inf")


def _fmt_db(v):
    if v is None:
        return "    -"
    if math.isnan(v):
        return "  nan"
    if math.isinf(v):
        return "  inf"
    return f"{v:5.1f}"


def _fmt_pct(v):
    return "    -" if v is None else f"{100 * v:4.1f}%"


def _exp_hist(counts, label, width=40):
    """ASCII histogram over a shared-exponent counts dict (mx_debug._int_hist style)."""
    if not counts:
        return f"  {label}: (empty)"
    items = sorted((int(e), c) for e, c in counts.items())
    n = sum(c for _, c in items)
    cmax = max(c for _, c in items) or 1
    lines = [f"  {label}  (n={n})"]
    for e, c in items:
        bar = "█" * round(width * c / cmax)
        lines.append(f"    e={e:+4d} {c:7d} {bar}")
    return "\n".join(lines)


def _print_table(stats, log=None):
    _log(log, "Quant stats per layer (worst first):")
    hdr = (f"  {'Layer':<40} {'Type':<18} {'Fmt(w/a)':<16} "
           f"{'W-SQNR':>7} {'A-SQNR':>7} {'Out-SQNR':>8} "
           f"{'W-ufl%':>7} {'A-ufl%':>7} {'A-dyn(p50)':>11} {'W-exp':>10}")
    _log(log, hdr)
    for name, e in stats["layers"].items():
        w, a = e.get("weight"), e.get("activation")
        w_err = (w or {}).get("error") or {}
        a_err = (a or {}).get("error") or {}
        o_err = (e.get("output_error") or {}).get("isolated") or {}
        a_dyn = (((a or {}).get("block") or {}).get("dyn_range") or {})
        w_exp = (w or {}).get("shared_exp")
        exp_str = f"[{w_exp['min']}..{w_exp['max']}]" if w_exp else "-"
        fmt = f"{e.get('w_elem_format')}/{e.get('a_elem_format')}"
        _log(log, f"  {name:<40} {e['layer_type']:<18} {fmt:<16} "
                  f"{_fmt_db(w_err.get('sqnr_db')):>7} "
                  f"{_fmt_db(a_err.get('sqnr_db')):>7} "
                  f"{_fmt_db(o_err.get('sqnr_db')):>8} "
                  f"{_fmt_pct((w or {}).get('underflow_rate')):>7} "
                  f"{_fmt_pct((a or {}).get('underflow_rate')):>7} "
                  f"{a_dyn.get('p50', float('nan')):>11.2f} {exp_str:>10}")
    net = stats["network"]
    _log(log, f"  Overall: out {_fmt_db(net['mean_out_sqnr_db'])} dB | "
              f"w {_fmt_db(net['mean_w_sqnr_db'])} dB | "
              f"a {_fmt_db(net['mean_a_sqnr_db'])} dB")


def _print_histograms(stats, state, log=None):
    for name, e in stats["layers"].items():
        _log(log, f"\n  --- {name} ({e['layer_type']}) ---")
        st = state.get(name, {})
        w_sink, a_sink = st.get("weight"), st.get("act")
        if w_sink is not None:
            _log(log, _ascii_hist(w_sink.block["dyn_range"].res,
                                  label="weight block dyn_range"))
            _log(log, _ascii_hist(w_sink.block_sqnr.res,
                                  label="weight block SQNR (dB)"))
            if e["weight"] and e["weight"]["shared_exp"]:
                _log(log, _exp_hist(e["weight"]["shared_exp"]["counts"],
                                    "weight shared exponents"))
        if a_sink is not None and a_sink.n_blocks > 0:
            _log(log, _ascii_hist(a_sink.block["dyn_range"].res,
                                  label="act block dyn_range"))
            _log(log, _ascii_hist(a_sink.block_sqnr.res,
                                  label="act block SQNR (dB)"))
            if e["activation"] and e["activation"]["shared_exp"]:
                _log(log, _exp_hist(e["activation"]["shared_exp"]["counts"],
                                    "act shared exponents"))


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return _json_sanitize(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def _dump_json(stats, path):
    with open(path, "w") as f:
        json.dump(_json_sanitize(stats), f, indent=2)


def collect_stats(model, data=None, forward_fn=None, *,
                  max_batches=32, histograms=False, detail=False,
                  output_error=True, save_path=None, log=None):
    """Collect per-block / per-layer quantization statistics on an MX model.

    Args:
        model: model already quantized by MXQuantizer.quant().
        data: optional iterable of calibration batches. Without it only
            static weight stats are computed.
        forward_fn: optional forward_fn(model, batch); defaults to the same
            dispatch as MXQuantizer (tuple->m(b[0]), dict->m(**b), tensor->m(b)).
        max_batches: cap on calibration batches (<=0 means all).
        histograms: print ASCII histograms and embed bin data in the result.
        detail: keep raw per-block vectors for weights (large!).
        output_error: compute isolated layer-output error (one extra fp32
            conv/linear per layer per batch).
        save_path: if set, dump the sanitized stats dict to this JSON path.
        log: optional logger; falls back to print.

    Returns:
        dict: {"meta", "layers" (worst-first), "network"} — see module docstring.
    """
    mx_layers = [(n, m) for n, m in model.named_modules()
                 if isinstance(m, _MX_TYPES)]
    if not mx_layers:
        _log(log, "collect_stats | WARNING: no MX layers found, skipping.")
        return {}

    state = {}
    for name, m in mx_layers:
        state[name] = {"weight": _weight_stats(m, detail=detail),
                       "act": _TensorStats(),
                       "out_err": _ErrAccum(),
                       "n_calls": 0}

    n_batches = 0
    if data is not None:
        handles = [m.register_forward_hook(
                       _make_stats_hook(n, state, output_error=output_error))
                   for n, m in mx_layers]

        def _fwd(m, batch):
            if forward_fn is not None:
                forward_fn(m, batch)
            elif isinstance(batch, (list, tuple)):
                m(batch[0])
            elif isinstance(batch, dict):
                m(**batch)
            else:
                m(batch)

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for i, batch in enumerate(data):
                    if max_batches > 0 and i >= max_batches:
                        break
                    _fwd(model, batch)
                    n_batches += 1
        finally:
            for h in handles:
                h.remove()
            if was_training:
                model.train()

    layers = {}
    for name, m in mx_layers:
        st = state[name]
        sp = m.mx_specs
        act_axes, wt_axes = _layer_quant_axes(m)
        entry = {
            "layer_type": _layer_type_name(m),
            "w_elem_format": sp["w_elem_format"],
            "a_elem_format": sp["a_elem_format"],
            "block_size": sp["block_size"],
            "act_axes": list(act_axes), "wt_axes": list(wt_axes),
        }
        if st["weight"] is not None:
            entry["weight"] = dict(st["weight"].finalize(histograms=histograms),
                                   shape=list(m.weight.shape),
                                   ptq_note="weights may already be on the MX "
                                            "lattice if PTQ ran (error ~0 is "
                                            "expected then)")
        else:
            entry["weight"] = None
        if st["act"].n_blocks > 0:
            entry["activation"] = dict(st["act"].finalize(histograms=histograms),
                                       n_calls=st["n_calls"])
        else:
            entry["activation"] = None
        entry["output_error"] = {"isolated": st["out_err"].summary(),
                                 "propagated": None}
        layers[name] = entry

    layers = dict(sorted(layers.items(), key=lambda kv: _sort_key(kv[1])))

    def _mean_db(vals):
        vals = [v for v in vals
                if v is not None and not math.isnan(v) and not math.isinf(v)]
        return sum(vals) / len(vals) if vals else None

    def _rate(num_attr, den_attr, section):
        num = sum(getattr(state[n][section], num_attr) for n in state
                  if state[n][section] is not None)
        den = sum(getattr(state[n][section], den_attr) for n in state
                  if state[n][section] is not None)
        return num / den if den else None

    network = {
        "mean_out_sqnr_db": _mean_db([(e["output_error"]["isolated"] or {}).get("sqnr_db")
                                      for e in layers.values()]),
        "mean_w_sqnr_db": _mean_db([((e["weight"] or {}).get("error") or {}).get("sqnr_db")
                                    for e in layers.values()]),
        "mean_a_sqnr_db": _mean_db([((e["activation"] or {}).get("error") or {}).get("sqnr_db")
                                    for e in layers.values()]),
        "worst_layers": list(layers.keys())[:5],
        "total_underflow_rate_w": _rate("uf_cnt", "nz_cnt", "weight"),
        "total_underflow_rate_a": _rate("uf_cnt", "nz_cnt", "act"),
    }

    stats = {
        "meta": {"n_batches": n_batches, "max_batches": max_batches,
                 "n_layers": len(layers), "torch_version": torch.__version__,
                 "detail": detail},
        "layers": layers,
        "network": network,
    }

    _print_table(stats, log)
    if histograms:
        _print_histograms(stats, state, log)
    if save_path is not None:
        _dump_json(stats, save_path)
        _log(log, f"collect_stats | saved to {save_path}")
    return stats
