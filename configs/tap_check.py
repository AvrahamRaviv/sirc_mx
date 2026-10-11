"""
tap_check.py - the ALG side's own consistency check, written from the spec.

Two things, both plain torch with no MX lib import:

  * `mx_decode` turns a fake-quant FP tensor (already on the MXINT lattice) into
    the integer form HW stores: intN mantissas plus one biased shared-exponent
    code per block. It asserts the tensor really is on the lattice, so a tap
    that does not describe what the layer used fails here instead of in a
    comparison with HW.
  * `ref_conv` rebuilds a layer's accumulator word from those integer taps:
    per-product multiply, signed shift, saturating add, bias, in the order the
    contract in `main_HW.py` specifies.

`main_ALG.run` calls both on every MXConv2dHW layer and requires the rebuilt
accumulator to equal the one MXConv2dHW computed, bit for bit. If it does not,
the taps we would hand to arch do not explain our own output, and every
comparison against them would be chasing a ghost.

This is NOT the HW side. main_HW is written independently by arch from the
spec; this file only proves our taps are self-consistent.

Conventions (the contract):

  value      = mantissa * 2**(E - mant_bias),  mant_bias = mbits - 2
  E          = floor(log2(max |x| over the block)), clamped to
               [-emax, emax], emax = 2**(scale_bits - 1) - 1
  code       = E + emax                 unsigned, 0 .. 2**scale_bits - 2
  zero block = code 0 (its exponent is undefined; mantissas are all 0)
  mantissa   = round(x * 2**(mant_bias - E)) in [-(2**(mbits-1) - 1), +...]
"""

import torch
import torch.nn.functional as F


def fmt_bits(fmt):
    """'int8' -> 8. Only MXINT formats have an integer form."""
    if not (isinstance(fmt, str) and fmt.startswith("int")):
        raise ValueError(f"no integer form for element format {fmt!r}")
    return int(fmt[3:])


def scale_emax(scale_bits):
    return (1 << (int(scale_bits) - 1)) - 1


# =============================================================================
# Decode
# =============================================================================

def mx_decode(q_fp, axis, bs, mbits, scale_bits):
    """FP tensor on the MXINT lattice -> (mantissas, per-block exponent codes).

    Blocks run along `axis` in steps of `bs`; a tail shorter than `bs` is its
    own block (zero-padding it does not move a max-abs exponent).

    Returns:
        q    : same shape as `q_fp`, int8 (mbits <= 8) or int16
        code : `q_fp` shape with `axis` reduced to ceil(n / bs), int16
    """
    axis = axis % q_fp.dim()
    x = q_fp.detach().to(torch.float64).movedim(axis, -1)
    n = x.shape[-1]
    pad = (-n) % bs
    if pad:
        x = F.pad(x, (0, pad))
    xb = x.reshape(*x.shape[:-1], -1, bs)

    emax = scale_emax(scale_bits)
    mant_bias = mbits - 2
    max_val = (1 << (mbits - 1)) - 1

    m = xb.abs().amax(-1, keepdim=True)
    nz = m > 0
    E = torch.floor(torch.log2(torch.where(nz, m, torch.ones_like(m))))
    if bool((E[nz] > emax).any()):
        raise ValueError(f"shared exponent above +{emax}: the lib makes that "
                         f"NaN, there is no integer form")
    E = E.clamp(min=-emax)
    E = torch.where(nz, E, torch.full_like(E, -emax))

    qf = xb * torch.pow(2.0, mant_bias - E)
    q = torch.round(qf)
    if not bool((q == qf).all()):
        bad = int((q != qf).sum())
        raise AssertionError(f"mx_decode: {bad} elements are not on the "
                             f"MXINT{mbits} lattice - the tap does not "
                             f"describe a quantized tensor")
    if bool((q.abs() > max_val).any()):
        raise AssertionError(f"mx_decode: mantissa beyond +-{max_val}")

    dt = torch.int8 if mbits <= 8 else torch.int16
    q = q.reshape(*x.shape)[..., :n].movedim(-1, axis).to(dt).contiguous()
    code = (E.squeeze(-1) + emax).to(torch.int16).movedim(-1, axis).contiguous()
    return q, code


def expand_exp(code, axis, bs, n, scale_bits):
    """Per-block codes -> per-element real exponent E along `axis` (int64)."""
    E = code.long() - scale_emax(scale_bits)
    return E.repeat_interleave(bs, dim=axis).narrow(axis, 0, n)


# =============================================================================
# Reference conv
# =============================================================================

def ref_conv(a_int, a_code, w_int, w_code, bias, *, stride, padding, dilation,
             e_layer_min, bits, sat_mode, bs, mbits, scale_bits, blockify,
             bias_round="half_even", rshift="floor"):
    """Rebuild the accumulator word of one HW conv from its integer taps.

    Accumulation order (it matters once `per_product` saturates):
      npe      for each input channel c, for each kernel tap (kh, kw) - i.e.
               along the flattened filter stream, C-major.
      channel  for each channel block, for each kernel tap, for each lane.
    `per_block` (channel only) clamps after each (block, tap) group of lanes.

    Bias is projected onto the accumulator grid, round(b * 2**-e_layer_min),
    added after the last product, then clamped once more.

    `bias_round` and `rshift` default to what MXConv2dHW does (half-to-even,
    arithmetic shift = floor). The alternatives exist to show what a HW that
    chose differently would produce: rshift='trunc' rounds toward zero,
    bias_round='half_away' rounds ties away from zero.

    Returns (acc int64 [B,O,Ho,Wo], sat bool [B,O,Ho,Wo]).
    """
    B, C, H, W = a_int.shape
    O, _, kH, kW = w_int.shape
    kK = kH * kW
    if blockify == "npe":
        Ea = expand_exp(a_code, 3, bs, W, scale_bits)
        Ew = expand_exp(w_code, 1, bs, C * kK, scale_bits).reshape(O, C, kH, kW)
    else:
        Ea = expand_exp(a_code, 1, bs, C, scale_bits)
        Ew = expand_exp(w_code, 1, bs, C, scale_bits)
        Ew = Ew.reshape(O, C, kH, kW) if Ew.dim() == 4 else Ew

    kw = dict(kernel_size=(kH, kW), dilation=dilation, padding=padding,
              stride=stride)
    # unfold pads with 0: a pad position has mantissa 0, so its product is 0
    # whatever exponent sits next to it.
    ai = F.unfold(a_int.double(), **kw).long()          # [B, C*kK, L]
    ae = F.unfold(Ea.double(), **kw).long()
    wi = w_int.reshape(O, -1).long()                    # [O, C*kK]
    we = Ew.reshape(O, -1).long()
    L = ai.shape[-1]

    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    two_bias = 2 * (mbits - 2)
    acc = torch.zeros(B, O, L, dtype=torch.long)
    sat = torch.zeros(B, O, L, dtype=torch.bool)

    def clamp_(a):
        nonlocal sat
        sat |= (a < lo) | (a > hi)
        return a.clamp(lo, hi)

    if blockify == "npe":
        groups = [[c * kK + k] for c in range(C) for k in range(kK)]
        if sat_mode != "per_product":
            raise ValueError("npe blockify supports sat_mode='per_product' only")
    else:
        nb = (C + bs - 1) // bs
        groups = [[c * kK + k for c in range(b * bs, min((b + 1) * bs, C))]
                  for b in range(nb) for k in range(kK)]

    for g in groups:
        part = None
        for idx in g:
            p = ai[:, idx, :].unsqueeze(1) * wi[:, idx].view(1, O, 1)
            sh = ae[:, idx, :].unsqueeze(1) + we[:, idx].view(1, O, 1) \
                - two_bias - e_layer_min
            up = sh.clamp(min=0)
            if bool(((p != 0) & (up > 62 - 2 * (mbits - 1))).any()):
                raise OverflowError("ref_conv: left shift would overflow int64")
            # Arithmetic right shift rounds toward -inf (see the open
            # question in main_HW.py); 'trunc' shifts |p| and restores the sign.
            dn = (-sh).clamp(min=0)
            if rshift == "floor":
                r = p >> dn
            elif rshift == "trunc":
                r = torch.sign(p) * (p.abs() >> dn)
            else:
                raise ValueError(f"rshift {rshift!r}")
            s = torch.where(sh >= 0, p << up, r)
            if sat_mode == "per_product":
                acc = clamp_(acc + s)
            else:
                part = s if part is None else part + s
        if part is not None:
            acc = clamp_(acc + part)

    if bias is not None:
        b = bias.detach().double() * 2.0 ** -e_layer_min
        if bias_round == "half_even":
            bi = torch.round(b)
        elif bias_round == "half_away":
            bi = torch.sign(b) * torch.floor(b.abs() + 0.5)
        else:
            raise ValueError(f"bias_round {bias_round!r}")
        acc = clamp_(acc + bi.long().view(1, O, 1))

    sH, sW = (stride, stride) if isinstance(stride, int) else stride
    pH, pW = (padding, padding) if isinstance(padding, int) else padding
    dH, dW = (dilation, dilation) if isinstance(dilation, int) else dilation
    Ho = (H + 2 * pH - dH * (kH - 1) - 1) // sH + 1
    Wo = (W + 2 * pW - dW * (kW - 1) - 1) // sW + 1
    return acc.view(B, O, Ho, Wo), sat.view(B, O, Ho, Wo)


def check_layer(name, taps, mod, *, e_layer_min, bits, sat_mode, bs, mbits,
                scale_bits, blockify):
    """Assert one layer's integer taps rebuild its own accumulator and output.

    Raises AssertionError naming the first disagreeing stage. Returns a short
    dict for the run summary.
    """
    acc, sat = ref_conv(
        taps[f"{name}/a_int"], taps[f"{name}/a_exp"],
        taps[f"{name}/w_int"], taps[f"{name}/w_exp"],
        mod.bias, stride=tuple(mod.stride), padding=tuple(mod.padding),
        dilation=tuple(mod.dilation), e_layer_min=e_layer_min, bits=bits,
        sat_mode=sat_mode, bs=bs, mbits=mbits, scale_bits=scale_bits,
        blockify=blockify)
    got_acc, got_sat = taps[f"{name}/acc"], taps[f"{name}/sat"].bool()
    if acc.shape != got_acc.shape or not torch.equal(acc, got_acc):
        n = int((acc != got_acc).sum()) if acc.shape == got_acc.shape else -1
        raise AssertionError(f"tap_check {name}: acc rebuilt from the taps "
                             f"differs from MXConv2dHW in {n} elements")
    if not torch.equal(sat, got_sat):
        raise AssertionError(f"tap_check {name}: sat map differs in "
                             f"{int((sat != got_sat).sum())} elements")
    out = taps[f"{name}/out"]
    want = (acc.to(torch.float32) * float(2.0 ** e_layer_min))
    if not torch.equal(want, out):
        raise AssertionError(f"tap_check {name}: /out is not fp32(acc) * "
                             f"2^{e_layer_min} in {int((want != out).sum())} "
                             f"elements")
    lossy = int((acc.to(torch.float32).double() != acc.double()).sum())
    return {"acc_fp32_lossy": lossy, "sat": int(sat.sum())}
