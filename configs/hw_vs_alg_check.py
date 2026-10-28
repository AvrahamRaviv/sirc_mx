"""
hw_vs_alg_check.py - ALG (MXConv2dHW) vs HW (hw/npe_mx_emu): one test vector
per known arithmetic difference. Prints numbers; changes nothing.

    python hw_vs_alg_check.py

Rerun when HW sends new code: a section whose numbers change means that
difference moved. Sections:
  T1 activation quantizer (floor exp, truncate vs round, no pre-round)
  T7 their mx_dequantizer vs the value their pipe uses
  T3 HW accumulator exactness (int64 p_sum)
  T4 PCU rounding (round half up vs floor)
  T5 bias projection (half up vs half to even)
  T6 accumulator overflow (two's-complement wrap vs clamp-and-continue)
  T8 weight observer blocking (per_row_flatten) vs weight quantizer blocking
  T9 4-bit exponent wrap
  T10 4-bit exponent window: ALG fixed vs HW per-tensor offset
  T11 PLA (PReLU in the PCU), the PCU stage alone: the same accumulator word
      into ALG's PLA + window and into HW's pcu; (a) positive / identity,
      ReLU, (b) PReLU, (c) rounding and clamp
--config: default mx_config_npe_e4.json; mx_config_npe_hw_exact.json = lib HW-exact switches.
"""
import math, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import torch, torch.nn.functional as F
import cases as C, main_ALG as _A, main_HW as _H, sim_config as SC

# --config: default mx_config_npe_e4.json (NPE 4-bit exponent, ALG defaults);
# mx_config_npe_hw_exact.json = the lib's HW-exact switches M1-M8, with grids.
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--config", default="mx_config_npe_e4.json")
CFG = SC.load_config(_ap.parse_args().config)
print(f"config: {os.path.basename(CFG['file'])}")


class A:
    run = staticmethod(lambda case: _A.run(case, CFG))


class H:
    run = staticmethod(lambda case: _H.run(case, CFG))

from npe_mx_emu import base_mx_quantizer, mx_dequantizer, MXObserver, bcu, pcu_div
torch.set_printoptions(precision=6)

def hdr(s): print(f"\n### {s}")

# ---------------------------------------------------------------- T1 quantizer
hdr("T1 activation quantizer, conv3x3_p0 x")
c = C.build("conv3x3_p0"); x = c["x"]
a = A.run(c); h = H.run(c)
cal = int(h["x_cal"])
# ALG value
ebias = (1 << (int(CFG["mx_specs"]["scale_bits"]) - 1)) - 1      # code = E + ebias
qa = a["a_int"].double() * 2.0 ** (a["a_exp"].double().repeat_interleave(32, -1)[..., :40] - ebias - 6)
# HW value: x = data * 2^(shared - cal)
hd = h["hw_a_mant"].double().reshape(1, 4, 6, -1)[..., :40]
he = h["hw_a_exp"].double().repeat_interleave(32, -1)[..., :40]
qh = hd * 2.0 ** (he - cal)
# my reading of their rule (not an input layer: no pre-round):
# trunc(x 2^c / 2^(floor(log2 amax)-6))
xi = x.double() * 2 ** cal
am = F.pad(xi, (0, 24)).reshape(1, 4, 6, 2, 32).abs().amax(-1)
E = torch.where(am > 0, torch.log2(am.clamp(min=1e-30)).floor(), torch.zeros_like(am))
mine = torch.trunc(F.pad(xi, (0, 24)).reshape(1, 4, 6, 2, 32) / 2 ** (E - 6).unsqueeze(-1)).reshape(1, 4, 6, 64)[..., :40]
print("my model of HW rule == their data:", bool((mine == hd).all()))
print(f"max|mant|  ALG {int(a['a_int'].abs().max())}   HW {int(hd.abs().max())}")
print(f"quant err rms  ALG {float(((qa - x) ** 2).mean().sqrt()):.5f}   HW {float(((qh - x) ** 2).mean().sqrt()):.5f}")
print(f"same value: {float((qa == qh).double().mean()) * 100:.1f}% of elements")
print(f"HW bias (mean err, trunc->toward 0 of |x|): {float(((qh.abs() - x.abs())).mean()):.5f}  ALG {float(((qa.abs() - x.abs())).mean()):.5f}")

# ---------------------------------------------------------------- T7 dequantizer
hdr("T7 their mx_dequantizer vs the value their pipe uses")
mx = base_mx_quantizer(x * 2 ** cal, calibrate_offset=cal)
dq = mx_dequantizer(mx)
print(f"mx_dequantizer / true value ratio: {float((dq[x.abs() > 0.1] / qh.float()[x.abs() > 0.1]).median()):.6f}  (1.0 = consistent; 2^-6 = {2**-6:.6f})")

# ---------------------------------------------------------------- T3 acc exactness
def exact_acc(h, w_shape, x_shape):
    B, Cin, Hh, W = x_shape; O, _, kh, kw = w_shape
    ad = h["hw_a_mant"].long().reshape(B, Cin, Hh, -1)[..., :W]
    ae = h["hw_a_exp"].long().repeat_interleave(32, -1)[..., :W]
    wd = h["hw_w_mant"].long().reshape(O, -1)[:, :Cin * kh * kw]
    we = h["hw_w_exp"].long().repeat_interleave(32, -1)[:, :Cin * kh * kw]
    A_ = F.unfold((ad.double()), (kh, kw)).long(); AE = F.unfold(ae.double(), (kh, kw)).long()
    acc = torch.zeros(B, O, A_.shape[-1], dtype=torch.int64)
    for o in range(O):
        acc[:, o] = ((A_ * wd[o].view(1, -1, 1)) << (AE + we[o].view(1, -1, 1))).sum(1)
    return acc

hdr("T3 HW accumulator exactness (int64 p_sum)")
for name in ["conv1x1", "conv3x3_p0"]:
    cc = C.build(name); hh = H.run(cc)
    ex = exact_acc(hh, cc["w"].shape, cc["x"].shape).reshape(hh["hw_acc"].shape)
    print(f"{name}: max|acc| {int(ex.abs().max())} (2^{math.log2(ex.abs().max()):.1f})  hw_acc==exact int: {bool((hh['hw_acc'].double() == ex.double()).all())}")
# worst case on their bcu directly: a big and a tiny product in one dot
# product (exps legal 0..15, sum fits 48 bits) - float32 dropped the low bits
xd = torch.tensor([[63.0, 1.0]]); xe = torch.tensor([[10.0, 0.0]])
wd = torch.tensor([63.0, 1.0]); we = torch.tensor([10.0, 0.0])
exact = 63 * 63 * 2 ** 20 + 1
got = int(bcu(xd, xe, wd, we).item())
print(f"bcu 63*63*2^20 + 1: exact {exact}  HW {got}  diff {exact - got}")

# ---------------------------------------------------------------- T4 PCU rounding
hdr("T4 PCU rounding: HW round half up (ties to +inf) vs ALG window")
v = torch.tensor([10, 11, 13, -6, -10, -11])   # /4 -> 2.5 2.75 3.25 -1.5 -2.5 -2.75
print("in            ", v.tolist(), "shift -2 (x/4)")
print("pcu_div round ", pcu_div(v, -2, "round").tolist())
print("half up       ", (v / 4 + 0.5).floor().long().tolist())
print("ALG floor     ", (v >> 2).tolist(), "  ALG nearest", ((v + 2) >> 2).tolist(),
      "(xblock_accum.out_round)")
cc = C.build("conv3x3_p0"); hh = H.run(cc)
e = int(hh["out_scale_log2"]) - int(hh["x_cal"]) - int(hh["w_cal"])   # incl. static shift
print(f"conv3x3_p0: out_raw == floor(acc*2^{e} + 1/2): "
      f"{bool((hh['out_raw'] == (hh['hw_acc'].double() * 2.0 ** e + 0.5).floor()).all())}")

# ---------------------------------------------------------------- T5 bias
hdr("T5 bias projection")
cc = C.build("conv3x3_p0_bias")
hh = H.run(cc); aa = A.run(cc)
bc = int(hh["x_cal"]) + int(hh["w_cal"])
b = cc["b"].double()
print(f"HW bias = floor(b*2^{bc} + 1/2) (acc LSB 2^-{bc}), err max {float((b - (b * 2 ** bc + 0.5).floor() / 2 ** bc).abs().max()):.2e}")
ea = -int(aa.get("e_layer_min", -20))
half_up = CFG["mx_specs"].get("xblock_accum", {}).get("bias_round", "even") == "half_up"
rb = (lambda v: (v + 0.5).floor()) if half_up else (lambda v: v.round())
print(f"ALG bias = {'floor(v+1/2)' if half_up else 'round half to even'} of b*2^{ea},  "
      f"err max {float((b - rb(b * 2 ** ea) / 2 ** ea).abs().max()):.2e}")
t = torch.tensor([0.5, 1.5, 2.5, -0.5, -2.5], dtype=torch.float64)
print(f"ties b*2^k {t.tolist()}: HW {(t + 0.5).floor().long().tolist()}  ALG {rb(t).long().tolist()}")
print(f"with bias end-to-end: max|ALG-fp32| {float((aa['out'] - F.conv2d(cc['x'], cc['w'], cc['b'])).abs().max()):.4f}  max|HW-fp32| {float((hh['out'] - F.conv2d(cc['x'], cc['w'], cc['b'])).abs().max()):.4f}")

# ---------------------------------------------------------------- T6 saturation
hdr("T6 accumulator overflow (bcu, acc_bit_width=48 -> range [-2^47, 2^47-1])")
lo, hi = -2 ** 47, 2 ** 47 - 1
for signs in ([1, 1, -1], [1, 1, 1]):
    xd = torch.tensor([signs]); xe = torch.full((1, 3), 46)   # each product = +-2^46
    wd = torch.ones(3, dtype=torch.long); we = torch.zeros(3, dtype=torch.long)
    acc = 0
    for s_ in signs:
        acc = max(lo, min(hi, acc + s_ * 2 ** 46))
    print(f"products {['%+d*2^46' % s_ for s_ in signs]}: true {sum(signs) * 2 ** 46}  "
          f"HW wrap {int(bcu(xd, xe, wd, we, acc_bit_width=48).item())}  ALG clamp {acc}"
          + ("  (ALG overflow 'raise': eval raises here)" if CFG["mx_specs"].get("xblock_accum", {}).get("overflow") == "raise" else ""))

# ---------------------------------------------------------------- T8 observer blocking
hdr("T8 weight observer blocks vs weight quantizer blocks")
w = C.build("conv3x3_p0")["w"]
o_theirs = MXObserver(mode="mx"); o_theirs.update(w)
o_flat = MXObserver(mode="mx"); o_flat.update(w, per_row_flatten=True)
print(f"observer default (blocks along kW=3): w_cal {o_theirs.calibrate_offset()}   per_row_flatten=True (= quantizer blocks, main_HW uses it): {o_flat.calibrate_offset()}")
mxw = H.run(C.build("conv3x3_p0"))["hw_w_exp"]
print(f"resulting weight block exps used: {mxw.unique().tolist()}  (min 0 = anchored; >0 = headroom spent)")

# ---------------------------------------------------------------- T9 exponent wrap
hdr("T9 4-bit exponent wrap (block below the anchor)")
t = torch.zeros(1, 64); t[0, :32] = 64.0; t[0, 32:] = 3.0   # block2 E=2 -> 2-6=-4 -> &0xF=12
m = base_mx_quantizer(t, calibrate_offset=0)
print("block shared exps:", m.shared_exp.tolist(), " block2 data:", m.data[0, 1, :4].tolist(), "(true 3.0 -> wiped)")

# ---------------------------------------------------------------- T10 exponent window
hdr("T10 4-bit exponent window: ALG fixed [-7,7] vs HW per-tensor offset (weights x0.01)")
c = C.build("conv3x3_p0"); c["w"] = c["w"] * 0.01
r = F.conv2d(c["x"], c["w"])
try:
    a = A.run(c)
except RuntimeError as e:      # HW-exact: weights x0.01 leave the config's register
    print(f"ALG raises (grids in the config are for the unscaled weights): {str(e)[:150]}")
    a = None
h = H.run(c)
if a is not None:
  print(f"ALG: w_exp codes [{int(a['w_exp'].min())}..{int(a['w_exp'].max())}] (0 = clamped at E=-7)  "
        f"w_int max {int(a['w_int'].abs().max())}  max err {float((a['out'] - r).abs().max()):.2e}")
print(f"HW : w_cal {int(h['w_cal'])} slides the window  "
      f"w_mant max {int(h['hw_w_mant'].abs().max())}  max err {float((h['out'] - r).abs().max()):.2e}")

# ---------------------------------------------------------------- T11 PLA
# HW PLA = per-channel bins on the accumulator word (their answer to Q10):
# out = S_k * acc + B_k -> pcu_div (round half up) -> relux. Their emulator is
# still ONE bin; main_HW.bins_pcu is our stand-in (their pcu per bin).
# Same int64 accumulator word N into both PCUs, so only the PCU differs:
# ALG = the lib's pla_bins + _apply_pla + _apply_out_window.
# Grids aligned: acc LSB 2^-16 (x_cal + w_cal = 16), output 2^-12 on both.
from fixed_point.mx_fixed_point_hw import (PLAParams, pla_bins, _apply_pla,
                                           _apply_out_window)
from npe_mx_emu import Cstruct, pcu

BCAL, EOUT, WIN, FRAC = 16, -12, 22, 6
E_HW = 12 - BCAL                         # their e_min_layer sum


def bins_of(b, m_neg, m_pos=64, tie="upper"):
    bd, sc, bi = pla_bins(torch.tensor([m_pos]), torch.tensor([m_neg]),
                          torch.tensor([b], dtype=torch.float64), -BCAL, FRAC)
    return PLAParams(torch.tensor([m_pos]), torch.tensor([m_neg]), FRAC, "bins",
                     bd, sc, bi, tie)


def alg_bins(N, pla):
    w = _apply_pla(N.view(1, 1, -1), pla)
    o, sat = _apply_out_window(w, -BCAL - FRAC, EOUT, WIN, "nearest")
    return o.view(-1)


def hw_bins(N, pla):
    d = {"bounds": pla.bounds.tolist(), "scales": pla.scales.tolist(),
         "biases": pla.biases.tolist(), "tie": pla.tie, "frac": FRAC,
         "lo": -2.0 ** 23, "hi": 2.0 ** 23 - 1}
    return _H.bins_pcu(N, d, 0, E_HW)


def hw_one_bin(N, pla):
    cs = Cstruct(bias=int(pla.biases[0, -1]), scale=int(pla.scales[0, -1]), shift=-FRAC,
                 round_type="round", min_val=-2.0 ** 23, max_val=2.0 ** 23 - 1)
    return pcu(N, c_struct=cs, e_min_layer=E_HW).long()


hdr("T11 PLA bins: the PCU stage alone (same acc word, acc LSB 2^-16, out 2^-12)")
g = torch.Generator().manual_seed(11)
N = torch.randint(-2 ** 20, 2 ** 20, (20000,), generator=g, dtype=torch.int64)
print("N: 20000 random in [-2^20, 2^20)")
for b in (0.0371, -0.0371):
    for name, m_neg in (("identity", 64), ("ReLU", 0), ("PReLU 0.25", 16), ("PReLU 0.05", 3)):
        pla = bins_of(b, m_neg)
        a_, (h_, k) = alg_bins(N, pla), hw_bins(N, pla)
        one = hw_one_bin(N, pla)
        low = k == 0
        print(f"  bias {b:+.4f} {name:10s} bound {int(pla.bounds[0, 0]):6d} S {pla.scales[0].tolist()} "
              f"B {pla.biases[0].tolist()}: lib vs stand-in {int((a_ != h_).sum())} differ | "
              f"vs their 1 bin {int((a_ != one).sum())} differ "
              f"({int((a_ != one)[~low].sum())} in the upper bin)")

print("(tie) acc == boundary, both tie rules - on the fine grid 2^-(16+6) (no output")
print("      shift), where the two bins give different words; on 2^-12 both round to 0")
def alg_fine(N, pla):
    w = _apply_pla(N.view(1, 1, -1), pla)
    return _apply_out_window(w, -BCAL - FRAC, -BCAL - FRAC, WIN, "nearest")[0].view(-1)
def hw_fine(N, pla):
    d = {"bounds": pla.bounds.tolist(), "scales": pla.scales.tolist(),
         "biases": pla.biases.tolist(), "tie": pla.tie, "frac": FRAC,
         "lo": -2.0 ** 23, "hi": 2.0 ** 23 - 1}
    return _H.bins_pcu(N, d, 0, FRAC)      # their shift: -FRAC + FRAC = 0
for b in (0.0371, -0.0371):
    for tie in ("upper", "lower"):
        pla = bins_of(b, 16, tie=tie)
        t = int(pla.bounds[0, 0])
        Nt = torch.tensor([t - 1, t, t + 1], dtype=torch.int64)
        a_, (h_, k) = alg_fine(Nt, pla), hw_fine(Nt, pla)
        print(f"  bias {b:+.4f} tie {tie:5s}: N = bound-1, bound, bound+1 -> bins {k.tolist()}  "
              f"lib {a_.tolist()}  stand-in {h_.tolist()}  "
              f"{'equal' if bool((a_ == h_).all()) else 'DIFFER'}")
print("(c) rounding ties and clamp (identity bins, bias 0)")
pla = bins_of(0.0, 64)
P = torch.tensor([2, 6, 10, -2, -6, -10, 3, -3], dtype=torch.int64) * 2 ** (BCAL - 12 - 2)
print(f"  x/4 of [2,6,10,-2,-6,-10,3,-3]: lib {alg_bins(P, pla).tolist()}  "
      f"stand-in {hw_bins(P, pla)[0].tolist()}")
big = torch.tensor([2 ** 22 - 1, 2 ** 22, 5_000_000, 2 ** 23, -2 ** 22, -2 ** 23 - 5],
                   dtype=torch.int64)
Nb = big * 2 ** (BCAL - 12)
print(f"  out (LSB)    {big.tolist()}")
print(f"  lib window   {alg_bins(Nb, pla).tolist()}  (+-(2^22-1))")
print(f"  their relux  {hw_bins(Nb, pla)[0].tolist()}  ([-2^23, 2^23-1]; HW: >= 2^22 is "
      f"unreachable by PTQ, no clamp needed)")
