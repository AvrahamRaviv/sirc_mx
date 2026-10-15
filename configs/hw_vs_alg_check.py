"""
hw_vs_alg_check.py - ALG (MXConv2dHW) vs HW (hw/npe_mx_emu): one test vector
per known arithmetic difference. Prints numbers; changes nothing.

    python hw_vs_alg_check.py

Rerun when HW sends new code: a section whose numbers change means that
difference moved. Sections:
  T1 activation quantizer (exp ceil vs floor, truncate vs round, pre-round)
  T7 their mx_dequantizer vs the value their pipe uses
  T3 HW accumulator exactness (float32 p_sum)
  T4 PCU rounding (floor before round)
  T5 bias projection
  T6 accumulator saturation (sticky vs clamp-and-continue)
  T8 weight observer blocking vs weight quantizer blocking
  T9 4-bit exponent wrap
"""
import math, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import torch, torch.nn.functional as F
import cases as C, main_ALG as A, main_HW as H
from npe_mx_emu import base_mx_quantizer, mx_dequantizer, MXObserver, bcu, pcu_div
torch.set_printoptions(precision=6)

def hdr(s): print(f"\n### {s}")

# ---------------------------------------------------------------- T1 quantizer
hdr("T1 activation quantizer, conv3x3_p0 x")
c = C.build("conv3x3_p0"); x = c["x"]
a = A.run(c); h = H.run(c)
cal = int(h["x_cal"])
# ALG value
qa = a["a_int"].double() * 2.0 ** (a["a_exp"].double().repeat_interleave(32, -1)[..., :40] - 127 - 6)
# HW value: x = data * 2^(shared - cal)
hd = h["hw_a_mant"].double().reshape(1, 4, 6, -1)[..., :40]
he = h["hw_a_exp"].double().repeat_interleave(32, -1)[..., :40]
qh = hd * 2.0 ** (he - cal)
# my reading of their rule: trunc(round(x 2^c) / 2^(ceil(log2 amax)-6))
xi = (x.double() * 2 ** cal).round()
am = F.pad(xi, (0, 24)).reshape(1, 4, 6, 2, 32).abs().amax(-1)
E = torch.where(am > 0, torch.log2(am.clamp(min=1e-30)).ceil(), torch.zeros_like(am))
mine = torch.trunc(F.pad(xi, (0, 24)).reshape(1, 4, 6, 2, 32) / 2 ** (E - 6).unsqueeze(-1)).reshape(1, 4, 6, 64)[..., :40]
print("my model of HW rule == their data:", bool((mine == hd).all()))
print(f"max|mant|  ALG {int(a['a_int'].abs().max())}   HW {int(hd.abs().max())}")
print(f"quant err rms  ALG {float(((qa - x) ** 2).mean().sqrt()):.5f}   HW {float(((qh - x) ** 2).mean().sqrt()):.5f}")
print(f"same value: {float((qa == qh).double().mean()) * 100:.1f}% of elements")
print(f"HW bias (mean err, trunc->toward 0 of |x|): {float(((qh.abs() - x.abs())).mean()):.5f}  ALG {float(((qa.abs() - x.abs())).mean()):.5f}")

# ---------------------------------------------------------------- T7 dequantizer
hdr("T7 their mx_dequantizer vs the value their pipe uses")
mx = base_mx_quantizer((x * 2 ** cal).round(), calibrate_offset=cal)
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

hdr("T3 HW accumulator exactness (float32 p_sum)")
for name in ["conv1x1", "conv3x3_p0"]:
    cc = C.build(name); hh = H.run(cc)
    ex = exact_acc(hh, cc["w"].shape, cc["x"].shape).reshape(hh["hw_acc"].shape)
    print(f"{name}: max|acc| {int(ex.abs().max())} (2^{math.log2(ex.abs().max()):.1f})  hw_acc==exact int: {bool((hh['hw_acc'].double() == ex.double()).all())}")
# worst case on their bcu directly: a big and a tiny product in one dot
# product (exps legal 0..15, sum fits 46 bits) - float32 drops the low bits
xd = torch.tensor([[63.0, 1.0]]); xe = torch.tensor([[10.0, 0.0]])
wd = torch.tensor([63.0, 1.0]); we = torch.tensor([10.0, 0.0])
exact = 63 * 63 * 2 ** 20 + 1
got = int(bcu(xd, xe, wd, we).item())
print(f"bcu 63*63*2^20 + 1: exact {exact}  HW {got}  diff {exact - got}")

# ---------------------------------------------------------------- T4 PCU rounding
hdr("T4 PCU: floor then round")
v = torch.tensor([10.0, 11.0, 13.0, -11.0, -13.0])   # /4 -> 2.5 2.75 3.25 -2.75 -3.25
print("in            ", v.tolist(), "shift -2 (x/4)")
print("pcu_div round ", pcu_div(v, -2, "round").tolist())
print("true round    ", (v / 4).round().tolist())
cc = C.build("conv3x3_p0"); hh = H.run(cc)
e = 12 - int(hh["x_cal"]) - int(hh["w_cal"])
print(f"conv3x3_p0: out_raw == floor(acc*2^{e}): {bool((hh['out_raw'] == (hh['hw_acc'] * 2.0 ** e).floor()).all())}")

# ---------------------------------------------------------------- T5 bias
hdr("T5 bias projection")
cc = C.build("conv3x3_bias"); cc["params"]["padding"] = 0
hh = H.run(cc); aa = A.run(cc)
bc = int(hh["x_cal"]) + int(hh["w_cal"])
b = cc["b"].double()
print(f"HW bias = floor(b*2^{bc}) (acc LSB 2^-{bc}), err max {float((b - (b * 2 ** bc).floor() / 2 ** bc).abs().max()):.2e}")
print(f"ALG bias = round(b*2^20),  err max {float((b - (b * 2 ** 20).round() / 2 ** 20).abs().max()):.2e}")
print(f"with bias end-to-end: max|ALG-fp32| {float((aa['out'] - F.conv2d(cc['x'], cc['w'], cc['b'])).abs().max()):.4f}  max|HW-fp32| {float((hh['out'] - F.conv2d(cc['x'], cc['w'], cc['b'])).abs().max()):.4f}")

# ---------------------------------------------------------------- T6 saturation
hdr("T6 accumulator saturation (bcu, acc_bit_width=8 -> range [-128,127])")
xd = torch.tensor([[100.0, 100.0, -100.0]]); z = torch.zeros(1, 3)
wd = torch.tensor([1.0, 1.0, 1.0]); wz = torch.zeros(3)
print("products +100 +100 -100")
print("HW bcu:", bcu(xd, z, wd, wz, acc_bit_width=8).tolist(), " (sticky: once at limit, later products dropped)")
acc = 0
for p in [100, 100, -100]:
    acc = max(-128, min(127, acc + p))
print("ALG per_product clamp-and-continue:", acc)
print("HW pos limit check uses >= 2^(n-1)-1 -> a sum landing exactly on 127 is also frozen:",
      bcu(torch.tensor([[127.0, -5.0]]), torch.zeros(1, 2), torch.tensor([1.0, 1.0]), torch.zeros(2), acc_bit_width=8).tolist())

# ---------------------------------------------------------------- T8 observer blocking
hdr("T8 weight observer blocks vs weight quantizer blocks")
w = C.build("conv3x3_p0")["w"]
o_theirs = MXObserver(mode="mx"); o_theirs.update(w)
o_flat = MXObserver(mode="mx"); o_flat.update(w.reshape(w.shape[0], -1))
print(f"observer on w as given (blocks along kW=3): w_cal {o_theirs.calibrate_offset()}   on flattened filter (= quantizer blocks): {o_flat.calibrate_offset()}")
mxw = H.run(C.build("conv3x3_p0"))["hw_w_exp"]
print(f"resulting weight block exps used: {mxw.unique().tolist()}  (min 0 = anchored; >0 = headroom spent)")

# ---------------------------------------------------------------- T9 exponent wrap
hdr("T9 4-bit exponent wrap (block below the anchor)")
t = torch.zeros(1, 64); t[0, :32] = 64.0; t[0, 32:] = 3.0   # block2 E=2 -> 2-6=-4 -> &0xF=12
m = base_mx_quantizer(t, calibrate_offset=0)
print("block shared exps:", m.shared_exp.tolist(), " block2 data:", m.data[0, 1, :4].tolist(), "(true 3.0 -> wiped)")
