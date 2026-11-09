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
  T12 warp (main_ALG.warp vs main_HW.warp on their CalibratedMXWarp): one
      vector per ALG-vs-HW difference of the warp deep run, each labelled
      [D<n> | HW Q<m>] with its question in the list sent to the HW team
--config: default mx_config_npe_e4.json; mx_config_npe_hw_exact.json = lib HW-exact switches.
--only warp: T12 alone.
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
_ap.add_argument("--only", choices=("warp",), default=None, help="run T12 alone")
ARGS = _ap.parse_args()
CFG = SC.load_config(ARGS.config)
print(f"config: {os.path.basename(CFG['file'])}")


class A:
    run = staticmethod(lambda case: _A.run(case, CFG))


class H:
    run = staticmethod(lambda case: _H.run(case, CFG))

from npe_mx_emu import base_mx_quantizer, mx_dequantizer, MXObserver, bcu, pcu_div
from npe_mx_emu import pack_warp_reference
torch.set_printoptions(precision=6)

def hdr(s): print(f"\n### {s}")


# ---------------------------------------------------------------- T12 warp
# One vector per difference D1-D10 of the warp deep run, labelled with the
# number of its question in the list sent to the HW team. Both sides through
# the sim (main_ALG.warp, main_HW.warp); where main_HW refuses (batch, signed
# reference), the refusal is shown and their code is called directly.
# Reference values are codes; acc = the bilinear sum in code units.
import main as _M


def _ramp(H, W, f):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    return f(xx, yy).view(1, 1, H, W)


def _const_flow(B, H, W, fx, fy):
    return C._flow((B, 2, H, W), ("const", fx, fy), None)


def _warp(codes, flow, rs=C.P2, os_=C.P2):
    case = C.warp_case("t12", codes, flow, rs, os_)
    a, h = A.run(case), H.run(case)
    acc_a = a["out_pre"].double() / rs
    acc_h = h["hw_acc"].double() / 2.0 ** int(h["frac_sum"])
    return case, a, h, acc_a, acc_h


def _v(label, text):
    print(f"  [{label}] {text}")


def warp_section():
    hdr("T12 warp: ALG (main_ALG) vs HW (their CalibratedMXWarp via main_HW)")
    sp = _H.warp_spec(CFG)
    print(f"  config grid: int{sp['grid_bits']}, frac {sp['frac_cfg']}, relative flow "
          f"(call site {sp['call_site']}); theirs: absolute position, frac from the "
          f"data. acc in reference code units.")
    # D1 border, Q1: ref[y,x] = 10(x+1)+y on 4x6, sample just outside the frame
    ref = _ramp(4, 6, lambda x, y: 10 * (x + 1) + y)
    for (fx, fy), (yy, xx), where in (((-0.25, 0.0), (1, 0), "x = -0.25, y = 1"),
                                      ((0.0, 0.25), (3, 2), "x = 2, y = H-1+0.25")):
        _, a, h, acc_a, acc_h = _warp(ref, _const_flow(1, 4, 6, fx, fy))
        _v("D1 | HW Q1", f"border, sample at {where}: ALG (zero outside) "
           f"{float(acc_a[0, 0, yy, xx]):.2f}, HW {float(acc_h[0, 0, yy, xx]):.2f}")
    # D1 identity, Q2: zero flow, last row (3x4) and last column (W = 32)
    ref = _ramp(3, 4, lambda x, y: 12 * y + 3 * x)
    _, a, h, acc_a, acc_h = _warp(ref, _const_flow(1, 3, 4, 0.0, 0.0))
    _v("D1 | HW Q2", f"zero flow, 3x4, last row: ref {ref[0, 0, -1].tolist()}, "
       f"ALG {acc_a[0, 0, -1].tolist()}, HW {acc_h[0, 0, -1].tolist()}")
    ref = _ramp(2, 32, lambda x, y: 3 * x)
    _, a, h, acc_a, acc_h = _warp(ref, _const_flow(1, 2, 32, 0.0, 0.0))
    _v("D1 | HW Q2", f"zero flow, 2x32, last column: ref {int(ref[0, 0, 0, -1])}, "
       f"ALG {float(acc_a[0, 0, 0, -1]):g}, HW {float(acc_h[0, 0, 0, -1]):g}")
    # D2 fraction bits, Q3; D3 their sum, Q4: zero flow per frame size
    sums = []
    for Hh, Ww in ((4, 6), (8, 8), (6, 40), (6, 56), (6, 64), (2, 160), (65, 65)):
        h = H.run(C.warp_case("t12", torch.zeros(1, 1, Hh, Ww),
                              _const_flow(1, Hh, Ww, 0.0, 0.0), C.P2, C.P2))
        sums.append((Hh, Ww, h["frac_cfg"].tolist(), h["hw_frac_bits"].tolist(),
                     int(h["frac_sum"])))
    _v("D2 | HW Q3", "fraction bits (x, y), config vs theirs: " + ", ".join(
        f"{Hh}x{Ww} {fc} vs {fh}" for Hh, Ww, fc, fh, _ in sums))
    g = torch.Generator().manual_seed(12)
    ref = torch.randint(0, 128, (1, 1, 2, 160), generator=g)
    flow = torch.round((torch.rand(1, 2, 2, 160, generator=g) * 2 - 1) * 3 * 256) / 256
    case, a, h, acc_a, acc_h = _warp(ref, flow)
    st = {r["stage"]: r for r in _M.warp_stages(case, a, h)}
    _v("D2 | HW Q3", f"2x160, flow on 1/256: their frac {h['hw_frac_bits'].tolist()}, "
       f"grid position differs on {st['grid px']['n_diff']}/{st['grid px']['n']} "
       f"(max {st['grid px']['max_d'] * 256:g}/256 px), interior acc on "
       f"{st['acc interior']['n_diff']}/{st['acc interior']['n']} "
       f"(max {st['acc interior']['max_d']:.3f})")
    _v("D3 | HW Q4", f"frx + fry (limit {_H.WARP_FRAC_SUM_LIMIT}): " + ", ".join(
        f"{Hh}x{Ww} {k}{' OVER' if k > _H.WARP_FRAC_SUM_LIMIT else ''}"
        for Hh, Ww, _, _, k in sums))
    # D4 grid ties, D6 reference ties, Q5
    ref = torch.full((1, 1, 2, 100), 50)
    for t in (0.5, -0.5):
        case, a, h, _, _ = _warp(ref, _const_flow(1, 2, 100, t / 256, 0.0))
        fr = int(h["hw_frac_bits"][0])
        dx_a = int(a["grid_code"][0, 0, 0, 10])
        dx_h = float(h["hw_grid_code"][0, 0, 0, 10]) / 2 ** fr - 10
        _v("D4 | HW Q5", f"grid tie, flow x {t:+}/256 (their frac x {fr}): ALG "
           f"{dx_a:+d}/256 px, HW {dx_h * 256:+g}/256 px")
    ref = torch.tensor([[[[2.5, 3.5, 4.0]] * 2]])
    _, a, h, _, _ = _warp(ref, _const_flow(1, 2, 3, 0.0, 0.0))
    _v("D6 | HW Q5", f"reference ties {ref[0, 0, 0].tolist()}: ALG codes "
       f"{a['ref_code'][0, 0, 0].tolist()}, HW {h['hw_ref_code'][0, 0, 0, :3].tolist()}")
    # D5 reference format, Q6
    ref = torch.tensor([[[[-50, 0, 127, 200]] * 2]])
    try:
        H.run(C.warp_case("t12", ref, _const_flow(1, 2, 4, 0.0, 0.0), C.P2, C.P2))
        refusal = "ran"
    except NotImplementedError as e:
        refusal = f"refused ({str(e)[:60]}...)"
    case = C.warp_case("t12", ref, _const_flow(1, 2, 4, 0.0, 0.0), C.P2, C.P2)
    their = pack_warp_reference(ref.double())
    _v("D5 | HW Q6", f"codes {ref[0, 0, 0].tolist()}: ALG int8 "
       f"{A.run(case)['ref_code'][0, 0, 0].tolist()}, their packer "
       f"{their.data[0, 0, 0, :4].tolist()}; main_HW {refusal}")
    # D7 output requant, Q7: acc 37.5, out step = ref step
    ref = _ramp(2, 40, lambda x, y: 37 + x % 2)
    _, a, h, acc_a, acc_h = _warp(ref, _const_flow(1, 2, 40, 0.5, 0.0))
    k = int(h["frac_sum"])
    pcu = pcu_div(h["hw_acc"][0, 0, 0, :1], -k, "round")
    _v("D7 | HW Q7", f"acc {float(acc_h[0, 0, 0, 0]):g}: ALG int8 code "
       f"{int(a['out_code'][0, 0, 0, 0])}, HW out {float(h['out_raw'][0, 0, 0, 0]):g} "
       f"(no PCU), their pcu_div half up {int(pcu)}; negative ties need a signed ref")
    # D8 batch, Q8: batch 1 moved +1 px
    ref = _ramp(2, 8, lambda x, y: 10 * (x + 1)).expand(2, 1, 2, 8).clone()
    flow = torch.zeros(2, 2, 2, 8)
    flow[1, 0] = 1.0
    try:
        H.run(C.warp_case("t12", ref, flow, C.P2, C.P2))
        refusal = "ran"
    except NotImplementedError as e:
        refusal = f"refused ({str(e)[:50]}...)"
    a = A.run(C.warp_case("t12", ref, flow, C.P2, C.P2))
    r = _H.run_their_warp(ref.double(), _H.abs_coords(flow), sp["grid_bits"],
                          _H.hw_acc_bits(CFG))
    _v("D8 | HW Q8", f"batch 1 moved +1 px, row 0 x 0..3: ALG "
       f"{(a['out_pre'][1, 0, 0, :4] / C.P2).tolist()}, their code "
       f"{r['out'][1, 0, 0, :4].tolist()}; main_HW {refusal}")
    # D9 float32 store of the acc, Q9: continuous flow, small frame
    g = torch.Generator().manual_seed(9)
    ref = torch.randint(0, 128, (1, 1, 8, 8), generator=g)
    flow = (torch.rand(1, 2, 8, 8, generator=g) * 2 - 1) * 1.5
    h = H.run(C.warp_case("t12", ref, flow, C.P2, C.P2))
    n, big, rounded = _M.warp_float32_store(h)
    back = h["out_raw"].double() * 2.0 ** int(h["frac_sum"])
    _v("D9 | HW Q9", f"8x8 continuous flow, frx+fry {int(h['frac_sum'])}: {big}/{n} acc "
       f"above 2^24, float32 output rounds {rounded} of them, by up to "
       f"{int((back - h['hw_acc'].double()).abs().max())} acc LSB (hw_acc is exact)")
    # D1 padding, Q10: x = W-1+0.5 on a constant 100, W = 40 vs 32
    for Ww in (40, 32):
        _, a, h, acc_a, acc_h = _warp(torch.full((1, 1, 2, Ww), 100),
                                      _const_flow(1, 2, Ww, 0.5, 0.0))
        _v("D1 | HW Q10", f"W {Ww} (padded to {h['hw_ref_code'].shape[-1]}), "
           f"x = W-1+0.5 on 100: ALG {float(acc_a[0, 0, 0, -1]):g}, "
           f"HW {float(acc_h[0, 0, 0, -1]):g}")
    print("  (D10, their float product in bcu, does not show here: the sim feeds "
          "float64 coordinates, as their notebooks do)")


if ARGS.only == "warp":
    warp_section()
    sys.exit(0)

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

warp_section()
