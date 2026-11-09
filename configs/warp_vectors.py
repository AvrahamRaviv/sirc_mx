"""
warp_vectors.py - the warp test vectors of hw_vs_alg_check.py (T12).

    for v in vectors(cfg):            # one per ALG-vs-HW difference
        print(v["label"], v["text"])  # "D1 | HW Q1", the numbers it shows
        v["alg"], v["hw"]             # each side's values (None: one side only)

One vector per difference D1-D10 of the warp deep run, labelled with the number
of its question in the list sent to the HW team. Both sides through the sim
(main_ALG.warp, main_HW.warp); where main_HW refuses (batch, signed reference),
the refusal is recorded and their code is called directly. Reference values are
codes; acc = the bilinear sum in reference code units. The text is built from
the vector's own values ("alg", "hw" and its extra keys: counts, refusal), so a
test of the values tests what is printed.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch

import cases as C
import main as M
import main_ALG as ALG
import main_HW as HW
from npe_mx_emu import pack_warp_reference, pcu_div

# frame sizes of the fraction-bit vectors (D2 / D3), zero flow
FRAC_SIZES = ((4, 6), (8, 8), (6, 40), (6, 56), (6, 64), (2, 160), (65, 65))


def _ramp(H, W, f):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    return f(xx, yy).view(1, 1, H, W)


def _vec(label, text, alg, hw, **more):
    return {"label": label, "text": text, "alg": alg, "hw": hw, **more}


class _Sim:
    """Both sides of the sim on one config, and the warp helpers on top."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = HW.warp_spec(cfg)

    def alg(self, case):
        return ALG.run(case, self.cfg)

    def hw(self, case):
        return HW.run(case, self.cfg)

    def warp(self, codes, flow, rs=C.P2, os_=C.P2):
        """Both sides on one warp case; acc of each side in ref code units."""
        case = C.warp_case("t12", codes, flow, rs, os_)
        a, h = self.alg(case), self.hw(case)
        acc_a = a["out_pre"].double() / rs
        acc_h = h["hw_acc"].double() / 2.0 ** int(h["frac_sum"])
        return case, a, h, acc_a, acc_h

    def hw_refusal(self, case, width):
        """What main_HW does with the case: "ran", or "refused (<why>)" with
        the first `width` characters of its message."""
        try:
            self.hw(case)
            return "ran"
        except NotImplementedError as e:
            return f"refused ({str(e)[:width]}...)"


# What a side raises on a config the warp cannot run on: main_ALG a missing
# fxp entry (KeyError) or a bad one (ValueError), main_HW a refusal or a
# missing grid format. Any other error is a bug and propagates.
CANNOT_RUN = (KeyError, ValueError, NotImplementedError, SystemExit)


def probe(cfg):
    """None when both sides run a plain warp case on `cfg`, else why not
    ("<side> <error>"): every vector would fail the same way."""
    case = C.warp_case("probe", _ramp(2, 4, lambda x, y: x + y),
                       C.const_flow(1, 2, 4, 0.0, 0.0), C.P2, C.P2)
    for side, run in (("main_ALG", ALG.run), ("main_HW", HW.run)):
        try:
            run(case, cfg)
        except CANNOT_RUN as e:
            return f"{side} {type(e).__name__}: {e}"
    return None


def vectors(cfg):
    """Every T12 vector on `cfg`, in print order."""
    sim = _Sim(cfg)
    return (_border(sim) + _identity(sim) + _frac(sim) + _ties(sim)
            + [_ref_format(sim), _out_requant(sim), _batch(sim), _float32(sim)]
            + _padding(sim))


def _border(sim):
    """D1, Q1: ref[y,x] = 10(x+1)+y on 4x6, sample just outside the frame."""
    ref = _ramp(4, 6, lambda x, y: 10 * (x + 1) + y)
    out = []
    for (fx, fy), (yy, xx), where in (((-0.25, 0.0), (1, 0), "x = -0.25, y = 1"),
                                      ((0.0, 0.25), (3, 2), "x = 2, y = H-1+0.25")):
        _, _, _, acc_a, acc_h = sim.warp(ref, C.const_flow(1, 4, 6, fx, fy))
        a, h = float(acc_a[0, 0, yy, xx]), float(acc_h[0, 0, yy, xx])
        out.append(_vec("D1 | HW Q1", f"border, sample at {where}: ALG (zero outside) "
                        f"{a:.2f}, HW {h:.2f}", a, h, ref=ref, at=(yy, xx)))
    return out


def _identity(sim):
    """D1, Q2: zero flow, last row (3x4) and last column (W = 32)."""
    ref = _ramp(3, 4, lambda x, y: 12 * y + 3 * x)
    _, _, _, acc_a, acc_h = sim.warp(ref, C.const_flow(1, 3, 4, 0.0, 0.0))
    a, h = acc_a[0, 0, -1].tolist(), acc_h[0, 0, -1].tolist()
    row = _vec("D1 | HW Q2", f"zero flow, 3x4, last row: ref {ref[0, 0, -1].tolist()}, "
               f"ALG {a}, HW {h}", a, h, ref=ref)
    ref = _ramp(2, 32, lambda x, y: 3 * x)
    _, _, _, acc_a, acc_h = sim.warp(ref, C.const_flow(1, 2, 32, 0.0, 0.0))
    a, h = float(acc_a[0, 0, 0, -1]), float(acc_h[0, 0, 0, -1])
    col = _vec("D1 | HW Q2", f"zero flow, 2x32, last column: ref {int(ref[0, 0, 0, -1])}, "
               f"ALG {a:g}, HW {h:g}", a, h, ref=ref)
    return [row, col]


def _frac(sim):
    """D2 fraction bits, Q3; D3 their sum, Q4: zero flow per frame size; and
    a flow on the config's 1/256 grid that their data-derived format moves."""
    sums = []
    for Hh, Ww in FRAC_SIZES:
        h = sim.hw(C.warp_case("t12", torch.zeros(1, 1, Hh, Ww),
                               C.const_flow(1, Hh, Ww, 0.0, 0.0), C.P2, C.P2))
        sums.append((Hh, Ww, h["frac_cfg"].tolist(), h["hw_frac_bits"].tolist(),
                     int(h["frac_sum"])))
    size = {(Hh, Ww): (fc, fh) for Hh, Ww, fc, fh, _ in sums}
    q3 = _vec("D2 | HW Q3", "fraction bits (x, y), config vs theirs: " + ", ".join(
        f"{Hh}x{Ww} {fc} vs {fh}" for Hh, Ww, fc, fh, _ in sums),
        {k: v[0] for k, v in size.items()}, {k: v[1] for k, v in size.items()})
    g = torch.Generator().manual_seed(12)
    ref = torch.randint(0, 128, (1, 1, 2, 160), generator=g)
    flow = torch.round((torch.rand(1, 2, 2, 160, generator=g) * 2 - 1) * 3 * 256) / 256
    case, a, h, _, _ = sim.warp(ref, flow)
    st = {r["stage"]: r for r in M.warp_stages(case, a, h)}
    grid, acc = st["grid px"], st["acc interior"]
    moved = _vec("D2 | HW Q3", f"2x160, flow on 1/256: their frac {h['hw_frac_bits'].tolist()}, "
                 f"grid position differs on {grid['n_diff']}/{grid['n']} "
                 f"(max {grid['max_d'] * 256:g}/256 px), interior acc on "
                 f"{acc['n_diff']}/{acc['n']} (max {acc['max_d']:.3f})",
                 a["grid_code"], h["hw_grid_code"], grid=grid, acc=acc)
    lim = HW.WARP_FRAC_SUM_LIMIT
    q4 = _vec("D3 | HW Q4", f"frx + fry (limit {lim}): " + ", ".join(
        f"{Hh}x{Ww} {k}{' OVER' if k > lim else ''}" for Hh, Ww, _, _, k in sums),
        None, {(Hh, Ww): k for Hh, Ww, _, _, k in sums})
    return [q3, moved, q4]


def _ties(sim):
    """D4 grid ties and D6 reference ties, Q5."""
    out = []
    ref = torch.full((1, 1, 2, 100), 50)
    for t in (0.5, -0.5):
        _, a, h, _, _ = sim.warp(ref, C.const_flow(1, 2, 100, t / 256, 0.0))
        fr = int(h["hw_frac_bits"][0])
        dx_a = int(a["grid_code"][0, 0, 0, 10])
        dx_h = (float(h["hw_grid_code"][0, 0, 0, 10]) / 2 ** fr - 10) * 256
        out.append(_vec("D4 | HW Q5", f"grid tie, flow x {t:+}/256 (their frac x {fr}): "
                        f"ALG {dx_a:+d}/256 px, HW {dx_h:+g}/256 px", dx_a, dx_h, tie=t))
    ref = torch.tensor([[[[2.5, 3.5, 4.0]] * 2]])
    _, a, h, _, _ = sim.warp(ref, C.const_flow(1, 2, 3, 0.0, 0.0))
    ca, ch = a["ref_code"][0, 0, 0].tolist(), h["hw_ref_code"][0, 0, 0, :3].tolist()
    out.append(_vec("D6 | HW Q5", f"reference ties {ref[0, 0, 0].tolist()}: ALG codes "
                    f"{ca}, HW {ch}", ca, ch, ref=ref[0, 0, 0]))
    return out


def _ref_format(sim):
    """D5 reference format, Q6: signed and above-int8 codes."""
    ref = torch.tensor([[[[-50, 0, 127, 200]] * 2]])
    case = C.warp_case("t12", ref, C.const_flow(1, 2, 4, 0.0, 0.0), C.P2, C.P2)
    refusal = sim.hw_refusal(case, 60)
    a = sim.alg(case)["ref_code"][0, 0, 0].tolist()
    h = pack_warp_reference(ref.double()).data[0, 0, 0, :4].tolist()
    return _vec("D5 | HW Q6", f"codes {ref[0, 0, 0].tolist()}: ALG int8 {a}, their packer "
                f"{h}; main_HW {refusal}", a, h, ref=ref[0, 0, 0], refusal=refusal)


def _out_requant(sim):
    """D7 output requant, Q7: acc 37.5, out step = ref step."""
    ref = _ramp(2, 40, lambda x, y: 37 + x % 2)
    _, a, h, _, acc_h = sim.warp(ref, C.const_flow(1, 2, 40, 0.5, 0.0))
    pcu = int(pcu_div(h["hw_acc"][0, 0, 0, :1], -int(h["frac_sum"]), "round"))
    acc = float(acc_h[0, 0, 0, 0])
    ch = float(h["out_raw"][0, 0, 0, 0])
    if "out_code" not in a:         # no output fxp: the lib leaves it unrounded
        ca = float(a["out_pre"][0, 0, 0, 0] / C.P2)
        alg = f"ALG unrounded {ca:g} (config has no output fxp)"
    else:
        ca = int(a["out_code"][0, 0, 0, 0])
        alg = f"ALG int8 code {ca}"
    return _vec("D7 | HW Q7", f"acc {acc:g}: {alg}, HW out {ch:g} (no PCU), "
                f"their pcu_div half up {pcu}; negative ties need a signed ref",
                ca, ch, acc=acc, pcu=pcu)


def _batch(sim):
    """D8 batch, Q8: batch 1 moved +1 px, batch 0 still."""
    ref = _ramp(2, 8, lambda x, y: 10 * (x + 1)).expand(2, 1, 2, 8).clone()
    flow = torch.zeros(2, 2, 2, 8)
    flow[1, 0] = 1.0
    case = C.warp_case("t12", ref, flow, C.P2, C.P2)
    refusal = sim.hw_refusal(case, 50)
    a = (sim.alg(case)["out_pre"][1, 0, 0, :4] / C.P2).tolist()
    r = HW.run_their_warp(ref.double(), HW.abs_coords(flow), sim.spec["grid_bits"],
                          HW.hw_acc_bits(sim.cfg))
    h = r["out"][1, 0, 0, :4].tolist()
    return _vec("D8 | HW Q8", f"batch 1 moved +1 px, row 0 x 0..3: ALG {a}, their code "
                f"{h}; main_HW {refusal}", a, h, ref=ref, refusal=refusal)


def _float32(sim):
    """D9 float32 store of the acc, Q9: continuous flow, small frame."""
    g = torch.Generator().manual_seed(9)
    ref = torch.randint(0, 128, (1, 1, 8, 8), generator=g)
    flow = (torch.rand(1, 2, 8, 8, generator=g) * 2 - 1) * 1.5
    h = sim.hw(C.warp_case("t12", ref, flow, C.P2, C.P2))
    n, big, rounded = M.warp_float32_store(h)
    back = h["out_raw"].double() * 2.0 ** int(h["frac_sum"])
    lsb = int((back - h["hw_acc"].double()).abs().max())
    return _vec("D9 | HW Q9", f"8x8 continuous flow, frx+fry {int(h['frac_sum'])}: {big}/{n} "
                f"acc above 2^24, float32 output rounds {rounded} of them, by up to "
                f"{lsb} acc LSB (hw_acc is exact)", None, h["hw_acc"],
                n=n, big=big, rounded=rounded, lsb=lsb)


def _padding(sim):
    """D1 padding, Q10: x = W-1+0.5 on a constant 100, W = 40 vs 32."""
    out = []
    for Ww in (40, 32):
        _, _, h, acc_a, acc_h = sim.warp(torch.full((1, 1, 2, Ww), 100),
                                         C.const_flow(1, 2, Ww, 0.5, 0.0))
        a, hv = float(acc_a[0, 0, 0, -1]), float(acc_h[0, 0, 0, -1])
        out.append(_vec("D1 | HW Q10", f"W {Ww} (padded to {h['hw_ref_code'].shape[-1]}), "
                        f"x = W-1+0.5 on 100: ALG {a:g}, HW {hv:g}", a, hv, W=Ww))
    return out


# D10, their float product in bcu, has no vector: the sim feeds float64
# coordinates, as their notebooks do.
D10_NOTE = ("(D10, their float product in bcu, does not show here: the sim feeds "
            "float64 coordinates, as their notebooks do)")
