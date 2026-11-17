"""
warp_vectors.py - the warp test vectors of hw_vs_alg_check.py (T12).

    for v in vectors(cfg):            # one per question sent to the HW team
        print(v["label"], v["text"])  # "D1 | HW Q1", the numbers it shows
        v["alg"], v["hw"], v["want"]  # ALG, their emulator, the HW answer
        v["status"]                   # "resolved" or "open: <what is open>"

One vector per difference D1-D10 of the warp deep run, labelled with the number
of its question in hw/warp_hw_questions_and_answers.md (the HW spec for the
warp). `want` is the value the HW team's ANSWER gives, worked out here from
the rule it states, never read from their code; `hw` is what their corrected
emulator (the reference, Q11) gives, through main_HW; `alg` what the lib
gives, through main_ALG. A vector whose question the answer leaves open says
so in `status` (its `want` is then the rule both sides follow today, or None
where the answer gives none). Reference values are codes; acc = the bilinear
sum in reference code units. The text is built from the vector's own values,
so a test of the values tests what is printed.
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
import sim_config as SC
from npe_mx_emu import pack_warp_reference

RESOLVED = "resolved"


def ramp(H, W, f):
    """A [1, 1, H, W] integer frame f(x, y), x along W."""
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    return f(xx, yy).view(1, 1, H, W)


def _vec(label, text, alg, hw, want, status, **more):
    """One vector. The text ends with the HW answer and its status."""
    tail = "resolved" if status == RESOLVED else status
    return {"label": label, "text": f"{text}; HW answer {want} [{tail}]", "alg": alg,
            "hw": hw, "want": want, "status": status, **more}


def gaps(vs):
    """(alg, hw): labels and texts of the vectors where ALG, or their
    emulator, gives another value than the HW answer, resolved ones only (an
    open question has no answer to differ from)."""
    def off(side):
        return [v for v in vs if v["status"] == RESOLVED and v[side] is not None
                and v[side] != v["want"]]
    return off("alg"), off("hw")


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
        acc_h = h["hw_acc"].double() / 2.0 ** int(h["frac_bits"].sum())
        return case, a, h, acc_a, acc_h

    def hw_refusal(self, case, width):
        """What main_HW does with the case: "ran", or "refused (<why>)" with
        the first `width` characters of its message."""
        try:
            self.hw(case)
            return "ran"
        except NotImplementedError as e:
            return f"refused ({str(e)[:width]}...)"


def probe(cfg):
    """None when both sides run a plain warp case on `cfg`, else why not
    ("<side> <error>"): every vector would fail the same way. Only a
    documented refusal (sim_config.REFUSED: a missing entry, named, or a
    per-site fxp list the lib refuses) is a why; any other error, a KeyError
    or a ValueError included, is a bug: it propagates and the run fails."""
    case = C.warp_case("probe", ramp(2, 4, lambda x, y: x + y),
                       C.const_flow(1, 2, 4, 0.0, 0.0), C.P2, C.P2)
    for side, run in (("main_ALG", ALG.run), ("main_HW", HW.run)):
        try:
            run(case, cfg)
        except SC.REFUSED as e:
            return f"{side} {type(e).__name__}: {e}"
    return None


def vectors(cfg):
    """Every T12 vector on `cfg`, in print order. Never empty."""
    sim = _Sim(cfg)
    vs = (_border(sim) + _identity(sim) + [_frac(sim), _frac_sum(sim)] + _ties(sim)
          + [_ref_format(sim)] + _out_requant(sim) + [_batch(sim), _int_store(sim)]
          + _padding(sim))
    assert vs, "T12 built no vector"
    return vs


def _border(sim):
    """D1, Q1: ref[y,x] = 10(x+1)+y on 4x6, sample just outside the frame.
    Answer: the whole coordinate saturates to the frame (no excess fraction),
    so the sample is the edge pixel. Stated for the far edge; the near edge
    (negative words) and the four-side symmetry were not established."""
    ref = ramp(4, 6, lambda x, y: 10 * (x + 1) + y)
    out = []
    for (fx, fy), (yy, xx), (wy, wx), where, status in (
            ((-0.25, 0.0), (1, 0), (1, 0), "x = -0.25, y = 1",
             "open: Q1 did not establish the near edge; ALG and their emulator clamp to 0"),
            ((0.0, 0.25), (3, 2), (3, 2), "x = 2, y = H-1+0.25", RESOLVED)):
        _, _, _, acc_a, acc_h = sim.warp(ref, C.const_flow(1, 4, 6, fx, fy))
        a, h = float(acc_a[0, 0, yy, xx]), float(acc_h[0, 0, yy, xx])
        want = float(ref[0, 0, wy, wx])
        out.append(_vec("D1 | HW Q1", f"border, sample at {where}: ALG {a:.2f}, HW {h:.2f}",
                        a, h, want, status, ref=ref, at=(yy, xx)))
    return out


def _identity(sim):
    """D1, Q2: zero flow, last row (3x4) and last column (W = 32). Answer:
    zero motion preserves the reference."""
    ref = ramp(3, 4, lambda x, y: 12 * y + 3 * x)
    _, _, _, acc_a, acc_h = sim.warp(ref, C.const_flow(1, 3, 4, 0.0, 0.0))
    a, h = acc_a[0, 0, -1].tolist(), acc_h[0, 0, -1].tolist()
    want = ref[0, 0, -1].double().tolist()
    row = _vec("D1 | HW Q2", f"zero flow, 3x4, last row: ALG {a}, HW {h}", a, h, want,
               RESOLVED, ref=ref)
    ref = ramp(2, 32, lambda x, y: 3 * x)
    _, _, _, acc_a, acc_h = sim.warp(ref, C.const_flow(1, 2, 32, 0.0, 0.0))
    a, h = float(acc_a[0, 0, 0, -1]), float(acc_h[0, 0, 0, -1])
    col = _vec("D1 | HW Q2", f"zero flow, 2x32, last column: ALG {a:g}, HW {h:g}", a, h,
               float(ref[0, 0, 0, -1]), RESOLVED, ref=ref)
    return [row, col]


def _frac(sim):
    """D2, Q3: the grid is relative motion in the supplied Q format (the
    config's fraction bits), not derived from the data. A flow on the
    config's grid: the motion word is flow * 2^frac exactly, on both sides,
    and so is every interior acc."""
    g = torch.Generator().manual_seed(12)
    ref = torch.randint(-128, 128, (1, 1, 2, 160), generator=g)
    fr = sim.spec["frac"]
    flow = torch.round((torch.rand(1, 2, 2, 160, generator=g) * 2 - 1) * 3 * 2 ** fr[0]) / 2 ** fr[0]
    case, a, h, _, _ = sim.warp(ref, flow)
    word = (flow.double() * 2.0 ** fr[0]).long()
    n_a = int((a["grid_code"].long() != word).sum())
    n_h = int((h["grid_code"].long() != word).sum())
    acc = {r["stage"]: r for r in M.warp_stages(case, a, h)}["acc interior"]
    return _vec("D2 | HW Q3", f"2x160, flow on 1/{2 ** fr[0]}: grid words off flow * 2^frac: "
                f"ALG {n_a}, HW {n_h} of {word.numel()}; their frac {h['frac_bits'].tolist()}, "
                f"config {list(fr)}; interior acc ALG vs HW differ on {acc['n_diff']}/{acc['n']}",
                n_a, n_h, 0, RESOLVED, frac=h["frac_bits"].tolist(), acc=acc)


def _frac_sum(sim):
    """D3, Q4: frx + fry up to 16 is legal (8 + 8). Their weight at zero
    fraction is then 2^(frx+fry); the encoding of 2^16 in the unsigned 16-bit
    field is open, so the vector records what their code does: the count of
    weights at that endpoint, on zero flow (every pixel's w00)."""
    H, W = 2, 4
    h = sim.hw(C.warp_case("t12", ramp(H, W, lambda x, y: x), C.const_flow(1, H, W, 0, 0),
                           C.P2, C.P2))
    k = int(h["frac_bits"].sum())
    n = int((h["hw_weights"] == 2 ** k).sum())
    return _vec("D3 | HW Q4", f"frx + fry = {k}: zero flow on {H}x{W}, {n} TRC weights are "
                f"2^{k} (their int64 weight)", None, n, None,
                "open: Q4 did not establish how 2^16 is encoded in the 16-bit field",
                frac_sum=k, n=H * W)


def _ties(sim):
    """D4 grid ties, Q5: the grid word comes from the producing layer's PCU,
    nearest with ties toward +inf (Q7), so +-0.5/256 px -> +1 / 0. ALG's word
    comes from the warp entry's grid fxp round. D6 reference ties, Q5: the
    float-to-code tie rule was not settled (the pipeline reference is
    already integer)."""
    out = []
    ref = torch.full((1, 1, 2, 100), 50)
    for t, want in ((0.5, 1), (-0.5, 0)):
        _, a, h, _, _ = sim.warp(ref, C.const_flow(1, 2, 100, t / 256, 0.0))
        dx_a, dx_h = int(a["grid_code"][0, 0, 0, 10]), int(h["grid_code"][0, 0, 0, 10])
        out.append(_vec("D4 | HW Q5", f"grid tie, flow x {t:+}/256: ALG {dx_a:+d}/256 px, HW "
                        f"{dx_h:+d}/256 px", dx_a, dx_h, want, RESOLVED, tie=t))
    ref = torch.tensor([[[[2.5, 3.5, 4.0]] * 2]])
    _, a, h, _, _ = sim.warp(ref, C.const_flow(1, 2, 3, 0.0, 0.0))
    ca, ch = a["ref_code"][0, 0, 0].tolist(), h["ref_code"][0, 0, 0].tolist()
    out.append(_vec("D6 | HW Q5", f"reference ties {ref[0, 0, 0].tolist()}: ALG codes {ca}, "
                    f"HW {ch}", ca, ch, None,
                    "open: Q5 did not settle the float-to-code tie rule", ref=ref[0, 0, 0]))
    return out


def _ref_format(sim):
    """D5, Q6: the reference is int8 [-128, 127] unless uint8 is selected."""
    ref = torch.tensor([[[[-50, 0, 127, 200]] * 2]])
    case = C.warp_case("t12", ref, C.const_flow(1, 2, 4, 0.0, 0.0), C.P2, C.P2)
    refusal = sim.hw_refusal(case, 60)
    a = sim.alg(case)["ref_code"][0, 0, 0].tolist()
    h = pack_warp_reference(ref.double()).data[0, 0, 0, :4].tolist()
    want = ref[0, 0, 0].clamp(-128, 127).tolist()
    return _vec("D5 | HW Q6", f"codes {ref[0, 0, 0].tolist()}: ALG int8 {a}, their packer "
                f"{h}; main_HW {refusal}", a, h, want, RESOLVED, ref=ref[0, 0, 0],
                refusal=refusal)


def _out_requant(sim):
    """D7, Q7: output through the PCU, int8, nearest with ties toward +inf:
    acc 37.5 -> 38, -37.5 -> -37. Out step = ref step here."""
    ref = torch.cat([ramp(2, 40, lambda x, y: 37 + x % 2),
                     ramp(2, 40, lambda x, y: -37 - x % 2)], 1)
    _, a, h, acc_a, acc_h = sim.warp(ref, C.const_flow(1, 2, 40, 0.5, 0.0))
    acc = acc_h[0, :, 0, 0].tolist()
    ch = h["out_code"][0, :, 0, 0].tolist()
    if "out_code" not in a:         # no output fxp: the lib leaves it unrounded
        ca = acc_a[0, :, 0, 0].tolist()
        alg = f"ALG unrounded {ca} (config has no output fxp)"
    else:
        ca = a["out_code"][0, :, 0, 0].tolist()
        alg = f"ALG int8 codes {ca}"
    return [_vec("D7 | HW Q7", f"acc {acc}: {alg}, HW {ch}", ca, ch, [38, -37], RESOLVED,
                 acc=acc)]


def _batch(sim):
    """D8, Q8: one reference frame and one grid per call; native batch is not
    established and not needed now. Batch 1 moved +1 px: ALG moves it; their
    code raises on two frames, main_HW refuses (named)."""
    ref = ramp(2, 8, lambda x, y: 10 * (x + 1)).expand(2, 1, 2, 8).clone()
    flow = torch.zeros(2, 2, 2, 8)
    flow[1, 0] = 1.0
    case = C.warp_case("t12", ref, flow, C.P2, C.P2)
    refusal = sim.hw_refusal(case, 50)
    a = (sim.alg(case)["out_pre"][1, 0, 0, :4] / C.P2).tolist()
    try:
        HW.run_their_warp(ref.double(), HW.grid_word(flow, sim.spec["frac"], sim.spec["grid_bits"]),
                          sim.spec["frac"], sim.spec["grid_bits"], HW.hw_acc_bits(sim.cfg))
        theirs = "ran"
    except ValueError as e:
        theirs = f"raise ValueError ({e})"
    ran = "refused" if refusal.startswith("refused") else "ran"
    return _vec("D8 | HW Q8", f"batch 1 moved +1 px, row 0 x 0..3: ALG {a}; their code "
                f"{theirs}; main_HW {refusal}", None, ran, "refused", RESOLVED,
                ref=ref, alg_out=a, theirs=theirs, refusal=refusal)


def _int_store(sim):
    """D9 (float32 output store) and D10 (float product in bcu), Q9:
    integer arithmetic through the PCU (int64 product, int64 store). Continuous flow on 1/256, random int8
    reference: their int8 output is the PCU rule (nearest, ties toward +inf,
    int8 saturation) of their own int64 acc on every pixel."""
    g = torch.Generator().manual_seed(9)
    ref = torch.randint(-128, 128, (1, 1, 8, 8), generator=g)
    flow = torch.round((torch.rand(1, 2, 8, 8, generator=g) * 2 - 1) * 1.5 * 256) / 256
    h = sim.hw(C.warp_case("t12", ref, flow, C.P2, C.P2))
    k = int(h["frac_bits"].sum())
    want = torch.floor(h["hw_acc"].double() / 2.0 ** k + 0.5).clamp(-128, 127)
    n = int((h["out_code"].double() != want).sum())
    dtypes = f"{h['hw_acc'].dtype}, {h['out_code'].dtype}"
    return _vec("D9 | HW Q9", f"8x8 continuous flow, frx+fry {k}: acc and output {dtypes}; "
                f"output off the PCU rule of their own acc on {n}/{want.numel()}",
                None, n, 0, RESOLVED, dtypes=dtypes)


def _padding(sim):
    """D1 padding, Q10: no padding, the border follows the frame's own width.
    x = W-1+0.5 on a constant 100 saturates to W-1: 100 at any W."""
    out = []
    for Ww in (40, 32):
        _, _, h, acc_a, acc_h = sim.warp(torch.full((1, 1, 2, Ww), 100),
                                         C.const_flow(1, 2, Ww, 0.5, 0.0))
        a, hv = float(acc_a[0, 0, 0, -1]), float(acc_h[0, 0, 0, -1])
        out.append(_vec("D1 | HW Q10", f"W {Ww} (their packed width {h['ref_code'].shape[-1]}), "
                        f"x = W-1+0.5 on 100: ALG {a:g}, HW {hv:g}", a, hv, 100.0, RESOLVED,
                        W=Ww))
    return out
