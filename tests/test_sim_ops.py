"""Single-op simulator (configs/): every case runs on the ALG side, and its
integer taps explain its own output - conv acc from a_int/w_int, warp out_code
from ref_code/grid_code."""

import copy
import json
import os
import re
import sys

import pytest
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(_ROOT, "configs"), _ROOT, os.path.dirname(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cases as C           # noqa: E402
import main_ALG as ALG      # noqa: E402
import main_HW as HW        # noqa: E402
import sim_cli as CLI       # noqa: E402
import sim_config as SC     # noqa: E402

E4 = SC.load_config("mx_config_npe_e4.json")   # NPE exponent width (4 bits)


def _expand(code, axis, n, bs=C.BS):
    """Per-block exponent codes -> per-element real exponent along `axis`."""
    return (code.double() - 127).repeat_interleave(bs, dim=axis).narrow(axis, 0, n)


@pytest.mark.parametrize("name", C.select("conv"))
def test_conv_acc_from_taps(name):
    case = C.build(name)
    t = ALG.run(case)
    a = t["a_int"].double() * 2.0 ** (_expand(t["a_exp"], 3, t["a_int"].shape[3]) - 6)
    w_int = t["w_int"]
    O = w_int.shape[0]
    Ew = _expand(t["w_exp"], 1, w_int[0].numel()).reshape(w_int.shape)
    w = w_int.double() * 2.0 ** (Ew - 6)
    b = None
    if case["b"] is not None:   # bias lands on the accumulator grid
        b = torch.round(case["b"].double() * 2.0 ** 20) * 2.0 ** -20
    p = case["params"]
    want = F.conv2d(a, w, b, p["stride"], p["padding"], p["dilation"])
    assert int(t["sat"].sum()) == 0
    assert torch.equal(t["acc"].double(), want * 2.0 ** 20)


@pytest.mark.parametrize("name", C.select("warp"))
def test_warp_out_from_taps(name):
    case = C.build(name)
    t = ALG.run(case)
    rs, os_ = case["warp"]["ref_scale"], case["warp"]["out_scale"]
    pre = C.Warp()(t["ref_code"].double() * rs, t["grid_code"].double() / 256)
    v = pre / os_
    want = (torch.sign(v) * torch.floor(v.abs() + 0.5)).clamp(-128, 127)
    assert torch.equal(t["out_code"].double(), want)


@pytest.mark.parametrize("name", C.select("convt"))
def test_convt_runs(name):
    t = ALG.run(C.build(name))
    assert t["out"].shape[2:] == (10, 32)


@pytest.mark.parametrize("name", ["conv1x1", "conv3x3_p0"])
def test_hw_conv_runs(name):
    """HW team's npe_mx_emu conv, wired in: runs, shapes line up with ALG."""
    t = HW.run(C.build(name), E4)
    assert t["out"].shape == ALG.run(C.build(name))["out"].shape
    assert t["hw_acc"].shape == t["out_raw"].shape


@pytest.mark.parametrize("name", ["conv3x3", "conv3x3_d2"])
def test_hw_conv_unsupported_raises(name):
    with pytest.raises(NotImplementedError):
        HW.run(C.build(name), E4)


def test_hw_placeholder():
    with pytest.raises(NotImplementedError, match="HW convt"):
        HW.run(C.build("convt_k3_s2"), E4)


def test_int_input_is_on_grid_and_hw_preround_is_noop():
    """--input int: x * 2^x_frac is an exact integer on both sides, so
    x * 2^x_cal is already integer while x_cal >= x_frac (HW rounds only an
    input layer now; either way nothing would move)."""
    case = C.build("conv3x3_p0", x_frac=4)
    v = case["x"].double() * 16
    assert bool((v == v.round()).all())
    t = HW.run(case, E4)
    assert int(t["x_cal"]) >= 4
    assert int(t["pre_round_moved"]) == 0


def test_exponent_width_comes_from_config():
    """scale_bits drives both sides: ALG codes are E + 2^(n-1)-1; HW picks its
    mx_type (and so its wrap mask) from it, and refuses a type it lacks."""
    c = C.build("conv3x3_p0")
    dof = SC.load_config("mx_config_dof.json")
    e8 = ALG.run(c, dof)
    e4 = ALG.run(c, E4)
    assert int(e8["a_exp"].min()) > 100          # 8-bit: E + 127
    assert int(e4["a_exp"].max()) <= 14          # 4-bit: E + 7, in [0, 14]
    assert bool((e8["a_exp"].long() - 127 == e4["a_exp"].long() - 7).all())
    assert HW.hw_format(E4) == ("mx-e4bm8", 32)
    assert HW.hw_format(dof) == ("mx-e8bm8", 32)
    e5 = {"mx_specs": dict(E4["mx_specs"], scale_bits=5)}
    with pytest.raises(NotImplementedError, match="not in npe_mx_emu"):
        HW.hw_format(e5)


def test_hw_conv_refuses_non_int8_mantissa():
    """Their factory has mx-e4bm16 (the warp weight lane); their conv on it
    returns numbers ~30x off instead of raising, so an int16 conv with a 4-bit
    register is refused, whichever field gives the 4 bits, and run() refuses
    it before reaching their conv."""
    i16 = dict(E4["mx_specs"], a_elem_format="int16", w_elem_format="int16")
    by_scale_bits = dict(i16, scale_bits=4)
    by_register = dict(i16, scale_bits=8, xblock_accum=dict(
        i16["xblock_accum"], in_exp_mode="infer", in_exp_bits=4))
    for sp in (by_scale_bits, by_register):
        with pytest.raises(NotImplementedError, match="int16.*mx-e4bm16 is the warp's"):
            HW.hw_format({"mx_specs": sp})
    with pytest.raises(NotImplementedError, match="warp's 16-bit lane"):
        HW.run(C.build("conv1x1"), {**E4, "mx_specs": by_scale_bits})


@pytest.mark.parametrize("fmt, scale_bits", [("int6", 4), ("int4", 4), ("int16", 8)])
def test_hw_conv_refusal_names_only_what_is_true(fmt, scale_bits):
    """A width with no lane in their factory is refused as that, never as
    "the warp lane" (only mx-e4bm16 is)."""
    sp = dict(E4["mx_specs"], a_elem_format=fmt, w_elem_format=fmt, scale_bits=scale_bits)
    with pytest.raises(NotImplementedError) as e:
        HW.hw_format({"mx_specs": sp})
    msg = str(e.value)
    assert f"npe_mx_emu has no mx-e{scale_bits}bm{fmt[3:]}" in msg
    assert "warp" not in msg


def test_hw_acc_width_comes_from_config():
    """xblock_accum.bits sets their accumulator width; unsupported widths are
    refused."""
    assert HW.hw_acc_bits(E4) == 48
    t = HW.run(C.build("conv1x1"), E4)
    assert int(t["acc_bits"]) == 48
    xa = dict(E4["mx_specs"]["xblock_accum"], bits=35)
    with pytest.raises(NotImplementedError, match="supports"):
        HW.hw_acc_bits({"mx_specs": dict(E4["mx_specs"], xblock_accum=xa)})


# --- PLA: activation folded into the PCU (HW bin form) -------------------------

PLA = SC.load_config("mx_config_npe_e4_pla.json")
PLA_CASES = ("conv_prelu", "conv_relu", "conv_prelu_negb")


@pytest.mark.parametrize("name", PLA_CASES)
def test_pla_both_sides_use_config_bins(name):
    """ALG (lib) and HW read the same bins from the config's hw_grids; the sim
    never calibrates."""
    g = PLA["hw_grids"][name]
    a = ALG.run(C.build(name), PLA)
    h = HW.run(C.build(name), PLA)
    assert int(a["e_layer_min"]) == g["e_layer_min"]
    assert int(a["e_qa_out"]) == g["e_qa_out"]
    for k in ("pla_bin_bounds", "pla_bin_scales", "pla_bin_biases"):
        assert a[k].tolist() == g[k] == h[k].tolist()
    assert bool((a["win"].double() * 2.0 ** g["e_qa_out"] == a["out"].double()).all())


def test_pla_without_grids_fails_clearly():
    cfg = dict(PLA, hw_grids={})
    with pytest.raises(SystemExit, match="does not calibrate"):
        ALG.run(C.build("conv_prelu"), cfg)


@pytest.mark.parametrize("name", PLA_CASES)
def test_pla_pcu_alone_lib_bins_equal_hw_standin(name):
    """The PCU alone: the lib's bins + window on HW's own accumulator words
    equals the bins stand-in (their pcu per bin) bit for bit."""
    case = C.build(name)
    h = HW.run(case, PLA)
    o, _ = ALG.pcu_on(h["hw_acc"], PLA, case, -int(h["out_scale_log2"]))
    assert bool((o == h["hw_out_raw_bins"]).all())


def test_pla_their_one_bin_differs_only_in_the_lower_bin():
    """Evidence for HW: their one-bin PCU equals the stand-in in the upper bin
    and differs in the lower bin wherever S_0 != S_1."""
    h = HW.run(C.build("conv_prelu"), PLA)
    low = h["hw_pla_bin"] == 0
    d = h["out_raw"].long() != h["hw_out_raw_bins"]
    assert not bool(d[~low].any())
    sc = h["pla_bin_scales"]
    other = (sc[:, 0] != sc[:, 1]).view(1, -1, 1, 1).expand_as(d)
    assert bool(d[low & other].float().mean() > 0.9)
    assert not bool(d[low & ~other].any())


@pytest.mark.parametrize("b", [0.0371, -0.0371])
@pytest.mark.parametrize("tie", ["upper", "lower"])
def test_pla_bins_standin_bit_exact_incl_tie(b, tie):
    """Same acc words (random + bound - 1, bound, bound + 1) into the lib's
    bins and the stand-in: equal for both tie rules and both bias signs, on the
    output grid and on the fine grid where the tie word differs per rule."""
    from fixed_point.mx_fixed_point_hw import (PLAParams, pla_bins, _apply_pla,
                                               _apply_out_window)
    bcal, frac = 16, 6
    for m_neg in (0, 16, 3, 64):
        bd, sc, bi = pla_bins(torch.tensor([64]), torch.tensor([m_neg]),
                              torch.tensor([b], dtype=torch.float64), -bcal, frac)
        pla = PLAParams(torch.tensor([64]), torch.tensor([m_neg]), frac, "bins",
                        bd, sc, bi, tie)
        t = int(bd[0, 0])
        g = torch.Generator().manual_seed(5)
        N = torch.cat([torch.randint(-2 ** 20, 2 ** 20, (2000,), generator=g),
                       torch.tensor([t - 1, t, t + 1])]).to(torch.int64)
        d = {"bounds": bd.tolist(), "scales": sc.tolist(), "biases": bi.tolist(),
             "tie": tie, "frac": frac, "lo": -2.0 ** 23, "hi": 2.0 ** 23 - 1}
        # fine grid: only words that fit the window there (|P| < 2^22; wider
        # words are the known 22- vs 24-bit clamp difference, not the bins)
        small = N[(N.abs() < 2 ** 14) | (N - t).abs().le(1)]
        for e_out, e_hw, n in ((-12, 12 - bcal, N), (-bcal - frac, frac, small)):
            w = _apply_pla(n.view(1, 1, -1), pla)
            a = _apply_out_window(w, -bcal - frac, e_out, 22, "nearest")[0].view(-1)
            h, k = HW.bins_pcu(n, d, 0, e_hw)
            assert bool((a == h).all()), (m_neg, e_out)
        assert int(k[-2]) == (1 if tie == "upper" else 0)   # acc == bound


def test_alg_convt_hw_path_taps():
    """xblock_accum.convT = "hw": main_ALG taps the lib's MXConvTranspose2dHW
    like a conv (compact input blocked along W, equivalent conv weight)."""
    cfg = json.loads(json.dumps(E4))
    cfg["mx_specs"]["xblock_accum"].update(convT="hw", backend="python")
    c = C.build("convt_k4_s2")
    t = ALG.run(c, cfg)
    p = c["params"]
    ref = F.conv_transpose2d(c["x"], c["w"], None, p["stride"], p["padding"],
                             p.get("output_padding", 0))
    assert t["a_int"].shape == c["x"].shape          # quantized compact, not dilated
    assert t["w_int"].shape == c["w"].transpose(0, 1).shape
    assert "acc" in t and float((t["out"] - ref).abs().max()) < 0.05


# --- HW-exact switches (M1-M8): both sides from one config -------------------

HWX = SC.load_config("mx_config_npe_hw_exact.json")
HWX_IN = SC.load_config("mx_config_npe_hw_exact_in.json")


def test_hw_exact_register_width_and_cal_from_config():
    """HW mx_type follows the NPE register (in_exp_bits 4), and their x_cal /
    w_cal come from the config's register bases: cal = 6 - E_min."""
    assert HW.hw_format(HWX) == ("mx-e4bm8", 32)
    g = HWX["hw_grids"]["conv3x3_p0"]
    t = HW.run(C.build("conv3x3_p0"), HWX)
    assert int(t["cal_from_config"]) == 1
    assert (int(t["x_cal"]), int(t["w_cal"])) == (6 - g["e_base_a"], 6 - g["e_base_w"])
    assert -int(t["out_scale_log2"]) == g["e_qa_out"]      # static shift moves the grid


@pytest.mark.parametrize("name", ["conv1x1", "conv3x3_p0", "conv3x3_p0_bias"])
@pytest.mark.parametrize("x_frac", [None, 4])
def test_hw_exact_conv_bit_exact(name, x_frac):
    c = C.build(name, x_frac=x_frac)
    a, h = ALG.run(c, HWX), HW.run(c, HWX)
    assert bool((a["out"] == h["out"]).all())


@pytest.mark.parametrize("name", PLA_CASES)
def test_hw_exact_pla_bit_exact_with_bins_standin(name):
    c = C.build(name)
    a, h = ALG.run(c, HWX), HW.run(c, HWX)
    assert bool((a["out"] == h["out_bins"]).all())
    assert not bool((a["out"] == h["out"]).all())     # their 1 bin: not PReLU


@pytest.mark.parametrize("cfg_name", ["mx_config_npe_hw_exact.json",
                                      "mx_config_npe_hw_exact_nearest.json"])
def test_hw_exact_pcu_on_saturates_as_the_layer(cfg_name):
    """overflow_saturate ['out_window']: the sim's lib PCU alone (pcu_on) on
    the layer's own accumulator words gives the layer's window word, also
    where it saturates (to +-127 * 2^15). Grid 6 bits short so it does."""
    cfg = copy.deepcopy(SC.load_config(cfg_name))
    case = C.build("conv_prelu")
    g = cfg["hw_grids"]["conv_prelu"]
    g["e_qa_out"] -= 6
    g["static_shift"] = -(g["e_qa_out"] + 12)
    g["shift_out"] = g["e_qa_out"] - g["e_layer_min"]
    t = ALG.run(case, cfg)
    sat = t["win_sat"].bool()
    assert bool(sat.any()) and not bool(sat.all())
    assert bool((t["win"][sat].abs() == 127 * 2 ** 15).all())
    o, s = ALG.pcu_on(t["acc"], cfg, case, g["e_qa_out"])
    assert torch.equal(s, sat) and torch.equal(o, t["win"])


def test_hw_exact_first_layer():
    """M8: conv1x1_in bit-exact; with a bias (conv3x3_in) a per-channel
    constant <= 2^-16: the lib's first-layer acc grid (2^e_layer_min, block
    exps kept on the int input) is finer than HW's 2^-(x_cal + w_cal), so the
    bias rounds on another grid."""
    c = C.build("conv1x1_in")
    assert bool((ALG.run(c, HWX_IN)["out"] == HW.run(c, HWX_IN)["out"]).all())
    c = C.build("conv3x3_in")
    a, h = ALG.run(c, HWX_IN), HW.run(c, HWX_IN)
    d = (a["out"].double() - h["out"].double())[0]
    assert float(d.abs().max()) <= 2.0 ** -16
    assert all(d[o].unique().numel() == 1 for o in range(d.shape[0]))


# --- HW warp: their CalibratedMXWarp, wired in (main_HW.warp) -----------------

import subprocess                     # noqa: E402

import main as M                      # noqa: E402
import warp_vectors as WV             # noqa: E402

WARP = C.select("warp")
# the warp cases HW runs: out step = ref step (their PCU only shifts; Q7)
WARP_HW = [n for n in WARP if C.CASES[n]["ref_scale"] == C.CASES[n]["out_scale"]]


def _half_up(cfg):
    """`cfg` with the warp's grid and output fxp rounding half up (ties
    toward +inf), the HW rule (warp Q&A Q5, Q7)."""
    cfg = copy.deepcopy(cfg)
    e = cfg["entries"]["model.warp"]
    for fxp in (e["inputs"][1]["fxp"], e["output"]["fxp"]):
        fxp["round"] = "half_up"
    return cfg


@pytest.mark.parametrize("name", WARP)
def test_hw_warp_runs_their_code(name):
    """Every warp case with a power-of-two out / ref step runs on their
    corrected code, the signed (randn) ones too: their packed reference holds
    the case's int8 codes, the grid word is the relative motion in Q8.8 (the
    config's fraction bits, passed in), the output is their int64 PCU word
    times the out step. warp_subpix (another step ratio) is refused."""
    case = C.build(name)
    s = case["warp"]
    if s["out_scale"] != s["ref_scale"]:             # warp_subpix: 0.0137 / 0.0331
        with pytest.raises(NotImplementedError, match=r"not a power of two.*HW Q7"):
            HW.run(case, E4)
        return
    t = HW.run(case, E4)
    B, Cc, H, W = case["x"].shape
    codes = (case["x"].double() / s["ref_scale"]).round().clamp(-128, 127).long()
    assert torch.equal(t["ref_code"], codes)
    assert torch.equal(t["grid_code"].long(), (case["flow"].double() * 256).long())
    assert t["frac_bits"].tolist() == [8, 8]          # mx_config_npe_e4.json grid fxp
    assert t["hw_acc"].dtype == t["out_code"].dtype == torch.int64
    assert t["hw_acc"].shape == t["out_code"].shape == (B, Cc, H, W)
    assert t["hw_weights"].shape == (H, W, 4) and int(t["out_shift"]) == 0
    assert torch.equal(t["out"], t["out_code"] * s["out_scale"])


def test_warp_hw_cases_are_most_of_them():
    assert len(WARP_HW) == len(WARP) - 1 > 0           # warp_subpix: refused, Q7 open


@pytest.mark.parametrize("name", WARP_HW)
def test_warp_alg_equals_hw_bit_exact_with_the_hw_rounding(name):
    """With the warp's grid and output rounding half up (the HW rule), every
    warp case HW runs is bit-exact, ALG vs their emulator: ref, grid and out
    codes, out, and the accumulator inside and on the border. As shipped
    (half away from zero), only ties of a negative value differ, in out_code;
    the cases without a negative tie are bit-exact as shipped too."""
    case = C.build(name)
    for cfg in (_half_up(E4), E4):
        a, h = ALG.run(case, cfg), HW.run(case, cfg)
        rows = {k: v for k, v, *_ in M.compare(a, h)}
        assert set(rows) >= {"ref_code", "grid_code", "out_code", "out"}
        st = {r["stage"]: r for r in M.warp_stages(case, a, h)}
        assert st["acc interior"]["n_diff"] == st["acc border"]["n_diff"] == 0
        acc = h["hw_acc"].double() / 2 ** 16 * case["warp"]["ref_scale"] / case["warp"]["out_scale"]
        neg_tie = bool(((acc - acc.floor()) == 0.5).logical_and(acc < 0).any())
        bad = [k for k, v in rows.items() if v not in ("bit-exact", "float")]
        if cfg is E4 and neg_tie:
            assert bad == ["out_code", "out"]
            d = a["out_code"].long() - h["out_code"]
            assert set(d.unique().tolist()) == {-1, 0}      # half away: one down
        else:
            assert bad == [], (name, bad)


def test_hw_warp_acc_is_their_int64_psum():
    """hw_acc is bcu's int64 p_sum taken in their loop: with a constant
    reference c and a continuous flow every acc, border included (the
    border replicates the edge), is c * 2^(frx+fry) exactly (their weights
    sum to 2^(frx+fry)); their int8 output is then c."""
    g = torch.Generator().manual_seed(3)
    flow = torch.round((torch.rand(1, 2, 8, 8, generator=g) * 2 - 1) * 1.5 * 256) / 256
    t = HW.run(C.warp_case("c", torch.full((1, 1, 8, 8), -7), flow, C.P2, C.P2), E4)
    assert bool((t["hw_acc"] == -7 * 2 ** 16).all())
    assert bool((t["out_code"] == -7).all())
    assert bool((t["hw_weights"].sum(-1) == 2 ** 16).all())


@pytest.mark.parametrize("name", ["warp_u8_subpix", "warp_u8_subpix_w64", "warp_oob"])
def test_hw_warp_acc_equals_alg_exact(name):
    """Flow on 1/256, frx = fry = 8: on every pixel, border included, their
    acc / 2^16 is the exact bilinear sum of ALG's own codes by the HW border
    rule (cases.Warp in float64, exact on the 1/256 grid)."""
    case = C.build(name)
    a, h = ALG.run(case, E4), HW.run(case, E4)
    want = C.Warp()(a["ref_code"].double(), a["grid_code"].double() / 256)
    got = h["hw_acc"].double() / 2.0 ** 16
    st = {r["stage"]: r for r in M.warp_stages(case, a, h)}
    assert st["acc border"]["n"] > 0 and st["acc interior"]["n"] > 0
    assert torch.equal(got, want)


@pytest.mark.parametrize("H, W", [(5, 32), (4, 40), (3, 7)])
def test_zero_motion_returns_the_reference_last_row_and_col(H, W):
    """Q2: zero motion preserves the reference, its last row and column
    included, at widths that are and are not a multiple of 32: ALG (cases.Warp)
    and HW (their trc) both. Their old trc read the row and column before
    (its clamp to H-2 / W-2)."""
    ref = WV.ramp(H, W, lambda x, y: (13 * y + 3 * x) % 200 - 100)
    case = C.warp_case("z", ref, C.const_flow(1, H, W, 0.0, 0.0), C.P2, C.P2)
    a, h = ALG.run(case, E4), HW.run(case, E4)
    assert torch.equal(a["out_code"].long(), ref.long())
    assert torch.equal(h["out_code"], ref.long())
    assert torch.equal(C.Warp()(ref.double(), torch.zeros(1, 2, H, W)), ref.double())


@pytest.mark.parametrize("fx, fy, at, want", [
    (2.75, 0.0, (1, 4), 61),           # x = 4 + 2.75 > W-1 = 5: the edge pixel ref[1, 5]
    (0.0, 9.5, (0, 2), 33),            # y beyond H-1: ref[3, 2]
    (-0.25, 0.0, (1, 0), 11),          # x < 0: ref[1, 0] (their emulator's rule, Q1 open)
    (-3.0, -3.0, (2, 2), 10),          # both below 0: ref[0, 0]
    (0.5, 0.0, (2, 5), 62),            # x = W-1+0.5: ref[2, 5], no excess fraction kept
    (0.25, 0.0, (2, 4), 54.5)])        # inside: 0.75 * 52 + 0.25 * 62
def test_border_saturates_the_whole_coordinate(fx, fy, at, want):
    """Q1 / Q10: the sample position saturates to [0, W-1] x [0, H-1], with
    no excess fraction, so beyond the edge the sample is the edge pixel. ref
    [y, x] = 10(x+1)+y on 4x6 (the Q&A's own frame). ALG (cases.Warp) and
    HW (their acc / 2^16) both give the value the rule gives, worked out by
    hand; zero outside (the old ALG rule) gives other values beyond the
    frame."""
    ref = WV.ramp(4, 6, lambda x, y: 10 * (x + 1) + y)
    flow = C.const_flow(1, 4, 6, fx, fy)
    y, x = at
    assert float(C.Warp()(ref.double(), flow.double())[0, 0, y, x]) == want
    h = HW.run(C.warp_case("b", ref, flow, C.P2, C.P2), E4)
    assert float(h["hw_acc"][0, 0, y, x]) / 2 ** 16 == want


def test_hw_warp_signed_reference_saturates_to_int8():
    """Q6: the reference is int8 [-128, 127]: negative codes run (the old
    packer clamped them to 0, main_HW refused), codes beyond saturate, on
    both sides the same."""
    ref = torch.tensor([[[[-200, -128, -50, 0, 127, 200]] * 2]])
    case = C.warp_case("s", ref, C.const_flow(1, 2, 6, 0.0, 0.0), C.P2, C.P2)
    a, h = ALG.run(case, E4), HW.run(case, E4)
    want = [-128, -128, -50, 0, 127, 127]
    assert a["ref_code"][0, 0, 0].tolist() == h["ref_code"][0, 0, 0].tolist() == want
    assert h["out_code"][0, 0, 0].tolist() == want


def test_hw_warp_output_ties_toward_plus_inf():
    """Q7: the PCU rounds to nearest, ties toward +inf, and saturates to int8:
    acc 37.5 -> 38, -37.5 -> -37. ALG gives the same with the output fxp
    rounding half up; as shipped (half away) -37.5 -> -38."""
    ref = torch.cat([WV.ramp(2, 40, lambda x, y: 37 + x % 2),
                     WV.ramp(2, 40, lambda x, y: -37 - x % 2)], 1)
    case = C.warp_case("t", ref, C.const_flow(1, 2, 40, 0.5, 0.0), C.P2, C.P2)
    h = HW.run(case, E4)
    assert h["hw_acc"][0, :, 0, 0].tolist() == [37.5 * 2 ** 16, -37.5 * 2 ** 16]
    assert h["out_code"][0, :, 0, 0].tolist() == [38, -37]
    assert ALG.run(case, _half_up(E4))["out_code"][0, :, 0, 0].tolist() == [38, -37]
    assert ALG.run(case, E4)["out_code"][0, :, 0, 0].tolist() == [38, -38]


def test_hw_warp_output_shift_from_the_steps():
    """Their PCU shifts the output by out_cal = log2(ref step / out step):
    out step twice the ref step halves the code (round half up), a quarter
    of it quadruples it, saturated to int8."""
    ref = WV.ramp(2, 8, lambda x, y: 10 * x - 35)            # -35, -25, ..., 35
    flow = C.const_flow(1, 2, 8, 0.0, 0.0)
    h = HW.run(C.warp_case("h", ref, flow, C.P2, 2 * C.P2), E4)
    assert int(h["out_shift"]) == -1
    r = ref[0, 0, 0].double()
    assert h["out_code"][0, 0, 0].tolist() == torch.floor(r / 2 + 0.5).tolist()
    h = HW.run(C.warp_case("q", ref, flow, C.P2, C.P2 / 4), E4)
    assert int(h["out_shift"]) == 2
    assert h["out_code"][0, 0, 0].tolist() == (r * 4).clamp(-128, 127).tolist()


def test_hw_grid_word_rounds_ties_up_and_saturates():
    """The grid word, as the producing layer's PCU gives it (Q5, Q7): nearest,
    ties toward +inf, saturated to the signed word."""
    flow = torch.tensor([0.5, -0.5, 1.5, -1.5, 0.25, 200.0, -200.0]).div(256).view(1, 1, 1, 7)
    flow = torch.cat([flow, flow * 256 * 256 / 64], 1)       # y: past +-128 px
    q = HW.grid_word(flow, (8, 8), 16)
    assert q.dtype == torch.int16
    assert q[0, 0, 0].tolist() == [1, 0, 2, -1, 0, 200, -200]
    assert q[0, 1, 0].tolist()[-2:] == [32767, -32768]


def test_hw_warp_refuses_what_their_code_does_not_take():
    """More than one frame (their forward raises, Q8: one frame per call,
    native batch not needed), a grid of another size than the reference (Q10),
    a non-finite reference or flow, grid fraction bits outside 0..8 (Q3), a
    grid wider than int16, a reference or output that is not signed 8-bit, a
    non power-of-two out / ref step (Q7: the multiplier format is open). Each
    is a named NotImplementedError (a refusal), never their ValueError (a
    failure). A grid fxp with a `scale` stops the run: the lib would use the
    scale, main_HW feeds their code frac_bits, so the sides would differ."""
    ok = torch.randint(-128, 128, (2, 1, 3, 8), generator=torch.Generator().manual_seed(1))
    f = C.const_flow(2, 3, 8, 0.5, 0.0)
    with pytest.raises(NotImplementedError, match=r"batch 2.*one\s+frame.*HW Q8"):
        HW.run(C.warp_case("b", ok, f, C.P2, C.P2), E4)          # even with equal flows
    ref = ok[:1].double().clone()
    ref[0, 0, 0, 0] = float("nan")
    with pytest.raises(NotImplementedError, match="non-finite reference"):
        HW.run(C.warp_case("n", ref, C.const_flow(1, 3, 8, 0, 0), C.P2, C.P2), E4)
    nan_flow = C.const_flow(1, 3, 8, 0, 0)
    nan_flow[0, 1, 0, 0] = float("inf")
    with pytest.raises(NotImplementedError, match="non-finite flow"):
        HW.run(C.warp_case("n", ok[:1], nan_flow, C.P2, C.P2), E4)
    with pytest.raises(NotImplementedError, match=r"same-size grid \(HW Q10\)"):
        HW.run(C.warp_case("g", ok[:1], C.const_flow(1, 3, 4, 0, 0), C.P2, C.P2), E4)

    def cfg_with(i, **fxp):
        cfg = copy.deepcopy(E4)
        e = cfg["entries"]["model.warp"]
        (e["inputs"][i] if i is not None else e["output"])["fxp"].update(fxp)
        return cfg
    z = C.warp_case("z", ok[:1], C.const_flow(1, 3, 8, 0, 0), C.P2, C.P2)
    for cfg, why in ((cfg_with(1, frac_bits=9), r"9 fraction bits.*0\.\.8 per axis \(HW Q3\)"),
                     (cfg_with(1, total_bits=32), "takes an int16 word"),
                     (cfg_with(0, total_bits=16), "ref is signed 16-bit"),
                     (cfg_with(0, signed=False), "ref is unsigned 8-bit"),
                     (cfg_with(None, total_bits=16), r"output fxp is 16-bit signed.*int8 \(HW Q7\)")):
        with pytest.raises(NotImplementedError, match=why):
            HW.run(z, cfg)
    cfg = copy.deepcopy(E4)
    cfg["entries"]["model.warp"]["output"]["fxp"] = True        # the lib's defaults: 16-bit
    with pytest.raises(NotImplementedError, match="output fxp is 16-bit"):
        HW.run(z, cfg)
    cfg["entries"]["model.warp"]["output"]["fxp"] = False       # off: their int8 runs
    assert "out_code" in HW.run(z, cfg)
    with pytest.raises(NotImplementedError, match="not a power of two"):
        HW.run(C.warp_case("s", ok[:1], C.const_flow(1, 3, 8, 0, 0), C.P2, 0.7 * C.P2), E4)
    with pytest.raises(SystemExit, match=r"grid has no frac_bits \(or a scale\)"):
        HW.run(z, cfg_with(1, scale=1 / 256))


def test_hw_warp_grid_format_from_config_and_reported():
    """frac_bits are the config's grid fxp at call site 0, passed to their
    pack_warp_grid (0..8 per axis, Q3); frx + fry = 16 is legal (Q4) and the
    2^16 weights it makes are named (encoding open); call sites reported."""
    def cfg_with(**grid):
        cfg = copy.deepcopy(E4)
        cfg["entries"]["model.warp"]["inputs"][1]["fxp"].update(grid)
        return cfg
    z = C.warp_case("z", torch.zeros(1, 1, 4, 40), C.const_flow(1, 4, 40, 0, 0), C.P2, C.P2)
    t = HW.run(z, cfg_with(frac_bits=6))
    assert t["frac_bits"].tolist() == [6, 6]
    assert bool((t["hw_weights"].sum(-1) == 2 ** 12).all())
    notes = " ".join(HW.warp_notes(t))
    assert "HW Q4" not in notes and "call_sites 3" in notes
    t = HW.run(z, E4)
    assert "160/640 TRC weights are 2^16" in " ".join(HW.warp_notes(t))
    assert int(HW.run(z, HWX)["call_sites"]) == 2
    cfg = copy.deepcopy(E4)                          # per-site list: site 0 is used
    g = cfg["entries"]["model.warp"]["inputs"][1]["fxp"]
    cfg["entries"]["model.warp"]["inputs"][1]["fxp"] = [dict(g, frac_bits=f) for f in (6, 7, 5)]
    t = HW.run(z, cfg)
    assert t["frac_bits"].tolist() == [6, 6] and int(t["call_site"]) == 0
    cfg = cfg_with(frac_bits=None)
    with pytest.raises(SystemExit, match="grid has no frac_bits"):
        HW.run(z, cfg)


def test_fxp_half_up_round_mode_is_opt_in():
    """fxp round 'half_up': nearest, ties toward +inf (the NPE PCU rule);
    the default stays half away from zero; the bias helper shares it."""
    from fixed_point.fxp_quant import OUT_QUANT_DEFAULTS, fake_quant_fxp, normalize_out_quant
    from fixed_point.mx_fixed_point_hw import _bias_to_grid
    v = torch.tensor([37.5, -37.5, 2.5, -2.5, 0.49, -0.51, 200.0, -200.0])
    q = fake_quant_fxp(v, frac_bits=0, total_bits=8, round_mode="half_up")
    assert q.tolist() == [38, -37, 3, -2, 0, -1, 127, -128]
    assert fake_quant_fxp(v, frac_bits=0, total_bits=8).tolist() == [38, -38, 3, -3, 0, -1,
                                                                      127, -128]
    assert OUT_QUANT_DEFAULTS["round"] == "half_away"
    assert normalize_out_quant({"round": "half_up"})["round"] == "half_up"
    with pytest.raises(ValueError, match="out_quant.round must be in"):
        normalize_out_quant({"round": "half_down"})
    assert _bias_to_grid(v, 0, True, "cpu").tolist() == [38, -37, 3, -2, 0, -1, 200, -200]


from mx_layers_act import MXActQuant   # noqa: E402


def test_hw_warp_per_site_list_of_wrong_length_refused():
    """A per-site fxp list must hold one spec per call site: the HW side
    refuses a short one, as the lib does, instead of indexing it."""
    cfg = copy.deepcopy(E4)
    e = cfg["entries"]["model.warp"]
    g = e["inputs"][1]["fxp"]
    e["inputs"][1]["fxp"] = [dict(g, frac_bits=f) for f in (6, 7)]   # call_sites 3
    with pytest.raises(ValueError, match="2 per-call-site fxp configs for call_sites=3"):
        HW.warp_spec(cfg)
    with pytest.raises(ValueError, match="2 per-call-site fxp configs for call_sites=3"):
        MXActQuant(torch.nn.Identity(), [None, None],
                   fxp_per_input=[None, e["inputs"][1]["fxp"]], call_sites=3)
    e["call_sites"] = 0
    with pytest.raises(ValueError, match="call_sites must be an int >= 1"):
        HW.warp_spec(cfg)


# --- ALG warp on every shape of warp entry the configs hold -----------------

DOF_RUN = SC.load_config("mx_config_dof_run.json")         # ref fxp: per-site list
# HWX (mx_config_npe_hw_exact.json): its warp entry has no output


def test_alg_warp_reads_a_per_site_ref_list_by_the_lib_rule():
    """dof_run's ref fxp is one spec per call site (call_sites 3): ALG runs
    call site 0 with the case's static scale in place of the file's, and a
    list of the wrong length is refused by the lib's rule (fxp_sites)."""
    assert isinstance(DOF_RUN["entries"]["model.warp"]["inputs"][0]["fxp"], list)
    case = C.build("warp_u8_half")
    t = ALG.run(case, DOF_RUN)
    codes = (case["x"].double() / case["warp"]["ref_scale"]).round()
    assert torch.equal(t["ref_code"].double(), codes)
    bad = copy.deepcopy(DOF_RUN)
    ref = bad["entries"]["model.warp"]["inputs"][0]
    ref["fxp"] = ref["fxp"][:2]
    with pytest.raises(ValueError, match="2 per-call-site fxp configs for call_sites=3"):
        ALG.run(case, bad)


@pytest.mark.parametrize("cfg", [DOF_RUN, HWX], ids=["dof_run", "hw_exact"])
def test_alg_warp_without_output_fxp_is_unrounded_as_the_lib(cfg):
    """A warp entry with no output: the lib passes the bilinear result
    through (MXActQuant.quant_output), so ALG out is out_pre, off the
    out_scale lattice, with no out_code, and the sim says why."""
    assert "output" not in cfg["entries"]["model.warp"]
    case = C.build("warp_subpix")              # out_scale 0.0137
    t = ALG.run(case, cfg)
    assert "out_code" not in t
    assert torch.equal(t["out"], t["out_pre"])
    v = t["out"].double() / case["warp"]["out_scale"]
    assert bool(((v - v.round()).abs() > 1e-3).any())
    assert "no output fxp" in ALG.warp_out_note(cfg)
    assert ALG.warp_out_note(E4) is None
    assert "out_code" in ALG.run(case, E4)


def test_alg_warp_non_fxp_output_has_no_out_code():
    """An MX (non-fxp) output: the lib holds no fixed-point output word, so
    ALG taps no out_code and says so; no format is invented."""
    cfg = copy.deepcopy(E4)
    cfg["entries"]["model.warp"]["output"] = {"axes": [1]}   # global mx_specs
    t = ALG.run(C.build("warp_u8_half"), cfg)
    assert "out_code" not in t
    assert "output is not fxp" in ALG.warp_out_note(cfg)


@pytest.mark.parametrize("fxp", [False, None, {"enabled": False}, [{"enabled": False}] * 3],
                         ids=["false", "null", "enabled_false", "per_site_off"])
def test_alg_warp_output_fxp_off_is_unrounded_as_the_lib(fxp):
    """An output fxp the lib reads as off (normalize_out_quant enabled False):
    the lib passes the warp result through, as with no output, so ALG out is
    out_pre, off the out_scale lattice, with no out_code, and the sim says
    why."""
    cfg = copy.deepcopy(E4)                    # warp call_sites 3
    cfg["entries"]["model.warp"]["output"]["fxp"] = fxp
    case = C.build("warp_subpix")
    t = ALG.run(case, cfg)
    assert "out_code" not in t
    assert torch.equal(t["out"], t["out_pre"])
    v = t["out"].double() / case["warp"]["out_scale"]
    assert bool(((v - v.round()).abs() > 1e-3).any())
    assert "output fxp is off" in ALG.warp_out_note(cfg)


@pytest.mark.parametrize("bad", [[], "two"], ids=["empty", "two_of_three"])
def test_alg_warp_output_site_list_of_wrong_length_refused_by_the_lib_rule(bad):
    """warp_out_note reads the output's call site by the lib's rule
    (sim_config.site_fxp): a list that is not one spec per call site is the
    lib's ValueError, not a crash of its own."""
    cfg = copy.deepcopy(E4)                    # warp call_sites 3
    out = cfg["entries"]["model.warp"]["output"]
    out["fxp"] = [] if bad == [] else [out["fxp"]] * 2
    n = len(out["fxp"])
    with pytest.raises(ValueError, match=f"{n} per-call-site fxp configs for call_sites=3"):
        ALG.warp_out_note(cfg)


@pytest.mark.parametrize("operand, shape", [
    ("output", "on_none_none"), ("output", "on_false_false"), ("input0", "on_none_none"),
    ("output", "true"), ("input0", "true")])
def test_alg_warp_off_sites_and_true_run_as_the_lib(operand, shape):
    """Shapes the lib accepts: a per-site list whose other sites are off
    (null / false) runs call site 0 as the single spec does, and `true` (the
    lib's defaults) runs with the case's scale; no crash in _fix_scale."""
    cfg = copy.deepcopy(E4)                    # warp call_sites 3
    e = cfg["entries"]["model.warp"]
    op = e["output"] if operand == "output" else e["inputs"][0]
    on = op["fxp"]
    op["fxp"] = {"on_none_none": [on, None, None], "on_false_false": [on, False, False],
                 "true": True}[shape]
    case = C.build("warp_u8_half")
    t = ALG.run(case, cfg)
    assert "out_code" in t and ALG.warp_out_note(cfg) is None
    if shape == "true":
        codes = (case["x"].double() / case["warp"]["ref_scale"]).round()
        assert operand == "output" or torch.equal(t["ref_code"].double(), codes)
        return
    ref = ALG.run(case, E4)
    assert all(torch.equal(t[k], ref[k]) for k in ref)


def test_t12_output_requant_q7_without_output_fxp():
    """D7 on a config with no output fxp: ALG gives the acc unrounded, off
    the HW answer, which their PCU gives."""
    (v,) = [x for x in WV.vectors(HWX) if x["label"] == "D7 | HW Q7"]
    assert (v["acc"], v["alg"], v["hw"], v["want"]) == ([37.5, -37.5], [37.5, -37.5],
                                                        [38, -37], [38, -37])
    assert "no output fxp" in v["text"]


# --- hw_vs_alg_check T12: the warp vectors, against the HW answers ------------

T12 = WV.vectors(E4)


def _t12(label):
    v = [x for x in T12 if x["label"] == label]
    assert v, f"no T12 vector {label}"
    return v


def test_t12_covers_every_hw_question():
    labels = [v["label"].split(" | ") for v in T12]
    assert labels
    assert {q for _, q in labels} == {f"HW Q{i}" for i in range(1, 11)}
    assert {d for d, _ in labels} == {f"D{i}" for i in range(1, 10)}
    assert all(v["status"] == WV.RESOLVED or v["status"].startswith("open: ") for v in T12)


# Each T12 vector's HW answer, worked out here from the answer's own words in
# hw/warp_hw_questions_and_answers.md (never from their code), and the open
# ones by label. Their emulator must give every resolved answer.
OPEN = {"D1 | HW Q1": 1, "D3 | HW Q4": 1, "D6 | HW Q5": 1}


def test_t12_open_ones_are_the_questions_the_answers_leave_open():
    got = {}
    for v in T12:
        if v["status"] != WV.RESOLVED:
            got[v["label"]] = got.get(v["label"], 0) + 1
    assert got == OPEN
    off_alg, off_hw = WV.gaps(T12)
    assert off_hw == []                                        # their emulator: every answer
    assert [v["label"] for v in off_alg] == ["D4 | HW Q5", "D7 | HW Q7"]   # E4: half away
    off_alg, off_hw = WV.gaps(WV.vectors(_half_up(E4)))
    assert off_alg == off_hw == []
    # the DOF HW config (owner and co-manager 2026-11-17): warp ref, grid,
    # output and model out_quant half up: no resolved vector off
    off_alg, off_hw = WV.gaps(WV.vectors(SE4_HW))
    assert off_alg == off_hw == []


SE4_HW = SC.load_config("infer_dof/se4_warp_convT_hw.json")


@pytest.mark.parametrize("name", WARP_HW)
def test_warp_bit_exact_on_the_dof_hw_config(name):
    """infer_dof/se4_warp_convT_hw.json rounds the warp grid and output half
    up (2026-11-17): every warp case HW runs is bit-exact, ALG vs their
    emulator, negative output ties included."""
    case = C.build(name)
    a, h = ALG.run(case, SE4_HW), HW.run(case, SE4_HW)
    bad = [k for k, v, *_ in M.compare(a, h) if v not in ("bit-exact", "float")]
    assert bad == [] and "out_code" in a


def test_t12_border_q1():
    """Q1: the complete coordinate saturates, no excess fraction kept.
    ref[y,x] = 10(x+1)+y on 4x6: x = 2, y = H-1+0.25 -> ref[3, 2] = 33 (the
    Q&A's 'Replicate' column); x = -0.25, y = 1 -> 11, the near edge, open."""
    left, bottom = _t12("D1 | HW Q1")
    assert (bottom["want"], bottom["alg"], bottom["hw"], bottom["status"]) == (
        33, 33, 33, WV.RESOLVED)
    assert (left["want"], left["alg"], left["hw"]) == (11, 11, 11)
    assert left["status"].startswith("open: Q1")


def test_t12_identity_q2():
    """Q2: zero motion preserves the reference: last row [24, 27, 30, 33]
    (the Q&A's own example), last column of 2x32 at 3 * 31 = 93."""
    row, col = _t12("D1 | HW Q2")
    assert row["want"] == row["alg"] == row["hw"] == [24, 27, 30, 33]
    assert col["want"] == col["alg"] == col["hw"] == 93


def test_t12_fraction_bits_q3():
    """Q3: relative motion in the supplied format, the config's 8 bits per
    axis: no grid word off flow * 2^8, and the interior acc equal."""
    (v,) = _t12("D2 | HW Q3")
    assert (v["want"], v["alg"], v["hw"], v["frac"]) == (0, 0, 0, [8, 8])
    assert v["acc"]["n"] > 0 and v["acc"]["n_diff"] == 0


def test_t12_frac_sum_q4_records_the_open_endpoint():
    """Q4: 8 + 8 = 16 is legal; zero flow makes every w00 2^16, which their
    int64 weight holds; the field's encoding of 2^16 is open."""
    (v,) = _t12("D3 | HW Q4")
    assert (v["frac_sum"], v["hw"], v["n"], v["want"]) == (16, 8, 8, None)
    assert v["status"].startswith("open: Q4")


def test_t12_ties_q5():
    """Q5 + Q7: the grid word comes from the producer's PCU, ties toward +inf:
    +0.5/256 -> +1, -0.5/256 -> 0; ALG as shipped rounds half away (-1). The
    reference float-to-code tie is open: ALG half away, their packer half to
    even."""
    g_pos, g_neg, ref = _t12("D4 | HW Q5") + _t12("D6 | HW Q5")
    assert (g_pos["tie"], g_pos["want"], g_pos["alg"], g_pos["hw"]) == (0.5, 1, 1, 1)
    assert (g_neg["tie"], g_neg["want"], g_neg["alg"], g_neg["hw"]) == (-0.5, 0, -1, 0)
    assert (ref["alg"], ref["hw"], ref["want"]) == ([3, 4, 4], [2, 4, 4], None)
    assert ref["status"].startswith("open: Q5")


def test_t12_reference_format_q6():
    """Q6: int8 [-128, 127]: [-50, 0, 127, 200] -> [-50, 0, 127, 127]; main_HW
    now runs a signed reference."""
    (v,) = _t12("D5 | HW Q6")
    assert v["want"] == v["alg"] == v["hw"] == [-50, 0, 127, 127]
    assert v["refusal"] == "ran"


def test_t12_output_requant_q7():
    """Q7: acc 37.5 -> 38, -37.5 -> -37 (the Q&A's own example)."""
    (v,) = _t12("D7 | HW Q7")
    assert (v["acc"], v["want"], v["hw"], v["alg"]) == ([37.5, -37.5], [38, -37], [38, -37],
                                                        [38, -38])


def test_t12_batch_q8():
    """Q8: one frame per call. ALG moves batch 1; their code raises, main_HW
    refuses with a name."""
    (v,) = _t12("D8 | HW Q8")
    ref1 = v["ref"][1, 0, 0].double()
    assert v["alg_out"] == ref1[1:5].tolist()
    assert (v["want"], v["hw"]) == ("refused", "refused")
    assert v["theirs"].startswith("raise ValueError")


def test_t12_integer_store_q9():
    """Q9: integer types through the PCU: their int8 output is the PCU rule
    of their own int64 acc on every pixel."""
    (v,) = _t12("D9 | HW Q9")
    assert (v["want"], v["hw"], v["dtypes"]) == (0, 0, "torch.int64, torch.int64")


def test_t12_padding_q10():
    """Q10: no padding: x = W-1+0.5 on a constant 100 is 100 at W = 40 and 32
    (the delivered emulator gave 50 at W = 40)."""
    w40, w32 = _t12("D1 | HW Q10")
    assert (w40["W"], w40["want"], w40["alg"], w40["hw"]) == (40, 100, 100, 100)
    assert (w32["W"], w32["want"], w32["alg"], w32["hw"]) == (32, 100, 100, 100)


def test_hw_vs_alg_check_prints_the_t12_vectors():
    """--only warp prints every vector's own text (built from its values),
    then the vectors off the HW answer and the open ones, by name."""
    p = subprocess.run([sys.executable, "hw_vs_alg_check.py", "--only", "warp"],
                       cwd=os.path.join(_ROOT, "configs"), capture_output=True,
                       text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    out = p.stdout.splitlines()
    lines = [f"  [{v['label']}] {v['text']}" for v in T12]
    assert lines and all(ln in out for ln in lines), p.stdout[-2000:]
    assert out[-3:] == ["  ALG off the HW answer: D4 | HW Q5, D7 | HW Q7",
                        "  their emulator off the HW answer: none",
                        "  open in the HW answers: D1 | HW Q1, D3 | HW Q4, D6 | HW Q5"]


def test_hw_vs_alg_check_runs_on_hw_exact():
    """The script's HW-exact mode runs T1-T12 to the end; that config's warp
    entry has no output fxp, so D7 shows ALG's unrounded value."""
    p = subprocess.run([sys.executable, "hw_vs_alg_check.py", "--config",
                        "mx_config_npe_hw_exact.json"],
                       cwd=os.path.join(_ROOT, "configs"), capture_output=True,
                       text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    assert "### T11" in p.stdout and "T12 skipped" not in p.stdout
    assert "(config has no output fxp)" in p.stdout, p.stdout[-2000:]


# --- CLIs over many cases: a refusal is reported, the run goes on -----------

def _cli(script, *args):
    return subprocess.run([sys.executable, script, *args],
                          cwd=os.path.join(_ROOT, "configs"), capture_output=True,
                          text=True, timeout=600)


def _refused_names(summary):
    """Case names after 'refused: ' in a CLI summary line."""
    part = summary.split("refused: ", 1)[1].split(", FAILED")[0]
    return part.split(", HW missing")[0].split(", ")


def test_main_hw_cli_reports_each_refusal_and_goes_on():
    """-m all on npe_e4: conv3x3 (padding) is refused by their code, and the
    run goes on to the end; every refusal is named with its reason and in
    the summary, never counted as ran; refusals and no failure: exit 2."""
    names = C.select("all")
    p = _cli("main_HW.py", "-m", "all", "--config", "mx_config_npe_e4.json")
    assert p.returncode == 2, p.stdout[-2000:] + p.stderr[-2000:]
    out = p.stdout.splitlines()
    assert "  refused: Padding not yet supported" in out
    assert "=== warp_u8_subpix_w64  HW ===" in out          # the last case ran
    refused = _refused_names(out[-1])
    assert "conv3x3" in refused and "conv1x1" not in refused
    assert out[-1].startswith(f"{len(names) - len(refused)}/{len(names)} ran")
    assert sum(ln.startswith("  refused: ") for ln in out) == len(refused)


def test_main_hw_cli_fails_on_an_error_that_is_not_a_refusal(monkeypatch, capsys):
    """An error other than a documented refusal is reported, the run goes
    on, and the exit code is 1."""
    real = HW.run

    def run(case, cfg=None):
        if case["name"] == "conv1x1":
            raise RuntimeError("boom")
        return real(case, cfg)

    monkeypatch.setattr(HW, "run", run)
    monkeypatch.setattr(sys, "argv", ["main_HW.py", "-m", "conv", "--config",
                                      "mx_config_npe_e4.json"])
    with pytest.raises(SystemExit) as ex:
        HW.main()
    out = capsys.readouterr().out.splitlines()
    assert ex.value.code == 1
    assert "  FAILED: RuntimeError: boom" in out
    assert "=== conv3x3_p0  HW ===" in out                    # went on
    assert ", FAILED: conv1x1" in out[-1]


def test_main_alg_cli_reports_each_refusal_and_goes_on(tmp_path):
    """main_ALG -m all on npe_hw_exact: the cases it holds no grids for are
    refused, with why, the run goes on to the last case, the summary names
    each, and the exit code is 2 (refused, nothing failed). --dump holds the
    cases that ran."""
    names = C.select("all")
    dump = tmp_path / "taps.pt"
    p = _cli("main_ALG.py", "-m", "all", "--config", "mx_config_npe_hw_exact.json",
             "--dump", str(dump))
    assert p.returncode == 2, p.stdout[-2000:] + p.stderr[-2000:]
    out = p.stdout.splitlines()
    assert "  refused: mx_config_npe_hw_exact.json: no hw_grids entry 'conv3x3' " \
           "and xblock_accum has no e_layer_min/e_qa_out, but out_window is on. " \
           "The sim does not calibrate: write the grids into the config " \
           "(gen_hw_grids.py)." in out
    assert "=== warp_u8_subpix_w64  ALG ===" in out          # the last case ran
    note = ("  ALG out: warp entry 'model.warp' has no output fxp; the lib leaves the "
            "warp output unrounded (float), so there is no out_code")
    assert out.count(note) == sum(C.CASES[n]["op"] == "warp" for n in names) > 0
    summary = out[-3]                          # then the --dump line
    refused = _refused_names(summary)
    assert {"conv3x3", "convt_k3_s2"} <= set(refused) and "conv1x1" not in refused
    assert summary.startswith(f"{len(names) - len(refused)}/{len(names)} ran")
    assert out[-1] == f"wrote {dump} ({len(names) - len(refused)} cases that ran)"
    assert sum(ln.startswith("  refused: ") for ln in out) == len(refused)
    ran = torch.load(dump)
    assert set(ran) == set(names) - set(refused)


def test_main_alg_cli_fails_on_an_error_that_is_not_a_refusal(monkeypatch, capsys):
    """An error other than a documented refusal is reported, the run goes
    on, and the exit code is 1, also beside refusals."""
    real = ALG.run

    def run(case, cfg=None):
        if case["name"] == "conv1x1":
            raise RuntimeError("boom")
        return real(case, cfg)

    monkeypatch.setattr(ALG, "run", run)
    monkeypatch.setattr(sys, "argv", ["main_ALG.py", "-m", "conv", "--config",
                                      "mx_config_npe_hw_exact.json"])
    with pytest.raises(SystemExit) as ex:
        ALG.main()
    out = capsys.readouterr().out.splitlines()
    assert ex.value.code == 1
    assert "  FAILED: RuntimeError: boom" in out
    assert "=== conv3x3_p0  ALG ===" in out                   # went on
    assert ", refused: conv3x3" in out[-1] and ", FAILED: conv1x1" in out[-1]


@pytest.mark.parametrize("script", ["main.py", "main_HW.py", "main_ALG.py",
                                    "hw_vs_alg_check.py"])
def test_cli_exits_2_on_a_config_it_refuses(script):
    """A config the sim refuses at load is said, with why, and exits 2 in
    every CLI (sim_cli): refused, nothing failed - never 0, never a crash."""
    p = _cli(script, "--config", "mx_config_auto_mixed.json")
    assert p.returncode == 2, p.stdout[-2000:] + p.stderr[-2000:]
    assert p.stdout.splitlines()[-1] == (
        "refused: mx_config_auto_mixed.json: no global mx_specs (per-layer groups "
        "only); the sim runs one layer on the file's mx_specs")
    assert "Traceback" not in p.stderr


def test_run_each_refuses_an_empty_case_list():
    with pytest.raises(ValueError, match="no cases to run"):
        CLI.run_each([], "ALG", lambda name: None)


def test_gen_hw_grids_leaves_no_tmp_file_on_a_refused_base(tmp_path):
    """A base the sim refuses (no element formats in mx_specs) stops
    gen_hw_grids with why, and its temporary config is removed."""
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"mx_specs": {"xblock_accum": {}}, "layers": []}))
    out = tmp_path / "out.json"
    p = _cli("gen_hw_grids.py", "--base", str(base), "--out", str(out),
             "--cases", "conv1x1")
    assert p.returncode == 1
    assert "mx_specs has no a_elem_format, w_elem_format" in p.stderr
    assert sorted(f.name for f in tmp_path.iterdir()) == ["base.json"]


def test_main_cli_reports_each_refusal_and_goes_on():
    """main.py -m all on npe_e4: the input-layer and PLA cases are refused
    (no grids in that config), the run goes on to the warp, the refusals are
    named, and the exit code says a case did not run (2), never a pass. The
    warp cases with a negative output tie mismatch (that config rounds half
    away, HW half up: Q7), which outranks the refusals: exit 1."""
    p = _cli("main.py", "-m", "all", "--config", "mx_config_npe_e4.json")
    assert p.returncode == 1, p.stdout[-2000:] + p.stderr[-2000:]
    out = p.stdout.splitlines()
    assert ", mismatched: warp_half_pixel, warp_neg_frac, warp_oob," in out[-1]
    assert any(ln.startswith("  HW refused: mx_config_npe_e4.json: input layer needs")
               for ln in out)
    assert "=== warp_u8_subpix_w64 ===" in out
    refused = _refused_names(out[-1])
    assert {"conv1x1_in", "conv3x3_in", "conv_prelu"} <= set(refused)
    p = _cli("main.py", "-m", "conv1x1_in", "--config", "mx_config_npe_e4.json")
    assert p.returncode == 2, p.stdout[-2000:]                # refused alone: 2
    assert p.stdout.splitlines()[-1] == "0/1 bit-exact, refused: conv1x1_in"


def test_main_cli_prints_the_warp_out_note():
    """main.py on a warp case of a config with no output fxp says why ALG
    has no out_code (warp_out_note), in its per-stage warp report."""
    p = _cli("main.py", "-m", "warp_u8_half", "--config", "mx_config_npe_hw_exact.json")
    assert p.returncode == 3, p.stdout[-2000:] + p.stderr[-2000:]  # out: float diff
    assert ("    ALG out: warp entry 'model.warp' has no output fxp; the lib leaves the "
            "warp output unrounded (float), so there is no out_code") in p.stdout.splitlines()


def test_main_cli_exit_3_on_float_differences_only():
    """Every integer tap equal, a float tap differs: exit 3, not 0 and not 1."""
    p = _cli("main.py", "-m", "conv1x1", "--config", "mx_config_npe_e4.json")
    assert p.returncode == 3, p.stdout[-2000:] + p.stderr[-2000:]
    assert p.stdout.splitlines()[-1] == "0/1 bit-exact, float-diff: conv1x1"


def test_hw_vs_alg_check_fails_on_an_error_that_is_not_a_refusal(monkeypatch):
    """A section's error that is not a documented refusal is not a skip: it
    propagates (exit 1), so a bug never reads as "skipped"."""
    import runpy
    real = HW.run

    def run(case, cfg=None):
        if case["name"] == "conv3x3_p0":
            raise RuntimeError("boom")
        return real(case, cfg)

    monkeypatch.setattr(HW, "run", run)
    monkeypatch.chdir(os.path.join(_ROOT, "configs"))
    monkeypatch.setattr(sys, "argv", ["hw_vs_alg_check.py", "--config", "mx_config_npe_e4.json"])
    with pytest.raises(RuntimeError, match="boom"):
        runpy.run_path(os.path.join(_ROOT, "configs", "hw_vs_alg_check.py"), run_name="__main__")


def test_main_cli_fails_on_an_error_that_is_not_a_refusal(monkeypatch, capsys):
    real = ALG.run

    def run(case, cfg=None):
        if case["name"] == "conv1x1":
            raise RuntimeError("boom")
        return real(case, cfg)

    monkeypatch.setattr(M.ALG, "run", run)
    monkeypatch.setattr(sys, "argv", ["main.py", "-m", "conv", "--config",
                                      "mx_config_npe_e4.json"])
    with pytest.raises(SystemExit) as ex:
        M.main()
    out = capsys.readouterr().out.splitlines()
    assert ex.value.code == 1
    assert "  ALG FAILED: RuntimeError: boom" in out
    assert "=== conv3x3_p0 ===" in out
    assert ", FAILED: conv1x1" in out[-1]


@pytest.mark.parametrize("cfg_name", ["mx_config_npe_e4_pla.json",
                                      "mx_config_npe_hw_exact_in.json"])
def test_hw_vs_alg_check_skips_sections_without_grids_and_goes_on(cfg_name):
    """These configs hold grids for their own cases only (PLA / first layer),
    not for the conv cases T1-T10 run on; the sim never calibrates. Each such
    section says it is skipped and why, the rest (T6, T9, T11, T12) runs, the
    last line names the skipped ones, and the exit code is 2, not a pass."""
    p = _cli("hw_vs_alg_check.py", "--config", cfg_name)
    assert p.returncode == 2, p.stdout[-2000:] + p.stderr[-2000:]
    out = p.stdout.splitlines()
    assert any(ln.startswith(f"  T1 T7 skipped on {cfg_name}: no hw_grids "
                             f"entry 'conv3x3_p0'") for ln in out)
    for t in ("### T6", "### T9", "### T11", "### T12"):
        assert any(ln.startswith(t) for ln in out), t
    assert any(ln.startswith("  [D7 | HW Q7]") for ln in out)
    assert out[-1] == f"skipped on {cfg_name}: T1 T7, T3, T4, T5, T8, T10"


def test_hw_vs_alg_check_exit_0_only_when_every_section_ran():
    p = _cli("hw_vs_alg_check.py", "--config", "mx_config_npe_e4.json")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "skipped" not in p.stdout
    p = _cli("hw_vs_alg_check.py", "--config", "mx_config_npe_cpu.json")
    assert p.returncode == 2
    assert p.stdout.splitlines()[-1] == "skipped on mx_config_npe_cpu.json: T12"


# --- configs the sim does not model are refused, with why -------------------

@pytest.mark.parametrize("cfg_name, why", [
    ("mx_config_auto_mixed.json", "no global mx_specs"),
    ("mx_config_auto_ladder.json", "mx_specs has no a_elem_format, w_elem_format"),
])
def test_load_config_refuses_a_file_without_global_formats(cfg_name, why):
    with pytest.raises(SystemExit, match=f"{cfg_name}: {why}"):
        SC.load_config(cfg_name)


@pytest.mark.parametrize("cfg_name, layer, blk", [
    ("mx_config_int8.json", "Conv2d", "channel / channel"),     # no xblock_accum
    ("mx_config_int8_hw.json", "MXConv2dHW", "channel / channel"),
])
def test_alg_conv_refuses_what_it_cannot_decode(cfg_name, layer, blk):
    """The conv taps decode NPE blocks of the lib's HW conv; another layer or
    blocking is refused instead of decoded on the wrong axis."""
    with pytest.raises(SystemExit, match=f"this config gives {layer}, blockify {blk}"):
        ALG.run(C.build("conv1x1"), SC.load_config(cfg_name))


def test_alg_convt_hw_path_refuses_what_it_cannot_decode():
    """xblock_accum.convT "hw" gives the lib's MXConvTranspose2dHW, whose taps
    decode NPE blocks as conv's: on NPE blocks it runs, on channel blocks it
    is refused (no shipped config reaches this path with grids: E4, no
    window, as test_alg_convt_hw_path_taps)."""
    cfg = copy.deepcopy(E4)                    # xblock / flatten
    cfg["mx_specs"]["xblock_accum"].update(convT="hw", backend="python")
    case = C.build("convt_k3_s2")
    assert "acc" in ALG.run(case, cfg)         # the HW path, NPE blocks: runs
    xa = cfg["mx_specs"]["xblock_accum"]
    xa["act_blockify"] = xa["weight_blockify"] = "channel"
    with pytest.raises(SystemExit, match="this config gives MXConvTranspose2dHW, "
                                         "blockify channel / channel"):
        ALG.run(case, cfg)


# --- a missing config entry is a named refusal; any other error a failure ----

def _drop(cfg, path):
    """A copy of `cfg` whose warp entry lacks what `path` names."""
    cfg = copy.deepcopy(cfg)
    e = cfg["entries"]["model.warp"]
    {"inputs": lambda: e.pop("inputs"),
     "input 1": lambda: e.__setitem__("inputs", e["inputs"][:1]),
     "input 0 fxp": lambda: e["inputs"][0].pop("fxp"),
     "input 1 fxp": lambda: e["inputs"][1].pop("fxp")}[path]()
    return cfg


@pytest.mark.parametrize("path, why", [
    ("inputs", "has no input 0 (ref); the sim reads the warp's ref format from the config"),
    ("input 1", "has no input 1 (grid); the sim reads the warp's grid format from the config"),
    ("input 0 fxp", "input 0 (ref) lacks ['fxp']; the sim reads the warp's ref format "
                    "from the config"),
    ("input 1 fxp", "input 1 (grid) lacks ['fxp']; the sim reads the warp's grid format "
                    "from the config")])
def test_warp_entry_missing_an_input_is_a_named_refusal(path, why):
    """Both sides refuse a warp entry that lacks an input or its fxp, naming
    the entry and why (sim_config.warp_sites), instead of a KeyError or an
    IndexError; T12's probe reports it as why the warp cannot run."""
    cfg = _drop(E4, path)
    msg = f"mx_config_npe_e4.json: warp entry 'model.warp' {why}"
    for run in (ALG.run, HW.run):
        with pytest.raises(SystemExit) as ex:
            run(C.build("warp_u8_half"), cfg)
        assert str(ex.value) == msg
    assert WV.probe(cfg) == f"main_ALG SystemExit: {msg}"


@pytest.mark.parametrize("role, i", [("ref", 0), ("grid", 1)])
def test_hw_warp_fxp_without_total_bits_is_a_named_refusal(role, i):
    cfg = copy.deepcopy(E4)
    cfg["entries"]["model.warp"]["inputs"][i]["fxp"].pop("total_bits")
    with pytest.raises(SystemExit, match=re.escape(
            f"mx_config_npe_e4.json: model.warp {role} fxp (call site 0) lacks "
            f"['total_bits']; their lane widths come from the config")):
        HW.warp_spec(cfg)


def test_pla_grids_missing_a_key_are_a_named_refusal():
    """ALG's PCU alone (pcu_on) and HW's PLA bins read the layer's hw_grids:
    a missing key is refused, naming the entry, not a KeyError."""
    cfg = copy.deepcopy(PLA)
    case = C.build("conv_prelu")
    cfg["hw_grids"]["conv_prelu"].pop("pla_m_pos")
    acc = torch.zeros(1, case["w"].shape[0], 2, 2, dtype=torch.int64)
    with pytest.raises(SystemExit, match=re.escape(
            "mx_config_npe_e4_pla.json: hw_grids 'conv_prelu' lacks ['pla_m_pos']; the "
            "lib's PCU reads its PLA bins and grids from the config")):
        ALG.pcu_on(acc, cfg, case, -12)
    cfg = copy.deepcopy(PLA)
    del cfg["mx_specs"]["xblock_accum"]
    with pytest.raises(SystemExit, match=re.escape(
            "mx_config_npe_e4_pla.json: mx_specs lacks ['xblock_accum']")):
        ALG.pcu_on(acc, cfg, case, -12)
    cfg = copy.deepcopy(PLA)
    cfg["hw_grids"]["conv_prelu"].pop("e_layer_min")
    with pytest.raises(SystemExit, match=re.escape(
            "mx_config_npe_e4_pla.json: hw_grids 'conv_prelu' lacks ['e_layer_min']; "
            "the bins sit on its accumulator grid")):
        HW.run(case, cfg)


SITE_LIST_WHY = ("mx_config_npe_e4.json: warp entry 'model.warp' input 1 (grid): 2 "
                 "per-call-site fxp configs for call_sites=3")


def test_probe_skips_only_a_refusal_or_the_libs_site_list_error():
    """T12's probe turns into a skip only a documented refusal, the lib's
    per-call-site length error (sim_config.SiteListError) among them, named
    by the config's entry; any other ValueError (here call_sites 0) or
    KeyError propagates: a bug fails."""
    cfg = copy.deepcopy(E4)                    # warp call_sites 3
    g = cfg["entries"]["model.warp"]["inputs"][1]["fxp"]
    cfg["entries"]["model.warp"]["inputs"][1]["fxp"] = [g, g]
    assert WV.probe(cfg) == f"main_ALG SiteListError: {SITE_LIST_WHY}"
    with pytest.raises(SC.SiteListError, match=re.escape(SITE_LIST_WHY)):
        HW.warp_spec(cfg)
    cfg["entries"]["model.warp"]["call_sites"] = 0
    with pytest.raises(ValueError, match="call_sites must be an int >= 1") as ex:
        WV.probe(cfg)
    assert type(ex.value) is ValueError


def _tmp_config(tmp_path, cfg_name, edit):
    """`cfg_name` of configs/, edited by edit(raw json), written to tmp_path."""
    with open(os.path.join(_ROOT, "configs", cfg_name)) as f:
        raw = json.load(f)
    edit(raw)
    p = tmp_path / cfg_name
    p.write_text(json.dumps(raw))
    return str(p)


def _no_ref_fxp(raw):
    warp = next(e for e in raw["layers"] if isinstance(e, dict) and e.get("kind") == "act_quant")
    warp["inputs"][0].pop("fxp")


def _short_grid_list(raw):
    warp = next(e for e in raw["layers"] if isinstance(e, dict) and e.get("kind") == "act_quant")
    g = warp["inputs"][1]["fxp"]
    warp["inputs"][1]["fxp"] = [g, g]              # call_sites 3


@pytest.mark.parametrize("edit, kind, why", [
    (_no_ref_fxp, "SystemExit",
     "mx_config_npe_e4.json: warp entry 'model.warp' input 0 (ref) lacks ['fxp']; "
     "the sim reads the warp's ref format from the config"),
    (_short_grid_list, "SiteListError", SITE_LIST_WHY)], ids=["no_ref_fxp", "short_site_list"])
def test_every_cli_refuses_a_bad_warp_entry_with_why(tmp_path, edit, kind, why):
    """A warp entry that lacks its ref fxp, or whose grid list does not hold
    one spec per call site (the lib's error): every CLI says refused (or T12
    skipped) and why, naming the entry, and exits 2, not 1 (a failure) and
    not 0."""
    path = _tmp_config(tmp_path, "mx_config_npe_e4.json", edit)
    for script, side in (("main_ALG.py", ""), ("main_HW.py", ""), ("main.py", "ALG ")):
        p = _cli(script, "-m", "warp_u8_half", "--config", path)
        assert p.returncode == 2, script + p.stdout[-2000:] + p.stderr[-2000:]
        assert f"  {side}refused: {why}" in p.stdout.splitlines(), script
    p = _cli("hw_vs_alg_check.py", "--only", "warp", "--config", path)
    assert p.returncode == 2, p.stdout[-2000:] + p.stderr[-2000:]
    out = p.stdout.splitlines()
    assert f"  T12 skipped on mx_config_npe_e4.json: main_ALG {kind}: {why}" in out
    assert out[-1] == "skipped on mx_config_npe_e4.json: T12"


_INJECT = """
import runpy, sys
sys.path.insert(0, {configs!r})
import {module} as m
def boom(*a, **k):
    raise KeyError("injected")
m.{name} = boom
sys.argv = ["hw_vs_alg_check.py", "--only", "warp"]
runpy.run_path({script!r}, run_name="__main__")
"""


@pytest.mark.parametrize("module, name", [("main_ALG", "_code"), ("main_HW", "grid_word")])
def test_hw_vs_alg_check_fails_on_a_key_error_inside_the_warp(tmp_path, module, name):
    """A KeyError raised inside either warp side (a bug, not a config entry
    the sim lacks) is a failure: traceback, exit 1, never "T12 skipped"."""
    configs = os.path.join(_ROOT, "configs")
    runner = tmp_path / "inject.py"
    runner.write_text(_INJECT.format(configs=configs, module=module, name=name,
                                     script=os.path.join(configs, "hw_vs_alg_check.py")))
    p = subprocess.run([sys.executable, str(runner)], cwd=configs, capture_output=True,
                       text=True, timeout=600)
    assert p.returncode == 1, p.stdout[-2000:] + p.stderr[-2000:]
    assert "KeyError: 'injected'" in p.stderr
    assert "skipped" not in p.stdout
