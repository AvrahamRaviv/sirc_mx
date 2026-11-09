"""Single-op simulator (configs/): every case runs on the ALG side, and its
integer taps explain its own output - conv acc from a_int/w_int, warp out_code
from ref_code/grid_code."""

import copy
import os
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
    import json
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

U8 = [n for n, c in C.CASES.items() if c["op"] == "warp" and "ref" in c]


def _u8_cases():
    assert U8, "no warp case with a code-lattice reference in cases.CASES"
    return U8


@pytest.mark.parametrize("name", _u8_cases())
def test_hw_warp_runs_their_code(name):
    """Every code-lattice warp case runs on their code: their packed
    reference holds the case's codes, out is their output times the ref
    step, the grid format comes from the config."""
    case = C.build(name)
    t = HW.run(case, E4)
    B, Cc, H, W = case["x"].shape
    codes = (case["x"].double() / case["warp"]["ref_scale"]).round().long()
    assert torch.equal(t["hw_ref_code"][..., :W], codes)
    assert t["hw_acc"].dtype == torch.int64 and t["hw_acc"].shape == (B, Cc, H, W)
    assert t["hw_weights"].shape == (H, W, 4)
    assert torch.equal(t["out"], t["out_raw"] * case["warp"]["ref_scale"])
    assert t["frac_cfg"].tolist() == [8, 8]          # mx_config_npe_e4.json grid fxp
    assert int(t["frac_sum"]) == int(t["hw_frac_bits"].sum())


def test_hw_warp_acc_is_their_int64_psum():
    """hw_acc is bcu's p_sum taken in their loop, not read back from their
    float32 output: with a constant reference c and a continuous flow every
    interior sample's acc is c * 2^(frx+fry) exactly (their weights sum to
    2^(frx+fry)), while their float32 output equals it only where |acc| <=
    2^24; on a random reference it rounds some."""
    g = torch.Generator().manual_seed(3)
    flow = (torch.rand(1, 2, 8, 8, generator=g) * 2 - 1) * 1.5
    t = HW.run(C.warp_case("c", torch.full((1, 1, 8, 8), 7), flow, C.P2, C.P2), E4)
    k = int(t["frac_sum"])
    pos = HW.abs_coords(flow)[0]
    inner = ((pos[..., 0] >= 0) & (pos[..., 0] < 7) & (pos[..., 1] >= 0)
             & (pos[..., 1] < 7)).expand(1, 1, 8, 8)
    assert int(inner.sum()) > 0
    assert bool((t["hw_acc"][inner] == 7 * 2 ** k).all())
    t = HW.run(C.warp_case("r", torch.randint(0, 128, (1, 1, 8, 8), generator=g),
                           flow, C.P2, C.P2), E4)
    acc, back = t["hw_acc"], t["out_raw"].double() * 2.0 ** int(t["frac_sum"])
    small = acc.abs() <= 2 ** 24
    assert bool(small.any()) and bool((back[small] == acc[small].double()).all())
    n, big, rounded = M.warp_float32_store(t)
    assert big > 0 and rounded > 0                  # float32 dropped acc bits
    assert bool((back != acc.double()).any())


@pytest.mark.parametrize("name", ["warp_u8_subpix", "warp_u8_subpix_w64"])
def test_hw_warp_interior_equals_alg_exact(name):
    """Codes 0..127, flow on 1/256, their fraction bits >= 8 per axis: inside
    the frame their acc / 2^(frx+fry) is the exact bilinear sum of ALG's own
    codes (cases.Warp in float64, exact on the 1/256 grid)."""
    case = C.build(name)
    a, h = ALG.run(case, E4), HW.run(case, E4)
    assert int(h["hw_frac_bits"].min()) >= 8
    want = C.Warp()(a["ref_code"].double(), a["grid_code"].double() / 256)
    got = h["hw_acc"].double() / 2.0 ** int(h["frac_sum"])
    st = {r["stage"]: r for r in M.warp_stages(case, a, h)}
    _, _, H, W = case["x"].shape
    pos = HW.abs_coords(case["flow"])[0]
    x0, y0 = pos[..., 0].floor(), pos[..., 1].floor()
    inside = ((x0 >= 0) & (x0 + 1 < W) & (y0 >= 0) & (y0 + 1 < H)).expand_as(got)
    assert int(inside.sum()) == st["acc interior"]["n"] > 0
    assert torch.equal(got[inside], want[inside])


def test_hw_warp_stages_name_the_border_rule():
    """main.warp_stages on the identity flow: ref, grid and interior equal,
    the border differs: their last row reads the row above and, W being a
    multiple of 32, their last column the column before (HW Q2)."""
    case = C.build("warp_u8_zero")
    a, h = ALG.run(case, E4), HW.run(case, E4)
    st = {r["stage"]: r for r in M.warp_stages(case, a, h)}
    assert all(r["n"] > 0 for r in st.values())
    assert st["ref code"]["n_diff"] == st["grid px"]["n_diff"] == 0
    assert st["acc interior"]["n_diff"] == 0
    assert st["acc border"]["n_diff"] > 0
    got = h["hw_acc"][0, 0].double() / 2.0 ** int(h["frac_sum"])
    ref = a["ref_code"][0, 0].double()
    assert torch.equal(got[-1, :-1], ref[-2, :-1])
    assert torch.equal(got[:-1, -1], ref[:-1, -2])
    assert torch.equal(got[:-1, :-1], ref[:-1, :-1])


def test_hw_warp_refuses_what_their_code_would_get_wrong():
    """Batch > 1 with different flows (their batch-0 grid), a reference their
    packer would clamp (signed, or above 255), a non-finite reference (their
    packer makes NaN code 0), H < 2, an axis whose fraction bits come out
    infinite or negative (an 8-bit grid on W = 200). Same flows over a batch
    run."""
    ok = torch.randint(0, 128, (2, 1, 3, 8), generator=torch.Generator().manual_seed(1))
    f = C.const_flow(2, 3, 8, 0.5, 0.0)
    t = HW.run(C.warp_case("b", ok, f, C.P2, C.P2), E4)
    one = HW.run(C.warp_case("b1", ok[1:], f[1:], C.P2, C.P2), E4)
    assert torch.equal(t["hw_acc"][1:], one["hw_acc"])
    f2 = f.clone()
    f2[1, 0] += 1.0
    with pytest.raises(NotImplementedError, match="batch > 1 with different flows"):
        HW.run(C.warp_case("b", ok, f2, C.P2, C.P2), E4)
    for bad in (-1, 256):
        ref = ok[:1].clone()
        ref[0, 0, 0, 0] = bad
        with pytest.raises(NotImplementedError, match="clamps to uint8"):
            HW.run(C.warp_case("r", ref, C.const_flow(1, 3, 8, 0, 0), C.P2, C.P2), E4)
    ref = ok[:1].double().clone()
    ref[0, 0, 0, 0] = float("nan")
    with pytest.raises(NotImplementedError, match="uint8"):
        HW.run(C.warp_case("n", ref, C.const_flow(1, 3, 8, 0, 0), C.P2, C.P2), E4)
    g8 = copy.deepcopy(E4)
    g8["entries"]["model.warp"]["inputs"][1]["fxp"]["total_bits"] = 8
    with pytest.raises(NotImplementedError, match="fraction bits"):   # x frac -1
        HW.run(C.warp_case("g8", torch.full((1, 1, 2, 200), 10),
                           C.const_flow(1, 2, 200, 0, 0), C.P2, C.P2), g8)
    with pytest.raises(NotImplementedError, match="H = 1"):
        HW.run(C.warp_case("h", ok[:1, :, :1], C.const_flow(1, 1, 8, 0, 0), C.P2, C.P2), E4)
    fy = C.const_flow(1, 3, 8, 0.0, 0.0)
    fy[0, 1] = -torch.arange(3.0).view(3, 1)        # every y position 0
    with pytest.raises(NotImplementedError, match="fraction bits"):
        HW.run(C.warp_case("y", ok[:1], fy, C.P2, C.P2), E4)
    with pytest.raises(NotImplementedError, match="uint8"):
        HW.run(C.build("warp_int_shift"), E4)       # randn reference: signed


def test_hw_warp_grid_format_from_config_and_reported():
    """frac_cfg is the config's grid fxp at call site 0; their own fraction
    bits are tapped beside it, a mismatch and frx + fry > 14 are named, the
    call sites are reported; lanes their code lacks are refused."""
    def cfg_with(**grid):
        cfg = copy.deepcopy(E4)
        cfg["entries"]["model.warp"]["inputs"][1]["fxp"].update(grid)
        return cfg
    z = lambda H, W: C.warp_case("z", torch.zeros(1, 1, H, W), C.const_flow(1, H, W, 0, 0),
                                 C.P2, C.P2)
    t = HW.run(z(4, 40), cfg_with(frac_bits=7))
    assert t["frac_cfg"].tolist() == [7, 7] and int(t["frac_match"]) == 0
    notes = " ".join(HW.warp_notes(t))
    assert "HW Q3" in notes and "HW Q4" in notes and int(t["frac_over"]) == 1
    assert "call_sites 3" in notes
    t = HW.run(z(65, 65), E4)                       # their rule gives 8, 8 here
    assert t["hw_frac_bits"].tolist() == [8, 8] and int(t["frac_match"]) == 1
    assert "HW Q3" not in " ".join(HW.warp_notes(t))
    assert int(HW.run(z(4, 40), HWX)["call_sites"]) == 2
    cfg = copy.deepcopy(E4)                          # per-site list: site 0 is used
    g = cfg["entries"]["model.warp"]["inputs"][1]["fxp"]
    cfg["entries"]["model.warp"]["inputs"][1]["fxp"] = [dict(g, frac_bits=f) for f in (6, 7, 5)]
    t = HW.run(z(4, 40), cfg)
    assert t["frac_cfg"].tolist() == [6, 6] and int(t["call_site"]) == 0
    with pytest.raises(NotImplementedError, match="int16"):
        HW.run(z(4, 40), cfg_with(total_bits=32))
    cfg = copy.deepcopy(E4)
    cfg["entries"]["model.warp"]["inputs"][0]["fxp"]["total_bits"] = 16
    with pytest.raises(NotImplementedError, match="8-bit"):
        HW.run(z(4, 40), cfg)


def test_hw_warp_setup_c_struct_is_not_read():
    """main_HW's minimal setup calibrates with their default fraction bits;
    the result does not depend on them, their forward never reads the PCU
    c_structs (PCU bypassed)."""
    case = C.build("warp_u8_half")
    ref = case["x"].double() / case["warp"]["ref_scale"]
    coords = HW.abs_coords(case["flow"])
    outs = []
    for fb in ((7, 7), (1, 2)):
        layer = HW._warp_layer(ref.shape[1], 16, 48)
        layer.calibrate(fb)
        with torch.no_grad():
            outs.append(layer(ref, coords))
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(outs[0], HW.run(case, E4)["out_raw"])


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
HW_EXACT = SC.load_config("mx_config_npe_hw_exact.json")   # warp entry: no output


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


@pytest.mark.parametrize("cfg", [DOF_RUN, HW_EXACT], ids=["dof_run", "hw_exact"])
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


def test_t12_output_requant_q7_without_output_fxp():
    """D7 on a config with no output fxp: ALG gives the acc unrounded."""
    (v,) = [x for x in WV.vectors(HW_EXACT) if x["label"] == "D7 | HW Q7"]
    assert (v["acc"], v["alg"], v["hw"], v["pcu"]) == (37.5, 37.5, 37.5, 38)
    assert "no output fxp" in v["text"]


# --- hw_vs_alg_check T12: the warp vectors sent with the HW questions --------

import warp_vectors as WV             # noqa: E402

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


def test_t12_border_q1():
    """ALG reads zero outside the frame; HW reads a frame pixel there."""
    left, bottom = _t12("D1 | HW Q1")
    ref = left["ref"][0, 0].double()
    # x = -0.25 at (y 1, x 0): ALG 0.25 * 0 + 0.75 * ref[1, 0]
    assert left["alg"] == pytest.approx(0.75 * float(ref[1, 0])) == pytest.approx(8.25)
    # their index is floor (q >> frac = -1), which get_ref_frame clamps to 0;
    # the fraction is the low bits, 0.75 -> 0.25 * ref[1, 0] + 0.75 * ref[1, 1]
    assert left["hw"] == pytest.approx(0.25 * float(ref[1, 0]) + 0.75 * float(ref[1, 1]))
    assert left["hw"] == pytest.approx(18.5)
    # y = H-1+0.25 at (y 3, x 2): ALG 0.75 * ref[3, 2]; HW (question 1): their
    # index clamped to H-2, the raw fraction 0.25 kept
    assert bottom["alg"] == pytest.approx(0.75 * float(ref[3, 2]))
    assert bottom["hw"] == pytest.approx(
        0.75 * float(ref[2, 2]) + 0.25 * float(ref[3, 2])) == pytest.approx(32.25)


def test_t12_identity_q2():
    """Zero flow: ALG returns the frame; HW the last row (column) from the
    row (column) before it."""
    row, col = _t12("D1 | HW Q2")
    ref = row["ref"][0, 0].double()
    assert row["alg"] == ref[-1].tolist()
    assert row["hw"] == ref[-2].tolist()
    ref = col["ref"][0, 0].double()
    assert col["alg"] == float(ref[0, -1]) and col["hw"] == float(ref[0, -2])


def test_t12_fraction_bits_q3_q4():
    """Their fraction bits come from the frame: 15 minus the bits of the
    largest position on that axis (16-bit signed word); the config's are
    fixed. Equal only where the frame gives the config's format (65x65)."""
    q3, moved = _t12("D2 | HW Q3")
    (q4,) = _t12("D3 | HW Q4")
    fcfg = list(HW.warp_spec(E4)["frac_cfg"])
    assert set(q3["hw"]) == set(WV.FRAC_SIZES)
    for (Hh, Ww), fh in q3["hw"].items():
        assert q3["alg"][(Hh, Ww)] == fcfg
        assert fh == [15 - (Ww - 1).bit_length(), 15 - (Hh - 1).bit_length()]
        assert (fh == fcfg) == ((Hh, Ww) == (65, 65))
        assert q4["hw"][(Hh, Ww)] == sum(fh) > HW.WARP_FRAC_SUM_LIMIT
    assert moved["grid"]["n_diff"] > 0 and moved["acc"]["n_diff"] > 0


def test_t12_ties_q5():
    """Grid ties: ALG rounds half away from zero, HW the tie to 0 px; the
    reference: ALG half up, HW half to even."""
    g_pos, g_neg, ref = _t12("D4 | HW Q5") + _t12("D6 | HW Q5")
    assert (g_pos["tie"], g_pos["alg"], g_pos["hw"]) == (0.5, 1, 0)
    assert (g_neg["tie"], g_neg["alg"], g_neg["hw"]) == (-0.5, -1, 0)
    r = ref["ref"].double()
    assert ref["alg"] == torch.floor(r + 0.5).tolist() == [3, 4, 4]
    assert ref["hw"] == torch.round(r).tolist() == [2, 4, 4]


def test_t12_reference_format_q6():
    """ALG saturates to int8, their packer clamps to uint8; main_HW refuses."""
    (v,) = _t12("D5 | HW Q6")
    r = v["ref"].double()
    assert v["alg"] == r.clamp(-128, 127).tolist()
    assert v["hw"] == r.clamp(0, 255).tolist()
    assert v["refusal"].startswith("refused")


def test_t12_output_requant_q7():
    """acc 37.5: ALG requantizes (half away -> 38), HW stores the acc as is."""
    (v,) = _t12("D7 | HW Q7")
    assert (v["acc"], v["alg"], v["hw"], v["pcu"]) == (37.5, 38, 37.5, 38)


def test_t12_batch_q8():
    """Batch 1 moved +1 px: ALG moves it, their code samples batch 1 on
    batch 0's (zero) grid; main_HW refuses."""
    (v,) = _t12("D8 | HW Q8")
    ref1 = v["ref"][1, 0, 0].double()
    assert v["alg"] == ref1[1:5].tolist()
    assert v["hw"] == ref1[:4].tolist()
    assert v["refusal"].startswith("refused")


def test_t12_float32_store_q9():
    (v,) = _t12("D9 | HW Q9")
    assert 0 < v["rounded"] <= v["big"] <= v["n"] and v["lsb"] > 0


def test_t12_padding_q10():
    """x = W-1+0.5 on 100: ALG 0.5 * 100 at both widths; HW reads its zero
    padding at W = 40 (padded to 64), the frame's own pixel at W = 32."""
    w40, w32 = _t12("D1 | HW Q10")
    assert (w40["W"], w40["alg"], w40["hw"]) == (40, 50, 50)
    assert (w32["W"], w32["alg"], w32["hw"]) == (32, 50, 100)


def test_hw_vs_alg_check_prints_the_t12_vectors():
    """--only warp prints every vector's own text (built from its values)."""
    p = subprocess.run([sys.executable, "hw_vs_alg_check.py", "--only", "warp"],
                       cwd=os.path.join(_ROOT, "configs"), capture_output=True,
                       text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    lines = [f"  [{v['label']}] {v['text']}" for v in T12]
    assert lines and all(ln in p.stdout.splitlines() for ln in lines), p.stdout[-2000:]


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
    the summary, never counted as ran; only refusals: exit 0."""
    names = C.select("all")
    p = _cli("main_HW.py", "-m", "all", "--config", "mx_config_npe_e4.json")
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
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


def test_main_cli_reports_each_refusal_and_goes_on():
    """main.py -m all on npe_e4: the input-layer and PLA cases are refused
    (no grids in that config), the run goes on to the warp, the refusals are
    named, and the exit code says a case did not run (2), never a pass."""
    p = _cli("main.py", "-m", "all", "--config", "mx_config_npe_e4.json")
    assert p.returncode == 2, p.stdout[-2000:] + p.stderr[-2000:]
    out = p.stdout.splitlines()
    assert any(ln.startswith("  HW refused: mx_config_npe_e4.json: input layer needs")
               for ln in out)
    assert "=== warp_u8_subpix_w64 ===" in out
    refused = _refused_names(out[-1])
    assert {"conv1x1_in", "conv3x3_in", "conv_prelu"} <= set(refused)


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
