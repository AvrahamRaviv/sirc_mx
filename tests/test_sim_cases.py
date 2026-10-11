"""Tests for the single-layer simulator (configs/): every case's taps rebuild
its own accumulator, and every corner case actually triggers the behaviour it
is named for - measured on the data, not taken from its note."""

import os
import sys

import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (os.path.join(_ROOT, "configs"), _ROOT, os.path.dirname(_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import models as M          # noqa: E402
import main_ALG as ALG      # noqa: E402
import tap_check as TC      # noqa: E402

_CACHE = {}


def _run(name, blockify="npe"):
    """Run a case once per session (check=True, so tap_check already ran)."""
    key = (name, blockify)
    if key not in _CACHE:
        case = M.build_case(name, blockify=blockify)
        taps = ALG.run(case, ALG.load_config(blockify=blockify))
        _CACHE[key] = (case, taps)
    return _CACHE[key]


def _mant_raw(x, code, axis, bs, mbits=8, scale_bits=8):
    """x * 2^(mant_bias - E) before rounding, with E from the exponent codes."""
    E = TC.expand_exp(code, axis, bs, x.shape[axis], scale_bits).double()
    return x.double() * torch.pow(2.0, (mbits - 2) - E)


def _variant(taps, case, layer="conv", **kw):
    """Re-run the reference accumulator for `layer` with one knob changed."""
    meta, cfg = case["meta"], case["meta"]["config"]
    mod = dict(case["model"].named_modules())[layer]
    xb = cfg["mx_specs"]["xblock_accum"]
    args = dict(stride=tuple(mod.stride), padding=tuple(mod.padding),
                dilation=tuple(mod.dilation),
                e_layer_min=meta["e_layer_min"][layer], bits=xb["bits"],
                sat_mode=xb["sat_mode"], bs=cfg["mx_specs"]["block_size"],
                mbits=TC.fmt_bits(cfg["mx_specs"]["a_elem_format"]),
                scale_bits=cfg["mx_specs"]["scale_bits"],
                blockify=ALG.config_blockify(cfg))
    args.update(kw)
    return TC.ref_conv(taps[f"{layer}/a_int"], taps[f"{layer}/a_exp"],
                       taps[f"{layer}/w_int"], taps[f"{layer}/w_exp"],
                       mod.bias, **args)


# =============================================================================
# Every case, both geometries: taps rebuild the accumulator (run() asserts it)
# =============================================================================

@pytest.mark.parametrize("blockify", ["npe", "channel"])
@pytest.mark.parametrize("name", list(M.CASES))
def test_case_taps_rebuild_accumulator(name, blockify):
    case, taps = _run(name, blockify)
    hw = [n for n, c in case["meta"]["datapath"].items() if c == "MXConv2dHW"]
    assert set(case["meta"]["tap_check"]) == set(hw)
    for n in hw:
        assert taps[f"{n}/acc"].dtype == torch.int64
        assert taps[f"{n}/sat"].shape == taps[f"{n}/acc"].shape


def test_only_known_cases_fall_back():
    """A new case that silently misses the HW datapath tests nothing there."""
    fallback = {n for n in M.CASES
                if any(c != "MXConv2dHW"
                       for c in _run(n)[0]["meta"]["datapath"].values())}
    assert fallback == {"depthwise", "grouped4", "linear", "dw_pw"}


def test_npe_exp_shapes():
    case, taps = _run("kernel_5x5")
    bs = M.BLOCK_SIZE
    B, C, H, W = taps["conv/a_int"].shape
    O = taps["conv/w_int"].shape[0]
    assert taps["conv/a_exp"].shape == (B, C, H, -(-W // bs))
    assert taps["conv/w_exp"].shape == (O, -(-(C * 25) // bs))


# =============================================================================
# Each corner case triggers what it is named for
# =============================================================================

def test_conv1x1_ramp_hand_value():
    _, taps = _run("conv1x1_ramp")
    assert bool((taps["conv/acc"] == 11440 * 2 ** 20).all())


def test_out_fxp_q8_8_word():
    _, taps = _run("out_fxp_q8_8")
    assert bool((taps["model/out_code"] == 256).all())


@pytest.mark.parametrize("operand", ["a", "w"])
def test_mx_round_ties_separates_rounding_rules(operand):
    case, taps = _run("mx_round_ties")
    bs = M.BLOCK_SIZE
    if operand == "a":
        raw = _mant_raw(case["x"], taps["conv/a_exp"], 3, bs)
    else:
        w = case["model"].conv.weight.detach()
        raw = _mant_raw(w.reshape(w.shape[0], -1), taps["conv/w_exp"], 1, bs)
    got = taps[f"conv/{operand}_int"].reshape(raw.shape).double()
    ties = (raw.abs() - raw.abs().floor()) == 0.5
    half_away = torch.sign(raw) * torch.floor(raw.abs() + 0.5)
    half_even = torch.round(raw)
    assert int(ties.sum()) > 100
    assert torch.equal(got, half_away)                  # what the lib does
    assert int((got != half_even).sum()) > 50           # what HW might do
    assert int((got != torch.trunc(raw)).sum()) > 50


def test_mx_mant_overflow_clamps_both_signs():
    case, taps = _run("mx_mant_overflow")
    raw = _mant_raw(case["x"], taps["conv/a_exp"], 3, M.BLOCK_SIZE)
    over = torch.floor(raw.abs() + 0.5) >= 128
    got = taps["conv/a_int"]
    assert int((over & (raw > 0)).sum()) > 0 and int((over & (raw < 0)).sum()) > 0
    assert bool((got[over].abs() == 127).all())


def test_zero_blocks_code_zero():
    _, taps = _run("zero_blocks")
    for op, axis in (("a", 3), ("w", 1)):
        code = taps[f"conv/{op}_exp"]
        q = taps[f"conv/{op}_int"]
        q = q if op == "a" else q.reshape(q.shape[0], -1)
        zero = (code == 0)
        assert int(zero.sum()) > 0
        per_el = zero.repeat_interleave(M.BLOCK_SIZE, dim=axis) \
            .narrow(axis, 0, q.shape[axis])
        assert bool((q[per_el] == 0).all())


def test_exp_underflow_clamp():
    _, taps = _run("exp_underflow_clamp")
    assert bool((taps["conv/a_exp"] == 0).all())
    assert int(taps["conv/a_int"].abs().max()) == 8


def test_scale_bits_5_clamps_exponent():
    _, taps = _run("scale_bits_5")
    code, q = taps["conv/a_exp"], taps["conv/a_int"]
    assert int(code.max()) <= 2 * 15
    assert bool((code == 0).all())                      # every act block clamped
    assert 0 < int(q.abs().max()) < 64                  # so mantissas shrank


@pytest.mark.parametrize("name,nb,tail", [("xblock_w32", 1, False),
                                          ("xblock_w33", 2, True),
                                          ("xblock_w64", 2, False)])
def test_xblock_widths(name, nb, tail):
    _, taps = _run(name)
    assert taps["conv/a_exp"].shape[-1] == nb
    assert (taps["conv/a_int"].shape[-1] % M.BLOCK_SIZE != 0) == tail


@pytest.mark.parametrize("name,partial", [("e_layer_min_m8", True),
                                          ("flush_to_zero", False)])
def test_right_shift_floor_vs_trunc(name, partial):
    case, taps = _run(name)
    floor, _ = _variant(taps, case)
    trunc, _ = _variant(taps, case, rshift="trunc")
    assert torch.equal(floor, taps["conv/acc"])
    assert int((floor != trunc).sum()) > 0
    if partial:
        # each lossy product moves by at most one LSB, and nothing flushes
        n_prod = taps["conv/w_int"][0].numel()
        assert int((floor - trunc).abs().max()) <= n_prod
        assert not bool((trunc == 0).all())
    else:
        assert bool((trunc == 0).all())


def test_accum35_partial_saturation():
    _, taps = _run("accum35_partial_sat")
    n = int(taps["conv/sat"].sum())
    assert 0 < n < taps["conv/sat"].numel()


def test_accum_sat_transient_is_transient():
    """Clamped running sum, but the unbounded final sum is inside the bound."""
    case, taps = _run("accum_sat_transient")
    wide, wide_sat = _variant(taps, case, bits=62)
    assert not bool(wide_sat.any())
    assert bool((wide.abs() < 2 ** 47).all())
    assert bool(taps["conv/sat"].all())
    assert not torch.equal(wide, taps["conv/acc"])


def test_bias_ties_separate_rounding_rules():
    case, taps = _run("bias_ties")
    away, _ = _variant(taps, case, bias_round="half_away")
    assert torch.equal(_variant(taps, case)[0], taps["conv/acc"])
    assert int((away != taps["conv/acc"]).sum()) > 0


@pytest.mark.parametrize("name,mmax", [("fmt_int10", 511), ("fmt_int16", 32767)])
def test_wide_formats_use_wide_mantissas(name, mmax):
    _, taps = _run(name)
    m = int(taps["conv/a_int"].abs().max())
    assert 127 < m <= mmax


def test_out_fxp_round_modes_differ():
    codes = [_run(n)[1]["model/out_code"] for n in
             ("out_fxp_round_modes", "out_fxp_round_half_even",
              "out_fxp_round_trunc")]
    assert not torch.equal(codes[0], codes[1])
    assert not torch.equal(codes[0], codes[2])
    assert not torch.equal(codes[1], codes[2])


def test_batch2_samples_differ():
    _, taps = _run("batch2")
    acc = taps["conv/acc"]
    assert acc.shape[0] == 2 and not torch.equal(acc[0], acc[1])


def test_acc_wider_than_fp32_is_flagged():
    """/out loses bits here; /acc is the bit-exact comparable."""
    case, _ = _run("with_bias")
    assert case["meta"]["tap_check"]["conv"]["acc_fp32_lossy"] > 0


# =============================================================================
# Narrow formats
# =============================================================================

def test_int4_ties_separate_rounding_rules():
    case, taps = _run("fmt_int4_ties")
    raw = _mant_raw(case["x"], taps["conv/a_exp"], 3, M.BLOCK_SIZE, mbits=4)
    got = taps["conv/a_int"].double()
    away = (torch.sign(raw) * torch.floor(raw.abs() + 0.5)).clamp(-7, 7)
    assert torch.equal(got, away)
    assert int((got != torch.round(raw).clamp(-7, 7)).sum()) > 0


# =============================================================================
# Warp: every case rebuilds, and each corner splits its alternatives
# =============================================================================

WARP_CASES = M.case_names("warp")


def _warp_variant(taps, pfx="warp", **kw):
    return TC.ref_warp(taps[f"{pfx}/ref_code"], taps[f"{pfx}/grid_code"],
                       float(taps[f"{pfx}/ref_scale"]),
                       float(taps[f"{pfx}/out_scale"]), **kw)[0]


@pytest.mark.parametrize("name", WARP_CASES)
def test_warp_case_rebuilds(name):
    case, taps = _run(name)
    wc = case["meta"]["warp_check"]
    assert wc and all(f"{p}/out_code" in taps for p in wc)
    for p in wc:
        assert taps[f"{p}/ref_code"].dtype == torch.int8
        assert taps[f"{p}/grid_code"].dtype == torch.int16


def test_warp_identity_returns_ref():
    _, taps = _run("warp_identity")
    assert torch.equal(taps["warp/out_code"], taps["warp/ref_code"])


def test_warp_int_shift_is_a_shift():
    _, taps = _run("warp_int_shift")
    ref, out = taps["warp/ref_code"], taps["warp/out_code"]
    # dx = +3, dy = -1: out[y, x] = ref[y - 1, x + 3]
    assert torch.equal(out[:, :, 1:, :-3], ref[:, :, :-1, 3:])
    assert bool((out[:, :, 0] == 0).all()) and bool((out[..., -3:] == 0).all())
    assert not torch.equal(_warp_variant(taps, grid="absolute"), out)


def test_warp_half_pixel_ties():
    _, taps = _run("warp_half_pixel")
    out = taps["warp/out_code"]
    assert int((_warp_variant(taps, out_round="half_even") != out).sum()) > 5
    assert int((_warp_variant(taps, out_round="trunc") != out).sum()) > 5


def test_warp_neg_frac_split():
    _, taps = _run("warp_neg_frac")
    assert not torch.equal(_warp_variant(taps, split="trunc"),
                           taps["warp/out_code"])


def test_warp_oob_zero_vs_border():
    _, taps = _run("warp_oob")
    out = taps["warp/out_code"]
    border = _warp_variant(taps, oob="border")
    assert int((border != out).sum()) > out.numel() // 2


def _fxp_ties(x, step):
    v = x.double() / step
    return (v - v.floor()) == 0.5


@pytest.mark.parametrize("name,inp,tap,step", [
    ("warp_grid_ties", "flow", "grid_code", 2.0 ** -8),
    ("warp_ref_ties", "x", "ref_code", 2.0 ** -5)])
def test_warp_quantizer_ties(name, inp, tap, step):
    case, taps = _run(name)
    v = case[inp].double() / step
    assert bool(_fxp_ties(case[inp], step).all())
    got = taps[f"warp/{tap}"].double()
    assert torch.equal(got, torch.sign(v) * torch.floor(v.abs() + 0.5))
    assert int((got != torch.round(v)).sum()) > v.numel() // 4


def test_warp_grid_saturates():
    _, taps = _run("warp_grid_sat")
    g = taps["warp/grid_code"]
    assert int((g == 32767).sum()) > 0 and int((g == -32768).sum()) > 0


def test_warp_out_saturates_both_ways():
    _, taps = _run("warp_out_sat")
    out, ref = taps["warp/out_code"], taps["warp/ref_code"]
    assert int((out == 127).sum()) > 0 and int((out == -128).sum()) > 0
    assert int((ref.abs() >= 127).sum()) == 0          # only the out clips


def test_warp_subpix_is_fractional():
    _, taps = _run("warp_subpix")
    frac = taps["warp/grid_code"].long() & 255
    assert int((frac != 0).sum()) > frac.numel() * 0.9


def test_warp_levels_scale_per_site():
    _, taps = _run("warp_levels")
    s = [float(taps[f"warp@{i}/ref_scale"]) for i in range(3)]
    assert s[0] < s[1] < s[2]
    assert taps["warp@0/ref_code"].shape[-1] * 4 == taps["warp@2/ref_code"].shape[-1]


def test_conv_warp_handoff():
    """The warp ref is the conv's output requantised to the static ref step."""
    _, taps = _run("conv_warp")
    step = float(taps["warp/ref_scale"])
    v = taps["conv/out"].double() / step
    want = (torch.sign(v) * torch.floor(v.abs() + 0.5)).clamp(-128, 127)
    assert torch.equal(taps["warp/ref_code"].double(), want)
