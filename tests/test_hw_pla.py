"""PLA (xblock_accum.pla): the PReLU / LeakyReLU / ReLU after an MXConv2dHW
folded into the PCU multiply before the output window. Integer reference
for both bias orders, ReLU fold bit-exact, quantizer fold + hw_grids round
trip, post-PLA calibration, train mode untouched, convT, triton == python."""
import json
import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects")
sys.path.insert(0, "/home/avrahamra/PycharmProjects")

from microxcaling.mx import MxSpecs
from fixed_point.hw_grids import check_hw_grid, load_hw_grids
from fixed_point.mx_fixed_point import normalize_xblock_accum
from fixed_point.mx_fixed_point_hw import (calibrate_hw_grids, export_hw_grids,
                                           pla_multipliers)
from mx_layers_blocked import MXConv2dHW
from mx_layers_convT import MXConvTranspose2dHW
from mx_quantizer import MXQuantizer
from mx_stats import _fp32_functional

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False
_GPU = _HAS_TRITON and torch.cuda.is_available()

WIN = 22
MODES = {"channel": ("channel", "channel"), "xblock": ("xblock", "flatten")}


def _specs():
    sp = MxSpecs()
    for k, v in dict(w_elem_format="int8", a_elem_format="int8", block_size=32,
                     scale_bits=8, shared_exp_method="max",
                     custom_cuda=False).items():
        sp[k] = v
    return sp


def _xb(mode="xblock", backend="python", **kw):
    act, wt = MODES[mode]
    cfg = dict(mode="hw_fixed_point", bits=48, backend=backend,
               act_blockify=act, weight_blockify=wt, out_window="infer",
               verbose=0, pla="infer")
    cfg.update(kw)
    return normalize_xblock_accum(cfg)


def _block(act, convT=False, mode="xblock", backend="python", bias=0.3, **kw):
    """Sequential(HW conv, act) with the act folded, grids calibrated."""
    torch.manual_seed(0)
    if convT:
        conv = MXConvTranspose2dHW(32, 16, 4, stride=2, padding=1, mx_specs=_specs())
    else:
        conv = MXConv2dHW(32, 16, 3, padding=1, mx_specs=_specs())
    with torch.no_grad():
        conv.bias.uniform_(-bias, bias)
    conv.xblock_accum = _xb(mode, backend, **kw)
    conv._mx_layer_name = "conv"
    seq = nn.Sequential(conv, act)
    MXQuantizer._fold_pla(seq, verbose=0)
    calibrate_hw_grids(seq, _data(), num_batches=2, verbose=0)
    return seq


def _prelu(seed=5, n=16, lo=-0.4, hi=0.6):
    act = nn.PReLU(n)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        act.weight.copy_(torch.rand(n, generator=g) * (hi - lo) + lo)
    return act


def _data(n=2, seed=1, C=32, H=6, W=40):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, C, H, W, generator=g) * 2 for _ in range(n)]


def _win(P, e_word, e_out, out_round):
    """Window written out on its own: floor / round-half-up shift, saturate."""
    s = e_out - e_word
    if s <= 0:
        v = P * (1 << -s)
    else:
        v = torch.div(P + ((1 << (s - 1)) if out_round == "nearest" else 0),
                      1 << s, rounding_mode="floor")
    return v.clamp(-(2 ** WIN - 1), 2 ** WIN - 1)


def _run(seq, x):
    conv = seq[0]
    conv._tap_acc = True
    seq.eval()
    with torch.no_grad():
        y = seq(x)
    return y, conv._last_acc, conv._last_win


# ---------------------------------------------------------------------------
# Multipliers
# ---------------------------------------------------------------------------

def test_multipliers_round_half_up_and_auto_frac():
    m_pos, m_neg, frac = pla_multipliers(torch.tensor([0.25, -0.1, 1 / 128, 0.0]))
    assert frac == 6 and m_pos.tolist() == [64] * 4
    # 0.1 * 64 = 6.4 -> -6; 0.5 tie (1/128 * 64) -> 1, half up
    assert m_neg.tolist() == [16, -6, 1, 0]
    assert pla_multipliers(torch.tensor([2.5]))[2] == 5         # 2.5*64 > 127
    assert pla_multipliers(torch.tensor([0.25]), mult_bits=10)[2] == 8
    with pytest.raises(ValueError):
        pla_multipliers(torch.tensor([3.0]), frac_bits=6)
    with pytest.raises(ValueError):
        pla_multipliers(torch.tensor([0.25]), frac_bits=7)     # m_pos = 128


def test_config_validation():
    with pytest.raises(ValueError, match="out_window"):
        _xb(out_window="off")
    with pytest.raises(ValueError):
        _xb(pla="always")
    with pytest.raises(ValueError):
        _xb(pla_bias="mid")
    with pytest.raises(ValueError):
        _xb(pla_frac_bits=7)            # 2^7 does not fit signed 8b


# ---------------------------------------------------------------------------
# Integer reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("convT", [False, True])
@pytest.mark.parametrize("out_round", ["floor", "nearest"])
def test_prelu_pre_matches_int_reference(convT, out_round):
    seq = _block(_prelu(), convT=convT, out_round=out_round)
    conv = seq[0]
    y, acc, word = _run(seq, _data(1, seed=7)[0])
    pla = conv.pla_params()
    m = torch.where(acc < 0, pla.m_neg.view(1, -1, 1, 1), pla.m_pos.view(1, -1, 1, 1))
    want = _win(acc * m, conv.e_layer_min - pla.frac, conv.e_qa_out, out_round)
    assert torch.equal(word, want)
    assert torch.equal(y, want.float() * 2.0 ** conv.e_qa_out)
    assert (acc < 0).any() and (pla.m_neg != pla.m_pos).all()


def test_post_bias_matches_emulator_order():
    """'post' = pcu_pla: P = N * m(N) + bias on the multiplied grid."""
    seq = _block(_prelu(), pla_bias="post", bias=2.0)
    conv = seq[0]
    y, acc, word = _run(seq, _data(1, seed=7)[0])       # acc has no bias
    pla = conv.pla_params()
    e_p = conv.e_layer_min - pla.frac
    b = torch.floor(conv.bias.detach().double() * 2.0 ** -e_p + 0.5).long()
    m = torch.where(acc < 0, pla.m_neg.view(1, -1, 1, 1), pla.m_pos.view(1, -1, 1, 1))
    want = _win(acc * m + b.view(1, -1, 1, 1), e_p, conv.e_qa_out, "floor")
    assert torch.equal(word, want)
    # The two orders differ where N and N + bias have other signs.
    seq_pre = _block(_prelu(), bias=2.0)
    assert not torch.equal(_run(seq_pre, _data(1, seed=7)[0])[2], word)


def test_bias_rounds_half_up_with_pla():
    seq = _block(nn.ReLU())
    conv = seq[0]
    x = _data(1, seed=7)[0]
    with torch.no_grad():
        conv.bias.zero_()
    acc0 = _run(seq, x)[1]
    with torch.no_grad():
        conv.bias.fill_(2.5 * 2.0 ** conv.e_layer_min)   # exact tie
    assert torch.equal(_run(seq, x)[1] - acc0, torch.full_like(acc0, 3))
    conv.xblock_accum = _xb(pla="off")                   # today: half to even
    assert torch.equal(_run(seq, x)[1] - acc0, torch.full_like(acc0, 2))


@pytest.mark.parametrize("out_round", ["floor", "nearest"])
def test_relu_fold_bit_exact(out_round):
    seq = _block(nn.ReLU(), out_round=out_round)
    x = _data(1, seed=7)[0]
    y = _run(seq, x)[0]
    seq[0].xblock_accum = _xb(pla="off", out_round=out_round)
    seq[0]._pla_act = None
    y_off = _run(seq, x)[0]                     # window, then float ReLU
    assert torch.equal(y, y_off) and (y_off == 0).any()


# ---------------------------------------------------------------------------
# Layer behaviour
# ---------------------------------------------------------------------------

def test_act_passes_pla_output_through_and_train_mode_unchanged():
    seq = _block(_prelu())
    conv, act = seq
    x = _data(1, seed=7)[0].requires_grad_(True)
    seq.eval()
    with torch.no_grad():
        z = conv(x)
        assert torch.equal(act(z), z)           # already activated
    assert not getattr(torch.zeros(1), "_pla_folded", False)

    seq.train()
    y = seq(x)
    y.square().sum().backward()
    g = (x.grad.clone(), conv.weight.grad.clone(), act.weight.grad.clone())
    assert not conv._pla_last
    for p in (x, conv.weight, act.weight):
        p.grad = None
    conv.xblock_accum = _xb(pla="off")
    y2 = seq(x)
    y2.square().sum().backward()
    assert torch.equal(y, y2)
    for a, b in zip(g, (x.grad, conv.weight.grad, act.weight.grad)):
        assert torch.equal(a, b)


def test_calibration_sees_post_pla_output():
    """All outputs negative: max|PReLU(z)| = alpha * max|z|, alpha = 2^-3."""
    def e_out(pla):
        act = nn.PReLU(1, init=0.125)
        torch.manual_seed(0)
        conv = MXConv2dHW(32, 16, 3, padding=1, mx_specs=_specs())
        with torch.no_grad():
            conv.weight.mul_(0.01)
            conv.bias.fill_(-40.0)
        conv.xblock_accum = _xb(pla=pla)
        conv._mx_layer_name = "conv"
        seq = nn.Sequential(conv, act)
        MXQuantizer._fold_pla(seq, verbose=0)
        calibrate_hw_grids(seq, _data(), num_batches=2, verbose=0)
        return conv.e_qa_out
    assert e_out("infer") == e_out("off") - 3


def test_fp32_reference_includes_folded_act():
    seq = _block(_prelu())
    x = _data(1, seed=7)[0]
    y = _run(seq, x)[0]
    ref = _fp32_functional(seq[0], x)
    sqnr = 10 * torch.log10(ref.square().sum() / (ref - y).square().sum())
    assert sqnr > 30


# ---------------------------------------------------------------------------
# Quantizer: fold, hw_grids round trip, back compat
# ---------------------------------------------------------------------------

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.b = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), _prelu(n=32))
        self.c2 = nn.Conv2d(32, 8, 3, padding=1)

    def forward(self, x):
        return self.c2(self.b(x))


def _cfg(tmp, hw_grids=None, **xb):
    x = {"enabled": True, "mode": "hw_fixed_point", "bits": 48,
         "backend": "python", "e_layer_min": None, "out_window": "infer",
         "act_blockify": "xblock", "weight_blockify": "flatten", "verbose": 0}
    x.update(xb)
    cfg = {"mx_specs": {"w_elem_format": "int8", "a_elem_format": "int8",
                        "block_size": 32, "custom_cuda": False,
                        "xblock_accum": x},
           "layers": ["b.0", "c2"], "ptq": False, "measure_error": False}
    if hw_grids is not None:
        cfg["hw_grids"] = hw_grids
    path = os.path.join(tmp, f"cfg_{len(os.listdir(tmp))}.json")
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def _eval(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)


def test_quantizer_round_trip_and_state_dict():
    tmp = tempfile.mkdtemp()
    q = MXQuantizer(config_path=_cfg(tmp, pla="infer"), save_dir=tmp)
    model = q.quant(_Net())
    assert model.b[0].pla_on() and not model.c2.pla_on()   # c2: nothing after it
    assert isinstance(model.b[1], nn.PReLU)
    assert set(model.state_dict()) == set(_Net().state_dict())
    calibrate_hw_grids(model, _data(3), num_batches=3, verbose=0)
    x = _data(1, seed=9)[0]
    y = _eval(model, x)

    path = q.save_hw_grids(model)
    grids = load_hw_grids(path)
    p = model.b[0].pla_params()
    assert grids["b.0"]["pla_m_neg"] == p.m_neg.tolist()
    assert grids["b.0"]["pla_frac"] == 6 and "pla_frac" not in grids["c2"]

    model2 = MXQuantizer(config_path=path, save_dir=tmp).quant(_Net())
    with torch.no_grad():
        model2.b[1].weight.fill_(0.9)       # the sim reads the config only
    assert torch.equal(_eval(model2, x), y)


def test_old_grids_load_and_pla_off_ignores_nothing():
    tmp = tempfile.mkdtemp()
    q = MXQuantizer(config_path=_cfg(tmp), save_dir=tmp)       # pla off
    model = q.quant(_Net())
    calibrate_hw_grids(model, _data(3), num_batches=3, verbose=0)
    old = export_hw_grids(model)
    assert all("pla_frac" not in e for e in old.values())
    # An old grids file loads with PLA on (multipliers from the module).
    m_on = MXQuantizer(config_path=_cfg(tmp, hw_grids=old, pla="infer"),
                       save_dir=tmp).quant(_Net())
    assert m_on.b[0].pla_on() and m_on.b[0]._pla_grid is None
    _eval(m_on, _data(1)[0])
    # PLA multipliers in the grids but pla off: refuse, not silently drop.
    new = dict(old, **{"b.0": dict(old["b.0"], pla_frac=6, pla_m_pos=[64] * 32,
                                   pla_m_neg=[16] * 32)})
    with pytest.raises(ValueError, match="no activation was folded"):
        MXQuantizer(config_path=_cfg(tmp, hw_grids=new), save_dir=tmp).quant(_Net())


def test_hw_grid_pla_keys_checked():
    base = {"e_layer_min": -30, "e_qa_out": -19}
    check_hw_grid("c", dict(base, pla_frac=6, pla_m_pos=[64], pla_m_neg=[-3]))
    with pytest.raises(ValueError, match="needs all"):
        check_hw_grid("c", dict(base, pla_frac=6))
    with pytest.raises(TypeError):
        check_hw_grid("c", dict(base, pla_frac=6, pla_m_pos=64, pla_m_neg=[1]))
    with pytest.raises(ValueError, match="lengths"):
        check_hw_grid("c", dict(base, pla_frac=6, pla_m_pos=[64, 64], pla_m_neg=[1]))


# ---------------------------------------------------------------------------
# Triton == python (GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _GPU, reason="needs triton + CUDA")
@pytest.mark.parametrize("mode,variant", [("channel", "reuse"), ("xblock", "reuse"),
                                          ("xblock", "opt")])
@pytest.mark.parametrize("bias_order", ["pre", "post"])
@pytest.mark.parametrize("convT", [False, True])
def test_triton_matches_python(mode, variant, bias_order, convT):
    outs = []
    for backend in ("python", "triton"):
        seq = _block(_prelu(), convT=convT, mode=mode, backend=backend,
                     pla_bias=bias_order, npe_triton_variant=variant, bias=2.0)
        seq = seq.cuda().eval()
        with torch.no_grad():
            outs.append(seq(_data(1, seed=7)[0].cuda()).cpu())
    assert torch.equal(outs[0], outs[1])
