"""ConvTranspose2d HW path (MXConvTranspose2dHW): zero-insert conv on the
MXConv2dHW path. Bit-exact accumulator vs an fp64 conv_transpose2d on the
same quantized operands, quantizer wiring, hw_grids round trip, calibration
on the real (undilated) input, STE grads, triton == python."""
import json
import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects")
sys.path.insert(0, "/home/avrahamra/PycharmProjects")

from microxcaling.mx import MxSpecs
from microxcaling.mx.transpose_convolution import ConvTranspose2d as MXConvTranspose2d
from fixed_point.hw_grids import load_hw_grids
from fixed_point.mx_fixed_point import normalize_xblock_accum
from fixed_point.mx_fixed_point_hw import (_min_grid_exps, calibrate_hw_grids,
                                           dilate_input, e_qa_out_from_absmax)
from mx_layers_convT import MXConvTranspose2dHW
from mx_quantizer import MXQuantizer
from mx_stats import _fp32_functional

try:
    import triton  # noqa: F401
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False

MODES = {"channel": ("channel", "channel"), "xblock": ("xblock", "flatten")}
# Cin, Cout, k, stride, padding, output_padding, H, W
SHAPES = [(96, 2, 4, 2, 1, 0, 5, 36), (48, 2, 4, 2, 1, 0, 4, 40),
          (32, 2, 4, 2, 1, 0, 3, 70), (32, 3, 3, 2, 1, 1, 4, 40)]


def _specs():
    sp = MxSpecs()
    for k, v in dict(w_elem_format="int8", a_elem_format="int8", block_size=32,
                     scale_bits=8, shared_exp_method="max",
                     custom_cuda=False).items():
        sp[k] = v
    return sp


def _layer(shape, mode, backend="python", **xb):
    Cin, Co, k, s, p, op, _, _ = shape
    torch.manual_seed(0)
    m = MXConvTranspose2dHW(Cin, Co, k, stride=s, padding=p, output_padding=op,
                            mx_specs=_specs())
    with torch.no_grad():
        m.weight.mul_(4)
    act, wt = MODES[mode]
    m.xblock_accum = normalize_xblock_accum(dict(
        mode="hw_fixed_point", bits=48, backend=backend, act_blockify=act,
        weight_blockify=wt, verbose=0, **xb))
    return m


def _x(shape, seed=1):
    Cin, _, _, _, _, _, H, W = shape
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, Cin, H, W, generator=g).relu() * 3


def _to_convT(qw, Cin):
    """Conv-layout (maybe channel-padded) quantized weight -> convT layout."""
    return qw[:, :Cin].flip(-1, -2).transpose(0, 1)


def _set_exact_e_min(m, x, mode):
    """e_layer_min = min Ea + min Ew - 12: every shift >= 0, acc exact."""
    m.e_layer_min = -60
    m._tap_acc = True
    m.eval()
    with torch.no_grad():
        m(x)
    act, wt = MODES[mode]
    e_a, e_w = _min_grid_exps(m._last_qi, m._last_qw, 32,
                              act_blockify=act, weight_blockify=wt)
    m.e_layer_min = e_a + e_w
    return m


# ---------------------------------------------------------------------------
# Exactness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("shape", SHAPES)
def test_acc_bit_exact_vs_fp64_convT(shape, mode):
    Cin, _, _, s, p, op, _, _ = shape
    m = _layer(shape, mode)
    x = _x(shape)
    _set_exact_e_min(m, x, mode)
    with torch.no_grad():
        y = m(x)
    e = m.e_layer_min
    qi, qw = m._last_qi.double()[:, :Cin], _to_convT(m._last_qw.double(), Cin)
    # The kernel ran on the convT weight itself (re-layout), not a copy of it.
    w = m.weight.double()
    assert ((qw - w).norm() / w.norm()) < 0.02
    ref = F.conv_transpose2d(qi, qw, None, s, p, op) * 2.0 ** -e
    ref = ref + (m.bias.double() * 2.0 ** -e).round().view(1, -1, 1, 1)
    assert m._last_acc.shape == ref.shape == y.shape
    assert not m._last_sat.any()
    assert torch.equal(m._last_acc.double(), ref)


@pytest.mark.parametrize("mode", list(MODES))
def test_matches_fp32_convT(mode):
    shape = SHAPES[0]
    m = _layer(shape, mode)
    x = _x(shape)
    _set_exact_e_min(m, x, mode)
    with torch.no_grad():
        y = m(x)
        ref = F.conv_transpose2d(x, m.weight, m.bias, 2, 1)
    sqnr = 10 * torch.log10((ref ** 2).sum() / ((y - ref) ** 2).sum())
    assert sqnr > 30


def test_dilate_input_noop_and_shape():
    t = torch.randn(1, 2, 3, 4)
    assert dilate_input(t) is t
    d = dilate_input(t, 2, 1)
    assert d.shape == (1, 2, 6, 8)
    assert torch.equal(d[..., :5:2, :7:2], t) and d.abs().sum() == t.abs().sum()


def test_groups_and_crop_rejected():
    with pytest.raises(NotImplementedError, match="groups"):
        MXConvTranspose2dHW(2, 2, 4, stride=2, padding=1, groups=2, mx_specs=_specs())
    with pytest.raises(NotImplementedError, match="crop"):
        MXConvTranspose2dHW(8, 2, 2, stride=2, padding=2, mx_specs=_specs())


# ---------------------------------------------------------------------------
# STE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("shape", [SHAPES[0], SHAPES[3]])
def test_ste_grads(shape, mode):
    Cin, _, _, s, p, op, _, _ = shape
    m = _layer(shape, mode)
    x = _x(shape)
    _set_exact_e_min(m, x, mode)
    m.train()
    x = x.clone().requires_grad_(True)
    y = m(x)
    g = torch.randn_like(y)
    y.backward(g)
    qi = m._last_qi[:, :Cin].detach()
    qw = _to_convT(m._last_qw.detach(), Cin).contiguous()
    xr = qi.clone().requires_grad_(True)
    wr = qw.clone().requires_grad_(True)
    yr = F.conv_transpose2d(xr, wr, None, s, p, op)
    gx, gw = torch.autograd.grad(yr, (xr, wr), g)
    assert m.weight.grad.shape == m.weight.shape        # convT layout
    assert torch.allclose(x.grad, gx, atol=1e-4, rtol=1e-4)
    assert torch.allclose(m.weight.grad, gw, atol=1e-3, rtol=1e-4)
    assert torch.allclose(m.bias.grad, g.sum((0, 2, 3)), atol=1e-4)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class _One(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.up = m

    def forward(self, x):
        return self.up(x)


@pytest.mark.parametrize("mode", list(MODES))
def test_calibration_on_undilated_input(mode):
    """Grids come from the real input: min Ea over real blocks only (the
    inserted zeros would pull it to 0 - 6), e_qa_out from the convT output."""
    shape = SHAPES[0]
    m = _layer(shape, mode, out_window="infer")
    g = torch.Generator().manual_seed(4)
    x = (torch.rand(2, 96, 5, 36, generator=g) + 0.6) * 4    # every block E >= 1
    calibrate_hw_grids(_One(m), [x], num_batches=1, verbose=0)
    e_min, e_qa_out = m.e_layer_min, m.e_qa_out

    _set_exact_e_min(m, x, mode)        # taps the operands the layer quantizes
    act, wt = MODES[mode]
    e_a, e_w = _min_grid_exps(m._last_qi, m._last_qw, 32,
                              act_blockify=act, weight_blockify=wt)
    assert e_min == e_a + e_w
    e_a_dil, _ = _min_grid_exps(dilate_input(m._last_qi, 2), m._last_qw, 32,
                                act_blockify=act, weight_blockify=wt)
    assert e_a_dil == -6 < e_a

    out = F.conv_transpose2d(m._last_qi[:, :96], _to_convT(m._last_qw, 96),
                             m.bias, 2, 1)
    assert e_qa_out == e_qa_out_from_absmax(float(out.abs().max()), 22)


# ---------------------------------------------------------------------------
# Quantizer wiring, hw_grids round trip, stats
# ---------------------------------------------------------------------------

class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.c1 = nn.Conv2d(32, 32, 3, padding=1)
        self.up = nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1)
        self.dw = nn.ConvTranspose2d(2, 2, 4, stride=2, padding=1, groups=2)

    def forward(self, x):
        return self.dw(self.up(torch.relu(self.c1(x))))


def _write_cfg(tmp, convT="hw", hw_grids=None, top=None):
    xb = {"enabled": True, "mode": "hw_fixed_point", "bits": 48,
          "backend": "python", "sat_mode": "per_product", "out_window": "infer",
          "act_blockify": "xblock", "weight_blockify": "flatten", "verbose": 0}
    if convT is not None:
        xb["convT"] = convT
    cfg = {"mx_specs": {"w_elem_format": "int8", "a_elem_format": "int8",
                        "block_size": 32, "custom_cuda": False,
                        "xblock_accum": xb},
           "layers": ["c1", "up", "dw"], "ptq": False, "measure_error": False}
    if hw_grids is not None:
        cfg["hw_grids"] = hw_grids
    cfg.update(top or {})
    with open(os.path.join(tmp, "mx_config.json"), "w") as f:
        json.dump(cfg, f)


def _data(n=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(2, 32, 6, 40, generator=g) * 3 for _ in range(n)]


def _eval(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)


def test_default_stays_plain():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp, convT=None)
    model = MXQuantizer(save_dir=tmp).quant(_Net())
    assert type(model.up) is MXConvTranspose2d
    assert type(model.dw) is MXConvTranspose2d


def test_bad_convT_key_raises():
    with pytest.raises(ValueError, match="convT"):
        normalize_xblock_accum({"convT": "yes"})


def test_wiring_state_dict_and_stats():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp)
    net = _Net()
    sd = {k: v.shape for k, v in net.state_dict().items()}
    model = MXQuantizer(save_dir=tmp).quant(net)
    assert type(model.up) is MXConvTranspose2dHW
    assert model.up._mx_layer_name == "up"
    assert type(model.dw) is MXConvTranspose2d           # grouped: plain MX
    assert {k: v.shape for k, v in model.state_dict().items()} == sd
    x = torch.randn(2, 32, 6, 40)
    assert torch.allclose(_fp32_functional(model.up, x),
                          F.conv_transpose2d(x, model.up.weight, model.up.bias, 2, 1))


def test_hw_grids_round_trip_bit_identical():
    tmp = tempfile.mkdtemp()
    _write_cfg(tmp)
    q = MXQuantizer(save_dir=tmp)
    model = q.quant(_Net())
    calibrate_hw_grids(model, _data(), num_batches=3, verbose=0)
    x = _data(1, seed=9)[0]
    y = _eval(model, x)
    assert model.up._win_sat_seen_life == 0

    path = q.save_hw_grids(model)
    grids = load_hw_grids(path)
    assert set(grids) == {"c1", "up"}
    assert grids["up"]["e_layer_min"] == model.up.e_layer_min
    assert grids["up"]["e_qa_out"] == model.up.e_qa_out

    model2 = MXQuantizer(save_dir=tmp, config_path=path).quant(_Net())
    assert model2.up._grid_src == {"e_layer_min": "config", "e_qa_out": "config"}
    assert torch.equal(_eval(model2, x), y)


# ---------------------------------------------------------------------------
# Triton == python (GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not (torch.cuda.is_available() and _HAS_TRITON),
                    reason="needs CUDA + triton")
@pytest.mark.parametrize("variant", ["reuse", "opt"])
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("window", [False, True])
def test_triton_matches_python(shape, mode, variant, window):
    if mode == "channel" and variant == "opt":
        pytest.skip("'opt' is NPE only")
    x = _x(shape).cuda()
    outs = []
    for backend in ("python", "triton"):
        m = _layer(shape, mode, backend=backend, npe_triton_variant=variant,
                   out_window="infer" if window else "off").cuda()
        _set_exact_e_min(m, x, mode)
        m.e_layer_min += 2                     # some products shift right
        if window:
            m.e_qa_out = m.e_layer_min + 12
        with torch.no_grad():
            outs.append(m(x))
    assert torch.equal(outs[0], outs[1])
