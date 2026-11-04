"""xblock_accum.train_path 'fast': the training forward of MXConv2dHW as a
float conv on the HW path's MX operands, with the accumulator range, the bias
grid and the output window as fake-quant. Eval stays on the bit-exact kernel.

What is pinned here:
  * the default ('hw') trains on the kernel, unchanged;
  * 'fast' equals the kernel while the sum fits fp32 (24 bits), and is within
    1 output LSB of it when it does not (a wide-range input);
  * the gradients are the kernel's STE gradients, ste_mask included;
  * eval with 'fast' set is the kernel, bit for bit;
  * calibrate_hw_grids(force=True) in the middle of QAT works with 'fast';
  * the fast training step reads nothing back from the device.
"""
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

from fixed_point.mx_fixed_point import _get_xblock_cfg, normalize_xblock_accum
from fixed_point.mx_fixed_point_hw import calibrate_hw_grids, clamp_block_exp, floor_log2
from mx_layers_blocked import MXConv2dHW
from mx_quantizer import MXQuantizer

NPE = {"weight_blockify": "flatten", "act_blockify": "xblock"}


class _Net(nn.Module):
    """A DOF-shaped stack: stem, stride 2, a 2x convT, PReLU between."""

    def __init__(self, cin=6):
        super().__init__()
        torch.manual_seed(3)
        self.body = nn.Sequential(
            nn.Conv2d(cin, 16, 3, padding=1), nn.PReLU(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.PReLU(32),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.PReLU(16),
            nn.Conv2d(16, 8, 3, padding=1))

    def forward(self, x):
        return self.body(x)


LAYERS = ["body.0", "body.2", "body.4", "body.6"]


def _model(train_path=None, net=None, layers=LAYERS, groups=None, **xblock):
    tmp = tempfile.mkdtemp()
    xb = {"enabled": True, "mode": "hw_fixed_point", "bits": 48,
          "backend": "python", "sat_mode": "per_product", "e_layer_min": None,
          "out_window": "always", "convT": "hw", "verbose": 0, **NPE, **xblock}
    if train_path is not None:
        xb["train_path"] = train_path
    cfg = {"mx_specs": {"w_elem_format": "int8", "a_elem_format": "int8",
                        "block_size": 32, "custom_cuda": False, "xblock_accum": xb},
           "layers": layers, "ptq": False, "measure_error": False}
    if groups:
        cfg["groups"] = groups
    with open(os.path.join(tmp, "mx_config.json"), "w") as f:
        json.dump(cfg, f)
    return MXQuantizer(save_dir=tmp).quant(net or _Net())


def _data(n=3, seed=1, cin=6, tiny=None):
    """Inputs; `tiny` scales two rows by 2^tiny (the DOF wide-range input)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randn(1, cin, 8, 40, generator=g)
        if tiny is not None:
            x[..., :2, :] *= 2.0 ** tiny
        out.append(x)
    return out


def _hw_layers(model):
    layers = [m for m in model.modules() if isinstance(m, MXConv2dHW)]
    assert layers, "no MXConv2dHW in the model"
    return layers


def _set_path(model, path):
    for m in _hw_layers(model):
        _get_xblock_cfg(m)["train_path"] = path


def _layer_io(model, x):
    """(layer, its input, its output) of every HW layer in one forward."""
    rec = []
    hooks = [m.register_forward_hook(lambda mod, a, o: rec.append((mod, a[0].detach(), o.detach())))
             for m in _hw_layers(model)]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    assert rec
    return rec


def _fast_vs_hw(model, x):
    """Per HW layer, both train-mode outputs on the input the 'hw' run gave it,
    with that layer's output LSB (2^e_qa_out)."""
    model.train()
    _set_path(model, "hw")
    rows = []
    for m, xin, y_hw in _layer_io(model, x):
        _get_xblock_cfg(m)["train_path"] = "fast"
        with torch.no_grad():
            y_fast = m(xin)
        _get_xblock_cfg(m)["train_path"] = "hw"
        rows.append((m, y_hw, y_fast, 2.0 ** int(m.e_qa_out)))
    return rows


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_train_path_key():
    assert normalize_xblock_accum(True)["train_path"] == "hw"
    assert normalize_xblock_accum({"train_path": "fast"})["train_path"] == "fast"
    with pytest.raises(ValueError, match="train_path"):
        normalize_xblock_accum({"train_path": "float"})


def test_clamp_block_exp_count_without_sync():
    q = torch.tensor([2.0 ** -30, 3.0, -2.0 ** -40])
    y, n = clamp_block_exp(q, -15, 4)
    y2, n2 = clamp_block_exp(q, -15, 4, sync=False)
    assert torch.is_tensor(n2) and n2.dim() == 0 and int(n2) == n == 3   # 2 cut, 1 sat
    assert torch.equal(y, y2)


# ---------------------------------------------------------------------------
# default 'hw' trains on the kernel
# ---------------------------------------------------------------------------

def test_default_trains_on_the_kernel():
    """e_layer_min pinned above the products' grid: the kernel floors every
    product, the float sum does not, so the two differ and the default must
    give the kernel's answer."""
    data = _data()
    a, b = _model(), _model(train_path="hw")
    for m in (a, b):
        calibrate_hw_grids(m, data, num_batches=3, verbose=0)
        for layer in _hw_layers(m):
            layer.e_layer_min += 4
        m.train()
    x = data[0]
    with torch.no_grad():
        ya, yb = a(x), b(x)
    assert torch.equal(ya, yb)
    rows = _fast_vs_hw(a, x)
    assert any(not torch.equal(y_hw, y_fast) for _, y_hw, y_fast, _ in rows)


# ---------------------------------------------------------------------------
# fast forward vs the kernel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("xblock", [
    {},
    {"out_round": "nearest", "bias_round": "half_up", "mant_round": "truncate"},
    {"overflow": "raise", "overflow_saturate": ["out_window"]},
    {"in_exp_bits": 4, "in_exp_mode": "always"},
    {"out_window": "off"},
], ids=["window", "hw_rounding", "sat_word", "in_exp", "no_window"])
def test_fast_equals_kernel_when_sum_fits_fp32(xblock):
    model = _model(**xblock)
    data = _data()
    calibrate_hw_grids(model, data, num_batches=3, verbose=0)
    rows = _fast_vs_hw(model, data[0])
    assert len(rows) == 4
    for m, y_hw, y_fast, _ in rows:
        assert torch.equal(y_hw, y_fast), m._mx_layer_name


def test_fast_takes_the_in_exp_register():
    """Rows 2^-20 below the rest: the 4b register cuts them in training
    ('always'), base raised 6 codes so it cuts real values too, in both
    paths alike. The wide input leaves a low e_layer_min
    (calibration sees the uncut input), so the sum passes 24 bits: 1 LSB."""
    model = _model(in_exp_bits=4, in_exp_mode="always")
    data = _data(tiny=-20)
    calibrate_hw_grids(model, data, num_batches=3, verbose=0)
    for m in _hw_layers(model):
        m.e_base_a += 6                 # the register also cuts the normal rows
    rows = _fast_vs_hw(model, data[0])
    assert int(rows[0][0]._in_exp_cut_life[0]) > 0
    for m, y_hw, y_fast, lsb in rows:
        assert ((y_fast - y_hw).abs() <= lsb).all(), m._mx_layer_name


@pytest.mark.parametrize("xblock", [{}, {"overflow": "raise",
                                      "overflow_saturate": ["out_window"]}],
                         ids=["clamp", "sat_word"])
def test_fast_saturates_the_window_like_the_kernel(xblock):
    """Output grid 3 bits too fine: outputs saturate the window, to
    +-(2^22 - 1) or to the HW word. One layer, so its input stays on the
    calibrated grid (a finer input puts products below it: see
    test_default_trains_on_the_kernel)."""
    net = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1))
    model = _model(net=net, layers=["0"], **xblock)
    data = _data()
    calibrate_hw_grids(model, data, num_batches=3, verbose=0)
    m, = _hw_layers(model)
    m.e_qa_out -= 3
    (m, y_hw, y_fast, lsb), = _fast_vs_hw(model, data[0])
    top = (127 * 2 ** 15 if xblock else 2 ** 22 - 1) * lsb
    assert (y_hw.abs() == top).any()
    assert torch.equal(y_hw, y_fast)


@pytest.mark.parametrize("out_round", ["floor", "nearest"])
def test_fast_within_one_lsb_on_wide_range_input(out_round):
    """Rows 2^-20 below the rest: the accumulator holds more than 24 bits, the
    fp32 sum rounds, and the window lands at most 1 LSB off, near a tie."""
    net = nn.Sequential(nn.Conv2d(256, 32, 3, padding=1))
    model = _model(net=net, layers=["0"], out_round=out_round)
    data = _data(cin=256, tiny=-20)
    calibrate_hw_grids(model, data, num_batches=3, verbose=0)
    (m, y_hw, y_fast, lsb), = _fast_vs_hw(model, data[0])
    diff = (y_fast - y_hw).abs() / lsb
    assert diff.max() <= 1
    assert 0 < (diff > 0).float().mean() < 0.2      # it rounds, and rarely


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

def _grads(model, x, path):
    """Grads of a loss linear in the output (a fixed upstream gradient), so
    they compare the backward alone, not two forwards."""
    model.train()
    _set_path(model, path)
    model.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_(True)
    y = model(x)
    up = torch.randn(y.shape, generator=torch.Generator().manual_seed(11))
    (y * up).sum().backward()
    g = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    g["x"] = x.grad.clone()
    return g


@pytest.mark.parametrize("net,layers,xblock", [
    (nn.Sequential(nn.Conv2d(6, 16, 3, padding=1)), ["0"], {}),
    (nn.Sequential(nn.Conv2d(6, 16, 3, stride=2, padding=1, dilation=1)), ["0"],
     {"in_exp_bits": 4, "in_exp_mode": "always"}),
    (nn.Sequential(nn.ConvTranspose2d(6, 16, 4, stride=2, padding=1)), ["0"], {}),
    (nn.Sequential(nn.ConvTranspose2d(6, 16, 3, stride=2, padding=1,
                                      output_padding=1)), ["0"], {"out_window": "off"}),
], ids=["conv", "conv_s2_in_exp", "convT", "convT_outpad"])
def test_fast_grads_equal_kernel_ste(net, layers, xblock):
    """One layer, same input: weight, bias and input grads equal the kernel's
    STE (`HWFxpConv2dFn.backward`), bit for bit."""
    torch.manual_seed(0)
    model = _model(net=net, layers=layers, **xblock)
    data = _data()
    calibrate_hw_grids(model, data, num_batches=3, verbose=0)
    g_hw, g_fast = _grads(model, data[0], "hw"), _grads(model, data[0], "fast")
    assert set(g_hw) == set(g_fast) and len(g_hw) == 3
    for k in g_hw:
        assert torch.equal(g_hw[k], g_fast[k]), k


def test_fast_grads_ste_mask_zero_saturated_lanes():
    """A 32b accumulator on a fine grid saturates some lanes: with ste_mask
    both paths zero their gradient."""
    net = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1))
    model = _model(net=net, layers=["0"], bits=32, ste_mask=True, out_window="off")
    data = _data()
    calibrate_hw_grids(model, data, num_batches=3, verbose=0)
    m, = _hw_layers(model)
    model.eval()
    with torch.no_grad():
        top = floor_log2(float(model(data[0]).abs().max()))
    m.e_layer_min = top - 31          # the largest outputs pass 2^31 grid steps
    g_hw = _grads(model, data[0], "hw")
    sat_hw = int(m._sat_seen_life)
    g_fast = _grads(model, data[0], "fast")
    sat_fast = int(m._sat_seen_life) - sat_hw
    assert 0 < sat_fast == sat_hw < m._sat_total_life / 2    # some, not all
    for k in g_hw:
        assert torch.equal(g_hw[k], g_fast[k]), k
    (_, y_hw, y_fast, _), = _fast_vs_hw(model, data[0])
    hi = (2 ** 31 - 1) * 2.0 ** m.e_layer_min
    sat = y_hw.abs() >= hi
    assert sat.any() and torch.equal(y_hw[sat], y_fast[sat])   # clamped alike


def test_fast_counters_never_mutate_a_saved_value():
    """The fast path counts on the device (0-d tensors). A caller that saves
    a counter and subtracts later (mx_debug) must get the step's count: a
    counter is replaced, never added to in place."""
    net = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1))
    model = _model(net=net, layers=["0"], bits=32)
    data = _data()
    calibrate_hw_grids(model, data, num_batches=3, verbose=0)
    m, = _hw_layers(model)
    model.eval()
    with torch.no_grad():
        top = floor_log2(float(model(data[0]).abs().max()))
    m.e_layer_min = top - 31          # some lanes saturate the accumulator
    m.e_qa_out -= 3                   # and the window clips the largest outputs
    _set_path(model, "fast")
    model.train()
    counters = ("_sat_seen_life", "_sat_seen_window", "_win_sat_seen_life")
    with torch.no_grad():
        start = {k: int(getattr(m, k)) for k in counters}
        model(data[0])                # from here on the counters are tensors
        saved = {k: getattr(m, k) for k in counters}
        first = {k: int(v) - start[k] for k, v in saved.items()}
        model(data[0])
    for k in counters:
        assert first[k] > 0, k
        assert int(saved[k]) == start[k] + first[k], f"{k} added to in place"
        assert int(getattr(m, k)) == start[k] + 2 * first[k], k


# ---------------------------------------------------------------------------
# eval, calibration, syncs, shared layers, int input
# ---------------------------------------------------------------------------

def test_eval_with_fast_is_the_kernel():
    """Wide-range input, where fast and the kernel differ in train mode: in
    eval both settings give the kernel's output, bit for bit."""
    data = _data(tiny=-20)
    a, b = _model(), _model(train_path="fast")
    for m in (a, b):
        calibrate_hw_grids(m, data, num_batches=3, verbose=0)
        for layer in _hw_layers(m):
            layer.e_layer_min += 4          # the kernel floors: fast would differ
        m.eval()
    with torch.no_grad():
        assert torch.equal(a(data[0]), b(data[0]))
    b.train()
    with torch.no_grad():
        assert not torch.equal(a.eval()(data[0]), b(data[0]))


def test_calibrate_force_during_fast_training():
    data = _data()
    a, b = _model(), _model(train_path="fast")
    ra = calibrate_hw_grids(a, data, num_batches=3, verbose=0)
    b.train()
    rb = calibrate_hw_grids(b, data, num_batches=3, verbose=0, force=True)
    assert b.training
    assert ra.keys() == rb.keys() and len(ra) == 4
    for k in ra:
        assert (ra[k]["e_layer_min"], ra[k]["e_qa_out"]) == \
            (rb[k]["e_layer_min"], rb[k]["e_qa_out"]), k
    x = data[0].clone().requires_grad_(True)
    b(x).sum().backward()
    assert x.grad.abs().sum() > 0


class _Reads:
    """Counts tensor -> host reads (a device sync on CUDA)."""

    def __init__(self):
        self.n = 0

    def __enter__(self):
        self._orig = {k: getattr(torch.Tensor, k) for k in
                      ("item", "__float__", "__int__", "__bool__", "tolist")}
        for k, f in self._orig.items():
            def w(t, *a, _f=f, **kw):
                self.n += 1
                return _f(t, *a, **kw)
            setattr(torch.Tensor, k, w)
        return self

    def __exit__(self, *exc):
        for k, f in self._orig.items():
            setattr(torch.Tensor, k, f)


def test_fast_training_step_reads_nothing_back():
    model = _model(train_path="fast")
    data = _data()
    calibrate_hw_grids(model, data, num_batches=3, verbose=0)
    model.train()
    with _Reads() as fast:
        model(data[0]).sum().backward()
    _set_path(model, "hw")
    with _Reads() as hw:
        model(data[0]).sum().backward()
    assert fast.n == 0
    assert hw.n >= 4                    # the kernel path counts on the host
    m = _hw_layers(model)[0]
    assert m._fwd_count == 2 and m._sat_total_life > 0


def test_fast_shared_layer_called_twice():
    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(5)
            self.c = nn.Conv2d(6, 6, 3, padding=1)

        def forward(self, x):
            return self.c(self.c(x))

    data = _data()
    model = _model(train_path="fast", net=Twice(), layers=["c"])
    calibrate_hw_grids(model, data, num_batches=3, verbose=0)
    m, = _hw_layers(model)
    assert m.hw_calls == 2
    model.train()
    model(data[0]).sum().backward()
    _set_path(model, "hw")
    with torch.no_grad():
        y_hw = model(data[0])
    _set_path(model, "fast")
    with torch.no_grad():
        assert torch.equal(model(data[0]), y_hw)


def test_fast_int_input_layer():
    """in_mode 'int' (the first layer): fast sees the same int8 input."""
    groups = {"inp": {"w_elem_format": "int8", "a_elem_format": "int8",
                      "block_size": 32, "custom_cuda": False,
                      "xblock_accum": {"enabled": True, "mode": "hw_fixed_point",
                                       "bits": 48, "backend": "python",
                                       "e_layer_min": None, "out_window": "always",
                                       "verbose": 0, "in_mode": "int", **NPE}}}
    net = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1))
    model = _model(net=net, layers=[{"name": "0", "group": "inp"}], groups=groups)
    data = _data()
    calibrate_hw_grids(model, data, num_batches=3, verbose=0)
    (m, y_hw, y_fast, _), = _fast_vs_hw(model, data[0])
    assert m.in_scale is not None
    assert torch.equal(y_hw, y_fast)
