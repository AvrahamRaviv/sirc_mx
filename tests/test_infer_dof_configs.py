"""configs/infer_dof/: one config per DOF inference experiment, each made to
generate its static HW grids offline (quant() with calibrate_hw_grids, which
writes <save_dir>/mx_config_grids.json).

Each config is checked as the owner runs it, on a CPU stand-in of the DOF net
that has the same layer names and kinds (HW convs, dense ConvTranspose2d,
grouped ConvTranspose2d, the shared warp called at two pyramid levels):
  * it is the named change from its parent config, and nothing else;
  * quant() with data writes mx_config_grids.json holding a grid for every
    layer of every kind the experiment puts on the HW path, the warp's static
    scales when the warp is on, and the shared-exponent bases with se4;
  * that grids file then loads and runs with no data (inference never
    calibrates).
The expected layers per config are declared here from configs/infer_dof/
README.md, never read back from what the library wrote.
"""

import copy
import glob
import json
import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "configs"))
sys.path.insert(0, _ROOT)
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects")
sys.path.insert(0, "/home/avrahamra/PycharmProjects")

from cases import Warp                    # noqa: E402  the float warp spec
from mx_layers_act import MXActQuant      # noqa: E402
from mx_layers_blocked import MXConv2dHW  # noqa: E402
from mx_layers_convT import MXConvTranspose2dHW  # noqa: E402
from mx_quantizer import MXQuantizer      # noqa: E402
from mx_quantizer import MXConvTranspose2d  # noqa: E402  plain MX convT

DIR = os.path.join(_ROOT, "configs", "infer_dof")
CH = 32                                   # one NPE block of channels

# DOF layer names, as the configs list them.
CONVS = ([f"model.block{b}.conv0.{i}.0" for b in (2, 3, 4) for i in (0, 1)]
         + [f"model.block{b}.convblock.{i}.0" for b in (2, 3, 4) for i in range(6)]
         + [f"model.feature_pyramid_extractor.convs.{i}.{j}.0"
            for i in range(3) for j in (0, 1)])
CONVT_HW = [f"model.block{b}.conv1" for b in (2, 3, 4)]               # dense
CONVT_PLAIN = [f"model.block{b}.convtranspose_2" for b in (2, 3)]     # grouped
WARP = "model.warp"
WARP_SITES = 2
# The warp operands as arch set them (2026-10-08): ref plain int8 with a
# calibrated static scale, grid Q8.8 signed 16-bit.
WARP_REF = {"total_bits": 8, "signed": True, "calibrate": True,
            "round": "half_away", "saturate": True}
WARP_GRID = {"total_bits": 16, "frac_bits": 8, "signed": True,
             "round": "half_away", "saturate": True}
# Where the README says the warp and convT entries come from.
SOURCE = os.path.join(_ROOT, "configs", "mx_config_dof_run_win_cal_convT.json")

# Config -> (parent, convT on, warp on, 4-bit SE register on): README.md.
EXPECTED = {
    "base":           (None,  False, False, False),
    "se4":            ("base", False, False, True),
    "se4_warp":       ("se4",  False, True,  True),
    "se4_convT":      ("se4",  True,  False, True),
    "se4_warp_convT": ("se4",  True,  True,  True),
}


def _load(name):
    with open(os.path.join(DIR, name + ".json")) as f:
        return json.load(f)


def _act(n=CH):
    return nn.Sequential(nn.Conv2d(n, n, 3, padding=1), nn.LeakyReLU(0.1))


class _Block(nn.Module):
    """conv0 (2) + convblock (6) on the HW path, a dense convT (conv1), and
    for block2/3 a grouped convT (convtranspose_2): DOF's names and kinds."""

    def __init__(self, grouped_convT):
        super().__init__()
        self.conv0 = nn.Sequential(_act(), _act())
        self.convblock = nn.Sequential(*[_act() for _ in range(6)])
        self.conv1 = nn.ConvTranspose2d(CH, CH, 4, stride=2, padding=1)
        self.convtranspose_2 = (nn.ConvTranspose2d(CH, CH, 4, stride=2, padding=1,
                                                   groups=CH)
                                if grouped_convT else None)

    def forward(self, x):
        y = self.convblock(self.conv0(x))
        up = self.conv1(y)
        if self.convtranspose_2 is not None:
            up = up + self.convtranspose_2(y)
        return F.avg_pool2d(up, 2)


class _FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.Sequential(_act(), _act()) for _ in range(3)])

    def forward(self, x):
        for c in self.convs:
            x = c(x)
        return x


class _DOF(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_pyramid_extractor = _FPN()
        self.warp = Warp()
        self.block2 = _Block(True)
        self.block3 = _Block(True)
        self.block4 = _Block(False)

    def forward(self, x, flow):
        f = self.warp(self.feature_pyramid_extractor(x), flow)   # call site 0
        f = self.block2(f)
        f = self.warp(f, flow)                                   # call site 1
        return self.block4(self.block3(f))


class _Wrapper(nn.Module):
    """The trainer's wrapper: DOF layer names start with "model."."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.model = _DOF()

    def forward(self, x, flow):
        return self.model(x, flow)


def _fwd(m, b):
    return m(b[0], b[1])


def _data(n=2):
    g = torch.Generator().manual_seed(1)
    return [(torch.randn(1, CH, 8, 8, generator=g),
             torch.randn(1, 2, 8, 8, generator=g)) for _ in range(n)]


def _cpu(cfg):
    """The configs are the owner's GPU run configs. With no CUDA, custom_cuda
    goes off (the sim does the same); triton falls back to python on CPU."""
    cfg = copy.deepcopy(cfg)
    if not torch.cuda.is_available():
        cfg["mx_specs"]["custom_cuda"] = False
        for g in cfg.get("groups", {}).values():
            g["custom_cuda"] = False
    return cfg


def _quant(cfg, tmp, data):
    with open(os.path.join(tmp, "mx_config.json"), "w") as f:
        json.dump(_cpu(cfg), f)
    return MXQuantizer(tmp).quant(_Wrapper(), data=data, forward_fn=_fwd)


def _names(model, cls):
    return sorted(n for n, m in model.named_modules() if type(m) is cls)


def test_folder_holds_exactly_the_documented_configs():
    found = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(DIR, "*.json")))
    assert found, f"no configs in {DIR}"
    assert found == sorted(EXPECTED)
    with open(os.path.join(DIR, "README.md")) as f:
        readme = f.read()
    for name in found:
        assert f"`{name}.json`" in readme, f"{name}.json is not in README.md"


def _name(entry):
    return entry if isinstance(entry, str) else entry["name"]


def _source_entries(names):
    """The `layers` entries for `names`, as the source config holds them."""
    with open(SOURCE) as f:
        src = json.load(f)
    by_name = {_name(l): l for l in src["layers"]}
    return src, [by_name[n] for n in names]


def _without(cfg, layers=(), xblock=(), top=()):
    c = copy.deepcopy(cfg)
    c["layers"] = [l for l in c["layers"]
                   if _name(l) not in layers]
    for k in xblock:
        del c["mx_specs"]["xblock_accum"][k]
    for k in top:
        del c[k]
    return c


@pytest.mark.parametrize("name", [n for n, e in EXPECTED.items() if e[0]])
def test_config_is_its_parent_plus_the_named_change(name):
    """Remove the named change and the parent is what is left, key for key."""
    parent, convT, warp, se4 = EXPECTED[name]
    cfg, par = _load(name), _load(parent)
    p_convT, p_warp, p_se4 = EXPECTED[parent][1:]
    xb = cfg["mx_specs"]["xblock_accum"]
    rm_layers, rm_xblock, rm_top = [], [], []
    if se4 and not p_se4:
        assert (xb["in_exp_bits"], xb["in_exp_mode"]) == (4, "always")
        rm_xblock += ["in_exp_bits", "in_exp_mode"]
    if convT and not p_convT:
        assert xb["convT"] == "hw"
        rm_layers += CONVT_HW + CONVT_PLAIN
    if warp and not p_warp:
        rm_layers.append(WARP)
        rm_top.append("calibrate_act_scales")
    assert _without(cfg, rm_layers, rm_xblock, rm_top) == par

    # What was added is the source's entries, key for key.
    added = [l for l in cfg["layers"] if _name(l) in rm_layers]
    assert sorted(map(_name, added)) == sorted(rm_layers)
    src, expected = _source_entries(rm_layers)
    for got, want in zip(sorted(added, key=_name), sorted(expected, key=_name)):
        assert got == want, f"{name}: {_name(got)} differs from {SOURCE}"
    if warp and not p_warp:
        assert cfg["calibrate_act_scales"] == src["calibrate_act_scales"]
        entry = next(l for l in added if _name(l) == WARP)
        assert entry["kind"] == "act_quant" and entry["call_sites"] == WARP_SITES
        assert [i["fxp"] for i in entry["inputs"]] == [WARP_REF, WARP_GRID]
        assert "output" not in entry      # README: the warp output stays float


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_quant_writes_grids_for_every_hw_layer_then_infers_from_them(name, tmp_path):
    _, convT, warp, se4 = EXPECTED[name]
    tmp = str(tmp_path)
    model = _quant(_load(name), tmp, _data())

    # The stand-in took the paths the experiment asks for.
    hw_convs = _names(model, MXConv2dHW)
    assert hw_convs, "no HW conv in the quantized stand-in"
    assert hw_convs == sorted(CONVS)
    assert _names(model, MXConvTranspose2dHW) == (sorted(CONVT_HW) if convT else [])
    assert _names(model, MXConvTranspose2d) == (sorted(CONVT_PLAIN) if convT else [])
    assert _names(model, MXActQuant) == ([WARP] if warp else [])

    path = os.path.join(tmp, "mx_config_grids.json")
    assert os.path.exists(path), "quant() wrote no mx_config_grids.json"
    with open(path) as f:
        out = json.load(f)
    grids = out["hw_grids"]
    assert grids, "mx_config_grids.json has an empty hw_grids"

    kinds = {"conv": CONVS, "convT (HW)": CONVT_HW if convT else []}
    for kind, layers in kinds.items():
        missing = [n for n in layers if n not in grids]
        assert not missing, f"{name}: {kind} layers with no grid: {missing}"
    assert sorted(grids) == sorted(CONVS + (CONVT_HW if convT else []))
    for n, g in grids.items():
        assert isinstance(g["e_layer_min"], int) and isinstance(g["e_qa_out"], int), n
        has_bases = "e_base_a" in g and "e_base_w" in g
        assert has_bases == se4, f"{name}: {n} shared-exponent bases {g}"

    entry = [l for l in out["layers"] if isinstance(l, dict) and l["name"] == WARP]
    if warp:
        ref = entry[0]["inputs"][0]["fxp"]
        assert isinstance(ref, list) and len(ref) == WARP_SITES, \
            f"{name}: warp ref scales not written per call site: {ref}"
        assert all(s["scale"] > 0 and "calibrate" not in s for s in ref)
    else:
        assert not entry

    # Inference: the grids file alone, no data, nothing calibrated.
    infer = os.path.join(tmp, "infer")
    os.makedirs(infer)
    with open(os.path.join(infer, "mx_config.json"), "w") as f:
        json.dump(out, f)
    q = MXQuantizer(infer).quant(_Wrapper())
    assert not os.path.exists(os.path.join(infer, "mx_config_grids.json"))
    q.eval()
    with torch.no_grad():
        y = _fwd(q, _data(1)[0])
    assert torch.isfinite(y).all()
