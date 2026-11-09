"""
sim_config.py - the one config both sides read (pure json, no MX lib import).

    cfg = load_config("mx_config_dof.json")
    cfg["mx_specs"]["scale_bits"]     # shared-exponent width, ALG and HW

A framework mx_config.json, read as is. main_ALG builds its MXQuantizer
config from it; main_HW maps the same mx_specs onto the HW emulator.
Per-call-site fxp lists are read by the lib's one rule (fixed_point/fxp_sites,
pure python), so both sides accept and refuse the same configs.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from fixed_point.fxp_sites import check_site_list, entry_call_sites, expand_sites   # noqa: E402

DEFAULT_CONFIG = os.path.join(_HERE, "mx_config_dof.json")


# What a side raises when it cannot run a case as configured - a documented
# refusal, with a message naming what and why: NotImplementedError (the HW
# code cannot run it, or would give wrong numbers without an error) and
# SystemExit (the config lacks what the sim reads, e.g. static grids: the sim
# never calibrates). A CLI over many cases reports each and goes on; any other
# error is a failure.
REFUSED = (NotImplementedError, SystemExit)


def require(d, keys, where, why):
    """Refuse (SystemExit, a REFUSED) unless mapping `d` holds every key:
    `<where> lacks [<missing keys>]; <why>`. `where` names the config entry
    (file first), `why` what the sim reads from it."""
    miss = [k for k in keys if k not in d]
    if miss:
        raise SystemExit(f"{where} lacks {miss}; {why}")


class SiteListError(ValueError):
    """The lib's per-call-site length error (fixed_point/fxp_sites
    check_site_list: a per-site fxp list that does not hold one spec per call
    site), as the sim raises it: a config the lib itself refuses, told by
    type from any other ValueError (a bug)."""


def load_config(path=DEFAULT_CONFIG, layer=None, backend=None):
    """Framework mx_config.json -> sim config dict.

    Refused (SystemExit) unless the file has global mx_specs with the
    element formats. Returns {"file", "mx_specs", "groups", "entries",
    "hw_grids", "layer", "backend"}. Nothing in the file is changed except what --layer / --backend
    name. hw_grids: the file's static per-layer grids (calibrated offline; the
    sim never calibrates), {} when the file has none.
    """
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(_HERE, path)
    with open(path) as f:
        raw = json.load(f)
    _need_global_formats(raw, path)
    entries = {}
    for e in raw.get("layers", []):
        e = {"name": e} if isinstance(e, str) else e
        entries[e["name"]] = e
    if layer is not None and layer not in entries:
        raise SystemExit(f"--layer {layer!r} not in {path} layers")
    return {"file": path, "mx_specs": raw["mx_specs"],
            "groups": raw.get("groups", {}), "entries": entries,
            "hw_grids": raw.get("hw_grids", {}),
            "layer": layer, "backend": backend}


def _need_global_formats(raw, path):
    """The sim runs one layer on the file's global mx_specs, both sides: a
    file with none (groups only), or whose mx_specs leave the element
    formats to a per-layer assignment (auto mixed precision), is refused."""
    sp = raw.get("mx_specs")
    if not isinstance(sp, dict):
        raise SystemExit(f"{os.path.basename(path)}: no global mx_specs (per-layer "
                         f"groups only); the sim runs one layer on the file's mx_specs")
    miss = [k for k in ("a_elem_format", "w_elem_format") if k not in sp]
    if miss:
        raise SystemExit(f"{os.path.basename(path)}: mx_specs has no {', '.join(miss)} "
                         f"(formats assigned per layer); the sim reads them from mx_specs")


def entry_name(cfg, op):
    """Name of the `layers` entry of the layer under test: --layer if given,
    else for warp the file's first act_quant entry, else None (conv / convT
    run on the plain global mx_specs)."""
    name = cfg["layer"]
    if name is None and op == "warp":
        name = next((n for n, e in cfg["entries"].items()
                     if e.get("kind") == "act_quant"), None)
        if name is None:
            raise SystemExit(f"{cfg['file']}: no act_quant entry for warp")
    return name


# The sim runs one forward of a fresh module on each side, so both read the
# config's call site 0 of a per-site (call_sites) entry.
WARP_CALL_SITE = 0


def sites_fxp(entry, fxp, where):
    """One operand's fxp spec per call site of `entry`, by the lib's rule: a
    per-site list must hold one spec per call site (SiteListError, a
    ValueError, otherwise), a single spec is shared by every site, and so is
    a bare null / true / false (the lib reads them as off / defaults / off)."""
    n = entry_call_sites(entry, where)
    if fxp is None or isinstance(fxp, bool):
        return [fxp] * n
    try:
        check_site_list(fxp, n, where)
    except ValueError as e:
        raise SiteListError(str(e)) from e
    return expand_sites(fxp, n, where)


def site_fxp(entry, fxp, site, where):
    """One operand's fxp spec at call site `site` of `entry` (sites_fxp)."""
    return sites_fxp(entry, fxp, where)[site]


# The warp entry's inputs, in order: the reference frame and the flow grid.
WARP_INPUTS = ("ref", "grid")


def warp_sites(cfg, entry):
    """The fxp specs per call site (sites_fxp) of the warp entry's inputs, in
    WARP_INPUTS order. `entry` is the warp entry of `cfg` (or a copy). Refused,
    naming the entry and the input, when it has no such input or the input
    has no `fxp`: the sim reads both operand formats from the config."""
    f = os.path.basename(cfg["file"])
    name = entry_name(cfg, "warp")
    inputs = entry.get("inputs")
    inputs = inputs if isinstance(inputs, list) else []
    sites = []
    for i, role in enumerate(WARP_INPUTS):
        why = f"the sim reads the warp's {role} format from the config"
        if i >= len(inputs) or not isinstance(inputs[i], dict):
            raise SystemExit(f"{f}: warp entry {name!r} has no input {i} ({role}); {why}")
        require(inputs[i], ("fxp",), f"{f}: warp entry {name!r} input {i} ({role})", why)
        sites.append(sites_fxp(entry, inputs[i]["fxp"], f"{entry['name']} input {i}"))
    return sites


def grid_key(cfg, case):
    """Name of the case's hw_grids entry: --layer if given, else the case name."""
    return cfg["layer"] or case["name"]


def window_on(cfg):
    """True when the config runs the static output window (needs e_qa_out)."""
    xa = cfg["mx_specs"].get("xblock_accum")
    return isinstance(xa, dict) and xa.get("out_window", "off") != "off"


def hw_grid(cfg, case):
    """The case's hw_grids entry (a copy), or None. Raises when the window is on
    and neither hw_grids nor xblock_accum gives the grids: the sim reads static
    grids from the config and never calibrates."""
    key = grid_key(cfg, case)
    g = cfg["hw_grids"].get(key)
    if g is not None:
        return dict(g)
    xa = cfg["mx_specs"].get("xblock_accum") or {}
    if window_on(cfg) and (xa.get("e_layer_min") is None or xa.get("e_qa_out") is None):
        raise SystemExit(
            f"{os.path.basename(cfg['file'])}: no hw_grids entry {key!r} and "
            f"xblock_accum has no e_layer_min/e_qa_out, but out_window is on. "
            f"The sim does not calibrate: write the grids into the config "
            f"(gen_hw_grids.py).")
    return None
