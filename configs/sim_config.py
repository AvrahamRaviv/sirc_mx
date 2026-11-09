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

from fixed_point.fxp_sites import entry_call_sites, expand_sites   # noqa: E402

DEFAULT_CONFIG = os.path.join(_HERE, "mx_config_dof.json")


def load_config(path=DEFAULT_CONFIG, layer=None, backend=None):
    """Framework mx_config.json -> sim config dict.

    Returns {"file", "mx_specs", "groups", "entries", "hw_grids", "layer",
    "backend"}. Nothing in the file is changed except what --layer / --backend
    name. hw_grids: the file's static per-layer grids (calibrated offline; the
    sim never calibrates), {} when the file has none.
    """
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(_HERE, path)
    with open(path) as f:
        raw = json.load(f)
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


def site_fxp(entry, fxp, site, where):
    """One operand's fxp spec at call site `site` of `entry`, by the lib's
    rule: a per-site list must hold one spec per call site (ValueError
    otherwise), a single spec is shared by every site."""
    return expand_sites(fxp, entry_call_sites(entry, where), where)[site]


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
