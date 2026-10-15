"""
sim_config.py - the one config both sides read (pure json, no MX lib import).

    cfg = load_config("mx_config_dof.json")
    cfg["mx_specs"]["scale_bits"]     # shared-exponent width, ALG and HW

A framework mx_config.json, read as is. main_ALG builds its MXQuantizer
config from it; main_HW maps the same mx_specs onto the HW emulator.
"""

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(_HERE, "mx_config_dof.json")


def load_config(path=DEFAULT_CONFIG, layer=None, backend=None):
    """Framework mx_config.json -> sim config dict.

    Returns {"file", "mx_specs", "groups", "entries", "layer", "backend"}.
    Nothing in the file is changed except what --layer / --backend name.
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
            "layer": layer, "backend": backend}
