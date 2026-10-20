"""Per-layer HW grids stored in an MX config: the top-level "hw_grids" section.

    "hw_grids": {
        "<layer name>": {"e_layer_min": -16, "e_qa_out": -20,
                         "win_bits": 22, "shift_out": -4}
    }

Names are the ones in "layers". Per layer:
  e_layer_min  accumulator grid E_acc (min(Ea) + min(Ew) - 2*mant_bias)
  e_qa_out     output grid the static window extracts at
  win_bits     window magnitude bits, saturation at +-(2^win_bits - 1)
  shift_out    e_qa_out - e_layer_min, the one extraction shift
               (> 0 right shift, < 0 left shift)
  e_base_a     input shared-exponent base (only with xblock_accum.in_exp_bits):
               input block exponents limited to [e_base_a, e_base_a + 2^bits - 1]
  e_base_w     same for the weights

e_layer_min and e_qa_out are required; win_bits and shift_out are derived
values, written by `export_hw_grids` and checked on load. Calibration runs
once offline (`calibrate_hw_grids`, `MXQuantizer.save_hw_grids`); inference
and the single-layer sim only read this section.

No torch / microxcaling import here, so the sim can read a config
standalone.
"""
import json

HW_GRID_KEYS = ("e_layer_min", "e_qa_out", "win_bits", "shift_out",
                "e_base_a", "e_base_w")
_REQUIRED = ("e_layer_min", "e_qa_out")


def check_hw_grid(name, entry, win_bits=None):
    """Validate one hw_grids entry and return it.

    `win_bits`, if given, is the width the layer's format and config imply;
    an entry that names a different one is rejected.
    """
    if not isinstance(entry, dict):
        raise TypeError(f"hw_grids[{name!r}] must be a dict, got "
                        f"{type(entry).__name__}")
    unknown = sorted(set(entry) - set(HW_GRID_KEYS))
    if unknown:
        raise ValueError(f"hw_grids[{name!r}]: unknown keys {unknown}; "
                         f"valid keys: {list(HW_GRID_KEYS)}")
    missing = [k for k in _REQUIRED if k not in entry]
    if missing:
        raise ValueError(f"hw_grids[{name!r}]: missing {missing}")
    for k, v in entry.items():
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"hw_grids[{name!r}][{k!r}] must be int, got {v!r}")
    shift = entry["e_qa_out"] - entry["e_layer_min"]
    if "shift_out" in entry and entry["shift_out"] != shift:
        raise ValueError(
            f"hw_grids[{name!r}]: shift_out={entry['shift_out']} but "
            f"e_qa_out - e_layer_min = {entry['e_qa_out']} - "
            f"{entry['e_layer_min']} = {shift}")
    if "win_bits" in entry:
        if entry["win_bits"] < 1:
            raise ValueError(f"hw_grids[{name!r}]: win_bits must be >= 1")
        if win_bits is not None and entry["win_bits"] != int(win_bits):
            raise ValueError(
                f"hw_grids[{name!r}]: win_bits={entry['win_bits']} but the "
                f"layer's format/config give {int(win_bits)}")
    return entry


def load_hw_grids(cfg):
    """The validated "hw_grids" section of a config (dict or json path);
    {} when absent."""
    if isinstance(cfg, str):
        with open(cfg) as f:
            cfg = json.load(f)
    grids = (cfg or {}).get("hw_grids") or {}
    if not isinstance(grids, dict):
        raise TypeError("hw_grids must be a dict of layer name -> entry")
    for name, entry in grids.items():
        check_hw_grid(name, entry)
    return grids


def hw_grid_for(cfg, name):
    """One layer's grids from a config (dict or json path), with shift_out
    filled in. Raises if the layer has no entry: inference never falls back
    to calibration."""
    grids = load_hw_grids(cfg)
    if name not in grids:
        raise KeyError(f"hw_grids has no entry for {name!r} "
                       f"({len(grids)} layers present)")
    entry = dict(grids[name])
    entry.setdefault("shift_out", entry["e_qa_out"] - entry["e_layer_min"])
    return entry
