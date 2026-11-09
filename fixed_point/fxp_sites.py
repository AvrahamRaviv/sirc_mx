"""Per-call-site fxp configs of an act_quant entry: the one rule for reading them.

    n = entry_call_sites(entry, "act_quant 'warp'")      # validated int >= 1
    per_site = expand_sites(fxp, n, "input 0")           # one config per call

A shared module instance called N times per forward (`call_sites: N`) can hold
one static fxp config per call. Everything that reads such an entry - the
quantizer's config parser, MXActQuant, and the simulator's HW side, which must
not import the MX lib - goes through these functions, so they all accept and
refuse the same configs. Pure python (no torch), for that reason.
"""

import copy


def entry_call_sites(entry, where):
    """The entry's `call_sites` (default 1). Raises unless an int >= 1."""
    n = entry.get("call_sites", 1)
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"{where}: call_sites must be an int >= 1, got {n!r}")
    return n


def check_site_list(fxp, call_sites, where):
    """Raise when a per-site fxp list does not hold exactly one config per
    call site; a single config passes."""
    if isinstance(fxp, (list, tuple)) and len(fxp) != call_sites:
        raise ValueError(
            f"{where}: {len(fxp)} per-call-site fxp configs for "
            f"call_sites={call_sites}")


def expand_sites(fxp, call_sites, where):
    """Normalize one operand's fxp config into a per-call-site list.

    A list is taken as given. A single config is shared across call sites
    when its scale is already known, and deep-copied per site when the
    scale is to be calibrated - a learned scale is a property of the tensor
    a given call sees, not of the module.
    """
    if fxp is None:
        return None
    check_site_list(fxp, call_sites, where)
    if isinstance(fxp, (list, tuple)):
        return [copy.deepcopy(c) for c in fxp]
    if fxp.get("calibrate") and call_sites > 1:
        return [copy.deepcopy(fxp) for _ in range(call_sites)]
    return [fxp] * call_sites
