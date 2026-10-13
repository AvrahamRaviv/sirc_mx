"""
taps.py - log any intermediate value under a name.

    taps = {}
    tap(taps, "a_int", a_int)      # anywhere in ALG or HW
    show(taps, "conv3x3 ALG")

A tap dict keeps insertion order, so it reads in pipeline order. Both sides use
the same names; main.py compares the names they share.
"""

import torch


def tap(taps, name, t):
    """Record `t` (tensor or number) as `name`. Returns `t`, so it can wrap an
    expression: y = tap(taps, "acc", acc)."""
    taps[name] = t.detach().cpu() if torch.is_tensor(t) else torch.tensor(t)
    return t


def is_int(t):
    return not (t.is_floating_point() or t.is_complex())


def show(taps, title=""):
    print(f"\n=== {title} ===")
    print(f"  {'tap':<12} {'shape':<20} {'dtype':<9} range")
    for k, t in taps.items():
        rng = f"[{t.min().item():g} .. {t.max().item():g}]" if t.numel() else "[]"
        dt = str(t.dtype).replace("torch.", "")
        print(f"  {k:<12} {str(tuple(t.shape)):<20} {dt:<9} {rng}")
