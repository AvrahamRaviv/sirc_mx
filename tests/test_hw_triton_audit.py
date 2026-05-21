"""Deep-debug audit for HW triton kernel.

Three suspected bugs in the saturating int64 accumulator path:

1. **int64 overflow at bits=64** — `acc + s` wraps int64 silently before the
   `acc > hi` / `acc < lo` saturation check runs, because the configured
   accumulator window already uses the entire int64 range. Saturation
   misfires and clamps to the wrong rail.

2. **Shift overflow `p << pos_shift`** — at int16, `p` has up to ~30 bits.
   When `Ea + Ew` range produces a shift > ~33, `p << shift` exceeds 2^63
   and wraps int64 silently. Bigger formats hit it sooner; bigger
   activation/weight dynamic range hits it more often.

3. **Bias projection overflow** — `bias_int = round(b * 2^-e_layer_min)`
   for very negative `e_layer_min` (int16 with large block exponent ranges)
   can exceed int64.

Both kernels (python ref + triton) share the same int64 arithmetic, so
the python ref reproduces the same misbehaviour and we can demonstrate
the bugs on CPU.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, "/Users/avrahamraviv/PycharmProjects")
sys.path.insert(0, "/home/avrahamra/PycharmProjects")

from mx_fixed_point_hw import _hw_fxp_conv2d_ref


def _make_synthetic(shape_qi=(1, 32, 8, 8), shape_qw=(8, 32, 3, 3),
                    bs=32, fmt_mbits=16,
                    Ea_range=(-2, 4), Ew_range=(-2, 4)):
    """Build int operands + per-block exponents directly, controlling the
    shift = Ea + Ew - 2*mant_bias - e_layer_min distribution.
    """
    B, C, H, W = shape_qi
    O, _, kH, kW = shape_qw
    nb = C // bs
    torch.manual_seed(0)

    max_val = (1 << (fmt_mbits - 1)) - 1
    store_dtype = torch.int8 if fmt_mbits <= 8 else torch.int16

    qi_i = torch.randint(-max_val, max_val + 1, shape_qi, dtype=store_dtype)
    qw_i = torch.randint(-max_val, max_val + 1, shape_qw, dtype=store_dtype)
    Ea = torch.randint(Ea_range[0], Ea_range[1] + 1, (B, nb, H, W), dtype=torch.int16)
    Ew = torch.randint(Ew_range[0], Ew_range[1] + 1, (O, nb, kH, kW), dtype=torch.int16)
    return qi_i, qw_i, Ea, Ew


def _exact_fp32_ref(qi_i, qw_i, Ea, Ew, mant_bias, stride=1, padding=1, dilation=1):
    """Compute the same dot product in FP32 by reconstructing operands to fp64.

    Each int element x with shared exponent E maps to value `x * 2^(E - mant_bias)`.
    """
    import torch.nn.functional as F

    B, C, H, W = qi_i.shape
    O, _, kH, kW = qw_i.shape
    nb = Ea.shape[1]
    bs = C // nb

    a_scale = torch.pow(2.0, Ea.to(torch.float64) - mant_bias)  # [B, nb, H, W]
    a_scale = a_scale.unsqueeze(2).expand(B, nb, bs, H, W).reshape(B, C, H, W)
    w_scale = torch.pow(2.0, Ew.to(torch.float64) - mant_bias)  # [O, nb, kH, kW]
    w_scale = w_scale.unsqueeze(2).expand(O, nb, bs, kH, kW).reshape(O, C, kH, kW)

    a_f = qi_i.to(torch.float64) * a_scale
    w_f = qw_i.to(torch.float64) * w_scale

    return F.conv2d(a_f.float(), w_f.float(),
                    stride=stride, padding=padding, dilation=dilation)


# ----------------------------------------------------------------------
# Bug A — bits=64 saturation misfires due to int64 wrap of `acc + s`.
# ----------------------------------------------------------------------

def test_bug_A_bits_64_saturation_misfires_when_acc_near_int64_limit():
    """At bits=64, acc has no headroom. `acc + s` can wrap int64 before
    saturation check runs. Demonstrate by accumulating values that
    individually fit but together cross 2^63."""
    qi_i, qw_i, Ea, Ew = _make_synthetic(
        shape_qi=(1, 32, 4, 4), shape_qw=(4, 32, 3, 3),
        bs=32, fmt_mbits=16, Ea_range=(8, 10), Ew_range=(8, 10),
    )
    # Set e_layer_min so all shifts are large positive (-28 = -2*14).
    # Combined with int16 products (~30 bits) and shifts ~12 → ~42-bit s.
    # Then nb*kK*bs accumulations push acc into int64 range.
    e_min = -2 * 14  # min(Ea)+min(Ew)-28 = 8+8-28 = -12. Use -28 to push shift even bigger.

    # Width-64: no headroom — wrong saturation.
    out64, sat64 = _hw_fxp_conv2d_ref(
        qi_i, qw_i, Ea, Ew, e_min,
        (1, 1), (1, 1), (1, 1), 32, 64, "per_product",
        mant_bias=14,
    )
    # Width-62: 2-bit headroom should make saturation work correctly for same shifts.
    out62, sat62 = _hw_fxp_conv2d_ref(
        qi_i, qw_i, Ea, Ew, e_min,
        (1, 1), (1, 1), (1, 1), 32, 62, "per_product",
        mant_bias=14,
    )

    fp = _exact_fp32_ref(qi_i, qw_i, Ea, Ew, mant_bias=14,
                         stride=1, padding=1, dilation=1)
    # At wider bits with proper saturation we'd expect mostly clamped output
    # (everything pinned to ±hi). At bits=64 with broken saturation, output
    # diverges in a way that does NOT match the FP32 truth.
    B, O = qi_i.shape[0], qw_i.shape[0]
    H_out, W_out = 4, 4
    out64_4d = out64.view(B, O, H_out, W_out)
    out62_4d = out62.view(B, O, H_out, W_out)

    # Capture diagnostic — print magnitudes to see the failure mode.
    print("\n[bits=64 vs bits=62] max|out64 - fp|", (out64_4d - fp).abs().max().item())
    print("[bits=64 vs bits=62] max|out62 - fp|", (out62_4d - fp).abs().max().item())
    print("[sat counts] bits=64:", int(sat64.sum().item()),
          " bits=62:", int(sat62.sum().item()))

    # The two windows should produce identical answers when neither saturates,
    # or both saturate to their respective rails. If bits=64 misfires, the
    # outputs diverge wildly compared to bits=62.
    assert sat62.any() or torch.allclose(out64_4d, out62_4d, atol=1.0)


# ----------------------------------------------------------------------
# Bug B — shift overflow when p << pos_shift wraps int64.
# ----------------------------------------------------------------------

def test_bug_B_shift_overflow_at_int16_with_wide_Ea_Ew_range():
    """For int16, p has up to ~30 bits; if shift > 33, p << shift overflows
    int64 silently. Demonstrate with a contrived Ea/Ew range."""
    qi_i, qw_i, Ea, Ew = _make_synthetic(
        shape_qi=(1, 32, 4, 4), shape_qw=(2, 32, 1, 1),
        bs=32, fmt_mbits=16,
        Ea_range=(20, 20), Ew_range=(20, 20),  # huge constant shift
    )
    # e_layer_min picks min(Ea)+min(Ew)-2*mant_bias = 40 - 28 = 12.
    # shift = Ea + Ew - 28 - e_min = 0 always. Now contrive to force a big shift:
    Ea[0, 0, 0, 0] = 50          # one lane has a huge exponent
    e_min = 20 + 20 - 28          # baseline shift 0; outlier shift = 30+0 = 30
    # Push the outlier higher to trigger overflow:
    Ea[0, 0, 0, 0] = 60           # now shift at that lane = 60 + 20 - 28 - 12 = 40

    out, sat = _hw_fxp_conv2d_ref(
        qi_i, qw_i, Ea, Ew, e_min,
        (1, 1), (0, 0), (1, 1), 32, 62, "per_product",
        mant_bias=14,
    )
    print("\n[bug-B] sat lanes:", int(sat.sum().item()), "/", sat.numel())
    print("[bug-B] max |out|:", out.abs().max().item())
    # Expect: saturated lanes >= 1 because shift=40 takes 30-bit p to 70 bits,
    # well past any reasonable accumulator. If the kernel is correct, sat is
    # set; if it's wrong (int64 wrap before clamp), out value at that lane
    # will be implausibly small or wrong sign.
    assert sat.any(), "Expected at least one saturated lane with shift=40"


# ----------------------------------------------------------------------
# Bug C — bias projection overflow at very negative e_layer_min.
# ----------------------------------------------------------------------

def test_bug_C_bias_int_overflows_for_very_negative_e_layer_min():
    """bias_int = round(bias * 2^-e_layer_min). For e_layer_min = -62 and
    bias ≈ 4, bias_int = 4 * 2^62 ≈ 1.8e19 > int64 max (9.2e18). Wraps."""
    bias = torch.tensor([4.0, -4.0])
    e_layer_min = -62
    scale = 2.0 ** (-e_layer_min)
    bias_int_safe = bias.to(torch.float64) * scale
    print("\n[bug-C] bias * 2^-e_min:", bias_int_safe.tolist())
    print("[bug-C] int64 max:", (1 << 63) - 1)
    overflows = bias_int_safe.abs() > ((1 << 63) - 1)
    assert overflows.any(), "Expected at least one bias_int to overflow int64"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
