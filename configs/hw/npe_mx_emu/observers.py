"""Observers: collect block-amax statistics over calibration samples and derive
MX calibration parameters.

Restriction #1 of the NPE MX flow: the unsigned 4-bit exponent register can
only hold values in [0, 15], so calibrate_offset must scale the data such that
NO block exponent goes negative.  The observer enforces this by construction:

  mx mode    : anchor the smallest non-zero block amax so its exponent lands at 0
               (block exponent extraction is floor(log2) -- HW-confirmed Q1 --
               so the anchor condition is a*2^cal >= 2^mantissa_offset)
               calibrate_offset = mantissa_offset - floor(log2(a_min))
  input mode : integer sensor input uses a zero shared exponent and raw ints --
               the scale must make the data fit the signed int8 mantissa range
               calibrate_offset = mantissa_offset - ceil(log2(a_max))

and raises loudly when the residual dynamic range overflows the 4-bit exponent
window (meaning per-channel calibration is required).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from npe_mx_emu.mx_format import SIGN_BIT, GREATER_THAN_ONE_BIT, mx_config_factory


def _block_amax_reduction(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Reshape the last dim into blocks (padded with zeros) and return per-block amax."""
    width = x.shape[-1]
    num_blocks = math.ceil(width / block_size)
    pad_amount = num_blocks * block_size - width
    if pad_amount > 0:
        x = F.pad(x, (0, pad_amount), mode="constant", value=0)
    blocks = x.view(*x.shape[:-1], num_blocks, block_size)
    return blocks.abs().amax(dim=-1)


class MXObserver:
    """Running block-amax statistics over calibration batches.

    mode:
        "mx"    - intermediate activations / weights (exponent-anchored rule)
        "input" - first-layer integer sensor input (int8 range-fit rule)

    Blocking follows the HW data-consumption order (HW restriction): activations
    are blocked along W per channel row; weights are blocked per flattened
    kernel.  Pass ``per_row_flatten=True`` to update() when observing weights,
    otherwise the observer's anchor does not match the quantizer's blocks
    (see algo table item Q9).
    reduction:
        "minmax" - hard global min/max over all update() calls.  Required for
                   the no-negative-exponent guarantee to hold on unseen data.
        "ema"    - exponential moving average of the per-batch stats.  Smoother
                   against outliers, but can break the guarantee on unseen data
                   (a batch with a smaller block amax than the EMA will wrap).
    """

    def __init__(
        self,
        block_size: int = 32,
        mx_type: str = "mx-e4bm8",
        mode: str = "mx",
        reduction: str = "minmax",
        ema_momentum: float = 0.99,
    ):
        if mode not in ("mx", "input"):
            raise ValueError(f"unknown mode {mode!r} (expected 'mx' or 'input')")
        if reduction not in ("minmax", "ema"):
            raise ValueError(f"unknown reduction {reduction!r} (expected 'minmax' or 'ema')")

        self.block_size = block_size
        self.mx_type = mx_type
        self.mode = mode
        self.reduction = reduction
        self.ema_momentum = ema_momentum

        self.a_max: Optional[float] = None  # largest block amax seen
        self.a_min: Optional[float] = None  # smallest non-zero block amax seen
        self.n_samples = 0

    def update(self, x: torch.Tensor, per_row_flatten: bool = False) -> "MXObserver":
        """Collect block-amax statistics from one calibration sample.

        Blocking must match the HW data-consumption order of the tensor kind:
          * activations (default): blocks of ``block_size`` along W, per channel
            row -- a block never crosses a line.
          * weights (``per_row_flatten=True``): each output-channel kernel is
            flattened to ``C*Kh*Kw`` elements and THEN blocked, mirroring
            ``mx_weight_quantizer`` (HW receives weights as flat per-kernel
            streams).  For dim <= 2 the tensor is observed as-is, same as the
            quantizer's shallow branch.
        """
        x = x.detach().float()
        if per_row_flatten and x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        batch_amax = _block_amax_reduction(x, self.block_size)
        b_max = batch_amax.max().item()
        nz = batch_amax[batch_amax > 0]
        b_min = nz.min().item() if nz.numel() > 0 else None

        if self.reduction == "minmax":
            self.a_max = b_max if self.a_max is None else max(self.a_max, b_max)
            if b_min is not None:
                self.a_min = b_min if self.a_min is None else min(self.a_min, b_min)
        else:  # ema
            m = self.ema_momentum
            self.a_max = b_max if self.a_max is None else m * self.a_max + (1 - m) * b_max
            if b_min is not None:
                self.a_min = b_min if self.a_min is None else m * self.a_min + (1 - m) * b_min

        self.n_samples += 1
        return self

    __call__ = update

    def _stats(self):
        if self.a_max is None or self.a_min is None:
            raise RuntimeError(
                "MXObserver has no statistics: call update() with at least one "
                "batch containing non-zero values"
            )
        return self.a_max, self.a_min

    def mantissa_offset(self) -> int:
        return int(mx_config_factory[self.mx_type].mantissa_bits - SIGN_BIT - GREATER_THAN_ONE_BIT)

    def max_block_exponent(self, calibrate_offset: int) -> int:
        """Exponent of the largest block after scaling (the exponent-register endpoint).

        Matches the quantizer's floor(log2) extraction (Q1)."""
        a_max, _ = self._stats()
        return int(math.floor(math.log2(a_max * 2.0 ** calibrate_offset)) - self.mantissa_offset())

    def calibrate_offset(self) -> int:
        a_max, a_min = self._stats()
        m_off = self.mantissa_offset()
        sat_val = 2 ** int(mx_config_factory[self.mx_type].exponent_bits) - 1

        if self.mode == "mx":
            # anchor the smallest non-zero block exponent at 0 -> no negative exponents
            # (floor extraction: floor(log2(a*2^cal)) >= m_off <=> a*2^cal >= 2^m_off)
            cal = m_off - math.floor(math.log2(a_min))
        else:  # "input": raw int storage with zero shared exponent, fit int8 range
            cal = m_off - math.ceil(math.log2(a_max))

        if self.mode == "mx":
            max_exp = self.max_block_exponent(cal)
            if max_exp > sat_val:
                raise ValueError(
                    f"observer stats (a_min={a_min:.6g}, a_max={a_max:.6g}) exceed the "
                    f"{int(mx_config_factory[self.mx_type].exponent_bits)}-bit exponent window: "
                    f"smallest block would sit at exponent 0 but the largest block needs "
                    f"exponent {max_exp} > {sat_val}. "
                    f"Per-channel calibration is required for this tensor."
                )
        return cal

    def e_min_layer(self) -> int:
        return self.mantissa_offset() - self.calibrate_offset()
