"""Pure accounting for component-response directional derivatives."""

# BQGATE: LIBRARY
from __future__ import annotations

from typing import Sequence

import torch


class PathConditionedScoreError(ValueError):
    pass


def directional_effects(gradient, base, donor, position_banks: Sequence[Sequence[int]]):
    """Return per-row ``sum_position <gradient, donor - base>`` effects."""
    if (gradient.ndim != 3 or base.shape != gradient.shape or donor.shape != gradient.shape
            or gradient.shape[0] != len(position_banks)):
        raise PathConditionedScoreError("gradient, component tensors, or row banks disagree")
    effects = []
    for row, bank in enumerate(position_banks):
        positions = tuple(bank)
        if (not positions or len(positions) != len(set(positions))
                or any(type(position) is not int or not 0 <= position < gradient.shape[1]
                       for position in positions)):
            raise PathConditionedScoreError("position bank is empty, repeated, or out of range")
        value = gradient.new_zeros(())
        for position in positions:
            value = value + gradient[row, position].float() @ (
                donor[row, position].float() - base[row, position].float())
        effects.append(value)
    return torch.stack(effects)
