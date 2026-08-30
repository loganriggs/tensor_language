"""Pure fixed-mask tensor contractions for a previous-token circuit.

The functions in this module consume an attention head's *already-computed* score
tensor and value tensor.  They do not compute Q/K/V projections, apply a causal
mask, load a model, or choose support from the data.  Consequently every arm is a
fixed tensor network once the sequence lengths and offset are fixed.

Offset convention
-----------------
``offset`` means ``key_position = query_position + offset``.  In particular,
``offset=-1`` is the previous-token diagonal.  An offset outside the rectangular
query/key array simply has empty support and therefore contracts to zero.
"""

from __future__ import annotations

from enum import Enum

import torch


class PreviousTokenArm(str, Enum):
    """Registered fixed-support arms for the previous-token component."""

    NATIVE = "native_full_head"
    REMOVE_PREVIOUS = "remove_shift_minus_1"
    EXTRACT_PREVIOUS = "extract_shift_minus_1_only"
    DERANGED_MINUS_2 = "deranged_shift_minus_2_only"
    DERANGED_PLUS_2 = "deranged_shift_plus_2_only"


def fixed_shift_mask(
    query_length: int,
    key_length: int,
    offset: int,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return the boolean mask ``key = query + offset`` with shape ``[Q, K]``."""

    if type(query_length) is not int or query_length <= 0:
        raise ValueError("query_length must be a positive Python integer")
    if type(key_length) is not int or key_length <= 0:
        raise ValueError("key_length must be a positive Python integer")
    if type(offset) is not int:
        raise ValueError("offset must be a Python integer")
    queries = torch.arange(query_length, device=device).unsqueeze(1)
    keys = torch.arange(key_length, device=device).unsqueeze(0)
    return (keys == queries + offset).contiguous()


def _validate_scores_and_values(
    scores: torch.Tensor,
    values: torch.Tensor,
) -> tuple[int, int]:
    if not torch.is_tensor(scores) or scores.ndim < 2:
        raise ValueError("scores must be a tensor with shape [..., query, key]")
    if not torch.is_tensor(values) or values.ndim != scores.ndim:
        raise ValueError("values must have shape [..., key, value]")
    if scores.device != values.device:
        raise ValueError("scores and values must be on the same device")
    if scores.dtype != values.dtype:
        raise ValueError("scores and values must have the same dtype")
    if scores.shape[:-2] != values.shape[:-2]:
        raise ValueError("scores and values must have identical leading axes")
    if scores.shape[-1] != values.shape[-2]:
        raise ValueError("the score key axis must equal the value key axis")
    if scores.shape[-2] <= 0 or scores.shape[-1] <= 0 or values.shape[-1] <= 0:
        raise ValueError("query, key, and value dimensions must be nonempty")
    if not (scores.dtype.is_floating_point or scores.dtype.is_complex):
        raise ValueError("scores and values must use a floating or complex dtype")
    return int(scores.shape[-2]), int(scores.shape[-1])


def contract_full_head(scores: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Contract every supplied score with its key's value: ``scores @ values``."""

    _validate_scores_and_values(scores, values)
    return torch.matmul(scores, values)


def contract_fixed_shift(
    scores: torch.Tensor,
    values: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    """Contract only score/value edges on one fixed relative-position diagonal."""

    query_length, key_length = _validate_scores_and_values(scores, values)
    mask = fixed_shift_mask(
        query_length, key_length, offset, device=scores.device,
    ).to(dtype=scores.dtype)
    return torch.matmul(scores * mask, values)


def contract_without_fixed_shift(
    scores: torch.Tensor,
    values: torch.Tensor,
    offset: int,
) -> torch.Tensor:
    """Contract the supplied full head after deleting one fixed diagonal."""

    query_length, key_length = _validate_scores_and_values(scores, values)
    mask = fixed_shift_mask(
        query_length, key_length, offset, device=scores.device,
    )
    return torch.matmul(scores.masked_fill(mask, 0), values)


def run_previous_token_arm(
    scores: torch.Tensor,
    values: torch.Tensor,
    arm: PreviousTokenArm,
) -> torch.Tensor:
    """Evaluate one fixed arm without data-dependent routing or model calls."""

    if type(arm) is not PreviousTokenArm:
        raise ValueError("arm must be a PreviousTokenArm")
    if arm is PreviousTokenArm.NATIVE:
        return contract_full_head(scores, values)
    if arm is PreviousTokenArm.REMOVE_PREVIOUS:
        return contract_without_fixed_shift(scores, values, -1)
    if arm is PreviousTokenArm.EXTRACT_PREVIOUS:
        return contract_fixed_shift(scores, values, -1)
    if arm is PreviousTokenArm.DERANGED_MINUS_2:
        return contract_fixed_shift(scores, values, -2)
    if arm is PreviousTokenArm.DERANGED_PLUS_2:
        return contract_fixed_shift(scores, values, 2)
    raise AssertionError("unreachable PreviousTokenArm")


__all__ = [
    "PreviousTokenArm",
    "contract_fixed_shift",
    "contract_full_head",
    "contract_without_fixed_shift",
    "fixed_shift_mask",
    "run_previous_token_arm",
]
