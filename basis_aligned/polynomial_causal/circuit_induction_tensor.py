"""Pure equality-and-shift tensor contractions for an induction fetch path.

For token one-hots ``e_t``, the support entry

    M[q,k] = <e[token[q]], e[token[k-1]]> 1[1 <= k <= q]

is a contraction with the vocabulary equality tensor and fixed position masks.  The
integer-token implementation below is its compiled sparse evaluation; it does not
choose a source with argmax, nearest-match, or TopK.
"""

from __future__ import annotations

import torch


def induction_fetch_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Return ``[B,Q,K]`` support for keys immediately following equal query tokens."""

    if not torch.is_tensor(tokens) or tokens.ndim != 2 or tokens.dtype != torch.long:
        raise ValueError("tokens must be a rank-2 int64 tensor")
    if tokens.shape[1] < 2 or bool((tokens < 0).any()):
        raise ValueError("tokens must have nonnegative IDs and length at least two")
    batch, length = tokens.shape
    mask = torch.zeros(batch, length, length, dtype=torch.bool, device=tokens.device)
    query_equals_predecessor = tokens[:, :, None] == tokens[:, None, :-1]
    mask[:, :, 1:] = query_equals_predecessor
    causal = torch.tril(torch.ones(
        length, length, dtype=torch.bool, device=tokens.device,
    ))
    return mask & causal


def contract_induction_fetch(
    scores: torch.Tensor,
    values: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Contract only equality-matched successor edges of one attention head."""

    if not torch.is_tensor(scores) or scores.ndim != 3:
        raise ValueError("scores must have shape [batch, query, key]")
    if not torch.is_tensor(values) or values.ndim != 3:
        raise ValueError("values must have shape [batch, key, value]")
    if scores.shape[:1] != tokens.shape[:1] or scores.shape[1:] != (
        tokens.shape[1], tokens.shape[1]
    ) or values.shape[:2] != tokens.shape:
        raise ValueError("scores, values, and tokens have incompatible shapes")
    if scores.device != values.device or scores.device != tokens.device:
        raise ValueError("scores, values, and tokens must share a device")
    if scores.dtype != values.dtype or not scores.is_floating_point():
        raise ValueError("scores and values must share a floating dtype")
    support = induction_fetch_mask(tokens).to(scores.dtype)
    return torch.bmm(scores * support, values)


def contract_without_induction_fetch(
    scores: torch.Tensor,
    values: torch.Tensor,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Contract the full head after removing all equality-matched successor edges."""

    if not torch.is_tensor(scores) or scores.ndim != 3:
        raise ValueError("scores must have shape [batch, query, key]")
    support = induction_fetch_mask(tokens)
    if values.ndim != 3 or scores.shape[:2] != support.shape[:2] or (
        scores.shape[2] != support.shape[2] or values.shape[:2] != tokens.shape
    ):
        raise ValueError("scores, values, and tokens have incompatible shapes")
    if scores.device != values.device or scores.device != tokens.device:
        raise ValueError("scores, values, and tokens must share a device")
    if scores.dtype != values.dtype or not scores.is_floating_point():
        raise ValueError("scores and values must share a floating dtype")
    return torch.bmm(scores.masked_fill(support, 0), values)


__all__ = [
    "contract_induction_fetch",
    "contract_without_induction_fetch",
    "induction_fetch_mask",
]
