"""Pure target-side context classes used by circuit consequence audits."""

from __future__ import annotations

import torch


CLASSES = ("induction", "repeat", "novel")


@torch.no_grad()
def target_token_classes(
    input_tokens: torch.Tensor, target_tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Partition next-token positions using information available at that position.

    At prediction position ``j``, ``input_tokens[:, :j + 1]`` is the available
    context and ``target_tokens[:, j]`` is the next token.  ``induction`` means
    there is a strictly earlier ``p < j`` with the same current token and with
    the current target as its successor.  ``repeat`` means the target occurs in
    the available context but the induction predicate is false.  ``novel`` means
    the target does not occur in the available context.
    """

    if input_tokens.ndim != 2 or target_tokens.shape != input_tokens.shape:
        raise ValueError("input_tokens and target_tokens must have the same [batch, length] shape")
    if input_tokens.device != target_tokens.device:
        raise ValueError("input_tokens and target_tokens must be on the same device")

    batch, length = input_tokens.shape
    positions = torch.arange(length, device=input_tokens.device)
    # Matrix axes are [current position j, candidate source position p].
    strictly_prior = positions.unsqueeze(0) < positions.unsqueeze(1)
    available_context = positions.unsqueeze(0) <= positions.unsqueeze(1)
    successor = torch.cat([
        input_tokens[:, 1:],
        torch.full(
            (batch, 1), -1, device=input_tokens.device, dtype=input_tokens.dtype,
        ),
    ], dim=1)

    same_current = input_tokens.unsqueeze(1) == input_tokens.unsqueeze(2)
    same_successor = successor.unsqueeze(1) == target_tokens.unsqueeze(2)
    induction = (
        same_current & same_successor & strictly_prior.unsqueeze(0)
    ).any(dim=2)
    target_seen = (
        (input_tokens.unsqueeze(1) == target_tokens.unsqueeze(2))
        & available_context.unsqueeze(0)
    ).any(dim=2)

    # A valid induction source p < j has successor p+1 <= j, so induction is a
    # subset of target_seen.  Keeping novel as ~target_seen makes that invariant
    # explicit rather than hiding a future-looking induction label in the novel cell.
    return {
        "induction": induction,
        "repeat": target_seen & ~induction,
        "novel": ~target_seen,
    }
