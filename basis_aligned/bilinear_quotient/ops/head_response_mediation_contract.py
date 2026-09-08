"""Pure construction and scoring contract for a downstream-head mediation cube.

The two parent executions differ only by an upstream intervention.  Crossing the
upstream background with the source of one downstream head's absolute response
gives four cells:

``00`` upstream off / head from off, ``01`` upstream off / head from on,
``10`` upstream on / head from off, and ``11`` upstream on / head from on.

This distinguishes rescue (01-00), reset loss (11-10), the direct/bypass arm
(10-00), and their interaction without treating a responsive head as a mediator.
"""

from __future__ import annotations

import math


CELLS = ("00", "01", "10", "11")


class MediationContractError(ValueError):
    pass


def absolute_head_hybrid(torch, background, head_source, *, head, semantic_positions):
    """Return ``background`` with one head copied from ``head_source`` on-prefix."""
    if background.shape != head_source.shape or background.ndim != 4:
        raise MediationContractError("head captures must be equal [batch, token, head, width] tensors")
    batch, tokens, heads, _width = background.shape
    if not 0 <= int(head) < heads:
        raise MediationContractError("selected head is out of range")
    if len(semantic_positions) != batch:
        raise MediationContractError("semantic-position count must equal batch size")
    result = background.clone()
    for row, stop in enumerate(semantic_positions):
        stop = int(stop)
        if not 0 <= stop < tokens:
            raise MediationContractError("semantic position is out of range")
        result[row, : stop + 1, int(head)] = head_source[row, : stop + 1, int(head)].to(result)
    return result


def build_absolute_cells(torch, upstream_off, upstream_on, *, head, semantic_positions):
    """Build the four absolute response tensors for the mediation intervention."""
    return {
        "00": absolute_head_hybrid(
            torch, upstream_off, upstream_off, head=head, semantic_positions=semantic_positions),
        "01": absolute_head_hybrid(
            torch, upstream_off, upstream_on, head=head, semantic_positions=semantic_positions),
        "10": absolute_head_hybrid(
            torch, upstream_on, upstream_off, head=head, semantic_positions=semantic_positions),
        "11": absolute_head_hybrid(
            torch, upstream_on, upstream_on, head=head, semantic_positions=semantic_positions),
    }

def _vector(values, name):
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise MediationContractError(f"{name} must be a nonempty finite vector")
    return result


def decompose_margin_cells(cells):
    """Return exact rowwise rescue, reset-loss, bypass, and interaction effects."""
    if set(cells) != set(CELLS):
        raise MediationContractError(f"cells must be exactly {CELLS}")
    vectors = {cell: _vector(cells[cell], cell) for cell in CELLS}
    lengths = {len(values) for values in vectors.values()}
    if len(lengths) != 1:
        raise MediationContractError("cell vectors must have equal length")
    subtract = lambda left, right: [a - b for a, b in zip(vectors[left], vectors[right])]
    rescue = subtract("01", "00")
    reset_loss = subtract("11", "10")
    bypass = subtract("10", "00")
    full = subtract("11", "00")
    interaction = [reset - rescued for reset, rescued in zip(reset_loss, rescue)]
    closure = [b + r + i - f for b, r, i, f in zip(bypass, rescue, interaction, full)]
    return {
        "full_upstream_effect": full,
        "head_rescue": rescue,
        "head_reset_loss": reset_loss,
        "upstream_bypass_with_head_reset": bypass,
        "mobius_interaction": interaction,
        "closure_max_abs_error": max(abs(value) for value in closure),
    }
