"""Pure tensor algebra for rung 536 weight-compiled product-space DAS.

This module contains no data loading, model loading, loss evaluation, or fitting. The
real-model wrapper must supply audited token/context parts and frozen donor pairs.
"""

from __future__ import annotations

import torch


def product_features(
    token: torch.Tensor,
    context: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    state = token + context
    return torch.nn.functional.linear(state, left) * torch.nn.functional.linear(state, right)


def product_branches(
    token: torch.Tensor,
    context: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> dict[str, torch.Tensor]:
    left_token = torch.nn.functional.linear(token, left)
    right_token = torch.nn.functional.linear(token, right)
    left_context = torch.nn.functional.linear(context, left)
    right_context = torch.nn.functional.linear(context, right)
    return {
        "T": left_token * right_token,
        "I": left_token * right_context + left_context * right_token,
        "C": left_context * right_context,
    }


def token_hybrid_pair(
    base_token: torch.Tensor,
    base_context: torch.Tensor,
    donor_token: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full hybrid delta and exact token-only target delta."""
    base = product_features(base_token, base_context, left, right)
    hybrid = product_features(donor_token, base_context, left, right)
    base_parts = product_branches(base_token, base_context, left, right)
    donor_parts = product_branches(donor_token, base_context, left, right)
    return hybrid - base, donor_parts["T"] - base_parts["T"]


def interaction_hybrid_pair(
    base_token: torch.Tensor,
    base_context: torch.Tensor,
    donor_context: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full hybrid delta and exact token-by-context target delta."""
    base = product_features(base_token, base_context, left, right)
    hybrid = product_features(base_token, donor_context, left, right)
    base_parts = product_branches(base_token, base_context, left, right)
    donor_parts = product_branches(base_token, donor_context, left, right)
    return hybrid - base, donor_parts["I"] - base_parts["I"]


def projected_interchange(
    base_product: torch.Tensor,
    full_hybrid_delta: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Apply an orthonormal-basis DAS projector without materializing P=UU^T."""
    return base_product + (full_hybrid_delta @ basis) @ basis.transpose(-1, -2)


def compile_basis(
    left: torch.Tensor,
    right: torch.Tensor,
    down: torch.Tensor,
    basis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile U into symmetric input quadratics Q and output directions d."""
    ordered = torch.einsum("hi,hk,hj->kij", left, basis, right)
    forms = 0.5 * (ordered + ordered.transpose(-1, -2))
    output_directions = down @ basis
    return forms, output_directions


def compiled_output(
    state: torch.Tensor,
    forms: torch.Tensor,
    output_directions: torch.Tensor,
) -> torch.Tensor:
    coordinates = torch.einsum("...i,kij,...j->...k", state, forms, state)
    return coordinates @ output_directions.transpose(-1, -2)
