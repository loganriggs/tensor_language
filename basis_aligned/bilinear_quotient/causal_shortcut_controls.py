"""Linear-algebra helpers for detecting output-steering shortcut subspaces.

These functions do not claim that a raw unembedding span is the complete local
readout at an early residual site.  It is a cheap, globally defined diagnostic.
For a causal gate at an earlier site, callers should additionally construct a
site-local tangent span from gradients of the registered endpoints.
"""

from __future__ import annotations

import torch


def orthonormal_column_span(matrix: torch.Tensor, *, rtol: float = 1e-6) -> torch.Tensor:
    """Return a deterministic orthonormal basis for the numerical column span."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    if singular.numel() == 0:
        return matrix[:, :0]
    keep = singular > (rtol * singular.max())
    return u[:, keep]


def centered_token_readout_span(readout_weight: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    """Span of pairwise output-weight differences for a registered token set.

    ``readout_weight`` has shape ``[vocabulary, hidden]``.  Centering the selected
    rows removes their common mode, so four tokens produce rank at most three.
    """
    if readout_weight.ndim != 2:
        raise ValueError("readout_weight must have shape [vocabulary, hidden]")
    if len(set(token_ids)) != len(token_ids) or len(token_ids) < 2:
        raise ValueError("token_ids must contain at least two distinct tokens")
    rows = readout_weight[token_ids]
    centered = rows - rows.mean(dim=0, keepdim=True)
    return orthonormal_column_span(centered.T)


def subspace_overlap_fraction(candidate: torch.Tensor, shortcut: torch.Tensor) -> float:
    """Fraction of an orthonormal candidate subspace lying in shortcut span."""
    if candidate.ndim != 2 or shortcut.ndim != 2 or candidate.shape[0] != shortcut.shape[0]:
        raise ValueError("candidate and shortcut must share an ambient dimension")
    if candidate.shape[1] == 0:
        return 0.0
    value = (shortcut.T @ candidate).square().sum() / candidate.shape[1]
    return float(value.clamp(0, 1))


def deflate_and_orthonormalize(candidate: torch.Tensor, shortcut: torch.Tensor,
                               *, rtol: float = 1e-6) -> torch.Tensor:
    """Project candidate columns off the shortcut span and remove lost columns."""
    residual = candidate - shortcut @ (shortcut.T @ candidate)
    return orthonormal_column_span(residual, rtol=rtol)


def pooled_endpoint_tangent_span(endpoint_gradients: torch.Tensor, *, rtol: float = 1e-6) -> torch.Tensor:
    """Span of site-local endpoint gradients pooled across rows and endpoints.

    The input is ``[..., hidden]``.  Every leading-index gradient becomes one
    candidate readout direction.  This operational span accounts for all
    downstream layers at the chosen site, unlike raw unembedding weights.
    """
    if endpoint_gradients.ndim < 2:
        raise ValueError("endpoint_gradients must have at least two dimensions")
    hidden = endpoint_gradients.shape[-1]
    return orthonormal_column_span(endpoint_gradients.reshape(-1, hidden).T, rtol=rtol)
