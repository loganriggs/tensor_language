"""Activation-conditioned hidden-unit scores for an exact output subspace map."""
# BQGATE: LIBRARY
from __future__ import annotations


def contribution_energy(delta_hidden, hidden_factor):
    """Per-unit diagonal energy of delta_h_i * A_i in subspace coordinates.

    Cross-unit cancellation is intentionally not assigned to either unit; exact
    closure is measured after selected units are summed by the causal runner.
    """
    if delta_hidden.ndim != 2 or hidden_factor.ndim != 2:
        raise ValueError("expected [observations, hidden] and [hidden, rank]")
    if delta_hidden.shape[1] != hidden_factor.shape[0]:
        raise ValueError("hidden dimensions differ")
    return delta_hidden.square().sum(0) * hidden_factor.square().sum(1)


def top_fraction_mask(scores, fraction):
    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a nonempty vector")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    import math
    count = math.ceil(float(fraction) * scores.numel())
    indices = scores.argsort(descending=True)[:count]
    mask = scores.new_zeros(scores.shape, dtype=__import__('torch').bool)
    mask[indices] = True
    return mask


def compiled_delta(delta_hidden, hidden_factor, basis, mask=None):
    if mask is not None:
        delta_hidden, hidden_factor = delta_hidden[..., mask], hidden_factor[mask]
    return (delta_hidden @ hidden_factor) @ basis.T
