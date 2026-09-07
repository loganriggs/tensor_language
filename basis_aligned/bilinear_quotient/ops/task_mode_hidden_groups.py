"""Gauge-safe task-pair directions and hidden-writer participation scores."""
# BQGATE: LIBRARY
from __future__ import annotations


def canonical_pair_directions(torch, temporal, iswas, *, eps=1e-7):
    """Return orthonormal mean/contrast axes for two equal-rank subspaces.

    The axes are the canonical-angle sum and difference vectors.  They are
    physical directions in the parent coordinate space: ``mean`` is the
    shared direction of each canonical pair and ``contrast`` is the direction
    that distinguishes the two tasks.  Degenerate zero-norm axes are omitted.
    """
    if temporal.ndim != 2 or iswas.ndim != 2 or temporal.shape != iswas.shape:
        raise ValueError("task bases must be equal-shape matrices")
    qt = torch.linalg.qr(temporal, mode="reduced").Q
    qi = torch.linalg.qr(iswas, mode="reduced").Q
    u, cosine, vh = torch.linalg.svd(qt.T @ qi, full_matrices=False)
    left, right = qt @ u, qi @ vh.T
    cosine = cosine.clamp(0.0, 1.0)
    mean_raw, contrast_raw = left + right, left - right
    mean_norm, contrast_norm = mean_raw.norm(dim=0), contrast_raw.norm(dim=0)
    # Decide degeneracy from the canonical cosine, rather than a noisy norm of
    # two nearly identical float vectors.
    keep_contrast = (1.0 - cosine) > eps
    mean = mean_raw / mean_norm.clamp_min(eps)
    contrast = contrast_raw[:, keep_contrast] / contrast_norm[keep_contrast]
    return {"mean": mean, "contrast": contrast, "principal_cosines": cosine}


def participation_energy(torch, delta_hidden, weight_map, directions):
    """Per-hidden-unit isolated energy written into named physical directions."""
    if delta_hidden.ndim != 2 or weight_map.ndim != 2:
        raise ValueError("delta_hidden and weight_map must be matrices")
    if delta_hidden.shape[1] != weight_map.shape[0]:
        raise ValueError("hidden width mismatch")
    output = {}
    activation_energy = delta_hidden.square().sum(dim=0)
    for name, basis in directions.items():
        if name == "principal_cosines":
            continue
        if basis.ndim != 2 or basis.shape[0] != weight_map.shape[1]:
            raise ValueError(f"bad direction basis for {name}")
        output[name] = activation_energy * (weight_map @ basis).square().sum(dim=1)
    return output
