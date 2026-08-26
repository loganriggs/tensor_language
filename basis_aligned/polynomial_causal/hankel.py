"""Small empirical Hankel-matrix diagnostics."""

import torch


def row_column_baseline(matrix, observed, steps=50):
    """Fit mean + row + column effects on observed entries."""
    matrix = matrix.double()
    observed = observed.bool()
    mean = matrix[observed].mean()
    row = torch.zeros(matrix.shape[0], dtype=torch.float64)
    column = torch.zeros(matrix.shape[1], dtype=torch.float64)
    for _ in range(steps):
        residual = matrix - mean - column[None, :]
        for i in range(matrix.shape[0]):
            mask = observed[i]
            if mask.any():
                row[i] = residual[i, mask].mean()
        residual = matrix - mean - row[:, None]
        for j in range(matrix.shape[1]):
            mask = observed[:, j]
            if mask.any():
                column[j] = residual[mask, j].mean()
    return mean + row[:, None] + column[None, :]


def complete_low_rank(matrix, observed, rank, steps=100, ridge=1e-6):
    """Alternating least-squares completion after row/column main effects."""
    matrix = matrix.double()
    observed = observed.bool()
    baseline = row_column_baseline(matrix, observed)
    residual = matrix - baseline
    filled = torch.where(observed, residual, torch.zeros_like(residual))
    u0, singular, vh0 = torch.linalg.svd(filled, full_matrices=False)
    scale = singular[:rank].sqrt()
    u = u0[:, :rank] * scale
    v = vh0[:rank].T * scale
    eye = ridge * torch.eye(rank, dtype=torch.float64)
    for _ in range(steps):
        for i in range(matrix.shape[0]):
            mask = observed[i]
            design = v[mask]
            if design.shape[0] >= rank:
                u[i] = torch.linalg.solve(design.T @ design + eye,
                                          design.T @ residual[i, mask])
        for j in range(matrix.shape[1]):
            mask = observed[:, j]
            design = u[mask]
            if design.shape[0] >= rank:
                v[j] = torch.linalg.solve(design.T @ design + eye,
                                          design.T @ residual[mask, j])
    return baseline + u @ v.T


def heldout_rmse(prediction, matrix, observed):
    heldout = ~observed.bool()
    return float((prediction[heldout] - matrix.double()[heldout]).square().mean().sqrt())


def spectrum(matrix):
    centered = matrix.double()
    centered = centered - centered.mean(0, keepdim=True)
    centered = centered - centered.mean(1, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    cumulative = energy.cumsum(0) / energy.sum().clamp_min(1e-30)
    rank90 = int(torch.searchsorted(cumulative, 0.90).item()) + 1
    rank95 = int(torch.searchsorted(cumulative, 0.95).item()) + 1
    stable_rank = float(energy.sum() / energy.max().clamp_min(1e-30))
    return {"singular_values": singular, "rank90": rank90, "rank95": rank95,
            "stable_rank": stable_rank}
