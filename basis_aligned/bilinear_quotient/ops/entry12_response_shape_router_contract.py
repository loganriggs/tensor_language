"""Low-dimensional projective shape invariants for two entry-12 responses."""

from __future__ import annotations

import entry12_finite_router_contract as base


class ResponseShapeError(ValueError):
    pass


def _shape(torch, matrix):
    matrix = matrix.float()
    energy = matrix.square().sum().clamp_min(1e-12)
    singular2 = torch.linalg.svdvals(matrix).square()
    top_fraction = singular2.max() / energy
    participation = energy.square() / singular2.square().sum().clamp_min(1e-12)
    participation = participation / max(1, min(matrix.shape))
    semantic_fraction = matrix[-1].square().sum() / energy
    if len(matrix) > 1:
        roughness = (matrix[1:] - matrix[:-1]).square().sum() / energy
    else:
        roughness = torch.zeros((), device=matrix.device)
    return torch.stack((torch.log1p(energy / matrix.numel()), top_fraction,
                        participation, semantic_fraction, roughness))


def features(torch, off, expert_states, semantic_positions):
    """Return 14 fixed scalars invariant to simultaneous sign and width rotation."""
    if off.ndim != 3 or set(expert_states) != {"A1", "A2"}:
        raise ResponseShapeError("off and exactly two expert state tensors are required")
    if any(value.shape != off.shape for value in expert_states.values()):
        raise ResponseShapeError("expert state shapes must match off")
    if len(semantic_positions) != len(off):
        raise ResponseShapeError("one semantic position is required per row")
    deltas = {name: state.float() - off.float() for name, state in expert_states.items()}
    rows = []
    for row, stop_value in enumerate(semantic_positions):
        stop = int(stop_value)
        if stop < 0 or stop >= off.shape[1]:
            raise ResponseShapeError("semantic position out of range")
        a, b = deltas["A1"][row, :stop + 1], deltas["A2"][row, :stop + 1]
        ea, eb = a.square().sum().clamp_min(1e-12), b.square().sum().clamp_min(1e-12)
        prefix_cosine = (a * b).sum() / (ea.sqrt() * eb.sqrt())
        sa, sb = a[-1], b[-1]
        semantic_cosine = (sa @ sb) / (sa.norm() * sb.norm()).clamp_min(1e-12)
        token_gram_cosine = ((a @ a.T) * (b @ b.T)).sum() / (
            (a @ a.T).norm() * (b @ b.T).norm()).clamp_min(1e-12)
        log_energy_ratio = torch.log(ea / eb)
        rows.append(torch.cat((_shape(torch, a), _shape(torch, b), torch.stack((
            prefix_cosine, semantic_cosine, token_gram_cosine, log_energy_ratio)))))
    return torch.stack(rows)


def fit(torch, values, labels):
    if values.ndim != 2 or values.shape[1] != 14:
        raise ResponseShapeError("response-shape router requires fourteen features")
    mean = values.float().mean(0)
    scale = values.float().std(0, unbiased=False).clamp_min(1e-8)
    standardized = (values.float() - mean) / scale
    return {"mean": mean, "scale": scale,
            "centroids": base.fit_centroids(torch, standardized, labels)}


def predict(torch, values, fitted):
    standardized = (values.float() - fitted["mean"]) / fitted["scale"]
    return base.predict(torch, standardized, fitted["centroids"])
