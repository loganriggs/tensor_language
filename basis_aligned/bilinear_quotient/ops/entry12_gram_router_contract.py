"""Sign-gauge-invariant Gram features and centroids for entry12 source routing."""

from __future__ import annotations

import entry12_finite_router_contract as base


class GramRouterError(ValueError):
    pass


def gram_features(torch, off, expert_states, semantic_positions):
    if set(expert_states) != {"A1", "A2"} or off.ndim != 3:
        raise GramRouterError("Gram features require off and two expert state tensors")
    a = expert_states["A1"].float() - off.float()
    b = expert_states["A2"].float() - off.float()
    semantic_a = base.semantic_features(torch, a, semantic_positions)
    semantic_b = base.semantic_features(torch, b, semantic_positions)
    rows = []
    for row, stop in enumerate(semantic_positions):
        stop = int(stop)
        pa, pb = a[row, :stop + 1], b[row, :stop + 1]
        rows.append(torch.stack((
            torch.log1p(semantic_a[row].square().mean()),
            torch.log1p(semantic_b[row].square().mean()),
            (semantic_a[row] * semantic_b[row]).mean(),
            torch.log1p(pa.square().mean()),
            torch.log1p(pb.square().mean()),
            (pa * pb).mean(),
        )))
    return torch.stack(rows)


def fit(torch, features, labels):
    if features.ndim != 2 or features.shape[1] != 6:
        raise GramRouterError("Gram router requires six features")
    mean = features.float().mean(0)
    scale = features.float().std(0, unbiased=False).clamp_min(1e-8)
    standardized = (features.float() - mean) / scale
    centroids = base.fit_centroids(torch, standardized, labels)
    return {"mean": mean, "scale": scale, "centroids": centroids}


def predict(torch, features, fitted):
    standardized = (features.float() - fitted["mean"]) / fitted["scale"]
    return base.predict(torch, standardized, fitted["centroids"])
