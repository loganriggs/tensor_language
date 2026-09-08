"""Direction-invariant unordered token-pair features from frozen tied embeddings."""

from __future__ import annotations


class EmbeddingRouterError(ValueError):
    pass


def feature(torch, embedding_weight, signature):
    if embedding_weight.ndim != 2:
        raise EmbeddingRouterError("embedding weight must be [vocab,width]")
    width = embedding_weight.shape[1]
    if not signature:
        return torch.zeros(2 * width, device=embedding_weight.device, dtype=torch.float32)
    pairs = torch.tensor(signature, device=embedding_weight.device, dtype=torch.long)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or bool((pairs < 0).any()) or bool((pairs >= embedding_weight.shape[0]).any()):
        raise EmbeddingRouterError("signature token ids are invalid")
    values = embedding_weight.float()[pairs]
    values = torch.nn.functional.rms_norm(values, (width,))
    midpoint = values.sum(1).mean(0)
    squared_difference = values.diff(dim=1).squeeze(1).square().mean(0)
    normalize = lambda value: value / value.norm().clamp_min(1e-30)
    return torch.cat((normalize(midpoint), normalize(squared_difference)))


def features(torch, embedding_weight, signatures):
    if not signatures:
        raise EmbeddingRouterError("at least one signature is required")
    return torch.stack([feature(torch, embedding_weight, item) for item in signatures])


def fit(torch, values, labels, n_classes=3):
    if values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels):
        raise EmbeddingRouterError("feature/label shape mismatch")
    normalized = values / values.norm(dim=1, keepdim=True).clamp_min(1e-30)
    centroids = []
    for label in range(n_classes):
        selected = normalized[labels == label]
        if not len(selected):
            raise EmbeddingRouterError("every class must be present")
        centroid = selected.mean(0)
        centroids.append(centroid / centroid.norm().clamp_min(1e-30))
    return torch.stack(centroids)


def predict(torch, values, centroids):
    normalized = values / values.norm(dim=1, keepdim=True).clamp_min(1e-30)
    scores = normalized @ centroids.T
    return scores.argmax(-1), scores
