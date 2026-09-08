"""Exact, tangent, and raw weight responses to residual-state interventions."""

from __future__ import annotations


class NormalizedReaderError(ValueError):
    pass


def rms_jvp(torch, state, delta, *, eps=None):
    """Apply the exact Jacobian of RMSNorm at ``state`` to ``delta`` rowwise."""
    if state.shape != delta.shape or state.ndim != 2:
        raise NormalizedReaderError("state and delta must have equal [sample,width] shape")
    if eps is None:
        eps = torch.finfo(state.dtype).eps
    scale = (state.square().mean(-1, keepdim=True) + float(eps)).sqrt()
    radial = (state * delta).mean(-1, keepdim=True)
    return delta / scale - state * radial / scale.pow(3)


def reader_response(torch, matrix, state, delta, *, eps=None):
    """Return W times raw, RMS-tangent, and exact finite normalized changes."""
    if matrix.ndim != 2 or state.ndim != 2 or matrix.shape[1] != state.shape[1]:
        raise NormalizedReaderError("matrix must be [output,width] and states [sample,width]")
    if state.shape != delta.shape:
        raise NormalizedReaderError("state/delta shape mismatch")
    if eps is None:
        eps = torch.finfo(state.dtype).eps
    normalized = lambda value: torch.nn.functional.rms_norm(value, (value.shape[-1],), eps=float(eps))
    raw_input = delta
    tangent_input = rms_jvp(torch, state, delta, eps=eps)
    exact_input = normalized(state + delta) - normalized(state)
    apply = lambda value: value @ matrix.T
    raw, tangent, exact = map(apply, (raw_input, tangent_input, exact_input))
    matrix_norm = matrix.norm().clamp_min(1e-30)
    def strength(output, input_delta):
        return float(output.norm() / (matrix_norm * input_delta.norm().clamp_min(1e-30)))
    def cosine(left, right):
        return float((left.reshape(-1) @ right.reshape(-1)) /
                     (left.norm() * right.norm()).clamp_min(1e-30))
    return {
        "raw_strength": strength(raw, raw_input),
        "tangent_strength": strength(tangent, tangent_input),
        "exact_strength": strength(exact, exact_input),
        "tangent_exact_cosine": cosine(tangent, exact),
        "tangent_exact_relative_l2": float((tangent - exact).norm() / exact.norm().clamp_min(1e-30)),
        "raw_input_norm": float(raw_input.norm()),
        "tangent_input_norm": float(tangent_input.norm()),
        "exact_input_norm": float(exact_input.norm()),
    }


def ranked(records, key):
    rows = sorted(records, key=lambda row: (-row[key], row["label"]))
    denominator = max(1, len(rows) - 1)
    for index, row in enumerate(rows):
        row[f"{key}_percentile"] = (len(rows) - 1 - index) / denominator
    return rows


def spearman(left, right, key):
    labels = sorted({row["label"] for row in left} & {row["label"] for row in right})
    if len(labels) < 2:
        return float("nan")
    order_left = {row["label"]: rank for rank, row in enumerate(sorted(left, key=lambda row: (-row[key], row["label"])))}
    order_right = {row["label"]: rank for rank, row in enumerate(sorted(right, key=lambda row: (-row[key], row["label"])))}
    a = [order_left[label] for label in labels]
    b = [order_right[label] for label in labels]
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denominator = (sum((x - mean_a) ** 2 for x in a) * sum((y - mean_b) ** 2 for y in b)) ** .5
    return numerator / denominator if denominator else float("nan")
