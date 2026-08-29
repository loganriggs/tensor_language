"""Pure metrics for the prospectively frozen MLP2 error-Rayleigh pilot.

This module contains no model or artifact access.  It fixes the numerical definitions
used by the future collector and makes them independently testable before outcomes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


EPS = 1e-12


def _finite(value: torch.Tensor, *, ndim: int | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or (
        ndim is not None and value.ndim != ndim
    ) or not bool(torch.isfinite(value).all()):
        raise ValueError("expected a finite floating tensor")
    return value.double()


def symmetric_jvp(plus: torch.Tensor, minus: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return the centered finite directional derivative at amplitudes +/- alpha."""
    plus, minus = _finite(plus), _finite(minus)
    if plus.shape != minus.shape or not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError("symmetric JVP inputs or amplitude are malformed")
    return (plus - minus) / (2.0 * float(alpha))


def categorical_fisher_quadratic(
    native_logits: torch.Tensor, delta_logits: torch.Tensor,
) -> torch.Tensor:
    """Exact categorical-Fisher quadratic, reduced to one value per document.

    Inputs are ``[document, position, vocabulary]`` capped logits and their directional
    derivative.  The result averages over positions but not documents.
    """
    native_logits = _finite(native_logits, ndim=3)
    delta_logits = _finite(delta_logits, ndim=3)
    if native_logits.shape != delta_logits.shape or native_logits.shape[-1] < 2:
        raise ValueError("logit tensors must have one shared nontrivial shape")
    probabilities = F.softmax(native_logits, dim=-1)
    first = (probabilities * delta_logits.square()).sum(dim=-1)
    second = (probabilities * delta_logits).sum(dim=-1).square()
    value = (first - second).mean(dim=-1)
    if bool((value < -1e-10).any()):
        raise RuntimeError("categorical Fisher quadratic is materially negative")
    return value.clamp_min(0).contiguous()


def teacher_kl(native_logits: torch.Tensor, changed_logits: torch.Tensor) -> torch.Tensor:
    """Teacher KL(native || changed), averaged over positions per document."""
    native_logits = _finite(native_logits, ndim=3)
    changed_logits = _finite(changed_logits, ndim=3)
    if native_logits.shape != changed_logits.shape:
        raise ValueError("teacher-KL logit shapes differ")
    native_logp = F.log_softmax(native_logits, dim=-1)
    changed_logp = F.log_softmax(changed_logits, dim=-1)
    probability = native_logp.exp()
    return (probability * (native_logp - changed_logp)).sum(dim=-1).mean(dim=-1)


def normalized_response_energy(delta: torch.Tensor, native: torch.Tensor) -> torch.Tensor:
    """Per-document response energy divided by native consumer-write energy."""
    delta, native = _finite(delta), _finite(native)
    if delta.shape != native.shape or delta.ndim < 2:
        raise ValueError("consumer response tensors must share [document, ...] shape")
    axes = tuple(range(1, delta.ndim))
    denominator = native.square().sum(dim=axes)
    if bool((denominator <= EPS).any()):
        raise ValueError("native consumer energy is degenerate")
    return (delta.square().sum(dim=axes) / denominator).contiguous()


def _average_ranks(value: torch.Tensor) -> torch.Tensor:
    value = _finite(value, ndim=1)
    order = torch.argsort(value, stable=True)
    sorted_value = value[order]
    ranks = torch.empty_like(value)
    start = 0
    while start < len(value):
        stop = start + 1
        while stop < len(value) and sorted_value[stop] == sorted_value[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    """Spearman correlation with average ranks for ties."""
    left, right = _finite(left, ndim=1), _finite(right, ndim=1)
    if left.shape != right.shape or len(left) < 3:
        raise ValueError("Spearman inputs must have one shared length >=3")
    left, right = _average_ranks(left), _average_ranks(right)
    left, right = left - left.mean(), right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= EPS:
        raise ValueError("Spearman input is constant")
    return float((left @ right) / denominator)


def tangent_scale_gate(q_small: torch.Tensor, q_large: torch.Tensor) -> dict[str, Any]:
    """Frozen 1/16-versus-1/8 agreement gate on document-level quadratics."""
    q_small, q_large = _finite(q_small, ndim=1), _finite(q_large, ndim=1)
    if q_small.shape != q_large.shape or bool((q_small < 0).any()) or bool((q_large < 0).any()):
        raise ValueError("tangent quadratic vectors are malformed")
    mean_small, mean_large = float(q_small.mean()), float(q_large.mean())
    relative = abs(mean_small - mean_large) / max(abs(mean_small), abs(mean_large), EPS)
    correlation = spearman(q_small, q_large)
    return {
        "aggregate_relative_disagreement": relative,
        "document_spearman": correlation,
        "passes": relative <= 0.20 and correlation >= 0.80,
    }


def fisher_kl_gate(
    observed_teacher_kl: torch.Tensor, q_logit: torch.Tensor, alpha: float,
) -> dict[str, Any]:
    """Frozen finite-KL versus half-alpha-squared-Fisher gate."""
    observed_teacher_kl = _finite(observed_teacher_kl, ndim=1)
    q_logit = _finite(q_logit, ndim=1)
    if observed_teacher_kl.shape != q_logit.shape or alpha == 0 or (
        bool((observed_teacher_kl < -1e-10).any()) or bool((q_logit <= EPS).any())
    ):
        raise ValueError("Fisher/KL gate inputs are malformed")
    prediction = 0.5 * float(alpha) ** 2 * q_logit
    ratio = float(observed_teacher_kl.mean() / prediction.mean())
    correlation = spearman(observed_teacher_kl, prediction)
    return {
        "aggregate_ratio": ratio,
        "document_spearman": correlation,
        "passes": 0.80 <= ratio <= 1.25 and correlation >= 0.60,
    }


def predictor_gate(
    target: torch.Tensor, local_prediction: torch.Tensor,
    final_prediction: torch.Tensor, full_prediction: torch.Tensor,
) -> dict[str, Any]:
    """Frozen held-out improvement gate for the three DESIGN-fitted predictors."""
    values = [_finite(x, ndim=1) for x in (
        target, local_prediction, final_prediction, full_prediction,
    )]
    if len({tuple(x.shape) for x in values}) != 1 or len(target) < 3:
        raise ValueError("predictor gate inputs must share one document axis")
    target, local_prediction, final_prediction, full_prediction = values

    def mse(prediction: torch.Tensor) -> float:
        return float((target - prediction).square().mean())

    local_mse, final_mse, full_mse = map(mse, (
        local_prediction, final_prediction, full_prediction,
    ))
    local_gain = (local_mse - full_mse) / max(local_mse, EPS)
    final_gain = (final_mse - full_mse) / max(final_mse, EPS)
    correlation = spearman(target, full_prediction)
    return {
        "local_mse": local_mse, "final_mse": final_mse, "full_mse": full_mse,
        "relative_gain_over_local": local_gain,
        "relative_gain_over_final": final_gain,
        "document_spearman": correlation,
        "passes": local_gain >= 0.25 and final_gain >= 0.10 and correlation >= 0.50,
    }


def finite_interaction_gate(
    predicted: torch.Tensor, observed: torch.Tensor, tolerance: float = 0.0025,
) -> dict[str, Any]:
    """Require correct nonzero signs and absolute program-wise error <= tolerance."""
    predicted, observed = _finite(predicted, ndim=1), _finite(observed, ndim=1)
    if predicted.shape != observed.shape or len(predicted) != 3 or tolerance <= 0 or (
        bool((observed.abs() <= EPS).any())
    ):
        raise ValueError("finite interaction vectors are malformed")
    error = (predicted - observed).abs()
    signs = torch.sign(predicted) == torch.sign(observed)
    return {
        "absolute_errors": [float(x) for x in error],
        "signs_correct": [bool(x) for x in signs],
        "passes": bool(signs.all()) and bool((error <= tolerance).all()),
    }
