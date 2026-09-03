#!/usr/bin/env python3
"""Pure Grassmann geometry and frozen scoring for rung 524."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch


D = 64
RANK = 4
TARGET_COUNT = 3
MAP_COUNT = 4
OUTPUT_DIMENSION = 12
FIT_EXAMPLES = 96
VALIDATION_EXAMPLES = 192
OOD_EXAMPLES = 192
FIT_COUNT = 15
UPDATES = 200
INITIAL_STEP = 0.5
ARMIJO = 1e-4
BACKTRACK_FACTOR = 0.5
MAX_BACKTRACKS = 16
CONTROL_COEFFICIENT = 24.0


def canonical_qr(value: torch.Tensor) -> torch.Tensor:
    """Return a deterministic orthonormal basis for a full-column-rank matrix."""
    if value.ndim != 2 or value.shape[0] < value.shape[1]:
        raise ValueError("value must be a tall matrix")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("value must be finite and floating")
    q, r = torch.linalg.qr(value, mode="reduced")
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return q * signs.unsqueeze(0)


def grassmann_tangent(frame: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    """Project an ambient gradient onto the horizontal Grassmann tangent space."""
    if frame.ndim != 2 or gradient.shape != frame.shape:
        raise ValueError("frame and gradient shapes differ")
    return gradient - frame @ (frame.mT @ gradient)


def retract(frame: torch.Tensor, tangent_step: torch.Tensor) -> torch.Tensor:
    """Retract a tangent displacement to an orthonormal frame using QR."""
    if frame.shape != tangent_step.shape:
        raise ValueError("frame and tangent-step shapes differ")
    return canonical_qr(frame + tangent_step)


def projector_error(frame: torch.Tensor, planted: torch.Tensor) -> float:
    """Relative Frobenius distance between two rank-matched projectors."""
    if frame.shape != planted.shape:
        raise ValueError("frame and planted shapes differ")
    p = frame @ frame.mT
    p_star = planted @ planted.mT
    return float(torch.linalg.vector_norm(p - p_star) / torch.linalg.vector_norm(p_star))


def minimum_principal_cosine(frame: torch.Tensor, planted: torch.Tensor) -> float:
    """Smallest singular value of the overlap between two orthonormal frames."""
    if frame.shape != planted.shape:
        raise ValueError("frame and planted shapes differ")
    return float(torch.linalg.svdvals(planted.mT @ frame).min())


@dataclass(frozen=True)
class PretestFit:
    accepted_updates: int
    finite_losses: bool
    finite_gradients: bool
    maximum_evaluated_loss: float
    orthonormality_error: float
    initial_fit_loss: float
    final_fit_loss: float
    initial_validation_loss: float
    final_validation_loss: float
    projector_error: float
    minimum_principal_cosine: float


def score_pretest(fits: Sequence[PretestFit]) -> dict[str, object]:
    """Apply the preregistered numerical, validation, and recovery gates."""
    if len(fits) != FIT_COUNT:
        raise ValueError(f"expected exactly {FIT_COUNT} fits")
    records = []
    for index, fit in enumerate(fits):
        failures = []
        finite_scalars = all(math.isfinite(value) for value in (
            fit.maximum_evaluated_loss,
            fit.orthonormality_error,
            fit.initial_fit_loss,
            fit.final_fit_loss,
            fit.initial_validation_loss,
            fit.final_validation_loss,
            fit.projector_error,
            fit.minimum_principal_cosine,
        ))
        if fit.accepted_updates != UPDATES:
            failures.append("accepted_update_count")
        if not fit.finite_losses or not finite_scalars:
            failures.append("nonfinite_loss_or_score")
        if not fit.finite_gradients:
            failures.append("nonfinite_gradient")
        if not math.isfinite(fit.orthonormality_error) or fit.orthonormality_error > 1e-5:
            failures.append("orthonormality")
        if (
            not math.isfinite(fit.initial_fit_loss)
            or not math.isfinite(fit.final_fit_loss)
            or fit.final_fit_loss > 0.05 * fit.initial_fit_loss
        ):
            failures.append("fit_reduction")
        if (
            not math.isfinite(fit.initial_validation_loss)
            or not math.isfinite(fit.final_validation_loss)
            or fit.final_validation_loss > 0.05 * fit.initial_validation_loss
        ):
            failures.append("validation_reduction")
        if not math.isfinite(fit.maximum_evaluated_loss) or fit.maximum_evaluated_loss > 100:
            failures.append("loss_above_100")
        if not math.isfinite(fit.projector_error) or fit.projector_error > 0.10:
            failures.append("projector_recovery")
        if (
            not math.isfinite(fit.minimum_principal_cosine)
            or fit.minimum_principal_cosine < 0.995
        ):
            failures.append("principal_cosine")
        records.append({"fit_index": index, "passes": not failures, "failures": failures})
    a_failures = {
        "accepted_update_count", "nonfinite_loss_or_score", "nonfinite_gradient",
        "orthonormality", "fit_reduction", "validation_reduction", "loss_above_100",
    }
    b_failures = {"projector_recovery", "principal_cosine"}
    prediction_a = all(not a_failures.intersection(record["failures"]) for record in records)
    prediction_b = all(not b_failures.intersection(record["failures"]) for record in records)
    return {
        "fit_count": len(fits),
        "passing_fit_count": sum(record["passes"] for record in records),
        "prediction_a_numerical_and_validation": prediction_a,
        "prediction_b_subspace_recovery": prediction_b,
        "pretest_passes": prediction_a and prediction_b,
        "fits": records,
    }


def score_ood(losses: Sequence[float]) -> dict[str, object]:
    """Apply the frozen OOD response-transfer rule after the pretest seal opens."""
    if len(losses) != FIT_COUNT:
        raise ValueError(f"expected exactly {FIT_COUNT} OOD losses")
    finite = all(math.isfinite(float(value)) for value in losses)
    maximum = max(float(value) for value in losses) if finite else math.inf
    return {
        "fit_count": len(losses),
        "finite": finite,
        "maximum_normalized_ood_loss": maximum,
        "prediction_c_ood_transfer": finite and maximum <= 0.05,
    }


def final_decision(pretest: Mapping[str, object], ood: Mapping[str, object] | None) -> dict[str, object]:
    """Fail closed: OOD must be absent before pretest or present after a pass."""
    pretest_passes = pretest.get("pretest_passes") is True
    if not pretest_passes:
        if ood is not None:
            raise ValueError("OOD result exists despite failed pretest seal")
        return {
            "instrument_passes": False,
            "licenses_one_model_calibration": False,
            "next_action": "pivot_to_mlp0_exact_branch_decomposition",
        }
    if ood is None:
        raise ValueError("pretest passed but OOD result is absent")
    passed = ood.get("prediction_c_ood_transfer") is True
    return {
        "instrument_passes": passed,
        "licenses_one_model_calibration": passed,
        "next_action": (
            "one_unchanged_model_direct_subspace_calibration"
            if passed else "pivot_to_mlp0_exact_branch_decomposition"
        ),
    }


__all__ = [
    "ARMIJO", "BACKTRACK_FACTOR", "CONTROL_COEFFICIENT", "D", "FIT_COUNT",
    "FIT_EXAMPLES", "INITIAL_STEP", "MAP_COUNT", "MAX_BACKTRACKS", "OOD_EXAMPLES",
    "OUTPUT_DIMENSION", "PretestFit", "RANK", "TARGET_COUNT", "UPDATES",
    "VALIDATION_EXAMPLES", "canonical_qr", "final_decision", "grassmann_tangent",
    "minimum_principal_cosine", "projector_error", "retract", "score_ood",
    "score_pretest",
]
