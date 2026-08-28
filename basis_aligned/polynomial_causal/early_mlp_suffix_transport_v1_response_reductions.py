"""Pure reductions for one paired finite-response batch.

The observed backend owns the multidimensional tensors.  This module turns the
shared exact-teacher and one student baseline/positive/negative triplet into the
only row-level scalar sufficient statistics allowed to leave that backend.

Both antithetic edits are occurrences.  For a triplet ``(x0, x+, x-)`` the two
responses are ``x+ - x0`` and ``x- - x0``; they are not replaced by a central
difference.  Centered-logit response subtracts the vocabulary mean separately
from every response vector, exactly as preregistered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

import early_mlp_suffix_transport_v1_runtime as runtime
import early_mlp_suffix_transport_v1_statistics as statistics


BATCH_SIZE = 4
SCORED_POSITION_COUNT = 192
CODE_DIM = 64


@dataclass(frozen=True, slots=True)
class ResponseTriplet:
    """One subject's unedited, positive-edit, and negative-edit tensors."""

    baseline: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor

    def __post_init__(self) -> None:
        values = (self.baseline, self.positive, self.negative)
        if any(not torch.is_tensor(value) for value in values):
            raise TypeError("response triplet values must be tensors")
        shape = self.baseline.shape
        if self.baseline.ndim < 2 or any(value.shape != shape for value in values):
            raise ValueError("response triplet values must have one common shape")
        if shape[0] <= 0 or any(value.requires_grad or value.grad_fn is not None for value in values):
            raise ValueError("response triplet must be detached and nonempty")
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("response triplet contains nonfinite values")


@dataclass(frozen=True, slots=True)
class BatchResponseReduction:
    """Per-row response inner products for both antithetic occurrences."""

    error_sum: torch.Tensor
    teacher_sum: torch.Tensor
    student_sum: torch.Tensor
    dot_sum: torch.Tensor
    unit_identity: str

    def __post_init__(self) -> None:
        payload = {
            "error_sum": self.error_sum,
            "teacher_sum": self.teacher_sum,
            "student_sum": self.student_sum,
            "dot_sum": self.dot_sum,
            "unit_identity": self.unit_identity,
        }
        checked = statistics.validate_response_sufficient_statistics(payload)
        for name, value in checked.items():
            object.__setattr__(self, name, value.detach().clone().contiguous())

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            name: runtime.tensor_identity_sha256(getattr(self, name))
            for name in statistics.RESPONSE_KEYS
        } | {"unit_identity": self.unit_identity})


@dataclass(frozen=True, slots=True)
class BatchOutputKLReduction:
    """Per-row edited student KL divided later by exact-teacher edit KL."""

    numerator_sum: torch.Tensor
    denominator_sum: torch.Tensor
    unit_identity: str

    def __post_init__(self) -> None:
        payload = {
            "numerator_sum": self.numerator_sum,
            "denominator_sum": self.denominator_sum,
            "unit_identity": self.unit_identity,
        }
        checked = statistics.validate_output_kl_sufficient_statistics(payload)
        for name, value in checked.items():
            object.__setattr__(self, name, value.detach().clone().contiguous())

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            name: runtime.tensor_identity_sha256(getattr(self, name))
            for name in statistics.OUTPUT_KL_KEYS
        } | {"unit_identity": self.unit_identity})


def _unit_identity(value: Any) -> str:
    if not runtime._sha256_text(value):
        raise ValueError("response unit identity must be a SHA-256 identity")
    return value


def _responses(
    triplet: ResponseTriplet, *, center_last_dimension: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(triplet, ResponseTriplet):
        raise TypeError("response reduction requires a typed triplet")
    positive = triplet.positive.detach().cpu().double() - triplet.baseline.detach().cpu().double()
    negative = triplet.negative.detach().cpu().double() - triplet.baseline.detach().cpu().double()
    if center_last_dimension:
        positive = positive - positive.mean(dim=-1, keepdim=True)
        negative = negative - negative.mean(dim=-1, keepdim=True)
    return positive.contiguous(), negative.contiguous()


def _reduce_vector_response(
    *, teacher: ResponseTriplet, student: ResponseTriplet,
    unit_identity: str, center_last_dimension: bool = False,
) -> BatchResponseReduction:
    """Reduce aligned teacher/student responses without releasing the vectors."""

    if not isinstance(center_last_dimension, bool):
        raise TypeError("response centering flag must be boolean")
    if teacher.baseline.shape != student.baseline.shape:
        raise ValueError("teacher and student response supports differ")
    teacher_responses = _responses(
        teacher, center_last_dimension=center_last_dimension,
    )
    student_responses = _responses(
        student, center_last_dimension=center_last_dimension,
    )
    batch = teacher.baseline.shape[0]
    teacher_flat = torch.stack(teacher_responses, dim=1).reshape(batch, -1)
    student_flat = torch.stack(student_responses, dim=1).reshape(batch, -1)
    error = student_flat - teacher_flat
    return BatchResponseReduction(
        error_sum=error.square().sum(dim=1),
        teacher_sum=teacher_flat.square().sum(dim=1),
        student_sum=student_flat.square().sum(dim=1),
        dot_sum=(student_flat * teacher_flat).sum(dim=1),
        unit_identity=_unit_identity(unit_identity),
    )


def reduce_code_response(
    *, teacher: ResponseTriplet, student: ResponseTriplet, unit_identity: str,
) -> BatchResponseReduction:
    """Reduce MLP1-coordinate responses on every scored position 64--255."""

    expected = (BATCH_SIZE, SCORED_POSITION_COUNT, CODE_DIM)
    if teacher.baseline.shape != expected or student.baseline.shape != expected:
        raise ValueError("code response support must be [4,192,64]")
    return _reduce_vector_response(
        teacher=teacher, student=student, unit_identity=unit_identity,
        center_last_dimension=False,
    )


def reduce_centered_logit_response(
    *, teacher: ResponseTriplet, student: ResponseTriplet, unit_identity: str,
) -> BatchResponseReduction:
    """Reduce vocabulary-centered logit responses on positions 64--255."""

    shapes = (teacher.baseline.shape, student.baseline.shape)
    if any(len(shape) != 3 or shape[:2] != (
        BATCH_SIZE, SCORED_POSITION_COUNT,
    ) or shape[2] <= 1 for shape in shapes):
        raise ValueError("logit response support must be [4,192,vocab]")
    return _reduce_vector_response(
        teacher=teacher, student=student, unit_identity=unit_identity,
        center_last_dimension=True,
    )


def _row_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    if reference_logits.shape != candidate_logits.shape or reference_logits.ndim < 2:
        raise ValueError("KL logits must share a shape with a vocabulary dimension")
    reference = reference_logits.detach().cpu().double()
    candidate = candidate_logits.detach().cpu().double()
    reference_log_prob = torch.log_softmax(reference, dim=-1)
    candidate_log_prob = torch.log_softmax(candidate, dim=-1)
    reference_prob = reference_log_prob.exp()
    elementwise = reference_prob * (reference_log_prob - candidate_log_prob)
    return elementwise.reshape(elementwise.shape[0], -1).sum(dim=1).contiguous()


def reduce_output_kl_response(
    *, teacher: ResponseTriplet, student: ResponseTriplet, unit_identity: str,
) -> BatchOutputKLReduction:
    """Reduce the two edited output KL occurrences for every row.

    Numerator: KL(exact teacher edit || matching student edit).
    Denominator: KL(exact teacher edit || exact teacher baseline).
    Student baselines are intentionally irrelevant to this registered ratio.
    """

    shapes = (teacher.baseline.shape, student.baseline.shape)
    if any(len(shape) != 3 or shape[:2] != (
        BATCH_SIZE, SCORED_POSITION_COUNT,
    ) or shape[2] <= 1 for shape in shapes):
        raise ValueError("output-KL support must be [4,192,vocab]")
    numerator = _row_kl(teacher.positive, student.positive) + _row_kl(
        teacher.negative, student.negative,
    )
    denominator = _row_kl(teacher.positive, teacher.baseline) + _row_kl(
        teacher.negative, teacher.baseline,
    )
    return BatchOutputKLReduction(
        numerator_sum=numerator, denominator_sum=denominator,
        unit_identity=_unit_identity(unit_identity),
    )
