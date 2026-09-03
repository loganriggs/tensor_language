"""Pure CPU-safe mathematics for the rung-522 selective projector.

This module imports no model or data-loading code.  It defines the registered
rank-four projector action, the exact max-over-target training objective, a
deterministic optimizer, gauge-invariant diagnostics, and response metrics.
The model entrypoint may call these functions later, but no function here
touches CUDA or performs model inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import torch

import das_shared_private_lib as daslib


@dataclass(frozen=True)
class OptimizerConfig:
    """Pre-outcome optimizer choices proposed by the CPU preflight addendum."""

    rank: int = 4
    control_coefficient: float = 24.0
    learning_rate: float = 0.03
    updates: int = 200
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    loss_epsilon: float = 1e-12
    health_window: int = 20
    orthonormality_atol: float = 1e-5
    minimum_projector_distance: float = 0.02

    def validate(self, dimension: int) -> None:
        if not isinstance(dimension, int) or dimension < self.rank:
            raise ValueError("dimension must be an integer at least as large as rank")
        if not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("rank must be a positive integer")
        positive = {
            "learning_rate": self.learning_rate,
            "adam_epsilon": self.adam_epsilon,
            "loss_epsilon": self.loss_epsilon,
            "orthonormality_atol": self.orthonormality_atol,
            "minimum_projector_distance": self.minimum_projector_distance,
        }
        if any(not math.isfinite(value) or value <= 0 for value in positive.values()):
            raise ValueError("all positive optimizer constants must be finite and positive")
        if not math.isfinite(self.control_coefficient) or self.control_coefficient < 0:
            raise ValueError("control_coefficient must be finite and nonnegative")
        if not isinstance(self.updates, int) or self.updates < 2 * self.health_window:
            raise ValueError("updates must cover the first and final health windows")
        if not isinstance(self.health_window, int) or self.health_window <= 0:
            raise ValueError("health_window must be a positive integer")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("Adam beta values must lie in [0, 1)")


@dataclass(frozen=True)
class TargetResponse:
    """Signed response vectors for one target in one optimizer update."""

    full_member: torch.Tensor
    projected_member: torch.Tensor
    projected_control: torch.Tensor


@dataclass(frozen=True)
class TargetLoss:
    member: torch.Tensor
    control: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class ObjectiveResult:
    """Exact maximum objective and its per-target terms."""

    maximum: torch.Tensor
    per_target: Mapping[str, TargetLoss]
    maximizing_target: str


@dataclass(frozen=True)
class FitResult:
    frame: torch.Tensor
    initial_frame: torch.Tensor
    loss_history: tuple[float, ...]
    maximizing_target_history: tuple[str, ...]
    initial_validation_objective: float
    final_validation_objective: float
    final_orthonormality_error: float
    projector_distance_from_initialization: float
    healthy: bool
    health_failures: tuple[str, ...]


ResponseFunction = Callable[[torch.Tensor, int], Mapping[str, TargetResponse]]


def _one_dimensional_finite(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must have a floating dtype")
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise ValueError(f"{name} contains a non-finite value")


def target_loss(
    response: TargetResponse,
    *,
    control_coefficient: float,
    epsilon: float,
) -> TargetLoss:
    """Compute the two normalized losses frozen in the rung-522 preregistration."""
    _one_dimensional_finite(response.full_member, "full_member")
    _one_dimensional_finite(response.projected_member, "projected_member")
    _one_dimensional_finite(response.projected_control, "projected_control")
    if response.full_member.shape != response.projected_member.shape:
        raise ValueError("full and projected member responses must have identical shape")
    if response.full_member.device != response.projected_control.device:
        raise ValueError("member and control responses must occupy the same device")
    if not math.isfinite(control_coefficient) or control_coefficient < 0:
        raise ValueError("control_coefficient must be finite and nonnegative")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    denominator = response.full_member.square().mean() + epsilon
    member = (response.projected_member - response.full_member).square().mean() / denominator
    control = response.projected_control.square().mean() / denominator
    return TargetLoss(member=member, control=control, total=member + control_coefficient * control)


def exact_max_target_objective(
    responses: Mapping[str, TargetResponse],
    *,
    control_coefficient: float,
    epsilon: float,
) -> ObjectiveResult:
    """Return ``max_target (L_member + coefficient L_control)``, never an average."""
    if len(responses) < 1:
        raise ValueError("at least one target response is required")
    ordered_names = tuple(sorted(responses))
    losses = {
        name: target_loss(
            responses[name], control_coefficient=control_coefficient, epsilon=epsilon
        )
        for name in ordered_names
    }
    stacked = torch.stack(tuple(losses[name].total for name in ordered_names))
    maximum, index = torch.max(stacked, dim=0)
    return ObjectiveResult(
        maximum=maximum,
        per_target=losses,
        maximizing_target=ordered_names[int(index.detach().cpu())],
    )


def deterministic_haar_frame(
    dimension: int,
    rank: int,
    seed: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Seed on CPU deterministically, then move the orthonormal frame to ``device``."""
    if not isinstance(dimension, int) or not isinstance(rank, int) or dimension < rank or rank <= 0:
        raise ValueError("require integer dimension >= rank > 0")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("frame dtype must be float32 or float64")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    raw = torch.randn(dimension, rank, generator=generator, dtype=torch.float64)
    return daslib.symmetric_polar_retraction(raw).to(device=device, dtype=dtype)


def differentiable_qr_retraction(matrix: torch.Tensor) -> torch.Tensor:
    """Stable differentiable Stiefel retraction with deterministic QR signs.

    The existing symmetric-polar helper differentiates through an eigendecomposition
    of ``Q^T Q``.  At an exactly orthonormal iterate all four eigenvalues coincide,
    for which PyTorch's eigenvector derivative is undefined and produces NaNs.
    Reduced QR is an equivalent orthonormal retraction and has a finite derivative
    at the registered full-rank iterates.  Column signs are fixed from ``diag(R)``;
    scientific comparisons remain projector-gauge invariant.
    """
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
        raise ValueError("matrix must be rank two")
    if matrix.shape[0] < matrix.shape[1] or matrix.shape[1] == 0:
        raise ValueError("matrix must have shape dimension >= rank > 0")
    if matrix.dtype not in (torch.float32, torch.float64):
        raise ValueError("QR retraction requires float32 or float64")
    if not bool(torch.isfinite(matrix).all().detach().cpu()):
        raise ValueError("matrix contains a non-finite value")
    frame, triangular = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.diagonal(triangular)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return frame * signs.unsqueeze(0)


def projected_bilinear_response(
    displacement: torch.Tensor, reader: torch.Tensor, frame: torch.Tensor
) -> torch.Tensor:
    """Toy/local-linear signed response ``x QQ^T g`` for aligned row pairs."""
    if displacement.shape != reader.shape or displacement.ndim != 2:
        raise ValueError("displacement and reader must be equal-shaped matrices")
    if displacement.shape[1] != frame.shape[0]:
        raise ValueError("response vectors do not match the frame dimension")
    if displacement.device != frame.device or reader.device != frame.device:
        raise ValueError("response vectors and frame must occupy the same device")
    return ((displacement @ frame) * (reader @ frame)).sum(dim=-1)


def full_bilinear_response(displacement: torch.Tensor, reader: torch.Tensor) -> torch.Tensor:
    """Toy/local-linear full-component response ``x g`` for aligned row pairs."""
    if displacement.shape != reader.shape or displacement.ndim != 2:
        raise ValueError("displacement and reader must be equal-shaped matrices")
    return (displacement * reader).sum(dim=-1)


def signed_response_metrics(projected: torch.Tensor, full: torch.Tensor) -> dict[str, float]:
    """Cosine, best-scaled residual, and aligned recovery used by rung 522."""
    _one_dimensional_finite(projected, "projected")
    _one_dimensional_finite(full, "full")
    if projected.shape != full.shape:
        raise ValueError("projected and full responses must have identical shape")
    projected = projected.double()
    full = full.double()
    dot = float(projected @ full)
    projected_norm2 = float(projected @ projected)
    full_norm2 = float(full @ full)
    cosine = dot / math.sqrt(max(projected_norm2 * full_norm2, 1e-30))
    scale = dot / max(projected_norm2, 1e-30)
    residual = float((scale * projected - full).norm()) / math.sqrt(max(full_norm2, 1e-30))
    recovery = dot / max(full_norm2, 1e-30)
    return {
        "signed_cosine": cosine,
        "optimal_scale_projected_to_full": scale,
        "relative_residual": residual,
        "aligned_recovery": recovery,
    }


def response_concentration(member: torch.Tensor, control: torch.Tensor) -> dict[str, float]:
    """RMS selectivity statistic used by rung 522 and its 25%-control gate."""
    _one_dimensional_finite(member, "member")
    _one_dimensional_finite(control, "control")
    member_rms = float(member.double().square().mean().sqrt())
    control_rms = float(control.double().square().mean().sqrt())
    return {
        "member_rms": member_rms,
        "control_rms": control_rms,
        "control_to_member_ratio": control_rms / max(member_rms, 1e-30),
        "member_to_control_concentration": member_rms / max(control_rms, 1e-30),
    }


def _evaluate_objective(
    frame: torch.Tensor,
    response_function: ResponseFunction,
    step: int,
    config: OptimizerConfig,
) -> ObjectiveResult:
    return exact_max_target_objective(
        response_function(frame, step),
        control_coefficient=config.control_coefficient,
        epsilon=config.loss_epsilon,
    )


def fit_projector(
    dimension: int,
    seed: int,
    training_responses: ResponseFunction,
    validation_responses: ResponseFunction,
    *,
    config: OptimizerConfig = OptimizerConfig(),
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> FitResult:
    """Fit one projector with the frozen Adam/max-target/QR rule.

    The response callbacks retain responsibility for batching and, in future
    science code, executing the frozen model suffix.  They must return every
    required target on every update.  The core never averages target losses.
    """
    config.validate(dimension)
    initial = deterministic_haar_frame(
        dimension, config.rank, seed, dtype=dtype, device=device
    )
    raw = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam(
        (raw,),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
    )
    with torch.no_grad():
        initial_validation = float(
            _evaluate_objective(initial, validation_responses, -1, config).maximum.detach()
        )

    history: list[float] = []
    maximizing: list[str] = []
    failures: list[str] = []
    expected_targets: tuple[str, ...] | None = None
    for update in range(config.updates):
        optimizer.zero_grad(set_to_none=True)
        frame = differentiable_qr_retraction(raw)
        responses = training_responses(frame, update)
        names = tuple(sorted(responses))
        if expected_targets is None:
            expected_targets = names
        elif names != expected_targets:
            raise ValueError("training callback changed target identities across updates")
        objective = exact_max_target_objective(
            responses,
            control_coefficient=config.control_coefficient,
            epsilon=config.loss_epsilon,
        )
        if not bool(torch.isfinite(objective.maximum).detach().cpu()):
            raise FloatingPointError(f"non-finite objective at update {update}")
        objective.maximum.backward()
        if raw.grad is None or not bool(torch.isfinite(raw.grad).all().detach().cpu()):
            raise FloatingPointError(f"non-finite or absent frame gradient at update {update}")
        optimizer.step()
        history.append(float(objective.maximum.detach()))
        maximizing.append(objective.maximizing_target)

    final = differentiable_qr_retraction(raw.detach())
    final_validation = float(
        _evaluate_objective(final, validation_responses, -1, config).maximum.detach()
    )
    orth_error = float(daslib.orthonormality_error(final))
    distance = float(daslib.projector_frobenius_distance(initial, final))
    window = config.health_window
    if not all(math.isfinite(value) for value in history):
        failures.append("nonfinite_loss")
    if sum(history[-window:]) / window >= sum(history[:window]) / window:
        failures.append("final_window_not_below_initial_window")
    if final_validation >= initial_validation:
        failures.append("validation_not_better_than_initialization")
    if orth_error > config.orthonormality_atol:
        failures.append("orthonormality")
    if distance <= config.minimum_projector_distance:
        failures.append("projector_did_not_move")
    return FitResult(
        frame=final,
        initial_frame=initial,
        loss_history=tuple(history),
        maximizing_target_history=tuple(maximizing),
        initial_validation_objective=initial_validation,
        final_validation_objective=final_validation,
        final_orthonormality_error=orth_error,
        projector_distance_from_initialization=distance,
        healthy=not failures,
        health_failures=tuple(failures),
    )


def assert_parameters_have_no_gradients(parameters: Sequence[torch.nn.Parameter]) -> None:
    """Fail if any frozen model parameter acquired a gradient."""
    offenders = [index for index, parameter in enumerate(parameters) if parameter.grad is not None]
    if offenders:
        raise RuntimeError(f"frozen model parameters acquired gradients at indices {offenders[:8]}")
