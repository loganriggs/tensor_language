"""Pure algebra contract for the prospective early-MLP suffix/transport test.

This module deliberately contains no model, row, fitting, or scoring access.  It
defines the physical cross-site operator and its orthogonal interface gauge so the
future numerical runner cannot accidentally interpret coordinate sparsity as a
physical mechanism.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch


D_MODEL = 1152
CODE_DIM = 64


def _matrix(name: str, value: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if not torch.is_tensor(value) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def validate_orthonormal_basis(name: str, basis: torch.Tensor) -> None:
    basis = _matrix(name, basis, (D_MODEL, CODE_DIM)).double()
    error = (basis.T @ basis - torch.eye(CODE_DIM, dtype=torch.float64)).abs().max()
    if float(error) > 2e-4:
        raise ValueError(f"{name} is not orthonormal: max Gram error {float(error)}")


def validate_orthogonal_gauge(name: str, gauge: torch.Tensor) -> None:
    gauge = _matrix(name, gauge, (CODE_DIM, CODE_DIM)).double()
    error = (gauge.T @ gauge - torch.eye(CODE_DIM, dtype=torch.float64)).abs().max()
    if float(error) > 1e-10:
        raise ValueError(f"{name} is not orthogonal: max Gram error {float(error)}")


def physical_cross_map(
    basis0: torch.Tensor, cross: torch.Tensor, basis1: torch.Tensor,
) -> torch.Tensor:
    """Return the gauge-invariant physical map ``B0 A B1^T``."""

    validate_orthonormal_basis("basis0", basis0)
    validate_orthonormal_basis("basis1", basis1)
    cross = _matrix("cross", cross, (CODE_DIM, CODE_DIM))
    dtype = torch.promote_types(torch.promote_types(basis0.dtype, cross.dtype), basis1.dtype)
    return basis0.to(dtype) @ cross.to(dtype) @ basis1.to(dtype).T


def rewrite_cross_map_gauge(
    basis0: torch.Tensor,
    cross: torch.Tensor,
    basis1: torch.Tensor,
    gauge0: torch.Tensor,
    gauge1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rewrite both code bases without changing the physical cross-site map.

    With row-vector coordinates, ``p0' = p0 Q0`` and ``p1' = p1 Q1``.  Therefore
    the transported-coordinate matrix must transform as ``A' = Q0^T A Q1``.
    """

    validate_orthonormal_basis("basis0", basis0)
    validate_orthonormal_basis("basis1", basis1)
    validate_orthogonal_gauge("gauge0", gauge0)
    validate_orthogonal_gauge("gauge1", gauge1)
    cross = _matrix("cross", cross, (CODE_DIM, CODE_DIM))
    dtype = torch.promote_types(torch.promote_types(cross.dtype, gauge0.dtype), gauge1.dtype)
    q0, q1 = gauge0.to(dtype), gauge1.to(dtype)
    return (
        basis0.to(dtype) @ q0,
        q0.T @ cross.to(dtype) @ q1,
        basis1.to(dtype) @ q1,
    )


def transported_physical_write(
    source_write: torch.Tensor,
    basis0: torch.Tensor,
    cross: torch.Tensor,
    basis1: torch.Tensor,
) -> torch.Tensor:
    """Transport a physical row-vector write through the explicit code map."""

    if source_write.shape[-1] != D_MODEL or not torch.isfinite(source_write).all():
        raise ValueError("source_write must be finite and end in d_model")
    operator = physical_cross_map(basis0, cross, basis1)
    return source_write.to(operator.dtype) @ operator


def rewrite_affine_output_gauge(
    weight: torch.Tensor, bias: torch.Tensor, gauge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rewrite ``x W + b`` into coordinates ``p' = p Q``."""

    weight = _matrix("weight", weight, (D_MODEL, CODE_DIM))
    if not torch.is_tensor(bias) or tuple(bias.shape) != (CODE_DIM,) or not (
        torch.isfinite(bias).all()
    ):
        raise ValueError("bias must be a finite code vector")
    validate_orthogonal_gauge("gauge", gauge)
    dtype = torch.promote_types(torch.promote_types(weight.dtype, bias.dtype), gauge.dtype)
    moved = gauge.to(dtype)
    return weight.to(dtype) @ moved, bias.to(dtype) @ moved


def rewrite_code_gauge(code: torch.Tensor, gauge: torch.Tensor) -> torch.Tensor:
    """Rewrite executable codes, target labels, or perturbations by ``Q``."""

    if code.shape[-1] != CODE_DIM or not torch.isfinite(code).all():
        raise ValueError("code must be finite and end in code_dim")
    validate_orthogonal_gauge("gauge", gauge)
    dtype = torch.promote_types(code.dtype, gauge.dtype)
    return code.to(dtype) @ gauge.to(dtype)


def apply_physical_code_edit(
    residual: torch.Tensor, delta: torch.Tensor, basis: torch.Tensor,
) -> torch.Tensor:
    """Apply a row-code edit as the physical row-vector write ``delta B^T``."""

    if residual.shape[-1] != D_MODEL or not torch.isfinite(residual).all():
        raise ValueError("residual must be finite and end in d_model")
    if delta.shape[-1] != CODE_DIM or not torch.isfinite(delta).all():
        raise ValueError("delta must be finite and end in code_dim")
    validate_orthonormal_basis("basis", basis)
    dtype = torch.promote_types(torch.promote_types(residual.dtype, delta.dtype), basis.dtype)
    return residual.to(dtype) + delta.to(dtype) @ basis.to(dtype).T


def covariance_shaped_directions(
    codes: torch.Tensor,
    *,
    count: int = 32,
    seed_start: int = 2026083200,
) -> Mapping[str, torch.Tensor]:
    """Construct the frozen float64 covariance-shaped Rademacher direction bank."""

    if (
        not torch.is_tensor(codes)
        or codes.ndim != 2
        or tuple(codes.shape)[1] != CODE_DIM
        or tuple(codes.shape)[0] < 2
        or not torch.isfinite(codes).all()
    ):
        raise ValueError("codes must be a finite [n>=2, code_dim] tensor")
    if count <= 0:
        raise ValueError("count must be positive")

    values = codes.detach().cpu().double()
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / (values.shape[0] - 1)
    trace = torch.trace(covariance)
    if not torch.isfinite(trace) or float(trace) <= 0:
        raise ValueError("code covariance must have positive finite trace")
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = trace * (1e-12 / CODE_DIM)
    clipped = torch.clamp(eigenvalues, min=floor)
    square_root = (eigenvectors * torch.sqrt(clipped).unsqueeze(0)) @ eigenvectors.T

    raw_signs = []
    directions = []
    for index in range(count):
        generator = torch.Generator(device="cpu").manual_seed(seed_start + index)
        sign = 2 * torch.randint(0, 2, (CODE_DIM,), generator=generator) - 1
        sign = sign.double()
        direction = sign @ square_root
        rms = torch.sqrt(torch.mean(direction.square()))
        if not torch.isfinite(rms) or float(rms) <= 0:
            raise ValueError("direction RMS must be positive and finite")
        raw_signs.append(sign)
        directions.append(direction / rms)

    return {
        "mean": mean,
        "covariance": covariance,
        "eigenvalues": eigenvalues,
        "clipped_eigenvalues": clipped,
        "covariance_square_root": square_root,
        "code_rms": torch.sqrt(torch.mean(centered.square())),
        "raw_signs": torch.stack(raw_signs),
        "directions": torch.stack(directions),
    }


def pooled_response_metrics(
    student_response: torch.Tensor,
    teacher_response: torch.Tensor,
    *,
    center_last_dimension: bool = False,
) -> Mapping[str, float]:
    """Compute the preregistered pooled NRE, response R2, and cosine."""

    if (
        not torch.is_tensor(student_response)
        or not torch.is_tensor(teacher_response)
        or student_response.shape != teacher_response.shape
        or student_response.ndim < 2
        or not torch.isfinite(student_response).all()
        or not torch.isfinite(teacher_response).all()
    ):
        raise ValueError("responses must be same-shaped finite tensors with ndim >= 2")
    student = student_response.detach().cpu().double()
    teacher = teacher_response.detach().cpu().double()
    if center_last_dimension:
        student = student - student.mean(dim=-1, keepdim=True)
        teacher = teacher - teacher.mean(dim=-1, keepdim=True)

    error_sum = torch.sum((student - teacher).square())
    teacher_sum = torch.sum(teacher.square())
    student_sum = torch.sum(student.square())
    if float(teacher_sum) <= 1e-12:
        raise ValueError("teacher-response denominator is too small")
    cosine_denominator = torch.sqrt(student_sum) * torch.sqrt(teacher_sum)
    if float(cosine_denominator) <= 1e-12:
        raise ValueError("response cosine denominator is too small")
    nre = torch.sqrt(error_sum / teacher_sum)
    cosine = torch.sum(student * teacher) / cosine_denominator
    return {
        "error_sum": float(error_sum),
        "teacher_sum": float(teacher_sum),
        "student_sum": float(student_sum),
        "nre": float(nre),
        "r2": float(1 - nre.square()),
        "cosine": float(cosine),
    }


def finite_null_rank(primary: float, nulls: torch.Tensor) -> int:
    """Return ``1 + count(null >= primary)`` for the frozen higher-is-better bank."""

    if not torch.is_tensor(nulls) or nulls.ndim != 1 or not torch.isfinite(nulls).all():
        raise ValueError("nulls must be a finite vector")
    primary_tensor = torch.tensor(primary, dtype=torch.float64)
    if not torch.isfinite(primary_tensor):
        raise ValueError("primary must be finite")
    return 1 + int(torch.sum(nulls.detach().cpu().double() >= primary_tensor))


def incremental_price() -> Mapping[str, Any]:
    """Frozen v1 price of the uncentered dense 64-to-64 cross-map."""

    reals = CODE_DIM * CODE_DIM
    return {
        "grammar": "dense_uncentered_64x64",
        "cross_rank": CODE_DIM,
        "incremental_reals": int(reals),
        "float32_bits": int(32 * reals),
        "incremental_multiplies_per_token": int(reals),
        "basis_and_parent_program_price_excluded": True,
    }
