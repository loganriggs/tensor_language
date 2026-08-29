"""Weights-only mixed-block leakage for a fixed quadratic-program projector."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class LeakageEstimate:
    samples: int
    numerator: float
    denominator: float
    leakage: float


def _validate_factors(
    left: torch.Tensor, right: torch.Tensor, decoder: torch.Tensor, basis: torch.Tensor,
) -> None:
    values = (left, right, decoder, basis)
    if any(
        not torch.is_tensor(value) or value.device.type != "cpu" or value.ndim != 2
        or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
        for value in values
    ):
        raise ValueError("factors and basis must be finite CPU float32 matrices")
    if left.shape != right.shape or decoder.shape != (left.shape[1], left.shape[0]):
        raise ValueError("expected Left/Right [K,d] and decoder [d,K]")
    if basis.shape[0] != left.shape[1] or not 1 <= basis.shape[1] < basis.shape[0]:
        raise ValueError("basis must have shape [d,r] with 1 <= r < d")
    gram_error = float((basis.T @ basis - torch.eye(basis.shape[1])).abs().max())
    if gram_error > 2e-4:
        raise ValueError(f"basis is not orthonormal: {gram_error}")


def orthonormal_union(*bases: torch.Tensor) -> torch.Tensor:
    if len(bases) < 1 or any(value.ndim != 2 for value in bases):
        raise ValueError("at least one matrix basis is required")
    if len({(value.shape[0], value.dtype, value.device.type) for value in bases}) != 1:
        raise ValueError("union bases must share width, dtype, and device")
    q, r = torch.linalg.qr(torch.cat(bases, dim=1), mode="reduced")
    diagonal = r.diagonal().abs()
    tolerance = max(float(diagonal.max()), 1.0) * 1e-5
    rank = int((diagonal > tolerance).sum())
    if rank != sum(value.shape[1] for value in bases):
        raise RuntimeError("union bases are not jointly independent")
    return q[:, :rank].contiguous()


def haar_basis(width: int, rank: int, seed: int) -> torch.Tensor:
    if type(width) is not int or type(rank) is not int or not 1 <= rank < width:
        raise ValueError("invalid Haar dimensions")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(width, rank, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.contiguous()


def _project(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return (rows @ basis) @ basis.T


def _polarized(
    x: torch.Tensor,
    y: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    decoder: torch.Tensor,
) -> torch.Tensor:
    products = 0.5 * (
        (x @ left.T) * (y @ right.T) + (y @ left.T) * (x @ right.T)
    )
    return products @ decoder.T


@torch.no_grad()
def estimate_mixed_block_leakage(
    left: torch.Tensor,
    right: torch.Tensor,
    decoder: torch.Tensor,
    basis: torch.Tensor,
    *,
    samples: int,
    seed: int,
    batch_size: int = 16,
) -> LeakageEstimate:
    """Estimate normalized direct-sum tensor error with Gaussian contractions."""
    _validate_factors(left, right, decoder, basis)
    if type(samples) is not int or samples < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("samples and batch_size must be positive integers")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    numerator = 0.0
    denominator = 0.0
    width = left.shape[1]
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        x = torch.randn(size, width, generator=generator, dtype=torch.float32)
        y = torch.randn(size, width, generator=generator, dtype=torch.float32)
        px, py = _project(x, basis), _project(y, basis)
        qx, qy = x - px, y - py
        full = _polarized(x, y, left, right, decoder)
        pp = _project(_polarized(px, py, left, right, decoder), basis)
        qq_raw = _polarized(qx, qy, left, right, decoder)
        qq = qq_raw - _project(qq_raw, basis)
        residual = full - pp - qq
        numerator += float(residual.double().square().sum())
        denominator += float(full.double().square().sum())
    if not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("quadratic tensor has no measurable energy")
    return LeakageEstimate(samples, numerator, denominator, numerator / denominator)

