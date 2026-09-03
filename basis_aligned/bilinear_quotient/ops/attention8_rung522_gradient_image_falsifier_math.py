"""Gauge-invariant CPU mathematics for the conditional post-rung-522 falsifier.

This module does not load a model or data and does not treat low rank as a circuit
claim.  It measures two different statements for a frozen projector ``P=QQ^T``:

1. whether downstream loss gradients themselves lie in ``im(P)``; and
2. whether the first-order effect along observed donor displacements is carried
   by ``P`` even when the complete gradient contains unrelated directions.

The second statement is weaker and is the one most directly connected to the
finite attention-write swaps in rung 522.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class GradientImageSummary:
    sample_count: int
    dimension: int
    rank: int
    gradient_energy: float
    gradient_inside_fraction: float
    full_tangent_rms: float
    projected_tangent_rms: float
    orthogonal_tangent_rms: float
    projected_to_full_signed_cosine: float
    projected_to_full_relative_residual: float
    projected_to_full_aligned_recovery: float
    excitation_singular_values: tuple[float, ...]
    excitation_relative_singular_values: tuple[float, ...]
    excitation_numerical_rank: int


def _finite_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.device.type != "cpu"
        or not value.is_floating_point()
        or value.shape[0] == 0
        or value.shape[1] == 0
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} must be a nonempty finite floating CPU matrix")
    return value.double()


def summarize_gradient_image(
    frame: torch.Tensor,
    gradients: torch.Tensor,
    donor_displacements: torch.Tensor,
    *,
    excitation_relative_tolerance: float = 1e-3,
) -> GradientImageSummary:
    """Summarize gradient containment and observed-direction causal transport.

    For row ``i``, let ``g_i`` be the downstream CE gradient with respect to the
    attention8 write and ``z_i`` a donor-minus-recipient write.  The complete
    first-order response is ``b_i=<g_i,z_i>``.  The response carried by the
    frozen projector is

    ``a_i=<Q^T g_i,Q^T z_i>=<g_i,QQ^T z_i>``.

    Every reported value is invariant to replacing ``Q`` by ``QR`` for an
    orthogonal within-frame change of coordinates.
    """
    q = _finite_matrix(frame, "frame")
    g = _finite_matrix(gradients, "gradients")
    z = _finite_matrix(donor_displacements, "donor_displacements")
    if g.shape != z.shape:
        raise ValueError("gradients and donor displacements must have equal shape")
    if q.shape[0] != g.shape[1] or q.shape[1] > q.shape[0]:
        raise ValueError("frame shape is incompatible with the gradient dimension")
    identity = torch.eye(q.shape[1], dtype=torch.float64)
    if float((q.mT @ q - identity).abs().amax()) > 1e-5:
        raise ValueError("frame columns must be orthonormal")
    if (
        not math.isfinite(excitation_relative_tolerance)
        or not 0 < excitation_relative_tolerance < 1
    ):
        raise ValueError("excitation_relative_tolerance must lie strictly in (0,1)")

    gq = g @ q
    zq = z @ q
    gradient_energy = float(g.square().sum())
    inside_energy = float(gq.square().sum())
    inside_fraction = inside_energy / max(gradient_energy, 1e-30)

    full = (g * z).sum(1)
    projected = (gq * zq).sum(1)
    orthogonal = full - projected
    full_norm2 = float(full @ full)
    projected_norm2 = float(projected @ projected)
    dot = float(projected @ full)
    cosine = dot / math.sqrt(max(projected_norm2 * full_norm2, 1e-30))
    residual = float(orthogonal.norm()) / math.sqrt(max(full_norm2, 1e-30))
    recovery = dot / max(full_norm2, 1e-30)

    # Divide by sqrt(n) so singular values have the RMS scale of an observed
    # coordinate. Numerical rank is relative to the strongest excited direction.
    singular = torch.linalg.svdvals(zq) / math.sqrt(g.shape[0])
    largest = float(singular[0]) if singular.numel() else 0.0
    relative = singular / max(largest, 1e-30)
    numerical_rank = int((relative >= excitation_relative_tolerance).sum())

    rms = lambda value: float(value.square().mean().sqrt())
    return GradientImageSummary(
        sample_count=g.shape[0],
        dimension=g.shape[1],
        rank=q.shape[1],
        gradient_energy=gradient_energy,
        gradient_inside_fraction=inside_fraction,
        full_tangent_rms=rms(full),
        projected_tangent_rms=rms(projected),
        orthogonal_tangent_rms=rms(orthogonal),
        projected_to_full_signed_cosine=cosine,
        projected_to_full_relative_residual=residual,
        projected_to_full_aligned_recovery=recovery,
        excitation_singular_values=tuple(float(value) for value in singular),
        excitation_relative_singular_values=tuple(float(value) for value in relative),
        excitation_numerical_rank=numerical_rank,
    )


__all__ = ["GradientImageSummary", "summarize_gradient_image"]
