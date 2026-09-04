"""Exact compilation of a bilinear-MLP product-space projector.

This module is deliberately model-free.  It converts an already learned
orthonormal basis ``U`` into quadratic forms; it does not learn or validate the
causal meaning of that basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompiledProductProjector:
    """Factorized quadratic weights for ``W_D U U.T g(x)``.

    ``quadratic_forms[l]`` is the symmetric input-space matrix ``Q_l`` and
    ``output_directions[:, l]`` is ``W_D U[:, l]``.  Keeping this factorization
    avoids materializing one input-by-input matrix for every output coordinate.
    """

    basis: torch.Tensor
    quadratic_forms: torch.Tensor
    output_directions: torch.Tensor

    @property
    def projector(self) -> torch.Tensor:
        """Return the orthogonal product-space projector ``U U.T``."""
        return self.basis @ self.basis.transpose(-1, -2)

    def evaluate(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate the compiled selected output on ``state[..., input]``."""
        coordinates = torch.einsum(
            "...i,kij,...j->...k", state, self.quadratic_forms, state
        )
        return coordinates @ self.output_directions.transpose(-1, -2)

    def dense_quadratic_weights(self) -> torch.Tensor:
        """Return ``Q_out`` such that output[..., o] = x.T Q_out[o] x."""
        return torch.einsum(
            "ok,kij->oij", self.output_directions, self.quadratic_forms
        )


def _validate_matrix(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got shape {tuple(value.shape)}")
    if not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{name} must have a real floating dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _default_orthogonality_tolerances(basis: torch.Tensor) -> tuple[float, float]:
    epsilon = torch.finfo(basis.dtype).eps
    tolerance = 32.0 * epsilon
    return tolerance, tolerance


def _orthonormalize_span(
    basis: torch.Tensor,
    *,
    rank_tolerance: float | None,
) -> torch.Tensor:
    """Return an orthonormal basis for the column span, dropping null columns."""
    if basis.shape[1] == 0:
        return basis.clone()
    left_vectors, singular_values, _ = torch.linalg.svd(basis, full_matrices=False)
    if rank_tolerance is None:
        scale = float(singular_values.max()) if singular_values.numel() else 0.0
        rank_tolerance = max(basis.shape) * torch.finfo(basis.dtype).eps * scale
    if rank_tolerance < 0:
        raise ValueError("rank_tolerance must be nonnegative")
    retained = singular_values > rank_tolerance
    return left_vectors[:, retained]


def compile_product_projector(
    left: torch.Tensor,
    right: torch.Tensor,
    down: torch.Tensor,
    basis: torch.Tensor,
    *,
    normalize_basis: bool = False,
    orthogonality_atol: float | None = None,
    orthogonality_rtol: float | None = None,
    rank_tolerance: float | None = None,
) -> CompiledProductProjector:
    """Compile ``W_D U U.T ((W_L x) * (W_R x))`` into quadratic weights.

    Shapes are ``left,right: (product,input)``, ``down: (output,product)``,
    and ``basis: (product,rank)``.  By default, ``basis`` must already have
    orthonormal columns.  Setting ``normalize_basis=True`` instead compiles the
    orthogonal projector onto its column span; linearly dependent columns are
    removed using an SVD.
    """
    for name, value in (
        ("left", left),
        ("right", right),
        ("down", down),
        ("basis", basis),
    ):
        _validate_matrix(name, value)

    if left.shape != right.shape:
        raise ValueError("left and right must have the same (product, input) shape")
    product_dimension = left.shape[0]
    if down.shape[1] != product_dimension:
        raise ValueError("down's second dimension must equal the product dimension")
    if basis.shape[0] != product_dimension:
        raise ValueError("basis's first dimension must equal the product dimension")

    reference = left
    for name, value in (("right", right), ("down", down), ("basis", basis)):
        if value.dtype != reference.dtype:
            raise TypeError(f"{name} must have dtype {reference.dtype}")
        if value.device != reference.device:
            raise ValueError(f"{name} must be on device {reference.device}")

    default_atol, default_rtol = _default_orthogonality_tolerances(basis)
    atol = default_atol if orthogonality_atol is None else orthogonality_atol
    rtol = default_rtol if orthogonality_rtol is None else orthogonality_rtol
    if atol < 0 or rtol < 0:
        raise ValueError("orthogonality tolerances must be nonnegative")

    gram = basis.transpose(-1, -2) @ basis
    identity = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    is_orthonormal = torch.allclose(gram, identity, atol=atol, rtol=rtol)
    if not is_orthonormal:
        if not normalize_basis:
            max_error = float((gram - identity).abs().max())
            raise ValueError(
                "basis columns must be orthonormal; "
                f"maximum |U.T U - I| is {max_error:.6g}. "
                "Pass normalize_basis=True to compile the projector onto col(U)."
            )
        basis = _orthonormalize_span(basis, rank_tolerance=rank_tolerance)

    # For each basis vector u_l, the ordered form is
    # W_L.T diag(u_l) W_R.  Only its symmetric part contributes to x.T Q_l x.
    ordered_forms = torch.einsum("hi,hk,hj->kij", left, basis, right)
    quadratic_forms = 0.5 * (
        ordered_forms + ordered_forms.transpose(-1, -2)
    )
    output_directions = down @ basis
    return CompiledProductProjector(
        basis=basis,
        quadratic_forms=quadratic_forms,
        output_directions=output_directions,
    )
