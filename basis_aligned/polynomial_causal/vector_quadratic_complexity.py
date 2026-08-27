"""Certified lower bounds for vector-valued quadratic product programs.

For a quadratic map F: R^n -> R^m, the grammar is

    F(x) = bias + linear(x) + sum_i c_i (a_i @ x) (b_i @ x).

Only the quadratic tensor is priced here. One term costs one scalar
multiplication; its output vector ``c_i`` may be reused across every coordinate.
The exact minimum is a partially-symmetric tensor-rank problem, so this module
provides inexpensive, gauge-invariant lower bounds and an explicit-factor upper
bound rather than claiming to solve the generally hard minimization problem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


def tensor_from_product_factors(output: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return T with F_y(x) = sum_pq T[y,p,q] x[p] x[q]."""
    output = np.asarray(output, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if output.ndim != 2 or left.ndim != 2 or right.ndim != 2:
        raise ValueError("all factors must be matrices")
    if left.shape != right.shape or output.shape[1] != left.shape[0]:
        raise ValueError("expected output=(m,k), left=right=(k,n)")
    raw = np.einsum("yk,ki,kj->yij", output, left, right, optimize=True)
    return 0.5 * (raw + raw.swapaxes(-1, -2))


def evaluate_tensor(tensor: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a quadratic tensor on one vector or a batch of row vectors."""
    tensor = np.asarray(tensor, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if tensor.ndim != 3 or tensor.shape[1] != tensor.shape[2]:
        raise ValueError("tensor must have shape (m,n,n)")
    if x.shape[-1] != tensor.shape[-1]:
        raise ValueError("input dimension mismatch")
    return np.einsum("...i,yij,...j->...y", x, tensor, x, optimize=True)


def evaluate_product_factors(output: np.ndarray, left: np.ndarray, right: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return ((x @ np.asarray(left).T) * (x @ np.asarray(right).T)) @ np.asarray(output).T


def numerical_rank(matrix: np.ndarray, rtol: float | None = None) -> int:
    matrix = np.asarray(matrix, dtype=np.float64)
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0 or singular[0] == 0:
        return 0
    if rtol is None:
        rtol = max(matrix.shape) * np.finfo(np.float64).eps
    return int(np.count_nonzero(singular > singular[0] * rtol))


def output_flattening_rank(tensor: np.ndarray, rtol: float | None = None) -> int:
    """Rank of Y* -> Sym^2(X*), a lower bound on product count."""
    tensor = np.asarray(tensor, dtype=np.float64)
    return numerical_rank(tensor.reshape(tensor.shape[0], -1), rtol)


def input_flattening_rank(tensor: np.ndarray, rtol: float | None = None) -> int:
    """Rank of X -> Y tensor X*, whose half-rank lower-bounds products."""
    tensor = np.asarray(tensor, dtype=np.float64)
    matrix = tensor.transpose(1, 0, 2).reshape(tensor.shape[1], -1)
    return numerical_rank(matrix, rtol)


def scalar_inertia(matrix: np.ndarray, rtol: float | None = None) -> tuple[int, int, int]:
    """Return positive, negative, and numerical-zero inertia."""
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)
    eig = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eig), initial=0.0)), 1.0)
    if rtol is None:
        rtol = max(matrix.shape) * np.finfo(np.float64).eps
    tol = rtol * scale
    pos = int(np.count_nonzero(eig > tol))
    neg = int(np.count_nonzero(eig < -tol))
    return pos, neg, matrix.shape[0] - pos - neg


def contraction_inertia_bound(tensor: np.ndarray, output_directions: Iterable[np.ndarray], rtol: float | None = None) -> int:
    """Max scalar product complexity among supplied output contractions."""
    tensor = np.asarray(tensor, dtype=np.float64)
    best = 0
    for direction in output_directions:
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != (tensor.shape[0],):
            raise ValueError("output direction has wrong shape")
        pos, neg, _ = scalar_inertia(np.tensordot(direction, tensor, axes=(0, 0)), rtol)
        best = max(best, pos, neg)
    return best


@dataclass(frozen=True)
class ProductBounds:
    explicit_upper: int | None
    output_flattening_lower: int
    input_flattening_lower: int
    contraction_inertia_lower: int
    certified_lower: int


def product_bounds(
    tensor: np.ndarray,
    *,
    explicit_products: int | None = None,
    output_directions: Iterable[np.ndarray] = (),
    rtol: float | None = None,
) -> ProductBounds:
    """Compute three valid lower bounds and an optional factor-count upper bound."""
    out_rank = output_flattening_rank(tensor, rtol)
    in_rank = input_flattening_rank(tensor, rtol)
    in_lower = (in_rank + 1) // 2
    inertia = contraction_inertia_bound(tensor, output_directions, rtol)
    lower = max(out_rank, in_lower, inertia)
    if explicit_products is not None and explicit_products < lower:
        raise ValueError("explicit upper bound contradicts computed lower bound")
    return ProductBounds(explicit_products, out_rank, in_lower, inertia, lower)


def certificate_dict(bounds: ProductBounds) -> dict[str, int | None | str]:
    result = asdict(bounds)
    result["grammar"] = "sum_i c_i * (a_i dot x) * (b_i dot x)"
    result["status"] = "bounds_not_minimum" if bounds.explicit_upper != bounds.certified_lower else "minimum_certified"
    return result

