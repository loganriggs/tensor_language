"""Gauge-quotiented Jacobian rank checks for small tensor programs.

The observable map of a factored program is polynomial in its parameters, but the
parameters themselves are not unique.  For example ``A @ B`` is unchanged by

    A -> A G,       B -> G^{-1} B.

Consequently raw parameter count and raw Hessian/Jacobian conditioning are not
structural simplicity measures.  The rank of the parameter-to-observable Jacobian is
the local dimension of the image at a regular point.  Comparing its nullspace with
the tangent space of the known gauge orbit distinguishes expected gauge freedom from
additional local non-identifiability.

This module is deliberately a CPU known-answer gate.  It does not certify a bilin18
decomposition.  Its next use is on a fitted CP/BTD or typed polynomial candidate,
where the observables must be physical folded coefficients or lawful causal-response
cells rather than a particular internal gauge.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def numerical_rank(matrix: np.ndarray, *, relative_tolerance: float = 1e-10) -> int:
    """Return SVD rank using a scale-relative, explicitly reported tolerance."""
    singular_values = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0
    return int(np.count_nonzero(singular_values > relative_tolerance * singular_values[0]))


def greedy_observable_basis(
    jacobian: np.ndarray, *, relative_tolerance: float = 1e-10
) -> list[int]:
    """Select physical output coordinates spanning the Jacobian's row space.

    Rows are chosen by greedy residual norm (pivoted Gram--Schmidt).  At a regular
    point these indices form a basis of the represented algebraic matroid: their
    differentials locally determine every other observable differential.  The basis
    need not be globally identifying and grouping/cost constraints are not handled by
    this toy routine.
    """
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    residuals = jacobian.copy()
    scale = float(np.max(np.linalg.norm(residuals, axis=1), initial=0.0))
    if scale == 0.0:
        return []
    selected: list[int] = []
    orthonormal_rows: list[np.ndarray] = []
    while True:
        norms = np.linalg.norm(residuals, axis=1)
        pivot = int(np.argmax(norms))
        if norms[pivot] <= relative_tolerance * scale:
            break
        direction = residuals[pivot] / norms[pivot]
        selected.append(pivot)
        orthonormal_rows.append(direction)
        residuals -= np.outer(residuals @ direction, direction)
        residuals[selected, :] = 0.0
    return selected


def matrix_product_jacobian(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Jacobian of ``vec(left @ right)`` in row-major parameter order."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("left and right must be compatible matrices")
    m, rank = left.shape
    _, n = right.shape
    columns: list[np.ndarray] = []
    for i in range(m):
        for q in range(rank):
            derivative = np.zeros((m, n), dtype=np.float64)
            derivative[i, :] = right[q, :]
            columns.append(derivative.ravel())
    for q in range(rank):
        for j in range(n):
            derivative = np.zeros((m, n), dtype=np.float64)
            derivative[:, j] = left[:, q]
            columns.append(derivative.ravel())
    return np.column_stack(columns)


def matrix_gauge_tangents(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Parameter tangents ``(left H, -H right)`` for a basis of hidden-edge gauges."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    rank = left.shape[1]
    tangents: list[np.ndarray] = []
    for a in range(rank):
        for b in range(rank):
            generator = np.zeros((rank, rank), dtype=np.float64)
            generator[a, b] = 1.0
            tangents.append(
                np.concatenate(((left @ generator).ravel(), (-generator @ right).ravel()))
            )
    return np.column_stack(tangents)


def cp_tensor(factor_a: np.ndarray, factor_b: np.ndarray, factor_c: np.ndarray) -> np.ndarray:
    """Materialize ``sum_q a_q outer b_q outer c_q`` for row-stacked factors."""
    return np.einsum("qi,qj,qk->ijk", factor_a, factor_b, factor_c)


def cp_jacobian(factor_a: np.ndarray, factor_b: np.ndarray, factor_c: np.ndarray) -> np.ndarray:
    """Jacobian of a third-order CP tensor in component-major parameter order."""
    factor_a = np.asarray(factor_a, dtype=np.float64)
    factor_b = np.asarray(factor_b, dtype=np.float64)
    factor_c = np.asarray(factor_c, dtype=np.float64)
    if not (factor_a.ndim == factor_b.ndim == factor_c.ndim == 2):
        raise ValueError("CP factors must be matrices")
    if not (factor_a.shape[0] == factor_b.shape[0] == factor_c.shape[0]):
        raise ValueError("CP factors must have the same component count")
    component_count = factor_a.shape[0]
    columns: list[np.ndarray] = []
    for q in range(component_count):
        for i in range(factor_a.shape[1]):
            basis = np.zeros(factor_a.shape[1], dtype=np.float64)
            basis[i] = 1.0
            columns.append(np.einsum("i,j,k->ijk", basis, factor_b[q], factor_c[q]).ravel())
        for j in range(factor_b.shape[1]):
            basis = np.zeros(factor_b.shape[1], dtype=np.float64)
            basis[j] = 1.0
            columns.append(np.einsum("i,j,k->ijk", factor_a[q], basis, factor_c[q]).ravel())
        for k in range(factor_c.shape[1]):
            basis = np.zeros(factor_c.shape[1], dtype=np.float64)
            basis[k] = 1.0
            columns.append(np.einsum("i,j,k->ijk", factor_a[q], factor_b[q], basis).ravel())
    return np.column_stack(columns)


def cp_scaling_tangents(
    factor_a: np.ndarray, factor_b: np.ndarray, factor_c: np.ndarray
) -> np.ndarray:
    """Two continuous scaling-gauge tangents per CP component."""
    factor_a = np.asarray(factor_a, dtype=np.float64)
    factor_b = np.asarray(factor_b, dtype=np.float64)
    factor_c = np.asarray(factor_c, dtype=np.float64)
    block = factor_a.shape[1] + factor_b.shape[1] + factor_c.shape[1]
    parameter_count = factor_a.shape[0] * block
    tangents: list[np.ndarray] = []
    for q in range(factor_a.shape[0]):
        first = np.zeros(parameter_count, dtype=np.float64)
        second = np.zeros(parameter_count, dtype=np.float64)
        start = q * block
        a_slice = slice(start, start + factor_a.shape[1])
        b_slice = slice(a_slice.stop, a_slice.stop + factor_b.shape[1])
        c_slice = slice(b_slice.stop, b_slice.stop + factor_c.shape[1])
        first[a_slice] = factor_a[q]
        first[c_slice] = -factor_c[q]
        second[b_slice] = factor_b[q]
        second[c_slice] = -factor_c[q]
        tangents.extend((first, second))
    return np.column_stack(tangents)


def run_known_answer_gate() -> dict[str, object]:
    """Exercise regular, regauged, and deliberately non-identifiable examples."""
    rng = np.random.default_rng(830)

    left = rng.normal(size=(7, 3))
    right = rng.normal(size=(3, 8))
    matrix_jacobian = matrix_product_jacobian(left, right)
    matrix_gauge = matrix_gauge_tangents(left, right)
    expected_matrix_rank = 3 * (7 + 8 - 3)

    factor_a = rng.normal(size=(3, 4))
    factor_b = rng.normal(size=(3, 5))
    factor_c = rng.normal(size=(3, 6))
    cp_jac = cp_jacobian(factor_a, factor_b, factor_c)
    cp_gauge = cp_scaling_tangents(factor_a, factor_b, factor_c)
    expected_cp_rank = 3 * (4 + 5 + 6 - 2)
    observable_basis = greedy_observable_basis(cp_jac)
    selected_jacobian = cp_jac[observable_basis, :]
    tangent = rng.normal(size=cp_jac.shape[1])
    all_differentials = cp_jac @ tangent
    reconstructed_tangent = np.linalg.pinv(selected_jacobian) @ all_differentials[
        observable_basis
    ]
    reconstructed_differentials = cp_jac @ reconstructed_tangent

    scale_a = np.exp(rng.normal(size=3))
    scale_b = np.exp(rng.normal(size=3))
    regauged_a = factor_a * scale_a[:, None]
    regauged_b = factor_b * scale_b[:, None]
    regauged_c = factor_c / (scale_a * scale_b)[:, None]
    regauged_jac = cp_jacobian(regauged_a, regauged_b, regauged_c)

    duplicate_a = factor_a.copy()
    duplicate_b = factor_b.copy()
    duplicate_c = factor_c.copy()
    duplicate_a[1] = duplicate_a[0]
    duplicate_b[1] = duplicate_b[0]
    duplicate_c[1] = duplicate_c[0]
    duplicate_jac = cp_jacobian(duplicate_a, duplicate_b, duplicate_c)

    return {
        "matrix_product": {
            "shape": [7, 3, 8],
            "raw_parameter_count": int(matrix_jacobian.shape[1]),
            "known_gauge_dimension": int(matrix_gauge.shape[1]),
            "expected_quotient_dimension": expected_matrix_rank,
            "measured_jacobian_rank": numerical_rank(matrix_jacobian),
            "jacobian_nullity": int(matrix_jacobian.shape[1] - numerical_rank(matrix_jacobian)),
            "max_abs_gauge_image": float(np.max(np.abs(matrix_jacobian @ matrix_gauge))),
        },
        "regular_cp": {
            "shape": [3, 4, 5, 6],
            "raw_parameter_count": int(cp_jac.shape[1]),
            "known_continuous_gauge_dimension": int(cp_gauge.shape[1]),
            "expected_quotient_dimension": expected_cp_rank,
            "measured_jacobian_rank": numerical_rank(cp_jac),
            "jacobian_nullity": int(cp_jac.shape[1] - numerical_rank(cp_jac)),
            "max_abs_gauge_image": float(np.max(np.abs(cp_jac @ cp_gauge))),
        },
        "algebraic_matroid_basis": {
            "all_observable_coordinates": int(cp_jac.shape[0]),
            "selected_coordinates": len(observable_basis),
            "selected_row_rank": numerical_rank(selected_jacobian),
            "relative_first_order_reconstruction_error": float(
                np.linalg.norm(reconstructed_differentials - all_differentials)
                / np.linalg.norm(all_differentials)
            ),
            "interpretation": (
                "At this regular CP point, 39 selected physical tensor entries span "
                "the differentials of all 120 entries. This is local and first-order, "
                "not a global identifiability certificate."
            ),
        },
        "regauge_control": {
            "max_abs_tensor_change": float(
                np.max(
                    np.abs(
                        cp_tensor(factor_a, factor_b, factor_c)
                        - cp_tensor(regauged_a, regauged_b, regauged_c)
                    )
                )
            ),
            "rank_before": numerical_rank(cp_jac),
            "rank_after": numerical_rank(regauged_jac),
            "jacobian_condition_before": float(np.linalg.cond(cp_jac)),
            "jacobian_condition_after": float(np.linalg.cond(regauged_jac)),
        },
        "duplicate_component_failure_control": {
            "regular_rank": numerical_rank(cp_jac),
            "duplicate_rank": numerical_rank(duplicate_jac),
            "rank_deficit": int(numerical_rank(cp_jac) - numerical_rank(duplicate_jac)),
            "interpretation": (
                "Coincident CP components are a singular, non-identifiable point; the "
                "Jacobian loses directions beyond the ordinary per-component scaling gauge."
            ),
        },
    }


def build_receipt() -> dict[str, object]:
    started = time.monotonic()
    result = run_known_answer_gate()
    return {
        "schema": "quotient_jacobian_minimality_toy_v1",
        "claim_boundary": (
            "CPU algebraic known-answer only. This verifies gauge-null and rank "
            "accounting for matrix-product and CP programs; it does not establish "
            "identifiability, semantic meaning, causal sufficiency, or minimality of "
            "any bilin18 replacement."
        ),
        "relative_rank_tolerance": 1e-10,
        "results": result,
        "runtime_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "quotient_jacobian_minimality_toy_receipt.json",
    )
    args = parser.parse_args()
    receipt = build_receipt()
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
