"""Ground-truth toys for exact minimum causal intervention subspaces.

For allowed changes spanning V and stacked linear readers R, an orthogonal
projector P preserves every allowed downstream response when R P V = R V.
Therefore rank(P) >= rank(R V).  Projecting onto V times the row space of R V
attains the bound, providing an exact minimum-rank certificate.
"""

# BQGATE: LIBRARY
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ToyScenario:
    name: str
    difficulty: int
    deltas: np.ndarray
    readers: dict[str, np.ndarray]
    description: str


def orthogonal_basis(matrix: np.ndarray, *, tolerance: float = 1e-10) -> np.ndarray:
    """Return an orthonormal basis for the column span of a 2-D matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if not matrix.size:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64)
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular > tolerance * max(1.0, float(singular[0]))))
    return left[:, :rank]


def stack_readers(readers: dict[str, np.ndarray], environments=None) -> np.ndarray:
    names = sorted(readers) if environments is None else list(environments)
    if not names:
        raise ValueError("at least one reader environment is required")
    return np.vstack([np.atleast_2d(np.asarray(readers[name], dtype=np.float64)) for name in names])


def exact_minimum_subspace(
    deltas: np.ndarray,
    readers: dict[str, np.ndarray],
    *,
    environments=None,
) -> tuple[np.ndarray, dict]:
    """Return the canonical exact minimum subspace and its rank certificate."""
    deltas = np.asarray(deltas, dtype=np.float64)
    change_basis = orthogonal_basis(deltas.T)
    reader = stack_readers(readers, environments)
    restricted = reader @ change_basis
    restricted_row_basis = orthogonal_basis(restricted.T)
    basis = change_basis @ restricted_row_basis
    certificate = {
        "allowed_change_rank": int(change_basis.shape[1]),
        "minimum_rank_lower_bound": int(np.linalg.matrix_rank(restricted, tol=1e-10)),
        "attained_rank": int(basis.shape[1]),
    }
    return basis, certificate


def difference_in_means_subspace(deltas: np.ndarray) -> np.ndarray:
    """Rank-one DIM analogue: the normalized mean intervention change."""
    mean = np.asarray(deltas, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12:
        return np.zeros((mean.size, 0), dtype=np.float64)
    return (mean / norm)[:, None]


def response_regression_subspace(
    deltas: np.ndarray,
    responses: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Recover the minimum input subspace from paired changes and responses.

    This estimator sees no reader weights. It least-squares fits the linear
    change-to-response map and returns the row space of that fitted map.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)
    if deltas.shape[0] != responses.shape[0]:
        raise ValueError("deltas and responses must have the same number of rows")
    fitted_map, _, _, _ = np.linalg.lstsq(deltas, responses, rcond=tolerance)
    return orthogonal_basis(fitted_map, tolerance=tolerance)


def contextual_response_regression_subspace(
    deltas: np.ndarray,
    context_gate: np.ndarray,
    responses: np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Recover base and context-interaction readers from y = xB + g*xC.

    The returned activation subspace is the union of the fitted base and
    interaction input row spaces. This is the smallest model class in the toy
    ladder that can represent a subset-conditional reader.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    gate = np.asarray(context_gate, dtype=np.float64).reshape(-1, 1)
    responses = np.asarray(responses, dtype=np.float64)
    if deltas.shape[0] != gate.shape[0] or deltas.shape[0] != responses.shape[0]:
        raise ValueError("deltas, context_gate, and responses must align by row")
    design = np.hstack([deltas, gate * deltas])
    fitted, _, _, _ = np.linalg.lstsq(design, responses, rcond=tolerance)
    dimension = deltas.shape[1]
    coefficient_union = np.hstack([fitted[:dimension], fitted[dimension:]])
    return orthogonal_basis(coefficient_union, tolerance=tolerance)


def preservation_error(
    deltas: np.ndarray,
    readers: dict[str, np.ndarray],
    basis: np.ndarray,
    *,
    environments=None,
) -> float:
    """Maximum absolute response error over all allowed changes and readers."""
    deltas = np.asarray(deltas, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    projector = basis @ basis.T
    reader = stack_readers(readers, environments)
    error = reader @ (np.eye(deltas.shape[1]) - projector) @ deltas.T
    return float(np.max(np.abs(error)))


def projector_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Frobenius distance between two subspace projectors."""
    return float(np.linalg.norm(first @ first.T - second @ second.T, ord="fro"))


def _rotation(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def _changes(directions: np.ndarray, nuisance: np.ndarray | None = None) -> np.ndarray:
    """Positive-mean first factor plus centered variation spanning every factor."""
    rank = directions.shape[1]
    rows = [directions[:, 0]]
    for index in range(rank):
        rows.extend([directions[:, 0] + directions[:, index], directions[:, 0] - directions[:, index]])
    if nuisance is not None:
        for index in range(nuisance.shape[1]):
            rows.extend([directions[:, 0] + nuisance[:, index], directions[:, 0] - nuisance[:, index]])
    return np.asarray(rows)


def make_scenarios() -> dict[str, ToyScenario]:
    """Construct a deterministic, progressively harder four-rung ladder."""
    dimension = 12
    axis = np.eye(dimension)
    rotated = _rotation(dimension, 1729)

    axis_deltas = _changes(axis[:, :1], axis[:, 5:7])
    rotated_directions = rotated[:, :3]
    rotated_deltas = _changes(rotated_directions, rotated[:, 7:9])
    shared_directions = rotated[:, 3:6]
    shared_deltas = _changes(shared_directions, rotated[:, 9:11])
    subset_directions = rotated[:, 6:8]
    subset_deltas = _changes(subset_directions, rotated[:, 10:12])

    return {
        "axis_aligned_rank1": ToyScenario(
            "axis_aligned_rank1",
            1,
            axis_deltas,
            {"all": axis[:, 0][None, :]},
            "Basis-aligned rank-one cause with two unread nuisance directions.",
        ),
        "rotated_rank3": ToyScenario(
            "rotated_rank3",
            2,
            rotated_deltas,
            {"all": rotated_directions.T},
            "Non-basis-aligned rank-three cause; DIM sees only the first mean direction.",
        ),
        "shared_plus_private": ToyScenario(
            "shared_plus_private",
            3,
            shared_deltas,
            {
                "task_a": shared_directions[:, [0, 1]].T,
                "task_b": shared_directions[:, [0, 2]].T,
            },
            "Two tasks share one direction and each has one private direction.",
        ),
        "subset_gated": ToyScenario(
            "subset_gated",
            4,
            subset_deltas,
            {
                "common_rows": subset_directions[:, 0][None, :],
                "rare_subset": subset_directions.T,
            },
            "A second factor is read only on a rare subset; common-only fitting memorizes rank one.",
        ),
    }


def _coefficient_grid() -> np.ndarray:
    return np.asarray([
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0],
    ])


def evaluate_environment_confounding() -> dict:
    """A nuisance is unidentifiable in one environment and canceled by augmentation."""
    rotation = _rotation(12, 2718)
    causal = rotation[:, :2]
    nuisance = rotation[:, 2]
    coefficients = _coefficient_grid()
    environment_a = (
        coefficients[:, :1] * causal[:, 0]
        + coefficients[:, 1:] * (causal[:, 1] + nuisance)
    )
    environment_b = (
        coefficients[:, :1] * causal[:, 0]
        + coefficients[:, 1:] * (causal[:, 1] - nuisance)
    )
    response = coefficients.copy()
    readers = {"true": causal.T}
    allowed = np.vstack([environment_a, environment_b])
    exact, certificate = exact_minimum_subspace(allowed, readers)
    one_environment = response_regression_subspace(environment_a, response)
    augmented = response_regression_subspace(
        allowed, np.vstack([response, response]))
    return {
        "description": (
            "One environment aliases causal factor two with a nuisance; a sign-reversed "
            "augmentation makes the two directions identifiable."
        ),
        "exact_certificate": certificate,
        "one_environment_rank": int(one_environment.shape[1]),
        "one_environment_training_error": preservation_error(
            environment_a, readers, one_environment),
        "one_environment_full_error": preservation_error(allowed, readers, one_environment),
        "one_environment_projector_distance": projector_distance(one_environment, exact),
        "augmented_rank": int(augmented.shape[1]),
        "augmented_full_error": preservation_error(allowed, readers, augmented),
        "augmented_projector_distance": projector_distance(augmented, exact),
    }


def evaluate_bilinear_context_gate() -> dict:
    """A scalar context gate activates a second, rotated causal direction."""
    rotation = _rotation(12, 3141)
    causal = rotation[:, :2]
    nuisance = rotation[:, 8:10]
    allowed = _changes(causal, nuisance)
    repeats = 8
    deltas = np.repeat(allowed, repeats, axis=0)
    gate = np.tile(np.asarray([0.0] * (repeats - 1) + [1.0]), allowed.shape[0])
    responses = (
        deltas @ causal[:, 0]
        + gate * (deltas @ causal[:, 1])
    )[:, None]
    readers = {
        "common_context": causal[:, 0][None, :],
        "rare_context": (causal[:, 0] + causal[:, 1])[None, :],
    }
    exact, certificate = exact_minimum_subspace(allowed, readers)
    pooled = response_regression_subspace(deltas, responses)
    contextual = contextual_response_regression_subspace(deltas, gate, responses)
    return {
        "description": (
            "A rare one-in-eight context activates a second reader through a bilinear gate."
        ),
        "rare_fraction": float(gate.mean()),
        "exact_certificate": certificate,
        "pooled_linear_rank": int(pooled.shape[1]),
        "pooled_linear_full_error": preservation_error(allowed, readers, pooled),
        "pooled_linear_projector_distance": projector_distance(pooled, exact),
        "contextual_bilinear_rank": int(contextual.shape[1]),
        "contextual_bilinear_full_error": preservation_error(allowed, readers, contextual),
        "contextual_bilinear_projector_distance": projector_distance(contextual, exact),
    }


def evaluate_ladder() -> dict:
    results = {}
    for name, scenario in make_scenarios().items():
        exact, certificate = exact_minimum_subspace(scenario.deltas, scenario.readers)
        dim = difference_in_means_subspace(scenario.deltas)
        reader = stack_readers(scenario.readers)
        responses = scenario.deltas @ reader.T
        regression = response_regression_subspace(scenario.deltas, responses)
        report = {
            "difficulty": scenario.difficulty,
            "description": scenario.description,
            "exact_certificate": certificate,
            "exact_preservation_error": preservation_error(scenario.deltas, scenario.readers, exact),
            "dim_rank": int(dim.shape[1]),
            "dim_preservation_error": preservation_error(scenario.deltas, scenario.readers, dim),
            "response_regression_rank": int(regression.shape[1]),
            "response_regression_preservation_error": preservation_error(
                scenario.deltas, scenario.readers, regression),
            "response_regression_projector_distance": projector_distance(regression, exact),
        }
        if name == "shared_plus_private":
            task_a, cert_a = exact_minimum_subspace(
                scenario.deltas, scenario.readers, environments=["task_a"])
            task_b, cert_b = exact_minimum_subspace(
                scenario.deltas, scenario.readers, environments=["task_b"])
            overlap = np.linalg.svd(task_a.T @ task_b, compute_uv=False)
            report["task_a_rank"] = cert_a["attained_rank"]
            report["task_b_rank"] = cert_b["attained_rank"]
            report["shared_dimension"] = int(np.sum(overlap > 1.0 - 1e-8))
        if name == "subset_gated":
            common, common_certificate = exact_minimum_subspace(
                scenario.deltas, scenario.readers, environments=["common_rows"])
            report["common_only_rank"] = common_certificate["attained_rank"]
            report["common_only_full_test_error"] = preservation_error(
                scenario.deltas, scenario.readers, common)
        results[name] = report
    results["environment_confounding"] = {
        "difficulty": 5,
        **evaluate_environment_confounding(),
    }
    results["bilinear_context_gate"] = {
        "difficulty": 6,
        **evaluate_bilinear_context_gate(),
    }
    return {
        "schema": "causal_subspace_ground_truth_toys_v1",
        "criterion": "universal_linear_response_preservation",
        "scenarios": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = evaluate_ladder()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
