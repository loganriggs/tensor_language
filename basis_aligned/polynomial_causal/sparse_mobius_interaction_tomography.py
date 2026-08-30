"""Small, auditable tools for causal interaction tomography on subset interventions.

For a set function f(S), the coefficients returned by ``mobius_transform`` satisfy

    f(S) = sum_{T subseteq S} m(T).

Thus m({i,j}) is the part of the joint intervention effect not explained by either
main effect, and higher-order coefficients are the corresponding inclusion/exclusion
interactions.  Masks are integers; bit i denotes component i.

The OMP routine is deliberately only a diagnostic for sparse low-degree structure.
Subset-containment (AND) features are highly correlated, so successful recovery on a
toy is not a theorem that arbitrary transformer interactions can be recovered this
way.  Exact full-cube inversion remains the reference implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def masks_up_to_degree(n_components: int, max_degree: int) -> list[int]:
    """Return subset masks of size at most ``max_degree`` in integer order."""
    if not 0 <= max_degree <= n_components:
        raise ValueError("max_degree must lie in [0, n_components]")
    return [m for m in range(1 << n_components) if m.bit_count() <= max_degree]


def zeta_design(query_masks: list[int], term_masks: list[int]) -> np.ndarray:
    """Design X[q,t] = 1 when interaction term t is contained in query q."""
    return np.asarray(
        [[float((query & term) == term) for term in term_masks] for query in query_masks],
        dtype=np.float64,
    )


def mobius_transform(full_values: np.ndarray, n_components: int) -> np.ndarray:
    """Exact Boolean-lattice Möbius inversion in O(n 2**n) operations."""
    values = np.asarray(full_values, dtype=np.float64)
    if values.shape != (1 << n_components,):
        raise ValueError("full_values must contain one value for every subset mask")
    coefficients = values.copy()
    for bit in range(n_components):
        for mask in range(1 << n_components):
            if mask & (1 << bit):
                coefficients[mask] -= coefficients[mask ^ (1 << bit)]
    return coefficients


def zeta_transform(coefficients: np.ndarray, n_components: int) -> np.ndarray:
    """Reconstruct all subset values from full-cube Möbius coefficients."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (1 << n_components,):
        raise ValueError("coefficients must contain one value for every subset mask")
    values = coefficients.copy()
    for bit in range(n_components):
        for mask in range(1 << n_components):
            if mask & (1 << bit):
                values[mask] += values[mask ^ (1 << bit)]
    return values


def omp_decode(
    design: np.ndarray,
    outcomes: np.ndarray,
    sparsity: int,
    *,
    intercept_column: int | None = 0,
) -> np.ndarray:
    """Orthogonal matching pursuit with normalized residual correlations.

    This is a transparent baseline, not a guaranteed sparse-Möbius decoder.  The
    intercept is selected first because the empty-set coefficient is generally large.
    """
    design = np.asarray(design, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    if design.ndim != 2 or outcomes.shape != (design.shape[0],):
        raise ValueError("incompatible design/outcome shapes")
    if not 1 <= sparsity <= design.shape[1]:
        raise ValueError("invalid sparsity")

    selected: list[int] = []
    if intercept_column is not None:
        selected.append(intercept_column)
    norms = np.linalg.norm(design, axis=0)

    while len(selected) < sparsity:
        if selected:
            fitted = np.linalg.lstsq(design[:, selected], outcomes, rcond=None)[0]
            residual = outcomes - design[:, selected] @ fitted
        else:
            residual = outcomes
        scores = np.abs(design.T @ residual) / np.maximum(norms, 1e-15)
        scores[selected] = -np.inf
        selected.append(int(np.argmax(scores)))

    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[selected] = np.linalg.lstsq(
        design[:, selected], outcomes, rcond=None
    )[0]
    return coefficients


def _mask_labels(names: list[str]) -> dict[int, str]:
    labels = {0: "EMPTY"}
    for mask in range(1, 1 << len(names)):
        labels[mask] = "+".join(
            name for i, name in enumerate(names) if mask & (1 << i)
        )
    return labels


def analyze_mlp0_cube(path: Path) -> dict[str, object]:
    """Run the exact known-answer gate on the completed TT/X/CC factorial."""
    artifact = json.loads(path.read_text())
    names = ["TT", "X", "CC"]
    labels = _mask_labels(names)
    role_results: dict[str, object] = {}
    for role in ("FIT", "SELECT"):
        pooled_ce = artifact["roles"][role]["pooled_ce"]
        values = np.asarray([-pooled_ce[labels[m]] for m in range(8)])
        coefficients = mobius_transform(values, 3)
        registered = artifact["roles"][role]["mobius_dividends_of_negative_ce"]
        max_registered_error = max(
            abs(coefficients[m] - registered[labels[m]]) for m in range(8)
        )
        degree_two = coefficients.copy()
        degree_two[7] = 0.0
        degree_two_predictions = zeta_transform(degree_two, 3)
        role_results[role] = {
            "coefficients": {
                labels[m]: float(coefficients[m]) for m in range(8)
            },
            "max_abs_error_vs_registered": float(max_registered_error),
            "degree_two_max_abs_prediction_error_nat": float(
                np.max(np.abs(degree_two_predictions - values))
            ),
            "third_order_abs_nat": float(abs(coefficients[7])),
            "largest_pair_abs_nat": float(
                max(abs(coefficients[m]) for m in (3, 5, 6))
            ),
        }
    return role_results


def run_planted_sparse_gate() -> dict[str, object]:
    """Recover a fixed sparse degree-3 interaction graph from a partial cube."""
    n_components, max_degree, sparsity = 12, 3, 8
    term_masks = masks_up_to_degree(n_components, max_degree)
    rng = np.random.default_rng(29)
    support_indices = [0] + list(
        rng.choice(np.arange(1, len(term_masks)), sparsity - 1, replace=False)
    )
    truth = np.zeros(len(term_masks), dtype=np.float64)
    truth[support_indices] = rng.normal(size=sparsity)

    query_rng = np.random.default_rng(0)
    query_masks = list(
        dict.fromkeys(
            [0]
            + [1 << i for i in range(n_components)]
            + list(map(int, query_rng.integers(0, 1 << n_components, 192)))
        )
    )
    design = zeta_design(query_masks, term_masks)
    recovered = omp_decode(design, design @ truth, sparsity)
    recovered_support = set(np.flatnonzero(np.abs(recovered) > 1e-10))

    holdout_rng = np.random.default_rng(7)
    holdout_masks = list(map(int, holdout_rng.integers(0, 1 << n_components, 512)))
    holdout_design = zeta_design(holdout_masks, term_masks)
    holdout_rmse = np.sqrt(np.mean((holdout_design @ (recovered - truth)) ** 2))

    return {
        "n_components": n_components,
        "candidate_terms_degree_at_most_3": len(term_masks),
        "full_cube_queries": 1 << n_components,
        "partial_queries": len(query_masks),
        "true_nonzero_terms": sparsity,
        "support_terms_recovered": len(recovered_support & set(support_indices)),
        "false_support_terms": len(recovered_support - set(support_indices)),
        "max_abs_coefficient_error": float(np.max(np.abs(recovered - truth))),
        "holdout_rmse": float(holdout_rmse),
    }


def run_dense_failure_control() -> dict[str, object]:
    """Show that an equal-budget sparse decoder fails on a dense interaction null."""
    n_components, max_degree, fitted_sparsity = 12, 3, 8
    term_masks = masks_up_to_degree(n_components, max_degree)
    rng = np.random.default_rng(101)
    truth = rng.normal(size=len(term_masks)) / np.sqrt(len(term_masks))
    query_masks = list(
        dict.fromkeys(
            [0]
            + [1 << i for i in range(n_components)]
            + list(map(int, rng.integers(0, 1 << n_components, 192)))
        )
    )
    design = zeta_design(query_masks, term_masks)
    recovered = omp_decode(design, design @ truth, fitted_sparsity)
    holdout_masks = list(map(int, rng.integers(0, 1 << n_components, 512)))
    holdout_design = zeta_design(holdout_masks, term_masks)
    target = holdout_design @ truth
    residual = holdout_design @ recovered - target
    return {
        "nonzero_truth_terms": int(np.count_nonzero(truth)),
        "fitted_terms": fitted_sparsity,
        "holdout_rmse": float(np.sqrt(np.mean(residual**2))),
        "holdout_normalized_rmse": float(
            np.sqrt(np.mean(residual**2)) / np.sqrt(np.mean(target**2))
        ),
    }


def build_receipt(mlp0_artifact: Path) -> dict[str, object]:
    started = time.monotonic()
    return {
        "schema": "sparse_mobius_interaction_tomography_toy_v1",
        "claim_boundary": (
            "CPU toy and exact reanalysis of an already-opened MLP0 factorial only; "
            "no model, rows, protected outcomes, circuit promotion, or real 10-circuit "
            "query-efficiency claim. OMP has no guarantee under coherent AND features."
        ),
        "mlp0_artifact": str(mlp0_artifact),
        "mlp0_artifact_sha256": hashlib.sha256(mlp0_artifact.read_bytes()).hexdigest(),
        "mlp0_known_answer": analyze_mlp0_cube(mlp0_artifact),
        "planted_sparse_gate": run_planted_sparse_gate(),
        "dense_failure_control": run_dense_failure_control(),
        "runtime_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parent
    parser.add_argument(
        "--mlp0-artifact",
        type=Path,
        default=base / "mlp0_token_context_tensor_factorial_discovery.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base / "sparse_mobius_interaction_tomography_toy_receipt.json",
    )
    args = parser.parse_args()
    receipt = build_receipt(args.mlp0_artifact.resolve())
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
