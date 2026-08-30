import json
from pathlib import Path

import numpy as np

from sparse_mobius_interaction_tomography import (
    analyze_mlp0_cube,
    mobius_transform,
    run_dense_failure_control,
    run_planted_sparse_gate,
    zeta_transform,
)


BASE = Path(__file__).resolve().parent


def test_exact_mobius_round_trip() -> None:
    rng = np.random.default_rng(5)
    coefficients = rng.normal(size=32)
    values = zeta_transform(coefficients, 5)
    recovered = mobius_transform(values, 5)
    np.testing.assert_allclose(recovered, coefficients, atol=1e-12, rtol=1e-12)


def test_planted_sparse_partial_cube_gate() -> None:
    result = run_planted_sparse_gate()
    assert result["partial_queries"] < result["full_cube_queries"] / 20
    assert result["support_terms_recovered"] == result["true_nonzero_terms"]
    assert result["false_support_terms"] == 0
    assert result["max_abs_coefficient_error"] < 1e-10
    assert result["holdout_rmse"] < 1e-10


def test_dense_failure_control_rejects_sparse_story() -> None:
    result = run_dense_failure_control()
    assert result["nonzero_truth_terms"] > 200
    assert result["holdout_normalized_rmse"] > 0.2


def test_mlp0_known_answer_and_degree_two_residual() -> None:
    result = analyze_mlp0_cube(
        BASE / "mlp0_token_context_tensor_factorial_discovery.json"
    )
    for role in ("FIT", "SELECT"):
        assert result[role]["max_abs_error_vs_registered"] < 1e-12
        assert np.isclose(
            result[role]["degree_two_max_abs_prediction_error_nat"],
            result[role]["third_order_abs_nat"],
        )
        assert result[role]["third_order_abs_nat"] < 0.03
        assert result[role]["largest_pair_abs_nat"] > 1.0
