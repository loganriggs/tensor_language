import numpy as np

from quotient_jacobian_minimality import (
    cp_jacobian,
    cp_scaling_tangents,
    cp_tensor,
    greedy_observable_basis,
    matrix_gauge_tangents,
    matrix_product_jacobian,
    numerical_rank,
    run_known_answer_gate,
)


def test_matrix_product_rank_is_raw_count_minus_gl_gauge() -> None:
    rng = np.random.default_rng(1)
    left = rng.normal(size=(7, 3))
    right = rng.normal(size=(3, 8))
    jacobian = matrix_product_jacobian(left, right)
    gauge = matrix_gauge_tangents(left, right)
    assert numerical_rank(jacobian) == 3 * (7 + 8 - 3)
    assert jacobian.shape[1] - numerical_rank(jacobian) == 3**2
    np.testing.assert_allclose(jacobian @ gauge, 0.0, atol=1e-12, rtol=0.0)


def test_regular_cp_rank_is_raw_count_minus_scaling_gauge() -> None:
    rng = np.random.default_rng(2)
    factors = (
        rng.normal(size=(3, 4)),
        rng.normal(size=(3, 5)),
        rng.normal(size=(3, 6)),
    )
    jacobian = cp_jacobian(*factors)
    gauge = cp_scaling_tangents(*factors)
    assert numerical_rank(jacobian) == 3 * (4 + 5 + 6 - 2)
    assert jacobian.shape[1] - numerical_rank(jacobian) == 2 * 3
    np.testing.assert_allclose(jacobian @ gauge, 0.0, atol=1e-12, rtol=0.0)


def test_cp_rank_and_tensor_are_invariant_to_scaling_gauge() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3, 4))
    b = rng.normal(size=(3, 5))
    c = rng.normal(size=(3, 6))
    scale_a = np.exp(rng.normal(size=3))
    scale_b = np.exp(rng.normal(size=3))
    regauged = (
        a * scale_a[:, None],
        b * scale_b[:, None],
        c / (scale_a * scale_b)[:, None],
    )
    np.testing.assert_allclose(cp_tensor(a, b, c), cp_tensor(*regauged), atol=1e-12)
    assert numerical_rank(cp_jacobian(a, b, c)) == numerical_rank(cp_jacobian(*regauged))


def test_duplicate_cp_component_is_detected_as_singular() -> None:
    result = run_known_answer_gate()
    failure = result["duplicate_component_failure_control"]
    assert failure["duplicate_rank"] < failure["regular_rank"]
    assert failure["rank_deficit"] >= 1


def test_observable_basis_spans_every_local_cp_differential() -> None:
    rng = np.random.default_rng(4)
    factors = (
        rng.normal(size=(3, 4)),
        rng.normal(size=(3, 5)),
        rng.normal(size=(3, 6)),
    )
    jacobian = cp_jacobian(*factors)
    selected = greedy_observable_basis(jacobian)
    selected_jacobian = jacobian[selected, :]
    assert len(selected) == numerical_rank(jacobian) == 39
    assert numerical_rank(selected_jacobian) == 39
    tangent = rng.normal(size=jacobian.shape[1])
    outputs = jacobian @ tangent
    recovered = jacobian @ (np.linalg.pinv(selected_jacobian) @ outputs[selected])
    np.testing.assert_allclose(recovered, outputs, atol=1e-10, rtol=1e-10)
