import pytest

from two_by_two_dependency_contract import (
    DependencyFactorialError, decompose_dependency_factorial, vector_metrics,
)


def test_additive_cells_have_zero_interaction_and_exact_conditionals():
    report = decompose_dependency_factorial({
        "00": [0.0, 0.0], "01": [1.0, 2.0],
        "10": [3.0, 4.0], "11": [4.0, 6.0],
    })
    assert report["components"]["upstream_live_attention"] == [3.0, 4.0]
    assert report["components"]["upstream_with_fixed_attention15"] == [3.0, 4.0]
    assert report["components"]["mobius_interaction"] == [0.0, 0.0]
    assert report["base_zero_pass"] and report["factorial_closure_pass"]


def test_interaction_is_the_change_in_either_conditional_effect():
    report = decompose_dependency_factorial({
        "00": [0.0], "01": [2.0], "10": [3.0], "11": [8.0],
    })
    assert report["components"]["mobius_interaction"] == [3.0]
    assert report["components"]["upstream_with_fixed_attention15"] == [6.0]
    assert report["components"]["upstream_live_attention"] == [3.0]


def test_vector_metrics_match_registered_projection_convention():
    report = vector_metrics([1.0, -2.0], [2.0, -4.0])
    assert report["signed_projection"] == pytest.approx(0.5)
    assert report["cosine"] == pytest.approx(1.0)
    assert report["direction_fraction"] == 1.0


def test_malformed_or_nonlinear_cells_fail_closed():
    with pytest.raises(DependencyFactorialError):
        decompose_dependency_factorial({"00": [0.0], "01": [1.0], "10": [1.0]})
    with pytest.raises(DependencyFactorialError):
        decompose_dependency_factorial({
            "00": [0.0], "01": [1.0, 2.0], "10": [1.0], "11": [2.0],
        })
    with pytest.raises(DependencyFactorialError):
        decompose_dependency_factorial({
            "00": [0.0], "01": [1.0], "10": [float("nan")], "11": [2.0],
        })
    with pytest.raises(DependencyFactorialError):
        vector_metrics([1.0], [0.0])
