import numpy as np

import equality_mixed_product_shared_response_rung478 as subject


def test_common_target_bisects_parent_directions():
    left = np.eye(3)
    right = np.eye(3)[:, ::-1]
    target, left_parent, right_parent = subject.common_target(left, right)
    assert np.isclose(np.linalg.norm(target), 1)
    assert np.allclose(left_parent, np.ones(3))
    assert np.allclose(right_parent, np.ones(3))


def test_signed_matching_pursuit_finds_two_term_target():
    matrix = np.eye(4)
    target = np.array([1.0, -1.0, 0.0, 0.0])
    fit = subject.matching_pursuit(matrix, target, "signed")
    assert fit["fit_passes"]
    assert set(fit["indices"]) == {0, 1}
    assert fit["fit_cosine"] > .999


def test_nonnegative_matching_pursuit_uses_positive_terms():
    matrix = np.eye(4)
    target = np.array([1.0, 1.0, 0.0, 0.0])
    fit = subject.matching_pursuit(matrix, target, "nonnegative")
    assert fit["fit_passes"]
    assert set(fit["indices"]) == {0, 1}
    assert np.min(fit["coefficients"]) >= 0


def test_scaled_error_is_scale_invariant():
    target = np.array([1.0, 2.0, 3.0])
    assert subject._scaled_error(target, target) < 1e-12
    assert subject._scaled_error(7 * target, target) < 1e-12
