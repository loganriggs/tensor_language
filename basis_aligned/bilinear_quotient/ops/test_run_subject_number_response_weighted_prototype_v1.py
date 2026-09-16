import numpy as np

import run_subject_number_response_weighted_prototype_v1 as run


def test_ridge_coefficients_matches_closed_form_and_is_finite():
    design = np.array([[1., 0.], [0., 2.], [1., 1.], [-1., 1.]])
    target = np.array([1., 2., 1.5, -.5])
    observed, ridge = run.ridge_coefficients(design, target)
    gram = design.T @ design
    expected = np.linalg.solve(gram + ridge * np.eye(2), design.T @ target)
    assert ridge > 0
    assert np.isfinite(observed).all()
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)


def test_stats_exact_replay():
    values = np.array([-2., -1., 1., 3.])
    report = run.stats(values, values)
    assert abs(report["cosine"] - 1.) <= 1e-15
    assert report["relative_l2_error"] == 0.
    assert report["sign_agreement"] == 1.
