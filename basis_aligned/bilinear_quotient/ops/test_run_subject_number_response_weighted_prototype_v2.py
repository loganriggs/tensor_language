import numpy as np

import run_subject_number_response_weighted_prototype_v2 as run


def test_corrected_ridge_is_one_hundred_times_v1():
    design = np.array([[1., 0.], [0., 2.], [1., 1.], [-1., 1.]])
    target = np.array([1., 2., 1.5, -.5])
    _, corrected = run.ridge_coefficients(design, target)
    _, original = run.base.ridge_coefficients(design, target, ridge_fraction=1e-2)
    np.testing.assert_allclose(corrected, 100 * original, rtol=0, atol=1e-15)
