import numpy as np

import early_mlp_context_mobius_diagnostic as diagnostic


def test_product_poset_mobius_round_trip():
    rng = np.random.default_rng(7)
    coefficients = rng.normal(size=(8, 8))
    values = diagnostic.reconstruct(coefficients)
    recovered = diagnostic.mobius_coefficients(values)
    assert np.allclose(recovered, coefficients, atol=1e-12, rtol=1e-12)


def test_interaction_removes_main_effects():
    row = np.arange(8, dtype=np.float64)[:, None]
    column = np.square(np.arange(8, dtype=np.float64))[None, :]
    assert np.array_equal(diagnostic.interaction(row + column), np.zeros((8, 8)))


def test_omp_recovers_sparse_mobius_program():
    x, _ = diagnostic.design_matrix()
    truth = np.zeros(49)
    truth[[2, 17, 41]] = [1.5, -0.75, 2.0]
    support, recovered = diagnostic.omp(x, x @ truth, 3)
    assert set(support) == {2, 17, 41}
    assert np.allclose(recovered, truth, atol=1e-10, rtol=1e-10)


def test_leave_one_out_sparse_prediction_known_program():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(30, 8))
    truth = np.zeros(8)
    truth[[1, 6]] = [0.4, -1.2]
    result = diagnostic.leave_one_cell_out(x, x @ truth, omp_count=2)
    assert result["rmse"] < 1e-9
    assert result["nre_to_zero_interaction"] < 1e-9
