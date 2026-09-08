import numpy as np

import run_temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1 as experiment


def test_shapley_exactly_recovers_additive_four_route_game():
    weights = np.array([.1, .2, .3, .4])
    values = np.array([sum(weights[i] for i in range(4) if mask & (1 << i))
                       for mask in range(16)])
    report = experiment.shapley(values)
    assert report["efficiency_residual"] < 1e-12
    assert np.allclose([report["allocations"][route] for route in experiment.ROUTES], weights)
    assert max(abs(value) for value in report["pair_interactions"].values()) < 1e-12


def test_price_matches_native_closures_and_nonempty_coalitions():
    expected_forwards = 2 + 2 + (2 ** len(experiment.ROUTES) - 1)
    assert expected_forwards == experiment.PRICE["model_forwards_exact"] == 19
    assert expected_forwards * 64 == experiment.PRICE["sequence_evaluations_exact"]
