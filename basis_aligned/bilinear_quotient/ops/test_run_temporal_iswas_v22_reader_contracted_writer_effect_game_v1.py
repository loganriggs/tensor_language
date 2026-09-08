import numpy as np

import run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1 as subject


def test_exact_shapley_recovers_additive_route_game():
    coefficients = np.linspace(-0.2, 0.7, len(subject.ROUTES))
    values = np.asarray([
        sum(coefficients[i] for i in range(len(subject.ROUTES)) if mask & (1 << i))
        for mask in range(1 << len(subject.ROUTES))
    ])
    report = subject.shapley(values)
    assert report["efficiency_residual"] < 1e-12
    assert np.allclose([report["allocations"][route] for route in subject.ROUTES], coefficients)
    assert max(abs(value) for value in report["pair_interactions"].values()) < 1e-12


def test_pair_interaction_recovers_pure_complementarity():
    values = np.asarray([
        float(bool(mask & 1) and bool(mask & 2))
        for mask in range(1 << len(subject.ROUTES))
    ])
    report = subject.shapley(values)
    assert abs(report["allocations"][subject.ROUTES[0]] - 0.5) < 1e-12
    assert abs(report["allocations"][subject.ROUTES[1]] - 0.5) < 1e-12
    assert abs(report["pair_interactions"][f"{subject.ROUTES[0]}|{subject.ROUTES[1]}"] - 1.0) < 1e-12
    assert report["efficiency_residual"] < 1e-12


def test_registered_inventory_and_price_are_exact():
    assert len(subject.ROUTES) == 10
    assert len(set(subject.ROUTES)) == 10
    assert set(subject.HEAD_ROUTES) <= set(subject.ROUTES)
    assert subject.PRICE["model_forwards_exact"] == 2 + 2 + (1 << len(subject.ROUTES)) - 1
    assert subject.PRICE["sequence_evaluations_exact"] == 64 * subject.PRICE["model_forwards_exact"]
