import numpy as np

import score_mlp0_c512_mlp1_interchange_v1 as score


def synthetic_ledgers(n_units, effects=None, n_cells=2):
    effects = effects or {}
    ledgers = {}
    for arm in score.ARMS:
        ledgers[arm] = {}
        value = effects.get(arm, 0.2)
        for consumer, margin in score.MARGINS.items():
            counts = np.ones((n_units, n_cells))
            sums = counts * margin * value
            ledgers[arm][consumer] = {"sums": sums.tolist(), "counts": counts.tolist()}
    return ledgers


def test_ce_is_two_sided_before_standardization():
    sums = np.array([-0.015, 0.0075])
    effects = score._effect("ce_abs", sums, np.ones(2))
    assert np.allclose(effects, [0.015, 0.0075])


def test_familywise_bounds_center_coordinates_before_global_max():
    coordinates = np.array([[.50, .49]])
    bootstrap = np.array([[[.50, .58]], [[.50, .58]], [[.50, .58]]])
    upper, _, correction, _ = score.familywise_bounds(coordinates, bootstrap)
    assert np.isclose(correction, .09)
    assert np.isclose(upper[0], .59)


def test_one_common_bootstrap_family_contains_every_registered_arm(monkeypatch):
    monkeypatch.setattr(score, "MIN_FINEWEB_DOCUMENTS_PER_CELL", 1)
    ledger = synthetic_ledgers(12)
    scope = score.score_scope(ledger, np.arange(12), minimum_support=1, n_bootstrap=20, seed=3)
    assert set(scope["arms"]) == set(score.ARMS)
    assert scope["bootstrap"]["replicates"] == 20


def test_rescue_comparison_is_positive_when_upstream_state_is_better(monkeypatch):
    effects = {
        "live/observational_CC": .8,
        "live/upstream_state": .2,
        "mlp2_omit/observational_CC": .7,
        "mlp2_omit/upstream_state": .3,
    }
    scope = score.score_scope(
        synthetic_ledgers(20, effects), np.arange(20), minimum_support=1,
        n_bootstrap=50, seed=4,
    )
    assert scope["rescue"]["comparisons"]["rescue_live"]["familywise_95pct_lcb_reduction"] > 0


def test_full_decision_requires_sensitive_positive_controls(monkeypatch):
    monkeypatch.setattr(score, "MIN_FINEWEB_DOCUMENTS_PER_CELL", 1)
    monkeypatch.setattr(score, "MIN_CODE_FILES_PER_CELL", 1)
    effects = {
        "live/native_write": 2.0,
        "mlp2_omit/native_write": 2.0,
    }
    payload = {
        "sufficient_statistics": {
            "fineweb": synthetic_ledgers(384, effects),
            "code": synthetic_ledgers(48, effects),
        },
        "coverage": {"fineweb": {"wave_A": 1.0, "wave_B": 1.0}, "code": 1.0},
    }
    result = score.score_result(payload, n_bootstrap=30, seed=5)
    assert result["decisions"]["downstream_null_on_registered_fineweb_backgrounds"]
    assert result["decisions"]["positive_control_each_background"] == {
        "live": True, "mlp2_omit": True
    }
