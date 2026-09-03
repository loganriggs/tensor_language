import os
os.environ["BQLIB_NO_MODEL"] = "1"
import pytest
import pending_opener_cross_family_das_rung540 as rung


def test_registered_fit_grid_and_price():
    assert len(rung.RANKS) * len(rung.SEEDS) * len(rung.TRAIN_SOURCES) == 45
    assert 45 * rung.STEPS == 10800
    assert rung.SPLITS == ("FIT", "SELECT")
    assert rung.CONTROL_DENOM_FLOOR == 0.05


def test_response_equivalence_is_functional_not_coordinate_based():
    cosine, rms = rung.cosine_rms([1.0, 0.0, 1.0], [1.0, 0.0, 1.0])
    assert cosine == pytest.approx(1.0) and rms == 0.0
    cosine, rms = rung.cosine_rms([1.0, 0.0], [0.0, 1.0])
    assert cosine == 0.0 and rms == 1.0


def test_bootstrap_lower_is_deterministic():
    assert rung.bootstrap_lower([1.0] * 16, 1) == 1.0
