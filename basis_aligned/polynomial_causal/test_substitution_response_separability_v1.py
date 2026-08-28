import copy
import json
from pathlib import Path

import pytest

import substitution_response_separability_v1 as subject


SOURCE = (
    Path(__file__).parents[1]
    / "bilinear_quotient"
    / "ops"
    / "substitution_direction_curve_results.json"
)


def test_committed_result_closes_and_prediction_is_literal():
    result = subject.analyze(json.loads(SOURCE.read_text()))
    assert result["cumulative_frobenius_energy"][0] > 0.98
    assert result["calibration_to_full_spearman"] > 0.95
    assert 2.5 < result["anchored_power_law"]["exponent_p"] < 3.2
    assert result["anchored_power_law"]["curve_relative_l2"] < 0.06
    assert result["predictions_for_length1_replication"] == {
        "rank1_energy_at_least": 0.98,
        "alpha_0_25_to_full_spearman_at_least": 0.95,
    }


def test_registry_drift_fails_closed():
    payload = json.loads(SOURCE.read_text())
    broken = copy.deepcopy(payload)
    broken["curve"].pop("mlp17")
    with pytest.raises(ValueError, match="34 site curves"):
        subject.analyze(broken)
