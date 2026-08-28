from pathlib import Path

import tensor_bilin18_rank640_predictive_validation as validation


def test_frozen_rank_and_price():
    assert validation.RANK == 640
    assert validation.EXPECTED_TOTAL == 516_707_766
    assert validation.EXPECTED_TOTAL < validation.DENSE_TOTAL


def test_preregistration_and_create_only_result_path():
    assert validation.PREREG.exists()
    assert validation.OUTPUT.name == "tensor_bilin18_rank640_predictive_validation_results.json"


def test_prediction_gate_accepts_exact_thresholds():
    comparisons = {
        "a": {"all_ce_harm": 0.0199, "covered_ce_harm": 0.0199, "unseen_ce_harm": 0.025},
        "b": {"all_ce_harm": 0.0199, "covered_ce_harm": 0.0199, "unseen_ce_harm": 0.025},
    }
    rank512 = {"comparisons": {
        "a": {"all_ce_harm": 0.018, "covered_ce_harm": 0.018},
        "b": {"all_ce_harm": 0.018, "covered_ce_harm": 0.018},
    }}
    cost = {
        "total_stored_values": validation.EXPECTED_TOTAL,
        "native_calls_per_forward": 0, "fitted_lookup_table_values": 0,
        "total_input_support": True,
    }
    built = {"storage_disjoint": True, "native_module_references": []}
    causal = {
        "status": "rank640_robust_pass",
        "candidates": {"640": {"robust_gate": True}},
    }
    assert all(validation.prediction_gates(cost, built, comparisons, rank512, causal).values())


def test_sources_include_preregistration_and_test():
    names = {Path(path).name for path in validation.SOURCES}
    assert validation.PREREG.name in names
    assert Path(__file__).name in names
