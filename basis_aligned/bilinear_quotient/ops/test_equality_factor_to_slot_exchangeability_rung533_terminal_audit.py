#!/usr/bin/env python3

import equality_factor_to_slot_exchangeability_rung533_terminal_audit as audit533


def test_terminal_audit_recomputes_registered_result_and_failure_modes():
    report = audit533.audit()
    assert report["status"] == "audit_passed"
    assert report["calls_reconciled"] == 2208
    assert report["registered_outcome"] == "invalid_identification_test_positive_control_failed"
    assert report["recomputed_checks"] == {
        "total_contexts": 8,
        "product_control_contexts_passing": 4,
        "mapping_contexts_passing": {
            "source_first_to_target_first": 5,
            "source_second_to_target_first": 6,
            "source_first_to_target_second": 5,
            "source_second_to_target_second": 6,
        },
        "background_stability_contexts_passing": 2,
        "total_background_stability_contexts": 16,
    }
    assert report["recomputed_predictions"] == {
        "pred_a_valid_physical_instrument": True,
        "pred_b_product_level_positive_control": False,
        "pred_c_both_source_factors_fill_target_first": False,
        "pred_d_both_source_factors_fill_target_second": False,
        "pred_e_branch_exchangeable_downstream_family": False,
        "pred_f_donor_background_stability": False,
    }
    descriptive = report["descriptive_not_rescored"]
    assert descriptive["product_control"]["positive_effect_only_contexts_passing"] == 7
    assert descriptive["product_control"]["failure_context_counts"]["matched_negative"] == 3
    for mapping in audit533.rung533.MAPPINGS:
        assert descriptive[mapping]["beats_own_key_control_by_0p15_contexts"] == 8
