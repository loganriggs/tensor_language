#!/usr/bin/env python3

import equality_factor_companion_causal_equivalence_rung532_terminal_audit as audit532


def test_terminal_audit_recomputes_frozen_result_and_failure_reason():
    report = audit532.audit()
    assert report["status"] == "audit_passed"
    assert report["calls_reconciled"] == 2625
    assert report["recomputed_checks"] == {
        "product_control_contexts_passing": 8,
        "swapped_first_contexts_passing": 0,
        "swapped_second_contexts_passing": 0,
        "total_contexts": 8,
        "composition_contexts_passing": 8,
    }
    assert report["strong_null"] is True
    descriptive = report["descriptive_not_rescored"]
    assert descriptive["swapped_first"]["base_and_slice_contexts_passing"] == 8
    assert descriptive["swapped_second"]["base_and_slice_contexts_passing"] == 8
    assert descriptive["direct_first"]["base_and_slice_contexts_passing"] == 8
    assert descriptive["direct_second"]["base_and_slice_contexts_passing"] == 8
    assert descriptive["permuted_first"]["base_and_slice_contexts_passing"] == 0
    assert descriptive["permuted_second"]["base_and_slice_contexts_passing"] == 0
    reason = descriptive["registered_failure_reason"]
    assert reason["swapped_first_beats_permuted_by_0p15_contexts"] == 8
    assert reason["swapped_second_beats_permuted_by_0p15_contexts"] == 8
    assert reason["swapped_first_beats_direct_by_0p15_contexts"] == 0
    assert reason["swapped_second_beats_direct_by_0p15_contexts"] == 0
