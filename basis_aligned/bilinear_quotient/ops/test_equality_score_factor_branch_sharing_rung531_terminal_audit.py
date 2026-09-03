#!/usr/bin/env python3

import equality_score_factor_branch_sharing_rung531_terminal_audit as audit531


def test_terminal_audit_recomputes_frozen_result():
    report = audit531.audit()
    assert report["status"] == "audit_passed"
    assert report["calls_reconciled"] == 125
    assert report["recomputed_predictions"] == {
        "pred_a_exact_authorized_instrument": True,
        "pred_b_both_score_factors_shared": False,
        "pred_c_exactly_one_score_factor_shared": False,
        "pred_d_factor_gauges_match_product": False,
        "strong_null": True,
    }
    assert report["both_factor_candidates"] == []
    assert report["one_factor_candidates"] == []
    assert report["gauge_consistent_candidates"] == []
