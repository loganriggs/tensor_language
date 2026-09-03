#!/usr/bin/env python3

import equality_product_shared_private_rung534_math as rung534


def test_exact_shared_private_product_recomposition():
    report = rung534.exact_algebra()
    assert report["maximum_recomposition_error"] <= 1e-15
    assert report["shared_is_gauge_invariant_complete_product"] is True


def test_parent_localizes_the_code_failure_to_specificity_not_copy_effect():
    report = rung534.analyze_parent()
    summary = report["code_donor_absent_summary"]
    assert min(summary["shared_positive_cosines"]) >= 0.98
    assert max(summary["shared_positive_relative_errors"]) <= 0.25
    assert min(summary["shared_positive_recovery_from_parent"]) >= 0.85
    assert min(summary["shared_matched_negative_ce_mismatch"]) >= 0.02
