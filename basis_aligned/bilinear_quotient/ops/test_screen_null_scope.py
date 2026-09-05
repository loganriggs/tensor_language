"""Tests for ops/screen_null_scope.py."""
import os
import sys

OPS = os.path.dirname(os.path.abspath(__file__))
BQ = os.path.dirname(OPS)
sys.path.insert(0, OPS)
import screen_null_scope as S

NULL = os.path.join(BQ, "circuits/fast_screens/numbered_list_cached_value_sufficiency_v3_result.json")
POS = os.path.join(BQ, "circuits/fast_screens/bracket_positive_control_v1_result.json")


def test_gating_bars_are_the_selection_bars_not_capability_bars():
    """Regression: the first version printed minimum_a1_capability_accuracy as if it gated recovery."""
    assert "minimum_a1_capability_accuracy" not in S.GATING_BARS
    assert "maximum_c_absolute_recovery" in S.GATING_BARS
    assert "minimum_target_family_recovery" in S.GATING_BARS


def test_null_best_site_is_rejected_only_on_the_control():
    s = S.scope(NULL)
    assert s["reason"] == "no_selective_causal_site"
    assert s["best_site"]["reasons"] == ["C_absolute_recovery_above_fixed_bar"]
    assert s["best_site"]["a1"]["mean_absolute_effect"] > 0.9, "target WAS recovered"


def test_positive_control_selects_a_site_with_low_control_recovery():
    s = S.scope(POS)
    assert s["reason"] == "selective_causal_site"
    assert s["selected"] == "resid:18"
    assert s["best_site"]["c_absolute_recovery"] < 0.05


def test_the_two_differ_on_control_recovery_not_target_recovery():
    n, p = S.scope(NULL)["best_site"], S.scope(POS)["best_site"]
    assert abs(n["a1"]["mean_absolute_effect"] - p["a1"]["mean_absolute_effect"]) < 0.2
    assert n["c_absolute_recovery"] > 20 * p["c_absolute_recovery"]
