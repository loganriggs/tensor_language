from __future__ import annotations

import pytest

from hybrid_tensor_class_oracle_stats import analyze_hybrid_losses


def test_additive_harms_have_zero_interaction_and_attention_dominates() -> None:
    result = analyze_hybrid_losses({
        "both_compiled": 4.5,
        "attention_native": 3.5,
        "mlp_native": 4.0,
        "both_native": 3.0,
    }, live_ce=3.0)
    assert result.harm["both_compiled"] == 1.5
    assert result.attention_restoration_gain == 1.0
    assert result.mlp_restoration_gain == 0.5
    assert result.interaction_harm == 0.0
    assert result.dominant_missing_contraction == "attention"


def test_positive_interaction_is_superadditive_compilation_harm() -> None:
    result = analyze_hybrid_losses({
        "both_compiled": 5.0,
        "attention_native": 3.4,
        "mlp_native": 3.6,
        "both_native": 3.0,
    }, live_ce=3.0)
    assert result.interaction_harm == pytest.approx(1.0)
    assert result.attention_restoration_gain == pytest.approx(1.6)
    assert result.mlp_restoration_gain == pytest.approx(1.4)


def test_tied_dominance_uses_explicit_tolerance() -> None:
    result = analyze_hybrid_losses({
        "both_compiled": 4.0,
        "attention_native": 3.5,
        "mlp_native": 3.5000005,
        "both_native": 3.0,
    }, live_ce=3.0, dominance_atol=1e-3)
    assert result.dominant_missing_contraction == "tied"


def test_missing_arm_nonfinite_or_bad_native_control_fails_closed() -> None:
    good = {
        "both_compiled": 4.0,
        "attention_native": 3.5,
        "mlp_native": 3.5,
        "both_native": 3.0,
    }
    with pytest.raises(ValueError, match="malformed"):
        analyze_hybrid_losses({**good, "extra": 3.0}, live_ce=3.0)
    with pytest.raises(ValueError, match="malformed"):
        analyze_hybrid_losses({**good, "mlp_native": float("nan")}, live_ce=3.0)
    with pytest.raises(ValueError, match="both-native"):
        analyze_hybrid_losses({**good, "both_native": 3.1}, live_ce=3.0)
