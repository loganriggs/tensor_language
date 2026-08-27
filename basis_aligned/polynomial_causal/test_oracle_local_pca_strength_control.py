import pytest

from oracle_local_pca_strength_control import (
    analyze_strength_control,
    match_monotone_scale,
    paired_gain,
)


def test_monotone_scale_match_uses_bounded_geometric_bisection():
    result = match_monotone_scale(9.0, lambda scale: scale * scale)
    assert result["scale"] == pytest.approx(3.0, rel=0.01)
    assert result["relative_error"] <= 0.01
    assert result["bounds"] == [0.1, 10.0]


def test_monotone_scale_match_fails_closed_on_bad_geometry():
    with pytest.raises(ValueError, match="not bracketed"):
        match_monotone_scale(200.0, lambda scale: scale)
    with pytest.raises(ValueError, match="decreases"):
        match_monotone_scale(1.0, lambda scale: 2.0 - scale / 10.0)


def test_paired_gain_validates_and_uses_positive_is_repair_sign():
    assert paired_gain([3.0, 5.0], [2.0, 4.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="equal positive"):
        paired_gain([1.0], [])


def _gain(mean):
    return {"mean": mean}


def test_strength_gate_uses_minimum_across_splits_and_exact_twenty_nulls():
    candidate = {"discovery": _gain(0.10), "heldout": _gain(0.08)}
    nulls = {
        f"null_{index:02d}": {
            "discovery": _gain(0.06 - index * 0.001),
            "heldout": _gain(0.07 - index * 0.001),
        }
        for index in range(20)
    }
    result = analyze_strength_control(
        candidate, nulls, full_heldout_gain=0.16,
        bootstrap_ci95=[0.03, 0.13],
    )
    assert result["candidate_joint_split_statistic"] == pytest.approx(0.08)
    assert result["exact_one_sided_p"] == pytest.approx(1 / 21)
    assert result["decision"]["passes"] is True

    nulls["null_19"]["discovery"] = _gain(0.20)
    nulls["null_19"]["heldout"] = _gain(0.09)
    failed = analyze_strength_control(
        candidate, nulls, full_heldout_gain=0.16,
        bootstrap_ci95=[0.03, 0.13],
    )
    assert failed["nulls_at_least_candidate"] == 1
    assert failed["decision"]["passes"] is False


def test_strength_gate_requires_positive_full_oracle_denominator():
    candidate = {"discovery": _gain(0.10), "heldout": _gain(0.08)}
    nulls = {
        f"null_{index:02d}": {"discovery": _gain(0.01), "heldout": _gain(0.01)}
        for index in range(20)
    }
    result = analyze_strength_control(
        candidate, nulls, full_heldout_gain=-0.1,
        bootstrap_ci95=[0.03, 0.13],
    )
    assert result["heldout_fraction_of_full_oracle_gain"] is None
    assert result["decision"]["passes"] is False
