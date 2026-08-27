from dataclasses import FrozenInstanceError
import math

import pytest

from joint_early_mlp_oracle_factorial import (
    EARLY_MLP_GROUPS,
    OracleCorrectionSpec,
    analyze_full_live_subset_rows,
    configure_oracle_corrections,
    freeze_oracle_corrections,
    resolve_oracle_correction,
)
from factorial_causal_attribution import powerset


def test_singleton_runtime_api_is_backward_compatible():
    legacy_basis = object()
    state = {
        "on": True,
        "site": 1,
        "basis": legacy_basis,
        "scale": 0.75,
        "corrections": freeze_oracle_corrections({0: {"basis": object()}}),
    }
    assert resolve_oracle_correction(state, 0) is None
    resolved = resolve_oracle_correction(state, 1)
    assert resolved is not None
    assert resolved.basis is legacy_basis
    assert resolved.scale == 0.75
    assert resolve_oracle_correction({**state, "on": False}, 1) is None


def test_joint_runtime_map_is_immutable_and_resolves_each_site():
    basis0, basis2 = object(), object()
    state = {}
    frozen = configure_oracle_corrections(state, {
        0: {"basis": basis0, "scale": 1.0},
        2: OracleCorrectionSpec(basis2, 0.5),
    })
    assert state["site"] is None
    assert resolve_oracle_correction(state, 0).basis is basis0
    assert resolve_oracle_correction(state, 1) is None
    assert resolve_oracle_correction(state, 2).scale == 0.5
    with pytest.raises(TypeError):
        frozen[1] = OracleCorrectionSpec(None)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        frozen[0].scale = 2.0  # type: ignore[misc]


def _gain_cube_from_terms(terms):
    return {
        arm: sum(value for term, value in terms.items() if set(term).issubset(arm))
        for arm in powerset(EARLY_MLP_GROUPS)
    }


def _row_ce_cube(gains):
    baseline = [10.0, 12.0, 14.0]
    return {arm: [value - gains[arm] for value in baseline]
            for arm in powerset(EARLY_MLP_GROUPS)}


def test_full_subset_analysis_closes_and_preserves_signed_interactions():
    terms = {
        (): 0.0,
        ("mlp0",): 1.0,
        ("mlp1",): 2.0,
        ("mlp2",): -1.0,
        ("mlp0", "mlp1"): 0.5,
        ("mlp0", "mlp2"): 0.2,
        ("mlp1", "mlp2"): -0.3,
        ("mlp0", "mlp1", "mlp2"): 0.6,
    }
    result = analyze_full_live_subset_rows(
        _row_ce_cube(_gain_cube_from_terms(terms)),
        mlp012_residual_nats=6.0,
    )
    assert result["joint_gain"] == pytest.approx(3.0)
    assert result["joint_gain_fraction_of_mlp012_residual"] == pytest.approx(0.5)
    assert result["pairwise_mobius"] == pytest.approx({
        "mlp0+mlp1": 0.5,
        "mlp0+mlp2": 0.2,
        "mlp1+mlp2": -0.3,
    })
    assert result["triple_mobius"] == pytest.approx(0.6)
    assert result["shapley_closure_error"] == pytest.approx(0.0, abs=1e-12)
    assert sum(result["shapley"].values()) == pytest.approx(result["joint_gain"])
    assert result["mlp2_conditional_marginal_after_mlp0_mlp1"] == pytest.approx(-0.5)


def test_joint_gain_fraction_is_absent_without_same_currency_denominator():
    gains = {arm: 0.1 * len(arm) for arm in powerset(EARLY_MLP_GROUPS)}
    result = analyze_full_live_subset_rows(_row_ce_cube(gains))
    assert result["joint_gain"] == pytest.approx(0.3)
    assert result["joint_gain_fraction_of_mlp012_residual"] is None


def test_subset_analyzer_rejects_incomplete_nonfinite_or_mismatched_rows():
    gains = {arm: 0.1 * len(arm) for arm in powerset(EARLY_MLP_GROUPS)}
    rows = _row_ce_cube(gains)
    rows.pop(("mlp2",))
    with pytest.raises(ValueError, match="cube mismatch"):
        analyze_full_live_subset_rows(rows)

    rows = _row_ce_cube(gains)
    rows[("mlp2",)] = rows[("mlp2",)][:-1]
    with pytest.raises(ValueError, match="same positive row count"):
        analyze_full_live_subset_rows(rows)

    rows = _row_ce_cube(gains)
    rows[("mlp2",)][0] = math.nan
    with pytest.raises(ValueError, match="must be finite"):
        analyze_full_live_subset_rows(rows)

    with pytest.raises(ValueError, match="finite and positive"):
        analyze_full_live_subset_rows(_row_ce_cube(gains), mlp012_residual_nats=0.0)
