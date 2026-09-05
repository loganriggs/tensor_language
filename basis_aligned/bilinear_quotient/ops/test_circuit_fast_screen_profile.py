#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import circuit_fast_screen_profile as profile


ROOT = Path(__file__).resolve().parents[1]


def _result(*, p_scale: float = 2.0, p_move: float = 0.2, c: float = 0.1) -> dict:
    site = {"site_id": "attn:08", "evidence_kind": "module"}
    return {
        "schema": "circuit_fast_screen_result_v1",
        "run": {
            "site_results": [{
                "site": site,
                "a1": {"mean_effect": 0.7, "direction_fraction": 1.0},
                "a2": {"mean_effect": 0.6, "direction_fraction": 1.0},
                "p_invariance_effect": p_move / p_scale,
                "c_absolute_recovery": c,
                "c_signed_recovery": c,
            }],
            "native_logits": [{
                "family": "P", "side": "base", "row_id": "p:1",
                "answer_logit": 3.0, "foil_logit": 1.0,
            }],
            "intervention_logits": [{
                "family": "P", "row_id": "p:1", "site": site,
                "answer_logit": 3.0 + p_move, "foil_logit": 1.0,
            }],
        },
    }


def test_target_selection_ignores_control_verdicts() -> None:
    clean = _result(c=0.01)
    shared = _result(c=0.9)
    output = profile.profile_results([("far-control", clean), ("near-control", shared)])
    assert output["mechanistic_target_candidates"] == ["attn:08"]
    assert output["site_profiles"][0]["target_pass_all_present"] is True


def test_raw_p_movement_is_stable_when_normalization_changes() -> None:
    first = _result(p_scale=2.0)
    second = _result(p_scale=8.0)
    responses = profile.profile_results([("a", first), ("b", second)])["site_profiles"][0]["responses"]
    assert responses[0]["p_normalized"] == pytest.approx(0.1)
    assert responses[1]["p_normalized"] == pytest.approx(0.025)
    assert responses[0]["p_raw_mean_margin_movement"] == pytest.approx(0.2)
    assert responses[1]["p_raw_mean_margin_movement"] == pytest.approx(0.2)


def test_target_failure_remains_visible_without_becoming_a_control_null() -> None:
    failed = _result()
    failed["run"]["site_results"][0]["a2"]["mean_effect"] = 0.49
    output = profile.profile_results([("member", failed)])
    assert output["mechanistic_target_candidates"] == []
    assert output["site_profiles"][0]["target_pass_any"] is False


def test_residual_boundary_is_reported_only_as_a_ceiling() -> None:
    result = _result()
    residual = copy.deepcopy(result["run"]["site_results"][0])
    residual["site"] = {"site_id": "resid:18", "evidence_kind": "residual"}
    result["run"]["site_results"].append(residual)
    intervention = copy.deepcopy(result["run"]["intervention_logits"][0])
    intervention["site"] = residual["site"]
    result["run"]["intervention_logits"].append(intervention)
    output = profile.profile_results([("member", result)])
    assert output["mechanistic_target_candidates"] == ["attn:08"]
    assert output["residual_ceiling_sites"] == ["resid:18"]


def test_missing_p_evidence_fails_closed() -> None:
    result = _result()
    result["run"]["intervention_logits"] = []
    with pytest.raises(profile.ProfileError, match="lacks P intervention"):
        profile.profile_results([("member", result)])


def test_numeric_sequence_history_is_a_carrier_with_distinct_control_responses() -> None:
    folder = ROOT / "circuits" / "fast_screens"
    names = {
        "near_C_digit_P": "numeric_sequence_cross_construction_v1_result.json",
        "far_C_digit_P": "numeric_sequence_control_choice_v1_result.json",
        "far_C_word_P": "numeric_sequence_p_family_v1_result.json",
        "alternate_A2": "numeric_sequence_a2_family_v1_result.json",
    }
    members = [
        (label, json.loads((folder / filename).read_text()))
        for label, filename in names.items()
    ]
    output = profile.profile_results(members)
    assert output["mechanistic_target_candidates"] == ["attn:08"]
    attn8 = next(item for item in output["site_profiles"] if item["site_id"] == "attn:08")
    responses = {item["member"]: item for item in attn8["responses"]}
    assert responses["near_C_digit_P"]["c_absolute_recovery"] == pytest.approx(0.9236163, abs=1e-6)
    assert responses["far_C_digit_P"]["c_absolute_recovery"] == pytest.approx(0.0136282, abs=1e-6)
    assert responses["far_C_digit_P"]["p_raw_mean_margin_movement"] == pytest.approx(0.6948982, abs=1e-6)
    assert responses["far_C_word_P"]["p_raw_mean_margin_movement"] == pytest.approx(0.1546568, abs=1e-6)
    # A2 changed the normalized P number, but the underlying P intervention did not.
    assert responses["far_C_word_P"]["p_raw_mean_margin_movement"] == pytest.approx(
        responses["alternate_A2"]["p_raw_mean_margin_movement"]
    )
