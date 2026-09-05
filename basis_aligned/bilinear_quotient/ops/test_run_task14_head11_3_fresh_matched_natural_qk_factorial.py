#!/usr/bin/env python3
"""Focused CPU tests for the licensed fresh Task14 score-by-value factorial."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_task14_head11_3_fresh_matched_natural_qk_factorial as run


def _exactness(value=0.0):
    return {
        "native_replay_max_absolute_logit_error": value,
        "source_term_sum_max_absolute_error": value,
        "same_batch_native_noop_endpoint_max_absolute_error": value,
        "installed_term_max_absolute_error": value,
        "complete_head_vector_max_absolute_error": value,
    }


def _synthetic_evidence():
    evidence = []
    for authority_row in run.build_rows():
        direction = authority_row["direction_id"]
        direction_sign = 1 if direction == "singular_to_plural" else -1
        cell = f"{direction}__{authority_row['template_id']}"
        values = {
            "same_score_same_value": (0.0, 0.0, 0.0),
            "opposite_score_same_value": (-direction_sign, -1.0, -0.4),
            "same_score_opposite_value": (0.0, 0.0, 0.0),
            "opposite_score_opposite_value": (direction_sign, 1.0, 0.5),
            "lexical_score_same_value": (-0.1*direction_sign, -0.1, 0.0),
            "lexical_score_opposite_value": (0.1*direction_sign, 0.1, 0.0),
            "complete_opposite_head": (direction_sign, 1.0, 0.8),
        }
        for condition in run.CONDITIONS:
            fixed_margin, donor_margin, ce = values[condition]
            evidence.append({
                "row_id": authority_row["row_id"], "cell_id": cell,
                "condition": condition,
                "are_minus_is_margin_effect": fixed_margin,
                "donor_margin_improvement": donor_margin,
                "donor_CE_improvement": ce,
            })
    return evidence


def test_plan_validates_exact_license_and_uses_holdout_only():
    plan = run.compile_plan()
    assert plan["preflight_validated"] is True
    assert plan["capability_license_sha256"] == run.EXPECTED_LICENSE_SHA256
    assert plan["candidate_id"] == run.CAUSAL_CANDIDATE_ID
    assert plan["split"] == "LICENSED_HOLDOUT"
    assert plan["row_count"] == 16
    assert plan["price"] == {
        "model_forwards": 4, "example_evaluations": 320,
        "backwards": 0, "parameter_updates": 0,
    }
    rows = run.build_rows()
    assert {row["phase"] for row in rows} == {"HOLDOUT"}
    assert {row["group_number"] for row in rows} == set(range(8, 16))


def test_dry_run_is_preflighted_and_never_loads_model():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [sys.executable, str(Path(run.__file__))], env=env,
        check=True, capture_output=True, text=True)
    assert json.loads(completed.stdout) == run.compile_plan()


def test_preflight_fails_closed_on_wrong_license_sha(monkeypatch):
    monkeypatch.setattr(run, "EXPECTED_LICENSE_SHA256", "0" * 64)
    with pytest.raises(run.licensing.CapabilityLicenseError,
                       match="license hash changed"):
        run.validate_preflight()


def test_patch_compiler_has_exact_seven_arms_and_source_algebra():
    torch = pytest.importorskip("torch")
    rows = run.build_rows()
    count, width = len(rows), 5
    recipient = {
        "p": torch.arange(count*9, dtype=torch.float32).reshape(count, 9)/100 + 1,
        "u": torch.arange(count*9*width, dtype=torch.float32).reshape(count, 9, width)/100,
        "head": torch.arange(count*width, dtype=torch.float32).reshape(count, width),
    }
    opposite = {key: value+1 for key, value in recipient.items()}
    lexical = {key: value+2 for key, value in recipient.items()}
    tokens = torch.tensor([row["endpoints"]["recipient"]["ids"] for row in rows])
    patch = run._compile_patch_batch(tokens, recipient, opposite, lexical, rows, torch)
    assert len(patch["specs"]) == count * len(run.CONDITIONS) == 112
    assert set(condition for _, condition, _ in patch["specs"]) == set(run.CONDITIONS)
    for index, (row_index, condition, _) in enumerate(patch["specs"]):
        native_term = recipient["p"][row_index, 8] * recipient["u"][row_index, 8]
        if condition == "complete_opposite_head":
            assert torch.equal(patch["replacement_heads"][index], opposite["head"][row_index])
        else:
            observed = patch["replacement_heads"][index] - recipient["head"][row_index] + native_term
            assert torch.allclose(observed, patch["expected_terms"][index])


def test_scoring_keeps_fixed_margin_and_answer_directed_ce_separate():
    scored = run.score(_synthetic_evidence(), _exactness())
    predictions = scored["predictions"]
    assert predictions == {
        "pred_a_instrument_live": True,
        "pred_b_number_score_discriminative": True,
        "pred_c_lexically_selective": True,
        "pred_d_bidirectional_task_use": True,
        "pred_e_directionally_asymmetric_task_use": False,
    }
    singular = scored["cells"]["singular_to_plural__across_beside"]
    plural = scored["cells"]["plural_to_singular__across_beside"]
    assert singular["number_score_opposite_value"]["mean_donor_CE_improvement"] == .5
    assert plural["number_score_opposite_value"]["mean"] == -1.0
    assert plural["number_score_opposite_value"]["mean_donor_margin_improvement"] == 1.0
    assert plural["number_score_opposite_value"]["mean_donor_CE_improvement"] == .5


def test_directionally_asymmetric_alternative_is_independently_reported():
    evidence = _synthetic_evidence()
    for item in evidence:
        if item["cell_id"].startswith("plural_to_singular") \
                and item["condition"] == "opposite_score_opposite_value":
            item["are_minus_is_margin_effect"] = 1.0
            item["donor_margin_improvement"] = -1.0
            item["donor_CE_improvement"] = -0.5
    scored = run.score(evidence, _exactness())
    assert not scored["predictions"]["pred_b_number_score_discriminative"]
    assert not scored["predictions"]["pred_d_bidirectional_task_use"]
    assert scored["predictions"]["pred_e_directionally_asymmetric_task_use"]


def test_exactness_or_complete_head_failure_invalidates_instrument():
    bad_exact = run.score(_synthetic_evidence(), _exactness(1.0))
    assert not bad_exact["predictions"]["pred_a_instrument_live"]
    evidence = _synthetic_evidence()
    for item in evidence:
        if item["condition"] == "complete_opposite_head":
            item["donor_margin_improvement"] = 0.0
            item["donor_CE_improvement"] = 0.0
    dead_control = run.score(evidence, _exactness())
    assert not dead_control["predictions"]["pred_a_instrument_live"]


def test_same_batch_native_noop_control_records_tolerance_and_can_fail():
    within = _exactness()
    within["same_batch_native_noop_endpoint_max_absolute_error"] = 6.9e-5
    assert run.score(_synthetic_evidence(), within)["predictions"]["pred_a_instrument_live"]
    above = _exactness()
    above["same_batch_native_noop_endpoint_max_absolute_error"] = 7.1e-5
    assert not run.score(_synthetic_evidence(), above)["predictions"]["pred_a_instrument_live"]


def test_arm_metrics_use_the_supplied_same_index_native_baseline():
    torch = pytest.importorskip("torch")
    row = next(row for row in run.build_rows()
               if row["direction_id"] == "singular_to_plural")
    native = torch.zeros(50257)
    observed = torch.zeros(50257)
    native[318], native[389] = 2.0, 0.0
    observed[318], observed[389] = 1.0, 3.0
    metrics = run._comparison_metrics(native, observed, row, torch)
    assert metrics["native_recipient_margin"] == 2.0
    assert metrics["native_donor_margin"] == -2.0
    assert metrics["recipient_margin"] == -2.0
    assert metrics["donor_margin"] == 2.0
    assert metrics["recipient_margin_improvement"] == -4.0
    assert metrics["donor_margin_improvement"] == 4.0
    assert metrics["are_minus_is_margin_effect"] == 4.0
    assert metrics["donor_CE_improvement"] > 0
    assert metrics["recipient_CE_improvement"] < 0


def test_lexical_control_can_falsify_selectivity_without_erasing_number_signal():
    evidence = _synthetic_evidence()
    for item in evidence:
        if item["condition"].startswith("lexical_score_"):
            item["are_minus_is_margin_effect"] *= 8
    scored = run.score(evidence, _exactness())
    assert scored["predictions"]["pred_b_number_score_discriminative"]
    assert not scored["predictions"]["pred_c_lexically_selective"]


def test_scoring_fails_closed_on_missing_or_duplicate_factorial_arm():
    evidence = _synthetic_evidence()
    with pytest.raises(run.FactorialError, match="exactly cover"):
        run.score(evidence[:-1], _exactness())
    duplicated = list(evidence)
    duplicated[-1] = dict(duplicated[0])
    with pytest.raises(run.FactorialError, match="exactly cover"):
        run.score(duplicated, _exactness())
