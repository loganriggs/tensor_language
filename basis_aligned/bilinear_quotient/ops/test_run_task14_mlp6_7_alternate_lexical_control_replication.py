#!/usr/bin/env python3
"""Focused CPU tests for the alternate Task14 MLP6/7 lexical control."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import run_task14_mlp6_7_alternate_lexical_control_replication as run


def test_plan_is_targeted_and_hash_bound():
    plan = run.compile_plan()
    assert plan["conditions"] == [
        "recipient", "lexical_EAUWY_full", "lexical_EAUWZ_full",
        "lexical_EAUWYZ_full"]
    assert plan["price"] == {"physical_model_forwards": 4,
                             "example_evaluations": 224,
                             "causal_interventions": 48,
                             "backwards": 0, "parameter_updates": 0}
    assert plan["parent_result_sha256"] == run.PARENT_RESULT_SHA256


def test_alternate_donors_are_one_token_same_number_and_distinct():
    original = run.parent.build_rows()
    rows = run.build_rows()
    for before, after in zip(original, rows):
        recipient = after["endpoints"]["recipient"]
        donor = after["endpoints"]["same_number_different_lemma"]
        old_donor = before["endpoints"]["same_number_different_lemma"]
        assert donor["subject"] != recipient["subject"]
        assert donor["subject"] != old_donor["subject"]
        assert len(donor["ids"]) == len(recipient["ids"])
        assert donor["ids"][:-1] == recipient["ids"][:-1]
        assert donor["answer_id"] == recipient["answer_id"]
        assert donor["foil_id"] == recipient["foil_id"]


def test_no_model_dry_run_matches_plan():
    env = dict(os.environ, BQLIB_NO_MODEL="1", PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                          check=True, capture_output=True, text=True)
    assert json.loads(done.stdout) == run.compile_plan()


def _evidence(mixed_y, mixed_z, completed):
    values = {"recipient": 0.0, "lexical_EAUWY_full": mixed_y,
              "lexical_EAUWZ_full": mixed_z,
              "lexical_EAUWYZ_full": completed}
    evidence = []
    for row in run.build_rows():
        cell = f"{row['direction_id']}__{row['template_id']}"
        for condition in run.CONDITIONS:
            value = values[condition] if cell == run.PRIMARY_CELL else 0.0
            evidence.append({"row_id": row["row_id"], "cell_id": cell,
                             "condition": condition,
                             "alternate_lexical_subject": "probe",
                             "lexical_target_margin_improvement": value,
                             "lexical_target_CE_improvement": value})
    return evidence


def test_score_separates_replication_from_control_sensitivity(monkeypatch):
    parent = json.loads(run.PARENT_RESULT.read_text())
    denominator = parent["score"]["cells"][run.PRIMARY_CELL]["opposite"][
        "full"]["margin"]["effects"]["EAUWYZ"]
    exact = {name: 0.0 for name in (
        "native_role_replay_max_absolute_logit_error",
        "role_input_state_closure_max_absolute_error",
        "role_input_normalized_closure_max_absolute_error",
        "full_source_input_max_absolute_error", "full_source_output_max_absolute_error",
        "full_source_propagated_slot_max_absolute_error",
        "full_source_installed_head_max_absolute_error",
        "recipient_noop_full_logit_max_absolute_error",
        "downstream_state_closure_max_absolute_error",
        "downstream_normalized_closure_max_absolute_error")}
    replicated = run.score(_evidence(.30 * denominator, .10 * denominator,
                                     .10 * denominator), exact, [1.0] * 16)
    assert replicated["predictions"] == {
        "pred_a_instrument_live": True,
        "pred_b_lexical_entanglement_replicates": True,
        "pred_c_control_family_sensitive": False,
        "pred_d_mlp7_completion_cancels": True,
        "pred_e_reciprocal_mixed_corner_quiet": True}
    sensitive = run.score(_evidence(.20 * denominator, .10 * denominator,
                                    .10 * denominator), exact, [1.0] * 16)
    assert sensitive["predictions"]["pred_c_control_family_sensitive"]
    assert not sensitive["predictions"]["pred_b_lexical_entanglement_replicates"]


def test_preflight_fails_closed(monkeypatch):
    monkeypatch.setattr(run, "PRIOR_ART_SHA256", "0" * 64)
    with pytest.raises(run.AlternateLexicalControlError, match="prior-art"):
        run.validate_preflight()
