#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

import circuit_fast_screen_publish as publish


SPEC_PATH = publish.BQ / "circuits/fast_screens/task14_subject_verb_agreement_full_state_v2_publication.json"
CROSS_SYNTAX_SPEC_PATH = publish.BQ / "circuits/fast_screens/task14_subject_verb_agreement_cross_syntax_v1_publication.json"
SELECT_CROSS_SYNTAX_SPEC_PATH = publish.BQ / "circuits/fast_screens/task14_subject_verb_agreement_select_cross_syntax_v1_publication.json"
CROSS_NOUN_SPEC_PATH = publish.BQ / "circuits/fast_screens/task14_subject_verb_agreement_select_cross_noun_v1_publication.json"
RECORD_PATH = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def _spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


def _cross_syntax_spec() -> dict:
    return json.loads(CROSS_SYNTAX_SPEC_PATH.read_text())


def _select_cross_syntax_spec() -> dict:
    return json.loads(SELECT_CROSS_SYNTAX_SPEC_PATH.read_text())


def _cross_noun_spec() -> dict:
    return json.loads(CROSS_NOUN_SPEC_PATH.read_text())


def test_task14_plan_binds_real_ledger_result_and_nontrivial_site() -> None:
    plan = publish.build_plan(_spec())
    assert plan["ledger_request_id"] == "task14-subject-verb-agreement-full-state-v2"
    assert plan["event"]["site_id"] == "attention.block11.output.final_position"
    metrics = {item["name"]: item["estimate"] for item in plan["event"]["metrics"]}
    assert metrics["A1_mean_donor_recovery"] == pytest.approx(0.6190234173)
    assert metrics["A2_mean_donor_recovery"] == pytest.approx(0.6061822740)
    assert metrics["P_normalized_margin_movement"] == pytest.approx(0.0331384574)
    assert metrics["C_absolute_recovery"] == pytest.approx(0.0356467249)
    assert metrics["passing_nonresidual_site_count"] == 2


def test_unknown_semantic_mapping_fails_before_a_plan_exists() -> None:
    spec = _spec()
    spec["transform_to_family_id"]["C"] = "invented_control"
    with pytest.raises(publish.FastScreenPublishError, match="unknown canonical family"):
        publish.build_plan(spec)


def test_cross_syntax_plan_preserves_its_fit_only_scope() -> None:
    plan = publish.build_cross_syntax_plan(_cross_syntax_spec())
    assert plan["ledger_request_id"] == "task14-subject-verb-agreement-cross-syntax-v1"
    assert plan["event"]["site_id"] == \
        "attention.block11.head3.pre_output_projection.final_position"
    assert plan["event"]["evaluation_role"] == \
        "FIT_VALIDATION_new_relations_not_unseen_text"
    metrics = {item["name"]: item["estimate"] for item in plan["event"]["metrics"]}
    assert metrics["cross_syntax_mean_donor_recovery"] == pytest.approx(0.5892548665)
    assert metrics["minimum_cross_syntax_cell_recovery"] == pytest.approx(0.4581604762)
    assert metrics["minimum_cross_syntax_direction_fraction"] == 1.0
    assert metrics["passing_preselected_site_count"] == 2


def test_cross_syntax_plan_rejects_an_unknown_family() -> None:
    spec = _cross_syntax_spec()
    spec["family_ids"][2] = "invented_cross_syntax_family"
    with pytest.raises(publish.FastScreenPublishError, match="unknown canonical family"):
        publish.build_cross_syntax_plan(spec)


def test_select_cross_syntax_plan_is_held_out_and_checkpoint_bound() -> None:
    plan = publish.build_cross_syntax_plan(_select_cross_syntax_spec())
    assert plan["event"]["evaluation_role"] == \
        "SELECT_HELD_OUT_unseen_nouns_and_prompt_templates"
    assert plan["event"]["checkpoint_sha256"] == \
        "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
    metrics = {item["name"]: item["estimate"] for item in plan["event"]["metrics"]}
    assert metrics["cross_syntax_mean_donor_recovery"] == pytest.approx(0.6272903086)
    assert metrics["minimum_cross_syntax_cell_recovery"] == pytest.approx(0.5038686541)


def test_cross_noun_plan_has_counterfactual_robustness_scope() -> None:
    plan = publish.build_cross_syntax_plan(_cross_noun_spec())
    assert plan["event"]["evaluation_role"] == \
        "SELECT_HELD_OUT_cross_noun_counterfactual_robustness"
    assert plan["event"]["checkpoint_sha256"] == \
        "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
    metrics = {item["name"]: item["estimate"] for item in plan["event"]["metrics"]}
    assert metrics["cross_syntax_mean_donor_recovery"] == pytest.approx(0.6289474868)
    assert metrics["minimum_cross_syntax_cell_recovery"] == pytest.approx(0.5066245814)


def test_residual_ceiling_cannot_be_promoted_as_mechanistic_site() -> None:
    spec = _spec()
    spec["result_site_id"] = "resid:18"
    with pytest.raises(publish.FastScreenPublishError, match="residual ceiling"):
        publish.build_plan(spec)


def test_apply_is_idempotent_and_finishes_one_append_only_prefix(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan(_spec())
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    copied = circuits / RECORD_PATH.name
    shutil.copy2(RECORD_PATH, copied)
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")

    publish.apply_plan(plan)
    first = json.loads(copied.read_text())
    publish.apply_plan(plan)
    second = json.loads(copied.read_text())
    assert first == second
    assert sum(
        event["event_id"] == plan["event"]["event_id"]
        for event in second["evidence_events"]
    ) == 1
    assert sum(
        claim["claim_id"] == plan["claim_revision"]["claim_id"]
        for claim in second["claims"]
    ) == 1


def test_event_id_collision_is_refused_without_adding_a_claim(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan(_spec())
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    copied = circuits / RECORD_PATH.name
    shutil.copy2(RECORD_PATH, copied)
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    publish.apply_plan(plan)
    changed = copy.deepcopy(plan)
    changed["event"]["notes"] += " incompatible mutation"
    with pytest.raises(publish.FastScreenPublishError, match="event id collision"):
        publish.apply_plan(changed)
    record = json.loads(copied.read_text())
    assert sum(claim["claim_id"] == plan["claim_revision"]["claim_id"] for claim in record["claims"]) == 1
