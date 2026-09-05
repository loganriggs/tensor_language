#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_task14_head11_3_below_head_results as publish


RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"
V10_SPEC = publish.RESULT_DIR / "task14_head11_3_below_head_v10_publication.json"


def test_plan_binds_complete_below_head_lineage_and_v9() -> None:
    plan = publish.build_plan()
    assert len(plan["events"]) == 15
    assert len(plan["artifacts"]) == 30
    assert sum(event["verdict"] == "invalid" for event in plan["events"]) == 4
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v9"
    assert plan["claim_revision"]["status"] == "site_live"
    sites = {site["site_id"] for site in plan["claim_revision"]["candidate_sites"]}
    assert {event["site_id"] for event in plan["events"]} <= sites
    qk = next(event for event in plan["events"] if ".ood_self_qk." in event["event_id"])
    assert all(">=0.70" in item["bar"] for item in qk["metrics"][:2])
    natural = next(event for event in plan["events"] if ".ood_natural_qk_specificity." in event["event_id"])
    assert natural["metrics"][0]["estimate"] == pytest.approx([0.005757905072626622, 0.057294400041296364])


def test_literal_hash_and_terminal_mutations_are_rejected() -> None:
    spec = json.loads(publish.SPEC.read_text())
    changed = copy.deepcopy(spec)
    changed["entries"][0][1] = "0" * 64
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan(changed)
    changed = copy.deepcopy(spec)
    changed["entries"][0][3] = "invented_terminal"
    with pytest.raises(publish.PublicationError, match="schema/terminal mismatch"):
        publish.build_plan(changed)


def test_v10_records_only_fresh_invalid_instrument_provenance() -> None:
    plan = publish.build_plan(json.loads(V10_SPEC.read_text()))
    assert len(plan["events"]) == 1
    assert len(plan["artifacts"]) == 2
    event = plan["events"][0]
    assert event["stage"] == event["verdict"] == "invalid"
    assert event["failure_kind"] == "invalid_instrument"
    assert event["evaluation_role"] == "FRESH_TEXT"
    assert "not scientific evidence" in event["notes"]
    metrics = {m["name"]: m for m in event["metrics"]}
    assert metrics["minimum_opposite_joint_expected_row_sign_count_of_8"]["estimate"] == 4
    assert metrics["minimum_native_correct_count_of_8"]["estimate"] == 8
    assert metrics["native_replay_max_absolute_logit_error"]["estimate"] == 0.0
    assert metrics["direct_score_identity_max_absolute_error"]["estimate"] == pytest.approx(5.21540641784668e-08)
    claim = plan["claim_revision"]
    assert claim["claim_id"] == "grammatical_subject_number.v10"
    assert claim["revision"] == 10
    assert claim["supersedes"] == "grammatical_subject_number.v9"
    assert claim["evidence_event_ids"][-1] == event["event_id"]
    assert "Fresh-text natural-QK confirmation remains missing" in claim["next_missing"]


def test_decisive_metrics_are_reduced_from_cell_scores() -> None:
    path = publish.RESULT_DIR / publish.RESULT_PATHS["subject_value_test"]
    result = json.loads(path.read_text())
    metrics = {m["name"]: m["estimate"] for m in publish.decisive_metrics("subject_value_test", result)}
    assert metrics["passing_subject_value_cells"] == 3
    assert metrics["minimum_subject_value_margin_recovery"] == pytest.approx(0.18667551116735692)
    qk_path = publish.RESULT_DIR / publish.RESULT_PATHS["ood_self_qk"]
    qk = {m["name"]: m["estimate"] for m in publish.decisive_metrics("ood_self_qk", json.loads(qk_path.read_text()))}
    assert qk["branch_interaction_absolute_fraction_range"] == pytest.approx([0.15426219152005102, 0.17771235940787158])


def test_apply_is_idempotent_on_isolated_record(tmp_path, monkeypatch) -> None:
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    copied = circuits / RECORD.name
    shutil.copy2(RECORD, copied)
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    plan = publish.build_plan()
    publish.apply_plan(plan, regenerate=False)
    first = copied.read_bytes()
    publish.apply_plan(plan, regenerate=False)
    assert copied.read_bytes() == first
    record = json.loads(first)
    assert record["claims"][-1]["claim_id"] == "grammatical_subject_number.v9"
    assert len({event["event_id"] for event in record["evidence_events"]}) == len(record["evidence_events"])
    v10 = publish.build_plan(json.loads(V10_SPEC.read_text()))
    publish.apply_plan(v10, regenerate=False)
    second = copied.read_bytes()
    publish.apply_plan(v10, regenerate=False)
    assert copied.read_bytes() == second
    record = json.loads(second)
    assert record["claims"][-1]["claim_id"] == "grammatical_subject_number.v10"
