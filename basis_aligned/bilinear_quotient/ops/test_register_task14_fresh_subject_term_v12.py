#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_task14_fresh_subject_term_v12 as publish

RECORD = publish.BQ / "circuits/task_subject_verb_number_agreement.json"


def event(plan: dict, suffix: str) -> dict:
    return next(x for x in plan["events"] if x["event_id"].endswith(suffix))


def test_v12_separates_instrument_nulls_and_localization() -> None:
    plan = publish.build_plan()
    assert plan["claim_revision"]["claim_id"] == "grammatical_subject_number.v12"
    assert plan["claim_revision"]["supersedes"] == "grammatical_subject_number.v11"
    assert plan["claim_revision"]["status"] == "site_live"
    assert len(plan["events"]) == 5
    assert event(plan, "instrument.complete.v1")["verdict"] == "held"
    assert event(plan, "interaction_repair.complete.v1")["verdict"] == "null"
    assert event(plan, "complement_independent_use.complete.v1")["verdict"] == "null"
    assert event(plan, "complement_asymmetry.complete.v1")["verdict"] == "null"
    held = event(plan, "exact_localization.complete.v1")
    assert held["verdict"] == "held"
    assert "does not establish value-vector semantics" in held["notes"]
    assert "value side" in plan["claim_revision"]["next_missing"]
    assert "downstream readers" in plan["claim_revision"]["next_missing"]


def test_localization_metrics_are_recomputed_from_cells() -> None:
    plan = publish.build_plan()
    metrics = {x["name"]: x["estimate"] for x in event(plan, "exact_localization.complete.v1")["metrics"]}
    assert metrics["subject_over_complete_mean_margin_range"] == pytest.approx([0.9558769831, 1.0982569364])
    assert metrics["subject_over_complete_mean_CE_range"] == pytest.approx([0.9542493737, 1.1033600160])
    assert metrics["maximum_absolute_complement_mean_margin"] == pytest.approx(0.0372903347)
    assert metrics["maximum_absolute_complement_mean_CE"] == pytest.approx(0.0389773846)
    assert metrics["maximum_absolute_interaction_mean_margin"] == pytest.approx(0.0030497313)
    assert metrics["maximum_absolute_interaction_mean_CE"] == pytest.approx(0.0023448467)


def test_exact_v11_prefix_and_global_event_ids(tmp_path, monkeypatch) -> None:
    plan = publish.build_plan()
    before = json.loads(RECORD.read_text())
    # The repository may contain either the exact v11 migration source or the
    # already-applied v12 record.  In both cases validate the immutable v11
    # prefix and the exact v12 evidence suffix.
    if before["claims"][-1]["claim_id"] == publish.NEW_CLAIM:
        expected_v11_claims = before["claims"][:-1]
        expected_v11_event_ids = expected_v11_claims[-1]["evidence_event_ids"]
    else:
        expected_v11_claims = before["claims"]
        expected_v11_event_ids = before["claims"][-1]["evidence_event_ids"]
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    copied = circuits / RECORD.name
    shutil.copy2(RECORD, copied)
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    publish.apply_plan(plan, regenerate=False)
    first = copied.read_bytes()
    publish.apply_plan(plan, regenerate=False)
    assert copied.read_bytes() == first
    after = json.loads(first)
    assert after["claims"][:-1] == expected_v11_claims
    assert after["claims"][-1]["evidence_event_ids"] == (
        expected_v11_event_ids + [x["event_id"] for x in plan["events"]]
    )
    ids = [x["event_id"] for x in after["evidence_events"]]
    assert len(ids) == len(set(ids))


def test_literal_artifact_hash_mutation_is_rejected(monkeypatch) -> None:
    changed = copy.deepcopy(publish.ARTIFACT_SPECS)
    path, _digest, kind = changed["task14_fresh_subject_term_result"]
    changed["task14_fresh_subject_term_result"] = (path, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACT_SPECS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan()
