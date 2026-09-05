#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json

import pytest

import register_induction_native_capability_results_rung580_r587 as publish


def test_v10_closes_r580_and_records_clean_replication_and_audit() -> None:
    record = publish.build_record()
    publish.registry.validate_v2(record)
    claim = record["claims"][-1]
    assert claim["claim_id"] == publish.NEW_CLAIM
    assert claim["supersedes"] == publish.OLD_CLAIM
    assert claim["status"] == "specified"
    assert "do not repeat the native-capability screen" in claim["next_missing"]
    invalid, held, audit = record["evidence_events"][-3:]
    assert invalid["event_id"] == publish.INVALID_EVENT
    assert invalid["stage"] == invalid["verdict"] == "invalid"
    assert invalid["failure_kind"] == "invalid_instrument"
    assert invalid["supersedes_event_id"] == publish.OPEN_EVENT
    assert held["event_id"] == publish.HELD_EVENT
    assert held["stage"] == "complete" and held["verdict"] == "held"
    assert held["supersedes_event_id"] == publish.INVALID_EVENT
    assert held["replicates_event_id"] == publish.INVALID_EVENT
    assert audit["event_id"] == publish.AUDIT_EVENT
    assert audit["test_type"] == "null_control" and audit["verdict"] == "held"
    superseded = {
        event["supersedes_event_id"] for event in record["evidence_events"]
        if event.get("supersedes_event_id")
    }
    active_open = [
        event["event_id"] for event in record["evidence_events"]
        if event["event_id"] not in superseded
        and event["stage"] == "preregistered" and event["verdict"] == "inconclusive"
    ]
    assert publish.OPEN_EVENT not in active_open


def test_exact_scientific_terminal_metrics_and_scope_are_preserved() -> None:
    invalid, held, audit = publish.build_record()["evidence_events"][-3:]
    invalid_metrics = {metric["name"]: metric["estimate"] for metric in invalid["metrics"]}
    assert invalid_metrics == {
        "scientific_capability_predicates": 3,
        "independent_full_envelope_audit": 0,
    }
    assert "one-item JSON list" in invalid["notes"]
    held_metrics = {metric["name"]: metric["estimate"] for metric in held["metrics"]}
    assert held_metrics["scientific_capability_predicates"] == 3
    assert held_metrics["unique_sequences"] == 3024
    assert held_metrics["execution_envelope"] == 95
    assert held_metrics["scalar_next_step"] == 1.0
    assert "FINAL_TEST/OOD stayed closed" in held["notes"]
    audit_metrics = {metric["name"]: metric["estimate"] for metric in audit["metrics"]}
    assert audit_metrics["raw_rows_recomputed"] == 3240
    assert audit_metrics["factorial_groups_recomputed"] == 108
    assert audit_metrics["bootstrap_cells_recomputed"] == 86
    assert audit_metrics["result_receipt_binding"] == 1.0
    assert "zero model calls" in audit["notes"]


def test_all_terminal_artifacts_are_exactly_hash_bound() -> None:
    record = publish.build_record()
    for artifact_id, (_, expected, _) in publish.ARTIFACTS.items():
        artifact = record["artifacts"][artifact_id]
        assert artifact["sha256"] == expected
        assert artifact["status"] == "frozen"


def test_hash_drift_is_rejected(monkeypatch) -> None:
    changed = dict(publish.ARTIFACTS)
    path, _, kind = changed["r586_capability_result"]
    changed["r586_capability_result"] = (path, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACTS", changed)
    with pytest.raises(publish.PublicationError, match="artifact hash mismatch"):
        publish.build_record()


def test_exact_v9_prefix_migrates_once_then_is_idempotent(tmp_path, monkeypatch) -> None:
    expected = publish.build_record()
    base = copy.deepcopy(expected)
    base["claims"] = base["claims"][:-1]
    base["evidence_events"] = base["evidence_events"][:-3]
    for artifact_id in publish.ARTIFACTS:
        base["artifacts"].pop(artifact_id)
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    path = circuits / "task_induction_selector_payload.json"
    path.write_text(json.dumps(base, indent=1) + "\n")
    assert publish._sha256(path) == publish.BASE_SHA256
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    publish.apply_record(expected, regenerate=False)
    first = path.read_bytes()
    publish.apply_record(expected, regenerate=False)
    assert path.read_bytes() == first


def test_build_record_against_applied_v10_is_idempotent(tmp_path, monkeypatch) -> None:
    expected = publish.build_record()
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    path = circuits / "task_induction_selector_payload.json"
    path.write_text(json.dumps(expected, indent=1) + "\n")
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    assert publish.build_record() == expected
    publish.apply_record(regenerate=False)
    assert json.loads(path.read_text()) == expected


def test_nonexact_existing_record_is_refused(tmp_path, monkeypatch) -> None:
    expected = publish.build_record()
    base = copy.deepcopy(expected)
    base["claims"] = base["claims"][:-1]
    base["evidence_events"] = base["evidence_events"][:-3]
    for artifact_id in publish.ARTIFACTS:
        base["artifacts"].pop(artifact_id)
    base["claims"][-1]["next_missing"] = "changed"
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    path = circuits / "task_induction_selector_payload.json"
    path.write_text(json.dumps(base, indent=1) + "\n")
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    with pytest.raises(publish.PublicationError, match="exact v9 prefix"):
        publish.apply_record(expected, regenerate=False)


def test_new_event_ids_are_globally_namespaced_and_unique() -> None:
    record = publish.build_record()
    event_ids = [event["event_id"] for event in record["evidence_events"]]
    assert len(event_ids) == len(set(event_ids))
    assert publish.INVALID_EVENT in event_ids
    assert publish.HELD_EVENT in event_ids
    assert publish.AUDIT_EVENT in event_ids
