#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json

import pytest

import register_narrative_tense_results as publish


def test_record_is_existing_registry_v2_schema_and_claim_is_bounded() -> None:
    record = publish.build_record()
    publish.registry.validate_v2(record)
    assert record["tag"] == "task.narrative_tense.past_vs_present"
    claim = record["claims"][-1]
    assert claim["status"] == "site_live"
    assert claim["next_missing"] == publish.NEXT_MISSING
    assert "predeclare A1 template capability selection on FIT" in claim["next_missing"]
    assert "untouched construction holdout" in claim["next_missing"]
    assert "only if the selected authority is capable" in claim["next_missing"]
    assert [claim["claim_id"] for claim in record["claims"]] == [
        "narrative_tense_at_final_position.v1", "narrative_tense_at_final_position.v2",
        "narrative_tense_at_final_position.v3",
    ]
    assert record["claims"][-1]["supersedes"] == record["claims"][1]["claim_id"]
    assert record["claims"][0]["evidence_event_ids"] == [
        event["event_id"] for event in record["evidence_events"][:5]
    ]
    assert record["claims"][1]["evidence_event_ids"] == [
        event["event_id"] for event in record["evidence_events"][:6]
    ]
    assert record["claims"][-1]["evidence_event_ids"] == [
        event["event_id"] for event in record["evidence_events"]
    ]
    assert len(record["evidence_events"]) == 7
    assert [event["verdict"] for event in record["evidence_events"]] == [
        "invalid", "held", "invalid", "invalid", "held", "invalid", "invalid"
    ]
    assert all(claim["status"] == "site_live" for claim in record["claims"])


def test_new_event_is_invalid_only_and_preserves_descriptive_results() -> None:
    event = publish.build_record()["evidence_events"][-2]
    assert event["claim_id"] == "narrative_tense_at_final_position.v2"
    assert event["stage"] == event["verdict"] == "invalid"
    assert event["failure_kind"] == "invalid_instrument"
    metrics = {metric["name"]: metric for metric in event["metrics"]}
    assert metrics["installed_noop_max_absolute_error"]["estimate"] == 0.0000591278076171875
    assert metrics["installed_noop_max_absolute_error"]["bar"] == "<=0.00005"
    assert metrics["C_R_joint_mean_absolute_normalized_movement"]["estimate"] == 0.05991756523480643
    assert metrics["C_complete_H3_mean_absolute_normalized_movement"]["estimate"] == 0.0589810113272569
    assert event["notes"]["descriptive_R_target_recovery"].endswith("explicitly not evidence")
    assert event["notes"]["cross_task_results"]["is_payload_transfer_passed"] is False
    assert event["notes"]["cross_task_results"]["was_payload_transfer_passed"] is False


def test_fresh_event_is_capability_invalid_with_exactness_only() -> None:
    event = publish.build_record()["evidence_events"][-1]
    assert event["claim_id"] == "narrative_tense_at_final_position.v3"
    assert event["stage"] == event["verdict"] == "invalid"
    assert event["failure_kind"] == "invalid_instrument"
    metrics = {metric["name"]: metric["estimate"] for metric in event["metrics"]}
    assert metrics["A1_past_minimum_native_capability"] == 0.75
    assert metrics["source_sum_max_absolute_error"] == 0.0
    assert metrics["same_batch_native_reinstall_max_absolute_error"] == 0.0
    assert metrics["pre_first_change_install_max_absolute_error"] == 0.0000171661376953125
    assert event["notes"]["descriptive_only"] == [
        "R-joint target recovery", "R effective-value target recovery",
        "post-last-change effective-value concentration", "P/C selectivity measurements",
    ]


def test_all_requested_artifacts_are_hash_bound() -> None:
    record = publish.build_record()
    assert set(record["artifacts"]) == set(publish.ARTIFACTS)
    for artifact_id, (_, expected, _) in publish.ARTIFACTS.items():
        assert record["artifacts"][artifact_id]["sha256"] == expected
        assert record["artifacts"][artifact_id]["status"] == "frozen"


def test_hash_drift_is_rejected(monkeypatch) -> None:
    changed = dict(publish.ARTIFACTS)
    relative, _, kind = changed["head3_complement_result"]
    changed["head3_complement_result"] = (relative, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACTS", changed)
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_record()


def test_event_ids_are_globally_namespaced_and_new() -> None:
    record = publish.build_record()
    event_ids = {event["event_id"] for event in record["evidence_events"]}
    assert len(event_ids) == len(record["evidence_events"])
    assert all(event_id.startswith("narrative_tense.") for event_id in event_ids)
    existing_ids = set()
    for path in publish.registry.CIRCUITS.glob("task_*.json"):
        if path.name == publish.OUTPUT_NAME:
            continue
        document = json.loads(path.read_text())
        existing_ids.update(event["event_id"] for event in document.get("evidence_events", []))
    assert event_ids.isdisjoint(existing_ids)


def test_apply_is_idempotent_in_temporary_registry(tmp_path, monkeypatch) -> None:
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    expected = publish.build_record()
    path = publish.apply_record(expected)
    first = path.read_bytes()
    publish.apply_record(expected)
    assert path.read_bytes() == first
    compact = json.loads((circuits / "registry.json").read_text())
    assert compact["circuits"][publish.TAG]["active_claim_id"] == expected["claims"][-1]["claim_id"]


def test_exact_v2_prefix_migrates_once_then_is_idempotent(tmp_path, monkeypatch) -> None:
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    expected = publish.build_record()
    base = copy.deepcopy(expected)
    base["claims"] = base["claims"][:2]
    base["claims"][-1]["evidence_event_ids"] = base["claims"][-1]["evidence_event_ids"][:6]
    base["evidence_events"] = base["evidence_events"][:6]
    base["artifacts"].pop("fresh_unchanged_carrier_prior_art")
    base["artifacts"].pop("fresh_unchanged_carrier_invalid_result")
    path = publish.registry.circuit_path(publish.TAG)
    path.write_text(json.dumps(base))
    publish.apply_record(expected)
    first = path.read_bytes()
    assert json.loads(first)["claims"][-1]["claim_id"] == "narrative_tense_at_final_position.v3"
    publish.apply_record(expected)
    assert path.read_bytes() == first


def test_v1_prefix_is_not_an_allowed_migration_anymore(tmp_path, monkeypatch) -> None:
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    expected = publish.build_record()
    v1 = copy.deepcopy(expected)
    v1["claims"] = v1["claims"][:1]
    v1["claims"][0]["evidence_event_ids"] = v1["claims"][0]["evidence_event_ids"][:5]
    v1["evidence_events"] = v1["evidence_events"][:5]
    for artifact_id in (
        "source_route_cross_task_prior_art", "source_route_cross_task_invalid_result",
        "fresh_unchanged_carrier_prior_art", "fresh_unchanged_carrier_invalid_result",
    ):
        v1["artifacts"].pop(artifact_id)
    publish.registry.circuit_path(publish.TAG).write_text(json.dumps(v1))
    with pytest.raises(publish.PublicationError, match="canonical record differs"):
        publish.apply_record(expected)


def test_existing_different_record_is_refused(tmp_path, monkeypatch) -> None:
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    record = publish.build_record()
    path = publish.registry.circuit_path(publish.TAG)
    changed = copy.deepcopy(record)
    changed["claims"][-1]["next_missing"] = "different"
    path.write_text(json.dumps(changed))
    with pytest.raises(publish.PublicationError, match="canonical record differs"):
        publish.apply_record(record)
