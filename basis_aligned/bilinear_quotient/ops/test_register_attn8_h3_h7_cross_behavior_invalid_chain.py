#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
import shutil

import pytest

import register_attn8_h3_h7_cross_behavior_invalid_chain as publish


def test_plan_is_invalid_only_and_preserves_claim_statuses() -> None:
    plan = publish.build_plan()
    assert len(plan["destinations"]) == 2
    for destination in plan["destinations"]:
        assert len(destination["artifacts"]) == 10
        assert len(destination["events"]) == 4
        namespace = publish.EVENT_NAMESPACES[destination["canonical_tag"]]
        assert all(event["event_id"].startswith(namespace + ".")
                   for event in destination["events"])
        assert all(event["stage"] == event["verdict"] == "invalid"
                   for event in destination["events"])
        assert destination["events"][0]["failure_kind"] == "implementation_failure"
        assert all(event["failure_kind"] == "invalid_instrument"
                   for event in destination["events"][1:])
        claim = destination["claim_revision"]
        if destination["canonical_tag"] == "task.numbered_list.index_successor":
            assert claim["claim_id"] == "numbered_list_index_successor.v10"
            assert claim["status"] == "weights_translated"
        else:
            assert claim["claim_id"] == "numeric_sequence_continuation.v7"
            assert claim["status"] == "specified"
        assert "descriptive only" in claim["next_missing"]
        assert "Do not repair thresholds or select rows" in claim["next_missing"]


def test_decisive_invalid_metrics_are_recomputed() -> None:
    destination = publish.build_plan()["destinations"][0]
    events = destination["events"]
    prefix = publish.EVENT_NAMESPACES[destination["canonical_tag"]] + "."
    metrics = {event["event_id"]: {item["name"]: item["estimate"]
                                  for item in event["metrics"]} for event in events}
    assert metrics[prefix + "attn8_h3_h7_cross_behavior.v2.invalid_implementation.v1"]["minimum_step_two_native_capability"] == 0.0
    assert metrics[prefix + "attn8_h3_h7_cross_behavior.v3.invalid_capability.v1"]["minimum_control_native_capability"] == 0.625
    assert metrics[prefix + "attn8_h3_h7_cross_behavior.v4.invalid_semantic_role.v1"]["semantic_role_registration_valid"] == 0.0
    v5 = metrics[prefix + "attn8_h3_h7_cross_behavior.v5.invalid_ood_instrument.v1"]
    assert v5["minimum_OOD_word_copy_native_capability"] == 0.71875
    assert v5["step_two_direction_specific_donor_answer_win_range"] == [0.0, 1.0]
    assert v5["exact_and_live"] == 1.0


def test_hash_and_schema_mutations_are_rejected() -> None:
    spec = json.loads(publish.SPEC.read_text())
    changed = copy.deepcopy(spec)
    changed["artifacts"][0][2] = "0" * 64
    with pytest.raises(publish.PublicationError, match="hash mismatch"):
        publish.build_plan(changed)
    changed = copy.deepcopy(spec)
    changed["schema"] = "wrong"
    with pytest.raises(publish.PublicationError, match="wrong publication schema"):
        publish.build_plan(changed)


def test_apply_is_idempotent_on_two_isolated_dossiers(tmp_path, monkeypatch) -> None:
    circuits = tmp_path / "circuits"
    circuits.mkdir()
    for destination in json.loads(publish.SPEC.read_text())["destinations"]:
        source = publish.registry.circuit_path(destination["canonical_tag"])
        shutil.copy2(source, circuits / source.name)
    monkeypatch.setattr(publish.registry, "CIRCUITS", circuits)
    monkeypatch.setattr(publish.registry, "REGISTRY", circuits / "registry.json")
    plan = publish.build_plan()
    publish.apply_plan(plan, regenerate=False)
    first = {path.name: path.read_bytes() for path in circuits.glob("*.json")
             if path.name != "registry.json"}
    publish.apply_plan(plan, regenerate=False)
    second = {path.name: path.read_bytes() for path in circuits.glob("*.json")
              if path.name != "registry.json"}
    assert second == first
    for destination in plan["destinations"]:
        record = json.loads((circuits / publish.registry.circuit_path(
            destination["canonical_tag"]).name).read_text())
        assert record["claims"][-1]["claim_id"] == destination["claim_revision"]["claim_id"]
        assert len({event["event_id"] for event in record["evidence_events"]}) == len(record["evidence_events"])


def test_interrupted_dual_apply_is_namespaced_without_dropping_evidence() -> None:
    destination = publish.build_plan()["destinations"][0]
    record = json.loads(publish.registry.circuit_path(
        destination["canonical_tag"]).read_text())
    # Recreate the exact bad intermediate form in memory.
    namespace = publish.EVENT_NAMESPACES[destination["canonical_tag"]]
    for event in record["evidence_events"]:
        if event["event_id"].startswith(namespace + ".attn8_h3_h7_cross_behavior"):
            event["event_id"] = event["event_id"].removeprefix(namespace + ".")
            parent = event.get("supersedes_event_id")
            if parent and parent.startswith(namespace + "."):
                event["supersedes_event_id"] = parent.removeprefix(namespace + ".")
    for claim in record["claims"]:
        claim["evidence_event_ids"] = [item.removeprefix(namespace + ".")
                                       if item.startswith(namespace + ".attn8_h3_h7_cross_behavior")
                                       else item for item in claim["evidence_event_ids"]]
    before = len(record["evidence_events"])
    repaired = publish._migrate_interrupted_dual_apply(record, destination)
    assert len(repaired["evidence_events"]) == before
    assert all(not event["event_id"].startswith("attn8_h3_h7_cross_behavior")
               for event in repaired["evidence_events"])
