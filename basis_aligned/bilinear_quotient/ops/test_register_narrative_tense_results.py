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
    assert "source-score and cached-value" in claim["next_missing"]
    assert "output-token confound control" in claim["next_missing"]
    assert len(record["evidence_events"]) == 5
    assert [event["verdict"] for event in record["evidence_events"]] == [
        "invalid", "held", "invalid", "invalid", "held"
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
