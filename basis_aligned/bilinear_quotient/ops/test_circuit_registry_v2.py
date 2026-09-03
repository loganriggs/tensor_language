import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


BQ = Path(__file__).resolve().parents[1]
REPO = BQ.parents[1]
sys.path.insert(0, str(BQ))
import circuit_registry_v2 as registry  # noqa: E402


TASK_TAGS = {
    "task.induction.selector_payload",
    "task.bracket.pending_opener",
    "task.successor.pointer",
    "task.increment.state",
    "subroutine.induction.equality_score",
}


def task_records():
    compact = json.loads(registry.REGISTRY.read_text())["circuits"]
    return {
        tag: json.loads((registry.CIRCUITS / compact[tag]["file"]).read_text())
        for tag in TASK_TAGS
    }


def test_all_task_records_validate_and_bind_existing_artifacts():
    for record in task_records().values():
        registry.validate_v2(record)
        for artifact in record["artifacts"].values():
            path = REPO / artifact["path"]
            assert path.is_file(), artifact["path"]
            assert artifact["status"] == "frozen"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_registry_contains_only_tagged_records_not_aggregate_json():
    compact = json.loads(registry.REGISTRY.read_text())["circuits"]
    assert TASK_TAGS <= set(compact)
    for row in compact.values():
        document = json.loads((registry.CIRCUITS / row["file"]).read_text())
        assert document.get("tag")
    assert "BATTERY" not in compact
    assert "REPERTOIRE" not in compact


def test_design_reuse_is_explicit_and_execution_keys_are_unique_and_recomputable():
    for record in task_records().values():
        designs = {}
        executions = []
        for event in record["evidence_events"]:
            assert registry.design_key(record, event) == event["design_key"]
            assert registry.execution_key(record, event) == event["execution_key"]
            prior_ids = designs.setdefault(event["design_key"], set())
            if prior_ids:
                assert event.get("supersedes_event_id") in prior_ids or (
                    event.get("replicates_event_id") in prior_ids
                )
            prior_ids.add(event["event_id"])
            executions.append(event["execution_key"])
        assert len(executions) == len(set(executions))


def test_changed_event_contract_invalidates_design_key():
    record = task_records()["task.bracket.pending_opener"]
    broken = copy.deepcopy(record)
    broken["claims"][0]["counterfactual_families"][0]["holds_fixed"].append("new confound")
    with pytest.raises(AssertionError):
        registry.validate_v2(broken)


def test_bad_tree_alias_cannot_be_attached_to_behavior_identity():
    record = copy.deepcopy(task_records()["task.increment.state"])
    record["identity"]["kind"] = "census_slice"
    record["identity"]["instance"] = None
    with pytest.raises(AssertionError):
        registry.validate_v2(record)


def test_append_artifacts_is_idempotent_and_refuses_hash_drift(tmp_path, monkeypatch):
    record = copy.deepcopy(task_records()["task.increment.state"])
    record["tag"] = "task.test.append_artifact"
    record["artifacts"]["new_result"] = {
        "path": "result.json", "sha256": "a" * 64,
        "kind": "result", "status": "frozen",
    }
    path = tmp_path / "task_test_append_artifact.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(registry, "CIRCUITS", tmp_path)
    monkeypatch.setattr(registry, "REGISTRY", tmp_path / "registry.json")
    value = record["artifacts"]["new_result"]
    registry.append_artifacts(record["tag"], {"new_result": value})
    assert json.loads(path.read_text())["artifacts"]["new_result"] == value
    changed = dict(value, sha256="b" * 64)
    with pytest.raises(ValueError, match="artifact id collision"):
        registry.append_artifacts(record["tag"], {"new_result": changed})


def test_registry_distinguishes_active_from_historical_invalid_events():
    compact = json.loads(registry.REGISTRY.read_text())["circuits"]
    bracket = compact["task.bracket.pending_opener"]
    # The original unverified-checkpoint event is historical because R538v2
    # supersedes it.  R540's selectivity null and R542's statistical-unit
    # correction are active negative evidence and must remain visible.
    assert bracket["negative_event_count"] == 3
    assert bracket["active_negative_event_count"] == 2
    assert bracket["latest_blocker"] == "pending_opener_split_integrity.r542.invalid_statistical_unit.v1"
    record = task_records()["task.bracket.pending_opener"]
    superseded = {event.get("supersedes_event_id") for event in record["evidence_events"]}
    active_ids = {event["event_id"] for event in record["evidence_events"]
                  if event["event_id"] not in superseded}
    assert bracket["latest_active_event"] in active_ids
    assert "invalid_unverified_checkpoint" not in bracket["latest_active_event"]
