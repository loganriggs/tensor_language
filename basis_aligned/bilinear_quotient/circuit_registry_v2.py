"""Lightweight, CPU-only I/O for version-2 circuit evidence records.

The model-facing ``census_lib`` re-exports these functions.  This module exists so
registry maintenance never imports a checkpoint or allocates GPU memory.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CIRCUITS = ROOT / "circuits"
REGISTRY = CIRCUITS / "registry.json"

CLAIM_STATUSES = {
    "proposed", "specified", "site_live", "activation_identified",
    "weights_translated", "adopted", "rejected", "superseded",
}
FAMILY_ROLES = {"interchange", "necessity", "invariance"}
FAMILY_STATUSES = {"proposed", "frozen", "validated", "failed"}
EVENT_STAGES = {"preregistered", "complete", "invalid"}
EVENT_VERDICTS = {"held", "failed", "null", "inconclusive", "invalid"}
TEST_TYPES = {
    "capability", "full_swap_ceiling", "das_interchange", "cross_family_transfer",
    "removal", "invariance", "composition", "ood", "seed_stability",
    "compiled_equivalence", "null_control",
}
FAILURE_KINDS = {
    None, "scientific_null", "insufficient_power", "invalid_instrument",
    "implementation_failure",
}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def circuit_path(tag: str) -> Path:
    if not tag or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in tag):
        raise ValueError(f"unsafe circuit tag: {tag!r}")
    return CIRCUITS / f"{tag.replace('.', '_')}.json"


@contextmanager
def _lock(name: str):
    CIRCUITS.mkdir(parents=True, exist_ok=True)
    with (CIRCUITS / f".{name}.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _atomic_json(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o664)
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_v2(record: dict) -> None:
    assert record["schema_version"] == 2
    assert record["tag"]
    identity = record["identity"]
    assert identity["kind"] in {"census_slice", "behavior_circuit", "shared_subroutine"}
    assert "instance" in identity and identity["identity_artifact_id"]
    if identity["kind"] == "census_slice":
        assert identity["instance"]
    elif identity["kind"] == "behavior_circuit":
        assert identity["instance"] is None
    assert isinstance(identity.get("aliases", []), list)
    artifact_ids = set(record["artifacts"])
    split_ids = {item["split_plan_id"] for item in record.get("split_plans", [])}
    assert len(split_ids) == len(record.get("split_plans", []))
    claim_ids = set()
    event_ids = set()
    for claim in record["claims"]:
        assert claim["claim_id"] not in claim_ids
        claim_ids.add(claim["claim_id"])
        assert claim["status"] in CLAIM_STATUSES
        variable = claim["causal_variable"]
        assert all(variable.get(key) for key in ("id", "domain", "read", "operation", "write", "endpoint"))
        family_ids = set()
        for family in claim["counterfactual_families"]:
            assert family["family_id"] not in family_ids
            family_ids.add(family["family_id"])
            assert family["role"] in FAMILY_ROLES
            assert family["status"] in FAMILY_STATUSES
            assert len(family["changes"]) >= 1 and len(family["holds_fixed"]) >= 2
            assert len(family["control_ids"]) >= 2
            if family.get("builder_artifact_id"):
                assert family["builder_artifact_id"] in artifact_ids
            if family.get("split_plan_id"):
                assert family["split_plan_id"] in split_ids
        assert set(claim.get("split_plan_ids", [])) <= split_ids
        for site in claim["candidate_sites"]:
            assert all(site.get(key) for key in ("site_id", "tensor_path", "shape", "intervention"))
    for event in record.get("evidence_events", []):
        assert event["event_id"] not in event_ids
        event_ids.add(event["event_id"])
        assert event["claim_id"] in claim_ids
        assert event["test_type"] in TEST_TYPES
        assert event["stage"] in EVENT_STAGES
        assert event["verdict"] in EVENT_VERDICTS
        assert event.get("failure_kind") in FAILURE_KINDS
        claim = _claim(record, event["claim_id"])
        known_families = {family["family_id"] for family in claim["counterfactual_families"]}
        assert set(event.get("family_ids", [])) <= known_families
        if event.get("split_plan_id"):
            assert event["split_plan_id"] in split_ids
        for key in ("prereg_artifact_id", "result_artifact_id"):
            if event.get(key):
                assert event[key] in artifact_ids
        assert set(event.get("input_artifact_ids", [])) <= artifact_ids
        assert event["design_key"] == design_key(record, event)
        assert event["execution_key"] == execution_key(record, event)
    for claim in record["claims"]:
        assert set(claim.get("evidence_event_ids", [])) <= event_ids
    for artifact in record["artifacts"].values():
        assert artifact["status"] in {"frozen", "legacy_unhashed", "missing"}
        if artifact["status"] == "frozen":
            assert len(artifact.get("sha256", "")) == 64


def _claim(record: dict, claim_id: str) -> dict:
    return next(claim for claim in record["claims"] if claim["claim_id"] == claim_id)


def design_key(record: dict, event: dict) -> str:
    claim = _claim(record, event["claim_id"])
    families = {
        family["family_id"]: family for family in claim["counterfactual_families"]
        if family["family_id"] in event.get("family_ids", [])
    }
    metric_contract = [
        {key: metric.get(key) for key in ("name", "bar")}
        for metric in event.get("metrics", [])
    ]
    return _canonical_hash({
        "claim_id": claim["claim_id"],
        "causal_variable_id": claim["causal_variable"]["id"],
        "families": {
            key: {
                "role": value["role"],
                "changes": value["changes"],
                "holds_fixed": value["holds_fixed"],
                "control_ids": value["control_ids"],
            }
            for key, value in sorted(families.items())
        },
        "site_id": event.get("site_id"),
        "test_type": event["test_type"],
        "metric_contract": metric_contract,
    })


def execution_key(record: dict, event: dict) -> str:
    split = next(
        (item for item in record.get("split_plans", []) if item["split_plan_id"] == event.get("split_plan_id")),
        None,
    )
    artifacts = record["artifacts"]
    bound = {}
    for key in ("prereg_artifact_id", "result_artifact_id"):
        artifact_id = event.get(key)
        if artifact_id:
            bound[artifact_id] = artifacts[artifact_id].get("sha256")
    for artifact_id in event.get("input_artifact_ids", []):
        bound[artifact_id] = artifacts[artifact_id].get("sha256")
    if split:
        for key in ("partition_artifact_id", "builder_artifact_id"):
            artifact_id = split.get(key)
            if artifact_id:
                bound[artifact_id] = artifacts[artifact_id].get("sha256")
    return _canonical_hash({
        "design_key": event["design_key"],
        "split": split,
        "seed": event.get("seed"),
        "checkpoint_sha256": event.get("checkpoint_sha256"),
        "artifacts": bound,
    })


def rebuild_registry_v2() -> dict:
    """Generate the compact registry from tagged circuit records only."""
    with _lock("registry"):
        rows = {}
        for path in sorted(CIRCUITS.glob("*.json")):
            if path.name == REGISTRY.name:
                continue
            try:
                document = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            tag = document.get("tag")
            if not tag:
                continue
            row = {
                "file": path.name,
                "schema_version": document.get("schema_version", 1),
                "n_members": document.get("members", {}).get("n"),
                "headline": document.get("story", {}).get("blind_name", ""),
                "cert": len(document.get("certification", [])),
                "record_sha256": file_sha256(path),
            }
            if document.get("schema_version") == 2:
                active = [claim for claim in document["claims"] if claim["status"] != "superseded"]
                claim = active[-1] if active else document["claims"][-1]
                events = document.get("evidence_events", [])
                superseded_event_ids = {
                    event.get("supersedes_event_id") for event in events
                    if event.get("supersedes_event_id")
                }
                active_events = [event for event in events if event["event_id"] not in superseded_event_ids]
                held_test_types = sorted({
                    event["test_type"] for event in active_events if event["verdict"] == "held"
                })
                blocking_test_types = sorted({
                    event["test_type"] for event in active_events
                    if event["verdict"] in {"failed", "null", "invalid"}
                })
                row.update({
                    "kind": document["identity"]["kind"],
                    "identity_instance": document["identity"]["instance"],
                    "active_claim_id": claim["claim_id"],
                    "claim_status": claim["status"],
                    "causal_variable_id": claim["causal_variable"]["id"],
                    "interchange_family_count": sum(f["role"] == "interchange" for f in claim["counterfactual_families"]),
                    "invariance_family_count": sum(f["role"] == "invariance" for f in claim["counterfactual_families"]),
                    "negative_event_count": sum(e["verdict"] in {"failed", "null", "invalid"} for e in events),
                    "active_negative_event_count": sum(
                        e["verdict"] in {"failed", "null", "invalid"} for e in active_events),
                    "latest_blocker": next((
                        e["event_id"] for e in reversed(active_events)
                        if e["verdict"] in {"failed", "null", "invalid"}), None),
                    "latest_active_event": active_events[-1]["event_id"] if active_events else None,
                    "active_event_ids": [event["event_id"] for event in active_events],
                    "held_test_types": held_test_types,
                    "active_blocking_test_types": blocking_test_types,
                    "design_keys": sorted({event["design_key"] for event in events}),
                })
            rows[tag] = row
        registry = {"schema": "generated_circuit_registry_v2", "circuits": rows}
        _atomic_json(REGISTRY, registry)
    return registry


def write_behavior_circuit(record: dict) -> Path:
    """Write a complete v2 behavior record and regenerate the compact registry."""
    validate_v2(record)
    path = circuit_path(record["tag"])
    with _lock("registry"):
        if path.exists():
            old = json.loads(path.read_text())
            if old != record:
                raise FileExistsError(f"refusing to overwrite existing record {record['tag']}")
        _atomic_json(path, record)
    rebuild_registry_v2()
    return path


def append_claim_revision(
    tag: str,
    claim: dict,
    *,
    artifacts: dict[str, dict] | None = None,
    split_plans: list[dict] | None = None,
) -> Path:
    """Append a claim revision and its pre-outcome authorities without mutating history."""
    path = circuit_path(tag)
    with _lock("registry"):
        record = json.loads(path.read_text())
        old_claims = record["claims"]
        assert all(item["claim_id"] != claim["claim_id"] for item in old_claims)
        assert claim["supersedes"] in {item["claim_id"] for item in old_claims}
        assert claim["revision"] > max(item["revision"] for item in old_claims)
        for artifact_id, value in (artifacts or {}).items():
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
                raise ValueError(f"artifact id collision: {artifact_id}")
            record["artifacts"][artifact_id] = value
        existing_splits = {item["split_plan_id"]: item for item in record.get("split_plans", [])}
        for split in split_plans or []:
            split_id = split["split_plan_id"]
            if split_id in existing_splits and existing_splits[split_id] != split:
                raise ValueError(f"split-plan id collision: {split_id}")
            if split_id not in existing_splits:
                record.setdefault("split_plans", []).append(split)
        old_claims.append(claim)
        validate_v2(record)
        _atomic_json(path, record)
    rebuild_registry_v2()
    return path


def append_artifacts(tag: str, artifacts: dict[str, dict]) -> Path:
    """Append frozen artifact authorities without rewriting claims or events.

    Result files exist before their completion event can bind them, so outcome
    registration needs a small atomic step that does not manufacture a new
    causal claim merely to add a hash.  Existing identical entries are
    idempotent; a changed artifact under the same id is always refused.
    """
    path = circuit_path(tag)
    with _lock("registry"):
        record = json.loads(path.read_text())
        for artifact_id, value in artifacts.items():
            if artifact_id in record["artifacts"] and record["artifacts"][artifact_id] != value:
                raise ValueError(f"artifact id collision: {artifact_id}")
            record["artifacts"][artifact_id] = value
        validate_v2(record)
        _atomic_json(path, record)
    rebuild_registry_v2()
    return path


def append_evidence_event(tag: str, event: dict) -> Path:
    """Append one immutable event, rejecting duplicate designs unless superseded."""
    path = circuit_path(tag)
    with _lock("registry"):
        record = json.loads(path.read_text())
        event = dict(event)
        event["design_key"] = design_key(record, event)
        event["execution_key"] = execution_key(record, event)
        old_events = record.setdefault("evidence_events", [])
        assert all(item["event_id"] != event["event_id"] for item in old_events)
        collisions = [item for item in old_events if item["design_key"] == event["design_key"]]
        if collisions:
            by_id = {item["event_id"]: item for item in collisions}
            referenced_id = event.get("supersedes_event_id") or event.get("replicates_event_id")
            if referenced_id not in by_id:
                raise ValueError(
                    "duplicate design; explicitly supersede or replicate one of "
                    + ", ".join(sorted(by_id))
                )
            if event.get("replicates_event_id") and event["execution_key"] == by_id[referenced_id]["execution_key"]:
                raise ValueError("a replication must change the split, seed, or another execution-bound input")
        old_events.append(event)
        validate_v2(record)
        _atomic_json(path, record)
    rebuild_registry_v2()
    return path
