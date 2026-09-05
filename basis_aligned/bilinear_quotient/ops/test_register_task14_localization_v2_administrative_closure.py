#!/usr/bin/env python3
# BQLANE: cpu

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

import register_task14_localization_v2_administrative_closure as publish


def test_closure_is_only_an_implementation_terminal() -> None:
    canonical_path = publish.registry.circuit_path(publish.TAG)
    current = json.loads(canonical_path.read_text())
    base = publish._base_from_current(current, canonical_path)
    base_claim = copy.deepcopy(base["claims"][-1])
    record = publish.build_record()
    event = record["evidence_events"][-1]
    assert event["event_id"] == publish.CLOSURE_EVENT
    assert event["stage"] == event["verdict"] == "invalid"
    assert event["failure_kind"] == "implementation_failure"
    assert event["supersedes_event_id"] == publish.OPEN_EVENT
    assert event["claim_id"] == publish.CLAIM_ID
    assert event["result_artifact_id"] == "localization_v2_administrative_stop"
    assert "never executed" in event["notes"]
    assert "no learned coordinate" in event["notes"]
    metrics = {item["name"]: item["estimate"] for item in event["metrics"]}
    assert metrics == {
        "model_calls": 0,
        "result_absent": True,
        "compiler_v1_valid": False,
        "compiler_v2_valid": False,
        "compiler_v3_frozen": False,
    }
    claim = record["claims"][-1]
    assert claim["claim_id"] == base_claim["claim_id"] == publish.CLAIM_ID
    assert claim["revision"] == base_claim["revision"] == 10
    assert claim["next_missing"] == base_claim["next_missing"]
    assert claim["evidence_event_ids"][:-1] == base_claim["evidence_event_ids"]
    assert claim["evidence_event_ids"][-1] == publish.CLOSURE_EVENT


def test_exact_artifacts_are_bound_and_hash_drift_fails(monkeypatch) -> None:
    record = publish.build_record()
    for artifact_id, expected in publish.REQUIRED_EXISTING_ARTIFACTS.items():
        assert record["artifacts"][artifact_id]["sha256"] == expected
    for artifact_id, (_, expected, _) in publish.ARTIFACTS.items():
        assert record["artifacts"][artifact_id]["sha256"] == expected
    changed = dict(publish.ARTIFACTS)
    relative, _, kind = changed["localization_v2_compiler_v1_block_review"]
    changed["localization_v2_compiler_v1_block_review"] = (relative, "0" * 64, kind)
    monkeypatch.setattr(publish, "ARTIFACTS", changed)
    with pytest.raises(publish.PublicationError, match="artifact hash mismatch"):
        publish.build_record()


def test_generated_open_audit_closes_exact_preregistration() -> None:
    record = publish.build_record()
    superseded = {
        event["supersedes_event_id"] for event in record["evidence_events"]
        if event.get("supersedes_event_id")
    }
    open_ids = {
        event["event_id"] for event in record["evidence_events"]
        if event["stage"] == "preregistered" and event["verdict"] == "inconclusive"
        and event["event_id"] not in superseded
    }
    assert publish.OPEN_EVENT not in open_ids
    assert not open_ids


def test_cli_apply_twice_is_byte_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "task_subject_verb_number_agreement.json"
    target.write_bytes(publish.registry.circuit_path(publish.TAG).read_bytes())
    command = [
        sys.executable, str(Path(publish.__file__)), "--apply", "--record", str(target),
        "--no-regenerate",
    ]
    subprocess.run(command, cwd=publish.REPO, check=True, capture_output=True, text=True)
    first = target.read_bytes()
    subprocess.run(command, cwd=publish.REPO, check=True, capture_output=True, text=True)
    assert target.read_bytes() == first
    assert json.loads(first)["evidence_events"][-1]["event_id"] == publish.CLOSURE_EVENT


def test_nonexact_base_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "task_subject_verb_number_agreement.json"
    base = json.loads(publish.registry.circuit_path(publish.TAG).read_text())
    base["claims"][-1]["next_missing"] = "silently changed"
    target.write_text(json.dumps(base, indent=1) + "\n")
    with pytest.raises(publish.PublicationError, match="exact v10 base"):
        publish.build_record(target)
