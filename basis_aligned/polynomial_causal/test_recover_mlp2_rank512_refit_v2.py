from __future__ import annotations

from pathlib import Path

import os
import pytest

import recover_mlp2_rank512_refit_v2 as recovery


def test_recovery_changes_only_output_and_lock_namespace(monkeypatch, tmp_path: Path) -> None:
    original_rows = recovery.assay.ROWS_RECEIPT
    original_sources = recovery.assay.SOURCE_PATHS
    recovery.configure_namespace()
    assert recovery.assay.ROWS_RECEIPT == original_rows
    assert recovery.assay.SOURCE_PATHS == original_sources
    assert all("v2_recovery" in path.name for path in (
        recovery.assay.AUTHORITY, recovery.assay.BUNDLE, recovery.assay.LEDGER,
        recovery.assay.RESULT, recovery.assay.RECEIPT, recovery.assay.FAILURE,
    ))


def test_recovery_lineage_requires_all_v1_success_paths_absent(monkeypatch) -> None:
    monkeypatch.setattr(recovery, "validate_recovery_audit",
                        lambda: ({"audited_source_commit": "a" * 40,
                                  "audited_source_hashes": {}}, "b" * 64))
    monkeypatch.setattr(recovery.assay, "committed_sources",
                        lambda: ("c" * 40, recovery.json.loads(
                            recovery.V1_FAILURE.read_bytes())["source_hashes"]))
    value = recovery.v1_lineage_snapshot()
    assert value["v1_failure_sha256"] == recovery.V1_FAILURE_SHA256
    assert len(value["v1_success_and_lock_absent"]) == 5


def _terminal_fixture(monkeypatch, tmp_path):
    lock = tmp_path / "run.lock"
    monkeypatch.setattr(recovery, "V2_LOCK", lock)
    monkeypatch.setattr(recovery, "V2_RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(recovery, "V2_FAILURE", tmp_path / "failure.json")
    lineage = {"bound": True}
    monkeypatch.setattr(recovery, "v1_lineage_snapshot", lambda: lineage)
    claim = recovery.assay.row_life.acquire_claim(lock)
    return claim, lineage


def test_late_failure_during_receipt_artifact_replay_is_caught(monkeypatch, tmp_path) -> None:
    claim, lineage = _terminal_fixture(monkeypatch, tmp_path)
    artifact = tmp_path / "artifact"; artifact.write_bytes(b"fixed")
    expected = recovery.file_sha256(artifact)
    original = recovery.assay.stable_bytes
    def race(path, digest):
        value = original(path, digest)
        recovery.V2_FAILURE.write_text("competitor")
        return value
    monkeypatch.setattr(recovery.assay, "stable_bytes", race)
    try:
        with pytest.raises(RuntimeError, match="raced"):
            recovery.terminal_boundary_guard(
                claim, expected_lineage=lineage, artifact_hashes={artifact: expected},
                publishing="receipt")
    finally:
        recovery.V2_FAILURE.unlink(missing_ok=True)
        recovery.assay.row_life.release_claim(claim, recovery.V2_LOCK)


def test_late_receipt_blocks_failure_publication(monkeypatch, tmp_path) -> None:
    claim, lineage = _terminal_fixture(monkeypatch, tmp_path)
    recovery.V2_RECEIPT.write_text("competitor")
    try:
        with pytest.raises(RuntimeError, match="competing"):
            recovery.terminal_boundary_guard(
                claim, expected_lineage=lineage, artifact_hashes={},
                publishing="failure")
    finally:
        recovery.V2_RECEIPT.unlink()
        recovery.assay.row_life.release_claim(claim, recovery.V2_LOCK)


def test_lock_replacement_during_final_lineage_replay_is_caught(monkeypatch, tmp_path) -> None:
    claim, lineage = _terminal_fixture(monkeypatch, tmp_path)
    calls = 0
    def replace_lock():
        nonlocal calls
        calls += 1
        if calls == 2:
            replacement = tmp_path / "replacement"
            replacement.write_text(claim.nonce + "\n")
            os.replace(replacement, recovery.V2_LOCK)
        return lineage
    monkeypatch.setattr(recovery, "v1_lineage_snapshot", replace_lock)
    try:
        with pytest.raises(RuntimeError, match="replaced"):
            recovery.terminal_boundary_guard(
                claim, expected_lineage=lineage, artifact_hashes={},
                publishing="receipt")
    finally:
        recovery.assay.row_life.release_claim(claim, recovery.V2_LOCK)
