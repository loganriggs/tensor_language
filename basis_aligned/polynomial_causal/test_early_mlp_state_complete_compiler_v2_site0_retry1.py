from __future__ import annotations

import json
import inspect

import pytest
import torch

import early_mlp_state_complete_compiler_v2_site0_retry1 as retry


def _write_pending_outputs(monkeypatch, tmp_path, *, wrong_parent: bool = False) -> None:
    artifact = tmp_path / "retry1_programs.pt"
    result_path = tmp_path / "retry1_results.json"
    manifest_path = tmp_path / "retry1_manifest.json"
    receipt_path = tmp_path / "retry1_receipt.json"
    source_hashes = {"runner.py": "abc"}
    parent = "wrong" if wrong_parent else retry.PINS[retry.PARENT_FAILURE]
    common = {
        "parent_failure_manifest_sha256": parent,
        "numeric_diagnostic_receipt_sha256": retry.PINS[retry.DIAGNOSTIC_RECEIPT],
        "retry1_protocol_sha256": retry.PINS[retry.PROTOCOL],
        "source_commit": "source-commit",
        "source_hashes": source_hashes,
    }
    torch.save({
        "status": "pending_site0_retry1_last_written_authority_receipt",
        "authority": "compiler_v2_site0_retry1_pending",
        "authorized_for_training": False,
        "training_license_sites": [],
        "selection": {"selected": "B8", "selected_family": "family"},
        "candidates": {"B8": {}},
        "controls": {"mean": {}, "shuffle": {}, "full_native": {}},
        **common,
    }, artifact)
    result = {
        "status": "completed_site0_validation_pending_authority_retry1",
        "authorized_for_training": False,
        "artifact_sha256": retry.file_sha256(artifact),
        **common,
    }
    result_path.write_text(json.dumps(result))
    manifest = {
        "status": "completed_integrity_pending_last_written_authority_receipt_retry1",
        "authorized_for_training": False,
        "artifact_sha256": retry.file_sha256(artifact),
        "result_sha256": retry.file_sha256(result_path),
        "component_tree_unchanged": True,
        "hook_restored_and_inert": True,
        "outer_sa_main_returned": True,
        "protected_before": {},
        "protected_after_outer": {},
        **common,
    }
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(retry, "ARTIFACT", artifact)
    monkeypatch.setattr(retry, "RESULT", result_path)
    monkeypatch.setattr(retry, "MANIFEST", manifest_path)
    monkeypatch.setattr(retry, "RECEIPT", receipt_path)


def test_retry1_pins_parent_failure_diagnostic_and_protocol() -> None:
    for path, expected in retry.PINS.items():
        assert retry.file_sha256(path) == expected
    protocol = json.loads(retry.PROTOCOL.read_text())
    assert protocol["licensed_change"].startswith("Scorer currency only.")
    receipt = json.loads(retry.DIAGNOSTIC_RECEIPT.read_text())
    assert receipt["scorer_only_retry_licensed"] is True
    assert receipt["representation_retry_licensed"] is False


def test_retry1_namespace_isolated_and_parent_outputs_remain_absent() -> None:
    assert all("retry1" in path.name for path in retry.OUTPUTS)
    assert not any(path.exists() for path in retry.ORIGINAL_ABSENT)
    assert retry.PARENT_FAILURE not in retry.OUTPUTS


def test_retry1_source_closure_contains_failed_and_diagnostic_sources() -> None:
    names = {path.name for path in retry.SOURCE_CLOSURE}
    assert "early_mlp_state_complete_compiler_v2_site0.py" in names
    assert "early_mlp_state_complete_compiler_v2_full_native_numeric_diagnostic_v1.py" in names
    assert "early_mlp_state_complete_compiler_v2_preflight_r2.py" in names
    assert "test_early_mlp_state_complete_compiler_v2_preflight_r2.py" in names
    assert "test_early_mlp_state_complete_compiler_v2_site0_retry1.py" in names


def test_retry1_parent_and_diagnostic_source_lineage_matches_live_files() -> None:
    for authority_path in (retry.PARENT_FAILURE, retry.DIAGNOSTIC_RECEIPT):
        authority = json.loads(authority_path.read_text())
        for relative, expected in authority["source_hashes"].items():
            assert retry.file_sha256(retry.ROOT / relative) == expected


def test_retry1_pending_validator_rejects_wrong_parent(monkeypatch, tmp_path) -> None:
    _write_pending_outputs(monkeypatch, tmp_path, wrong_parent=True)
    with pytest.raises(RuntimeError, match="parent failure"):
        retry.validate_pending_outputs()


def test_retry1_authority_validator_rejects_mismatched_result(monkeypatch, tmp_path) -> None:
    _write_pending_outputs(monkeypatch, tmp_path)
    payload, _, _ = retry.validate_pending_outputs()
    retry.RECEIPT.write_text(json.dumps({
        "status": "frozen_site0_retry1_last_written_authority",
        "authority": "compiler_v2_sequential_site0_validation_freeze_retry1",
        "authorized_for_scored_experiments": False,
        "authorized_for_training": True,
        "training_license_sites": [1],
        "artifact_path": str(retry.ARTIFACT.resolve()),
        "artifact_sha256": retry.file_sha256(retry.ARTIFACT),
        "artifact_bytes": retry.ARTIFACT.stat().st_size,
        "result_path": str(retry.RESULT.resolve()),
        "result_sha256": "wrong",
        "manifest_path": str(retry.MANIFEST.resolve()),
        "manifest_sha256": retry.file_sha256(retry.MANIFEST),
        "parent_failure_manifest_sha256": retry.PINS[retry.PARENT_FAILURE],
        "numeric_diagnostic_receipt_sha256": retry.PINS[retry.DIAGNOSTIC_RECEIPT],
        "retry1_protocol_sha256": retry.PINS[retry.PROTOCOL],
        "source_commit": payload["source_commit"],
        "source_hashes": payload["source_hashes"],
        "selected": "B8",
        "selected_family": "family",
    }))
    with pytest.raises(RuntimeError, match="authority receipt binding"):
        retry.validate_artifact()


def test_retry1_late_failure_leaves_no_training_authority(monkeypatch, tmp_path) -> None:
    _write_pending_outputs(monkeypatch, tmp_path)
    for path in (retry.ARTIFACT, retry.RESULT, retry.MANIFEST):
        path.unlink()
    lock = tmp_path / "retry1.lock"
    monkeypatch.setattr(retry, "LOCK", lock)
    monkeypatch.setattr(
        retry, "OUTPUTS", (retry.ARTIFACT, retry.RECEIPT, retry.RESULT, retry.MANIFEST)
    )
    monkeypatch.setattr(retry, "protected_snapshot", lambda: {})

    def fail_after_pending_outputs(_before) -> None:
        _write_pending_outputs(monkeypatch, tmp_path)
        raise RuntimeError("late integrity failure")

    monkeypatch.setattr(retry, "run_claimed", fail_after_pending_outputs)
    with pytest.raises(RuntimeError, match="late integrity failure"):
        retry.main()
    assert not retry.RECEIPT.exists()
    assert torch.load(retry.ARTIFACT, weights_only=True)["authorized_for_training"] is False
    assert json.loads(retry.RESULT.read_text())["authorized_for_training"] is False
    failed_manifest = json.loads(retry.MANIFEST.read_text())
    assert failed_manifest["status"] == "failed_compiler_v2_site0_retry1"
    assert failed_manifest["authorized_for_training"] is False


def test_retry1_receipt_is_last_content_write_after_outer_return() -> None:
    source = inspect.getsource(retry.run_claimed)
    outer_return = source.rfind("sa.main(oracle_content_screen=True)")
    authority_write = source.rfind("write_json_atomic(receipt, RECEIPT)")
    assert 0 <= outer_return < authority_write
    assert "write_" not in source[authority_write + len(
        "write_json_atomic(receipt, RECEIPT)"
    ):]
