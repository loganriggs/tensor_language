from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import prepare_mlp2_trajectory_robust_r512_v2_eval_rows as rows
import recover_mlp2_trajectory_robust_r512_v2_physical_eval as recovery
import run_mlp2_trajectory_robust_r512_v1_physical_eval as science


def fake_ledger(dce: float, kl: float | None = None) -> torch.Tensor:
    value = torch.zeros(192, 9, dtype=torch.float64)
    value[:, 0] = 10.0; value[:, 1] = 10.0 + dce * 192
    value[:, 2] = (dce if kl is None else kl) * 192
    value[:, 4] = 1.0; value[:, 5:8] = 192; value[:, 8] = 192
    return value


def test_recovery_admission_binds_failure_audit_science_and_absence() -> None:
    value = rows.recovery_admission()
    assert value["v1_failure_sha256"] == rows.V1_FAILURE_SHA
    assert value["original_audit_sha256"] == rows.ORIGINAL_AUDIT_SHA
    assert value["original_audited_commit"] == rows.ORIGINAL_AUDITED_COMMIT
    assert value["science_changed"] is False
    assert value["failure_phase"] == (
        "source_ancestry_before_ordered_source_registry_or_harvest"
    )
    assert value["v1_cache_receipt_lock_absent"]
    assert value["v1_evaluator_outputs_and_lock_absent"]


def test_recovery_namespaces_are_disjoint() -> None:
    assert len({rows.CACHE, rows.v1.CACHE}) == 2
    assert len({rows.RECEIPT, rows.v1.RECEIPT}) == 2
    assert len({rows.FAILURE, rows.v1.FAILURE}) == 2
    assert len({rows.LOCK, rows.v1.LOCK}) == 2
    old = science
    assert all(path not in {old.AUTHORITY, old.LEDGER, old.RESULT,
                            old.RECEIPT, old.FAILURE, old.LOCK}
               for path in (recovery.AUTHORITY, recovery.LEDGER, recovery.RESULT,
                            recovery.RECEIPT, recovery.FAILURE, recovery.LOCK))


def test_recovery_source_closure_contains_original_and_new_files() -> None:
    assert len(rows.SOURCE_PATHS) == len(set(rows.SOURCE_PATHS))
    assert set(rows.DIRECT_SOURCES).issubset(rows.SOURCE_PATHS)
    assert set(rows.v1.SOURCE_PATHS).issubset(rows.SOURCE_PATHS)
    assert rows.ORIGINAL_AUDIT not in rows.SOURCE_PATHS


def test_v2_result_is_exact_v1_pure_derivation_plus_schema_and_admission() -> None:
    ledgers = {arm: fake_ledger(0.0) for arm in science.ARMS}
    ledgers.update({
        "C512": fake_ledger(0.003, 0.004),
        "FULL512": fake_ledger(0.050, 0.055),
        "C512_FULL512": fake_ledger(0.064, 0.069),
        "CONTINUE512": fake_ledger(0.049, 0.054),
        "C512_CONTINUE512": fake_ledger(0.060, 0.066),
        "ROBUST512": fake_ledger(0.049, 0.054),
        "C512_ROBUST512": fake_ledger(0.053, 0.059),
    })
    curve = [{"step": step, "worst_normalized_mse": 0.44}
             for step in range(0, 1201, 25)]
    bundle = {"curves": {"ROBUST512": curve}}
    v1 = science.derive_result(ledgers, 1.0, bundle)
    v2 = recovery.derive_result(ledgers, 1.0, bundle)
    assert v2["schema"] == recovery.RESULT_SCHEMA
    assert v2.pop("recovery_admission") == rows.recovery_admission()
    v2["schema"] = v1["schema"]
    assert v2 == v1


def test_row_terminal_writer_rechecks_recovery_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "receipt.json"
    monkeypatch.setattr(rows, "RECEIPT", destination)
    calls = []
    monkeypatch.setattr(rows, "recovery_admission", lambda: calls.append(1) or {"ok": True})
    rows.recovery_write_json(destination, {"schema": "x"})
    value = json.loads(destination.read_text())
    assert value["recovery_admission"] == {"ok": True}
    assert len(calls) == 2


def test_late_v1_output_invalidates_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    late = tmp_path / "late-v1-output"
    original = rows._v1_absence_paths
    monkeypatch.setattr(rows, "_v1_absence_paths", lambda: (*original(), late))
    late.write_text("late")
    with pytest.raises(RuntimeError, match="v1 row or evaluation output appeared"):
        rows.recovery_admission()


def test_v1_output_created_during_science_replay_invalidates_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    late = tmp_path / "late-v1-output"
    original_paths = rows._v1_absence_paths
    original_hashes = rows.v1.source_hashes
    monkeypatch.setattr(rows, "_v1_absence_paths", lambda: (*original_paths(), late))

    def racing_hashes(commit: str):
        value = original_hashes(commit)
        late.write_text("appeared during replay")
        return value

    monkeypatch.setattr(rows.v1, "source_hashes", racing_hashes)
    with pytest.raises(RuntimeError, match="appeared during science replay"):
        rows.recovery_admission()


def test_row_terminal_guard_checks_claim_after_final_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "receipt.json"
    lock = tmp_path / "lock"
    monkeypatch.setattr(rows, "RECEIPT", destination)
    claim = rows.base.acquire_claim(lock)
    calls = 0

    def racing_admission():
        nonlocal calls
        calls += 1
        if calls == 2:
            lock.write_text("rival claim")
        return {"bound": True}

    monkeypatch.setattr(rows, "recovery_admission", racing_admission)
    with pytest.raises(RuntimeError):
        rows.recovery_write_json(
            destination, {"schema": "x"},
            pre_link_check=lambda: rows.base.require_claim(claim, lock),
        )
    assert not destination.exists()


def configure_eval_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    names = ("authority.json", "ledger.pt", "result.json", "receipt.json", "failure.json", "lock")
    paths = {name: tmp_path / name for name in names}
    for attribute, name in (
        ("AUTHORITY", "authority.json"), ("LEDGER", "ledger.pt"),
        ("RESULT", "result.json"), ("RECEIPT", "receipt.json"),
        ("FAILURE", "failure.json"), ("LOCK", "lock"),
    ):
        monkeypatch.setattr(recovery, attribute, paths[name])
    return paths


def test_v2_receipt_binds_recovery_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    admission = {"bound": True}
    monkeypatch.setattr(rows, "recovery_admission", lambda: admission)
    value = recovery.validate_receipt({
        "schema": recovery.RECEIPT_SCHEMA, "status": "result_complete_receipt_last",
        "authority_sha256": "a", "ledger_sha256": "l", "result_sha256": "r",
        "evaluation_opened": True, "recovery_admission": admission,
    }, "a", "l", "r")
    assert value["recovery_admission"] == admission


def test_v2_prelink_failure_publishes_with_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = configure_eval_namespace(tmp_path, monkeypatch)
    admission = {"bound": True}
    monkeypatch.setattr(rows, "recovery_admission", lambda: admission)
    claim = rows.base.acquire_claim(paths["lock"])
    try:
        value = recovery.publish_failure(
            claim, RuntimeError("injected prelink"), {"constructed": True},
            {"protected": True}, False,
        )
        assert value["authority_exists"] is False
        assert value["recovery_admission"] == admission
        assert paths["failure.json"].is_file()
        assert not paths["receipt.json"].exists()
    finally:
        rows.base.release_claim(claim, paths["lock"])


def test_v2_failure_guard_rechecks_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = configure_eval_namespace(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(rows, "recovery_admission", lambda: calls.append(1) or {"bound": True})
    expected = recovery.artifact_snapshot()
    claim = rows.base.acquire_claim(paths["lock"])
    try:
        recovery.failure_terminal_guard(claim, expected, None, None)
        assert len(calls) == 1
    finally:
        rows.base.release_claim(claim, paths["lock"])


def test_v2_failure_guard_rejects_terminal_created_during_final_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = configure_eval_namespace(tmp_path, monkeypatch)
    expected = recovery.artifact_snapshot()
    claim = rows.base.acquire_claim(paths["lock"])

    def racing_admission():
        paths["receipt.json"].write_text("rival terminal")
        return {"bound": True}

    monkeypatch.setattr(rows, "recovery_admission", racing_admission)
    try:
        with pytest.raises(RuntimeError, match="aggregate or terminal changed"):
            recovery.failure_terminal_guard(claim, expected, None, None)
    finally:
        rows.base.release_claim(claim, paths["lock"])
