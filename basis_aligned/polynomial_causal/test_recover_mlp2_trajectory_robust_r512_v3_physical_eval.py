from __future__ import annotations

import json
from pathlib import Path

import pytest

import recover_mlp2_trajectory_robust_r512_v2_physical_eval as v2
import recover_mlp2_trajectory_robust_r512_v3_physical_eval as v3


def test_v3_admission_binds_exact_preopen_v2_failure_and_rows() -> None:
    value = v3.recovery_admission()
    assert value["v2_failure_sha256"] == v3.V2_FAILURE_SHA
    assert value["v2_authority_exists"] is False
    assert value["v2_evaluation_may_have_opened"] is False
    assert value["v2_rows_receipt_sha256"] == v3.V2_ROWS_RECEIPT_SHA
    assert value["v2_rows_sha256"] == v3.V2_ROWS_SHA
    assert value["science_changed"] is False


def test_v3_source_closure_is_unique_and_transitive() -> None:
    assert len(v3.SOURCE_PATHS) == len(set(v3.SOURCE_PATHS))
    assert set(v2.row_life.SOURCE_PATHS).issubset(v3.SOURCE_PATHS)
    assert {v3.AMENDMENT, v3.RUNNER, v3.TEST}.issubset(v3.SOURCE_PATHS)


def test_v3_namespaces_are_disjoint() -> None:
    assert all(path not in {v2.AUTHORITY, v2.LEDGER, v2.RESULT,
                            v2.RECEIPT, v2.FAILURE, v2.LOCK}
               for path in (v3.AUTHORITY, v3.LEDGER, v3.RESULT,
                            v3.RECEIPT, v3.FAILURE, v3.LOCK))


def test_v3_rejects_late_v2_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    late = tmp_path / "late-v2-output"
    original = v3._v2_absence_paths
    monkeypatch.setattr(v3, "_v2_absence_paths", lambda: (*original(), late))
    late.write_text("late")
    with pytest.raises(RuntimeError, match="v2 evaluation output or lock appeared"):
        v3.recovery_admission()


def test_committed_sources_uses_audited_commit_not_mutable_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"audited_source_commit": "frozen-commit"}))
    monkeypatch.setattr(v3, "AUDIT", audit)
    monkeypatch.setattr(v3, "_OPEN_AUDIT_BINDING", None)
    monkeypatch.setattr(v3, "source_hashes", lambda commit: {"commit": commit})
    assert v3.committed_sources() == (
        "frozen-commit", {"commit": "frozen-commit"},
    )


def test_audit_swap_between_source_binding_and_validation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = tmp_path / "audit.json"
    sources = {"same": "hashes"}

    def value(commit: str) -> dict:
        return {
            "schema": "mlp2_trajectory_robust_r512_v3_physical_eval_independent_audit",
            "status": "GO", "outcome_access": False,
            "audited_source_commit": commit, "audited_source_hashes": sources,
            "tests_passed": 1, "reviewer": "test",
        }

    audit.write_text(json.dumps(value("commit-a")))
    monkeypatch.setattr(v3, "AUDIT", audit)
    monkeypatch.setattr(v3, "_OPEN_AUDIT_BINDING", None)
    monkeypatch.setattr(v3, "source_hashes", lambda _commit: sources)
    assert v3.committed_sources() == ("commit-a", sources)
    audit.write_text(json.dumps(value("commit-b")))
    with pytest.raises(RuntimeError, match="audit commit binding changed"):
        v3.validate_independent_audit(sources, audit)


def test_recovery_admission_survives_v2_namespace_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "AUTHORITY", "LEDGER", "RESULT", "RECEIPT", "FAILURE", "LOCK",
        "AUTHORITY_SCHEMA", "LEDGER_SCHEMA", "RESULT_SCHEMA", "RECEIPT_SCHEMA",
        "FAILURE_SCHEMA", "committed_sources", "validate_row_receipt",
    ):
        monkeypatch.setattr(v2, name, getattr(v2, name))
    for name in ("source_hashes", "validate_independent_audit", "recovery_admission"):
        monkeypatch.setattr(v2.row_life, name, getattr(v2.row_life, name))
    v3.configure()
    assert v3.recovery_admission()["v2_evaluation_outputs_and_lock_absent"] is True
