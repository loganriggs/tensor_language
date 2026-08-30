from __future__ import annotations

import copy
import json

import pytest

import recover_mlp1_sparse_c512_continue_factorial_v2 as recovery


def test_equivalent_validator_requires_both_commits_replay_same_sources(monkeypatch):
    sources = {"science.py": "a" * 64}
    audit = {"audited_source_commit": recovery.ROWS_AUDIT_SOURCE_COMMIT}
    value = {"source_commit": recovery.ROWS_RECEIPT_SOURCE_COMMIT}
    calls = []
    monkeypatch.setattr(
        recovery.assay.rows_life, "source_hashes", lambda commit: dict(sources),
    )
    monkeypatch.setattr(
        recovery, "_BASE_VALIDATE_ROWS",
        lambda normalized, observed, observed_audit, observed_sha: calls.append(
            (normalized, observed, observed_audit, observed_sha)
        ),
    )
    recovery.validate_equivalent_row_receipt(
        value, sources, audit, recovery.ROWS_AUDIT_SHA256,
    )
    assert calls[0][0]["source_commit"] == recovery.ROWS_AUDIT_SOURCE_COMMIT
    assert calls[0][1] == sources


def test_equivalent_validator_rejects_one_commit_hash_mismatch(monkeypatch):
    sources = {"science.py": "a" * 64}
    audit = {"audited_source_commit": recovery.ROWS_AUDIT_SOURCE_COMMIT}
    value = {"source_commit": recovery.ROWS_RECEIPT_SOURCE_COMMIT}

    def hashes(commit):
        return sources if commit == recovery.ROWS_AUDIT_SOURCE_COMMIT else {
            "science.py": "b" * 64,
        }

    monkeypatch.setattr(recovery.assay.rows_life, "source_hashes", hashes)
    with pytest.raises(RuntimeError, match="not source-equivalent"):
        recovery.validate_equivalent_row_receipt(
            value, sources, audit, recovery.ROWS_AUDIT_SHA256,
        )


def test_recovery_audit_requires_exact_three_file_source_family(tmp_path, monkeypatch):
    audit_path = tmp_path / "audit.json"
    sources = {str(path.relative_to(recovery.ROOT)): "a" * 64
               for path in recovery.RECOVERY_SOURCES}
    value = {
        "schema": "mlp1_sparse_c512_continue_factorial_v2_recovery_independent_audit",
        "status": "GO", "outcome_access": False,
        "audited_source_commit": "c" * 40,
        "audited_source_hashes": sources,
        "tests_passed": 1, "reviewer": "independent-test",
    }
    audit_path.write_text(json.dumps(value))
    monkeypatch.setattr(recovery, "AUDIT", audit_path)
    monkeypatch.setattr(recovery, "recovery_sources", lambda commit: dict(sources))
    parsed, digest = recovery.validate_recovery_audit()
    assert parsed == value and len(digest) == 64
    bad = copy.deepcopy(value); bad["outcome_access"] = True
    audit_path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="not an exact GO"):
        recovery.validate_recovery_audit()
    bad = copy.deepcopy(value); bad["reviewer"] = ""
    audit_path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="not an exact GO"):
        recovery.validate_recovery_audit()


def test_lineage_replay_rejects_v1_artifact_injected_mid_check(tmp_path, monkeypatch):
    v1_authority = tmp_path / "v1_authority.json"
    v1_paths = (v1_authority, tmp_path / "v1_bundle.pt")
    sources = {"science.py": "a" * 64}
    receipt = {"source_hashes": sources}
    audit = {"audited_source_commit": recovery.ROWS_AUDIT_SOURCE_COMMIT}
    monkeypatch.setattr(recovery, "V1_PATHS", v1_paths)
    monkeypatch.setattr(
        recovery.assay, "stable_json",
        lambda path, expected=None: (receipt, recovery.ROWS_RECEIPT_SHA256),
    )
    monkeypatch.setattr(
        recovery.assay.rows_life, "validate_independent_audit",
        lambda observed: (audit, recovery.ROWS_AUDIT_SHA256),
    )

    def inject(*args):
        v1_authority.write_text("{}\n")

    monkeypatch.setattr(recovery, "validate_equivalent_row_receipt", inject)
    monkeypatch.setattr(
        recovery, "validate_recovery_audit",
        lambda: ({"audited_source_commit": "c" * 40, "audited_source_hashes": {}}, "d" * 64),
    )
    with pytest.raises(RuntimeError, match="raced lineage replay"):
        recovery.recovery_lineage_snapshot()


def test_configuration_changes_only_execution_namespace_and_lineage_hooks(monkeypatch):
    original_rows = recovery.assay.ROWS_RECEIPT
    original_science = (
        recovery.assay.SEEDS, recovery.assay.STEPS, recovery.assay.BATCH_SIZE,
        recovery.assay.LEARNING_RATE, recovery.assay.SCORING,
    )
    names = (
        "AUTHORITY", "BUNDLE", "RESULT", "RECEIPT", "FAILURE", "LOCK",
        "validate_row_receipt", "protected_snapshot", "write_json_create_only",
    )
    originals = {name: getattr(recovery.assay, name) for name in names}
    monkeypatch.setattr(recovery, "recovery_lineage_snapshot", lambda: {"bound": True})
    try:
        recovery.configure_namespace()
        assert recovery.assay.ROWS_RECEIPT == original_rows
        assert (
            recovery.assay.SEEDS, recovery.assay.STEPS, recovery.assay.BATCH_SIZE,
            recovery.assay.LEARNING_RATE, recovery.assay.SCORING,
        ) == original_science
        assert recovery.assay.AUTHORITY == recovery.V2_AUTHORITY
        assert recovery.assay.BUNDLE == recovery.V2_BUNDLE
        assert recovery.assay.RESULT == recovery.V2_RESULT
        assert recovery.assay.RECEIPT == recovery.V2_RECEIPT
        assert recovery.assay.FAILURE == recovery.V2_FAILURE
        assert recovery.assay.LOCK == recovery.V2_LOCK
    finally:
        for name, value in originals.items():
            setattr(recovery.assay, name, value)


def test_composite_protected_binds_recovery_lineage(monkeypatch):
    monkeypatch.setattr(recovery, "_BASE_PROTECTED", lambda *args: {"science": "v1"})
    monkeypatch.setattr(
        recovery, "recovery_lineage_snapshot", lambda: {"commits_equivalent": True},
    )
    assert recovery.composite_protected("c", {}, "a", "r") == {
        "science": "v1",
        "v2_lineage_recovery": {"commits_equivalent": True},
    }


def test_recovery_json_marks_authority_and_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery, "V2_AUTHORITY", tmp_path / "authority.json")
    monkeypatch.setattr(recovery, "V2_RECEIPT", tmp_path / "receipt.json")
    monkeypatch.setattr(recovery, "V2_FAILURE", tmp_path / "failure.json")
    monkeypatch.setattr(recovery, "recovery_lineage_snapshot", lambda: {"bound": True})
    observed = []
    monkeypatch.setattr(
        recovery, "_BASE_WRITE_JSON",
        lambda path, value, pre_link_check=None: observed.append((path, copy.deepcopy(value))),
    )
    authority = {"schema": "v1"}
    recovery.recovery_write_json(recovery.V2_AUTHORITY, authority)
    receipt = {"schema": "v1"}
    recovery.recovery_write_json(recovery.V2_RECEIPT, receipt)
    assert observed[0][1]["schema"].endswith("v2_fit_authority")
    assert observed[1][1]["schema"].endswith("v2_fit_receipt")
    assert all(value["v2_lineage_recovery"] == {"bound": True} for _, value in observed)
