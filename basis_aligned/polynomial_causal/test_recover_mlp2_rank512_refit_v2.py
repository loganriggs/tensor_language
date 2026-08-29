from __future__ import annotations

from pathlib import Path

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
