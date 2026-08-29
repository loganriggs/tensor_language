from __future__ import annotations

from pathlib import Path

import recover_mlp2_rank512_refit_v2 as recovery


def test_recovery_changes_only_output_and_lock_namespace(monkeypatch, tmp_path: Path) -> None:
    original_rows = recovery.assay.ROWS_RECEIPT
    original_sources = recovery.assay.SOURCE_PATHS
    monkeypatch.setattr(recovery, "HERE", tmp_path)
    recovery.configure_namespace()
    assert recovery.assay.ROWS_RECEIPT == original_rows
    assert recovery.assay.SOURCE_PATHS == original_sources
    assert all("v2_recovery" in path.name for path in (
        recovery.assay.AUTHORITY, recovery.assay.BUNDLE, recovery.assay.LEDGER,
        recovery.assay.RESULT, recovery.assay.RECEIPT, recovery.assay.FAILURE,
    ))
