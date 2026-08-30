#!/usr/bin/env python3
"""Lineage-only recovery for the unrun sparse-MLP1 FIT/SELECT transaction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import run_mlp1_sparse_c512_continue_factorial_v1_fit as assay


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AMENDMENT = HERE / "MLP1_SPARSE_C512_CONTINUE_FACTORIAL_V2_LINEAGE_RECOVERY_AMENDMENT.md"
RECOVERY = Path(__file__).resolve()
TEST = HERE / "test_recover_mlp1_sparse_c512_continue_factorial_v2.py"
AUDIT = HERE / "mlp1_sparse_c512_continue_factorial_v2_recovery_independent_audit.json"
RECOVERY_SOURCES = (AMENDMENT, RECOVERY, TEST)

ROWS_RECEIPT_SHA256 = "ce4a6f8eeb20840711bb20677ff8310f1a39db55b50106face1157cd2feeef7f"
ROWS_AUDIT_SHA256 = "ef223f4294cfae84603a22cc4d11c239315579260cbd6f88bcaa0ce6d84f2ebc"
ROWS_RECEIPT_SOURCE_COMMIT = "236ae134ce80c78144b9eae1420336be06399c83"
ROWS_AUDIT_SOURCE_COMMIT = "15ed37b9fec29685a415c7b940e026011f0c20ef"

V1_PATHS = (
    assay.AUTHORITY, assay.BUNDLE, assay.RESULT, assay.RECEIPT, assay.FAILURE, assay.LOCK,
)
V2_AUTHORITY = HERE / "mlp1_sparse_c512_continue_factorial_v2_fit_authority.json"
V2_BUNDLE = HERE / "mlp1_sparse_c512_continue_factorial_v2_fit_bundle.pt"
V2_RESULT = HERE / "mlp1_sparse_c512_continue_factorial_v2_fit_result.json"
V2_RECEIPT = HERE / "mlp1_sparse_c512_continue_factorial_v2_fit_receipt.json"
V2_FAILURE = HERE / "mlp1_sparse_c512_continue_factorial_v2_fit_failure.json"
V2_LOCK = Path("/workspace/runs/.mlp1_sparse_c512_continue_factorial_v2_fit.lock")

_BASE_VALIDATE_ROWS = assay.validate_row_receipt
_BASE_PROTECTED = assay.protected_snapshot
_BASE_WRITE_JSON = assay.write_json_create_only


def file_sha256(path: Path) -> str:
    return assay.file_sha256(path)


def recovery_sources(commit: str) -> dict[str, str]:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    output: dict[str, str] = {}
    for path in RECOVERY_SOURCES:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"uncommitted sparse-MLP1 recovery source: {relative}")
        output[relative] = digest
    return output


def validate_recovery_audit() -> tuple[dict[str, Any], str]:
    if not AUDIT.is_file():
        raise RuntimeError("independent sparse-MLP1 recovery audit is absent")
    raw = AUDIT.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    if file_sha256(AUDIT) != digest or set(value) != {
        "schema", "status", "outcome_access", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    } or value.get("schema") != (
        "mlp1_sparse_c512_continue_factorial_v2_recovery_independent_audit"
    ) or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or not isinstance(value.get("tests_passed"), int) or value["tests_passed"] < 1:
        raise RuntimeError("independent sparse-MLP1 recovery audit is not an exact GO")
    commit = value.get("audited_source_commit")
    sources = value.get("audited_source_hashes")
    if not isinstance(commit, str) or recovery_sources(commit) != sources:
        raise RuntimeError("sparse-MLP1 recovery audit binding changed")
    return value, digest


def validate_equivalent_row_receipt(
    value: Mapping[str, Any], sources: Mapping[str, str], audit: Mapping[str, Any],
    audit_sha: str,
) -> None:
    """Apply v1 semantics after proving the two historical labels are byte-equivalent."""

    if value.get("source_commit") != ROWS_RECEIPT_SOURCE_COMMIT \
            or audit.get("audited_source_commit") != ROWS_AUDIT_SOURCE_COMMIT \
            or audit_sha != ROWS_AUDIT_SHA256 \
            or assay.rows_life.source_hashes(ROWS_RECEIPT_SOURCE_COMMIT) != dict(sources) \
            or assay.rows_life.source_hashes(ROWS_AUDIT_SOURCE_COMMIT) != dict(sources):
        raise RuntimeError("sparse-MLP1 row/audit commits are not source-equivalent")
    normalized = dict(value)
    normalized["source_commit"] = ROWS_AUDIT_SOURCE_COMMIT
    _BASE_VALIDATE_ROWS(normalized, sources, audit, audit_sha)


def recovery_lineage_snapshot() -> dict[str, Any]:
    if any(path.exists() for path in V1_PATHS):
        raise RuntimeError("v1 sparse-MLP1 execution namespace is no longer pristine")
    receipt, receipt_sha = assay.stable_json(assay.ROWS_RECEIPT, ROWS_RECEIPT_SHA256)
    sources = receipt.get("source_hashes")
    if not isinstance(sources, dict):
        raise RuntimeError("sparse-MLP1 row source map changed")
    audit, audit_sha = assay.rows_life.validate_independent_audit(sources)
    validate_equivalent_row_receipt(receipt, sources, audit, audit_sha)
    recovery_audit, recovery_audit_sha = validate_recovery_audit()
    return {
        "v1_execution_paths_absent": [str(path) for path in V1_PATHS],
        "rows_receipt_sha256": receipt_sha,
        "rows_audit_sha256": audit_sha,
        "rows_receipt_source_commit": ROWS_RECEIPT_SOURCE_COMMIT,
        "rows_audit_source_commit": ROWS_AUDIT_SOURCE_COMMIT,
        "identical_scientific_source_hashes": dict(sources),
        "recovery_audit_sha256": recovery_audit_sha,
        "recovery_audited_source_commit": recovery_audit["audited_source_commit"],
        "recovery_source_hashes": recovery_audit["audited_source_hashes"],
    }


def composite_protected(
    commit: str, sources: Mapping[str, str], audit_sha: str, row_receipt_sha: str,
) -> dict[str, Any]:
    value = _BASE_PROTECTED(commit, sources, audit_sha, row_receipt_sha)
    value["v2_lineage_recovery"] = recovery_lineage_snapshot()
    return value


def recovery_write_json(path: Path, value: Any, *, pre_link_check=None) -> None:
    if path == V2_AUTHORITY:
        value["schema"] = "mlp1_sparse_c512_continue_factorial_v2_fit_authority"
        value["v2_lineage_recovery"] = recovery_lineage_snapshot()
    elif path == V2_RECEIPT:
        value["schema"] = "mlp1_sparse_c512_continue_factorial_v2_fit_receipt"
        value["v2_lineage_recovery"] = recovery_lineage_snapshot()
    elif path == V2_FAILURE:
        value["schema"] = "mlp1_sparse_c512_continue_factorial_v2_fit_failure"
        value["v2_lineage_recovery"] = recovery_lineage_snapshot()
    _BASE_WRITE_JSON(path, value, pre_link_check=pre_link_check)


def configure_namespace() -> None:
    recovery_lineage_snapshot()
    assay.AUTHORITY, assay.BUNDLE = V2_AUTHORITY, V2_BUNDLE
    assay.RESULT, assay.RECEIPT, assay.FAILURE = V2_RESULT, V2_RECEIPT, V2_FAILURE
    assay.LOCK = V2_LOCK
    assay.validate_row_receipt = validate_equivalent_row_receipt
    assay.protected_snapshot = composite_protected
    assay.write_json_create_only = recovery_write_json


def main() -> None:
    configure_namespace()
    assay.main()


if __name__ == "__main__":
    main()
