#!/usr/bin/env python3
"""Lineage-bound recovery from the pre-authority branch-pointer failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import run_mlp0_c512_mlp2_full512_composition_v1 as assay

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AMENDMENT = HERE / "MLP0_C512_MLP2_FULL512_COMPOSITION_V2_EXECUTION_RECOVERY_AMENDMENT.md"
RECOVERY = Path(__file__).resolve()
TEST = HERE / "test_recover_mlp0_c512_mlp2_full512_composition_v2.py"
AUDIT = HERE / "mlp0_c512_mlp2_full512_composition_v2_execution_recovery_audit.json"
RECOVERY_SOURCES = (AMENDMENT, RECOVERY, TEST)

V1_FAILURE = HERE / "mlp0_c512_mlp2_full512_composition_v1_failure.json"
V1_FAILURE_SHA = "6c375e461def332a38584e11a542ab4ec3c6822e6a385a31c46d8e51c98e42e1"
V1_AUTHORITY = assay.AUTHORITY
V1_LEDGER = assay.LEDGER
V1_RESULT = assay.RESULT
V1_RECEIPT = assay.RECEIPT
V1_LOCK = assay.LOCK
V2_AUTHORITY = HERE / "mlp0_c512_mlp2_full512_composition_v2_authority.json"
V2_LEDGER = HERE / "mlp0_c512_mlp2_full512_composition_v2_ledger.pt"
V2_RESULT = HERE / "mlp0_c512_mlp2_full512_composition_v2_result.json"
V2_RECEIPT = HERE / "mlp0_c512_mlp2_full512_composition_v2_receipt.json"
V2_FAILURE = HERE / "mlp0_c512_mlp2_full512_composition_v2_failure.json"
V2_LOCK = Path("/workspace/runs/.mlp0_c512_mlp2_full512_composition_v2.lock")

_BASE_PROTECTED = assay.protected_snapshot
_BASE_ATOMIC_JSON = assay.atomic_json


def file_sha256(path: Path) -> str:
    return assay.file_sha256(path)


def stable_v1_failure() -> dict[str, Any]:
    value, _ = assay.stable_json(V1_FAILURE, V1_FAILURE_SHA)
    if value != {
        "artifact_hashes": {}, "authority_exists": False,
        "error": value.get("error"), "evaluation_may_have_opened": False,
        "protected_snapshot": None,
        "schema": "mlp0_c512_mlp2_full512_composition_v1_failure",
        "status": "terminal_failure_no_receipt",
    } or "merge-base" not in value["error"]:
        raise RuntimeError("v1 composition execution failure lineage changed")
    if any(path.exists() for path in (
        V1_AUTHORITY, V1_LEDGER, V1_RESULT, V1_RECEIPT, V1_LOCK,
    )):
        raise RuntimeError("v1 composition success or lock artifact appeared")
    return value


def recovery_sources(commit: str) -> dict[str, str]:
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    output = {}
    for path in RECOVERY_SOURCES:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError("uncommitted recovery source")
        output[relative] = digest
    return output


def validate_recovery_audit() -> tuple[dict[str, Any], str]:
    raw = AUDIT.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    required = {"schema", "status", "outcome_access", "audited_source_commit",
                "audited_source_hashes", "tests_passed", "reviewer"}
    if file_sha256(AUDIT) != digest or set(value) != required or value.get("schema") != (
        "mlp0_c512_mlp2_full512_composition_v2_execution_recovery_audit"
    ) or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or not isinstance(value.get("tests_passed"), int) or value["tests_passed"] < 1:
        raise RuntimeError("execution recovery audit is not GO")
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str) or recovery_sources(commit) != value.get(
        "audited_source_hashes"
    ):
        raise RuntimeError("execution recovery audit binding changed")
    return value, digest


def science_sources() -> tuple[str, dict[str, str]]:
    rows, _ = assay.stable_json(assay.ROWS_RECEIPT)
    commit, sources = rows["source_commit"], rows["source_hashes"]
    assay.verify_sources(commit, sources)
    assay.row_life.validate_independent_audit(sources)
    return commit, sources


def recovery_admission() -> dict[str, Any]:
    stable_v1_failure()
    audit, audit_sha = validate_recovery_audit()
    return {
        "v1_failure_sha256": V1_FAILURE_SHA,
        "v1_success_and_lock_absent": True,
        "recovery_audit_sha256": audit_sha,
        "recovery_audited_source_commit": audit["audited_source_commit"],
        "recovery_source_hashes": audit["audited_source_hashes"],
    }


def composite_protected(authority: dict[str, Any]) -> dict[str, Any]:
    value = _BASE_PROTECTED(authority)
    current = recovery_admission()
    if authority.get("recovery_admission") != current:
        raise RuntimeError("execution recovery admission changed")
    value["recovery_admission"] = current
    return value


def recovery_atomic_json(path: Path, value: Any, *, pre_link_check=None) -> None:
    if path == V2_AUTHORITY:
        value["schema"] = "mlp0_c512_mlp2_full512_composition_v2_authority"
        value["recovery_admission"] = recovery_admission()
    elif path == V2_RECEIPT:
        value["schema"] = "mlp0_c512_mlp2_full512_composition_v2_receipt"
        value["recovery_admission"] = recovery_admission()
    elif path == V2_FAILURE:
        value["schema"] = "mlp0_c512_mlp2_full512_composition_v2_failure"
        value["recovery_admission"] = recovery_admission()
    _BASE_ATOMIC_JSON(path, value, pre_link_check=pre_link_check)


def configure() -> None:
    stable_v1_failure(); validate_recovery_audit(); science_sources()
    assay.AUTHORITY, assay.LEDGER, assay.RESULT = V2_AUTHORITY, V2_LEDGER, V2_RESULT
    assay.RECEIPT, assay.FAILURE, assay.LOCK = V2_RECEIPT, V2_FAILURE, V2_LOCK
    assay.committed_sources = science_sources
    assay.protected_snapshot = composite_protected
    assay.atomic_json = recovery_atomic_json


def main() -> None:
    configure()
    assay.main()


if __name__ == "__main__":
    main()
