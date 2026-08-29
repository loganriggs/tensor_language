#!/usr/bin/env python3
"""One-change namespace recovery for the pre-EVALUATION MLP2 refit v1 failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import run_mlp2_rank512_refit_v1 as assay

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECOVERY = Path(__file__).resolve()
AMENDMENT = HERE / "MLP2_RANK512_REFIT_V2_RECOVERY_AMENDMENT.md"
TEST = HERE / "test_recover_mlp2_rank512_refit_v2.py"
AUDIT = HERE / "mlp2_rank512_refit_v2_recovery_independent_audit.json"
V1_FAILURE = HERE / "mlp2_rank512_refit_v1_failure.json"
V1_AUTHORITY = HERE / "mlp2_rank512_refit_v1_execution_authority.json"
V1_FAILURE_SHA256 = "9000830570023bcb9f797d6fcd7bfa1e5f73e791e7783ebf2f56009febd79e26"
RECOVERY_SOURCES = (AMENDMENT, RECOVERY, TEST)


def file_sha256(path: Path) -> str:
    return assay.file_sha256(path)


def recovery_source_hashes(commit: str) -> dict[str, str]:
    output = {}
    for path in RECOVERY_SOURCES:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"uncommitted recovery source: {relative}")
        output[relative] = digest
    return output


def validate_admission() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    origin = subprocess.check_output(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True,
    ).strip()
    if commit != origin:
        raise RuntimeError("recovery admission requires synchronized HEAD/origin")
    sources = recovery_source_hashes(commit)
    if not AUDIT.is_file():
        raise RuntimeError("independent recovery audit is absent")
    audit_raw = AUDIT.read_bytes()
    audit = json.loads(audit_raw)
    if set(audit) != {
        "schema", "status", "outcome_access", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    } or audit.get("schema") != "mlp2_rank512_refit_v2_recovery_independent_audit" \
            or audit.get("status") != "GO" or audit.get("outcome_access") is not False \
            or audit.get("audited_source_hashes") != sources:
        raise RuntimeError("independent recovery audit binding changed")
    if file_sha256(V1_FAILURE) != V1_FAILURE_SHA256:
        raise RuntimeError("V1 failure bytes changed")
    failure = json.loads(V1_FAILURE.read_bytes())
    if failure.get("status") != "terminal_failure_no_receipt" or (
        failure.get("bundle_exists") is not False
    ) or failure.get("evaluation_may_have_opened") is not False or (
        failure.get("artifact_hashes") != {}
    ) or failure.get("source_hashes") != assay.committed_sources()[1] or (
        failure.get("authority_sha256") != file_sha256(V1_AUTHORITY)
    ):
        raise RuntimeError("V1 was not an exact pre-candidate/pre-EVALUATION failure")
    return {
        "recovery_commit": commit, "recovery_source_hashes": sources,
        "recovery_audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "v1_failure_sha256": V1_FAILURE_SHA256,
    }


def configure_namespace() -> None:
    assay.AUTHORITY = HERE / "mlp2_rank512_refit_v2_recovery_execution_authority.json"
    assay.BUNDLE = HERE / "mlp2_rank512_refit_v2_recovery_bundle.pt"
    assay.LEDGER = HERE / "mlp2_rank512_refit_v2_recovery_ledger.pt"
    assay.RESULT = HERE / "mlp2_rank512_refit_v2_recovery_result.json"
    assay.RECEIPT = HERE / "mlp2_rank512_refit_v2_recovery_receipt.json"
    assay.FAILURE = HERE / "mlp2_rank512_refit_v2_recovery_failure.json"
    assay.LOCK = Path("/workspace/runs/.mlp2_rank512_refit_v2_recovery.lock")


def main() -> None:
    admission = validate_admission()
    configure_namespace()
    assay.main()
    receipt = json.loads(assay.RECEIPT.read_bytes())
    receipt["recovery_admission"] = admission
    # The assay receipt is already immutable; write a sibling binding rather than
    # mutating it after publication.
    assay.atomic_json(
        HERE / "mlp2_rank512_refit_v2_recovery_admission_receipt.json",
        {"schema": "mlp2_rank512_refit_v2_recovery_admission_receipt",
         "assay_receipt_sha256": file_sha256(assay.RECEIPT), **admission},
    )


if __name__ == "__main__":
    main()
