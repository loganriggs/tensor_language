#!/usr/bin/env python3
"""One-change, lineage-bound recovery of the pre-EVALUATION V1 failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import run_mlp2_rank512_refit_v1 as assay

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RECOVERY = Path(__file__).resolve()
AMENDMENT = HERE / "MLP2_RANK512_REFIT_V2_RECOVERY_AMENDMENT.md"
TEST = HERE / "test_recover_mlp2_rank512_refit_v2.py"
AUDIT = HERE / "mlp2_rank512_refit_v2_recovery_independent_audit.json"
RECOVERY_SOURCES = (AMENDMENT, RECOVERY, TEST)

V1_FAILURE = HERE / "mlp2_rank512_refit_v1_failure.json"
V1_AUTHORITY = HERE / "mlp2_rank512_refit_v1_execution_authority.json"
V1_BUNDLE = HERE / "mlp2_rank512_refit_v1_bundle.pt"
V1_LEDGER = HERE / "mlp2_rank512_refit_v1_ledger.pt"
V1_RESULT = HERE / "mlp2_rank512_refit_v1_result.json"
V1_RECEIPT = HERE / "mlp2_rank512_refit_v1_receipt.json"
V1_LOCK = Path("/workspace/runs/.mlp2_rank512_refit_v1.lock")
V1_FAILURE_SHA256 = "9000830570023bcb9f797d6fcd7bfa1e5f73e791e7783ebf2f56009febd79e26"
V1_AUTHORITY_SHA256 = "bf4384bc2c1f3a3858562e5b3980b4cecda7b6e737faeefa27650d7423e352cc"

V2_AUTHORITY = HERE / "mlp2_rank512_refit_v2_recovery_execution_authority.json"
V2_BUNDLE = HERE / "mlp2_rank512_refit_v2_recovery_bundle.pt"
V2_LEDGER = HERE / "mlp2_rank512_refit_v2_recovery_ledger.pt"
V2_RESULT = HERE / "mlp2_rank512_refit_v2_recovery_result.json"
V2_RECEIPT = HERE / "mlp2_rank512_refit_v2_recovery_receipt.json"
V2_FAILURE = HERE / "mlp2_rank512_refit_v2_recovery_failure.json"
V2_LOCK = Path("/workspace/runs/.mlp2_rank512_refit_v2_recovery.lock")

_BASE_PROTECTED = assay.protected_snapshot
_V1_SOURCE_HASHES: dict[str, str] | None = None


def file_sha256(path: Path) -> str:
    return assay.file_sha256(path)


def _committed_blob(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def validate_recovery_audit() -> tuple[dict[str, Any], str]:
    if not AUDIT.is_file():
        raise RuntimeError("independent recovery audit is absent")
    raw = AUDIT.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if file_sha256(AUDIT) != digest:
        raise RuntimeError("recovery audit changed while reading")
    value = json.loads(raw)
    if set(value) != {
        "schema", "status", "outcome_access", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    } or value.get("schema") != "mlp2_rank512_refit_v2_recovery_independent_audit" \
            or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or not isinstance(value.get("tests_passed"), int) or value["tests_passed"] < 1:
        raise RuntimeError("independent recovery audit is not an exact GO")
    commit = value.get("audited_source_commit")
    hashes = value.get("audited_source_hashes")
    expected_paths = {str(path.relative_to(ROOT)) for path in RECOVERY_SOURCES}
    if not isinstance(commit, str) or not isinstance(hashes, dict) or set(hashes) != expected_paths:
        raise RuntimeError("recovery audit source family changed")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    for path in RECOVERY_SOURCES:
        relative = str(path.relative_to(ROOT))
        expected = hashlib.sha256(_committed_blob(commit, relative)).hexdigest()
        if hashes[relative] != expected or file_sha256(path) != expected:
            raise RuntimeError("recovery source changed after audit")
    return value, digest


def v1_lineage_snapshot() -> dict[str, Any]:
    global _V1_SOURCE_HASHES
    failure, _ = assay.stable_json(V1_FAILURE, V1_FAILURE_SHA256)
    authority, _ = assay.stable_json(V1_AUTHORITY, V1_AUTHORITY_SHA256)
    if failure.get("status") != "terminal_failure_no_receipt" or (
        failure.get("bundle_exists") is not False
    ) or failure.get("evaluation_may_have_opened") is not False or (
        failure.get("artifact_hashes") != {}
    ) or failure.get("authority_sha256") != V1_AUTHORITY_SHA256:
        raise RuntimeError("V1 is not the exact pre-candidate/pre-EVALUATION failure")
    absent = (V1_BUNDLE, V1_LEDGER, V1_RESULT, V1_RECEIPT, V1_LOCK)
    if any(path.exists() for path in absent):
        raise RuntimeError("a V1 success or lock artifact appeared")
    sources = failure.get("source_hashes")
    if not isinstance(sources, dict) or set(sources) != {
        str(path.relative_to(ROOT)) for path in assay.SOURCE_PATHS
    }:
        raise RuntimeError("V1 source family changed")
    for path in assay.SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        if file_sha256(path) != sources[relative]:
            raise RuntimeError("V1 scientific source bytes changed")
    _V1_SOURCE_HASHES = dict(sources)
    audit, audit_hash = validate_recovery_audit()
    return {
        "v1_authority_sha256": V1_AUTHORITY_SHA256,
        "v1_failure_sha256": V1_FAILURE_SHA256,
        "v1_success_and_lock_absent": [str(path) for path in absent],
        "v1_source_hashes": sources,
        "v1_source_commit": authority["source_commit"],
        "recovery_audit_sha256": audit_hash,
        "recovery_audited_source_commit": audit["audited_source_commit"],
        "recovery_source_hashes": audit["audited_source_hashes"],
    }


def stable_v1_sources() -> tuple[str, dict[str, str]]:
    lineage = v1_lineage_snapshot()
    sources = lineage["v1_source_hashes"]
    # Return the exact source commit already spent by V1; branch-pointer movement is
    # irrelevant when every executable byte is checked directly.
    return lineage["v1_source_commit"], sources


def composite_protected(authority: dict[str, Any], sources: dict[str, str]):
    value = _BASE_PROTECTED(authority, sources)
    value["recovery_lineage"] = v1_lineage_snapshot()
    return value


def configure_namespace() -> None:
    assay.AUTHORITY, assay.BUNDLE, assay.LEDGER = V2_AUTHORITY, V2_BUNDLE, V2_LEDGER
    assay.RESULT, assay.RECEIPT, assay.FAILURE = V2_RESULT, V2_RECEIPT, V2_FAILURE
    assay.LOCK = V2_LOCK
    assay.committed_sources = stable_v1_sources
    assay.protected_snapshot = composite_protected


def terminal_boundary_guard(
    claim: assay.row_life.RunClaim, *, expected_lineage: dict[str, Any],
    artifact_hashes: dict[Path, str], publishing: str,
) -> None:
    if publishing not in {"receipt", "failure"}:
        raise ValueError("unknown recovery terminal")
    if V2_RECEIPT.exists() or V2_FAILURE.exists():
        raise RuntimeError("competing recovery terminal appeared")
    if v1_lineage_snapshot() != expected_lineage:
        raise RuntimeError("recovery lineage changed at terminal boundary")
    for path, expected in artifact_hashes.items():
        assay.stable_bytes(path, expected)
    # Repeat both checks after the potentially expensive lineage/artifact replay;
    # the owned claim is deliberately the last callback operation before linking.
    if V2_RECEIPT.exists() or V2_FAILURE.exists() or (
        v1_lineage_snapshot() != expected_lineage
    ):
        raise RuntimeError("recovery terminal or lineage raced publication")
    assay.row_life.require_claim(claim, V2_LOCK)


def prepare_recovery_authority(claim: assay.row_life.RunClaim) -> dict[str, Any]:
    source_commit, sources = stable_v1_sources()
    lineage = v1_lineage_snapshot()
    v1_audit, v1_audit_hash = assay.row_life.validate_independent_audit(sources)
    rows_receipt, rows_hash = assay.stable_json(assay.ROWS_RECEIPT)
    assay.validate_row_receipt(rows_receipt, sources)
    mean_receipt, mean_receipt_hash = assay.stable_json(assay.MEAN_RECEIPT)
    mean_bundle, mean_hash = assay.stable_torch(
        assay.MEAN_BUNDLE, mean_receipt.get("bundle_sha256"))
    assay.validate_mean_parent(mean_receipt, mean_bundle)
    entries = rows_receipt["entries"]
    authority = {
        "schema": "mlp2_rank512_refit_v1_execution_authority",
        "status": "spent_before_train_or_model_access",
        "source_commit": source_commit, "source_hashes": sources,
        "independent_audit_sha256": v1_audit_hash,
        "independent_audit_reviewer": v1_audit["reviewer"],
        "parents": {
            "rows_receipt": rows_hash,
            "train_rows": entries["TRAIN"]["file_sha256"],
            "evaluation_rows": entries["EVALUATION"]["file_sha256"],
            "mean_receipt": mean_receipt_hash, "mean_bundle": mean_hash,
        },
        "recovery_admission": lineage,
        "outcome_access_before_authority": {
            "train_rows_opened": False, "evaluation_rows_opened": False,
            "mean_bundle_semantically_validated": True, "checkpoint_loaded": False,
            "model_forward_calls": 0,
        },
    }

    def guard() -> None:
        assay.row_life.require_claim(claim, V2_LOCK)
        if any(path.exists() for path in (
            V2_AUTHORITY, V2_BUNDLE, V2_LEDGER, V2_RESULT, V2_RECEIPT, V2_FAILURE,
        )) or v1_lineage_snapshot() != lineage:
            raise RuntimeError("recovery admission changed before authority")
        if any(path.exists() for path in (
            V2_AUTHORITY, V2_BUNDLE, V2_LEDGER, V2_RESULT, V2_RECEIPT, V2_FAILURE,
        )):
            raise RuntimeError("recovery namespace raced authority publication")
        assay.row_life.require_claim(claim, V2_LOCK)
    assay.atomic_json(V2_AUTHORITY, authority, pre_link_check=guard)
    return authority


def main() -> None:
    configure_namespace()
    if any(path.exists() for path in (
        V2_AUTHORITY, V2_BUNDLE, V2_LEDGER, V2_RESULT, V2_RECEIPT, V2_FAILURE, V2_LOCK,
    )):
        raise RuntimeError("V2 recovery namespace already exists")
    claim = assay.row_life.acquire_claim(V2_LOCK)
    authority: dict[str, Any] | None = None
    protected: dict[str, Any] | None = None
    try:
        authority = prepare_recovery_authority(claim)
        result, _, protected = assay.run(claim)

        def result_guard() -> None:
            assay.verify_protected_snapshot(
                protected, authority, authority["source_hashes"], claim)
            if V2_RESULT.exists() or V2_RECEIPT.exists() or V2_FAILURE.exists():
                raise RuntimeError("recovery terminal raced result publication")
        assay.atomic_json(V2_RESULT, result, pre_link_check=result_guard)
        reloaded_result, result_hash = assay.stable_json(V2_RESULT)
        reloaded_bundle, bundle_hash = assay.stable_torch(V2_BUNDLE)
        reloaded_bundle = assay.validate_bundle(
            reloaded_bundle, result["parents"], authority["source_hashes"],
            expected_commit=authority["source_commit"])
        reloaded_ledger, ledger_hash = assay.stable_torch(V2_LEDGER)
        reloaded_ledger = assay.validate_ledger(
            reloaded_ledger, bundle_hash=bundle_hash,
            evaluation_hash=authority["parents"]["evaluation_rows"])
        replay = assay.derive_result(
            reloaded_ledger, reloaded_bundle, bundle_hash=bundle_hash,
            ledger_hash=ledger_hash, runtime_seconds=result["runtime_seconds"])
        if reloaded_result != result or replay != result:
            raise RuntimeError("recovery result semantic replay changed")
        receipt = {
            "schema": "mlp2_rank512_refit_v2_recovery_receipt",
            "status": "result_complete_receipt_last",
            "authority_sha256": file_sha256(V2_AUTHORITY),
            "bundle_sha256": bundle_hash, "ledger_sha256": ledger_hash,
            "result_sha256": result_hash,
            "source_commit": authority["source_commit"],
            "source_hashes": authority["source_hashes"],
            "recovery_admission": authority["recovery_admission"],
            "evaluation_opened": True,
        }

        def receipt_guard() -> None:
            assay.verify_protected_snapshot(
                protected, authority, authority["source_hashes"], claim)
            terminal_boundary_guard(
                claim, expected_lineage=authority["recovery_admission"],
                artifact_hashes={V2_AUTHORITY: receipt["authority_sha256"],
                                 V2_BUNDLE: bundle_hash, V2_LEDGER: ledger_hash,
                                 V2_RESULT: result_hash}, publishing="receipt")
        assay.atomic_json(V2_RECEIPT, receipt, pre_link_check=receipt_guard)
        if assay.stable_json(V2_RECEIPT)[0] != receipt:
            raise RuntimeError("recovery receipt replay changed")
        print(json.dumps(result, sort_keys=True, indent=2))
    except BaseException as exc:
        failure = {
            "schema": "mlp2_rank512_refit_v2_recovery_failure",
            "status": "terminal_failure_no_receipt", "error": repr(exc),
            "authority_sha256": file_sha256(V2_AUTHORITY) if V2_AUTHORITY.is_file() else None,
            "recovery_lineage": v1_lineage_snapshot(),
            "protected_snapshot": protected,
            "artifact_hashes": {path.name: file_sha256(path) for path in (
                V2_BUNDLE, V2_LEDGER, V2_RESULT) if path.is_file()},
            "evaluation_may_have_opened": V2_BUNDLE.exists(),
        }
        if not V2_RECEIPT.exists() and not V2_FAILURE.exists():
            frozen_artifacts = dict(failure["artifact_hashes"])

            def failure_guard() -> None:
                current = {path: frozen_artifacts[path.name] for path in (
                    V2_BUNDLE, V2_LEDGER, V2_RESULT) if path.name in frozen_artifacts}
                terminal_boundary_guard(
                    claim, expected_lineage=failure["recovery_lineage"],
                    artifact_hashes=current, publishing="failure")
            assay.atomic_json(V2_FAILURE, failure, pre_link_check=failure_guard)
        raise
    finally:
        assay.row_life.release_claim(claim, V2_LOCK)


if __name__ == "__main__":
    main()
