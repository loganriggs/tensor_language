#!/usr/bin/env python3
"""One-change row recovery after the v1 preselection ancestry failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import prepare_mlp2_trajectory_robust_r512_v1_eval_rows as v1

base = v1.base
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
AMENDMENT = HERE / "MLP2_TRAJECTORY_ROBUST_R512_V2_ROW_RECOVERY_AMENDMENT.md"
FREEZER = Path(__file__).resolve()
RUNNER = HERE / "recover_mlp2_trajectory_robust_r512_v2_physical_eval.py"
TEST = HERE / "test_recover_mlp2_trajectory_robust_r512_v2_physical_eval.py"
AUDIT = HERE / "mlp2_trajectory_robust_r512_v2_physical_eval_independent_audit.json"
DIRECT_SOURCES = (AMENDMENT, FREEZER, RUNNER, TEST)
SOURCE_PATHS = tuple(dict.fromkeys((*DIRECT_SOURCES, *v1.SOURCE_PATHS)))

CACHE = BQ / ".rowcache_mlp2_trajectory_robust_r512_v2_physical_eval"
RECEIPT = BQ / "mlp2_trajectory_robust_r512_v2_physical_eval_rows_receipt.json"
FAILURE = BQ / "mlp2_trajectory_robust_r512_v2_physical_eval_rows_failure.json"
LOCK = Path("/workspace/runs/.mlp2_trajectory_robust_r512_v2_physical_eval_rows.lock")
RECEIPT_SCHEMA = "mlp2_trajectory_robust_r512_v2_physical_eval_rows"
FAILURE_SCHEMA = "mlp2_trajectory_robust_r512_v2_physical_eval_rows_failure"

V1_FAILURE = v1.FAILURE
V1_FAILURE_SHA = "ef2ff4687fe057fb2d83db0c97429b67d46cc99dc0b422e9d4fa65edd4c4de81"
ORIGINAL_AUDIT = HERE / "mlp2_trajectory_robust_r512_v1_physical_eval_independent_audit.json"
ORIGINAL_AUDIT_SHA = "16b7d002040dd820d4e5fbe5a30209417c381b32e0ad69fe5558535def242e68"
ORIGINAL_AUDITED_COMMIT = "b2d8b31fb2bbb0c69d9cb2d34a22bdd50dde6049"

_BASE_WRITE_JSON = base.write_json_create_only


def file_sha256(path: Path) -> str:
    return base.file_sha256(path)


def stable_json(path: Path, expected: str) -> dict[str, Any]:
    before = file_sha256(path)
    if before != expected:
        raise RuntimeError(f"recovery parent hash changed: {path}")
    raw = path.read_bytes()
    if file_sha256(path) != before or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"recovery parent raced read: {path}")
    return json.loads(raw)


def _v1_absence_paths() -> tuple[Path, ...]:
    # Import here to avoid source import order affecting the outcome-blind freezer.
    import run_mlp2_trajectory_robust_r512_v1_physical_eval as assay
    return (v1.CACHE, v1.RECEIPT, v1.LOCK, assay.AUTHORITY, assay.LEDGER,
            assay.RESULT, assay.RECEIPT, assay.FAILURE, assay.LOCK)


def recovery_admission() -> dict[str, Any]:
    failure = stable_json(V1_FAILURE, V1_FAILURE_SHA)
    audit = stable_json(ORIGINAL_AUDIT, ORIGINAL_AUDIT_SHA)
    if failure.get("schema") != v1.FAILURE_SCHEMA \
            or failure.get("status") != "terminal_failure_no_receipt" \
            or failure.get("cache_exists") is not False \
            or failure.get("receipt_exists") is not False \
            or "merge-base" not in failure.get("error", "") \
            or "origin/main" not in failure.get("error", ""):
        raise RuntimeError("v1 row failure semantics changed")
    if any(path.exists() for path in _v1_absence_paths()):
        raise RuntimeError("v1 row or evaluation output appeared")
    if audit.get("schema") != (
        "mlp2_trajectory_robust_r512_v1_physical_eval_independent_audit"
    ) or audit.get("status") != "GO" or audit.get("outcome_access") is not False \
            or audit.get("audited_source_commit") != ORIGINAL_AUDITED_COMMIT \
            or v1.source_hashes(ORIGINAL_AUDITED_COMMIT) != audit.get("audited_source_hashes"):
        raise RuntimeError("original science audit or source closure changed")
    return {
        "v1_failure_path": str(V1_FAILURE.resolve()),
        "v1_failure_sha256": V1_FAILURE_SHA,
        "v1_failure_schema": v1.FAILURE_SCHEMA,
        "v1_failure_status": "terminal_failure_no_receipt",
        "v1_failure_error_class": "CalledProcessError",
        "failure_phase": "source_ancestry_before_ordered_source_registry_or_harvest",
        "v1_cache_receipt_lock_absent": True,
        "v1_evaluator_outputs_and_lock_absent": True,
        "original_audit_path": str(ORIGINAL_AUDIT.resolve()),
        "original_audit_sha256": ORIGINAL_AUDIT_SHA,
        "original_audited_commit": ORIGINAL_AUDITED_COMMIT,
        "original_source_hashes": audit["audited_source_hashes"],
        "science_changed": False,
    }


def source_hashes(commit: str) -> dict[str, str]:
    if len(SOURCE_PATHS) != len(set(SOURCE_PATHS)) \
            or not set(DIRECT_SOURCES).issubset(SOURCE_PATHS):
        raise RuntimeError("v2 recovery source closure changed")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    output = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"v2 recovery source is not committed: {relative}")
        output[relative] = digest
    return output


def validate_independent_audit(
    sources: Mapping[str, str], path: Path = AUDIT,
) -> tuple[dict[str, Any], str]:
    recovery_admission()
    raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest(); value = json.loads(raw)
    required = {"schema", "status", "outcome_access", "audited_source_commit",
                "audited_source_hashes", "tests_passed", "reviewer"}
    if file_sha256(path) != digest or set(value) != required or value.get("schema") != (
        "mlp2_trajectory_robust_r512_v2_physical_eval_independent_audit"
    ) or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or value.get("audited_source_hashes") != dict(sources) \
            or not isinstance(value.get("tests_passed"), int) or value["tests_passed"] < 1:
        raise RuntimeError("v2 recovery audit is not an exact source-bound GO")
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str) or source_hashes(commit) != dict(sources):
        raise RuntimeError("v2 recovery audit commit binding changed")
    recovery_admission()
    return value, digest


def recovery_write_json(path: Path, value: Mapping[str, Any], *, pre_link_check=None) -> None:
    if path in (RECEIPT, FAILURE):
        value = dict(value)
        value["recovery_admission"] = recovery_admission()
        original_guard = pre_link_check

        def combined_guard() -> None:
            if original_guard is not None:
                original_guard()
            recovery_admission()

        pre_link_check = combined_guard
    _BASE_WRITE_JSON(path, value, pre_link_check=pre_link_check)


def configure() -> None:
    recovery_admission(); v1.configure()
    base.FREEZER, base.RUNNER, base.TEST = FREEZER, RUNNER, TEST
    base.AUDIT, base.SOURCE_PATHS = AUDIT, SOURCE_PATHS
    base.CACHE, base.RECEIPT, base.FAILURE, base.LOCK = CACHE, RECEIPT, FAILURE, LOCK
    base.RECEIPT_SCHEMA, base.FAILURE_SCHEMA = RECEIPT_SCHEMA, FAILURE_SCHEMA
    base.source_hashes = source_hashes
    base.validate_independent_audit = validate_independent_audit
    base.write_json_create_only = recovery_write_json


def freeze():
    configure()
    return base.freeze()


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True, indent=2))
