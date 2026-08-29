#!/usr/bin/env python3
"""V3 execution-only recovery from V2's pre-open shared-HEAD race."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import recover_mlp2_trajectory_robust_r512_v2_physical_eval as v2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AMENDMENT = HERE / "MLP2_TRAJECTORY_ROBUST_R512_V3_EXECUTION_RECOVERY_AMENDMENT.md"
RUNNER = Path(__file__).resolve()
TEST = HERE / "test_recover_mlp2_trajectory_robust_r512_v3_physical_eval.py"
AUDIT = HERE / "mlp2_trajectory_robust_r512_v3_physical_eval_independent_audit.json"
SOURCE_PATHS = tuple(dict.fromkeys((AMENDMENT, RUNNER, TEST, *v2.row_life.SOURCE_PATHS)))

V2_FAILURE = v2.FAILURE
V2_FAILURE_SHA = "a66b51c6bcc2c2e66e9f372344e41d38d9743cea5c5973c529215a9323e90ae5"
V2_ROWS_RECEIPT = v2.ROWS_RECEIPT
V2_ROWS_RECEIPT_SHA = "efe44941388878cfac1467508e2763b6a5db9b881be224c9db86314deaa61c2a"
V2_ROWS = v2.BQ / ".rowcache_mlp2_trajectory_robust_r512_v2_physical_eval/evaluation_192.pt"
V2_ROWS_SHA = "3e0bdab49a3413423cb1bae71fbcb8ab627bc2ace561e81ea6fec35fa94e02d9"

AUTHORITY = HERE / "mlp2_trajectory_robust_r512_v3_physical_eval_authority.json"
LEDGER = HERE / "mlp2_trajectory_robust_r512_v3_physical_eval_ledger.pt"
RESULT = HERE / "mlp2_trajectory_robust_r512_v3_physical_eval_result.json"
RECEIPT = HERE / "mlp2_trajectory_robust_r512_v3_physical_eval_receipt.json"
FAILURE = HERE / "mlp2_trajectory_robust_r512_v3_physical_eval_failure.json"
LOCK = Path("/workspace/runs/.mlp2_trajectory_robust_r512_v3_physical_eval.lock")

AUTHORITY_SCHEMA = "mlp2_trajectory_robust_r512_v3_physical_eval_authority"
LEDGER_SCHEMA = "mlp2_trajectory_robust_r512_v3_physical_eval_ledger"
RESULT_SCHEMA = "mlp2_trajectory_robust_r512_v3_physical_eval_result"
RECEIPT_SCHEMA = "mlp2_trajectory_robust_r512_v3_physical_eval_receipt"
FAILURE_SCHEMA = "mlp2_trajectory_robust_r512_v3_physical_eval_failure"

_V2_ADMISSION = v2.row_life.recovery_admission


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_json(path: Path, expected: str) -> tuple[dict[str, Any], str]:
    before = file_sha256(path)
    if before != expected:
        raise RuntimeError(f"v3 recovery parent hash changed: {path}")
    raw = path.read_bytes()
    if file_sha256(path) != before or hashlib.sha256(raw).hexdigest() != before:
        raise RuntimeError(f"v3 recovery parent raced read: {path}")
    return json.loads(raw), before


def _v2_absence_paths() -> tuple[Path, ...]:
    return (v2.AUTHORITY, v2.LEDGER, v2.RESULT, v2.RECEIPT, v2.LOCK)


def recovery_admission() -> dict[str, Any]:
    parent = _V2_ADMISSION()
    failure, _ = stable_json(V2_FAILURE, V2_FAILURE_SHA)
    if failure.get("schema") != v2.FAILURE_SCHEMA \
            or failure.get("status") != "terminal_failure_no_receipt" \
            or failure.get("authority_exists") is not False \
            or failure.get("evaluation_may_have_opened") is not False \
            or any(value is not None for value in failure.get("artifact_snapshot", {}).values()) \
            or "merge-base" not in failure.get("error", "") \
            or "origin/main" not in failure.get("error", ""):
        raise RuntimeError("v2 execution failure semantics changed")
    if any(path.exists() for path in _v2_absence_paths()):
        raise RuntimeError("v2 evaluation output or lock appeared")
    rows_receipt, _ = stable_json(V2_ROWS_RECEIPT, V2_ROWS_RECEIPT_SHA)
    if rows_receipt.get("outcome_access") != {"model_loaded": False, "training_run": False} \
            or rows_receipt.get("entries", {}).get("EVALUATION", {}).get("file_sha256") != V2_ROWS_SHA \
            or file_sha256(V2_ROWS) != V2_ROWS_SHA:
        raise RuntimeError("v2 rows or outcome-blind receipt changed")
    if any(path.exists() for path in _v2_absence_paths()):
        raise RuntimeError("v2 evaluation output or lock appeared during replay")
    return {
        "v2_failure_sha256": V2_FAILURE_SHA,
        "v2_failure_status": "terminal_failure_no_receipt",
        "v2_authority_exists": False,
        "v2_evaluation_may_have_opened": False,
        "v2_evaluation_outputs_and_lock_absent": True,
        "v2_rows_receipt_sha256": V2_ROWS_RECEIPT_SHA,
        "v2_rows_sha256": V2_ROWS_SHA,
        "science_changed": False,
        "parent_v2_row_recovery_admission": parent,
    }


def source_hashes(commit: str) -> dict[str, str]:
    if len(SOURCE_PATHS) != len(set(SOURCE_PATHS)):
        raise RuntimeError("v3 source closure contains duplicates")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    output = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"v3 source is not committed: {relative}")
        output[relative] = digest
    return output


def validate_independent_audit(
    sources: Mapping[str, str], path: Path = AUDIT,
) -> tuple[dict[str, Any], str]:
    recovery_admission()
    raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest(); value = json.loads(raw)
    required = {"schema", "status", "outcome_access", "audited_source_commit",
                "audited_source_hashes", "tests_passed", "reviewer"}
    if file_sha256(path) != digest or set(value) != required \
            or value.get("schema") != "mlp2_trajectory_robust_r512_v3_physical_eval_independent_audit" \
            or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or value.get("audited_source_hashes") != dict(sources) \
            or not isinstance(value.get("tests_passed"), int) or value["tests_passed"] < 1:
        raise RuntimeError("v3 audit is not an exact source-bound GO")
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str) or source_hashes(commit) != dict(sources):
        raise RuntimeError("v3 audit commit binding changed")
    recovery_admission()
    return value, digest


def committed_sources() -> tuple[str, dict[str, str]]:
    value = json.loads(AUDIT.read_text())
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str):
        raise RuntimeError("v3 audit lacks source commit")
    return commit, source_hashes(commit)


def validate_row_receipt(value: Any, _sources: dict[str, str]) -> dict[str, Any]:
    expected, _ = stable_json(V2_ROWS_RECEIPT, V2_ROWS_RECEIPT_SHA)
    if value != expected or file_sha256(V2_ROWS) != V2_ROWS_SHA:
        raise RuntimeError("v3 inherited row receipt changed")
    recovery_admission()
    return value


def configure() -> None:
    recovery_admission()
    for name, value in {
        "AUTHORITY": AUTHORITY, "LEDGER": LEDGER, "RESULT": RESULT,
        "RECEIPT": RECEIPT, "FAILURE": FAILURE, "LOCK": LOCK,
        "AUTHORITY_SCHEMA": AUTHORITY_SCHEMA, "LEDGER_SCHEMA": LEDGER_SCHEMA,
        "RESULT_SCHEMA": RESULT_SCHEMA, "RECEIPT_SCHEMA": RECEIPT_SCHEMA,
        "FAILURE_SCHEMA": FAILURE_SCHEMA,
    }.items():
        setattr(v2, name, value)
    v2.committed_sources = committed_sources
    v2.validate_row_receipt = validate_row_receipt
    v2.row_life.source_hashes = source_hashes
    v2.row_life.validate_independent_audit = validate_independent_audit
    v2.row_life.recovery_admission = recovery_admission


def main() -> None:
    configure()
    v2.main()


if __name__ == "__main__":
    main()
