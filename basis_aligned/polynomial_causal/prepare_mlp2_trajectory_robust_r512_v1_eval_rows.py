#!/usr/bin/env python3
"""Fresh row-family wrapper for the frozen robust-r512 physical evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import prepare_mlp2_rank512_refit_v1_rows as base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
ADDENDUM = HERE / "MLP2_TRAJECTORY_ROBUST_R512_V1_PHYSICAL_EVALUATION_ADDENDUM.md"
PARENT_PREREG = HERE / "MLP2_TRAJECTORY_ROBUST_R512_V1_PREREGISTRATION.md"
FREEZER = Path(__file__).resolve()
RUNNER = HERE / "run_mlp2_trajectory_robust_r512_v1_physical_eval.py"
TEST = HERE / "test_mlp2_trajectory_robust_r512_v1_physical_eval.py"
AUDIT = HERE / "mlp2_trajectory_robust_r512_v1_physical_eval_independent_audit.json"
DIRECT_SOURCES = (ADDENDUM, PARENT_PREREG, FREEZER, RUNNER, TEST)
SOURCE_PATHS = tuple(dict.fromkeys((*DIRECT_SOURCES, *base.SOURCE_PATHS)))

CACHE = BQ / ".rowcache_mlp2_trajectory_robust_r512_v1_physical_eval"
RECEIPT = BQ / "mlp2_trajectory_robust_r512_v1_physical_eval_rows_receipt.json"
FAILURE = BQ / "mlp2_trajectory_robust_r512_v1_physical_eval_rows_failure.json"
LOCK = Path("/workspace/runs/.mlp2_trajectory_robust_r512_v1_physical_eval_rows.lock")
START_DOCUMENT_INDEX = 120_000
RECEIPT_SCHEMA = "mlp2_trajectory_robust_r512_v1_physical_eval_rows"
FAILURE_SCHEMA = "mlp2_trajectory_robust_r512_v1_physical_eval_rows_failure"


def file_sha256(path: Path) -> str:
    return base.file_sha256(path)


def source_hashes(commit: str) -> dict[str, str]:
    if len(SOURCE_PATHS) != len(set(SOURCE_PATHS)) \
            or not set(DIRECT_SOURCES).issubset(SOURCE_PATHS):
        raise RuntimeError("physical-eval direct source closure changed")
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    output = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"physical-eval source is not committed: {relative}")
        output[relative] = digest
    return output


def validate_independent_audit(
    sources: Mapping[str, str], path: Path = AUDIT,
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    required = {"schema", "status", "outcome_access", "audited_source_commit",
                "audited_source_hashes", "tests_passed", "reviewer"}
    if file_sha256(path) != digest or set(value) != required or value.get("schema") != (
        "mlp2_trajectory_robust_r512_v1_physical_eval_independent_audit"
    ) or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or value.get("audited_source_hashes") != dict(sources) \
            or not isinstance(value.get("tests_passed"), int) \
            or value["tests_passed"] < 1:
        raise RuntimeError("physical-eval audit is not an exact source-bound GO")
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str) or source_hashes(commit) != dict(sources):
        raise RuntimeError("physical-eval audit commit binding changed")
    return value, digest


def configure() -> None:
    base.PREREG, base.FREEZER, base.RUNNER, base.TEST = ADDENDUM, FREEZER, RUNNER, TEST
    base.AUDIT, base.SOURCE_PATHS = AUDIT, SOURCE_PATHS
    base.START_DOCUMENT_INDEX = START_DOCUMENT_INDEX
    base.DOCUMENTS_PER_ROLE, base.TOTAL_DOCUMENTS = 192, 192
    base.ROLE_NAMES = ("EVALUATION",)
    base.ROLE_AUTHORIZATIONS = {
        "EVALUATION": {"authorized_for_training": False,
                       "authorized_for_evaluation": True},
    }
    base.CACHE, base.RECEIPT, base.FAILURE, base.LOCK = CACHE, RECEIPT, FAILURE, LOCK
    base.RECEIPT_SCHEMA, base.FAILURE_SCHEMA = RECEIPT_SCHEMA, FAILURE_SCHEMA
    base.source_hashes = source_hashes
    base.validate_independent_audit = validate_independent_audit


def freeze():
    configure()
    return base.freeze()


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True, indent=2))
