#!/usr/bin/env python3
"""Lineage-bound row-freezer recovery after the v1 registry-proof failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import prepare_mlp0_c512_mlp2_full512_composition_v1_rows as v1

base = v1.base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
AMENDMENT = HERE / "MLP0_C512_MLP2_FULL512_COMPOSITION_V2_ROW_RECOVERY_AMENDMENT.md"
FREEZER = Path(__file__).resolve()
RUNNER = HERE / "run_mlp0_c512_mlp2_full512_composition_v1.py"
TEST = HERE / "test_mlp0_c512_mlp2_full512_composition_v1.py"
AUDIT = HERE / "mlp0_c512_mlp2_full512_composition_v2_independent_audit.json"
SOURCE_PATHS = tuple(dict.fromkeys((AMENDMENT, FREEZER, *v1.SOURCE_PATHS)))

CACHE = BQ / ".rowcache_mlp0_c512_mlp2_full512_composition_v2"
RECEIPT = BQ / "mlp0_c512_mlp2_full512_composition_v2_rows_receipt.json"
FAILURE = BQ / "mlp0_c512_mlp2_full512_composition_v2_rows_failure.json"
LOCK = Path("/workspace/runs/.mlp0_c512_mlp2_full512_composition_v2_rows.lock")
RECEIPT_SCHEMA = "mlp0_c512_mlp2_full512_composition_v2_rows"
FAILURE_SCHEMA = "mlp0_c512_mlp2_full512_composition_v2_rows_failure"

V1_FAILURE = BQ / "mlp0_c512_mlp2_full512_composition_v1_rows_failure.json"
V1_FAILURE_SHA = "0c760bbd6798960eb037dbdd01e820fa1a924c98aff6e2b645c01964592055c3"


def file_sha256(path: Path) -> str:
    return v1.file_sha256(path)


def source_hashes(commit: str) -> dict[str, str]:
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    output = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"composition v2 source is not committed: {relative}")
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
        "mlp0_c512_mlp2_full512_composition_v2_independent_audit"
    ) or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or value.get("audited_source_hashes") != dict(sources) \
            or not isinstance(value.get("tests_passed"), int) or value["tests_passed"] < 1:
        raise RuntimeError("composition v2 audit is not an exact source-bound GO")
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str) or source_hashes(commit) != dict(sources):
        raise RuntimeError("composition v2 audit commit binding changed")
    return value, digest


def validate_v1_failure() -> None:
    if file_sha256(V1_FAILURE) != V1_FAILURE_SHA or v1.CACHE.exists() \
            or v1.RECEIPT.exists():
        raise RuntimeError("v1 row failure lineage changed")
    failure = json.loads(V1_FAILURE.read_text())
    if failure.get("status") != "terminal_failure_no_receipt" \
            or failure.get("cache_exists") is not False \
            or failure.get("receipt_exists") is not False \
            or "unregistered missing row-like reference" not in failure.get("error", ""):
        raise RuntimeError("v1 row failure is not the registered preselection failure")


def configure() -> None:
    validate_v1_failure()
    v1.configure()
    base = v1.base
    base.FREEZER, base.RUNNER, base.TEST = FREEZER, RUNNER, TEST
    base.AUDIT, base.SOURCE_PATHS = AUDIT, SOURCE_PATHS
    base.CACHE, base.RECEIPT, base.FAILURE, base.LOCK = CACHE, RECEIPT, FAILURE, LOCK
    base.RECEIPT_SCHEMA, base.FAILURE_SCHEMA = RECEIPT_SCHEMA, FAILURE_SCHEMA
    base.source_hashes = source_hashes
    base.validate_independent_audit = validate_independent_audit


def freeze():
    configure()
    return v1.base.freeze()


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True, indent=2))
