#!/usr/bin/env python3
"""Fresh, registry-disjoint rows for the frozen C512 × FULL512 cross."""

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
PREREG = HERE / "MLP0_C512_MLP2_FULL512_COMPOSITION_V1_PREREGISTRATION.md"
FREEZER = Path(__file__).resolve()
RUNNER = HERE / "run_mlp0_c512_mlp2_full512_composition_v1.py"
TEST = HERE / "test_mlp0_c512_mlp2_full512_composition_v1.py"
AUDIT = HERE / "mlp0_c512_mlp2_full512_composition_v1_independent_audit.json"
SOURCE_PATHS = tuple(dict.fromkeys((
    PREREG, FREEZER, RUNNER, TEST,
    HERE / "mlp0_native_down_program.py",
    HERE / "test_mlp0_native_down_program.py",
    *base.SOURCE_PATHS,
)))

CACHE = BQ / ".rowcache_mlp0_c512_mlp2_full512_composition_v1"
RECEIPT = BQ / "mlp0_c512_mlp2_full512_composition_v1_rows_receipt.json"
FAILURE = BQ / "mlp0_c512_mlp2_full512_composition_v1_rows_failure.json"
LOCK = Path("/workspace/runs/.mlp0_c512_mlp2_full512_composition_v1_rows.lock")
START_DOCUMENT_INDEX = 110_000
RECEIPT_SCHEMA = "mlp0_c512_mlp2_full512_composition_v1_rows"
FAILURE_SCHEMA = "mlp0_c512_mlp2_full512_composition_v1_rows_failure"


def file_sha256(path: Path) -> str:
    return base.file_sha256(path)


def source_hashes(commit: str) -> dict[str, str]:
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    output = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"composition source is not committed: {relative}")
        output[relative] = digest
    return output


def validate_independent_audit(
    sources: Mapping[str, str], path: Path = AUDIT,
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if file_sha256(path) != digest:
        raise RuntimeError("composition audit changed while reading")
    value = json.loads(raw)
    required = {
        "schema", "status", "outcome_access", "audited_source_commit",
        "audited_source_hashes", "tests_passed", "reviewer",
    }
    if set(value) != required or value.get("schema") != (
        "mlp0_c512_mlp2_full512_composition_v1_independent_audit"
    ) or value.get("status") != "GO" or value.get("outcome_access") is not False \
            or value.get("audited_source_hashes") != dict(sources) \
            or not isinstance(value.get("tests_passed"), int) \
            or value["tests_passed"] < 1:
        raise RuntimeError("composition audit is not an exact source-bound GO")
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str) or source_hashes(commit) != dict(sources):
        raise RuntimeError("composition audit commit binding changed")
    return value, digest


def configure() -> None:
    base.PREREG, base.FREEZER, base.RUNNER, base.TEST = PREREG, FREEZER, RUNNER, TEST
    base.AUDIT, base.SOURCE_PATHS = AUDIT, SOURCE_PATHS
    base.START_DOCUMENT_INDEX = START_DOCUMENT_INDEX
    base.CACHE, base.RECEIPT, base.FAILURE, base.LOCK = CACHE, RECEIPT, FAILURE, LOCK
    base.RECEIPT_SCHEMA, base.FAILURE_SCHEMA = RECEIPT_SCHEMA, FAILURE_SCHEMA
    base.source_hashes = source_hashes
    base.validate_independent_audit = validate_independent_audit


def freeze():
    configure()
    return base.freeze()


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True, indent=2))
