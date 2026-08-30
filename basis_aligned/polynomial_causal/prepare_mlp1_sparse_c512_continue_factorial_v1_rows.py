#!/usr/bin/env python3
"""Outcome-blind fresh roles for the MLP1 sparse/C512/CONTINUE factorial."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import prepare_mlp2_rank512_refit_v1_rows as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"

PREREG = HERE / "MLP1_SPARSE_C512_CONTINUE_FACTORIAL_V1_PREREGISTRATION.md"
RUNNER = HERE / "run_mlp1_sparse_c512_continue_factorial_v1_fit.py"
RUNNER_TEST = HERE / "test_run_mlp1_sparse_c512_continue_factorial_v1_fit.py"
PROGRAM = HERE / "mlp1_sparse_down_program_v1.py"
PROGRAM_TEST = HERE / "test_mlp1_sparse_down_program_v1.py"
FACADE = HERE / "bilin18_observed_model_facade.py"
FACADE_TEST = HERE / "test_bilin18_observed_model_facade.py"
FREEZER = Path(__file__).resolve()
TEST = HERE / "test_prepare_mlp1_sparse_c512_continue_factorial_v1_rows.py"
AUDIT = HERE / "mlp1_sparse_c512_continue_factorial_v1_rows_independent_audit.json"

DIRECT_SOURCES = (
    PREREG,
    RUNNER,
    RUNNER_TEST,
    PROGRAM,
    PROGRAM_TEST,
    FACADE,
    FACADE_TEST,
    FREEZER,
    TEST,
    HERE / "prepare_mlp2_rank512_refit_v1_rows.py",
)
SOURCE_PATHS = tuple(dict.fromkeys((*DIRECT_SOURCES, *base.BASE.SOURCE_PATHS)))

START_DOCUMENT_INDEX = 122_000
DOCUMENTS_PER_ROLE = 96
TOTAL_DOCUMENTS = 288
ROLE_NAMES = ("FIT", "SELECT", "FINAL")
ROLE_AUTHORIZATIONS = {
    "FIT": {
        "authorized_for_training": True,
        "authorized_for_selection": False,
        "authorized_for_final": False,
    },
    "SELECT": {
        "authorized_for_training": False,
        "authorized_for_selection": True,
        "authorized_for_final": False,
    },
    "FINAL": {
        "authorized_for_training": False,
        "authorized_for_selection": False,
        "authorized_for_final": True,
    },
}

CACHE = BQ / ".rowcache_mlp1_sparse_c512_continue_factorial_v1"
RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
FAILURE = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_failure.json"
LOCK = Path("/workspace/runs/.mlp1_sparse_c512_continue_factorial_v1_rows.lock")
RECEIPT_SCHEMA = "mlp1_sparse_c512_continue_factorial_v1_rows"
FAILURE_SCHEMA = "mlp1_sparse_c512_continue_factorial_v1_rows_failure"


def file_sha256(path: Path) -> str:
    return base.file_sha256(path)


def source_hashes(commit: str) -> dict[str, str]:
    if len(SOURCE_PATHS) != len(set(SOURCE_PATHS)) or not set(
        DIRECT_SOURCES
    ).issubset(SOURCE_PATHS) or not set(base.BASE.SOURCE_PATHS).issubset(SOURCE_PATHS):
        raise RuntimeError("MLP1 factorial row-freezer source closure changed")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT,
        check=True,
    )
    output: dict[str, str] = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"uncommitted MLP1 factorial row-freezer source: {relative}")
        output[relative] = digest
    return output


def validate_independent_audit(
    sources: Mapping[str, str], path: Path = AUDIT,
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise RuntimeError("independent MLP1 factorial row audit is absent")
    before = file_sha256(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if file_sha256(path) != before or digest != before:
        raise RuntimeError("independent MLP1 factorial row audit changed while reading")
    value = json.loads(raw)
    required = {
        "schema",
        "status",
        "outcome_access",
        "audited_source_commit",
        "audited_source_hashes",
        "tests_passed",
        "reviewer",
    }
    if (
        set(value) != required
        or value.get("schema")
        != "mlp1_sparse_c512_continue_factorial_v1_rows_independent_audit"
        or value.get("status") != "GO"
        or value.get("outcome_access") is not False
        or not isinstance(value.get("tests_passed"), int)
        or value["tests_passed"] < 1
        or not isinstance(value.get("reviewer"), str)
        or not value["reviewer"]
        or value.get("audited_source_hashes") != dict(sources)
    ):
        raise RuntimeError(
            "independent MLP1 factorial row audit is not an exact source-bound GO"
        )
    commit = value.get("audited_source_commit")
    if not isinstance(commit, str) or source_hashes(commit) != dict(sources):
        raise RuntimeError("independent MLP1 factorial row audit commit binding changed")
    return value, digest


_CONFIGURED_NAMES = (
    "PREREG",
    "FREEZER",
    "TEST",
    "AUDIT",
    "SOURCE_PATHS",
    "START_DOCUMENT_INDEX",
    "DOCUMENTS_PER_ROLE",
    "TOTAL_DOCUMENTS",
    "ROLE_NAMES",
    "ROLE_AUTHORIZATIONS",
    "CACHE",
    "RECEIPT",
    "FAILURE",
    "LOCK",
    "RECEIPT_SCHEMA",
    "FAILURE_SCHEMA",
    "source_hashes",
    "validate_independent_audit",
)


@contextmanager
def configured_base():
    """Install this immutable namespace for one inherited freezer transaction."""

    original = {name: getattr(base, name) for name in _CONFIGURED_NAMES}
    configured = {
        "PREREG": PREREG,
        "FREEZER": FREEZER,
        "TEST": TEST,
        "AUDIT": AUDIT,
        "SOURCE_PATHS": SOURCE_PATHS,
        "START_DOCUMENT_INDEX": START_DOCUMENT_INDEX,
        "DOCUMENTS_PER_ROLE": DOCUMENTS_PER_ROLE,
        "TOTAL_DOCUMENTS": TOTAL_DOCUMENTS,
        "ROLE_NAMES": ROLE_NAMES,
        "ROLE_AUTHORIZATIONS": ROLE_AUTHORIZATIONS,
        "CACHE": CACHE,
        "RECEIPT": RECEIPT,
        "FAILURE": FAILURE,
        "LOCK": LOCK,
        "RECEIPT_SCHEMA": RECEIPT_SCHEMA,
        "FAILURE_SCHEMA": FAILURE_SCHEMA,
        "source_hashes": source_hashes,
        "validate_independent_audit": validate_independent_audit,
    }
    try:
        for name, value in configured.items():
            setattr(base, name, value)
        yield base
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def split_rows(rows, records):
    with configured_base() as configured:
        return configured.split_rows(rows, records)


def validate_selected(rows, records, prior):
    with configured_base() as configured:
        return configured.validate_selected(rows, records, prior)


def freeze():
    with configured_base() as configured:
        return configured.freeze()


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True, indent=2))
