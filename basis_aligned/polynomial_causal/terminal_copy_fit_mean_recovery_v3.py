#!/usr/bin/env python3
"""Narrow recovery for the v2 bfloat16 integrity-hash failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import terminal_copy_fit_mean_lifecycle as life
import terminal_copy_fit_mean_recovery_v2 as v2


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "TERMINAL_COPY_FIT_MEAN_RECOVERY_V3_PREREGISTRATION.md"
RUNNER = Path(__file__).resolve()
TEST = HERE / "test_terminal_copy_fit_mean_recovery_v3.py"
AUDIT = HERE / "terminal_copy_fit_mean_recovery_v3_independent_audit.json"
OWNER = HERE / "terminal_copy_fit_mean_owner.py"
OWNER_TEST = HERE / "test_terminal_copy_fit_mean_owner.py"

V2_AUTHORITY = HERE / "terminal_copy_fit_means_v2_authority.json"
V2_FAILURE = HERE / "terminal_copy_fit_means_v2_failure.json"
V2_BANK = HERE / "terminal_copy_fit_means_v2_bank.pt"
V2_RESULT = HERE / "terminal_copy_fit_means_v2_result.json"
V2_MANIFEST = HERE / "terminal_copy_fit_means_v2_manifest.json"
V2_RECEIPT = HERE / "terminal_copy_fit_means_v2_receipt.json"
V2_LOCK = Path("/workspace/runs/.terminal_copy_fit_means_v2.lock")
V2_ABSENT_OUTPUTS = (V2_BANK, V2_RESULT, V2_MANIFEST, V2_RECEIPT, V2_LOCK)
V2_AUTHORITY_SHA256 = "49b1b8cb7960fbf389118bee356a8509c029046cbe498dbbbd39f11757a91451"
V2_FAILURE_SHA256 = "ce2d0dc53f3e52a2e8707ab3f46a55e93ea16279cbeda5da48b2f2920560616a"

AUTHORITY = HERE / "terminal_copy_fit_means_v3_authority.json"
BANK = HERE / "terminal_copy_fit_means_v3_bank.pt"
RESULT = HERE / "terminal_copy_fit_means_v3_result.json"
MANIFEST = HERE / "terminal_copy_fit_means_v3_manifest.json"
RECEIPT = HERE / "terminal_copy_fit_means_v3_receipt.json"
FAILURE = HERE / "terminal_copy_fit_means_v3_failure.json"
LOCK = Path("/workspace/runs/.terminal_copy_fit_means_v3.lock")

RECOVERY_SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/TERMINAL_COPY_FIT_MEAN_RECOVERY_V3_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/terminal_copy_fit_mean_recovery_v3.py",
    "basis_aligned/polynomial_causal/test_terminal_copy_fit_mean_recovery_v3.py",
    "basis_aligned/polynomial_causal/terminal_copy_fit_mean_recovery_v3_independent_audit.json",
    "basis_aligned/polynomial_causal/terminal_copy_fit_means_v2_authority.json",
    "basis_aligned/polynomial_causal/terminal_copy_fit_means_v2_failure.json",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_v2_failure_lineage() -> None:
    if (
        file_sha256(V2_AUTHORITY) != V2_AUTHORITY_SHA256
        or file_sha256(V2_FAILURE) != V2_FAILURE_SHA256
    ):
        raise RuntimeError("terminal-copy fit v2 authority/failure bytes changed")
    authority = json.loads(V2_AUTHORITY.read_text())
    failure = json.loads(V2_FAILURE.read_text())
    if (
        failure.get("status") != "terminal_failure_no_success_receipt"
        or failure.get("bank_exists") is not False
        or failure.get("result_exists") is not False
        or failure.get("manifest_exists") is not False
        or failure.get("receipt_exists") is not False
        or failure.get("authority_sha256") != authority.get("authority_sha256")
        or "BFloat16" not in str(failure.get("exception_message"))
        or any(path.exists() for path in V2_ABSENT_OUTPUTS)
    ):
        raise RuntimeError("terminal-copy fit v2 failure semantics changed")


def validate_recovery_audit() -> dict[str, Any]:
    value = json.loads(AUDIT.read_text())
    reviewed_paths = (PREREG, RUNNER, TEST, OWNER, OWNER_TEST, V2_AUTHORITY, V2_FAILURE)
    reviewed = {
        str(path.relative_to(ROOT)): file_sha256(path) for path in reviewed_paths
    }
    if (
        value.get("schema") != "terminal_copy_fit_mean_recovery_v3_independent_audit"
        or value.get("status") != "approved_narrow_bfloat16_hash_recovery"
        or value.get("approved") is not True
        or value.get("outcome_access") is not False
        or value.get("reviewer") != "independent_artifact_audit_agent"
        or value.get("reviewed_source_sha256s") != reviewed
        or value.get("scientific_protocol_changed") is not False
    ):
        raise RuntimeError("terminal-copy fit v3 recovery audit changed")
    return value


def configure() -> None:
    v2.configure()
    validate_v2_failure_lineage()
    life.AUTHORITY = AUTHORITY
    life.BANK = BANK
    life.RESULT = RESULT
    life.MANIFEST = MANIFEST
    life.RECEIPT = RECEIPT
    life.FAILURE = FAILURE
    life.LOCK = LOCK
    life.SOURCE_PATHS = tuple(dict.fromkeys((*life.SOURCE_PATHS, *RECOVERY_SOURCE_PATHS)))
    life.PROTECTED_PATHS = tuple(dict.fromkeys((*life.PROTECTED_PATHS, *V2_ABSENT_OUTPUTS)))


def freeze_authority() -> dict[str, Any]:
    configure()
    validate_recovery_audit()
    return life.freeze_execution_authority(life.AUDIT)


def execute(*, device: str = "cuda", batch_size: int = 4) -> dict[str, Any]:
    configure()
    validate_recovery_audit()
    return life.execute_fit_mean_collection(device=device, batch_size=batch_size)


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))
