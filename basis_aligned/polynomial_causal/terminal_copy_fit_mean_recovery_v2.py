#!/usr/bin/env python3
"""Narrow recovery for the v1 cross-device Rotary identity-check failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import terminal_copy_fit_mean_lifecycle as life


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREREG = HERE / "TERMINAL_COPY_FIT_MEAN_RECOVERY_V2_PREREGISTRATION.md"
RUNNER = Path(__file__).resolve()
TEST = HERE / "test_terminal_copy_fit_mean_recovery_v2.py"
AUDIT = HERE / "terminal_copy_fit_mean_recovery_v2_independent_audit.json"
V1_AUTHORITY = HERE / "terminal_copy_fit_means_v1_authority.json"
V1_FAILURE = HERE / "terminal_copy_fit_means_v1_failure.json"
V1_AUTHORITY_SHA256 = "66541d42a89eeede2be83724e130037183eb1f9ced9c6150785a30448126aca8"
V1_FAILURE_SHA256 = "42c102693cb4388384fa40d04bb1091e6efdc878209e12cea3e91c8ed99e7ce4"

AUTHORITY = HERE / "terminal_copy_fit_means_v2_authority.json"
BANK = HERE / "terminal_copy_fit_means_v2_bank.pt"
RESULT = HERE / "terminal_copy_fit_means_v2_result.json"
MANIFEST = HERE / "terminal_copy_fit_means_v2_manifest.json"
RECEIPT = HERE / "terminal_copy_fit_means_v2_receipt.json"
FAILURE = HERE / "terminal_copy_fit_means_v2_failure.json"
LOCK = Path("/workspace/runs/.terminal_copy_fit_means_v2.lock")

RECOVERY_SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/TERMINAL_COPY_FIT_MEAN_RECOVERY_V2_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/terminal_copy_fit_mean_recovery_v2.py",
    "basis_aligned/polynomial_causal/test_terminal_copy_fit_mean_recovery_v2.py",
    "basis_aligned/polynomial_causal/terminal_copy_fit_mean_recovery_v2_independent_audit.json",
    "basis_aligned/polynomial_causal/terminal_copy_attention_dispatcher.py",
    "basis_aligned/polynomial_causal/test_terminal_copy_attention_dispatcher.py",
    "basis_aligned/polynomial_causal/terminal_copy_fit_means_v1_authority.json",
    "basis_aligned/polynomial_causal/terminal_copy_fit_means_v1_failure.json",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_v1_failure_lineage() -> None:
    if (
        file_sha256(V1_AUTHORITY) != V1_AUTHORITY_SHA256
        or file_sha256(V1_FAILURE) != V1_FAILURE_SHA256
    ):
        raise RuntimeError("terminal-copy fit v1 authority/failure bytes changed")
    failure = json.loads(V1_FAILURE.read_text())
    if (
        failure.get("status") != "terminal_failure_no_success_receipt"
        or failure.get("bank_exists") is not False
        or failure.get("result_exists") is not False
        or failure.get("manifest_exists") is not False
        or failure.get("receipt_exists") is not False
        or "same device" not in str(failure.get("exception_message"))
    ):
        raise RuntimeError("terminal-copy fit v1 failure semantics changed")


def validate_recovery_audit() -> dict[str, Any]:
    value = json.loads(AUDIT.read_text())
    reviewed = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in (PREREG, RUNNER, TEST, HERE / "terminal_copy_attention_dispatcher.py",
                     HERE / "test_terminal_copy_attention_dispatcher.py",
                     V1_AUTHORITY, V1_FAILURE)
    }
    if (
        value.get("schema") != "terminal_copy_fit_mean_recovery_v2_independent_audit"
        or value.get("status") != "approved_narrow_cross_device_identity_recovery"
        or value.get("approved") is not True
        or value.get("outcome_access") is not False
        or value.get("reviewer") != "independent_artifact_audit_agent"
        or value.get("reviewed_source_sha256s") != reviewed
        or value.get("scientific_protocol_changed") is not False
    ):
        raise RuntimeError("terminal-copy fit v2 recovery audit changed")
    return value


def configure() -> None:
    validate_v1_failure_lineage()
    life.AUTHORITY = AUTHORITY
    life.BANK = BANK
    life.RESULT = RESULT
    life.MANIFEST = MANIFEST
    life.RECEIPT = RECEIPT
    life.FAILURE = FAILURE
    life.LOCK = LOCK
    life.SOURCE_PATHS = tuple(dict.fromkeys((*life.SOURCE_PATHS, *RECOVERY_SOURCE_PATHS)))


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
