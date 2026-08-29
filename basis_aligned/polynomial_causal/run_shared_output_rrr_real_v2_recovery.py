#!/usr/bin/env python3
"""Narrow recovery wrapper for shared-output RRR after v1's device mismatch.

The scientific protocol is unchanged.  This module binds the spent v1 authority and
failure, opens a fresh v2 namespace, and delegates to the source-closed base runner
whose coverage lookup now follows evaluation tokens onto their execution device.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_shared_output_rrr_real_v1 as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNNER = HERE / "run_shared_output_rrr_real_v2_recovery.py"
TEST = HERE / "test_run_shared_output_rrr_real_v2_recovery.py"
PREREG = HERE / "SHARED_OUTPUT_RRR_REAL_V2_RECOVERY_PREREGISTRATION.md"
V1_AUTHORITY = HERE / "shared_output_rrr_real_v1_authority.json"
V1_FAILURE = HERE / "shared_output_rrr_real_v1_failure.json"
V1_RESULTS = HERE / "shared_output_rrr_real_v1_results.json"
V1_RECEIPT = HERE / "shared_output_rrr_real_v1_receipt.json"

V1_AUTHORITY_FILE_SHA256 = "4ac2839267abc99179fda32f161ec33eabe091384053b53ec78dca5f6b54e122"
V1_FAILURE_FILE_SHA256 = "1162039e29635f35978a267324b6b9f1da3333ec01b410d04e61b614664bfd88"
V1_AUTHORITY_SHA256 = "fd7a73c3a7a5a275f802d76483b2001ec15a85942c806536e52bce8114742392"
V1_SOURCE_COMMIT = "f690c6fd18c595ba17b3bf09f0743691ed809226"

BASE_SOURCE_PATHS = tuple(base.SOURCE_PATHS)
BASE_FILE_PINS = dict(base.FILE_PINS)


def restore_base_defaults() -> None:
    """Restore import defaults after isolated CPU tests; production never calls this."""
    base.PROTOCOL_VERSION = "v1"
    base.AUTHORITY = HERE / "shared_output_rrr_real_v1_authority.json"
    base.RESULTS = HERE / "shared_output_rrr_real_v1_results.json"
    base.FAILURE = HERE / "shared_output_rrr_real_v1_failure.json"
    base.RECEIPT = HERE / "shared_output_rrr_real_v1_receipt.json"
    base.LOCK = Path("/workspace/runs/.shared_output_rrr_real_v1.lock")
    base.SOURCE_PATHS = BASE_SOURCE_PATHS
    base.FILE_PINS = dict(BASE_FILE_PINS)
    base.RECOVERY_PARENT = None


def configure_base() -> None:
    base.PROTOCOL_VERSION = "v2_recovery"
    base.AUTHORITY = HERE / "shared_output_rrr_real_v2_recovery_authority.json"
    base.RESULTS = HERE / "shared_output_rrr_real_v2_recovery_results.json"
    base.FAILURE = HERE / "shared_output_rrr_real_v2_recovery_failure.json"
    base.RECEIPT = HERE / "shared_output_rrr_real_v2_recovery_receipt.json"
    base.LOCK = Path("/workspace/runs/.shared_output_rrr_real_v2_recovery.lock")
    additions = (RUNNER, TEST, PREREG)
    base.SOURCE_PATHS = tuple(dict.fromkeys(
        (*BASE_SOURCE_PATHS, *(str(path.relative_to(ROOT)) for path in additions))
    ))
    base.FILE_PINS = {
        **BASE_FILE_PINS,
        str(V1_AUTHORITY.relative_to(ROOT)): V1_AUTHORITY_FILE_SHA256,
        str(V1_FAILURE.relative_to(ROOT)): V1_FAILURE_FILE_SHA256,
    }
    base.RECOVERY_PARENT = {
        "version": "v1",
        "authority_file_sha256": V1_AUTHORITY_FILE_SHA256,
        "authority_sha256": V1_AUTHORITY_SHA256,
        "failure_file_sha256": V1_FAILURE_FILE_SHA256,
        "failure_stage": "first evaluation-role coverage-mask lookup",
        "scientific_metrics_observed": False,
        "only_change": "move immutable coverage mask to evaluation token device",
    }


def verify_spent_parent() -> None:
    if base.file_sha256(V1_AUTHORITY) != V1_AUTHORITY_FILE_SHA256 or base.file_sha256(
        V1_FAILURE
    ) != V1_FAILURE_FILE_SHA256 or V1_RESULTS.exists() or V1_RECEIPT.exists():
        raise RuntimeError("shared-RRR v1 recovery parent artifacts changed")
    authority = json.loads(V1_AUTHORITY.read_text())
    failure = json.loads(V1_FAILURE.read_text())
    if authority.get("schema") != "shared_output_rrr_real_v1_authority" or authority.get(
        "authority_sha256"
    ) != V1_AUTHORITY_SHA256 or authority.get("source_closure", {}).get(
        "commit"
    ) != V1_SOURCE_COMMIT:
        raise RuntimeError("shared-RRR v1 authority semantics changed")
    if failure != {
        "authority_exists": True,
        "authority_file_sha256": V1_AUTHORITY_FILE_SHA256,
        "error": "indices should be either on cpu or on the same device as the indexed tensor (cpu)",
        "error_type": "RuntimeError",
        "receipt_exists": False,
        "results_exists": False,
        "results_file_sha256": None,
        "schema": "shared_output_rrr_real_v1_failure",
        "status": "terminal_failure_no_receipt",
    }:
        raise RuntimeError("shared-RRR v1 failure semantics changed")


def run(*, device: str = "cuda"):
    configure_base()
    verify_spent_parent()
    return base.run(device=device)


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
