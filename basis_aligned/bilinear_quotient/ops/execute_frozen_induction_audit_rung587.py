#!/usr/bin/env python3
"""Queue adapter for the already frozen, outcome-blind R587 auditor."""

# BQLANE: cpu

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
AUDITOR = ROOT / "ops" / "audit_induction_selector_payload_native_capability_rung587.py"
AUDITOR_TEST = ROOT / "ops" / "test_audit_induction_selector_payload_native_capability_rung587.py"
AUDITOR_PREREG = ROOT.parent / "polynomial_causal" / (
    "INDUCTION_SELECTOR_PAYLOAD_NATIVE_CAPABILITY_AUDIT_RUNG587_PREREGISTRATION.md"
)
AUDITOR_DRYRUN = ROOT / "induction_selector_payload_native_capability_audit_rung587_dryrun.json"
SOURCE_RESULT = ROOT / "induction_selector_payload_native_capability_rung586_results.json"
SOURCE_RECEIPT = ROOT / "induction_selector_payload_native_capability_rung586_receipt.json"
AUDIT_RESULT = ROOT / "induction_selector_payload_native_capability_audit_rung587.json"

FROZEN_HASHES = {
    AUDITOR: "a31ba01c3b5009ff7125a3ff0dec049389aa8b44511d242ef62a9a2799b8aad0",
    AUDITOR_TEST: "50aa271a9e2fd06ee1b4df0638c8a435ae62948da55c837850a5d3c110dcfb6a",
    AUDITOR_PREREG: "1f8e51ca7dcb4c8c9bb73ba13403c098871e13b593d995ff516ed839c2a9c771",
    AUDITOR_DRYRUN: "ececa57c88a423b93a234a956c7057de3622142fa28227a3d2d1936989487800",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_inputs() -> dict[str, str]:
    observed = {}
    for path, expected in FROZEN_HASHES.items():
        if not path.is_file():
            raise RuntimeError(f"frozen R587 file missing: {path}")
        digest = sha256(path)
        if digest != expected:
            raise RuntimeError(f"frozen R587 file changed: {path}")
        observed[str(path)] = digest
    if not SOURCE_RESULT.is_file() or not SOURCE_RECEIPT.is_file():
        raise RuntimeError("R586 result and receipt must both exist before R587 execution")
    if AUDIT_RESULT.exists():
        raise RuntimeError("R587 audit namespace already exists")
    return observed


def plan() -> dict[str, object]:
    observed = validate_frozen_inputs()
    return {
        "schema": "execute_frozen_induction_audit_rung587_plan_v1",
        "pred_a_frozen_auditor_hashes_match": True,
        "pred_b_complete_source_pair_exists": True,
        "pred_c_cpu_only_zero_model_calls": True,
        "frozen_sha256": observed,
        "source_pair_contents_opened": False,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "model_weights_updated": False,
        "evaluated_splits": [],
        "forbidden_splits_opened": [],
        "next_step": "execute_the_exact_hash_pinned_R587_auditor",
    }


def main() -> None:
    execution_plan = plan()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(execution_plan, indent=2, allow_nan=False))
        return
    os.environ.pop("BQLIB_DRYRUN", None)
    os.execv(sys.executable, [sys.executable, str(AUDITOR)])


if __name__ == "__main__":
    main()
