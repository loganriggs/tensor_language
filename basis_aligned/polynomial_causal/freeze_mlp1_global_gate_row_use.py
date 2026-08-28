#!/usr/bin/env python3
"""Freeze the permitted scientific uses of the registry-fresh MLP1 rows.

The parent receipt authorizes scored experiments but forbids training.  This contract
states prospectively that response-based support selection and linear analysis fitting
are scored analysis on wave A, while all model-weight optimization remains forbidden.
Wave B is evaluation-only.  No model outcome is read or created here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
ROWS_RECEIPT = BQ / "mlp1_global_gate_v1_rows_receipt.json"
OUT = HERE / "mlp1_global_gate_row_use_authority.json"
RESULT = HERE / "tensor_bilin18_mlp1_global_gate_results.json"
BUNDLE = HERE / "tensor_bilin18_mlp1_global_gate_bundle.pt"
EXPECTED_ROWS_RECEIPT_SHA256 = "63d35040a22c5da69a889cd94ece37cf7c6d353c41ebda3fdbaa12114303b3cd"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def build_authority() -> dict[str, Any]:
    if file_sha256(ROWS_RECEIPT) != EXPECTED_ROWS_RECEIPT_SHA256:
        raise RuntimeError("MLP1 global-gate row receipt identity changed")
    rows = json.loads(ROWS_RECEIPT.read_text())
    if (
        rows.get("status") != "frozen_before_any_global_gate_model_forward"
        or rows.get("authorized_for_scored_experiments") is not True
        or rows.get("authorized_for_training") is not False
        or not all(rows.get("disjointness_gates", {}).values())
        or rows.get("selection", {}).get("n_source_documents") != 32
    ):
        raise RuntimeError("MLP1 global-gate parent row authority changed")
    if RESULT.exists() or BUNDLE.exists():
        raise RuntimeError("cannot freeze row use after global-gate outcomes exist")
    value: dict[str, Any] = {
        "status": "mlp1_global_gate_row_use_frozen_no_model_outcomes",
        "parent_rows_receipt_sha256": EXPECTED_ROWS_RECEIPT_SHA256,
        "parent_scored_experiments_authorized": True,
        "parent_training_authorized": False,
        "interpretation": (
            "support selection and regularized linear analysis-coefficient fitting are "
            "registered scored-analysis operations, not optimization of model weights"
        ),
        "wave_A_fit_authorized": {
            "compute_model_responses": True,
            "select_physical_gate_support": True,
            "fit_regularized_css_interpolant": True,
            "fit_regularized_all_on_coefficients": True,
            "choose_unregistered_hyperparameters": False,
            "optimize_any_model_parameter_or_buffer": False,
        },
        "wave_B_evaluation_only": {
            "compute_model_responses": True,
            "evaluate_frozen_support_and_coefficients": True,
            "select_or_modify_support": False,
            "fit_or_modify_coefficients": False,
            "choose_or_modify_thresholds": False,
            "optimize_any_model_parameter_or_buffer": False,
        },
        "probe_half_first_fit_only": True,
        "probe_half_second_replication_only": True,
        "model_training_forbidden": True,
        "finite_gate_scaling_forbidden": True,
        "result_computed": False,
        "bundle_computed": False,
    }
    value["authority_fingerprint"] = canonical_sha256(value)
    return value


def main() -> None:
    if OUT.exists():
        raise RuntimeError("MLP1 row-use authority is create-only and already exists")
    payload = json.dumps(build_authority(), indent=2, sort_keys=True) + "\n"
    descriptor = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    print(json.dumps({"status": "frozen", "path": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
