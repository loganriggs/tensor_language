#!/usr/bin/env python3
"""Audit the narrow length-seven parent-mediator boundary in low-rank V3."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OPS = ROOT / "basis_aligned/bilinear_quotient/ops"
BINDING = HERE / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V3_BINDING.json"
RESULT = HERE / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V3_RESULT.json"
OUT = HERE / "SUCCESSOR_POINTER_LENGTH7_PARENT_BOUNDARY_V1_AUDIT.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    binding = json.loads(BINDING.read_text()); result = json.loads(RESULT.read_text())
    paths = {
        "fit_rows": HERE / "SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_ROWS.json",
        "holdout_rows": HERE / "SUCCESSOR_FIXED_POINTER_LENGTH7_LOW_RANK_HOLDOUT_V1_ROWS.json",
        "preregistration": HERE / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V1_PREREGISTRATION.md",
        "interaction_result": HERE / "SUCCESSOR_POINTER_CROSS_TYPE_INTERACTION_V1_RESULT.json",
        "confirmation_result": HERE / "SUCCESSOR_FIXED_POINTER_LENGTH6_CONFIRMATION_V1_RESULT.json",
        "correction": HERE / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V2_READOUT_CORRECTION.md",
        "v1_binding": HERE / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V1_BINDING.json",
        "v1_runner": OPS / "run_successor_pointer_interaction_low_rank_v1.py",
        "v1_invalid_result": HERE / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V1_RESULT.json",
        "v3_correction": HERE / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V3_SERIALIZATION_CORRECTION.md",
        "v2_binding": HERE / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V2_BINDING.json",
        "v2_runner": OPS / "run_successor_pointer_interaction_low_rank_v2.py",
        "v2_invalid_result": HERE / "SUCCESSOR_POINTER_INTERACTION_LOW_RANK_V2_RESULT.json",
    }
    checks = {
        "binding_files_match": all(sha(paths[name]) == expected for name, expected in binding["files"].items()),
        "result_runner_matches": result["runner_sha256"] == sha(OPS / "run_successor_pointer_interaction_low_rank_v3.py"),
        "fit_valid_and_rank8_selected": result["fit"]["instrument"] is True and result["fit"]["selected_rank"] == 8,
        "holdout_opened_at_full_price": result["price"]["observed_forwards"] == 22 and result["price"]["observed_sequences"] == 154 and result["price"]["observed_readout_rows"] == 90,
        "holdout_exact_instruments": result["holdout"]["self_error"] == 0.0 and result["holdout"]["ceiling_error"] == 0.0 and result["holdout"]["direct_readout_error"] == 0.0,
        "holdout_native_capability": all(value == 1.0 for cells in result["holdout"]["capability"].values() for value in cells.values()),
        "holdout_target_live": all(value >= 1.0 for value in result["holdout"]["family_mean_target"].values()),
        "digit_joint_passes": result["holdout"]["joint"]["digit"]["passes"] is True,
        "month_joint_only_projection_fails": result["holdout"]["joint"]["month"]["passes"] is False and result["holdout"]["joint"]["month"]["projection"] < 0.50 and result["holdout"]["joint"]["month"]["cosine"] >= 0.70 and result["holdout"]["joint"]["month"]["late_forward_control_fraction"] <= 0.50 and result["holdout"]["joint"]["month"]["early_backward_control_fraction"] <= 0.50,
        "low_rank_claim_unopened": result["terminal"] == "invalid" and result["predictions"]["pred_c_holdout_transfer"] is False,
    }
    payload = {
        "schema": "successor_pointer_length7_parent_boundary_v1_audit",
        "terminal": "length7_parent_mediator_boundary" if all(checks.values()) else "audit_failed",
        "checks": checks,
        "metrics": {
            "digit_joint_projection": result["holdout"]["joint"]["digit"]["projection"],
            "month_joint_projection": result["holdout"]["joint"]["month"]["projection"],
            "month_joint_cosine": result["holdout"]["joint"]["month"]["cosine"],
            "month_late_forward_control_fraction": result["holdout"]["joint"]["month"]["late_forward_control_fraction"],
            "month_early_backward_control_fraction": result["holdout"]["joint"]["month"]["early_backward_control_fraction"],
        },
        "claim_boundary": "parent typed-group length transfer only; V1-V3 low-rank interaction outcomes remain invalid",
        "binding_sha256": sha(BINDING), "result_sha256": sha(RESULT),
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
