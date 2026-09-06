#!/usr/bin/env python3
"""Capability-filtered repair of the shared L9H1/H4 value-subspace test."""

# BQGATE: EXPERIMENT pred_a_authority_capability_capture_and_split pred_b_each_task_basis_is_sufficient pred_c_shared_geometric_mode_exists pred_d_shared_mode_is_causally_material pred_e_task_specific_complements_are_secondary pred_f_joint_basis_is_compact_and_sufficient
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import run_aspectual_tense_l9h1h4_shared_value_weight_subspace_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_l9h1h4_shared_value_weight_subspace_v2.json"
CAPABILITY = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_capability_manifest_v1_result.json"
V1_RUNNER = ROOT / "ops/run_aspectual_tense_l9h1h4_shared_value_weight_subspace_v1.py"
OUT = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_weight_subspace_v2_result.json"
CANDIDATE_ID = "aspectual_tense.l9h1h4_shared_value_weight_subspace_v2"
EXPECTED = {
    "prior": "7d41b61bfaae00c03b124678d48235c66aaa70096a7d715180c4a08428de54ab",
    "capability": "9299fe3501995b72cec637a58838fdaf85f056034a1566de3a6d6bb04e38edd6",
    "v1_runner": "1c041f793af9ae8ff516a6650aba41518e57574da063d896cc142c2382ff4333",
}
SPLIT_COUNTS = {"has_fit": 16, "has_heldout": 15, "has_a2": 31,
                "is_fit": 8, "is_heldout": 6, "is_a2": 15}
REGISTERED_PREDICTIONS = {
    "pred_a_authority_capability_capture_and_split": "delegated unchanged to v1 executor",
    "pred_b_each_task_basis_is_sufficient": "delegated unchanged to v1 executor",
    "pred_c_shared_geometric_mode_exists": "delegated unchanged to v1 executor",
    "pred_d_shared_mode_is_causally_material": "delegated unchanged to v1 executor",
    "pred_e_task_specific_complements_are_secondary": "delegated unchanged to v1 executor",
    "pred_f_joint_basis_is_compact_and_sufficient": "delegated unchanged to v1 executor",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_v2():
    paths = {"prior": PRIOR, "capability": CAPABILITY, "v1_runner": V1_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise parent.ExperimentError("v2 repair authority hash changed")
    prior = json.loads(PRIOR.read_text())
    capability = json.loads(CAPABILITY.read_text())
    if (prior.get("candidate_id") != CANDIDATE_ID
            or capability.get("terminal") != "manifest"
            or not all(capability.get("predictions", {}).values())):
        raise parent.ExperimentError("v2 prior or capability manifest invalid")
    has_rows, _spec = parent.has_factor.validate_static()
    is_rows = parent.is_factor.validate_static()
    filtered = {}
    for task, rows in (("has", has_rows), ("is", is_rows)):
        ids = capability["jointly_capable_row_ids"][task]
        selected = [row for row in rows
                    if row["row_id"] in ids[row["transform_id"]]]
        a1 = [row for row in selected if row["transform_id"] == "A1"]
        fit, heldout = parent.stratified(a1)
        filtered[f"{task}_fit"] = fit
        filtered[f"{task}_heldout"] = heldout
        filtered[f"{task}_a2"] = [row for row in selected if row["transform_id"] == "A2"]
    if {name: len(rows) for name, rows in filtered.items()} != SPLIT_COUNTS:
        raise parent.ExperimentError("v2 frozen capable split changed")
    for name in ("has_fit", "has_heldout", "is_fit", "is_heldout"):
        counts = [sum(row["direction_id"] == direction for row in filtered[name])
                  for direction in sorted({row["direction_id"] for row in filtered[name]})]
        if max(counts) - min(counts) > 1:
            raise parent.ExperimentError("v2 split lost direction balance")
    return filtered


def dryrun_v2():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False,
            "split_counts": SPLIT_COUNTS, "arms": list(parent.ARMS),
            "model_forwards": 48, "example_evaluations": 766, "records": 402,
            "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}


def configure():
    parent.PRIOR = PRIOR
    parent.OUT = OUT
    parent.CANDIDATE_ID = CANDIDATE_ID
    parent.EXPECTED = EXPECTED
    parent.FORWARDS = 48
    parent.EVALUATIONS = 766
    parent.RECORDS = 402
    parent.validate_static = validate_v2
    parent.dryrun_receipt = dryrun_v2


if __name__ == "__main__":
    configure()
    parent.main()
