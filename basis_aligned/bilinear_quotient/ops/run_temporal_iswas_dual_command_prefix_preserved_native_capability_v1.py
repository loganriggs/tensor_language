#!/usr/bin/env python3
"""Native gate for the prefix-preserved same-sequence successor bank."""

# BQGATE: EXPERIMENT pred_a_authority_positions_finiteness_and_exact_price pred_b_fit_every_joint_cell_and_role_is_capable pred_c_holdout_every_joint_cell_and_role_is_capable pred_d_dual_command_license_is_issued
import hashlib
import json
import os
from pathlib import Path

import circuit_candidate_temporal_iswas_dual_command_v2 as candidate
import run_temporal_iswas_dual_command_native_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_dual_command_prefix_preserved_native_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_v2.py"
PREDECESSOR = ROOT / "circuits/followups/temporal_iswas_dual_command_native_capability_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_temporal_iswas_dual_command_native_capability_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_dual_command_prefix_preserved_native_capability_v1_result.json"
EXPECTED_WRAPPER = {
    "prior": "55f6f66ef6d6ff9503dc4c71ba1212259d9a6b91bf995fd17e33db40623cf328",
    "builder": "72da11860ce5bf1c03ea126bc10fcb6c46dd2648169edb00a1e9e1dbeee8a0d0",
    "predecessor": "7209be34d22c435c3f04e7231fe10c34452c3ee4df362dcbdac9c68adc402842",
    "base_runner": "0ebc5d1bd8cbcde0db8bee0aa09e7a5384ce35f58df0002f87e4fb0809d689f9",
}
PREDICTION_KEYS = (
    "pred_a_authority_positions_finiteness_and_exact_price",
    "pred_b_fit_every_joint_cell_and_role_is_capable",
    "pred_c_holdout_every_joint_cell_and_role_is_capable",
    "pred_d_dual_command_license_is_issued",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "builder": BUILDER, "predecessor": PREDECESSOR,
             "base_runner": BASE_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    predecessor = json.loads(PREDECESSOR.read_text())
    if observed != EXPECTED_WRAPPER or predecessor.get("terminal") != "native_capability_null":
        raise RuntimeError(f"prefix-preserved capability authority changed: {observed}")

    experiment.authority = candidate
    experiment.PRIOR = PRIOR
    experiment.BUILDER = BUILDER
    experiment.OUT = OUT
    experiment.EXPECTED = {"prior": EXPECTED_WRAPPER["prior"],
                           "builder": EXPECTED_WRAPPER["builder"],
                           "audit": experiment.EXPECTED["audit"],
                           "producer": experiment.EXPECTED["producer"]}
    original_write = experiment.atomic_create_json

    def write_result(_path, result):
        result["schema"] = "temporal_iswas_dual_command_prefix_preserved_native_capability_result_v1"
        result["wrapper_authority_sha256"] = EXPECTED_WRAPPER
        original_write(OUT, result)

    experiment.atomic_create_json = write_result
    experiment.main()


if __name__ == "__main__": main()
