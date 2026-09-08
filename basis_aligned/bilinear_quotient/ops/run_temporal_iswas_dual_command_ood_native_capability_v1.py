#!/usr/bin/env python3
"""Native capability gate for the fresh dual-command OOD authority."""

# BQGATE: EXPERIMENT pred_a_authority_positions_finiteness_and_exact_price pred_b_fit_every_joint_cell_and_role_is_capable pred_c_holdout_every_joint_cell_and_role_is_capable pred_d_dual_command_ood_license_is_issued
import hashlib
import json
import os
from pathlib import Path

import circuit_candidate_temporal_iswas_dual_command_ood_v1 as candidate
import run_temporal_iswas_dual_command_native_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_dual_command_ood_native_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_ood_v1.py"
TEMPORAL_CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ISWAS_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v2_audit_result.json"
BASE_RUNNER = ROOT / "ops/run_temporal_iswas_dual_command_native_capability_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_dual_command_ood_native_capability_v1_result.json"
EXPECTED_WRAPPER = {
    "prior": "b6ab07a6762633bf1dc76518fd45de34e4453031a8d1d6218e72db34de9dcade",
    "builder": "6062f2b8dfaa54da71477b43cb9fc63650db389c5883c9e64125f0a200afeb5a",
    "temporal_capability": "4758b02cd026c85289dc3eaf352cc496d238057c6f8b52dfc6fe49ae17893324",
    "iswas_capability": "a1c2baf0bd9548e189ccc4ba4d11c4905f85af1ff7013d408c143f6f2a6434e3",
    "base_runner": "0ebc5d1bd8cbcde0db8bee0aa09e7a5384ce35f58df0002f87e4fb0809d689f9",
}
PREDICTION_KEYS = (
    "pred_a_authority_positions_finiteness_and_exact_price",
    "pred_b_fit_every_joint_cell_and_role_is_capable",
    "pred_c_holdout_every_joint_cell_and_role_is_capable",
    "pred_d_dual_command_ood_license_is_issued",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "builder": BUILDER,
             "temporal_capability": TEMPORAL_CAPABILITY,
             "iswas_capability": ISWAS_CAPABILITY, "base_runner": BASE_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    temporal_result, iswas_result = (json.loads(path.read_text()) for path in
        (TEMPORAL_CAPABILITY, ISWAS_CAPABILITY))
    sources_capable = bool(
        temporal_result.get("terminal") == "manifest"
        and all(temporal_result.get("predictions", {}).values())
        and iswas_result.get("terminal") == "manifest"
        and iswas_result.get("predictions", {}).get("pred_b_native_cells_and_corrected_joint_bar") is True)
    if observed != EXPECTED_WRAPPER or not sources_capable:
        raise RuntimeError(f"OOD capability authority changed: {observed}")

    experiment.authority = candidate
    experiment.PRIOR = PRIOR
    experiment.BUILDER = BUILDER
    experiment.OUT = OUT
    experiment.EXPECTED = {"prior": EXPECTED_WRAPPER["prior"],
                           "builder": EXPECTED_WRAPPER["builder"],
                           "audit": experiment.EXPECTED["audit"],
                           "producer": experiment.EXPECTED["producer"]}
    experiment.PREDICTION_KEYS = PREDICTION_KEYS
    original_write = experiment.atomic_create_json

    def write_result(_path, result):
        result["schema"] = "temporal_iswas_dual_command_ood_native_capability_result_v1"
        result["wrapper_authority_sha256"] = EXPECTED_WRAPPER
        if result.get("terminal") == "dual_command_license_issued":
            result["terminal"] = "dual_command_ood_license_issued"
        original_write(OUT, result)

    experiment.atomic_create_json = write_result
    experiment.main()


if __name__ == "__main__": main()
