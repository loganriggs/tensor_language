#!/usr/bin/env python3
"""Zero-model admission gate for the conditional attention-15 mediation atlas."""

# BQLANE: cpu
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCY = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1_result.json"
DEPENDENCY_RUNNER = ROOT / "ops/run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1.py"
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_attention15_head_module_mediation_atlas_v1.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_attention15_head_module_mediation_atlas_v1_admission.json"
EXPECTED_RUNNER_SHA256 = "4d435dfa6c6f29a34de2b5ba7679aafb3b622cdf0fbf2ea779c58bea982b1fe3"
REQUIRED = (
    "pred_a_authority_alignment_parent_replay_zero_arm_factorial_closure_finiteness_and_price",
    "pred_b_each_oracle_transfers_own_v15_target_with_live_attention15",
    "pred_c_live_attention_retains_most_fixed_background_effect",
)


class AdmissionError(RuntimeError):
    pass


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def build_manifest(result_bytes, runner_bytes, prior_bytes):
    result, prior = json.loads(result_bytes), json.loads(prior_bytes)
    if digest(runner_bytes) != EXPECTED_RUNNER_SHA256:
        raise AdmissionError("dependency runner hash changed")
    if result.get("candidate_id") != "temporal_auxiliary.iswas_v15_construction_oracle_attention15_dependency_factorial_v1":
        raise AdmissionError("unexpected dependency candidate")
    if prior.get("candidate_id") != "temporal_auxiliary.iswas_v15_attention15_head_module_mediation_atlas_v1":
        raise AdmissionError("unexpected atlas prior")
    predictions = result.get("predictions", {})
    observed = {key: predictions.get(key) for key in REQUIRED}
    admitted = all(value is True for value in observed.values())
    return {
        "schema": "temporal_iswas_v15_attention15_head_module_mediation_atlas_admission_v1",
        "candidate_id": prior["candidate_id"],
        "dependency_result_sha256": digest(result_bytes),
        "dependency_runner_sha256": digest(runner_bytes),
        "atlas_prior_sha256": digest(prior_bytes),
        "required_predictions": observed,
        "admitted": admitted,
        "disposition": "admit_head_module_atlas" if admitted else "close_weight_reader_branch",
        "model_forwards": 0,
        "model_loaded": False,
    }


def main():
    missing = [str(path) for path in (DEPENDENCY, DEPENDENCY_RUNNER, PRIOR) if not path.exists()]
    if missing:
        raise AdmissionError(f"missing authorities: {missing}")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    manifest = build_manifest(DEPENDENCY.read_bytes(), DEPENDENCY_RUNNER.read_bytes(), PRIOR.read_bytes())
    atomic_create_json(OUT, manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
