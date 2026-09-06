#!/usr/bin/env python3
"""Index-order-only repair of the auxiliary value source-subset atlas."""

# BQGATE: EXPERIMENT pred_a_authority_partition_value_replay_finiteness_and_price pred_b_at_least_one_small_source_program_is_sufficient pred_c_source_program_is_panel_stable pred_d_both_auxiliary_layers_have_material_value_reads pred_e_zero_fit_exact_subset_inventory
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import run_iswas_mlp8_auxiliary_value_source_subset_atlas_v1 as experiment


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_auxiliary_value_source_subset_atlas_v2.json"
V1_RESULT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_value_source_subset_atlas_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_iswas_mlp8_auxiliary_value_source_subset_atlas_v1.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_value_source_subset_atlas_v2_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_auxiliary_value_source_subset_atlas_v2"
EXPECTED = {
    "prior": "2068950302f2f9722e265338159640a6091088bf64033facdc9e37500a28a811",
    "v1_result": "4764ff8692d3bcbc3ca3ae103f880b0dc2878e53967f7019567f4e282e37503e",
    "base_runner": "31ef7422c1f43113a62c42388b6be4168256b6387501d7648b70ef54ee990472",
}
REGISTERED_PREDICTIONS = (
    "pred_a_authority_partition_value_replay_finiteness_and_price",
    "pred_b_at_least_one_small_source_program_is_sufficient",
    "pred_c_source_program_is_panel_stable",
    "pred_d_both_auxiliary_layers_have_material_value_reads",
    "pred_e_zero_fit_exact_subset_inventory",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = {"prior": PRIOR, "v1_result": V1_RESULT, "base_runner": BASE_RUNNER}
    if {key: sha(value) for key, value in paths.items()} != EXPECTED:
        raise RuntimeError("v2 source-index repair authority changed")
    v1 = json.loads(V1_RESULT.read_text())
    if v1.get("terminal") != "invalid" or v1.get("instrument", {}).get("value_source_closure_max_abs", 0) < .5:
        raise RuntimeError("v1 is not the registered source-index invalid")
    experiment.PRIOR = PRIOR
    experiment.OUT = OUT
    experiment.CANDIDATE_ID = CANDIDATE_ID
    experiment.RESULT_SCHEMA = "iswas_mlp8_auxiliary_value_source_subset_atlas_result_v2"
    experiment.EXPECTED = dict(experiment.EXPECTED, prior=EXPECTED["prior"])
    experiment.main()


if __name__ == "__main__":
    main()
