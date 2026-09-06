#!/usr/bin/env python3
"""Numeric-only rerun of the exact H1/H4 query source-factor tensor."""

# BQGATE: EXPERIMENT pred_a_authority_partition_factor_native_weight_closure_finiteness_and_price pred_b_postcue_sources_dominate_query_conversion pred_c_base_pattern_value_transport_is_dominant pred_d_pattern_value_interaction_is_secondary pred_e_zero_fit_literal_weight_interface
import hashlib
from pathlib import Path
import run_iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v2.json"
V1_RESULT = ROOT / "circuits/followups/iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v1.py"
EXPECTED = {"prior": "7176a23f06facd4d50db977b19323ab6b040b2e4183d32832ffb5c51635aac2f",
    "v1_result": "af6cbb3a720f0faf34aa5aee76f5a558c12ad7ab049fa5fabfcfbc34247f36b4",
    "base_runner": "7eca9a08bd913831339b037da77c09f1a9bb398ed993e209806cdaefed565aff"}
PREDICTION_KEYS = ("pred_a_authority_partition_factor_native_weight_closure_finiteness_and_price",
    "pred_b_postcue_sources_dominate_query_conversion",
    "pred_c_base_pattern_value_transport_is_dominant",
    "pred_d_pattern_value_interaction_is_secondary",
    "pred_e_zero_fit_literal_weight_interface")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if {"prior": sha(PRIOR), "v1_result": sha(V1_RESULT), "base_runner": sha(BASE_RUNNER)} != EXPECTED:
        raise RuntimeError("v2 source-factor numeric-repair authority changed")
    experiment.PRIOR = PRIOR
    experiment.OUT = ROOT / "circuits/followups/iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v2_result.json"
    experiment.CANDIDATE_ID = "cross_task.iswas_mlp8_attn9_h1h4_query_source_factor_tensor_v2"
    experiment.RESULT_SCHEMA = "iswas_mlp8_attn9_h1h4_query_source_factor_tensor_result_v2"
    experiment.COMPLETE_RESID18_ABS_TOLERANCE = .05
    experiment.EXPECTED = dict(experiment.EXPECTED, prior=EXPECTED["prior"])
    experiment.main()


if __name__ == "__main__": main()
