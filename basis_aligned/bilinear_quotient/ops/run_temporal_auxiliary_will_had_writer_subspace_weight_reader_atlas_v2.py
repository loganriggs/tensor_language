#!/usr/bin/env python3
"""Gauge-only repair wrapper for the temporal writer weight-reader atlas."""

# BQGATE: EXPERIMENT pred_a_authority_capability_capture_and_split pred_b_writer_positive_control pred_c_known_readers_are_weight_enriched pred_d_value_beats_routing_for_causal_prediction pred_e_top_weight_readers_are_causal pred_f_price_coverage_and_zero_causal_leakage
from pathlib import Path

import run_temporal_auxiliary_will_had_writer_subspace_weight_reader_atlas_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
parent.PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_writer_subspace_weight_reader_atlas_v2.json"
parent.OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_writer_subspace_weight_reader_atlas_v2_result.json"
parent.CANDIDATE_ID = "temporal_auxiliary.will_vs_had.writer_subspace_weight_reader_atlas_v2"
parent.EXPECTED["prior"] = "e4d12ec1ec7d45c33dbcca53ea5af4f2bffb9c60498d490223945b64a52ad32f"
PREDICTIONS = {
    "pred_a_authority_capability_capture_and_split": "unchanged",
    "pred_b_writer_positive_control": "unchanged",
    "pred_c_known_readers_are_weight_enriched": "unchanged",
    "pred_d_value_beats_routing_for_causal_prediction": "unchanged",
    "pred_e_top_weight_readers_are_causal": "unchanged",
    "pred_f_price_coverage_and_zero_causal_leakage": "unchanged",
}


if __name__ == "__main__":
    parent.main()
