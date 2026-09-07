#!/usr/bin/env python3
"""Bidirectional source-noise validation of the frozen rank46 support."""
# BQGATE: EXPERIMENT pred_a_authority_hash_noise_tripwire_finiteness_and_exact_price pred_b_all_noisy_bidirectional_coordinates_transfer pred_c_all_noisy_bidirectional_behaviors_transfer pred_d_all_noisy_bidirectional_controls_are_selective pred_e_seed_and_direction_stability
from pathlib import Path

import run_temporal_five_mlp_rank48_pooled_bidirectional_source_noise_ood_v1 as engine

ROOT = Path(__file__).resolve().parents[1]
RANK46 = ROOT / "circuits/followups/temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_v1_result.json"
CORRECTION = ROOT / "circuits/followups/temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_v1_count_correction.json"
RANK45_BOUNDARY = ROOT / "circuits/followups/temporal_five_mlp_rank46_pooled_greedy_rank45_deletion_v1_result.json"
REGISTERED_PREDICTIONS = (
    "pred_a_authority_hash_noise_tripwire_finiteness_and_exact_price",
    "pred_b_all_noisy_bidirectional_coordinates_transfer",
    "pred_c_all_noisy_bidirectional_behaviors_transfer",
    "pred_d_all_noisy_bidirectional_controls_are_selective",
    "pred_e_seed_and_direction_stability",
)

engine.PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank46_pooled_bidirectional_source_noise_ood_v1.json"
engine.SUPPORT = RANK46
engine.OUT = ROOT / "circuits/followups/temporal_five_mlp_rank46_pooled_bidirectional_source_noise_ood_v1_result.json"
engine.CANDIDATE = "temporal_auxiliary.five_mlp_rank46_pooled_bidirectional_source_noise_ood_v1"
engine.RESULT_SCHEMA = "temporal_five_mlp_rank46_pooled_bidirectional_source_noise_ood_result_v1"
engine.SUCCESS_TERMINAL = "pooled_bidirectional_source_noise_robust_rank46_program"
engine.FAILURE_TERMINAL = "rank46_source_noise_failure"
engine.EXPECTED_SUPPORT_COUNT = 46
engine.EXTRA_AUTHORITIES = {"rank46_count_correction": CORRECTION, "rank45_boundary": RANK45_BOUNDARY}
engine.EXPECTED = dict(engine.EXPECTED,
    support="dc8b66d826acede98bde996babe42420dd9e805981f6b81de458d568f29eea1d",
    rank46_count_correction="fc1b3ac442e93a841131996bd73f826a18279c80947089deac4ed7060a988c50",
    rank45_boundary="0d70ba386846ead59ff034ef22be745b9811417d7d488b8ed05ffcc9694bfaaf")


if __name__ == "__main__": engine.main()
