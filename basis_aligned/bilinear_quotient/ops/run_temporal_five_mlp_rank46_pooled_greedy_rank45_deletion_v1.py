#!/usr/bin/env python3
"""Final frozen greedy deletion under the pooled bidirectional interface."""
# BQGATE: EXPERIMENT pred_a_authority_hash_disjointness_finiteness_and_exact_price pred_b_at_least_one_rank47_arm_has_bidirectional_coordinate_fidelity pred_c_at_least_one_rank47_arm_has_bidirectional_behavior_fidelity pred_d_at_least_one_rank47_arm_is_bidirectionally_selective pred_e_selected_rank47_arm_is_direction_stable
from pathlib import Path

import run_temporal_five_mlp_rank48_pooled_joint_rank47_deletion_v1 as engine

ROOT = Path(__file__).resolve().parents[1]
RANK46 = ROOT / "circuits/followups/temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_v1_result.json"
CORRECTION = ROOT / "circuits/followups/temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_v1_count_correction.json"
REGISTERED_PREDICTIONS = (
    "pred_a_authority_hash_disjointness_finiteness_and_exact_price",
    "pred_b_at_least_one_rank47_arm_has_bidirectional_coordinate_fidelity",
    "pred_c_at_least_one_rank47_arm_has_bidirectional_behavior_fidelity",
    "pred_d_at_least_one_rank47_arm_is_bidirectionally_selective",
    "pred_e_selected_rank47_arm_is_direction_stable",
)

engine.PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank46_pooled_greedy_rank45_deletion_v1.json"
engine.RANK48 = RANK46
engine.OUT = ROOT / "circuits/followups/temporal_five_mlp_rank46_pooled_greedy_rank45_deletion_v1_result.json"
engine.REMOVALS = ("L10H5",)
engine.CANDIDATE = "temporal_auxiliary.five_mlp_rank46_pooled_greedy_rank45_deletion_v1"
engine.RESULT_SCHEMA = "temporal_five_mlp_rank46_pooled_greedy_rank45_deletion_result_v1"
engine.SUCCESS_TERMINAL = "pooled_bidirectional_rank45_program"
engine.FAILURE_TERMINAL = "rank46_greedy_deletion_boundary"
engine.EXPECTED_BASE_SUPPORT_COUNT = 46
engine.EXTRA_AUTHORITIES = {"rank46_count_correction": CORRECTION}
engine.EXPECTED = dict(engine.EXPECTED,
    rank48="dc8b66d826acede98bde996babe42420dd9e805981f6b81de458d568f29eea1d",
    rank46_count_correction="fc1b3ac442e93a841131996bd73f826a18279c80947089deac4ed7060a988c50")


if __name__ == "__main__": engine.main()
