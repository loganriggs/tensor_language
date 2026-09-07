#!/usr/bin/env python3
"""Second greedy deletion step under the pooled bidirectional interface."""
# BQGATE: EXPERIMENT pred_a_authority_hash_disjointness_finiteness_and_exact_price pred_b_at_least_one_rank47_arm_has_bidirectional_coordinate_fidelity pred_c_at_least_one_rank47_arm_has_bidirectional_behavior_fidelity pred_d_at_least_one_rank47_arm_is_bidirectionally_selective pred_e_selected_rank47_arm_is_direction_stable
import hashlib
from pathlib import Path

import run_temporal_five_mlp_rank48_pooled_joint_rank47_deletion_v1 as engine

ROOT = Path(__file__).resolve().parents[1]
RANK47 = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_joint_rank47_deletion_v1_result.json"

engine.PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_v1.json"
engine.RANK48 = RANK47
engine.OUT = ROOT / "circuits/followups/temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_v1_result.json"
engine.REMOVALS = ("L3H7", "L10H5")
engine.CANDIDATE = "temporal_auxiliary.five_mlp_rank47_pooled_greedy_rank46_deletion_v1"
engine.RESULT_SCHEMA = "temporal_five_mlp_rank47_pooled_greedy_rank46_deletion_result_v1"
engine.SUCCESS_TERMINAL = "pooled_bidirectional_rank46_program"
engine.FAILURE_TERMINAL = "rank47_greedy_deletion_boundary"
engine.EXPECTED = dict(engine.EXPECTED, rank48=hashlib.sha256(RANK47.read_bytes()).hexdigest())


if __name__ == "__main__": engine.main()
