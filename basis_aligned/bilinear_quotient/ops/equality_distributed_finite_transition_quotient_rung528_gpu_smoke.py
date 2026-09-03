#!/usr/bin/env python3
"""Managed CUDA smoke for rung 528's raw post-MLP12 state instrument.

pred_a: native analytical replay and self-inserted boundaries are bit-exact
pred_b: all four registered equality transitions are live
pred_c: both fixed downstream continuation patches are live
"""

# BQGATE: EXPERIMENT

from equality_distributed_finite_transition_quotient_rung528_run import gpu_smoke


REGISTERED_PREDICTIONS = (
    "pred_a_native_replay_and_self_inserted_boundaries_are_exact",
    "pred_b_all_four_registered_equality_transitions_are_live",
    "pred_c_both_fixed_downstream_continuation_patches_are_live",
)


if __name__ == "__main__":
    gpu_smoke()
