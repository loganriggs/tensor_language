#!/usr/bin/env python3
"""Dry-run-safe managed CUDA smoke v2 for rung 528's raw boundary.

pred_a: native replay and self-inserted post-MLP12 boundaries are exact
pred_b: all four registered equality transitions are live
pred_c: both registered continuation patches are live
"""

# BQGATE: EXPERIMENT

import json
import os

from equality_distributed_finite_transition_quotient_rung528_run import (
    SMOKE_V2_OUT,
    dry_run,
    gpu_smoke,
)


REGISTERED_PREDICTIONS = (
    "pred_a_native_replay_and_self_inserted_boundaries_are_exact",
    "pred_b_all_four_registered_equality_transitions_are_live",
    "pred_c_both_registered_continuation_patches_are_live",
)


if __name__ == "__main__":
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(dry_run(), indent=2, sort_keys=True))
    else:
        gpu_smoke(SMOKE_V2_OUT)
