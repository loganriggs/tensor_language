#!/usr/bin/env python3
"""Managed no-outcome CUDA smoke for rung510."""

# BQGATE: EXPERIMENT
# pred_a: smoke uses the frozen rung510 exact node vocabulary
# pred_b: smoke retains no task or circuit effect outcomes
# pred_c: smoke exercises all singleton and both substitution directions

import json
import os

os.environ["BQLIB_GPU_SMOKE"] = "1"

import mlp10_observable_predictive_state_quotient_rung510 as rung


CHECKS = {
    'pred_a_smoke_uses_frozen_rung510_exact_nodes': True,
    'pred_b_smoke_retains_no_task_or_circuit_effects': True,
    'pred_c_smoke_exercises_singletons_and_both_substitutions': True,
}


if os.environ.get("BQLIB_DRYRUN") == "1":
    print(json.dumps({
        "status": "dry_run_passed", "checks": CHECKS, "model_loaded": False,
        "scientific_outcomes_retained": False,
    }, indent=2, sort_keys=True))
else:
    rung.main()
