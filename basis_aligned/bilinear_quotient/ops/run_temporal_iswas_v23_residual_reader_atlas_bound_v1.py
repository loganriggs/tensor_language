#!/usr/bin/env python3
"""Managed fail-closed launcher for the preregistered v23 residual-reader atlas.

This file adds no scientific degrees of freedom.  It converts the immutable
block-11 factorial receipt into the already-defined binding and then delegates
to the frozen atlas runner.  Queueing it after the factorial prevents a result
boundary from becoming an idle handoff while preserving the conditional gate.
"""

# BQGATE: EXPERIMENT pred_a_authority_residual_replay_hooks_finiteness_and_exact_price pred_b_no_single_complete_downstream_module_is_a_dominant_reader pred_c_any_fit_candidate_validates_without_reselection pred_d_any_validated_candidate_is_selective pred_e_residual_branch_remains_reporter_split_stable
from __future__ import annotations

import json
import os

import run_temporal_iswas_v23_residual_carrier_downstream_module_reader_atlas_v1 as atlas
import temporal_iswas_v23_residual_reader_atlas_binding as binding


REGISTERED_PREDICTIONS = {
    "pred_a_authority_residual_replay_hooks_finiteness_and_exact_price":
        "Delegate unchanged to the frozen atlas runner.",
    "pred_b_no_single_complete_downstream_module_is_a_dominant_reader":
        "Delegate unchanged to the frozen atlas runner.",
    "pred_c_any_fit_candidate_validates_without_reselection":
        "Delegate unchanged to the frozen atlas runner.",
    "pred_d_any_validated_candidate_is_selective":
        "Delegate unchanged to the frozen atlas runner.",
    "pred_e_residual_branch_remains_reporter_split_stable":
        "Delegate unchanged to the frozen atlas runner.",
}


def main():
    dry_run = bool(os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"))
    status = binding.create_binding(dry_run=dry_run)
    if dry_run:
        print(json.dumps({
            "candidate_id":
                "cross_task.temporal_iswas.v23_residual_carrier_downstream_module_reader_atlas_v1",
            "conditional_launcher": True,
            "gpu_accessed": False,
            "model_loaded": False,
            "queue_touched": False,
            "binding_status": status,
            "registered_predictions": REGISTERED_PREDICTIONS,
            "price": atlas.PRICE,
        }, sort_keys=True))
        return
    if status.get("status") != "bound":
        raise RuntimeError(f"residual-reader atlas dependency is not licensed: {status}")
    atlas.main()


if __name__ == "__main__":
    main()
