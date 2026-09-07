#!/usr/bin/env python3
"""Transferred-population repair for the exact source task-pair weight groups."""
# BQGATE: EXPERIMENT pred_a_authority_replay_closure_finiteness_and_price pred_b_task_pair_blocks_are_crossfit_stable pred_c_own_task_beats_cross_task pred_d_mean_plus_contrast_is_functional_and_full_equivalent pred_e_union_complement_is_insufficient pred_f_weight_grouping_adds_no_parent_collateral
import hashlib
import json
import os
from pathlib import Path

import circuit_das_subspace as das
import run_temporal_iswas_rank16_hidden_mask30_construction_holdout_v1 as transfer
import run_temporal_iswas_rank16_source_task_pair_weight_groups_v1 as base

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank16_source_task_pair_weight_groups_v2.json"
OUT = ROOT / "circuits/followups/temporal_iswas_rank16_source_task_pair_weight_groups_v2_result.json"
INVALID = ROOT / "circuits/followups/temporal_iswas_rank16_source_task_pair_weight_groups_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_temporal_iswas_rank16_source_task_pair_weight_groups_v1.py"
TRANSFER_RUNNER = ROOT / "ops/run_temporal_iswas_rank16_hidden_mask30_construction_holdout_v1.py"
PRECHECK = {
    "prior": "4a2a7f04611784e6be875199169ed1524006991958442728811cc67d01d9869a",
    "invalid_v1": "20199f66edb3a3694ae75ec0c49f41542b83c8b35cf7cd9485c9186795b52e92",
    "base_runner": "8ec435c1e551c37d5ba86cf4be7cd4541de8a7c6cb1f798707309c4119f9adb7",
    "transfer_runner": "3a8aac81e646cab9da9ea06130cba5b14eb844f7c1e142be8cbaf69041c875e3",
}
REGISTERED_PREDICTIONS = (
    "pred_a_authority_replay_closure_finiteness_and_price",
    "pred_b_task_pair_blocks_are_crossfit_stable",
    "pred_c_own_task_beats_cross_task",
    "pred_d_mean_plus_contrast_is_functional_and_full_equivalent",
    "pred_e_union_complement_is_insufficient",
    "pred_f_weight_grouping_adds_no_parent_collateral",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transferred_capture(backend, *_unused):
    tcap = json.loads(transfer.TCAP.read_text())
    icap = json.loads(transfer.ICAP.read_text())
    temporal, iswas, temporal_controls, iswas_controls = transfer.greedy.rows_and_controls(tcap, icap)
    rows = temporal + iswas
    controls = temporal_controls + iswas_controls
    batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    control_batch = das._batch(backend, controls, side="base")
    control_donor_batch = das._batch(backend, controls, side="donor")
    _base_output, base_full = base.atlas.capture_native(backend, batch)
    control_output, control_base_full = base.atlas.capture_native(backend, control_batch)
    return {
        "rows": rows, "controls": controls, "batch": batch, "donor_batch": donor_batch,
        "control_batch": control_batch, "control_donor_batch": control_donor_batch,
        "base_full": base_full, "control_base_full": control_base_full,
        "control_base_state": base.atlas.states(backend.torch, backend, control_output, controls),
    }


def main():
    observed = {
        "prior": sha(PRIOR), "invalid_v1": sha(INVALID), "base_runner": sha(BASE_RUNNER),
        "transfer_runner": sha(TRANSFER_RUNNER),
    }
    if observed != PRECHECK:
        raise RuntimeError(f"v2 population-repair authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({
            "candidate_id": "temporal_auxiliary.iswas_rank16_source_task_pair_weight_groups_v2",
            "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
            "population": "temporal-v13/iswas-v12", "arms": base.ARMS,
            "model_forwards_max": base.MAX_FORWARDS, "fit_updates": 0, "model_updates": 0,
            "transformer_backwards": 0,
        }, sort_keys=True))
        return
    base.PRIOR = PRIOR
    base.OUT = OUT
    base.EXPECTED = {
        "prior": PRECHECK["prior"],
        "weight_compiler_audit": sha(base.AUDIT),
        "mask30_holdout": sha(base.HOLDOUT),
        "mask30_control_attribution": sha(base.ATTRIBUTION),
        "task_mode_helper": sha(base.HELPER),
        "mathematical_review": sha(base.MATH_REVIEW),
    }
    base.EXPECTED_CONTROL_TASKS = set(base.TASKS)
    base.oodctx.capture = transferred_capture
    original_write = base.atomic_create_json

    def write_result(_path, result):
        result["schema"] = "temporal_iswas_rank16_source_task_pair_weight_groups_result_v2"
        result["candidate_id"] = "temporal_auxiliary.iswas_rank16_source_task_pair_weight_groups_v2"
        result["population_repair_authority_sha256"] = PRECHECK
        original_write(OUT, result)

    base.atomic_create_json = write_result
    base.main()


if __name__ == "__main__":
    main()
