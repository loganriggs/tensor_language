#!/usr/bin/env python3
"""Localize the behaviorally amplified complement residual across five source MLPs."""
# BQGATE: EXPERIMENT pred_a_authority_replay_closure_finiteness_and_price pred_b_one_site_explains_most_missing_iswas_behavior pred_c_residual_localization_is_not_linear_response_ranking pred_d_singletons_add_no_new_parent_collateral pred_e_singleton_effects_license_greedy_composition
import hashlib
import json
import os
from pathlib import Path

import run_temporal_iswas_rank16_source_task_pair_weight_groups_v1 as base
import run_temporal_iswas_rank16_source_task_pair_weight_groups_v2 as population

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank16_source_complement_site_localization_v1.json"
PARENT = ROOT / "circuits/followups/temporal_iswas_rank16_source_task_pair_weight_groups_v2_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_rank16_source_complement_site_localization_v1_result.json"
EXPECTED = {
    "prior": "5328cb30a03b5b3781b8d151e9778850093ef4e430da530b7ea7f51bc79dd9b8",
    "parent": "b69d0cbdba23f363f33a3fd834dbc63d1bfcfd41595ee061bb6587de8cb6aa5e",
    "population_runner": "7389053e5746120f4fa8611b4e182191363a29509088a4551542cf56c1b647fa",
    "weight_group_runner": "8ec435c1e551c37d5ba86cf4be7cd4541de8a7c6cb1f798707309c4119f9adb7",
}
SITE_ARMS = tuple(f"union_plus_{site}_complement" for site in base.SUPPORT)
EXECUTED_ARMS = (
    "full_rank16", "mean_plus_contrast", "mean_plus_contrast_complement",
    "own_task", "cross_task", *SITE_ARMS,
)
MAX_FORWARDS = 44


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    candidate_id = "temporal_auxiliary.iswas_rank16_source_complement_site_localization_v1"
    observed = {
        "prior": sha(PRIOR), "parent": sha(PARENT),
        "population_runner": sha(ROOT / "ops/run_temporal_iswas_rank16_source_task_pair_weight_groups_v2.py"),
        "weight_group_runner": sha(ROOT / "ops/run_temporal_iswas_rank16_source_task_pair_weight_groups_v1.py"),
    }
    if observed != EXPECTED:
        raise RuntimeError(f"complement-site authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({
            "candidate_id": candidate_id, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "arms": EXECUTED_ARMS,
            "model_forwards_max": MAX_FORWARDS, "fit_updates": 0, "model_updates": 0,
            "transformer_backwards": 0,
        }, sort_keys=True))
        return

    parent = json.loads(PARENT.read_text())
    if parent["terminal"] != "probe_basis_only" or not parent["predictions"]["pred_a_authority_replay_closure_finiteness_and_price"]:
        raise RuntimeError("parent is not the valid registered source-weight null")

    base.PRIOR = PRIOR
    base.OUT = OUT
    base.MAX_FORWARDS = MAX_FORWARDS
    base.ARMS = EXECUTED_ARMS
    base.EXPECTED_CONTROL_TASKS = set(base.TASKS)
    base.EXPECTED = {
        "prior": EXPECTED["prior"],
        "weight_compiler_audit": sha(base.AUDIT),
        "mask30_holdout": sha(base.HOLDOUT),
        "mask30_control_attribution": sha(base.ATTRIBUTION),
        "task_mode_helper": sha(base.HELPER),
        "mathematical_review": sha(base.MATH_REVIEW),
    }
    base.oodctx.capture = population.transferred_capture

    original_block_for = base.block_for
    original_run_blocks = base.run_blocks
    original_write = base.atomic_create_json
    cache = {}

    def block_for(blocks, site, arm, task):
        if arm.startswith("union_plus_") and arm.endswith("_complement"):
            selected = arm[len("union_plus_"):-len("_complement")]
            return blocks[site]["full"] if site == selected else blocks[site]["union"]
        return original_block_for(blocks, site, arm, task)

    def run_blocks(backend, batch, rows, base_out, delta_hidden, bases, maps, blocks, arm, *, control=False):
        canonical = {"own_task": "mean_plus_contrast", "cross_task": "mean_plus_contrast_complement"}.get(arm, arm)
        key = (id(batch), canonical, control)
        if key not in cache:
            cache[key] = original_run_blocks(
                backend, batch, rows, base_out, delta_hidden, bases, maps, blocks, canonical, control=control,
            )
        return cache[key]

    def write_result(_path, result):
        reports = result["reports"]
        union, full = reports["mean_plus_contrast"], reports["full_rank16"]
        union_control_flips = set(union["control"]["flipped_row_ids"])
        full_control_flips = set(full["control"]["flipped_row_ids"])
        allowed_flips = union_control_flips | full_control_flips
        missing_behavior = full["behavior_signed_projection"]["iswas"] - union["behavior_signed_projection"]["iswas"]
        singleton = {}
        for site, arm in zip(base.SUPPORT, SITE_ARMS):
            report = reports[arm]
            behavior_gain = report["behavior_signed_projection"]["iswas"] - union["behavior_signed_projection"]["iswas"]
            response_gain = report["response"]["iswas"]["signed_projection"] - union["response"]["iswas"]["signed_projection"]
            singleton[site] = {
                "arm": arm,
                "iswas_behavior_gain": behavior_gain,
                "iswas_behavior_gap_recovery": behavior_gain / max(missing_behavior, 1e-30),
                "iswas_response_gain": response_gain,
                "behavior_minus_response_gain": behavior_gain - response_gain,
                "report": report,
            }
        top_site = max(base.SUPPORT, key=lambda site: (singleton[site]["iswas_behavior_gain"], -base.SUPPORT.index(site)))
        pred_a = bool(result["predictions"]["pred_a_authority_replay_closure_finiteness_and_price"])
        pred_b = singleton[top_site]["iswas_behavior_gap_recovery"] >= .60 and base.source.functional(singleton[top_site]["report"])
        pred_c = singleton[top_site]["behavior_minus_response_gain"] >= .015
        pred_d = all(
            set(item["report"]["control"]["flipped_row_ids"]).issubset(allowed_flips)
            and item["report"]["control"]["median_kl"] <= full["control"]["median_kl"] + .002
            and all(item["report"]["control"]["margin_rms_fraction"][task] <= full["control"]["margin_rms_fraction"][task] + .02 for task in base.TASKS)
            for item in singleton.values()
        )
        additive_residual = {
            task: sum(item["report"]["behavior_signed_projection"][task] - union["behavior_signed_projection"][task] for item in singleton.values())
            - (full["behavior_signed_projection"][task] - union["behavior_signed_projection"][task])
            for task in base.TASKS
        }
        pred_e = all(abs(value) <= .015 for value in additive_residual.values())
        predictions = {
            "pred_a_authority_replay_closure_finiteness_and_price": pred_a,
            "pred_b_one_site_explains_most_missing_iswas_behavior": bool(pred_b),
            "pred_c_residual_localization_is_not_linear_response_ranking": bool(pred_c),
            "pred_d_singletons_add_no_new_parent_collateral": bool(pred_d),
            "pred_e_singleton_effects_license_greedy_composition": bool(pred_e),
        }
        if not pred_a:
            terminal = "invalid"
        elif not pred_d:
            terminal = "selectivity_blocked_residual"
        elif not pred_e:
            terminal = "conditional_complement_interaction"
        elif pred_b and pred_c:
            terminal = "localized_nonlinear_complement_site"
        else:
            terminal = "distributed_additive_complement"
        result.update({
            "schema": "temporal_iswas_rank16_source_complement_site_localization_result_v1",
            "candidate_id": candidate_id, "authority_sha256": EXPECTED,
            "parent_terminal": parent["terminal"], "singleton_localization": singleton,
            "top_behavior_site": top_site, "additive_behavior_residual": additive_residual,
            "predictions": predictions, "terminal": terminal,
        })
        original_write(OUT, result)

    base.block_for = block_for
    base.run_blocks = run_blocks
    base.atomic_create_json = write_result
    base.main()


if __name__ == "__main__":
    main()
