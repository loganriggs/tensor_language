#!/usr/bin/env python3
"""Complete five-site composition lattice for the distributed source complement."""
# BQGATE: EXPERIMENT pred_a_authority_replay_closure_finiteness_and_price pred_b_strict_subset_is_full_equivalent pred_c_frozen_greedy_finds_global_minimum pred_d_all_subset_effects_are_nearly_additive pred_e_selected_sites_are_minimal_and_selective
import hashlib
import json
import math
import os
from pathlib import Path

import run_temporal_iswas_rank16_source_task_pair_weight_groups_v1 as base
import run_temporal_iswas_rank16_source_task_pair_weight_groups_v2 as population

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_rank16_source_complement_composition_lattice_v1.json"
LOCALIZER = ROOT / "circuits/followups/temporal_iswas_rank16_source_complement_site_localization_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_rank16_source_complement_composition_lattice_v1_result.json"
EXPECTED = {
    "prior": "0df2f26571b97bc24e8277cfa37ccea824015c84c5dfeef372dc2b3cef846b75",
    "localizer_result": "cfe4914d653da25ba65f6e73cf447d54403d5f3b3f3036100584b4c552becbaa",
    "localizer_runner": "ee8d39c36909a0db7975cef4814d7ba4f95d4e42bf2aea6b3d270dbcc361688c",
    "population_runner": "7389053e5746120f4fa8611b4e182191363a29509088a4551542cf56c1b647fa",
    "weight_group_runner": "8ec435c1e551c37d5ba86cf4be7cd4541de8a7c6cb1f798707309c4119f9adb7",
}
MASK_ARMS = tuple(f"addback_mask_{mask:02d}" for mask in range(32))
BASE_ARMS = (
    "full_rank16", "shared_mean", "task_contrast", "mean_plus_contrast",
    "own_task", "cross_task", "mean_plus_contrast_complement",
)
EXECUTED_ARMS = BASE_ARMS + MASK_ARMS
GREEDY_ORDER = ("MLP3", "MLP6", "MLP1", "MLP2", "MLP0")
MAX_FORWARDS = 96
EQUIVALENCE_TOLERANCE = 0.015


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def mask_from_arm(arm):
    return int(arm.rsplit("_", 1)[1])


def target_gap(report, full):
    values = []
    for task in base.TASKS:
        values.append(abs(report["behavior_signed_projection"][task] - full["behavior_signed_projection"][task]))
        values.append(abs(report["response"][task]["signed_projection"] - full["response"][task]["signed_projection"]))
    return max(values)


def control_ok(report, full):
    left, right = report["control"], full["control"]
    return (
        set(left["flipped_row_ids"]).issubset(right["flipped_row_ids"])
        and left["median_kl"] <= right["median_kl"] + .002
        and set(left["margin_rms_fraction"]) == set(right["margin_rms_fraction"])
        and all(left["margin_rms_fraction"][task] <= right["margin_rms_fraction"][task] + .02 for task in right["margin_rms_fraction"])
    )


def main():
    candidate_id = "temporal_auxiliary.iswas_rank16_source_complement_composition_lattice_v1"
    dry = {
        "candidate_id": candidate_id, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "sites": base.SUPPORT,
        "masks": list(range(32)), "frozen_greedy_order": GREEDY_ORDER,
        "model_forwards_max": MAX_FORWARDS, "fit_updates": 0,
        "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    observed = {
        "prior": sha(PRIOR), "localizer_result": sha(LOCALIZER),
        "localizer_runner": sha(ROOT / "ops/run_temporal_iswas_rank16_source_complement_site_localization_v1.py"),
        "population_runner": sha(ROOT / "ops/run_temporal_iswas_rank16_source_task_pair_weight_groups_v2.py"),
        "weight_group_runner": sha(ROOT / "ops/run_temporal_iswas_rank16_source_task_pair_weight_groups_v1.py"),
    }
    if observed != EXPECTED:
        raise RuntimeError(f"composition authority changed: {observed}")
    localizer = json.loads(LOCALIZER.read_text())
    if localizer["terminal"] != "distributed_additive_complement":
        raise RuntimeError("singleton authority does not license composition")

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
        if arm.startswith("addback_mask_"):
            mask = mask_from_arm(arm)
            bit = base.SUPPORT.index(site)
            return blocks[site]["full"] if mask & (1 << bit) else blocks[site]["union"]
        return original_block_for(blocks, site, arm, task)

    def run_blocks(backend, batch, rows, base_out, delta_hidden, bases, maps, blocks, arm, *, control=False):
        canonical = {"addback_mask_00": "mean_plus_contrast", "addback_mask_31": "full_rank16"}.get(arm, arm)
        key = (id(batch), canonical, control)
        if key not in cache:
            cache[key] = original_run_blocks(
                backend, batch, rows, base_out, delta_hidden, bases, maps, blocks, canonical, control=control,
            )
        return cache[key]

    def write_result(_path, result):
        reports = result["reports"]
        union, full = reports["mean_plus_contrast"], reports["full_rank16"]
        mask_reports = {mask: reports[f"addback_mask_{mask:02d}"] for mask in range(32)}

        def eligible(mask):
            report = mask_reports[mask]
            return base.source.functional(report) and target_gap(report, full) <= EQUIVALENCE_TOLERANCE and control_ok(report, full)

        eligible_masks = [mask for mask in range(32) if eligible(mask)]
        selected = min(eligible_masks, key=lambda mask: (mask.bit_count(), target_gap(mask_reports[mask], full), mask)) if eligible_masks else None
        greedy_masks = [0]
        running = 0
        for site in GREEDY_ORDER:
            running |= 1 << base.SUPPORT.index(site)
            greedy_masks.append(running)
        greedy_selected = next((mask for mask in greedy_masks if mask in eligible_masks), None)

        additive_max = {"behavior": {task: 0.0 for task in base.TASKS}, "response": {task: 0.0 for task in base.TASKS}}
        for mask, report in mask_reports.items():
            for task in base.TASKS:
                expected_behavior = union["behavior_signed_projection"][task]
                expected_response = union["response"][task]["signed_projection"]
                for bit, site in enumerate(base.SUPPORT):
                    if mask & (1 << bit):
                        singleton = localizer["singleton_localization"][site]["report"]
                        expected_behavior += singleton["behavior_signed_projection"][task] - union["behavior_signed_projection"][task]
                        expected_response += singleton["response"][task]["signed_projection"] - union["response"][task]["signed_projection"]
                additive_max["behavior"][task] = max(additive_max["behavior"][task], abs(report["behavior_signed_projection"][task] - expected_behavior))
                additive_max["response"][task] = max(additive_max["response"][task], abs(report["response"][task]["signed_projection"] - expected_response))

        replay = max(target_gap(mask_reports[0], union), target_gap(mask_reports[31], full))
        pred_a = bool(
            result["predictions"]["pred_a_authority_replay_closure_finiteness_and_price"]
            and replay <= 1e-6 and finite(mask_reports)
            and result["price"]["model_forwards_observed"] <= MAX_FORWARDS
        )
        pred_b = selected is not None and selected.bit_count() <= 3
        pred_c = selected is not None and greedy_selected == selected
        pred_d = max(value for family in additive_max.values() for value in family.values()) <= .015
        selected_control_ok = selected is not None and control_ok(mask_reports[selected], full)
        leave_one_out = [] if selected is None else [selected & ~(1 << bit) for bit in range(5) if selected & (1 << bit)]
        pred_e = selected_control_ok and all(mask not in eligible_masks for mask in leave_one_out)
        predictions = {
            "pred_a_authority_replay_closure_finiteness_and_price": bool(pred_a),
            "pred_b_strict_subset_is_full_equivalent": bool(pred_b),
            "pred_c_frozen_greedy_finds_global_minimum": bool(pred_c),
            "pred_d_all_subset_effects_are_nearly_additive": bool(pred_d),
            "pred_e_selected_sites_are_minimal_and_selective": bool(pred_e),
        }
        if not pred_a:
            terminal = "invalid"
        elif not pred_b:
            terminal = "full_complement_required"
        elif not pred_d:
            terminal = "conditional_source_complement_program"
        elif not pred_c:
            terminal = "non_greedy_source_complement_program"
        elif pred_e:
            terminal = "minimal_source_complement_program"
        else:
            terminal = "full_complement_required"
        result.update({
            "schema": "temporal_iswas_rank16_source_complement_composition_lattice_result_v1",
            "candidate_id": candidate_id, "authority_sha256": EXPECTED,
            "parent_terminal": localizer["terminal"], "mask_reports": mask_reports,
            "eligible_masks": eligible_masks, "selected_mask": selected,
            "selected_sites": [] if selected is None else [site for bit, site in enumerate(base.SUPPORT) if selected & (1 << bit)],
            "selected_worst_target_gap": None if selected is None else target_gap(mask_reports[selected], full),
            "frozen_greedy_order": GREEDY_ORDER, "greedy_masks": greedy_masks,
            "greedy_selected_mask": greedy_selected, "leave_one_out_masks": leave_one_out,
            "additivity_max_abs_error": additive_max, "mask_replay_max_abs_error": replay,
            "predictions": predictions, "terminal": terminal,
        })
        original_write(OUT, result)

    base.block_for = block_for
    base.run_blocks = run_blocks
    base.atomic_create_json = write_result
    base.main()


if __name__ == "__main__":
    main()
