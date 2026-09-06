#!/usr/bin/env python3
"""Greedy multi-task-regularized shared carrier component program."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_instrument pred_b_shared_path_is_distributive pred_c_shared_path_generalizes_to_both_a1_panels pred_d_shared_path_transfers_to_both_a2_panels pred_e_multitask_regularization_repairs_is_heldout pred_f_price_and_coverage
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import attention_source_group_eval as source_groups
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import positioned_component_program_eval as positioned
import run_aspectual_tense_carrier_component_greedy_program_v1 as greedy
import run_aspectual_tense_l9h1h4_source_position_weight_validation_v1 as source_rank


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_multitask_shared_carrier_program_v1.json"
GREEDY_RESULT = ROOT / "circuits/followups/aspectual_tense_carrier_component_greedy_program_v1_result.json"
GREEDY_RUNNER = ROOT / "ops/run_aspectual_tense_carrier_component_greedy_program_v1.py"
OUT = ROOT / "circuits/followups/aspectual_tense_multitask_shared_carrier_program_v1_result.json"
CANDIDATE_ID = "aspectual_tense.multitask_shared_carrier_program_v1"
EXPECTED = {
    "prior": "cb58164c258256a3d728bebb160a60a412184ec65a945e80049d8b26887262aa",
    "greedy_result": "600fed015e2dd9046d885373c85cec70127ee67f922565ebcf3545a5598da4eb",
    "greedy_runner": "22da9d253d531ffb2167302ae269c3b71bdd4fd4e74bc38263dea78c2f80c413",
}
MAX_STEPS = 6
MAX_FORWARDS, MAX_EVALUATIONS, MAX_RECORDS = 270, 3744, 3562
PRIOR_IS_HELDOUT = 0.6542596410264659
PRIOR_HAS_HELDOUT_FLOOR = 0.6862649359677497


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    if {"prior": sha(PRIOR), "greedy_result": sha(GREEDY_RESULT),
            "greedy_runner": sha(GREEDY_RUNNER)} != EXPECTED:
        raise ExperimentError("prior or greedy authority hash changed")
    prior, parent = [json.loads(path.read_text()) for path in (PRIOR, GREEDY_RESULT)]
    splits, chosen, pools = greedy.validate_static()
    common = tuple(sorted(set(pools["has"]) & set(pools["is"])))
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "null"
            or len(common) != 17
            or parent["final_metrics"]["is"]["heldout"]["mean_recovery"] != PRIOR_IS_HELDOUT
            or abs(parent["final_metrics"]["has"]["heldout"]["mean_recovery"] - 0.05
                   - PRIOR_HAS_HELDOUT_FLOOR) > 1e-15):
        raise ExperimentError("common pool or frozen parent baseline changed")
    return splits, chosen, pools, common


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "common_pool": 17,
            "maximum_components": MAX_STEPS, "maximum_model_forwards": MAX_FORWARDS,
            "maximum_example_evaluations": MAX_EVALUATIONS,
            "maximum_intervention_records": MAX_RECORDS, "fitted_scalars": 0,
            "transformer_backwards": 0, "model_updates": 0}


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def main():
    splits, chosen, pools, common = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    items, records = {}, []
    forwards = evaluations = 0
    all_native_capable = True
    observed_bank_widths = {}
    for task in ("has", "is"):
        for suffix in ("fit", "heldout", "a2"):
            split, rows = f"{task}_{suffix}", splits[f"{task}_{suffix}"]
            base_batch = das._batch(backend, rows, side="base")
            donor_batch = das._batch(backend, rows, side="donor")
            base_output = backend.native(base_batch, capture=False)
            donor_output, cache = positioned.capture_full_components(
                backend, donor_batch, source_rank.capture_specs(chosen[task]))
            banks = source_rank.carrier_banks(task, base_batch, donor_batch)
            observed_bank_widths[split] = sorted(set(map(len, banks)))
            all_native_capable = all_native_capable and capable(base_output) and capable(donor_output)
            forwards += 2
            evaluations += 2 * len(rows)
            items[split] = {"rows": rows, "base_batch": base_batch, "donor_batch": donor_batch,
                "base_output": base_output, "donor_output": donor_output, "cache": cache, "banks": banks}

    selected, selection_steps, current_mse = [], [], 1.0
    shared_specs = pools["has"]
    for step in range(1, MAX_STEPS + 1):
        trials = []
        for label in sorted(set(common) - set(selected)):
            task_records, task_mse = {}, {}
            for task in ("has", "is"):
                item = items[f"{task}_fit"]
                output = positioned.patch_positioned_components(
                    backend, item["base_batch"], item["donor_batch"],
                    greedy.program_specs(selected + [label], shared_specs), item["cache"],
                    item["banks"], item["banks"])
                forwards += 1
                evaluations += len(item["rows"])
                arm_records = greedy.tagged(source_groups.recovery_records(
                    item["rows"], item["base_output"], item["donor_output"], output,
                    arm=f"select_step{step:02d}:{label}"), f"{task}_fit", task)
                records.extend(arm_records)
                task_records[task] = arm_records
                task_mse[task] = greedy.mse(arm_records)
            trials.append((statistics.fmean(task_mse.values()), label, task_mse))
        best_mse, best_label, best_task_mse = min(trials, key=lambda value: (value[0], value[1]))
        if not best_mse < current_mse:
            break
        selection_steps.append({"step": step, "added": best_label,
            "equal_task_mse_before": current_mse, "equal_task_mse_after": best_mse,
            "improvement": current_mse - best_mse, "task_mse_after": best_task_mse})
        selected.append(best_label)
        current_mse = best_mse

    final_metrics, singleton_benchmarks = {}, {}
    prefixes = [tuple(selected[:width]) for width in range(2, len(selected) + 1)]
    for task in ("has", "is"):
        final_metrics[task], singleton_benchmarks[task] = {}, {}
        for suffix in ("heldout", "a2"):
            split, item = f"{task}_{suffix}", items[f"{task}_{suffix}"]
            outputs = {}
            arms = [(f"singleton:{label}", (label,)) for label in common]
            arms.extend((f"prefix:{len(prefix):02d}", prefix) for prefix in prefixes)
            for arm, labels in arms:
                output = positioned.patch_positioned_components(
                    backend, item["base_batch"], item["donor_batch"],
                    greedy.program_specs(labels, shared_specs), item["cache"],
                    item["banks"], item["banks"])
                forwards += 1
                evaluations += len(item["rows"])
                arm_records = greedy.tagged(source_groups.recovery_records(
                    item["rows"], item["base_output"], item["donor_output"], output,
                    arm=arm), split, task)
                records.extend(arm_records)
                outputs[arm] = arm_records
            singles = {label: greedy.rmse(outputs[f"singleton:{label}"]) for label in common}
            singleton_benchmarks[task][suffix] = {"rmse_by_component": singles,
                "best_component": min(singles, key=lambda label: (singles[label], label)),
                "best_rmse": min(singles.values())}
            if selected:
                final_arm = f"singleton:{selected[0]}" if len(selected) == 1 else f"prefix:{len(selected):02d}"
                final_records = outputs[final_arm]
                final_metrics[task][suffix] = {**source_groups.summarize(final_records),
                    "unit_target_rmse": greedy.rmse(final_records),
                    "beats_best_singleton": greedy.rmse(final_records) < min(singles.values())}
            else:
                final_metrics[task][suffix] = None

    pred_a = bool(all_native_capable and observed_bank_widths == {
        "has_fit": [3], "has_heldout": [3], "has_a2": [3],
        "is_fit": [2], "is_heldout": [2], "is_a2": [2]})
    pred_b = bool(2 <= len(selected) <= MAX_STEPS and all(
        step["equal_task_mse_after"] < step["equal_task_mse_before"] for step in selection_steps))
    def passes(task, suffix):
        metric = final_metrics[task][suffix]
        return bool(metric and metric["mean_recovery"] >= 0.70
                    and metric["direction_fraction"] >= 0.80 and metric["beats_best_singleton"])
    pred_c = all(passes(task, "heldout") for task in ("has", "is"))
    pred_d = all(passes(task, "a2") for task in ("has", "is"))
    pred_e = bool(final_metrics["is"]["heldout"]
                  and final_metrics["is"]["heldout"]["mean_recovery"] > PRIOR_IS_HELDOUT
                  and final_metrics["has"]["heldout"]["mean_recovery"] >= PRIOR_HAS_HELDOUT_FLOOR)
    pred_f = bool(forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS
                  and len(records) <= MAX_RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {
        "pred_a_authority_capability_exact_instrument": pred_a,
        "pred_b_shared_path_is_distributive": pred_b,
        "pred_c_shared_path_generalizes_to_both_a1_panels": pred_c,
        "pred_d_shared_path_transfers_to_both_a2_panels": pred_d,
        "pred_e_multitask_regularization_repairs_is_heldout": pred_e,
        "pred_f_price_and_coverage": pred_f,
    }
    terminal = "invalid" if not pred_a or not pred_f else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_multitask_shared_carrier_program_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"all_native_capable": all_native_capable,
            "observed_carrier_bank_widths": observed_bank_widths},
        "common_candidate_pool": list(common), "selected_path": selected,
        "selection_steps": selection_steps, "final_metrics": final_metrics,
        "singleton_benchmarks": singleton_benchmarks, "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "intervention_records": len(records), "fitted_scalars": 0,
                  "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal,
        "reason": "multitask_regularized_shared_carrier_program_generalizes" if terminal == "screen"
                  else "shared_multitask_program_does_not_meet_all_generalization_bars" if terminal == "null"
                  else "authority_capability_instrument_coverage_or_price_invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "selected_path",
        "selection_steps", "final_metrics", "predictions", "price", "terminal", "reason")},
        sort_keys=True))


if __name__ == "__main__":
    main()
