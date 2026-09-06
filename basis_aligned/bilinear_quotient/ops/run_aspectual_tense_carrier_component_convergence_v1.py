#!/usr/bin/env python3
"""Converge source-resolved component programs by marginal fit improvement."""

# BQGATE: EXPERIMENT pred_a_authority_prefix_reproduction_capability_and_exact_price pred_b_is_was_reaches_marginal_convergence pred_c_converged_paths_preserve_out_of_sample_quality pred_d_programs_remain_sparse_and_shared pred_e_zero_fit_parameters
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
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_carrier_component_convergence_v1.json"
V2_RESULT = ROOT / "circuits/followups/aspectual_tense_carrier_component_greedy_program_v2_result.json"
GREEDY_RUNNER = ROOT / "ops/run_aspectual_tense_carrier_component_greedy_program_v1.py"
POSITIONED = ROOT / "ops/positioned_component_program_eval.py"
OUT = ROOT / "circuits/followups/aspectual_tense_carrier_component_convergence_v1_result.json"
CANDIDATE_ID = "aspectual_tense.carrier_component_convergence_v1"
EXPECTED = {
    "prior": "37d1e5f252dc600d2b54319d299e09024ed1f97491fc551e7e7085704abe95af",
    "v2_result": "a503afad73051d9bf589ac71bc9421d495f456075269b98c570d2fbf8d8253fd",
    "greedy_runner": "22da9d253d531ffb2167302ae269c3b71bdd4fd4e74bc38263dea78c2f80c413",
    "positioned": "4fc2ac355ca7c97a5a1270f88cb3302b1a39a85e62c0b0a8a1193fbc6f61bd0d",
}
MIN_IMPROVEMENT = 0.001
MAX_FORWARDS, MAX_EVALUATIONS, MAX_RECORDS = 70, 665, 515
STARTING_MSE = 0.09803752882624002
V2_BASELINES = {
    "has": {"heldout": 0.7550742490667587, "a2": 0.7668711904725405},
    "is": {"heldout": 0.694381731633972, "a2": 0.8244637221144694},
}


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "v2_result": V2_RESULT,
             "greedy_runner": GREEDY_RUNNER, "positioned": POSITIONED}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, v2 evidence, or implementation hash changed")
    prior, v2 = [json.loads(path.read_text()) for path in (PRIOR, V2_RESULT)]
    splits, chosen, pools = greedy.validate_static()
    has_path = v2["selected_paths"]["has"][:8]
    is_path = v2["selected_paths"]["is"][:10]
    if (prior.get("candidate_id") != CANDIDATE_ID or v2.get("terminal") != "null"
            or len(has_path) != 8 or len(is_path) != 10
            or abs(v2["selection_steps"]["has"][8]["improvement"] - 0.00048388576714038234) > 1e-15
            or abs(v2["selection_steps"]["is"][9]["mse_after"] - STARTING_MSE) > 1e-15):
        raise ExperimentError("frozen prefixes or convergence evidence changed")
    return splits, chosen, pools, {"has": has_path, "is": is_path}


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def main():
    splits, chosen, pools, paths = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False,
              "marginal_improvement_threshold": MIN_IMPROVEMENT,
              "starting_path_lengths": {task: len(path) for task, path in paths.items()},
              "maximum_model_forwards": MAX_FORWARDS,
              "maximum_example_evaluations": MAX_EVALUATIONS,
              "maximum_records": MAX_RECORDS, "fitted_scalars": 0,
              "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    forwards = evaluations = 0
    records, all_capable = [], True

    rows = splits["is_fit"]
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_output = backend.native(base_batch, capture=False)
    donor_output, cache = positioned.capture_full_components(
        backend, donor_batch, source_rank.capture_specs(chosen["is"]))
    banks = source_rank.carrier_banks("is", base_batch, donor_batch)
    forwards += 2
    evaluations += 2 * len(rows)
    all_capable = capable(base_output) and capable(donor_output)
    current_output = positioned.patch_positioned_components(
        backend, base_batch, donor_batch, greedy.program_specs(paths["is"], pools["is"]),
        cache, banks, banks)
    forwards += 1
    evaluations += len(rows)
    current_records = greedy.tagged(source_groups.recovery_records(
        rows, base_output, donor_output, current_output, arm="starting_prefix10"), "is_fit", "is")
    records.extend(current_records)
    current_mse = greedy.mse(current_records)
    reproduction_error = abs(current_mse - STARTING_MSE)
    continuation_steps, rejected_step = [], None
    for step in range(11, 21):
        trials = []
        for label in sorted(set(pools["is"]) - set(paths["is"])):
            candidate_path = paths["is"] + [label]
            output = positioned.patch_positioned_components(
                backend, base_batch, donor_batch,
                greedy.program_specs(candidate_path, pools["is"]), cache, banks, banks)
            forwards += 1
            evaluations += len(rows)
            trial_records = greedy.tagged(source_groups.recovery_records(
                rows, base_output, donor_output, output,
                arm=f"select_step{step:02d}:{label}"), "is_fit", "is")
            records.extend(trial_records)
            trials.append((greedy.mse(trial_records), label))
        best_mse, best_label = min(trials, key=lambda item: (item[0], item[1]))
        improvement = current_mse - best_mse
        if improvement <= MIN_IMPROVEMENT:
            rejected_step = {"step": step, "best_candidate": best_label,
                             "mse_before": current_mse, "best_mse": best_mse,
                             "best_improvement": improvement, "threshold": MIN_IMPROVEMENT}
            break
        paths["is"].append(best_label)
        continuation_steps.append({"step": step, "added": best_label,
            "mse_before": current_mse, "mse_after": best_mse, "improvement": improvement})
        current_mse = best_mse

    final_metrics = {"has": {}, "is": {}}
    observed_bank_widths = {}
    for task in ("has", "is"):
        for suffix in ("heldout", "a2"):
            split, panel_rows = f"{task}_{suffix}", splits[f"{task}_{suffix}"]
            base_batch, donor_batch = das._batch(backend, panel_rows, side="base"), das._batch(backend, panel_rows, side="donor")
            base_output = backend.native(base_batch, capture=False)
            donor_output, cache = positioned.capture_full_components(
                backend, donor_batch, source_rank.capture_specs(chosen[task]))
            banks = source_rank.carrier_banks(task, base_batch, donor_batch)
            output = positioned.patch_positioned_components(
                backend, base_batch, donor_batch,
                greedy.program_specs(paths[task], pools[task]), cache, banks, banks)
            forwards += 3
            evaluations += 3 * len(panel_rows)
            all_capable = all_capable and capable(base_output) and capable(donor_output)
            observed_bank_widths[split] = sorted(set(map(len, banks)))
            panel_records = greedy.tagged(source_groups.recovery_records(
                panel_rows, base_output, donor_output, output, arm="converged_program"), split, task)
            records.extend(panel_records)
            final_metrics[task][suffix] = {**source_groups.summarize(panel_records),
                                           "unit_target_rmse": greedy.rmse(panel_records)}

    pred_a = bool(reproduction_error <= 1e-5 and all_capable
                  and observed_bank_widths == {"has_heldout": [3], "has_a2": [3],
                                                "is_heldout": [2], "is_a2": [2]}
                  and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS
                  and len(records) <= MAX_RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    pred_b = bool(len(paths["is"]) > 10 and len(paths["is"]) < 20 and rejected_step
                  and rejected_step["best_improvement"] <= MIN_IMPROVEMENT
                  and all(step["improvement"] > MIN_IMPROVEMENT for step in continuation_steps))
    pred_c = bool(all(final_metrics["has"][panel]["mean_recovery"]
                      >= V2_BASELINES["has"][panel] - 0.005 for panel in ("heldout", "a2"))
                  and all(final_metrics["is"][panel]["mean_recovery"]
                          >= V2_BASELINES["is"][panel] for panel in ("heldout", "a2"))
                  and all(final_metrics[task][panel]["direction_fraction"] == 1.0
                          for task in ("has", "is") for panel in ("heldout", "a2")))
    shared = sorted(set(paths["has"]) & set(paths["is"]))
    pred_d = len(paths["has"]) < 20 and len(paths["is"]) < 20 and len(shared) >= 8
    pred_e = True
    predictions = {
        "pred_a_authority_prefix_reproduction_capability_and_exact_price": pred_a,
        "pred_b_is_was_reaches_marginal_convergence": pred_b,
        "pred_c_converged_paths_preserve_out_of_sample_quality": pred_c,
        "pred_d_programs_remain_sparse_and_shared": pred_d,
        "pred_e_zero_fit_parameters": pred_e,
    }
    terminal = "invalid" if not pred_a or not pred_e else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_carrier_component_convergence_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "starting_prefix_reproduction_error": reproduction_error,
        "converged_paths": paths, "continuation_steps": continuation_steps,
        "rejected_step": rejected_step, "shared_components": shared,
        "final_metrics": final_metrics, "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fitted_scalars": 0,
                  "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal,
        "reason": "marginally_converged_programs_preserve_out_of_sample_quality" if terminal == "screen"
                  else "marginal_convergence_does_not_preserve_quality_or_sparsity" if terminal == "null"
                  else "authority_reproduction_capability_coverage_finiteness_stopping_or_price_invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "starting_prefix_reproduction_error",
        "converged_paths", "continuation_steps", "rejected_step", "shared_components",
        "final_metrics", "predictions", "price", "terminal", "reason")}, sort_keys=True))


if __name__ == "__main__":
    main()
