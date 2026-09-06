#!/usr/bin/env python3
"""Greedy exact joint head/module programs at contextual carrier banks."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_instrument pred_b_both_greedy_paths_are_distributive pred_c_both_paths_generalize_to_a1 pred_d_both_paths_transfer_to_a2 pred_e_shared_program_machinery pred_f_price_and_coverage
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
import run_aspectual_tense_l9h1h4_source_position_weight_validation_v1 as source_rank


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_carrier_component_greedy_program_v1.json"
SOURCE_RESULT = ROOT / "circuits/followups/aspectual_tense_l9h1h4_source_position_weight_validation_v1_result.json"
SOURCE_RUNNER = ROOT / "ops/run_aspectual_tense_l9h1h4_source_position_weight_validation_v1.py"
POSITIONED = ROOT / "ops/positioned_component_program_eval.py"
OUT = ROOT / "circuits/followups/aspectual_tense_carrier_component_greedy_program_v1_result.json"
CANDIDATE_ID = "aspectual_tense.carrier_component_greedy_program_v1"
EXPECTED = {
    "prior": "2c8e3508b18386ad1c2d6cbfbb646db926e6b907326c1ddd0acd025b6a2ebf77",
    "source_result": "7692b9c3095e66935934a4a31c7263ea3986f3fd070d4faf271e8ddf6e5ec261",
    "source_runner": "c7570a2e25b444df84e40953e38d6bbc4b7b15c6d6f6657fda0696fb4eea3d34",
    "positioned": "4fc2ac355ca7c97a5a1270f88cb3302b1a39a85e62c0b0a8a1193fbc6f61bd0d",
}
MAX_STEPS = 6
MAX_FORWARDS, MAX_EVALUATIONS, MAX_RECORDS = 322, 4377, 4195


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def component_map(chosen_task):
    result = {}
    for kind in source_rank.TYPES:
        for component in chosen_task[kind]:
            label = component["label"]
            spec = source_rank.component_spec(kind, component)
            if label in result and result[label] != spec:
                raise ExperimentError("component label collision")
            result[label] = spec
    if len(result) != 20:
        raise ExperimentError("frozen component pool is not 20 unique labels")
    return result


def program_specs(labels, components):
    heads, mlps = {}, set()
    for label in labels:
        component = components[label]
        if component.kind == "attention_heads":
            heads.setdefault(component.layer, set()).update(component.heads)
        else:
            mlps.add(component.layer)
    return tuple(positioned.Component("attention_heads", layer, tuple(sorted(selected)))
                 for layer, selected in sorted(heads.items())) + tuple(
        positioned.Component("mlp", layer) for layer in sorted(mlps))


def mse(records):
    return statistics.fmean((1.0 - float(record["recovery"])) ** 2 for record in records)


def rmse(records):
    return math.sqrt(mse(records))


def tagged(records, split, task):
    return [dict(record, split=split, task=task) for record in records]


def validate_static():
    paths = {"prior": PRIOR, "source_result": SOURCE_RESULT,
             "source_runner": SOURCE_RUNNER, "positioned": POSITIONED}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, source authority, or implementation hash changed")
    prior, source = [json.loads(path.read_text()) for path in (PRIOR, SOURCE_RESULT)]
    splits, chosen, _query = source_rank.validate_static()
    pools = {task: component_map(chosen[task]) for task in ("has", "is")}
    if (prior.get("candidate_id") != CANDIDATE_ID or source.get("terminal") != "screen"
            or {name: len(rows) for name, rows in splits.items()} != {
                "has_fit": 16, "has_heldout": 15, "has_a2": 31,
                "is_fit": 8, "is_heldout": 6, "is_a2": 15}
            or any(len(pool) != 20 for pool in pools.values())):
        raise ExperimentError("frozen population, pool, or source screen changed")
    return splits, chosen, pools


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False,
            "split_counts": {"has_fit": 16, "has_heldout": 15, "has_a2": 31,
                             "is_fit": 8, "is_heldout": 6, "is_a2": 15},
            "candidate_pool_per_task": 20, "maximum_components": MAX_STEPS,
            "maximum_model_forwards": MAX_FORWARDS,
            "maximum_example_evaluations": MAX_EVALUATIONS,
            "maximum_intervention_records": MAX_RECORDS, "fitted_scalars": 0,
            "transformer_backwards": 0, "model_updates": 0}


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def main():
    splits, chosen, pools = validate_static()
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
            split = f"{task}_{suffix}"
            rows = splits[split]
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
            items[split] = {"rows": rows, "base_batch": base_batch,
                            "donor_batch": donor_batch, "base_output": base_output,
                            "donor_output": donor_output, "cache": cache, "banks": banks}

    selected_paths, selection_steps = {}, {}
    for task in ("has", "is"):
        item, pool = items[f"{task}_fit"], pools[task]
        selected, steps, current_mse = [], [], 1.0
        for step in range(1, MAX_STEPS + 1):
            trials = []
            for label in sorted(set(pool) - set(selected)):
                labels = selected + [label]
                output = positioned.patch_positioned_components(
                    backend, item["base_batch"], item["donor_batch"],
                    program_specs(labels, pool), item["cache"], item["banks"], item["banks"])
                forwards += 1
                evaluations += len(item["rows"])
                arm_records = tagged(source_groups.recovery_records(
                    item["rows"], item["base_output"], item["donor_output"], output,
                    arm=f"select_step{step:02d}:{label}"), f"{task}_fit", task)
                records.extend(arm_records)
                trials.append((mse(arm_records), label))
            best_mse, best_label = min(trials, key=lambda value: (value[0], value[1]))
            if not best_mse < current_mse:
                break
            steps.append({"step": step, "added": best_label, "mse_before": current_mse,
                          "mse_after": best_mse, "improvement": current_mse - best_mse})
            selected.append(best_label)
            current_mse = best_mse
        selected_paths[task], selection_steps[task] = selected, steps

    final_metrics, singleton_benchmarks = {}, {}
    for task in ("has", "is"):
        final_metrics[task], singleton_benchmarks[task] = {}, {}
        selected, pool = selected_paths[task], pools[task]
        prefixes = [tuple(selected[:width]) for width in range(2, len(selected) + 1)]
        for suffix in ("heldout", "a2"):
            split, item = f"{task}_{suffix}", items[f"{task}_{suffix}"]
            outputs = {}
            arms = [(f"singleton:{label}", (label,)) for label in sorted(pool)]
            arms.extend((f"prefix:{len(prefix):02d}", prefix) for prefix in prefixes)
            for arm, labels in arms:
                output = positioned.patch_positioned_components(
                    backend, item["base_batch"], item["donor_batch"],
                    program_specs(labels, pool), item["cache"], item["banks"], item["banks"])
                forwards += 1
                evaluations += len(item["rows"])
                arm_records = tagged(source_groups.recovery_records(
                    item["rows"], item["base_output"], item["donor_output"], output,
                    arm=arm), split, task)
                records.extend(arm_records)
                outputs[arm] = arm_records
            singles = {label: rmse(outputs[f"singleton:{label}"]) for label in pool}
            singleton_benchmarks[task][suffix] = {
                "rmse_by_component": singles,
                "best_component": min(singles, key=lambda label: (singles[label], label)),
                "best_rmse": min(singles.values()),
            }
            if not selected:
                final_metrics[task][suffix] = None
            else:
                final_arm = f"singleton:{selected[0]}" if len(selected) == 1 else f"prefix:{len(selected):02d}"
                final_records = outputs[final_arm]
                final_metrics[task][suffix] = {
                    **source_groups.summarize(final_records),
                    "unit_target_rmse": rmse(final_records),
                    "beats_best_singleton": rmse(final_records) < min(singles.values()),
                }

    pred_a = bool(all_native_capable and observed_bank_widths == {
        "has_fit": [3], "has_heldout": [3], "has_a2": [3],
        "is_fit": [2], "is_heldout": [2], "is_a2": [2]})
    pred_b = all(2 <= len(selected_paths[task]) <= MAX_STEPS and all(
        step["mse_after"] < step["mse_before"] for step in selection_steps[task])
        for task in ("has", "is"))
    pred_c = all(final_metrics[task]["heldout"] is not None
        and final_metrics[task]["heldout"]["mean_recovery"] >= 0.70
        and final_metrics[task]["heldout"]["direction_fraction"] >= 0.80
        and final_metrics[task]["heldout"]["beats_best_singleton"] for task in ("has", "is"))
    pred_d = all(final_metrics[task]["a2"] is not None
        and final_metrics[task]["a2"]["mean_recovery"] >= 0.65
        and final_metrics[task]["a2"]["direction_fraction"] >= 0.80
        and final_metrics[task]["a2"]["beats_best_singleton"] for task in ("has", "is"))
    shared = sorted(set(selected_paths["has"]) & set(selected_paths["is"]))
    pred_e = len(shared) >= 2
    pred_f = bool(forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS
                  and len(records) <= MAX_RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {
        "pred_a_authority_capability_exact_instrument": pred_a,
        "pred_b_both_greedy_paths_are_distributive": pred_b,
        "pred_c_both_paths_generalize_to_a1": pred_c,
        "pred_d_both_paths_transfer_to_a2": pred_d,
        "pred_e_shared_program_machinery": pred_e,
        "pred_f_price_and_coverage": pred_f,
    }
    terminal = "invalid" if not pred_a or not pred_f else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_carrier_component_greedy_program_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "instrument": {"all_native_capable": all_native_capable,
                  "observed_carrier_bank_widths": observed_bank_widths},
              "frozen_candidate_pools": {task: sorted(pools[task]) for task in ("has", "is")},
              "selected_paths": selected_paths, "selection_steps": selection_steps,
              "shared_selected_components": shared, "final_metrics": final_metrics,
              "singleton_benchmarks": singleton_benchmarks, "predictions": predictions,
              "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                        "intervention_records": len(records), "fitted_scalars": 0,
                        "transformer_backwards": 0, "model_updates": 0},
              "records": records, "terminal": terminal,
              "reason": "sparse_source_resolved_programs_generalize_and_share_machinery" if terminal == "screen"
                        else "greedy_source_programs_do_not_jointly_meet_generalization_and_sharing_bars" if terminal == "null"
                        else "authority_capability_instrument_coverage_or_price_invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
          "selected_paths", "selection_steps", "shared_selected_components", "final_metrics",
          "predictions", "price", "terminal", "reason")}, sort_keys=True))


if __name__ == "__main__":
    main()
