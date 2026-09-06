#!/usr/bin/env python3
"""Fit-only backward greedy compression of converged source-position programs."""

# BQGATE: EXPERIMENT pred_a_authority_reproduction_capability_finiteness_and_price pred_b_both_programs_compress pred_c_pruned_programs_preserve_out_of_sample_quality pred_d_is_was_beats_independent_core_compression pred_e_shared_backbone_survives
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_source_group_eval as source_groups
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import positioned_component_program_eval as positioned
import run_aspectual_tense_carrier_component_greedy_program_v1 as greedy
import run_aspectual_tense_l9h1h4_source_position_weight_validation_v1 as source_rank


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_carrier_program_backward_pruning_v1.json"
CONVERGENCE = ROOT / "circuits/followups/aspectual_tense_carrier_component_convergence_v1_result.json"
COMPOSITION = ROOT / "circuits/followups/aspectual_tense_core_program_composition_v1_result.json"
GREEDY = ROOT / "ops/run_aspectual_tense_carrier_component_greedy_program_v1.py"
POSITIONED = ROOT / "ops/positioned_component_program_eval.py"
SOURCE_RANK = ROOT / "ops/run_aspectual_tense_l9h1h4_source_position_weight_validation_v1.py"
SOURCE_GROUPS = ROOT / "ops/attention_source_group_eval.py"
DAS = ROOT / "ops/circuit_das_subspace.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/aspectual_tense_carrier_program_backward_pruning_v1_result.json"
CANDIDATE_ID = "aspectual_tense.carrier_program_backward_pruning_v1"
EXPECTED = {
    "prior": "ceb64bf33d8498bfbdf5505f27f56f1c1b89696fa10e4207b716789ad225c442",
    "convergence": "7f7872f550b98829f3b0255f3bdc3b16ff49b1dc9d53969d18f922d70645a455",
    "composition": "0757bfaf8f83aa1277123c0325d04f37774e26ccd1b014a3c9c3e788e6fbc9c6",
    "greedy": "22da9d253d531ffb2167302ae269c3b71bdd4fd4e74bc38263dea78c2f80c413",
    "positioned": "4fc2ac355ca7c97a5a1270f88cb3302b1a39a85e62c0b0a8a1193fbc6f61bd0d",
    "source_rank": "c7570a2e25b444df84e40953e38d6bbc4b7b15c6d6f6657fda0696fb4eea3d34",
    "source_groups": "6ecf5f40b92f94cb32bccf1a703e527a3d468281936e63d4c7e91e8af66b4348",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
START_MSE = {"has": 0.06863485941990788, "is": 0.09595690881187888}
MSE_BUDGET = 0.005
MAX_PRICE = {"model_forwards": 122, "example_evaluations": 1460, "records": 1238,
             "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
CORE_IS_BASELINE = {"heldout": 0.5938134454717784, "a2": 0.7664391620364538}
SHARED = {"MLP3", "MLP4", "MLP8"}


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "convergence": CONVERGENCE, "composition": COMPOSITION,
             "greedy": GREEDY, "positioned": POSITIONED, "source_rank": SOURCE_RANK,
             "source_groups": SOURCE_GROUPS, "das": DAS, "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    convergence, composition = [json.loads(path.read_text()) for path in (CONVERGENCE, COMPOSITION)]
    splits, chosen, pools = greedy.validate_static()
    programs = convergence["converged_paths"]
    if (convergence.get("terminal") != "null" or composition.get("terminal") != "null"
            or {task: len(path) for task, path in programs.items()} != {"has": 8, "is": 11}):
        raise ExperimentError("starting program authority changed")
    return convergence, splits, chosen, pools, programs


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def main():
    convergence, splits, chosen, pools, programs = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False,
              "starting_paths": programs, "cumulative_mse_budget": MSE_BUDGET,
              "maximum_price": MAX_PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    records, traces, pruned = [], {"has": [], "is": []}, {}
    all_capable, forwards, evaluations = True, 0, 0
    reproduction = {}
    for task in ("has", "is"):
        rows = splits[f"{task}_fit"]
        base_batch = das._batch(backend, rows, side="base")
        donor_batch = das._batch(backend, rows, side="donor")
        base_output = backend.native(base_batch, capture=False)
        donor_output, cache = positioned.capture_full_components(
            backend, donor_batch, source_rank.capture_specs(chosen[task]))
        banks = source_rank.carrier_banks(task, base_batch, donor_batch)
        all_capable = all_capable and capable(base_output) and capable(donor_output)
        current = list(programs[task])
        full_output = positioned.patch_positioned_components(
            backend, base_batch, donor_batch, greedy.program_specs(current, pools[task]),
            cache, banks, banks)
        full_records = greedy.tagged(source_groups.recovery_records(
            rows, base_output, donor_output, full_output, arm="starting_complete"),
            f"{task}_fit", task)
        records.extend(full_records)
        current_mse = greedy.mse(full_records)
        reproduction[task] = abs(current_mse - START_MSE[task])
        forwards += 3
        evaluations += 3 * len(rows)
        while len(current) > 1:
            candidates = []
            for removed in sorted(current):
                path = [label for label in current if label != removed]
                output = positioned.patch_positioned_components(
                    backend, base_batch, donor_batch, greedy.program_specs(path, pools[task]),
                    cache, banks, banks)
                candidate_records = greedy.tagged(source_groups.recovery_records(
                    rows, base_output, donor_output, output,
                    arm=f"remove_step{len(programs[task])-len(current)+1:02d}:{removed}"),
                    f"{task}_fit", task)
                records.extend(candidate_records)
                candidates.append((greedy.mse(candidate_records), removed, path))
                forwards += 1
                evaluations += len(rows)
            best_mse, removed, path = min(candidates, key=lambda item: (item[0], item[1]))
            accepted = best_mse <= START_MSE[task] + MSE_BUDGET
            traces[task].append({"step": len(programs[task]) - len(current) + 1,
                                 "removed": removed, "mse_before": current_mse,
                                 "mse_after": best_mse, "cumulative_increase": best_mse - START_MSE[task],
                                 "accepted": accepted})
            if not accepted:
                break
            current, current_mse = path, best_mse
        pruned[task] = current

    metrics, bank_widths = {"has": {}, "is": {}}, {}
    for task in ("has", "is"):
        for panel in ("heldout", "a2"):
            split, rows = f"{task}_{panel}", splits[f"{task}_{panel}"]
            base_batch = das._batch(backend, rows, side="base")
            donor_batch = das._batch(backend, rows, side="donor")
            base_output = backend.native(base_batch, capture=False)
            donor_output, cache = positioned.capture_full_components(
                backend, donor_batch, source_rank.capture_specs(chosen[task]))
            banks = source_rank.carrier_banks(task, base_batch, donor_batch)
            bank_widths[split] = sorted(set(map(len, banks)))
            all_capable = all_capable and capable(base_output) and capable(donor_output)
            metrics[task][panel] = {}
            for arm, path in (("complete", programs[task]), ("pruned", pruned[task])):
                output = positioned.patch_positioned_components(
                    backend, base_batch, donor_batch, greedy.program_specs(path, pools[task]),
                    cache, banks, banks)
                arm_records = greedy.tagged(source_groups.recovery_records(
                    rows, base_output, donor_output, output, arm=arm), split, task)
                records.extend(arm_records)
                metrics[task][panel][arm] = source_groups.summarize(arm_records)
            forwards += 4
            evaluations += 4 * len(rows)
    price = {"model_forwards": forwards, "example_evaluations": evaluations,
             "records": len(records), "fitted_scalars": 0,
             "transformer_backwards": 0, "model_updates": 0}
    finite = all(math.isfinite(float(record["recovery"])) for record in records)
    pred_a = bool(max(reproduction.values()) <= 1e-6 and all_capable and finite
                  and bank_widths == {"has_heldout": [3], "has_a2": [3],
                                      "is_heldout": [2], "is_a2": [2]}
                  and all(price[key] <= MAX_PRICE[key] for key in
                          ("model_forwards", "example_evaluations", "records")))
    pred_b = all(len(pruned[task]) < len(programs[task])
                 and any(step["accepted"] for step in traces[task])
                 and max(step["cumulative_increase"] for step in traces[task]
                         if step["accepted"]) <= MSE_BUDGET
                 for task in ("has", "is"))
    pred_c = all(metrics[task][panel]["pruned"]["mean_recovery"]
                 >= metrics[task][panel]["complete"]["mean_recovery"] - 0.03
                 and metrics[task][panel]["pruned"]["direction_fraction"] == 1.0
                 for task in ("has", "is") for panel in ("heldout", "a2"))
    pred_d = all(metrics["is"][panel]["pruned"]["mean_recovery"] > CORE_IS_BASELINE[panel]
                 for panel in ("heldout", "a2"))
    pred_e = all(SHARED <= set(pruned[task]) for task in ("has", "is"))
    predictions = {
        "pred_a_authority_reproduction_capability_finiteness_and_price": pred_a,
        "pred_b_both_programs_compress": pred_b,
        "pred_c_pruned_programs_preserve_out_of_sample_quality": pred_c,
        "pred_d_is_was_beats_independent_core_compression": pred_d,
        "pred_e_shared_backbone_survives": pred_e,
    }
    terminal = "invalid" if not pred_a else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_carrier_program_backward_pruning_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started,
              "authority_sha256": EXPECTED, "starting_paths": programs,
              "pruned_paths": pruned, "traces": traces, "fit_reproduction_error": reproduction,
              "metrics": metrics, "bank_widths": bank_widths,
              "predictions": predictions, "price": price, "records": records,
              "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "pruned_paths", "traces",
          "metrics", "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
