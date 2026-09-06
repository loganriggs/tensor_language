#!/usr/bin/env python3
"""Exact leave-one-component necessity for converged source-position programs."""

# BQGATE: EXPERIMENT pred_a_authority_capability_finiteness_and_exact_price pred_b_complete_programs_reproduce pred_c_majority_of_each_program_is_core pred_d_task_exclusive_components_are_functional pred_e_shared_mlp_backbone_is_jointly_necessary
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
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_carrier_program_leave_one_out_v1.json"
CONVERGENCE = ROOT / "circuits/followups/aspectual_tense_carrier_component_convergence_v1_result.json"
GREEDY = ROOT / "ops/run_aspectual_tense_carrier_component_greedy_program_v1.py"
POSITIONED = ROOT / "ops/positioned_component_program_eval.py"
SOURCE_RANK = ROOT / "ops/run_aspectual_tense_l9h1h4_source_position_weight_validation_v1.py"
SOURCE_GROUPS = ROOT / "ops/attention_source_group_eval.py"
DAS = ROOT / "ops/circuit_das_subspace.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/aspectual_tense_carrier_program_leave_one_out_v1_result.json"
CANDIDATE_ID = "aspectual_tense.carrier_program_leave_one_out_v1"
EXPECTED = {
    "prior": "64602f9e4b1a2686739c616940a8d72c735c6ce948d1c41502561c4504b7b70c",
    "convergence": "7f7872f550b98829f3b0255f3bdc3b16ff49b1dc9d53969d18f922d70645a455",
    "greedy": "22da9d253d531ffb2167302ae269c3b71bdd4fd4e74bc38263dea78c2f80c413",
    "positioned": "4fc2ac355ca7c97a5a1270f88cb3302b1a39a85e62c0b0a8a1193fbc6f61bd0d",
    "source_rank": "c7570a2e25b444df84e40953e38d6bbc4b7b15c6d6f6657fda0696fb4eea3d34",
    "source_groups": "6ecf5f40b92f94cb32bccf1a703e527a3d468281936e63d4c7e91e8af66b4348",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
PRICE = {"model_forwards": 50, "example_evaluations": 800, "records": 666,
         "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
CORE_THRESHOLD = 0.01


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "convergence": CONVERGENCE, "greedy": GREEDY,
             "positioned": POSITIONED, "source_rank": SOURCE_RANK,
             "source_groups": SOURCE_GROUPS, "das": DAS, "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    convergence = json.loads(CONVERGENCE.read_text())
    splits, chosen, pools = greedy.validate_static()
    programs = convergence["converged_paths"]
    if (convergence.get("terminal") != "null"
            or {task: len(path) for task, path in programs.items()} != {"has": 8, "is": 11}
            or any(not set(programs[task]) <= set(pools[task]) for task in programs)):
        raise ExperimentError("converged program authority changed")
    return convergence, splits, chosen, pools, programs


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def main():
    convergence, splits, chosen, pools, programs = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False,
              "program_lengths": {task: len(path) for task, path in programs.items()},
              "core_threshold": CORE_THRESHOLD, **PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    records, panel_metrics = [], {"has": {}, "is": {}}
    all_capable, bank_widths = True, {}
    forwards = evaluations = 0
    for task in ("has", "is"):
        for panel in ("heldout", "a2"):
            split = f"{task}_{panel}"
            rows = splits[split]
            base_batch = das._batch(backend, rows, side="base")
            donor_batch = das._batch(backend, rows, side="donor")
            base_output = backend.native(base_batch, capture=False)
            donor_output, cache = positioned.capture_full_components(
                backend, donor_batch, source_rank.capture_specs(chosen[task]))
            banks = source_rank.carrier_banks(task, base_batch, donor_batch)
            bank_widths[split] = sorted(set(map(len, banks)))
            all_capable = all_capable and capable(base_output) and capable(donor_output)
            outputs = {}
            arms = [("complete", programs[task])]
            arms += [(f"without:{label}", [item for item in programs[task] if item != label])
                     for label in programs[task]]
            for arm, labels in arms:
                outputs[arm] = positioned.patch_positioned_components(
                    backend, base_batch, donor_batch,
                    greedy.program_specs(labels, pools[task]), cache, banks, banks)
                arm_records = greedy.tagged(source_groups.recovery_records(
                    rows, base_output, donor_output, outputs[arm], arm=arm), split, task)
                records.extend(arm_records)
            forwards += 2 + len(arms)
            evaluations += (2 + len(arms)) * len(rows)
            complete = [record for record in records
                        if record.get("split") == split and record.get("arm") == "complete"]
            panel_metrics[task][panel] = {
                "complete": source_groups.summarize(complete),
                "without": {
                    label: source_groups.summarize([record for record in records
                        if record.get("split") == split and record.get("arm") == f"without:{label}"])
                    for label in programs[task]}}

    necessity, core = {"has": {}, "is": {}}, {"has": [], "is": []}
    for task in ("has", "is"):
        counts = {panel: panel_metrics[task][panel]["complete"]["count"]
                  for panel in ("heldout", "a2")}
        for label in programs[task]:
            scores = {panel: (panel_metrics[task][panel]["complete"]["mean_recovery"]
                              - panel_metrics[task][panel]["without"][label]["mean_recovery"])
                      for panel in ("heldout", "a2")}
            pooled = sum(scores[panel] * counts[panel] for panel in scores) / sum(counts.values())
            is_core = all(value > 0 for value in scores.values()) and pooled >= CORE_THRESHOLD
            necessity[task][label] = {**scores, "pooled": pooled, "core": is_core}
            if is_core:
                core[task].append(label)

    reproduction = max(abs(panel_metrics[task][panel]["complete"]["mean_recovery"]
                           - convergence["final_metrics"][task][panel]["mean_recovery"])
                       for task in ("has", "is") for panel in ("heldout", "a2"))
    finite = all(math.isfinite(float(record["recovery"])) for record in records)
    observed_price = {"model_forwards": forwards, "example_evaluations": evaluations,
                      "records": len(records), "fitted_scalars": 0,
                      "transformer_backwards": 0, "model_updates": 0}
    pred_a = bool(all_capable and finite and bank_widths == {
        "has_heldout": [3], "has_a2": [3], "is_heldout": [2], "is_a2": [2]}
        and observed_price == PRICE)
    pred_b = reproduction <= 1e-6
    pred_c = all(len(core[task]) >= math.ceil(len(programs[task]) / 2)
                 for task in ("has", "is"))
    exclusive = {task: set(programs[task]) - set(programs["is" if task == "has" else "has"])
                 for task in ("has", "is")}
    pred_d = all(bool(set(core[task]) & exclusive[task]) for task in ("has", "is"))
    shared_mlp = {label for label in set(programs["has"]) & set(programs["is"])
                  if label.startswith("MLP")}
    pred_e = len(shared_mlp & set(core["has"]) & set(core["is"])) >= 4
    predictions = {
        "pred_a_authority_capability_finiteness_and_exact_price": pred_a,
        "pred_b_complete_programs_reproduce": pred_b,
        "pred_c_majority_of_each_program_is_core": pred_c,
        "pred_d_task_exclusive_components_are_functional": pred_d,
        "pred_e_shared_mlp_backbone_is_jointly_necessary": pred_e,
    }
    terminal = "invalid" if not pred_a or not pred_b else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_carrier_program_leave_one_out_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started,
              "authority_sha256": EXPECTED, "core_threshold": CORE_THRESHOLD,
              "programs": programs, "panel_metrics": panel_metrics,
              "necessity": necessity, "core_programs": core,
              "reproduction_max_abs_error": reproduction,
              "bank_widths": bank_widths, "predictions": predictions,
              "price": observed_price, "records": records, "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "core_programs",
          "necessity", "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
