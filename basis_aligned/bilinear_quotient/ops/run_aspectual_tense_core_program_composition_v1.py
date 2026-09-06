#!/usr/bin/env python3
"""Test simultaneous core compression and shared/task-branch composition."""

# BQGATE: EXPERIMENT pred_a_authority_capability_finiteness_and_exact_price pred_b_simultaneous_core_compression_preserves_quality pred_c_shared_backbone_is_substantive pred_d_task_branches_add_task_typed_information pred_e_composition_is_nearly_additive
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
import run_aspectual_tense_carrier_program_leave_one_out_v1 as loo
import run_aspectual_tense_l9h1h4_source_position_weight_validation_v1 as source_rank


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_core_program_composition_v1.json"
NECESSITY = ROOT / "circuits/followups/aspectual_tense_carrier_program_leave_one_out_v1_result.json"
LOO_RUNNER = ROOT / "ops/run_aspectual_tense_carrier_program_leave_one_out_v1.py"
OUT = ROOT / "circuits/followups/aspectual_tense_core_program_composition_v1_result.json"
CANDIDATE_ID = "aspectual_tense.core_program_composition_v1"
EXPECTED = {"prior": "0b7263a4bc823c2e516935749653805db701cae128a84cabe8a288104b9f6b5a",
            "necessity": "92c6eca2b9dd27da62345657f1d57d4a5554b50e9206a0e345e39d5b6963c498",
            "loo_runner": "9ad4d07af58d2cb399fe0c2b2136e6bdd5318e875500b212cc95a0378f7aab3f"}
SHARED = ["MLP3", "MLP4", "MLP8"]
BRANCH = {"has": ["L8H1", "MLP2", "MLP6", "MLP7"], "is": ["MLP1"]}
PRICE = {"model_forwards": 24, "example_evaluations": 402, "records": 268,
         "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "necessity": NECESSITY, "loo_runner": LOO_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority hash changed")
    necessity = json.loads(NECESSITY.read_text())
    convergence, splits, chosen, pools, complete = loo.validate_static()
    core = necessity["core_programs"]
    if (necessity.get("terminal") != "null" or core != {
            "has": ["MLP4", "L8H1", "MLP3", "MLP6", "MLP8", "MLP2", "MLP7"],
            "is": ["MLP1", "MLP4", "MLP3", "MLP8"]}
            or set(core["has"]) & set(core["is"]) != set(SHARED)
            or any(set(core[task]) != set(SHARED) | set(BRANCH[task]) for task in core)):
        raise ExperimentError("frozen core decomposition changed")
    return necessity, splits, chosen, pools, complete, core


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def main():
    necessity, splits, chosen, pools, complete, core = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False,
              "programs": {"complete": complete, "core": core,
                           "shared": SHARED, "branches": BRANCH}, **PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    records, metrics, interactions = [], {"has": {}, "is": {}}, {"has": {}, "is": {}}
    all_capable, bank_widths = True, {}
    forwards = evaluations = 0
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
            arms = {"complete": complete[task], "core": core[task],
                    "shared": SHARED, "branch": BRANCH[task]}
            metrics[task][panel] = {}
            for arm, labels in arms.items():
                output = positioned.patch_positioned_components(
                    backend, base_batch, donor_batch,
                    greedy.program_specs(labels, pools[task]), cache, banks, banks)
                arm_records = greedy.tagged(source_groups.recovery_records(
                    rows, base_output, donor_output, output, arm=arm), split, task)
                records.extend(arm_records)
                metrics[task][panel][arm] = source_groups.summarize(arm_records)
            interactions[task][panel] = (metrics[task][panel]["core"]["mean_recovery"]
                                         - metrics[task][panel]["shared"]["mean_recovery"]
                                         - metrics[task][panel]["branch"]["mean_recovery"])
            forwards += 6
            evaluations += 6 * len(rows)
    observed_price = {"model_forwards": forwards, "example_evaluations": evaluations,
                      "records": len(records), "fitted_scalars": 0,
                      "transformer_backwards": 0, "model_updates": 0}
    finite = all(math.isfinite(float(record["recovery"])) for record in records)
    pred_a = bool(all_capable and finite and bank_widths == {
        "has_heldout": [3], "has_a2": [3], "is_heldout": [2], "is_a2": [2]}
        and observed_price == PRICE)
    pred_b = all(metrics[task][panel]["core"]["mean_recovery"]
                 >= metrics[task][panel]["complete"]["mean_recovery"] - 0.03
                 and metrics[task][panel]["core"]["direction_fraction"] == 1.0
                 for task in ("has", "is") for panel in ("heldout", "a2"))
    pred_c = all(metrics[task][panel]["shared"]["mean_recovery"] >= 0.50
                 for task in ("has", "is") for panel in ("heldout", "a2"))
    pred_d = all(metrics[task][panel]["core"]["mean_recovery"]
                 - metrics[task][panel]["shared"]["mean_recovery"] >= 0.01
                 for task in ("has", "is") for panel in ("heldout", "a2"))
    pred_e = all(abs(interactions[task][panel]) <= 0.10
                 for task in ("has", "is") for panel in ("heldout", "a2"))
    predictions = {
        "pred_a_authority_capability_finiteness_and_exact_price": pred_a,
        "pred_b_simultaneous_core_compression_preserves_quality": pred_b,
        "pred_c_shared_backbone_is_substantive": pred_c,
        "pred_d_task_branches_add_task_typed_information": pred_d,
        "pred_e_composition_is_nearly_additive": pred_e,
    }
    terminal = "invalid" if not pred_a else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_core_program_composition_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started,
              "authority_sha256": EXPECTED, "programs": {"complete": complete,
                  "core": core, "shared": SHARED, "branches": BRANCH},
              "metrics": metrics, "interactions": interactions,
              "bank_widths": bank_widths, "predictions": predictions,
              "price": observed_price, "records": records, "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "programs", "metrics",
          "interactions", "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
