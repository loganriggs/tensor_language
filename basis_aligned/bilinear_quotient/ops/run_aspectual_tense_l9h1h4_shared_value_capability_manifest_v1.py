#!/usr/bin/env python3
"""Freeze jointly native-capable aspectual/tense row IDs before subspace fitting."""

# BQGATE: EXPERIMENT pred_a_parent_counts_recur pred_b_stratified_joint_population_exists pred_c_exact_coverage
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_mlp4_h1h4_bank_routing_local_value_factorial_v1 as has_factor
import run_tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_v1 as is_factor


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_l9h1h4_shared_value_capability_manifest_v1.json"
INVALID_PARENT = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_weight_subspace_v1_result.json"
HAS_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp4_h1h4_bank_routing_local_value_factorial_v1.py"
IS_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_mlp4_h1h4_bank_routing_local_value_factorial_v1.py"
OUT = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_capability_manifest_v1_result.json"
CANDIDATE_ID = "aspectual_tense.l9h1h4_shared_value_capability_manifest_v1"
EXPECTED = {
    "prior": "b114a277745bb2ed4674978596b71ddc5b4276d96cddd64a3643cf8272d8985f",
    "invalid_parent": "1ed214e9e6857624216b975d9f981d2b3c3ce404850650d1f5ab48f21d8d2530",
    "has_runner": "33e208e2b256fa6916bb61f408ffddc376c4c4871fe419f555c8a76422006374",
    "is_runner": "6826d33fadd2af133000cb3c826b4d89c535f576c3f057e667e68dece98e7d39",
}


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "invalid_parent": INVALID_PARENT,
             "has_runner": HAS_RUNNER, "is_runner": IS_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority hash changed")
    prior = json.loads(PRIOR.read_text())
    parent = json.loads(INVALID_PARENT.read_text())
    has_rows, _spec = has_factor.validate_static()
    is_rows = is_factor.validate_static()
    rows = {"has": has_rows, "is": is_rows}
    counts = {task: {family: sum(row["transform_id"] == family for row in task_rows)
                     for family in ("A1", "A2")} for task, task_rows in rows.items()}
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "invalid"
            or counts != {"has": {"A1": 32, "A2": 32},
                          "is": {"A1": 16, "A2": 16}}):
        raise ExperimentError("candidate, invalid parent, or population changed")
    return rows


def main():
    rows = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "model_forwards": 8,
              "example_evaluations": 192, "records": 192,
              "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    records = []
    for task in ("has", "is"):
        for family in ("A1", "A2"):
            selected = [row for row in rows[task] if row["transform_id"] == family]
            for side in ("base", "donor"):
                output = backend.native(das._batch(backend, selected, side=side), capture=False)
                for row, (answer, foil) in zip(selected, output.answer_foil):
                    margin = float(answer) - float(foil)
                    records.append({"task": task, "row_id": row["row_id"], "family": family,
                                    "direction": row["direction_id"], "side": side,
                                    "answer_minus_foil": margin, "correct": margin > 0})

    counts = {task: {family: {side: sum(record["correct"] for record in records
                                         if record["task"] == task
                                         and record["family"] == family
                                         and record["side"] == side)
                              for side in ("base", "donor")}
                     for family in ("A1", "A2")} for task in ("has", "is")}
    jointly_capable = {}
    stratum_counts = {}
    for task in ("has", "is"):
        jointly_capable[task] = {}
        stratum_counts[task] = {}
        for family in ("A1", "A2"):
            selected = [row for row in rows[task] if row["transform_id"] == family]
            ids = sorted(row["row_id"] for row in selected if all(
                next(record["correct"] for record in records
                     if record["task"] == task and record["row_id"] == row["row_id"]
                     and record["family"] == family and record["side"] == side)
                for side in ("base", "donor")))
            jointly_capable[task][family] = ids
            stratum_counts[task][family] = {
                direction: sum(row["row_id"] in ids and row["direction_id"] == direction
                               for row in selected)
                for direction in sorted({row["direction_id"] for row in selected})}

    pred_a = counts == {
        "has": {"A1": {"base": 31, "donor": 32},
                "A2": {"base": 31, "donor": 32}},
        "is": {"A1": {"base": 15, "donor": 15},
               "A2": {"base": 15, "donor": 16}},
    }
    pred_b = all(count >= 6 for task in stratum_counts.values()
                 for family in task.values() for count in family.values())
    pred_c = bool(len(records) == 192
                  and len({(record["task"], record["row_id"], record["family"], record["side"])
                           for record in records}) == 192
                  and all(math.isfinite(record["answer_minus_foil"]) for record in records))
    predictions = {"pred_a_parent_counts_recur": pred_a,
                   "pred_b_stratified_joint_population_exists": pred_b,
                   "pred_c_exact_coverage": pred_c}
    result = {
        "schema": "aspectual_tense_l9h1h4_shared_value_capability_manifest_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "counts": counts, "jointly_capable_row_ids": jointly_capable,
        "jointly_capable_stratum_counts": stratum_counts, "predictions": predictions,
        "price": {"model_forwards": 8, "example_evaluations": 192,
                  "records": len(records), "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": "manifest" if all(predictions.values()) else "invalid",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "counts",
          "jointly_capable_stratum_counts", "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
