#!/usr/bin/env python3
"""Causal factorial for cue-conditioned response modes in fixed temporal readers."""

# BQGATE: EXPERIMENT pred_a_authority_and_closure pred_b_own_modes_stable pred_c_cross_modes_distinct pred_d_joint_span_sufficient pred_e_joint_span_selective pred_f_price_exact
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch

import attention_source_group_eval as source_groups
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fresh_builder
import circuit_fast_screen_candidate_temporal_auxiliary as original_builder
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_auxiliary_will_had_fresh_writer_to_reader_coefficients_v3 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_cross_cue_reader_modes_v1.json"
OOD_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_summed_writer_compiler_ood_v1_result.json"
FRESH_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
ORIGINAL_BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
PARENT_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_fresh_writer_to_reader_coefficients_v3.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_cross_cue_reader_modes_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.cross_cue_reader_modes_v1"
EXPECTED = {
    "prior": "c0d37d340954e2b3628e666d2d4c7085ead9117f9fd4724c604dee1a4654b2a7",
    "ood_result": "c7aa1a1f605467ccaf43199d3ec0eb08633b71f79089ca700fdf5a5cbc4614a6",
    "fresh_builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "original_builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
    "parent_runner": "ce7822a6a0ae41f330b478663ddc8b1a48f0ce0314609cfcad0c8bc35fbf24ab",
}
ARMS = ("full_response", "own_mode", "cross_mode", "joint_two_mode_span", "joint_span_complement")
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 72, 1760, 480


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def balanced(rows):
    a1 = [row for row in rows if row["transform_id"] == "A1"]
    fit = [row for index, row in enumerate(a1) if (index // 2) % 2 == 0]
    heldout = [row for index, row in enumerate(a1) if (index // 2) % 2 == 1]
    return fit, heldout


def validate_static():
    paths = {"prior": PRIOR, "ood_result": OOD_RESULT, "fresh_builder": FRESH_BUILDER,
             "original_builder": ORIGINAL_BUILDER, "parent_runner": PARENT_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, builder, or helper hash changed")
    prior = json.loads(PRIOR.read_text())
    ood_result = json.loads(OOD_RESULT.read_text())
    corpora = {"fresh": fresh_builder.build_rows(), "original": original_builder.build_rows()}
    splits = {}
    for corpus, rows in corpora.items():
        fit, heldout = balanced(rows)
        splits[f"{corpus}_fit"] = fit
        splits[f"{corpus}_heldout"] = heldout
        splits[f"{corpus}_a2"] = [row for row in rows if row["transform_id"] == "A2"]
        splits[f"{corpus}_p"] = [row for row in rows if row["transform_id"] == "P"]
    direction_counts = {name: {direction: sum(row["direction_id"] == direction for row in rows)
                               for direction in ("future_to_anterior", "anterior_to_future")}
                        for name, rows in splits.items() if name.endswith(("_fit", "_heldout"))}
    if (prior.get("candidate_id") != CANDIDATE_ID or ood_result.get("terminal") != "cue_specific"
            or any(counts != {"future_to_anterior": 8, "anterior_to_future": 8}
                   for counts in direction_counts.values())
            or any(len(rows) != (16 if name.endswith(("_fit", "_heldout")) else 32)
                   for name, rows in splits.items())
            or any(len(row["base_ids"]) != len(row["donor_ids"])
                   for rows in splits.values() for row in rows)):
        raise ExperimentError("authority, balanced split, population, or alignment changed")
    return splits


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "arms": list(ARMS),
            "model_forwards": MODEL_FORWARDS, "example_evaluations": EXAMPLE_EVALUATIONS,
            "answer_changing_records": RECORDS, "fitted_scalars": 0,
            "stored_axis_coordinates": 768, "transformer_backwards": 0, "model_updates": 0}


def project(matrix, q, complement=False):
    projected = (matrix @ q) @ q.T
    return matrix - projected if complement else projected


def main():
    split_rows = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    items = {name: parent.prepare_item(backend, rows) for name, rows in split_rows.items()}
    forwards = 6 * len(items)
    evaluations = sum(6 * len(rows) for rows in split_rows.values())
    matrices = {name: {"block9": parent.reader_matrix(item, 9, (1, 4)),
                       "block11": parent.reader_matrix(item, 11, (3,))}
                for name, item in items.items()}
    axes = {corpus: {reader: parent.first_axis(matrices[f"{corpus}_fit"][reader])[:, None]
                     for reader in ("block9", "block11")}
            for corpus in ("fresh", "original")}
    joint_axes = {reader: torch.linalg.qr(torch.cat((axes["fresh"][reader], axes["original"][reader]), dim=1))[0]
                  for reader in ("block9", "block11")}
    axis_cosines = {reader: float((axes["fresh"][reader][:, 0] @ axes["original"][reader][:, 0]).abs())
                    for reader in ("block9", "block11")}

    records, summaries = [], {}
    for corpus in ("fresh", "original"):
        other = "original" if corpus == "fresh" else "fresh"
        for split in ("heldout", "a2"):
            name = f"{corpus}_{split}"
            item = items[name]
            full = matrices[name]
            vector_sets = {
                "full_response": full,
                "own_mode": {reader: project(full[reader], axes[corpus][reader])
                             for reader in ("block9", "block11")},
                "cross_mode": {reader: project(full[reader], axes[other][reader])
                               for reader in ("block9", "block11")},
                "joint_two_mode_span": {reader: project(full[reader], joint_axes[reader])
                                        for reader in ("block9", "block11")},
                "joint_span_complement": {reader: project(full[reader], joint_axes[reader], complement=True)
                                          for reader in ("block9", "block11")},
            }
            outputs = {arm: parent.run_responses(backend, item, values["block9"], values["block11"])
                       for arm, values in vector_sets.items()}
            forwards += len(vector_sets)
            evaluations += len(vector_sets) * len(item["rows"])
            for arm in ARMS:
                for record in source_groups.recovery_records(
                        item["rows"], item["base_output"], item["donor_output"], outputs[arm], arm=arm):
                    record["split"] = name
                    records.append(record)
            summaries[name] = {arm: source_groups.summarize([
                record for record in records if record["split"] == name and record["arm"] == arm])
                for arm in ARMS}

    controls = {}
    for corpus in ("fresh", "original"):
        item = items[f"{corpus}_p"]
        full = matrices[f"{corpus}_p"]
        own = {reader: project(full[reader], axes[corpus][reader])
               for reader in ("block9", "block11")}
        joint = {reader: project(full[reader], joint_axes[reader])
                 for reader in ("block9", "block11")}
        outputs = {"own_mode": parent.run_responses(backend, item, own["block9"], own["block11"]),
                   "joint_two_mode_span": parent.run_responses(
                       backend, item, joint["block9"], joint["block11"])}
        forwards += 2
        evaluations += 2 * len(item["rows"])
        target_scale = statistics.median(abs(donor - base)
            for name in (f"{corpus}_heldout", f"{corpus}_a2")
            for base, donor in zip(parent.axis_values(items[name]["base_output"]),
                                   parent.axis_values(items[name]["donor_output"])))
        controls[corpus] = {arm: statistics.mean(abs(value - base) / target_scale
            for value, base in zip(parent.axis_values(output), parent.axis_values(item["base_output"])))
            for arm, output in outputs.items()}

    fractions = {name: {arm: values[arm]["mean_recovery"] / values["full_response"]["mean_recovery"]
                        for arm in ARMS[1:]}
                 for name, values in summaries.items()}
    identity_error = max(item["identity_error"] for item in items.values())
    reconstruction_error = max(item["reconstruction_error"] for item in items.values())
    pred_a = bool(identity_error <= 1e-4 and reconstruction_error <= 5e-4
                  and all(values["full_response"]["direction_fraction"] == 1.0
                          for values in summaries.values()))
    pred_b = all(0.70 <= fractions[name]["own_mode"] <= 1.20
                 and summaries[name]["own_mode"]["direction_fraction"] >= 0.90
                 for name in summaries)
    pred_c = all(fractions[f"{corpus}_heldout"]["own_mode"]
                 - fractions[f"{corpus}_heldout"]["cross_mode"] >= 0.20
                 for corpus in ("fresh", "original"))
    pred_d = all(0.85 <= fractions[name]["joint_two_mode_span"] <= 1.20
                 and summaries[name]["joint_two_mode_span"]["direction_fraction"] >= 0.90
                 and abs(fractions[name]["joint_span_complement"]) <= 0.20
                 for name in summaries)
    pred_e = all(values["joint_two_mode_span"] <= 0.20 for values in controls.values())
    pred_f = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["split"], record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {"pred_a_authority_and_closure": pred_a,
                   "pred_b_own_modes_stable": pred_b,
                   "pred_c_cross_modes_distinct": pred_c,
                   "pred_d_joint_span_sufficient": pred_d,
                   "pred_e_joint_span_selective": pred_e,
                   "pred_f_price_exact": pred_f}
    terminal = "invalid" if not pred_a or not pred_f else (
        "two_mode_quotient" if all(predictions.values()) else
        "shared_mode" if pred_b and pred_d and not pred_c else
        "unrelated_circuits" if pred_b and not pred_d else "null")
    result = {"schema": "temporal_auxiliary_cross_cue_reader_modes_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "instrument": {"capture_identity_max_abs": identity_error,
              "attention_reconstruction_max_abs": reconstruction_error},
              "cross_cue_axis_absolute_cosines": axis_cosines, "summaries": summaries,
              "fraction_of_full_response": fractions, "controls": controls,
              "predictions": predictions, "price": {"model_forwards": forwards,
              "example_evaluations": evaluations, "answer_changing_records": len(records),
              "fitted_scalars": 0, "stored_axis_coordinates": 768,
              "transformer_backwards": 0, "model_updates": 0},
              "records": records, "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
          "cross_cue_axis_absolute_cosines", "fraction_of_full_response", "controls",
          "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
