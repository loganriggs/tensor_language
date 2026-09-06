#!/usr/bin/env python3
"""Transfer the summed-writer response compiler to Tomorrow/Earlier prompts."""

# BQGATE: EXPERIMENT pred_a_ood_capability_and_writer_closure pred_b_ood_coefficient_prediction pred_c_ood_program_material pred_d_prediction_beats_intercept pred_e_ood_p_selective pred_f_price_exact
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
import circuit_fast_screen_candidate_temporal_auxiliary as ood_builder
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_auxiliary_will_had_fresh_writer_to_reader_coefficients_v3 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_summed_writer_compiler_ood_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_destination_resolved_coefficients_v1_result.json"
FRESH_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
OOD_BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
PARENT_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_fresh_writer_to_reader_coefficients_v3.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_summed_writer_compiler_ood_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.summed_writer_compiler_ood_v1"
RESULT_SCHEMA = "temporal_auxiliary_summed_writer_compiler_ood_result_v1"
EXPECTED_PARENT_TERMINAL = "writer_summary_insufficient"
PREDICTION_NAMES = (
    "pred_a_ood_capability_and_writer_closure",
    "pred_b_ood_coefficient_prediction",
    "pred_c_ood_program_material",
    "pred_d_prediction_beats_intercept",
    "pred_e_ood_p_selective",
    "pred_f_price_exact",
)
TERMINAL_NAMES = {"success": "screen", "transfer_fail": "cue_specific",
                  "behavioral_fail": "behavioral_null"}
EXPECTED = {
    "prior": "4069c4c5fc683d1cd241fffe6e958e7cbc72f1fefa6ba73f58b04b721c13043d",
    "parent_result": "951830b296fe0b8ce47d8b7a08e78e0875ba40a1ed3216d049cdc00c11d01929",
    "fresh_builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "ood_builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
    "parent_runner": "ce7822a6a0ae41f330b478663ddc8b1a48f0ce0314609cfcad0c8bc35fbf24ab",
}
ARMS = ("full_captured_response", "oracle_rank1_response", "predicted_rank1_response",
        "intercept_only_response", "zero_response")
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 38, 1104, 320


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "parent_result": PARENT_RESULT, "fresh_builder": FRESH_BUILDER,
             "ood_builder": OOD_BUILDER, "parent_runner": PARENT_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, builder, or helper hash changed")
    prior = json.loads(PRIOR.read_text())
    parent_result = json.loads(PARENT_RESULT.read_text())
    fresh = [row for row in fresh_builder.build_rows() if row["transform_id"] == "A1"]
    fit = [row for index, row in enumerate(fresh) if (index // 2) % 2 == 0]
    ood = ood_builder.build_rows()
    family_rows = {family: [row for row in ood if row["transform_id"] == family]
                   for family in ("A1", "A2", "P")}
    fit_directions = {direction: sum(row["direction_id"] == direction for row in fit)
                      for direction in ("future_to_anterior", "anterior_to_future")}
    if (prior.get("candidate_id") != CANDIDATE_ID
            or parent_result.get("terminal") != EXPECTED_PARENT_TERMINAL
            or fit_directions != {"future_to_anterior": 8, "anterior_to_future": 8}
            or {key: len(value) for key, value in family_rows.items()}
            != {"A1": 32, "A2": 32, "P": 32}
            or any(len(row["base_ids"]) != len(row["donor_ids"])
                   for selected in (fit, *family_rows.values()) for row in selected)):
        raise ExperimentError("fit authority, OOD population, or alignment changed")
    return {"fit": fit, "ood_a1": family_rows["A1"],
            "ood_a2": family_rows["A2"], "ood_p": family_rows["P"]}


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "answer_arms": list(ARMS),
            "model_forwards": MODEL_FORWARDS, "example_evaluations": EXAMPLE_EVALUATIONS,
            "answer_changing_records": RECORDS, "fitted_scalars": 4,
            "stored_axis_coordinates": 512, "transformer_backwards": 0, "model_updates": 0}


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
    matrices = {name: {"terms": parent.writer_terms(item),
                       "block9": parent.reader_matrix(item, 9, (1, 4)),
                       "block11": parent.reader_matrix(item, 11, (3,))}
                for name, item in items.items()}
    direct_errors = {}
    for name, item in items.items():
        direct = parent.run_direct_writer(backend, item, matrices[name]["terms"])
        direct_errors[name] = parent.pair_error(direct, item["writer_output"])
        forwards += 1
        evaluations += len(item["rows"])

    writer_fit = matrices["fit"]["terms"].sum(dim=1)
    writer_axis = parent.first_axis(writer_fit)
    reader_axes = {reader: parent.first_axis(matrices["fit"][reader])
                   for reader in ("block9", "block11")}
    x_fit = writer_fit @ writer_axis
    coefficients = {reader: parent.affine_fit(
        x_fit, matrices["fit"][reader] @ reader_axes[reader])
        for reader in ("block9", "block11")}
    intercepts = {reader: float((matrices["fit"][reader] @ reader_axes[reader]).mean())
                  for reader in ("block9", "block11")}
    predicted, correlations = {}, {}
    for name in ("ood_a1", "ood_a2", "ood_p"):
        x = matrices[name]["terms"].sum(dim=1) @ writer_axis
        predicted[name] = {reader: coefficients[reader][0] * x + coefficients[reader][1]
                           for reader in ("block9", "block11")}
        if name != "ood_p":
            correlations[name] = {reader: parent.correlation(
                predicted[name][reader], matrices[name][reader] @ reader_axes[reader])
                for reader in ("block9", "block11")}

    records, summaries = [], {}
    for name in ("ood_a1", "ood_a2"):
        item = items[name]
        full = {reader: matrices[name][reader] for reader in ("block9", "block11")}
        oracle = {reader: (full[reader] @ reader_axes[reader])[:, None] * reader_axes[reader][None, :]
                  for reader in ("block9", "block11")}
        predicted_vectors = {reader: predicted[name][reader][:, None] * reader_axes[reader][None, :]
                             for reader in ("block9", "block11")}
        intercept = {reader: torch.full_like(predicted[name][reader], intercepts[reader])[:, None]
                     * reader_axes[reader][None, :] for reader in ("block9", "block11")}
        vectors = {"full_captured_response": full, "oracle_rank1_response": oracle,
                   "predicted_rank1_response": predicted_vectors,
                   "intercept_only_response": intercept}
        outputs = {arm: parent.run_responses(backend, item, value["block9"], value["block11"])
                   for arm, value in vectors.items()}
        outputs["zero_response"] = item["base_output"]
        forwards += len(vectors)
        evaluations += len(vectors) * len(item["rows"])
        for arm in ARMS:
            for record in source_groups.recovery_records(
                    item["rows"], item["base_output"], item["donor_output"], outputs[arm], arm=arm):
                record["split"] = name
                records.append(record)
        summaries[name] = {arm: source_groups.summarize([
            record for record in records if record["split"] == name and record["arm"] == arm])
            for arm in ARMS}

    item = items["ood_p"]
    predicted_vectors = {reader: predicted["ood_p"][reader][:, None] * reader_axes[reader][None, :]
                         for reader in ("block9", "block11")}
    intercept = {reader: torch.full_like(predicted["ood_p"][reader], intercepts[reader])[:, None]
                 * reader_axes[reader][None, :] for reader in ("block9", "block11")}
    control_outputs = {
        "predicted_rank1_response": parent.run_responses(
            backend, item, predicted_vectors["block9"], predicted_vectors["block11"]),
        "intercept_only_response": parent.run_responses(
            backend, item, intercept["block9"], intercept["block11"])}
    forwards += 2
    evaluations += 2 * len(item["rows"])
    fractions = {name: {arm: summaries[name][arm]["mean_recovery"]
                        / summaries[name]["full_captured_response"]["mean_recovery"]
                        for arm in ARMS[1:]}
                 for name in ("ood_a1", "ood_a2")}
    target_scale = statistics.median(abs(donor - base)
        for name in ("ood_a1", "ood_a2")
        for base, donor in zip(parent.axis_values(items[name]["base_output"]),
                               parent.axis_values(items[name]["donor_output"])))
    controls = {arm: statistics.mean(abs(value - base) / target_scale
        for value, base in zip(parent.axis_values(output), parent.axis_values(item["base_output"])))
        for arm, output in control_outputs.items()}
    capability = {name: {"base_correct": sum(answer > foil for answer, foil in items[name]["base_output"].answer_foil),
                         "donor_correct": sum(answer > foil for answer, foil in items[name]["donor_output"].answer_foil),
                         "total": len(items[name]["rows"])}
                  for name in ("ood_a1", "ood_a2")}
    pred_a = bool(max(direct_errors.values()) <= 1e-4
                  and max(item["identity_error"] for item in items.values()) <= 1e-4
                  and max(item["reconstruction_error"] for item in items.values()) <= 5e-4
                  and all(value["base_correct"] == value["total"]
                          and value["donor_correct"] == value["total"] for value in capability.values()))
    pred_b = all(abs(correlations[name][reader]) >= 0.60
                 for name in ("ood_a1", "ood_a2") for reader in ("block9", "block11"))
    pred_c = all(fractions[name]["predicted_rank1_response"] >= 0.60
                 and summaries[name]["predicted_rank1_response"]["direction_fraction"] >= 0.75
                 for name in ("ood_a1", "ood_a2"))
    pred_d = all(fractions[name]["predicted_rank1_response"]
                 - fractions[name]["intercept_only_response"] >= 0.15
                 for name in ("ood_a1", "ood_a2"))
    pred_e = controls["predicted_rank1_response"] <= 0.20
    pred_f = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["split"], record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records)
                  and all(math.isfinite(correlations[name][reader])
                          for name in correlations for reader in correlations[name]))
    predictions = dict(zip(PREDICTION_NAMES, (pred_a, pred_b, pred_c, pred_d, pred_e, pred_f)))
    terminal = "invalid" if not pred_a or not pred_f else (
        TERMINAL_NAMES["success"] if all(predictions.values()) else
        TERMINAL_NAMES["transfer_fail"] if not pred_b or not pred_c or not pred_d
        else TERMINAL_NAMES["behavioral_fail"])
    result = {"schema": RESULT_SCHEMA,
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "capability": capability,
              "instrument": {"direct_writer_scored_logit_errors": direct_errors,
              "capture_identity_max_abs": max(item["identity_error"] for item in items.values()),
              "attention_reconstruction_max_abs": max(item["reconstruction_error"] for item in items.values())},
              "affine_coefficients": {reader: [float(v) for v in coefficients[reader]]
                                      for reader in ("block9", "block11")},
              "coefficient_correlations": correlations, "summaries": summaries,
              "fraction_of_full_response": fractions, "controls": controls,
              "predictions": predictions, "price": {"model_forwards": forwards,
              "example_evaluations": evaluations, "answer_changing_records": len(records),
              "fitted_scalars": 4, "stored_axis_coordinates": 512,
              "transformer_backwards": 0, "model_updates": 0},
              "records": records, "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability", "instrument",
          "coefficient_correlations", "fraction_of_full_response", "controls",
          "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
