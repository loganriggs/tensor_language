#!/usr/bin/env python3
"""Test destination-resolved writer features against a summed writer baseline."""

# BQGATE: EXPERIMENT pred_a_exact_writer_interface pred_b_destination_magnitude_prediction pred_c_destination_resolution_beats_sum pred_d_destination_program_material pred_e_aligned_p_selective pred_f_price_exact
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

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import attention_source_group_eval as source_groups
import run_temporal_auxiliary_will_had_fresh_writer_to_reader_coefficients_v3 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_destination_resolved_coefficients_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_to_reader_coefficients_v3_result.json"
RANK_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_two_reader_response_rank_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_fresh_writer_to_reader_coefficients_v3.py"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_destination_resolved_coefficients_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_destination_resolved_coefficients_v1"
EXPECTED = {
    "prior": "cdd023e933021744d765b321fec0935037d44378e69cc9f748a7e7133c5721b9",
    "parent_result": "1049e7be61f5bad58aa4d15ad18d836e89d27c0336efc0dfe12a8e8c9904ab57",
    "rank_result": "918888d78de2c10a24c57e80154f07fd4d9dd7ed5faf181385e1d4b22e587bcf",
    "parent_runner": "ce7822a6a0ae41f330b478663ddc8b1a48f0ce0314609cfcad0c8bc35fbf24ab",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
}
ARMS = ("full_captured_response", "oracle_rank1_response", "destination_predicted_response",
        "summed_predicted_response", "intercept_only_response", "zero_response")
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 41, 1008, 288


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "parent_result": PARENT_RESULT, "rank_result": RANK_RESULT,
             "parent_runner": PARENT_RUNNER, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or helper hash changed")
    prior, parent_result, rank_result = [json.loads(path.read_text())
                                         for path in (PRIOR, PARENT_RESULT, RANK_RESULT)]
    rows = candidate.build_rows()
    family_rows = {family: [row for row in rows if row["transform_id"] == family]
                   for family in ("A1", "A2", "P")}
    a1 = family_rows["A1"]
    fit = [row for index, row in enumerate(a1) if (index // 2) % 2 == 0]
    heldout = [row for index, row in enumerate(a1) if (index // 2) % 2 == 1]
    direction_counts = {split: {direction: sum(row["direction_id"] == direction for row in selected)
                                for direction in ("future_to_anterior", "anterior_to_future")}
                        for split, selected in (("fit", fit), ("heldout", heldout))}
    if (prior.get("candidate_id") != CANDIDATE_ID or parent_result.get("terminal") != "wrong_predictor"
            or rank_result.get("terminal") != "screen"
            or direction_counts != {"fit": {"future_to_anterior": 8, "anterior_to_future": 8},
                                    "heldout": {"future_to_anterior": 8, "anterior_to_future": 8}}
            or len(family_rows["A2"]) != 32 or len(family_rows["P"]) != 32
            or any(len(row["base_ids"]) != len(row["donor_ids"])
                   for selected in (fit, heldout, family_rows["A2"], family_rows["P"])
                   for row in selected)):
        raise ExperimentError("authority, balanced split, or alignment changed")
    return {"fit": fit, "heldout": heldout, "a2": family_rows["A2"], "p": family_rows["P"]}


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "answer_arms": list(ARMS),
            "model_forwards": MODEL_FORWARDS, "example_evaluations": EXAMPLE_EVALUATIONS,
            "answer_changing_records": RECORDS, "fitted_scalars_total": 10,
            "target_fitted_scalars": 6, "stored_axis_coordinates": 768,
            "transformer_backwards": 0, "model_updates": 0}


def affine_fit(features, target):
    design = torch.cat((features, torch.ones(features.shape[0], 1, device=features.device)), dim=1)
    return torch.linalg.lstsq(design, target[:, None]).solution[:, 0]


def affine_predict(features, coefficients):
    return features @ coefficients[:-1] + coefficients[-1]


def feature_matrices(terms, destination_axes, sum_axis):
    destination = torch.stack(tuple(terms[:, index] @ destination_axes[index] for index in range(2)), dim=1)
    summed = (terms.sum(dim=1) @ sum_axis)[:, None]
    return destination, summed


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

    destination_axes = tuple(parent.first_axis(matrices["fit"]["terms"][:, index]) for index in range(2))
    sum_axis = parent.first_axis(matrices["fit"]["terms"].sum(dim=1))
    reader_axes = {reader: parent.first_axis(matrices["fit"][reader])
                   for reader in ("block9", "block11")}
    features = {name: feature_matrices(values["terms"], destination_axes, sum_axis)
                for name, values in matrices.items()}
    target_coefficients, summed_coefficients, intercepts = {}, {}, {}
    for reader in ("block9", "block11"):
        target = matrices["fit"][reader] @ reader_axes[reader]
        target_coefficients[reader] = affine_fit(features["fit"][0], target)
        summed_coefficients[reader] = affine_fit(features["fit"][1], target)
        intercepts[reader] = float(target.mean())

    predicted, correlations = {}, {}
    for name in ("heldout", "a2", "p"):
        predicted[name] = {"destination": {}, "summed": {}}
        correlations[name] = {}
        for reader in ("block9", "block11"):
            actual = matrices[name][reader] @ reader_axes[reader]
            destination = affine_predict(features[name][0], target_coefficients[reader])
            summed = affine_predict(features[name][1], summed_coefficients[reader])
            predicted[name]["destination"][reader] = destination
            predicted[name]["summed"][reader] = summed
            if name != "p":
                correlations[name][reader] = {
                    "destination": parent.correlation(destination, actual),
                    "summed": parent.correlation(summed, actual)}

    records = []
    summaries = {}
    for name in ("heldout", "a2"):
        item = items[name]
        full = {reader: matrices[name][reader] for reader in ("block9", "block11")}
        oracle = {reader: (full[reader] @ reader_axes[reader])[:, None] * reader_axes[reader][None, :]
                  for reader in ("block9", "block11")}
        destination = {reader: predicted[name]["destination"][reader][:, None] * reader_axes[reader][None, :]
                       for reader in ("block9", "block11")}
        summed = {reader: predicted[name]["summed"][reader][:, None] * reader_axes[reader][None, :]
                  for reader in ("block9", "block11")}
        intercept = {reader: torch.full_like(predicted[name]["destination"][reader], intercepts[reader])[:, None]
                     * reader_axes[reader][None, :] for reader in ("block9", "block11")}
        vectors = {"full_captured_response": full, "oracle_rank1_response": oracle,
                   "destination_predicted_response": destination, "summed_predicted_response": summed,
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

    item = items["p"]
    control_vectors = {}
    for model in ("destination", "summed"):
        control_vectors[f"{model}_predicted_response"] = {
            reader: predicted["p"][model][reader][:, None] * reader_axes[reader][None, :]
            for reader in ("block9", "block11")}
    control_vectors["intercept_only_response"] = {
        reader: torch.full_like(predicted["p"]["destination"][reader], intercepts[reader])[:, None]
        * reader_axes[reader][None, :] for reader in ("block9", "block11")}
    control_outputs = {arm: parent.run_responses(backend, item, value["block9"], value["block11"])
                       for arm, value in control_vectors.items()}
    forwards += len(control_vectors)
    evaluations += len(control_vectors) * len(item["rows"])
    target_scale = statistics.median(abs(donor - base)
        for name in ("heldout", "a2")
        for base, donor in zip(parent.axis_values(items[name]["base_output"]),
                               parent.axis_values(items[name]["donor_output"])))
    controls = {arm: statistics.mean(abs(value - base) / target_scale
        for value, base in zip(parent.axis_values(output), parent.axis_values(item["base_output"])))
        for arm, output in control_outputs.items()}
    fractions = {name: {arm: summaries[name][arm]["mean_recovery"]
                        / summaries[name]["full_captured_response"]["mean_recovery"]
                        for arm in ARMS[1:]}
                 for name in ("heldout", "a2")}
    destination_mean = statistics.mean(abs(correlations["heldout"][reader]["destination"])
                                       for reader in ("block9", "block11"))
    summed_mean = statistics.mean(abs(correlations["heldout"][reader]["summed"])
                                  for reader in ("block9", "block11"))
    pred_a = bool(max(direct_errors.values()) <= 1e-4
                  and max(item["identity_error"] for item in items.values()) <= 1e-4
                  and max(item["reconstruction_error"] for item in items.values()) <= 5e-4)
    pred_b = all(abs(correlations[name][reader]["destination"]) >= 0.60
                 for name in ("heldout", "a2") for reader in ("block9", "block11"))
    pred_c = destination_mean - summed_mean >= 0.15
    pred_d = all(fractions[name]["destination_predicted_response"] >= 0.65
                 and summaries[name]["destination_predicted_response"]["direction_fraction"] >= 0.75
                 for name in ("heldout", "a2"))
    pred_e = controls["destination_predicted_response"] <= 0.20
    pred_f = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["split"], record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records)
                  and all(math.isfinite(float(correlations[name][reader][model]))
                          for name in ("heldout", "a2")
                          for reader in ("block9", "block11")
                          for model in ("destination", "summed")))
    predictions = {"pred_a_exact_writer_interface": pred_a,
                   "pred_b_destination_magnitude_prediction": pred_b,
                   "pred_c_destination_resolution_beats_sum": pred_c,
                   "pred_d_destination_program_material": pred_d,
                   "pred_e_aligned_p_selective": pred_e, "pred_f_price_exact": pred_f}
    terminal = "invalid" if not pred_a or not pred_f else (
        "screen" if all(predictions.values()) else
        "writer_summary_insufficient" if not pred_b or not pred_c else "behavioral_null")
    result = {"schema": "temporal_auxiliary_fresh_destination_resolved_coefficients_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "instrument": {"direct_writer_scored_logit_errors": direct_errors,
              "capture_identity_max_abs": max(item["identity_error"] for item in items.values()),
              "attention_reconstruction_max_abs": max(item["reconstruction_error"] for item in items.values())},
              "target_affine_coefficients": {reader: [float(v) for v in target_coefficients[reader]]
                                             for reader in ("block9", "block11")},
              "summed_affine_coefficients": {reader: [float(v) for v in summed_coefficients[reader]]
                                             for reader in ("block9", "block11")},
              "coefficient_correlations": correlations, "heldout_mean_correlation": {
              "destination": destination_mean, "summed": summed_mean,
              "improvement": destination_mean - summed_mean}, "summaries": summaries,
              "fraction_of_full_response": fractions, "controls": controls,
              "predictions": predictions, "price": {"model_forwards": forwards,
              "example_evaluations": evaluations, "answer_changing_records": len(records),
              "fitted_scalars_total": 10, "target_fitted_scalars": 6,
              "stored_axis_coordinates": 768, "transformer_backwards": 0, "model_updates": 0},
              "records": records, "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
          "coefficient_correlations", "heldout_mean_correlation", "fraction_of_full_response",
          "controls", "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
