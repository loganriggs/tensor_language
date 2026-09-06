#!/usr/bin/env python3
"""Typed A1/A2 coefficient calibration for the broad temporal reader mode."""

# BQGATE: EXPERIMENT pred_a_authority_capability_closure pred_b_frame_coefficients_predict pred_c_frame_program_material pred_d_a2_repairs_a1_only pred_e_p_selective pred_f_price_exact
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
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fresh
import circuit_fast_screen_candidate_temporal_auxiliary as original
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_auxiliary_will_had_fresh_writer_to_reader_coefficients_v3 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_frame_conditioned_compiler_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_broad_mode_reverse_compiler_v2_result.json"
MANIFEST = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_capability_manifest_v1_result.json"
FRESH_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
ORIGINAL_BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_frame_conditioned_compiler_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.frame_conditioned_compiler_v1"
EXPECTED = {
    "prior": "7b08e4bb23256d5a5cc40e066c32ad71e9833f655d2418419a8d9224a8b3dfb7",
    "parent_result": "dbf49804d98660255fcae5b590c4ce46e5cf5067bf04af054fa2a2cfcd36fcec",
    "manifest": "d59fdc0659f7db4632607a5fae860887bdf0f69f03af659b02ffcd6cc8c3be59",
    "fresh_builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "original_builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
}
ARMS = ("full_captured_response", "oracle_rank1_response", "predicted_rank1_response",
        "intercept_only_response", "zero_response")
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 45, 1161, 295


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def balanced_family(rows, family):
    selected = [row for row in rows if row["transform_id"] == family]
    return [row for index, row in enumerate(selected) if (index // 2) % 2 == 0]


def validate_static():
    paths = {"prior": PRIOR, "parent_result": PARENT_RESULT, "manifest": MANIFEST,
             "fresh_builder": FRESH_BUILDER, "original_builder": ORIGINAL_BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    prior = json.loads(PRIOR.read_text())
    parent_result = json.loads(PARENT_RESULT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    original_rows, fresh_rows = original.build_rows(), fresh.build_rows()
    fit_a1 = balanced_family(original_rows, "A1")
    fit_a2 = balanced_family(original_rows, "A2")
    allowed = manifest["jointly_capable_row_ids"]
    test_a1 = [row for row in fresh_rows if row["row_id"] in allowed["A1"]]
    test_a2 = [row for row in fresh_rows if row["row_id"] in allowed["A2"]]
    test_p = [row for row in fresh_rows if row["transform_id"] == "P"]
    splits = {"fit_a1": fit_a1, "fit_a2": fit_a2,
              "test_a1": test_a1, "test_a2": test_a2, "test_p": test_p}
    direction_counts = {name: {direction: sum(row["direction_id"] == direction for row in rows)
                               for direction in ("future_to_anterior", "anterior_to_future")}
                        for name, rows in splits.items() if name.startswith("fit")}
    if (prior.get("candidate_id") != CANDIDATE_ID or parent_result.get("terminal") != "coefficient_asymmetry"
            or direction_counts != {"fit_a1": {"future_to_anterior": 8, "anterior_to_future": 8},
                                    "fit_a2": {"future_to_anterior": 8, "anterior_to_future": 8}}
            or {name: len(rows) for name, rows in splits.items()}
            != {"fit_a1": 16, "fit_a2": 16, "test_a1": 29, "test_a2": 30, "test_p": 32}
            or any(len(row["base_ids"]) != len(row["donor_ids"])
                   for rows in splits.values() for row in rows)):
        raise ExperimentError("typed fit or capable test population changed")
    return splits


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "arms": list(ARMS),
            "model_forwards": MODEL_FORWARDS, "example_evaluations": EXAMPLE_EVALUATIONS,
            "answer_changing_records": RECORDS, "fitted_scalars": 8,
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

    joint_writer = torch.cat(tuple(matrices[name]["terms"].sum(dim=1)
                                   for name in ("fit_a1", "fit_a2")), dim=0)
    writer_axis = parent.first_axis(joint_writer)
    reader_axes = {reader: parent.first_axis(torch.cat(tuple(
        matrices[name][reader] for name in ("fit_a1", "fit_a2")), dim=0))
        for reader in ("block9", "block11")}
    coefficients, intercepts = {}, {}
    for frame in ("a1", "a2"):
        name = f"fit_{frame}"
        x = matrices[name]["terms"].sum(dim=1) @ writer_axis
        coefficients[frame] = {reader: parent.affine_fit(
            x, matrices[name][reader] @ reader_axes[reader]) for reader in ("block9", "block11")}
        intercepts[frame] = {reader: float((matrices[name][reader] @ reader_axes[reader]).mean())
                             for reader in ("block9", "block11")}

    predicted, correlations = {}, {}
    for frame in ("a1", "a2"):
        name = f"test_{frame}"
        x = matrices[name]["terms"].sum(dim=1) @ writer_axis
        predicted[name] = {reader: coefficients[frame][reader][0] * x + coefficients[frame][reader][1]
                           for reader in ("block9", "block11")}
        correlations[name] = {reader: parent.correlation(
            predicted[name][reader], matrices[name][reader] @ reader_axes[reader])
            for reader in ("block9", "block11")}
    x_p = matrices["test_p"]["terms"].sum(dim=1) @ writer_axis
    predicted["test_p"] = {reader: coefficients["a1"][reader][0] * x_p + coefficients["a1"][reader][1]
                           for reader in ("block9", "block11")}

    records, summaries = [], {}
    for frame in ("a1", "a2"):
        name = f"test_{frame}"
        item = items[name]
        full = {reader: matrices[name][reader] for reader in ("block9", "block11")}
        oracle = {reader: (full[reader] @ reader_axes[reader])[:, None] * reader_axes[reader][None, :]
                  for reader in ("block9", "block11")}
        predicted_vectors = {reader: predicted[name][reader][:, None] * reader_axes[reader][None, :]
                             for reader in ("block9", "block11")}
        intercept = {reader: torch.full_like(predicted[name][reader], intercepts[frame][reader])[:, None]
                     * reader_axes[reader][None, :] for reader in ("block9", "block11")}
        vectors = {"full_captured_response": full, "oracle_rank1_response": oracle,
                   "predicted_rank1_response": predicted_vectors, "intercept_only_response": intercept}
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

    item = items["test_p"]
    predicted_vectors = {reader: predicted["test_p"][reader][:, None] * reader_axes[reader][None, :]
                         for reader in ("block9", "block11")}
    intercept = {reader: torch.full_like(predicted["test_p"][reader], intercepts["a1"][reader])[:, None]
                 * reader_axes[reader][None, :] for reader in ("block9", "block11")}
    control_outputs = {"predicted_rank1_response": parent.run_responses(
                           backend, item, predicted_vectors["block9"], predicted_vectors["block11"]),
                       "intercept_only_response": parent.run_responses(
                           backend, item, intercept["block9"], intercept["block11"])}
    forwards += 2
    evaluations += 2 * len(item["rows"])
    fractions = {name: {arm: values[arm]["mean_recovery"]
                        / values["full_captured_response"]["mean_recovery"] for arm in ARMS[1:]}
                 for name, values in summaries.items()}
    target_scale = statistics.median(abs(donor - base)
        for name in ("test_a1", "test_a2")
        for base, donor in zip(parent.axis_values(items[name]["base_output"]),
                               parent.axis_values(items[name]["donor_output"])))
    controls = {arm: statistics.mean(abs(value - base) / target_scale
        for value, base in zip(parent.axis_values(output), parent.axis_values(item["base_output"])))
        for arm, output in control_outputs.items()}
    capability = {name: {"base_correct": sum(answer > foil for answer, foil in item["base_output"].answer_foil),
                         "donor_correct": sum(answer > foil for answer, foil in item["donor_output"].answer_foil),
                         "total": len(item["rows"])} for name, item in items.items() if name != "test_p"}
    pred_a = bool(max(direct_errors.values()) <= 1e-4
                  and max(item["identity_error"] for item in items.values()) <= 1e-4
                  and max(item["reconstruction_error"] for item in items.values()) <= 5e-4
                  and all(value["base_correct"] == value["total"]
                          and value["donor_correct"] == value["total"] for value in capability.values()))
    pred_b = all(abs(correlations[name][reader]) >= 0.60
                 for name in correlations for reader in correlations[name])
    pred_c = all(fractions[name]["predicted_rank1_response"] >= 0.65
                 and summaries[name]["predicted_rank1_response"]["direction_fraction"] >= 0.75
                 for name in ("test_a1", "test_a2"))
    pred_d = bool(fractions["test_a2"]["predicted_rank1_response"] >= 0.60
                  and fractions["test_a2"]["predicted_rank1_response"] - 0.4892859472904869 >= 0.10)
    pred_e = controls["predicted_rank1_response"] <= 0.20
    pred_f = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["split"], record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {"pred_a_authority_capability_closure": pred_a,
                   "pred_b_frame_coefficients_predict": pred_b,
                   "pred_c_frame_program_material": pred_c,
                   "pred_d_a2_repairs_a1_only": pred_d,
                   "pred_e_p_selective": pred_e, "pred_f_price_exact": pred_f}
    terminal = "invalid" if not pred_a or not pred_f else (
        "typed_compiler" if all(predictions.values()) else
        "cue_gain_missing" if pred_b and not pred_c else
        "axis_mismatch" if any(fractions[name]["oracle_rank1_response"] < 0.65
                               for name in ("test_a1", "test_a2")) else "null")
    result = {"schema": "temporal_auxiliary_frame_conditioned_compiler_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "capability": capability,
              "instrument": {"direct_writer_scored_logit_errors": direct_errors,
              "capture_identity_max_abs": max(item["identity_error"] for item in items.values()),
              "attention_reconstruction_max_abs": max(item["reconstruction_error"] for item in items.values())},
              "frame_affine_coefficients": {frame: {reader: [float(v) for v in values]
              for reader, values in readers.items()} for frame, readers in coefficients.items()},
              "coefficient_correlations": correlations, "summaries": summaries,
              "fraction_of_full_response": fractions, "controls": controls,
              "predictions": predictions, "price": {"model_forwards": forwards,
              "example_evaluations": evaluations, "answer_changing_records": len(records),
              "fitted_scalars": 8, "stored_axis_coordinates": 512,
              "transformer_backwards": 0, "model_updates": 0},
              "records": records, "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability", "instrument",
          "coefficient_correlations", "fraction_of_full_response", "controls",
          "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
