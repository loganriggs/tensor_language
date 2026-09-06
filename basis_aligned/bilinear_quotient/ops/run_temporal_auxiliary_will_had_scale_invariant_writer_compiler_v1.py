#!/usr/bin/env python3
"""Final writer-only test: normalize writer projection by its per-row norm."""

# BQGATE: EXPERIMENT pred_a_authority_capability_closure pred_b_normalized_coefficients_predict pred_c_normalized_program_material pred_d_normalization_repairs_raw pred_e_p_selective pred_f_price_exact
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
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_scale_invariant_writer_compiler_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_frame_conditioned_compiler_v1_result.json"
MANIFEST = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_capability_manifest_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_scale_invariant_writer_compiler_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.scale_invariant_writer_compiler_v1"
EXPECTED = {
    "prior": "9d0a684c9ee4fc0d8dc4e725b867fc624232d1bfeb39d181a547a437bd10f0dc",
    "parent_result": "61815c859ef5d8dfe4accf8f631555560919ba6f87553edb14c607fb7b42e54c",
    "manifest": "d59fdc0659f7db4632607a5fae860887bdf0f69f03af659b02ffcd6cc8c3be59",
}
ARMS = ("full_response", "oracle_rank1", "normalized_predicted", "raw_predicted",
        "intercept_only", "zero")
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 48, 1252, 354


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
    paths = {"prior": PRIOR, "parent_result": PARENT_RESULT, "manifest": MANIFEST}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    prior, parent_result, manifest = [json.loads(path.read_text())
                                      for path in (PRIOR, PARENT_RESULT, MANIFEST)]
    original_rows, fresh_rows = original.build_rows(), fresh.build_rows()
    allowed = manifest["jointly_capable_row_ids"]
    splits = {
        "fit_a1": balanced_family(original_rows, "A1"),
        "fit_a2": balanced_family(original_rows, "A2"),
        "test_a1": [row for row in fresh_rows if row["row_id"] in allowed["A1"]],
        "test_a2": [row for row in fresh_rows if row["row_id"] in allowed["A2"]],
        "test_p": [row for row in fresh_rows if row["transform_id"] == "P"],
    }
    if (prior.get("candidate_id") != CANDIDATE_ID or parent_result.get("terminal") != "cue_gain_missing"
            or {name: len(rows) for name, rows in splits.items()}
            != {"fit_a1": 16, "fit_a2": 16, "test_a1": 29, "test_a2": 30, "test_p": 32}
            or any(len(row["base_ids"]) != len(row["donor_ids"])
                   for rows in splits.values() for row in rows)):
        raise ExperimentError("fit or capable test population changed")
    return splits


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "arms": list(ARMS),
            "model_forwards": MODEL_FORWARDS, "example_evaluations": EXAMPLE_EVALUATIONS,
            "answer_changing_records": RECORDS, "fitted_scalars_total": 16,
            "stored_axis_coordinates": 512, "transformer_backwards": 0, "model_updates": 0}


def writer_features(matrix, axis):
    raw = matrix @ axis
    norms = matrix.norm(dim=1).clamp_min(1e-8)
    return {"raw": raw, "normalized": raw / norms}


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
    matrices = {name: {"writer": parent.writer_terms(item).sum(dim=1),
                       "block9": parent.reader_matrix(item, 9, (1, 4)),
                       "block11": parent.reader_matrix(item, 11, (3,))}
                for name, item in items.items()}
    direct_errors = {}
    for name, item in items.items():
        terms = parent.writer_terms(item)
        direct = parent.run_direct_writer(backend, item, terms)
        direct_errors[name] = parent.pair_error(direct, item["writer_output"])
        forwards += 1
        evaluations += len(item["rows"])

    fit_names = ("fit_a1", "fit_a2")
    writer_axis = parent.first_axis(torch.cat(tuple(matrices[name]["writer"] for name in fit_names)))
    reader_axes = {reader: parent.first_axis(torch.cat(tuple(matrices[name][reader] for name in fit_names)))
                   for reader in ("block9", "block11")}
    features = {name: writer_features(values["writer"], writer_axis) for name, values in matrices.items()}
    coefficients = {model: {} for model in ("raw", "normalized")}
    intercepts = {}
    for frame in ("a1", "a2"):
        fit_name = f"fit_{frame}"
        intercepts[frame] = {}
        for reader in ("block9", "block11"):
            target = matrices[fit_name][reader] @ reader_axes[reader]
            intercepts[frame][reader] = float(target.mean())
            for model in ("raw", "normalized"):
                coefficients[model].setdefault(frame, {})[reader] = parent.affine_fit(
                    features[fit_name][model], target)

    predicted, correlations = {}, {}
    for frame in ("a1", "a2"):
        name = f"test_{frame}"
        predicted[name], correlations[name] = {}, {}
        for model in ("raw", "normalized"):
            predicted[name][model], correlations[name][model] = {}, {}
            for reader in ("block9", "block11"):
                values = (coefficients[model][frame][reader][0] * features[name][model]
                          + coefficients[model][frame][reader][1])
                predicted[name][model][reader] = values
                correlations[name][model][reader] = parent.correlation(
                    values, matrices[name][reader] @ reader_axes[reader])
    predicted["test_p"] = {}
    for model in ("raw", "normalized"):
        predicted["test_p"][model] = {reader:
            coefficients[model]["a1"][reader][0] * features["test_p"][model]
            + coefficients[model]["a1"][reader][1] for reader in ("block9", "block11")}

    records, summaries = [], {}
    for frame in ("a1", "a2"):
        name, item = f"test_{frame}", items[f"test_{frame}"]
        full = {reader: matrices[name][reader] for reader in ("block9", "block11")}
        oracle = {reader: (full[reader] @ reader_axes[reader])[:, None] * reader_axes[reader][None, :]
                  for reader in ("block9", "block11")}
        projected = {model: {reader: predicted[name][model][reader][:, None] * reader_axes[reader][None, :]
                             for reader in ("block9", "block11")} for model in ("raw", "normalized")}
        intercept = {reader: torch.full_like(predicted[name]["raw"][reader], intercepts[frame][reader])[:, None]
                     * reader_axes[reader][None, :] for reader in ("block9", "block11")}
        vectors = {"full_response": full, "oracle_rank1": oracle,
                   "normalized_predicted": projected["normalized"], "raw_predicted": projected["raw"],
                   "intercept_only": intercept}
        outputs = {arm: parent.run_responses(backend, item, value["block9"], value["block11"])
                   for arm, value in vectors.items()}
        outputs["zero"] = item["base_output"]
        forwards += len(vectors)
        evaluations += len(vectors) * len(item["rows"])
        for arm in ARMS:
            for record in source_groups.recovery_records(
                    item["rows"], item["base_output"], item["donor_output"], outputs[arm], arm=arm):
                record["split"] = name
                records.append(record)
        summaries[name] = {arm: source_groups.summarize([
            record for record in records if record["split"] == name and record["arm"] == arm]) for arm in ARMS}

    item = items["test_p"]
    control_vectors = {f"{model}_predicted": {reader: predicted["test_p"][model][reader][:, None]
                       * reader_axes[reader][None, :] for reader in ("block9", "block11")}
                       for model in ("raw", "normalized")}
    control_vectors["intercept_only"] = {reader: torch.full_like(
        predicted["test_p"]["raw"][reader], intercepts["a1"][reader])[:, None]
        * reader_axes[reader][None, :] for reader in ("block9", "block11")}
    control_outputs = {arm: parent.run_responses(backend, item, value["block9"], value["block11"])
                       for arm, value in control_vectors.items()}
    forwards += len(control_vectors)
    evaluations += len(control_vectors) * len(item["rows"])
    fractions = {name: {arm: values[arm]["mean_recovery"] / values["full_response"]["mean_recovery"]
                        for arm in ARMS[1:]} for name, values in summaries.items()}
    target_scale = statistics.median(abs(donor - base)
        for name in ("test_a1", "test_a2")
        for base, donor in zip(parent.axis_values(items[name]["base_output"]),
                               parent.axis_values(items[name]["donor_output"])))
    controls = {arm: statistics.mean(abs(value - base) / target_scale
        for value, base in zip(parent.axis_values(output), parent.axis_values(item["base_output"])))
        for arm, output in control_outputs.items()}
    capability = {name: {"base": sum(a > f for a, f in item["base_output"].answer_foil),
                         "donor": sum(a > f for a, f in item["donor_output"].answer_foil),
                         "total": len(item["rows"])} for name, item in items.items() if name != "test_p"}
    pred_a = bool(max(direct_errors.values()) <= 1e-4
                  and max(item["identity_error"] for item in items.values()) <= 1e-4
                  and max(item["reconstruction_error"] for item in items.values()) <= 5e-4
                  and all(v["base"] == v["total"] and v["donor"] == v["total"] for v in capability.values()))
    pred_b = all(abs(correlations[name]["normalized"][reader]) >= 0.60
                 for name in correlations for reader in ("block9", "block11"))
    pred_c = all(fractions[name]["normalized_predicted"] >= 0.65
                 and summaries[name]["normalized_predicted"]["direction_fraction"] >= 0.75
                 for name in ("test_a1", "test_a2"))
    pred_d = bool(fractions["test_a2"]["normalized_predicted"] >= 0.60
                  and fractions["test_a2"]["normalized_predicted"]
                  - fractions["test_a2"]["raw_predicted"] >= 0.10)
    pred_e = controls["normalized_predicted"] <= 0.20
    pred_f = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["split"], record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {"pred_a_authority_capability_closure": pred_a,
                   "pred_b_normalized_coefficients_predict": pred_b,
                   "pred_c_normalized_program_material": pred_c,
                   "pred_d_normalization_repairs_raw": pred_d,
                   "pred_e_p_selective": pred_e, "pred_f_price_exact": pred_f}
    terminal = "invalid" if not pred_a or not pred_f else (
        "scale_invariant_compiler" if all(predictions.values()) else
        "writer_only_insufficient" if not pred_c or not pred_d else "null")
    result = {"schema": "temporal_auxiliary_scale_invariant_writer_compiler_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "capability": capability,
              "instrument": {"direct_writer_scored_logit_errors": direct_errors,
              "capture_identity_max_abs": max(item["identity_error"] for item in items.values()),
              "attention_reconstruction_max_abs": max(item["reconstruction_error"] for item in items.values())},
              "coefficient_correlations": correlations, "summaries": summaries,
              "fraction_of_full_response": fractions, "controls": controls,
              "predictions": predictions, "price": {"model_forwards": forwards,
              "example_evaluations": evaluations, "answer_changing_records": len(records),
              "fitted_scalars_total": 16, "stored_axis_coordinates": 512,
              "transformer_backwards": 0, "model_updates": 0},
              "records": records, "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability", "instrument",
          "coefficient_correlations", "fraction_of_full_response", "controls",
          "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
