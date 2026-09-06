#!/usr/bin/env python3
"""Predict temporal reader dose from writer state and base attention gain."""

# BQGATE: EXPERIMENT pred_a_authority_capability_closure pred_b_reader_gain_predicts_coefficients pred_c_reader_gain_program_material pred_d_reader_gain_beats_writer_only pred_e_p_selective pred_f_price_exact
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
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_reader_gain_compiler_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_scale_invariant_writer_compiler_v1_result.json"
MANIFEST = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_capability_manifest_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_reader_gain_compiler_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.reader_gain_compiler_v1"
EXPECTED = {"prior": "9875cec1a7bac2ec5fcc51aad16f97f2abe36d2b463839e569f1d6bceec4d122",
            "parent_result": "d115547e9026c8f39217a74632d76244d5ec1082f07dd64e2a7455f6b92fc7f4",
            "manifest": "d59fdc0659f7db4632607a5fae860887bdf0f69f03af659b02ffcd6cc8c3be59"}
ARMS = ("full_response", "oracle_rank1", "reader_gain_predicted", "writer_only_predicted", "zero")
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
    paths = {"prior": PRIOR, "parent_result": PARENT_RESULT, "manifest": MANIFEST}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    prior, parent_result, manifest = [json.loads(path.read_text()) for path in (PRIOR, PARENT_RESULT, MANIFEST)]
    original_rows, fresh_rows = original.build_rows(), fresh.build_rows()
    allowed = manifest["jointly_capable_row_ids"]
    splits = {"fit_a1": balanced_family(original_rows, "A1"),
              "fit_a2": balanced_family(original_rows, "A2"),
              "test_a1": [row for row in fresh_rows if row["row_id"] in allowed["A1"]],
              "test_a2": [row for row in fresh_rows if row["row_id"] in allowed["A2"]],
              "test_p": [row for row in fresh_rows if row["transform_id"] == "P"]}
    if (prior.get("candidate_id") != CANDIDATE_ID or parent_result.get("terminal") != "writer_only_insufficient"
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
            "answer_changing_records": RECORDS, "fitted_scalars_total": 11,
            "target_fitted_scalars": 7, "stored_axis_coordinates": 512,
            "transformer_backwards": 0, "model_updates": 0}


def reader_mass(item, layer, heads):
    capture = item[f"base{layer}"]
    values = []
    for index, query in enumerate(item["base_batch"].semantic_positions):
        positions = item["destinations"][index]
        values.append([float(capture["pattern"][index, head, int(query), list(positions)].sum())
                       for head in heads])
    return torch.tensor(values, device=capture["pattern"].device)


def design(s, masses, reader):
    ones = torch.ones_like(s)
    if reader == "block9":
        return torch.stack((s, s * masses[:, 0], s * masses[:, 1], ones), dim=1)
    return torch.stack((s, s * masses[:, 0], ones), dim=1)


def fit(design_matrix, target):
    return torch.linalg.lstsq(design_matrix, target[:, None]).solution[:, 0]


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
                       "block11": parent.reader_matrix(item, 11, (3,)),
                       "mass9": reader_mass(item, 9, (1, 4)),
                       "mass11": reader_mass(item, 11, (3,))}
                for name, item in items.items()}
    direct_errors = {}
    for name, item in items.items():
        direct = parent.run_direct_writer(backend, item, matrices[name]["terms"])
        direct_errors[name] = parent.pair_error(direct, item["writer_output"])
        forwards += 1
        evaluations += len(item["rows"])
    fit_names = ("fit_a1", "fit_a2")
    fit_writer = torch.cat(tuple(matrices[name]["terms"].sum(dim=1) for name in fit_names))
    writer_axis = parent.first_axis(fit_writer)
    reader_axes = {reader: parent.first_axis(torch.cat(tuple(matrices[name][reader] for name in fit_names)))
                   for reader in ("block9", "block11")}
    scalars, designs = {}, {}
    for name, values in matrices.items():
        writer = values["terms"].sum(dim=1)
        s = (writer @ writer_axis) / writer.norm(dim=1).clamp_min(1e-8)
        designs[name] = {"block9": design(s, values["mass9"], "block9"),
                         "block11": design(s, values["mass11"], "block11"),
                         "writer_only": torch.stack((s, torch.ones_like(s)), dim=1)}
    gain_coefficients, writer_coefficients = {}, {}
    for reader in ("block9", "block11"):
        target = torch.cat(tuple(matrices[name][reader] @ reader_axes[reader] for name in fit_names))
        gain_coefficients[reader] = fit(torch.cat(tuple(designs[name][reader] for name in fit_names)), target)
        writer_coefficients[reader] = fit(torch.cat(tuple(designs[name]["writer_only"] for name in fit_names)), target)
    predicted, correlations = {}, {}
    for name in ("test_a1", "test_a2", "test_p"):
        predicted[name] = {"gain": {}, "writer": {}}
        for reader in ("block9", "block11"):
            predicted[name]["gain"][reader] = designs[name][reader] @ gain_coefficients[reader]
            predicted[name]["writer"][reader] = designs[name]["writer_only"] @ writer_coefficients[reader]
        if name != "test_p":
            correlations[name] = {reader: parent.correlation(
                predicted[name]["gain"][reader], matrices[name][reader] @ reader_axes[reader])
                for reader in ("block9", "block11")}

    records, summaries = [], {}
    for name in ("test_a1", "test_a2"):
        item = items[name]
        full = {reader: matrices[name][reader] for reader in ("block9", "block11")}
        oracle = {reader: (full[reader] @ reader_axes[reader])[:, None] * reader_axes[reader][None, :]
                  for reader in ("block9", "block11")}
        gain = {reader: predicted[name]["gain"][reader][:, None] * reader_axes[reader][None, :]
                for reader in ("block9", "block11")}
        writer = {reader: predicted[name]["writer"][reader][:, None] * reader_axes[reader][None, :]
                  for reader in ("block9", "block11")}
        vectors = {"full_response": full, "oracle_rank1": oracle,
                   "reader_gain_predicted": gain, "writer_only_predicted": writer}
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
                       for model in ("gain", "writer")}
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
    pred_b = all(abs(correlations[name][reader]) >= 0.70
                 for name in correlations for reader in correlations[name])
    pred_c = all(fractions[name]["reader_gain_predicted"] >= 0.65
                 and summaries[name]["reader_gain_predicted"]["direction_fraction"] >= 0.75
                 for name in ("test_a1", "test_a2"))
    pred_d = bool(fractions["test_a2"]["reader_gain_predicted"] >= 0.60
                  and fractions["test_a2"]["reader_gain_predicted"]
                  - fractions["test_a2"]["writer_only_predicted"] >= 0.15)
    pred_e = controls["gain_predicted"] <= 0.20
    pred_f = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["split"], record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {"pred_a_authority_capability_closure": pred_a,
                   "pred_b_reader_gain_predicts_coefficients": pred_b,
                   "pred_c_reader_gain_program_material": pred_c,
                   "pred_d_reader_gain_beats_writer_only": pred_d,
                   "pred_e_p_selective": pred_e, "pred_f_price_exact": pred_f}
    terminal = "invalid" if not pred_a or not pred_f else (
        "reader_gain_compiler" if all(predictions.values()) else
        "base_gain_insufficient" if not pred_c or not pred_d else "null")
    result = {"schema": "temporal_auxiliary_reader_gain_compiler_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "capability": capability,
              "instrument": {"direct_writer_scored_logit_errors": direct_errors,
              "capture_identity_max_abs": max(item["identity_error"] for item in items.values()),
              "attention_reconstruction_max_abs": max(item["reconstruction_error"] for item in items.values())},
              "gain_coefficients": {reader: [float(v) for v in gain_coefficients[reader]]
                                    for reader in ("block9", "block11")},
              "coefficient_correlations": correlations, "summaries": summaries,
              "fraction_of_full_response": fractions, "controls": controls,
              "predictions": predictions, "price": {"model_forwards": forwards,
              "example_evaluations": evaluations, "answer_changing_records": len(records),
              "fitted_scalars_total": 11, "target_fitted_scalars": 7,
              "stored_axis_coordinates": 512, "transformer_backwards": 0, "model_updates": 0},
              "records": records, "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability", "instrument",
          "coefficient_correlations", "fraction_of_full_response", "controls",
          "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
