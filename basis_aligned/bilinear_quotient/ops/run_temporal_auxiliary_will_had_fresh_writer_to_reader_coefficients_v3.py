#!/usr/bin/env python3
"""Predict two low-rank reader responses from the exact block8H1 writer vector."""

# BQGATE: EXPERIMENT pred_a_exact_writer_vector pred_b_coefficient_prediction pred_c_predicted_program_material pred_d_prediction_beats_intercept pred_e_aligned_p_selective pred_f_price_exact
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

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as attention_eval
import attention_source_group_eval as source_groups
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_writer_to_reader_coefficients_v3.json"
RANK_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_two_reader_response_rank_v1_result.json"
PROGRAM_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_two_reader_response_program_v1_result.json"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_to_reader_coefficients_v3_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_writer_to_reader_coefficients_v3"
EXPECTED = {
    "prior": "9a4df529719ec615da54750749d063fd61df25e7214a466fee04203057180364",
    "rank_result": "918888d78de2c10a24c57e80154f07fd4d9dd7ed5faf181385e1d4b22e587bcf",
    "program_result": "3fe8a3e7edf8dc24f7977d0ce4a37f564cac35211b2a2043e76336ea33e023cb",
    "attention": "e948c1950ff3deac055cfb91a1ece9c417236580bcc131811730bf1fad4d9f9b",
    "mediation": "05d07578129db0cb8bf5e46d08e6cd6e9c2bb9c3883675586c83c3efd24b8fda",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
}
ANSWER_ARMS = ("full_captured_response", "oracle_rank1_response",
               "predicted_rank1_response", "intercept_only_response", "zero_response")
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 38, 928, 240


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def pair_error(first, second):
    return max(abs(float(a) - float(b)) for pa, pb in zip(first.answer_foil, second.answer_foil)
               for a, b in zip(pa, pb))


def axis_values(output):
    return [-(float(answer) - float(foil)) for answer, foil in output.answer_foil]


def validate_static():
    paths = {"prior": PRIOR, "rank_result": RANK_RESULT, "program_result": PROGRAM_RESULT,
             "attention": ATTENTION, "mediation": MEDIATION, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or implementation hash changed")
    prior, rank_result, program = [json.loads(path.read_text())
                                   for path in (PRIOR, RANK_RESULT, PROGRAM_RESULT)]
    rows = candidate.build_rows()
    family_rows = {family: [row for row in rows if row["transform_id"] == family]
                   for family in ("A1", "A2", "P")}
    if (prior.get("candidate_id") != CANDIDATE_ID or rank_result.get("terminal") != "screen"
            or program.get("terminal") != "screen"
            or {key: len(value) for key, value in family_rows.items()}
            != {"A1": 32, "A2": 32, "P": 32}
            or any(len(row["base_ids"]) != len(row["donor_ids"])
                   for family in family_rows.values() for row in family)):
        raise ExperimentError("population or response-program authority changed")
    return {"fit": family_rows["A1"][0::2], "heldout": family_rows["A1"][1::2],
            "a2": family_rows["A2"], "p": family_rows["P"]}


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "answer_arms": list(ANSWER_ARMS),
            "control_arms": ["predicted_rank1_response", "intercept_only_response"],
            "model_forwards": MODEL_FORWARDS, "example_evaluations": EXAMPLE_EVALUATIONS,
            "answer_changing_records": RECORDS, "fitted_scalars": 4,
            "stored_axis_coordinates": 512, "transformer_backwards": 0, "model_updates": 0}


def capture_with_writer(backend, item, layer):
    hook = mediation.fixed_source_delta_hook(
        backend, item["base_batch"], item["donor_batch"], item["writer_base"],
        item["writer_donor"], item["destinations"], ("cue",), selected_heads=(1,))
    handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(hook)
    try:
        return attention_eval.capture_layer_attention(backend, item["base_batch"], layer)
    finally:
        handle.remove()


def prepare_item(backend, rows):
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, writer_base = attention_eval.capture_layer_attention(backend, base_batch, 8)
    donor_output, writer_donor = attention_eval.capture_layer_attention(backend, donor_batch, 8)
    base9_output, base9 = attention_eval.capture_layer_attention(backend, base_batch, 9)
    base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
    item = {"rows": rows, "base_batch": base_batch, "donor_batch": donor_batch,
            "base_output": base_output, "donor_output": donor_output,
            "writer_base": writer_base, "writer_donor": writer_donor,
            "destinations": onset.positions_for_group(base_batch, donor_batch, "subject_onset"),
            "base9": base9, "base11": base11}
    writer_output9, item["changed9"] = capture_with_writer(backend, item, 9)
    writer_output11, item["changed11"] = capture_with_writer(backend, item, 11)
    item["writer_output"] = writer_output9
    item["identity_error"] = max(pair_error(base_output, base9_output),
                                 pair_error(base_output, base11_output),
                                 pair_error(writer_output9, writer_output11))
    item["reconstruction_error"] = max(float(capture["reconstruction_max_abs"])
        for capture in (writer_base, writer_donor, base9, base11, item["changed9"], item["changed11"]))
    return item


def reader_matrix(item, layer, heads):
    base, changed = item[f"base{layer}"], item[f"changed{layer}"]
    return torch.stack([torch.cat(tuple(
        changed["head_output"][index, int(query), head].float()
        - base["head_output"][index, int(query), head].float() for head in heads))
        for index, query in enumerate(item["base_batch"].semantic_positions)])


def writer_terms(item):
    partitions = attention_eval.batch_destination_partitions(
        item["base_batch"], item["donor_batch"], item["destinations"])
    rows = []
    for index, row_destinations in enumerate(item["destinations"]):
        destination_vectors = []
        for destination_index, destination in enumerate(row_destinations):
            cue_positions = partitions[index][destination_index]["cue"]
            delta = sum((
                item["writer_donor"]["pattern"][index, 1, destination, position]
                * item["writer_donor"]["value"][index, position, 1]
                - item["writer_base"]["pattern"][index, 1, destination, position]
                * item["writer_base"]["value"][index, position, 1]
                for position in cue_positions), start=torch.zeros_like(
                    item["writer_base"]["value"][index, 0, 1]))
            destination_vectors.append(delta.float())
        rows.append(torch.stack(destination_vectors))
    counts = {row.shape[0] for row in rows}
    if counts != {2}:
        raise ExperimentError(f"subject-onset destination count changed: {counts}")
    return torch.stack(rows)


def run_direct_writer(backend, item, terms):
    def patch(_module, arguments):
        flattened = arguments[0]
        changed = flattened.clone().view(len(item["rows"]), flattened.shape[1], 9, 128)
        for index, destinations in enumerate(item["destinations"]):
            for destination_index, destination in enumerate(destinations):
                changed[index, destination, 1] += terms[index, destination_index].to(
                    device=changed.device, dtype=changed.dtype)
        return (changed.reshape_as(flattened),) + tuple(arguments[1:])
    handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(patch)
    try:
        return backend.native(item["base_batch"], capture=False)
    finally:
        handle.remove()


def first_axis(matrix):
    _u, _s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return vh[0].contiguous()


def affine_fit(x, y):
    design = torch.stack((x, torch.ones_like(x)), dim=1)
    solution = torch.linalg.lstsq(design, y[:, None]).solution[:, 0]
    return solution


def correlation(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    denominator = float(xc.norm() * yc.norm())
    return float((xc @ yc) / denominator) if denominator > 1e-12 else float("nan")


def changed_with_vectors(item, layer, heads, vectors):
    base = item[f"base{layer}"]
    changed = dict(base)
    changed["head_output"] = base["head_output"].clone()
    width = vectors.shape[1] // len(heads)
    for index, query in enumerate(item["base_batch"].semantic_positions):
        for offset, head in enumerate(heads):
            changed["head_output"][index, int(query), head] += vectors[index, offset * width:(offset + 1) * width].to(
                device=changed["head_output"].device, dtype=changed["head_output"].dtype)
    return changed


def run_responses(backend, item, vectors9, vectors11):
    positions = tuple((int(query),) for query in item["base_batch"].semantic_positions)
    specs = ({"layer": 9, "base_capture": item["base9"],
              "changed_capture": changed_with_vectors(item, 9, (1, 4), vectors9),
              "selected_heads": (1, 4), "positions_by_row": positions},
             {"layer": 11, "base_capture": item["base11"],
              "changed_capture": changed_with_vectors(item, 11, (3,), vectors11),
              "selected_heads": (3,), "positions_by_row": positions})
    return attention_eval.intervene_ordered_head_output_deltas(backend, item["base_batch"], specs)


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
    items = {name: prepare_item(backend, rows) for name, rows in split_rows.items()}
    forwards = 6 * len(items)
    evaluations = sum(6 * len(rows) for rows in split_rows.values())
    matrices = {name: {"writer_terms": writer_terms(item),
                       "block9": reader_matrix(item, 9, (1, 4)),
                       "block11": reader_matrix(item, 11, (3,))}
                for name, item in items.items()}
    direct_errors = {}
    for name, item in items.items():
        direct = run_direct_writer(backend, item, matrices[name]["writer_terms"])
        direct_errors[name] = pair_error(direct, item["writer_output"])
        forwards += 1
        evaluations += len(item["rows"])

    writer_fit = matrices["fit"]["writer_terms"].sum(dim=1)
    writer_axis = first_axis(writer_fit)
    reader_axes = {reader: first_axis(matrices["fit"][reader])
                   for reader in ("block9", "block11")}
    x_fit = writer_fit @ writer_axis
    coefficients = {reader: affine_fit(x_fit, matrices["fit"][reader] @ reader_axes[reader])
                    for reader in ("block9", "block11")}
    intercepts = {reader: float((matrices["fit"][reader] @ reader_axes[reader]).mean())
                  for reader in ("block9", "block11")}
    predicted_coefficients, correlations = {}, {}
    for name in ("heldout", "a2", "p"):
        x = matrices[name]["writer_terms"].sum(dim=1) @ writer_axis
        predicted_coefficients[name] = {
            reader: coefficients[reader][0] * x + coefficients[reader][1]
            for reader in ("block9", "block11")}
        if name in {"heldout", "a2"}:
            correlations[name] = {reader: correlation(
                predicted_coefficients[name][reader], matrices[name][reader] @ reader_axes[reader])
                for reader in ("block9", "block11")}

    records, outputs = [], {}
    for name in ("heldout", "a2"):
        item = items[name]
        full9, full11 = matrices[name]["block9"], matrices[name]["block11"]
        oracle9 = (full9 @ reader_axes["block9"])[:, None] * reader_axes["block9"][None, :]
        oracle11 = (full11 @ reader_axes["block11"])[:, None] * reader_axes["block11"][None, :]
        pred9 = predicted_coefficients[name]["block9"][:, None] * reader_axes["block9"][None, :]
        pred11 = predicted_coefficients[name]["block11"][:, None] * reader_axes["block11"][None, :]
        int9 = torch.full_like(predicted_coefficients[name]["block9"], intercepts["block9"])[:, None] * reader_axes["block9"][None, :]
        int11 = torch.full_like(predicted_coefficients[name]["block11"], intercepts["block11"])[:, None] * reader_axes["block11"][None, :]
        arm_vectors = {"full_captured_response": (full9, full11),
                       "oracle_rank1_response": (oracle9, oracle11),
                       "predicted_rank1_response": (pred9, pred11),
                       "intercept_only_response": (int9, int11)}
        outputs[name] = {arm: run_responses(backend, item, *vectors)
                         for arm, vectors in arm_vectors.items()}
        outputs[name]["zero_response"] = item["base_output"]
        forwards += len(arm_vectors)
        evaluations += len(arm_vectors) * len(item["rows"])
        for arm in ANSWER_ARMS:
            for record in source_groups.recovery_records(
                    item["rows"], item["base_output"], item["donor_output"], outputs[name][arm], arm=arm):
                record["split"] = name
                records.append(record)

    control_outputs = {}
    for name in ("p",):
        item = items[name]
        pred9 = predicted_coefficients[name]["block9"][:, None] * reader_axes["block9"][None, :]
        pred11 = predicted_coefficients[name]["block11"][:, None] * reader_axes["block11"][None, :]
        int9 = torch.full_like(predicted_coefficients[name]["block9"], intercepts["block9"])[:, None] * reader_axes["block9"][None, :]
        int11 = torch.full_like(predicted_coefficients[name]["block11"], intercepts["block11"])[:, None] * reader_axes["block11"][None, :]
        control_outputs[name] = {
            "predicted_rank1_response": run_responses(backend, item, pred9, pred11),
            "intercept_only_response": run_responses(backend, item, int9, int11)}
        forwards += 2
        evaluations += 2 * len(item["rows"])

    summaries = {}
    for name in ("heldout", "a2"):
        family = "A1" if name == "heldout" else "A2"
        summaries[name] = {arm: source_groups.summarize([
            record for record in records if record["split"] == name and record["arm"] == arm])
            for arm in ANSWER_ARMS}
        summaries[name]["family"] = family
    fractions = {name: {arm: summaries[name][arm]["mean_recovery"]
                        / summaries[name]["full_captured_response"]["mean_recovery"]
                        for arm in ANSWER_ARMS[1:]}
                 for name in ("heldout", "a2")}
    target_scale = statistics.median(abs(donor - base)
        for name in ("heldout", "a2")
        for base, donor in zip(axis_values(items[name]["base_output"]), axis_values(items[name]["donor_output"])))
    controls = {name: {arm: statistics.mean(abs(value - base) / target_scale
        for value, base in zip(axis_values(output), axis_values(items[name]["base_output"])))
        for arm, output in control_outputs[name].items()} for name in ("p",)}
    pred_a = bool(max(direct_errors.values()) <= 1e-4
                  and max(item["identity_error"] for item in items.values()) <= 1e-4
                  and max(item["reconstruction_error"] for item in items.values()) <= 5e-4)
    pred_b = all(abs(correlations[name][reader]) >= (0.75 if name == "heldout" else 0.50)
                 for name in ("heldout", "a2") for reader in ("block9", "block11"))
    pred_c = all(fractions[name]["predicted_rank1_response"] >= 0.60
                 and summaries[name]["predicted_rank1_response"]["direction_fraction"] >= 0.75
                 for name in ("heldout", "a2"))
    pred_d = all(fractions[name]["predicted_rank1_response"]
                 - fractions[name]["intercept_only_response"] >= 0.15
                 for name in ("heldout", "a2"))
    pred_e = controls["p"]["predicted_rank1_response"] <= 0.20
    pred_f = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["split"], record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {"pred_a_exact_writer_vector": pred_a,
                   "pred_b_coefficient_prediction": pred_b,
                   "pred_c_predicted_program_material": pred_c,
                   "pred_d_prediction_beats_intercept": pred_d,
                   "pred_e_aligned_p_selective": pred_e,
                   "pred_f_price_exact": pred_f}
    terminal = "invalid" if not pred_a or not pred_f else (
        "screen" if all(predictions.values()) else
        "wrong_predictor" if not pred_b or not pred_c or not pred_d else "null")
    result = {"schema": "temporal_auxiliary_fresh_writer_to_reader_coefficients_result_v3",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "instrument": {"direct_writer_scored_logit_errors": direct_errors,
              "capture_identity_max_abs": max(item["identity_error"] for item in items.values()),
              "attention_reconstruction_max_abs": max(item["reconstruction_error"] for item in items.values())},
              "affine_coefficients": {reader: [float(value) for value in coefficients[reader]]
                                      for reader in ("block9", "block11")},
              "intercept_only_coefficients": intercepts, "coefficient_correlations": correlations,
              "summaries": summaries, "fraction_of_full_response": fractions, "controls": controls,
              "predictions": predictions, "price": {"model_forwards": forwards,
              "example_evaluations": evaluations, "answer_changing_records": len(records),
              "fitted_scalars": 4, "stored_axis_coordinates": 512,
              "transformer_backwards": 0, "model_updates": 0}, "records": records,
              "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
          "affine_coefficients", "coefficient_correlations", "fraction_of_full_response",
          "controls", "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
