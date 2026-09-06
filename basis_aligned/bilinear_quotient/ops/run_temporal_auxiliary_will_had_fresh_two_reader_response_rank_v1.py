#!/usr/bin/env python3
"""Compress the two verified temporal reader responses with separate SVD bases."""

# BQGATE: EXPERIMENT pred_a_authority_and_exact_instrument pred_b_rank4_program_sufficient pred_c_rank4_complement_inert pred_d_monotone_tensor_compression pred_e_rank_price_is_exact
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
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
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_two_reader_response_rank_v1.json"
PROGRAM = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_two_reader_response_program_v1_result.json"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_two_reader_response_rank_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_two_reader_response_rank_v1"
EXPECTED = {
    "prior": "d95b01d8b4de5e22ac7d6338e0755b5dc42df9b26b10b1e4f0f1404d2262806c",
    "program": "3fe8a3e7edf8dc24f7977d0ce4a37f564cac35211b2a2043e76336ea33e023cb",
    "attention": "e948c1950ff3deac055cfb91a1ece9c417236580bcc131811730bf1fad4d9f9b",
    "mediation": "05d07578129db0cb8bf5e46d08e6cd6e9c2bb9c3883675586c83c3efd24b8fda",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
}
RANKS = (1, 2, 4)
ARMS = ("full_response", "rank1_response", "rank1_complement", "rank2_response",
        "rank2_complement", "rank4_response", "rank4_complement")
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 32, 720, 336


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def pair_error(first, second):
    return max(abs(float(a) - float(b)) for pa, pb in zip(first.answer_foil, second.answer_foil)
               for a, b in zip(pa, pb))


def validate_static():
    paths = {"prior": PRIOR, "program": PROGRAM, "attention": ATTENTION,
             "mediation": MEDIATION, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or implementation hash changed")
    prior = json.loads(PRIOR.read_text())
    program = json.loads(PROGRAM.read_text())
    rows = candidate.build_rows()
    family_rows = {family: [row for row in rows if row["transform_id"] == family]
                   for family in ("A1", "A2")}
    if (prior.get("candidate_id") != CANDIDATE_ID or program.get("terminal") != "screen"
            or not all(program["predictions"].values())
            or {key: len(value) for key, value in family_rows.items()} != {"A1": 32, "A2": 32}):
        raise ExperimentError("population or executable-program authority changed")
    return {"fit": family_rows["A1"][0::2], "heldout": family_rows["A1"][1::2],
            "a2": family_rows["A2"]}


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "ranks": list(RANKS),
            "arms": list(ARMS), "model_forwards": MODEL_FORWARDS,
            "example_evaluations": EXAMPLE_EVALUATIONS, "records": RECORDS,
            "transformer_backwards": 0, "model_updates": 0}


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
    changed9_output, item["changed9"] = capture_with_writer(backend, item, 9)
    changed11_output, item["changed11"] = capture_with_writer(backend, item, 11)
    item["identity_error"] = max(pair_error(base_output, base9_output),
                                 pair_error(base_output, base11_output),
                                 pair_error(changed9_output, changed11_output))
    item["reconstruction_error"] = max(float(capture["reconstruction_max_abs"])
        for capture in (writer_base, writer_donor, base9, base11, item["changed9"], item["changed11"]))
    return item


def response_matrix(item, layer, heads):
    base, changed = item[f"base{layer}"], item[f"changed{layer}"]
    rows = []
    for index, query in enumerate(item["base_batch"].semantic_positions):
        rows.append(torch.cat(tuple(
            changed["head_output"][index, int(query), head].float()
            - base["head_output"][index, int(query), head].float() for head in heads)))
    return torch.stack(rows)


def basis(matrix):
    _u, _s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return vh.T.contiguous()


def project(matrix, q, complement=False):
    projected = (matrix @ q) @ q.T
    return matrix - projected if complement else projected


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


def response_specs(item, vectors9, vectors11):
    positions = tuple((int(query),) for query in item["base_batch"].semantic_positions)
    return ({"layer": 9, "base_capture": item["base9"],
             "changed_capture": changed_with_vectors(item, 9, (1, 4), vectors9),
             "selected_heads": (1, 4), "positions_by_row": positions},
            {"layer": 11, "base_capture": item["base11"],
             "changed_capture": changed_with_vectors(item, 11, (3,), vectors11),
             "selected_heads": (3,), "positions_by_row": positions})


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
    matrices = {name: {"block9": response_matrix(item, 9, (1, 4)),
                       "block11": response_matrix(item, 11, (3,))}
                for name, item in items.items()}
    bases = {reader: basis(matrices["fit"][reader]) for reader in ("block9", "block11")}
    explained = {}
    for split in ("heldout", "a2"):
        explained[split] = {}
        for reader in ("block9", "block11"):
            matrix = matrices[split][reader]
            denominator = float(matrix.square().sum())
            explained[split][reader] = {str(rank): float(project(matrix, bases[reader][:, :rank]).square().sum()) / denominator
                                        for rank in RANKS}

    records = []
    for split in ("heldout", "a2"):
        item = items[split]
        for arm in ARMS:
            if arm == "full_response":
                vectors9, vectors11 = matrices[split]["block9"], matrices[split]["block11"]
            else:
                rank = int(arm[4])
                complement = arm.endswith("complement")
                vectors9 = project(matrices[split]["block9"], bases["block9"][:, :rank], complement)
                vectors11 = project(matrices[split]["block11"], bases["block11"][:, :rank], complement)
            output = attention_eval.intervene_ordered_head_output_deltas(
                backend, item["base_batch"], response_specs(item, vectors9, vectors11))
            forwards += 1
            evaluations += len(item["rows"])
            for record in source_groups.recovery_records(
                    item["rows"], item["base_output"], item["donor_output"], output, arm=arm):
                record["split"] = split
                records.append(record)

    summaries = {}
    for split in ("heldout", "a2"):
        family = "A1" if split == "heldout" else "A2"
        summaries[split] = {arm: {family: source_groups.summarize([
            record for record in records
            if record["split"] == split and record["arm"] == arm])} for arm in ARMS}
    fractions = {split: {arm: summaries[split][arm]["A1" if split == "heldout" else "A2"]["mean_recovery"]
        / summaries[split]["full_response"]["A1" if split == "heldout" else "A2"]["mean_recovery"]
        for arm in ARMS[1:]} for split in ("heldout", "a2")}
    identity_error = max(item["identity_error"] for item in items.values())
    reconstruction_error = max(item["reconstruction_error"] for item in items.values())
    full = {split: summaries[split]["full_response"]["A1" if split == "heldout" else "A2"]["mean_recovery"]
            for split in ("heldout", "a2")}
    pred_a = bool(identity_error <= 1e-4 and reconstruction_error <= 5e-4
                  and abs(full["heldout"] - 0.17215762686594877) <= 0.03
                  and abs(full["a2"] - 0.11416906328249907) <= 0.03)
    pred_b = all(0.85 <= fractions[split]["rank4_response"] <= 1.15
                 and summaries[split]["rank4_response"]["A1" if split == "heldout" else "A2"]["direction_fraction"] >= 0.90
                 for split in ("heldout", "a2"))
    pred_c = all(abs(fractions[split]["rank4_complement"]) <= 0.20 for split in ("heldout", "a2"))
    pred_d = all(values["1"] <= values["2"] + 1e-8 and values["2"] <= values["4"] + 1e-8
                 for split in explained.values() for values in split.values())
    pred_e = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["split"], record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {"pred_a_authority_and_exact_instrument": pred_a,
                   "pred_b_rank4_program_sufficient": pred_b,
                   "pred_c_rank4_complement_inert": pred_c,
                   "pred_d_monotone_tensor_compression": pred_d,
                   "pred_e_rank_price_is_exact": pred_e}
    terminal = "invalid" if not pred_a or not pred_e else (
        "screen" if all(predictions.values()) else
        "higher_rank_needed" if not pred_b or not pred_c else "null")
    result = {"schema": "temporal_auxiliary_fresh_two_reader_response_rank_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "instrument": {"capture_identity_max_abs": identity_error,
              "attention_reconstruction_max_abs": reconstruction_error}, "explained_energy": explained,
              "summaries": summaries, "fraction_of_full_response": fractions,
              "predictions": predictions, "price": {"model_forwards": forwards,
              "example_evaluations": evaluations, "records": len(records),
              "basis_coordinates": {"rank1": 384, "rank2": 768, "rank4": 1536},
              "transformer_backwards": 0, "model_updates": 0}, "records": records,
              "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "explained_energy",
          "fraction_of_full_response", "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
