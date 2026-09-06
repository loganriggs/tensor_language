#!/usr/bin/env python3
"""Install captured block9/block11 reader responses without their upstream writer."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_response_instrument pred_b_writer_recurrence pred_c_each_installed_reader_is_material pred_d_joint_response_program_is_sufficient pred_e_unselected_head_response_is_inert pred_f_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as attention_eval
import attention_source_group_eval as source_groups
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_two_reader_response_program_v1.json"
JOINT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_joint_reader_mediation_v1_result.json"
H3 = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block11_h3_response_v1_result.json"
BLOCK9 = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_reader_mediation_v1_result.json"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_two_reader_response_program_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_two_reader_response_program_v1"
EXPECTED = {
    "prior": "d8cbad03f9cbce20634b8fa46c5da92fc53536c6b41998731372eb72b0c88080",
    "joint": "7132fab362c1f137ed650bf8144a2b08a175d161c21db7171c2e1b52f2ffa173",
    "h3": "ee95aef443d63ce936f011ce2d551b8a0b220aa701507ecad30a72383475405a",
    "block9": "505ce01c9b88e7f088554db9fd8d0fa2eafeabe43067aa61cdd9f2681f420f68",
    "attention": "e948c1950ff3deac055cfb91a1ece9c417236580bcc131811730bf1fad4d9f9b",
    "mediation": "05d07578129db0cb8bf5e46d08e6cd6e9c2bb9c3883675586c83c3efd24b8fda",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
}
ARMS = ("writer_only", "install_block9_response", "install_block11_response",
        "install_both_responses", "install_other_heads_response", "base_identity")
WRITER_TARGET = {"A1": 0.17215762686594877, "A2": 0.11321352225024732}
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 24, 768, 384


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
    paths = {"prior": PRIOR, "joint": JOINT, "h3": H3, "block9": BLOCK9,
             "attention": ATTENTION, "mediation": MEDIATION, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or evaluator hash changed")
    prior, joint, h3, block9 = [json.loads(path.read_text())
                                for path in (PRIOR, JOINT, H3, BLOCK9)]
    rows = [row for row in candidate.build_rows() if row["transform_id"] in {"A1", "A2"}]
    if (prior.get("candidate_id") != CANDIDATE_ID or joint.get("terminal") != "screen"
            or h3.get("terminal") != "screen" or block9.get("terminal") != "null"
            or not joint["predictions"]["pred_d_joint_readers_are_nearly_complete"]
            or len(rows) != 64):
        raise ExperimentError("population or reader authority changed")
    return rows


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "rows": 64,
            "arms": list(ARMS), "model_forwards": MODEL_FORWARDS,
            "example_evaluations": EXAMPLE_EVALUATIONS, "records": RECORDS,
            "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}


def capture_with_writer(backend, item, layer):
    hook = mediation.fixed_source_delta_hook(
        backend, item["base_batch"], item["donor_batch"], item["writer_base"],
        item["writer_donor"], item["destinations"], ("cue",), selected_heads=(1,))
    handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(hook)
    try:
        return attention_eval.capture_layer_attention(backend, item["base_batch"], layer)
    finally:
        handle.remove()


def response_spec(item, layer, heads):
    return {"layer": layer, "base_capture": item[f"base{layer}"],
            "changed_capture": item[f"changed{layer}"], "selected_heads": heads,
            "positions_by_row": tuple((int(query),) for query in item["base_batch"].semantic_positions)}


def main():
    rows = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    items, identity_error, reconstruction_error = [], 0.0, 0.0
    forwards = evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, writer_base = attention_eval.capture_layer_attention(backend, base_batch, 8)
        donor_output, writer_donor = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        base9_output, base9 = attention_eval.capture_layer_attention(backend, base_batch, 9)
        base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        item = {"rows": family_rows, "base_batch": base_batch, "donor_batch": donor_batch,
                "base_output": base_output, "donor_output": donor_output,
                "writer_base": writer_base, "writer_donor": writer_donor,
                "destinations": onset.positions_for_group(base_batch, donor_batch, "subject_onset"),
                "base9": base9, "base11": base11}
        changed9_output, item["changed9"] = capture_with_writer(backend, item, 9)
        changed11_output, item["changed11"] = capture_with_writer(backend, item, 11)
        forwards += 6
        evaluations += 6 * len(family_rows)
        identity_error = max(identity_error, pair_error(base_output, base9_output),
                             pair_error(base_output, base11_output),
                             pair_error(changed9_output, changed11_output))
        reconstruction_error = max(reconstruction_error, *(
            float(capture["reconstruction_max_abs"]) for capture in
            (writer_base, writer_donor, base9, base11, item["changed9"], item["changed11"])))
        items.append(item)

    records, base_identity_error = [], 0.0
    for arm in ARMS:
        for item in items:
            if arm == "writer_only":
                output, diagnostics = mediation.run_composed_multi_reader(
                    backend, item["base_batch"], item["donor_batch"], item["writer_base"],
                    item["writer_donor"], item["destinations"], ())
                reconstruction_error = max(reconstruction_error, *diagnostics.values(), 0.0)
            elif arm == "base_identity":
                output = backend.native(item["base_batch"], capture=False)
                base_identity_error = max(base_identity_error, pair_error(output, item["base_output"]))
            else:
                specs = []
                if arm in {"install_block9_response", "install_both_responses"}:
                    specs.append(response_spec(item, 9, (1, 4)))
                if arm in {"install_block11_response", "install_both_responses"}:
                    specs.append(response_spec(item, 11, (3,)))
                if arm == "install_other_heads_response":
                    specs = [response_spec(item, 9, (0, 2, 3, 5, 6, 7, 8)),
                             response_spec(item, 11, (0, 1, 2, 4, 5, 6, 7, 8))]
                output = attention_eval.intervene_ordered_head_output_deltas(
                    backend, item["base_batch"], specs)
            forwards += 1
            evaluations += len(item["rows"])
            records.extend(source_groups.recovery_records(
                item["rows"], item["base_output"], item["donor_output"], output, arm=arm))

    summaries = {arm: source_groups.summarize_by_family(
        [record for record in records if record["arm"] == arm]) for arm in ARMS}
    fractions = {arm: {family: summaries[arm][family]["mean_recovery"]
                       / summaries["writer_only"][family]["mean_recovery"]
                       for family in ("A1", "A2")} for arm in ARMS[1:5]}
    pred_a = identity_error <= 1e-4 and reconstruction_error <= 5e-4 and base_identity_error <= 1e-4
    pred_b = all(abs(summaries["writer_only"][family]["mean_recovery"] - WRITER_TARGET[family]) <= 0.03
                 and summaries["writer_only"][family]["direction_fraction"] == 1.0
                 for family in ("A1", "A2"))
    pred_c = all(0.25 <= fractions[arm][family] <= 0.75
                 for arm in ("install_block9_response", "install_block11_response")
                 for family in ("A1", "A2"))
    pred_d = all(0.75 <= fractions["install_both_responses"][family] <= 1.25
                 and summaries["install_both_responses"][family]["direction_fraction"] >= 0.75
                 for family in ("A1", "A2"))
    pred_e = all(abs(fractions["install_other_heads_response"][family]) <= 0.15
                 for family in ("A1", "A2"))
    pred_f = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {"pred_a_authority_capability_exact_response_instrument": pred_a,
                   "pred_b_writer_recurrence": pred_b,
                   "pred_c_each_installed_reader_is_material": pred_c,
                   "pred_d_joint_response_program_is_sufficient": pred_d,
                   "pred_e_unselected_head_response_is_inert": pred_e,
                   "pred_f_exact_zero_fit_price": pred_f}
    terminal = "screen" if all(predictions.values()) else (
        "mediation_only" if all((pred_a, pred_b, pred_c, pred_e, pred_f)) else
        "null" if pred_a and pred_b and pred_f else "invalid")
    result = {"schema": "temporal_auxiliary_fresh_two_reader_response_program_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "instrument": {"capture_identity_max_abs": identity_error,
              "attention_reconstruction_max_abs": reconstruction_error,
              "base_identity_scored_logit_max_abs": base_identity_error},
              "summaries": summaries, "fraction_of_writer": fractions,
              "predictions": predictions, "price": {"model_forwards": forwards,
              "example_evaluations": evaluations, "records": len(records), "fitted_scalars": 0,
              "transformer_backwards": 0, "model_updates": 0}, "records": records,
              "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "summaries",
          "fraction_of_writer", "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
