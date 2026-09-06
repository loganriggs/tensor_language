#!/usr/bin/env python3
"""Joint dynamic mediation of fresh block8H1 through block9 H1/H4 and block11 H3."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_multi_reader_instrument pred_b_writer_effect_recurrence pred_c_each_reader_is_material pred_d_joint_readers_are_nearly_complete pred_e_joint_path_is_source_selective pred_f_exact_zero_fit_price
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
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_writer_joint_reader_mediation_v1.json"
BLOCK9 = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_reader_mediation_v1_result.json"
H3 = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block11_h3_response_v1_result.json"
H3_SOURCE = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block11h3_source_response_v1_result.json"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
SOURCE = ROOT / "ops/attention_source_group_eval.py"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_joint_reader_mediation_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_writer_joint_reader_mediation_v1"
EXPECTED = {
    "prior": "66e5f7e58f7e71b49b6ef26e1a127cf2efbc18fc8c2bd4fe05259148e9e2acc6",
    "block9": "505ce01c9b88e7f088554db9fd8d0fa2eafeabe43067aa61cdd9f2681f420f68",
    "h3": "ee95aef443d63ce936f011ce2d551b8a0b220aa701507ecad30a72383475405a",
    "h3_source": "1fd089aeb63e2ec7e1771d54170461dc18b3bd105e2b7fc08413d8456a515cf1",
    "mediation": "05d07578129db0cb8bf5e46d08e6cd6e9c2bb9c3883675586c83c3efd24b8fda",
    "attention": "806bd970b773c839cf4eb8d74c1fdbf4102fda32d2188d22daa8a1d5624c2bdf",
    "source": "6ecf5f40b92f94cb32bccf1a703e527a3d468281936e63d4c7e91e8af66b4348",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
}
ARMS = (
    "writer_only", "writer_block9_subject_clamped", "writer_block11_subject_clamped",
    "writer_both_subject_clamped", "writer_both_non_subject_clamped",
    "base_both_subject_self_clamp",
)
WRITER_TARGET = {"A1": 0.17215762686594877, "A2": 0.11321352225024732}
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 20, 640, 384


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
    paths = {"prior": PRIOR, "block9": BLOCK9, "h3": H3, "h3_source": H3_SOURCE,
             "mediation": MEDIATION, "attention": ATTENTION, "source": SOURCE,
             "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or evaluator hash changed")
    prior, block9, h3, h3_source = [json.loads(path.read_text())
                                    for path in (PRIOR, BLOCK9, H3, H3_SOURCE)]
    rows_all = candidate.build_rows()
    rows = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    if (prior.get("candidate_id") != CANDIDATE_ID or block9.get("terminal") != "null"
            or h3.get("terminal") != "screen" or h3_source.get("terminal") != "screen"
            or not block9["predictions"]["pred_a_authority_capability_exact_mediation_instrument"]
            or not h3["predictions"]["pred_d_external_h3_dominates_fresh_response"]
            or len(rows) != 64 or len(ARMS) != 6):
        raise ExperimentError("population or reader authority changed")
    return rows, block9


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "rows": 64,
            "arms": list(ARMS), "model_forwards": MODEL_FORWARDS,
            "example_evaluations": EXAMPLE_EVALUATIONS, "records": RECORDS,
            "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}


def main():
    rows, block9_authority = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    items, capture_identity, native_reconstruction = [], 0.0, 0.0
    forwards = evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, writer_base = attention_eval.capture_layer_attention(backend, base_batch, 8)
        donor_output, writer_donor = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        base9_output, base9 = attention_eval.capture_layer_attention(backend, base_batch, 9)
        base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        forwards += 4
        evaluations += 4 * len(family_rows)
        capture_identity = max(capture_identity, pair_error(base_output, base9_output),
                               pair_error(base_output, base11_output))
        native_reconstruction = max(native_reconstruction, *(
            float(capture["reconstruction_max_abs"])
            for capture in (writer_base, writer_donor, base9, base11)))
        mappings = source_groups.batch_partitions(base_batch, donor_batch)
        subject = tuple(mapping["subject_onset"] for mapping in mappings)
        non_subject = tuple(tuple(position for name in source_groups.GROUP_ORDER
                                  if name != "subject_onset" for position in mapping[name])
                            for mapping in mappings)
        items.append({"rows": family_rows, "base_batch": base_batch, "donor_batch": donor_batch,
                      "base_output": base_output, "donor_output": donor_output,
                      "writer_base": writer_base, "writer_donor": writer_donor,
                      "destinations": onset.positions_for_group(base_batch, donor_batch, "subject_onset"),
                      "base9": base9, "base11": base11, "subject": subject,
                      "non_subject": non_subject})

    records, dynamic_reconstruction, self_clamp_error = [], 0.0, 0.0
    for arm in ARMS:
        for item in items:
            specs = []
            selected_positions = item["non_subject"] if arm == "writer_both_non_subject_clamped" else item["subject"]
            if arm in {"writer_block9_subject_clamped", "writer_both_subject_clamped",
                       "writer_both_non_subject_clamped", "base_both_subject_self_clamp"}:
                specs.append({"layer": 9, "base_capture": item["base9"], "heads": (1, 4),
                              "positions_by_row": selected_positions})
            if arm in {"writer_block11_subject_clamped", "writer_both_subject_clamped",
                       "writer_both_non_subject_clamped", "base_both_subject_self_clamp"}:
                specs.append({"layer": 11, "base_capture": item["base11"], "heads": (3,),
                              "positions_by_row": selected_positions})
            output, diagnostics = mediation.run_composed_multi_reader(
                backend, item["base_batch"], item["donor_batch"], item["writer_base"],
                item["writer_donor"], item["destinations"], specs,
                enable_writer=arm != "base_both_subject_self_clamp")
            forwards += 1
            evaluations += len(item["rows"])
            dynamic_reconstruction = max(dynamic_reconstruction, *diagnostics.values(), 0.0)
            if arm == "base_both_subject_self_clamp":
                self_clamp_error = max(self_clamp_error, pair_error(output, item["base_output"]))
            records.extend(source_groups.recovery_records(
                item["rows"], item["base_output"], item["donor_output"], output, arm=arm))

    summaries = {arm: source_groups.summarize_by_family(
        [record for record in records if record["arm"] == arm]) for arm in ARMS}
    retained = {arm: {family: summaries[arm][family]["mean_recovery"]
                      / summaries["writer_only"][family]["mean_recovery"]
                      for family in ("A1", "A2")} for arm in ARMS[1:5]}
    pred_a = bool(capture_identity <= 1e-4 and native_reconstruction <= 5e-4
                  and dynamic_reconstruction <= 5e-4 and self_clamp_error <= 1e-4
                  and block9_authority["instrument"]["base_subject_reader_self_clamp_scored_logit_max_abs"] <= 1e-4)
    pred_b = all(abs(summaries["writer_only"][family]["mean_recovery"] - WRITER_TARGET[family]) <= 0.03
                 and summaries["writer_only"][family]["direction_fraction"] == 1.0
                 for family in ("A1", "A2"))
    pred_c = all(retained[arm][family] <= 0.70 for arm in
                 ("writer_block9_subject_clamped", "writer_block11_subject_clamped")
                 for family in ("A1", "A2"))
    pred_d = all(retained["writer_both_subject_clamped"][family] <= 0.15
                 for family in ("A1", "A2"))
    pred_e = all(retained["writer_both_non_subject_clamped"][family] >= 0.90
                 for family in ("A1", "A2"))
    pred_f = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {
        "pred_a_authority_capability_exact_multi_reader_instrument": pred_a,
        "pred_b_writer_effect_recurrence": pred_b, "pred_c_each_reader_is_material": pred_c,
        "pred_d_joint_readers_are_nearly_complete": pred_d,
        "pred_e_joint_path_is_source_selective": pred_e,
        "pred_f_exact_zero_fit_price": pred_f,
    }
    terminal = "screen" if all(predictions.values()) else (
        "partial_path" if all((pred_a, pred_b, pred_c, pred_e, pred_f)) else
        "null" if pred_a and pred_b and pred_f else "invalid")
    result = {"schema": "temporal_auxiliary_fresh_writer_joint_reader_mediation_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "instrument": {"capture_identity_max_abs": capture_identity,
              "native_reconstruction_max_abs": native_reconstruction,
              "dynamic_reconstruction_max_abs": dynamic_reconstruction,
              "double_self_clamp_scored_logit_max_abs": self_clamp_error},
              "summaries": summaries, "retained_fraction_of_writer": retained,
              "predictions": predictions, "price": {"model_forwards": forwards,
              "example_evaluations": evaluations, "records": len(records), "fitted_scalars": 0,
              "transformer_backwards": 0, "model_updates": 0}, "records": records,
              "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "summaries",
          "retained_fraction_of_writer", "predictions", "price", "terminal")}, sort_keys=True))


if __name__ == "__main__":
    main()
