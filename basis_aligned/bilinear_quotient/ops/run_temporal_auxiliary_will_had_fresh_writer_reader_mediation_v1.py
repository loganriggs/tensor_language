#!/usr/bin/env python3
"""Selectively mediate the fresh block8H1 cue writer through block9 H1/H4."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_mediation_instrument pred_b_fresh_writer_effect_recurrence pred_c_subject_reader_mediates_writer pred_d_reader_path_is_source_selective pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as destination_source
import attention_source_group_eval as source_groups
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_writer_reader_mediation_v1.json"
WRITER_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8h1_fresh_cue_path_v1_result.json"
READER_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9h1h4_source_groups_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
SOURCE_EVAL = ROOT / "ops/attention_source_destination_eval.py"
MEDIATION_EVAL = ROOT / "ops/attention_path_mediation_eval.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_reader_mediation_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_writer_reader_mediation_v1"
EXPECTED = {
    "prior": "9521eb4ef759d5f7b889871d731cbb92271c35fa07d43b8139314803ec14a924",
    "writer_result": "2da5c4b424b620bbfe24cc98049a0520429102b7d37de45d49a48ef887181641",
    "reader_result": "c738d33cae91ee70f911ea5ea610d0156bc618b41b6574051d33ef6cd21e9f8d",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "source_eval": "3e66f65aa5ca4a84676a267f54b95f49fc2220c8efc6775cfac1a48ce972ab5b",
    "mediation_eval": "292a72049c19b2325465704567bfc0815a3acec7a4074858688603655a49c04c",
}
EXPECTED_ROWS_SHA256 = "c35d7f2159c47620b6f36f12c4f6c41c1630afae61a7f101ab9ff74e4e7e5e0a"
ARMS = (
    "writer_only",
    "writer_subject_reader_clamped",
    "writer_reader_complement_clamped",
    "writer_complete_h1h4_clamped",
    "base_subject_reader_self_clamp",
)
WRITER_TARGET = {"A1": 0.17215762686594877, "A2": 0.11321352225024732}
MODEL_FORWARDS = 16
EXAMPLE_EVALUATIONS = 512
RECORDS = 320


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def pair_error(first, second):
    return max(
        abs(float(a) - float(b))
        for pair_a, pair_b in zip(first.answer_foil, second.answer_foil)
        for a, b in zip(pair_a, pair_b)
    )


def validate_static():
    paths = {
        "prior": PRIOR,
        "writer_result": WRITER_RESULT,
        "reader_result": READER_RESULT,
        "builder": BUILDER,
        "source_eval": SOURCE_EVAL,
        "mediation_eval": MEDIATION_EVAL,
    }
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    writer = json.loads(WRITER_RESULT.read_text())
    reader = json.loads(READER_RESULT.read_text())
    rows_all = candidate.build_rows()
    if candidate.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("fresh row authority changed")
    rows = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    if (
        writer.get("terminal") != "screen"
        or not all(cell["passed"] for cell in writer["capability_cells"])
        or reader.get("terminal") != "null"
        or not reader["predictions"].get("pred_a_authority_capability_exact_partition")
        or len(rows) != 64
        or len(ARMS) != 5
    ):
        raise ExperimentError("population or writer/reader authority changed")
    return rows, writer["capability_cells"]


def dryrun_receipt():
    return {
        "candidate_id": CANDIDATE_ID,
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "rows": 64,
        "arms": list(ARMS),
        "model_forwards": MODEL_FORWARDS,
        "example_evaluations": EXAMPLE_EVALUATIONS,
        "writer_arm_records": 256,
        "self_clamp_controls": 64,
        "fitted_scalars": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }


def main():
    rows, capability_cells = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = utc_now()
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    if backend.model.config.n_head != 9:
        raise ExperimentError("frozen head inventory changed")
    batches = []
    capture_identity_error = 0.0
    reconstruction_error = 0.0
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, writer_base = destination_source.capture_layer_attention(
            backend, base_batch, layer=8
        )
        donor_output, writer_donor = destination_source.capture_layer_attention(
            backend, donor_batch, layer=8
        )
        reader_base_output, reader_base = destination_source.capture_layer_attention(
            backend, base_batch, layer=9
        )
        forward_calls += 3
        evaluations += 3 * len(family_rows)
        capture_identity_error = max(capture_identity_error, pair_error(base_output, reader_base_output))
        reconstruction_error = max(
            reconstruction_error,
            float(writer_base["reconstruction_max_abs"]),
            float(writer_donor["reconstruction_max_abs"]),
            float(reader_base["reconstruction_max_abs"]),
        )
        writer_destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        partitions = source_groups.batch_partitions(base_batch, donor_batch)
        subject_positions = tuple(mapping["subject_onset"] for mapping in partitions)
        complement_positions = tuple(
            tuple(
                position for name in source_groups.GROUP_ORDER if name != "subject_onset"
                for position in mapping[name]
            )
            for mapping in partitions
        )
        batches.append({
            "rows": family_rows,
            "base_batch": base_batch,
            "donor_batch": donor_batch,
            "base_output": base_output,
            "donor_output": donor_output,
            "writer_base": writer_base,
            "writer_donor": writer_donor,
            "writer_destinations": writer_destinations,
            "reader_base": reader_base,
            "subject_positions": subject_positions,
            "complement_positions": complement_positions,
        })

    records = []
    dynamic_reconstruction_error = 0.0
    self_clamp_error = 0.0
    for arm in ARMS:
        for item in batches:
            kwargs = {}
            enable_writer = arm != "base_subject_reader_self_clamp"
            if arm in {"writer_subject_reader_clamped", "base_subject_reader_self_clamp"}:
                kwargs["reader_positions_by_row"] = item["subject_positions"]
            elif arm == "writer_reader_complement_clamped":
                kwargs["reader_positions_by_row"] = item["complement_positions"]
            elif arm == "writer_complete_h1h4_clamped":
                kwargs["clamp_complete_reader"] = True
            output, current_error = mediation.run_composed(
                backend,
                item["base_batch"],
                item["donor_batch"],
                item["writer_base"],
                item["writer_donor"],
                item["writer_destinations"],
                item["reader_base"],
                enable_writer=enable_writer,
                **kwargs,
            )
            forward_calls += 1
            evaluations += len(item["rows"])
            dynamic_reconstruction_error = max(dynamic_reconstruction_error, current_error)
            if arm == "base_subject_reader_self_clamp":
                self_clamp_error = max(self_clamp_error, pair_error(output, item["base_output"]))
            records.extend(source_groups.recovery_records(
                item["rows"], item["base_output"], item["donor_output"], output, arm=arm
            ))

    summaries = {
        arm: source_groups.summarize_by_family([
            record for record in records if record["arm"] == arm
        ])
        for arm in ARMS
    }
    writer = summaries["writer_only"]
    subject_retained = {
        family: summaries["writer_subject_reader_clamped"][family]["mean_recovery"]
        / writer[family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    complement_retained = {
        family: summaries["writer_reader_complement_clamped"][family]["mean_recovery"]
        / writer[family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    complete_retained = {
        family: summaries["writer_complete_h1h4_clamped"][family]["mean_recovery"]
        / writer[family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    pred_a = bool(
        all(cell["passed"] for cell in capability_cells)
        and capture_identity_error <= 1.0e-4
        and reconstruction_error <= 5.0e-4
        and dynamic_reconstruction_error <= 5.0e-4
        and self_clamp_error <= 1.0e-4
        and len(records) == RECORDS
        and all(math.isfinite(float(record["recovery"])) for record in records)
    )
    pred_b = all(
        abs(writer[family]["mean_recovery"] - WRITER_TARGET[family]) <= 0.03
        and writer[family]["direction_fraction"] == 1.0
        for family in ("A1", "A2")
    )
    pred_c = all(subject_retained[family] <= 0.25 for family in ("A1", "A2"))
    pred_d = all(
        complement_retained[family] >= 0.75 and complete_retained[family] <= 0.25
        for family in ("A1", "A2")
    )
    pred_e = bool(
        forward_calls == MODEL_FORWARDS
        and evaluations == EXAMPLE_EVALUATIONS
        and len(records) == RECORDS
        and len({(record["arm"], record["row_id"]) for record in records}) == RECORDS
    )
    predictions = {
        "pred_a_authority_capability_exact_mediation_instrument": pred_a,
        "pred_b_fresh_writer_effect_recurrence": pred_b,
        "pred_c_subject_reader_mediates_writer": pred_c,
        "pred_d_reader_path_is_source_selective": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else (
        "null" if pred_a and pred_b and pred_e else "invalid"
    )
    reason = {
        "screen": "fresh_writer_effect_is_selectively_mediated_by_subject_reader_bank",
        "null": "fresh_writer_and_reader_endpoints_do_not_form_registered_mediated_path",
        "invalid": "authority_capability_dynamic_capture_identity_coverage_or_price_invalid",
    }[terminal]
    result = {
        "schema": "temporal_auxiliary_will_had_fresh_writer_reader_mediation_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "dryrun": dryrun,
        "instrument": {
            "capture_identity_scored_logit_max_abs": capture_identity_error,
            "native_source_reconstruction_max_abs": reconstruction_error,
            "dynamic_source_reconstruction_max_abs": dynamic_reconstruction_error,
            "base_subject_reader_self_clamp_scored_logit_max_abs": self_clamp_error,
        },
        "summaries": summaries,
        "retained_fraction_of_writer": {
            "subject_reader_clamped": subject_retained,
            "reader_complement_clamped": complement_retained,
            "complete_h1h4_clamped": complete_retained,
        },
        "predictions": predictions,
        "price": {
            "model_forwards": forward_calls,
            "example_evaluations": evaluations,
            "writer_arm_records": 256,
            "self_clamp_controls": 64,
            "total_records": len(records),
            "fitted_scalars": 0,
            "grid_evaluations": 0,
            "root_evaluations": 0,
            "transformer_backwards": 0,
            "model_updates": 0,
        },
        "records": records,
        "terminal": terminal,
        "reason": reason,
        "next_action": (
            "add unrelated-behavior collateral controls and then compile the identified temporal writer-reader interface"
            if terminal == "screen"
            else "retain separately identified writer and reader endpoints without composing them"
        ),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        key: result[key]
        for key in (
            "candidate_id", "instrument", "summaries", "retained_fraction_of_writer",
            "predictions", "price", "terminal", "reason", "next_action"
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
