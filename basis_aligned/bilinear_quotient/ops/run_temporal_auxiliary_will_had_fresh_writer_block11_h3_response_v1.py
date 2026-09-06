#!/usr/bin/env python3
"""Prospective external-H3 response test for the exact fresh temporal writer."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_head_response_instrument pred_b_direct_boundary11_recurrence pred_c_all_head_response_is_material pred_d_external_h3_dominates_fresh_response pred_e_exact_zero_fit_price
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
import attention_source_group_eval as source_score
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_das_subspace as das
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset
import written_state_block_factorial_eval as crossing


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_writer_block11_h3_response_v1.json"
OFFSET = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_subject_consumer_offset_v2_result.json"
WRITER = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8h1_fresh_cue_path_v1_result.json"
ASPECTUAL_H3 = ROOT / "circuits/followups/aspectual_anchor_attention11_15_single_head_confirmation_v1_result.json"
NUMBER_H3 = ROOT / "circuits/followups/head_localization_number_attn11_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
ATTENTION_EVAL = ROOT / "ops/attention_source_destination_eval.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
ONSET = ROOT / "ops/residual_source_onset_eval.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block11_h3_response_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_writer_block11_h3_response_v1"
EXPECTED = {
    "prior": "1991f97e29ccf6822c0ff946c710d2cd3e1011fd41f629fdff6047eca0d7feb5",
    "offset": "864f40e041cd4028c242fb96c816347875c59511d04e908372fd533b8c58c7ca",
    "writer": "2da5c4b424b620bbfe24cc98049a0520429102b7d37de45d49a48ef887181641",
    "aspectual_h3": "be2c405c4b7023c57d4a11baf9be0bc999fb3835908f34fb88c1011ab146353f",
    "number_h3": "70011cd36792a6040047d2c7d0d50f4818bde3b4a814750f6191d7de74a57559",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "attention_eval": "806bd970b773c839cf4eb8d74c1fdbf4102fda32d2188d22daa8a1d5624c2bdf",
    "mediation": "9180ef34ec376729103e200ae2b2a2ce93d5f8ed0b293b0b1b459a55d71a079d",
    "onset": "c276450cc9ec7c2b0a05e2be0e88bac3df9af7003e370b99b66552083c4f4b45",
}
ROWS_SHA256 = "c35d7f2159c47620b6f36f12c4f6c41c1630afae61a7f101ab9ff74e4e7e5e0a"
DIRECT_TARGET = {"A1": 0.09417874839356316, "A2": 0.059744184811890595}
ARMS = ("direct_boundary11", "all_heads", "h3", "complement")
MODEL_FORWARDS = 16
EXAMPLE_EVALUATIONS = 512
INTERVENTION_RECORDS = 256


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "offset": OFFSET, "writer": WRITER, "aspectual_h3": ASPECTUAL_H3, "number_h3": NUMBER_H3, "builder": BUILDER, "attention_eval": ATTENTION_EVAL, "mediation": MEDIATION, "onset": ONSET}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or evaluator hash changed")
    prior, offset, writer, aspectual = map(
        lambda path: json.loads(path.read_text()), (PRIOR, OFFSET, WRITER, ASPECTUAL_H3)
    )
    rows_all = candidate.build_rows()
    rows = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or candidate.validate_rows(rows_all) != ROWS_SHA256
        or offset.get("latest_material_boundary_at_0p015_and_0p75") != 11
        or writer.get("terminal") != "screen"
        or aspectual.get("terminal") != "screen"
        or aspectual["score"]["boundaries"]["11"]["head"] != 3
        or len(rows) != 64 or len(writer.get("capability_cells", ())) != 8
        or not all(cell["passed"] for cell in writer["capability_cells"])
    ):
        raise ExperimentError("population, capability, offset, or external H3 authority changed")
    return rows, writer


def dryrun_receipt():
    return {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": 64,
        "arms": list(ARMS), "model_forwards": MODEL_FORWARDS,
        "example_evaluations": EXAMPLE_EVALUATIONS,
        "intervention_records": INTERVENTION_RECORDS, "fitted_scalars": 0,
        "grid_evaluations": 0, "root_evaluations": 0,
        "transformer_backwards": 0, "model_updates": 0,
    }


def main():
    rows, writer_authority = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = utc_now()
    started = time.perf_counter()
    backend = onset.ResidualGroupBackend.load("cuda")
    if backend.model.config.n_head != 9:
        raise ExperimentError("frozen head inventory changed")
    records = []
    reconstruction_error = 0.0
    native_identity_error = 0.0
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, writer_base = attention_eval.capture_layer_attention(backend, base_batch, 8)
        donor_output, writer_donor = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        _writer_output, writer_states = mediation.capture_source_written_states(
            backend, base_batch, donor_batch, writer_base, writer_donor, destinations,
            maximum_boundary=12,
        )
        base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        direct_output, changed11 = attention_eval.capture_layer_attention(
            backend, base_batch, 11,
            call=lambda: backend.forward_states(
                base_batch, maximum_boundary=12, donor_batch=donor_batch,
                donor_states=writer_states, boundary=11, group_name="subject_onset",
            )[0],
        )
        forward_calls += 5
        evaluations += 5 * len(family_rows)
        reconstruction_error = max(
            reconstruction_error,
            *(float(capture["reconstruction_max_abs"]) for capture in (writer_base, writer_donor, base11, changed11)),
        )
        native_identity_error = max(native_identity_error, crossing.pair_error(base_output, base11_output))
        outputs = {"direct_boundary11": direct_output}
        for arm, heads in (("all_heads", range(9)), ("h3", (3,)), ("complement", (0, 1, 2, 4, 5, 6, 7, 8))):
            outputs[arm] = attention_eval.intervene_head_output_delta(
                backend, base_batch, base11, changed11, layer=11, selected_heads=heads
            )
            forward_calls += 1
            evaluations += len(family_rows)
        for arm in ARMS:
            records.extend(source_score.recovery_records(
                family_rows, base_output, donor_output, outputs[arm], arm=arm
            ))

    summaries = {
        arm: source_score.summarize_by_family([record for record in records if record["arm"] == arm])
        for arm in ARMS
    }
    retained = {
        family: summaries["h3"][family]["mean_recovery"] / summaries["all_heads"][family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    pred_a = bool(
        reconstruction_error <= 1e-4 and native_identity_error <= 1e-4
        and all(cell["passed"] for cell in writer_authority["capability_cells"])
    )
    pred_b = all(
        abs(summaries["direct_boundary11"][family]["mean_recovery"] - DIRECT_TARGET[family]) <= 1e-6
        and summaries["direct_boundary11"][family]["direction_fraction"] == 1.0
        for family in ("A1", "A2")
    )
    pred_c = all(
        summaries["all_heads"][family]["mean_recovery"] >= {"A1": 0.05, "A2": 0.03}[family]
        and summaries["all_heads"][family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")
    )
    pred_d = all(
        retained[family] >= 0.75 and summaries["h3"][family]["direction_fraction"] >= 0.75
        and abs(summaries["complement"][family]["mean_recovery"])
        <= 0.25 * abs(summaries["all_heads"][family]["mean_recovery"])
        for family in ("A1", "A2")
    )
    pred_e = bool(
        forward_calls == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
        and len(records) == INTERVENTION_RECORDS
        and len({(r["arm"], r["row_id"]) for r in records}) == INTERVENTION_RECORDS
        and all(math.isfinite(float(r["recovery"])) for r in records)
    )
    predictions = {
        "pred_a_authority_capability_exact_head_response_instrument": pred_a,
        "pred_b_direct_boundary11_recurrence": pred_b,
        "pred_c_all_head_response_is_material": pred_c,
        "pred_d_external_h3_dominates_fresh_response": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else (
        "null" if pred_a and pred_b and pred_c and pred_e else "invalid"
    )
    reason = {
        "screen": "external_h3_dominates_exact_fresh_writer_block11_response",
        "null": "valid_material_block11_response_but_external_h3_dominance_failed",
        "invalid": "authority_exactness_recurrence_coverage_or_price_invalid",
    }[terminal]
    result = {
        "schema": "temporal_auxiliary_will_had_fresh_writer_block11_h3_response_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "rows_sha256": ROWS_SHA256, "dryrun": dryrun, "predictions": predictions,
        "instrument": {"attention_reconstruction_max_abs": reconstruction_error, "native_identity_max_abs": native_identity_error, "model_head_count": 9},
        "summaries": summaries, "h3_retained_fraction": retained,
        "price": {"model_forwards": forward_calls, "example_evaluations": evaluations, "intervention_records": len(records), "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal, "reason": reason,
        "next_action": "localize the exact H3 source terms under the fresh writer" if terminal == "screen" else "screen all block11 heads under a new registered split",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "summaries": summaries, "h3_retained": retained, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
