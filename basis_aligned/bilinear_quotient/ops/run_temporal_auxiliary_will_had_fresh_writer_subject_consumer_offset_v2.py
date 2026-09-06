#!/usr/bin/env python3
"""Sweep the exact fresh block8H1 cue-written subject state through the suffix."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_written_state_instrument pred_b_writer_effect_recurrence pred_c_boundary10_bypass_recurrence pred_d_block15_is_last_material_consumer_window pred_e_inert_endpoint_and_exact_zero_fit_price
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
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_writer_subject_consumer_offset_v2.json"
WRITER = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8h1_fresh_cue_path_v1_result.json"
BYPASS = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block9_subject_bypass_cube_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
ONSET = ROOT / "ops/residual_source_onset_eval.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_subject_consumer_offset_v2_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_writer_subject_consumer_offset_v2"
EXPECTED = {
    "prior": "a5ea425d3e068a84be952ed14964ce72e7354089e8fcb239ba2da597ab9eeb29",
    "writer": "2da5c4b424b620bbfe24cc98049a0520429102b7d37de45d49a48ef887181641",
    "bypass": "f9cf002e526b474eff8d0f19014fbe4c9f9e336d6969189baa2046043ff589d0",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "mediation": "9180ef34ec376729103e200ae2b2a2ce93d5f8ed0b293b0b1b459a55d71a079d",
    "onset": "bad2e0fd28d26e9336cf2eaab9ef327ecc06ecdbdc5fe18ad777fb77294aa872",
}
ROWS_SHA256 = "c35d7f2159c47620b6f36f12c4f6c41c1630afae61a7f101ab9ff74e4e7e5e0a"
BOUNDARIES = tuple(range(10, 19))
WRITER_TARGET = {"A1": 0.17215762686594877, "A2": 0.11321352225024732}
BOUNDARY10_TARGET = {"A1": 0.09204826735617808, "A2": 0.05782733513712361}
MODEL_FORWARDS = 24
EXAMPLE_EVALUATIONS = 768
INTERVENTION_RECORDS = 576


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "writer": WRITER, "bypass": BYPASS, "builder": BUILDER, "mediation": MEDIATION, "onset": ONSET}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    prior = json.loads(PRIOR.read_text())
    writer = json.loads(WRITER.read_text())
    bypass = json.loads(BYPASS.read_text())
    rows_all = candidate.build_rows()
    rows = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or candidate.validate_rows(rows_all) != ROWS_SHA256
        or writer.get("terminal") != "screen"
        or bypass.get("terminal") != "null"
        or len(rows) != 64
        or len(writer.get("capability_cells", ())) != 8
        or not all(cell["passed"] for cell in writer["capability_cells"])
    ):
        raise ExperimentError("population, capability, or parent evidence changed")
    return rows, writer, bypass


def dryrun_receipt():
    return {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": 64,
        "boundaries": list(BOUNDARIES), "model_forwards": MODEL_FORWARDS,
        "example_evaluations": EXAMPLE_EVALUATIONS,
        "intervention_records": INTERVENTION_RECORDS, "fitted_scalars": 0,
        "grid_evaluations": 0, "root_evaluations": 0,
        "transformer_backwards": 0, "model_updates": 0,
    }


def main():
    rows, writer_authority, _bypass = validate_static()
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
    items = []
    writer_records = []
    reconstruction_error = 0.0
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, base_capture = destination_source.capture_layer_attention(backend, base_batch, layer=8)
        donor_output, donor_capture = destination_source.capture_layer_attention(backend, donor_batch, layer=8)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_output, writer_states = mediation.capture_source_written_states(
            backend, base_batch, donor_batch, base_capture, donor_capture, destinations,
            maximum_boundary=18,
        )
        reconstruction_error = max(
            reconstruction_error,
            float(base_capture["reconstruction_max_abs"]),
            float(donor_capture["reconstruction_max_abs"]),
        )
        forward_calls += 3
        evaluations += 3 * len(family_rows)
        writer_records.extend(source_groups.recovery_records(
            family_rows, base_output, donor_output, writer_output, arm="writer"
        ))
        items.append({
            "rows": family_rows, "base_batch": base_batch, "donor_batch": donor_batch,
            "base_output": base_output, "donor_output": donor_output,
            "donor_states": writer_states,
        })

    sweep = onset.sweep_precomputed_states(
        backend, items, boundaries=BOUNDARIES, group_name="subject_onset",
        maximum_boundary=18, recovery_bar=0.015, direction_bar=0.75,
    )
    forward_calls += sweep["forward_calls"]
    evaluations += sweep["example_evaluations"]
    writer_summary = source_groups.summarize_by_family(writer_records)
    by_boundary = {point["boundary"]: point for point in sweep["curve"]}
    boundary10 = by_boundary[10]["families"]
    boundary15 = by_boundary[15]["families"]
    boundary16 = by_boundary[16]["families"]
    latest_material = next(
        (point["boundary"] for point in reversed(sweep["curve"]) if point["passed"]), None
    )

    pred_a = bool(
        reconstruction_error <= 1e-4
        and all(cell["passed"] for cell in writer_authority["capability_cells"])
        and tuple(point["boundary"] for point in sweep["curve"]) == BOUNDARIES
    )
    pred_b = all(
        abs(writer_summary[family]["mean_recovery"] - WRITER_TARGET[family]) <= 1e-6
        and writer_summary[family]["direction_fraction"] == 1.0
        for family in ("A1", "A2")
    )
    pred_c = all(
        abs(boundary10[family]["mean_recovery"] - BOUNDARY10_TARGET[family]) <= 1e-6
        for family in ("A1", "A2")
    )
    pred_d = all(
        boundary15[family]["mean_recovery"] >= 0.015
        and boundary15[family]["direction_fraction"] >= 0.75
        and boundary16[family]["mean_recovery"] <= 0.5 * boundary15[family]["mean_recovery"]
        for family in ("A1", "A2")
    )
    unique_records = {(record["boundary"], record["row_id"]) for record in sweep["records"]}
    pred_e = bool(
        sweep["base_scored_logit_max_abs_by_boundary"]["18"] <= 1e-7
        and forward_calls == MODEL_FORWARDS
        and evaluations == EXAMPLE_EVALUATIONS
        and len(sweep["records"]) == INTERVENTION_RECORDS
        and len(unique_records) == INTERVENTION_RECORDS
        and all(math.isfinite(float(record["recovery"])) for record in sweep["records"])
    )
    predictions = {
        "pred_a_authority_capability_exact_written_state_instrument": pred_a,
        "pred_b_writer_effect_recurrence": pred_b,
        "pred_c_boundary10_bypass_recurrence": pred_c,
        "pred_d_block15_is_last_material_consumer_window": pred_d,
        "pred_e_inert_endpoint_and_exact_zero_fit_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else (
        "null" if pred_a and pred_b and pred_c and pred_e else "invalid"
    )
    reason = {
        "screen": "block15_is_final_material_subject_state_consumer_window",
        "null": "valid_curve_but_registered_block15_offset_prediction_failed",
        "invalid": "authority_exactness_recurrence_endpoint_or_price_invalid",
    }[terminal]
    result = {
        "schema": "temporal_auxiliary_will_had_fresh_writer_subject_consumer_offset_result_v2",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "rows_sha256": ROWS_SHA256, "dryrun": dryrun, "predictions": predictions,
        "instrument": {
            "source_reconstruction_max_abs": reconstruction_error,
            "boundary_base_scored_logit_max_abs": sweep["base_scored_logit_max_abs_by_boundary"],
            "model_head_count": backend.model.config.n_head,
        },
        "writer_summary": writer_summary, "curve": sweep["curve"],
        "latest_material_boundary_at_0p015_and_0p75": latest_material,
        "price": {
            "model_forwards": forward_calls, "example_evaluations": evaluations,
            "intervention_records": len(sweep["records"]), "fitted_scalars": 0,
            "grid_evaluations": 0, "root_evaluations": 0,
            "transformer_backwards": 0, "model_updates": 0,
        },
        "intervention_records": sweep["records"], "terminal": terminal, "reason": reason,
        "next_action": (
            "decompose block15 on the same fresh writer authority"
            if terminal == "screen" else
            "retain the full curve and preregister module decomposition at its empirically latest material crossing"
        ),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason,
        "predictions": predictions, "latest_material_boundary": latest_material,
        "curve": {str(point["boundary"]): {family: point["families"][family]["mean_recovery"] for family in ("A1", "A2")} for point in sweep["curve"]},
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
