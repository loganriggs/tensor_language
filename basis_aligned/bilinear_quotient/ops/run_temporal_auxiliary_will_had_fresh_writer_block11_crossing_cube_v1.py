#!/usr/bin/env python3
"""Exact block11 full-sequence component cube after the fresh cue writer."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_full_sequence_cube pred_b_boundary11_direct_ceiling_recurrence pred_c_full_sequence_cube_recovers_crossing pred_d_attention_is_dominant_transfer pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_source_destination_eval as destination_source
import attention_source_group_eval as source_score
import block_component_state_eval as components
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_das_subspace as das
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset
import written_state_block_factorial_eval as crossing_eval


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_writer_block11_crossing_cube_v1.json"
OFFSET = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_subject_consumer_offset_v2_result.json"
WRITER = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8h1_fresh_cue_path_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
COMPONENTS = ROOT / "ops/block_component_state_eval.py"
ONSET = ROOT / "ops/residual_source_onset_eval.py"
CROSSING = ROOT / "ops/written_state_block_factorial_eval.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block11_crossing_cube_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_writer_block11_crossing_cube_v1"
EXPECTED = {
    "prior": "9d91f65884dd8ed6773db15cf86ac9515c2eb8ff3308cb0c6832f1db2c2a6105",
    "offset": "864f40e041cd4028c242fb96c816347875c59511d04e908372fd533b8c58c7ca",
    "writer": "2da5c4b424b620bbfe24cc98049a0520429102b7d37de45d49a48ef887181641",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "components": "c44c5c392475fa4ead02c6402e44e6f14620caa055737d968111da80621d0379",
    "onset": "c276450cc9ec7c2b0a05e2be0e88bac3df9af7003e370b99b66552083c4f4b45",
    "crossing": "3ed7ccc4a02faf1a6130a29e685dc827ed57df0608698b3f6bca28f7a78dfa1c",
}
ROWS_SHA256 = "c35d7f2159c47620b6f36f12c4f6c41c1630afae61a7f101ab9ff74e4e7e5e0a"
DIRECT_TARGET = {"A1": 0.09417874839356316, "A2": 0.059744184811890595}
SUBJECT_ONLY_BOUNDARY12 = {"A1": 0.006125655754336495, "A2": 0.0024376408871950436}
MODEL_FORWARDS = 24
EXAMPLE_EVALUATIONS = 768
FACTORIAL_RECORDS = 512


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "offset": OFFSET, "writer": WRITER, "builder": BUILDER, "components": COMPONENTS, "onset": ONSET, "crossing": CROSSING}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or evaluator hash changed")
    prior = json.loads(PRIOR.read_text())
    offset = json.loads(OFFSET.read_text())
    writer = json.loads(WRITER.read_text())
    rows_all = candidate.build_rows()
    rows = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or candidate.validate_rows(rows_all) != ROWS_SHA256
        or offset.get("terminal") != "null"
        or offset.get("latest_material_boundary_at_0p015_and_0p75") != 11
        or writer.get("terminal") != "screen"
        or len(rows) != 64
        or len(writer.get("capability_cells", ())) != 8
        or not all(cell["passed"] for cell in writer["capability_cells"])
    ):
        raise ExperimentError("population, capability, or parent evidence changed")
    return rows, writer


def dryrun_receipt():
    return {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": 64,
        "components": list(components.COMPONENTS), "factorial_arms": 8,
        "model_forwards": MODEL_FORWARDS, "example_evaluations": EXAMPLE_EVALUATIONS,
        "factorial_records": FACTORIAL_RECORDS, "fitted_scalars": 0,
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
    items = []
    direct_records = []
    source_reconstruction_error = 0.0
    block_reconstruction_error = 0.0
    native_identity_error = 0.0
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, writer_base = destination_source.capture_layer_attention(backend, base_batch, layer=8)
        donor_output, writer_donor = destination_source.capture_layer_attention(backend, donor_batch, layer=8)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        crossing = crossing_eval.capture_crossing(
            backend, base_batch, donor_batch, writer_base, writer_donor, destinations,
            block_index=11,
        )
        forward_calls += 5
        evaluations += 5 * len(family_rows)
        source_reconstruction_error = max(
            source_reconstruction_error,
            float(writer_base["reconstruction_max_abs"]),
            float(writer_donor["reconstruction_max_abs"]),
        )
        block_reconstruction_error = max(
            block_reconstruction_error, crossing["block_reconstruction_max_abs"]
        )
        native_identity_error = max(
            native_identity_error, crossing_eval.pair_error(base_output, crossing["base_output"])
        )
        direct_records.extend(source_score.recovery_records(
            family_rows, base_output, donor_output, crossing["hybrid_output"], arm="direct_boundary11"
        ))
        items.append({
            "rows": family_rows, "base_batch": base_batch, "donor_batch": donor_batch,
            "base_output": base_output, "donor_output": donor_output, "crossing": crossing,
        })

    factorial = crossing_eval.run_full_sequence_factorial(backend, items, block_index=11)
    forward_calls += factorial["forward_calls"]
    evaluations += factorial["example_evaluations"]
    direct_summary = source_score.summarize_by_family(direct_records)
    full = factorial["summaries"]["entry+attention+mlp"]
    attention = factorial["summaries"]["attention"]
    attention_retained = {
        family: attention[family]["mean_recovery"] / full[family]["mean_recovery"]
        for family in ("A1", "A2")
    }
    pred_a = bool(
        source_reconstruction_error <= 1e-4
        and block_reconstruction_error <= 1e-4
        and native_identity_error <= 1e-4
        and factorial["empty_base_closure_max_abs"] <= 1e-4
        and factorial["full_state_closure_max_abs"] <= 1e-4
        and factorial["direct_full_scored_logit_max_abs"] <= 1e-4
        and factorial["shapley_efficiency_max_abs"] <= 1e-10
        and all(cell["passed"] for cell in writer_authority["capability_cells"])
    )
    pred_b = all(
        abs(direct_summary[family]["mean_recovery"] - DIRECT_TARGET[family]) <= 1e-6
        and direct_summary[family]["direction_fraction"] == 1.0
        for family in ("A1", "A2")
    )
    pred_c = bool(
        factorial["direct_full_scored_logit_max_abs"] <= 1e-4
        and all(full[family]["mean_recovery"] > SUBJECT_ONLY_BOUNDARY12[family] for family in ("A1", "A2"))
    )
    pred_d = all(
        factorial["shapley"][family]["attention"]
        == max(factorial["shapley"][family].values())
        and attention_retained[family] >= 0.75
        and attention[family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")
    )
    pred_e = bool(
        forward_calls == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
        and len(factorial["records"]) == FACTORIAL_RECORDS
        and len({(r["arm"], r["row_id"]) for r in factorial["records"]}) == FACTORIAL_RECORDS
        and all(math.isfinite(float(r["recovery"])) for r in factorial["records"])
    )
    predictions = {
        "pred_a_authority_capability_exact_full_sequence_cube": pred_a,
        "pred_b_boundary11_direct_ceiling_recurrence": pred_b,
        "pred_c_full_sequence_cube_recovers_crossing": pred_c,
        "pred_d_attention_is_dominant_transfer": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else (
        "null" if pred_a and pred_b and pred_c and pred_e else "invalid"
    )
    reason = {
        "screen": "attention11_dominates_fresh_writer_full_sequence_transfer",
        "null": "valid_crossing_cube_but_attention_dominance_failed",
        "invalid": "authority_exactness_recurrence_closure_coverage_or_price_invalid",
    }[terminal]
    result = {
        "schema": "temporal_auxiliary_will_had_fresh_writer_block11_crossing_cube_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "rows_sha256": ROWS_SHA256, "dryrun": dryrun, "predictions": predictions,
        "instrument": {
            "source_reconstruction_max_abs": source_reconstruction_error,
            "block_reconstruction_max_abs": block_reconstruction_error,
            "native_identity_max_abs": native_identity_error,
            "empty_base_closure_max_abs": factorial["empty_base_closure_max_abs"],
            "full_state_closure_max_abs": factorial["full_state_closure_max_abs"],
            "direct_full_scored_logit_max_abs": factorial["direct_full_scored_logit_max_abs"],
            "shapley_efficiency_max_abs": factorial["shapley_efficiency_max_abs"],
        },
        "direct_boundary11_summary": direct_summary,
        "summaries": factorial["summaries"], "component_shapley": factorial["shapley"],
        "attention_singleton_retained_fraction": attention_retained,
        "price": {
            "model_forwards": forward_calls, "example_evaluations": evaluations,
            "factorial_records": len(factorial["records"]), "fitted_scalars": 0,
            "grid_evaluations": 0, "root_evaluations": 0,
            "transformer_backwards": 0, "model_updates": 0,
        },
        "records": factorial["records"], "terminal": terminal, "reason": reason,
        "next_action": (
            "localize fresh block11 attention heads, then exact source terms"
            if terminal == "screen" else
            "retain the exact accounting and pursue the dominant or interacting block11 component"
        ),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason,
        "predictions": predictions, "direct": direct_summary,
        "shapley": factorial["shapley"], "attention_retained": attention_retained,
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
