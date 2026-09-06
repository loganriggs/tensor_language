#!/usr/bin/env python3
"""Partition exact block8H1 source terms at will/had subject destinations."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_partition pred_b_head1_effect_recurrence pred_c_literal_cue_dominates pred_d_prefix_and_local_secondary pred_e_exact_zero_fit_price
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
import circuit_das_subspace as das
import circuit_fast_screen_candidate_temporal_auxiliary as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_block8h1_subject_source_groups_v1.json"
HEAD_SCREEN = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8_subject_attention_heads_v2_result.json"
CUBE = ROOT / "circuits/followups/temporal_auxiliary_will_had_subject_onset_block8_component_cube_v2_result.json"
FAST_SCREEN = ROOT / "circuits/fast_screens/temporal_auxiliary_will_vs_had_v1_result.json"
EVALUATOR = ROOT / "ops/attention_source_destination_eval.py"
BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8h1_subject_source_groups_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.block8h1_subject_source_groups_v1"
EXPECTED = {
    "prior": "79ccfdc64d48b96d34c796e3d3ba7200fe210cd94b65c1c6748d0beb98c935ac",
    "head_screen": "febac51eeb2bf452f6bc80f1dd4d221714da899274888fd2d0bc950e7e03e55a",
    "cube": "973c0e490ed04c07fbc410e4b2960aa27db9a7ffc7f219948a68e5dd947a431b",
    "fast_screen": "71a833c52c39b98be6f576d11749522e2725c540a29d61a2352e315978684ec9",
    "evaluator": "3e66f65aa5ca4a84676a267f54b95f49fc2220c8efc6775cfac1a48ce972ab5b",
    "builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
}
EXPECTED_ROWS_SHA256 = "b74ff65cf6b59142bf38172dae7f70a25e5f471bbeaa038f9b3971caedf913da"
ARMS = ("full_h1", "all_sources") + destination_source.GROUPS
FULL_TARGET = {"A1": 0.23994593880671558, "A2": 0.14374750286867724}
MODEL_FORWARDS = 14
EXAMPLE_EVALUATIONS = 448
INTERVENTION_RECORDS = 320


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
        "head_screen": HEAD_SCREEN,
        "cube": CUBE,
        "fast_screen": FAST_SCREEN,
        "evaluator": EVALUATOR,
        "builder": BUILDER,
    }
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    head_screen = json.loads(HEAD_SCREEN.read_text())
    cube = json.loads(CUBE.read_text())
    fast_screen = json.loads(FAST_SCREEN.read_text())
    rows_all = candidate.build_rows()
    if candidate.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("row authority changed")
    rows = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    capability = [
        cell for cell in fast_screen["run"]["capability_cells"]
        if cell["family"] in {"A1", "A2"}
    ]
    if (
        head_screen.get("terminal") != "screen"
        or head_screen.get("shared_passing_heads") != [1]
        or not cube["predictions"].get("pred_d_attention8_is_material")
        or fast_screen.get("terminal") != "screen"
        or len(rows) != 64
        or len(capability) != 4
        or not all(cell["passed"] for cell in capability)
        or len(ARMS) != 5
    ):
        raise ExperimentError("population or parent writer evidence changed")
    return rows, fast_screen, capability


def dryrun_receipt():
    return {
        "candidate_id": CANDIDATE_ID,
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "rows": 64,
        "arms": list(ARMS),
        "destinations_per_row": 2,
        "selected_heads": [1],
        "model_forwards": MODEL_FORWARDS,
        "example_evaluations": EXAMPLE_EVALUATIONS,
        "intervention_records": INTERVENTION_RECORDS,
        "fitted_scalars": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }


def main():
    rows, fast_screen, capability_cells = validate_static()
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
    frozen_native = {
        (str(item["row_id"]), str(item["side"])): (
            float(item["answer_logit"]), float(item["foil_logit"])
        )
        for item in fast_screen["run"]["native_logits"]
        if item["family"] in {"A1", "A2"}
    }
    batches = []
    native_error = 0.0
    reconstruction_error = 0.0
    forward_calls = 0
    evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        outputs, captures = {}, {}
        for side, batch in (("base", base_batch), ("donor", donor_batch)):
            outputs[side], captures[side] = destination_source.capture_layer_attention(
                backend, batch, layer=8
            )
            frozen = producer.BatchOutput(
                tuple(frozen_native[(row_id, side)] for row_id in batch.row_ids), {}
            )
            native_error = max(native_error, pair_error(outputs[side], frozen))
            reconstruction_error = max(
                reconstruction_error, float(captures[side]["reconstruction_max_abs"])
            )
            forward_calls += 1
            evaluations += len(family_rows)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        if any(len(items) != 2 for items in destinations):
            raise ExperimentError("subject destination coverage changed")
        batches.append(
            (family_rows, base_batch, donor_batch, destinations, outputs, captures)
        )

    records = []
    arm_outputs = {}
    for arm in ARMS:
        for family_rows, base_batch, donor_batch, destinations, outputs, captures in batches:
            family = str(family_rows[0]["transform_id"])
            if arm == "full_h1":
                patched = destination_source.intervene_complete_heads(
                    backend, base_batch, donor_batch, captures["donor"], destinations,
                    layer=8, selected_heads=(1,),
                )
            else:
                names = destination_source.GROUPS if arm == "all_sources" else (arm,)
                patched = destination_source.intervene_source_groups(
                    backend, base_batch, donor_batch, captures["base"], captures["donor"],
                    destinations, names, layer=8, selected_heads=(1,),
                )
            arm_outputs[(arm, family)] = patched
            forward_calls += 1
            evaluations += len(family_rows)
            records.extend(source_score.recovery_records(
                family_rows, outputs["base"], outputs["donor"], patched, arm=arm
            ))

    closure_error = max(
        pair_error(arm_outputs[("full_h1", family)], arm_outputs[("all_sources", family)])
        for family in ("A1", "A2")
    )
    summaries = {
        arm: source_score.summarize_by_family([
            record for record in records if record["arm"] == arm
        ])
        for arm in ARMS
    }
    full = summaries["full_h1"]
    pred_a = bool(
        all(cell["passed"] for cell in capability_cells)
        and native_error <= 1.0e-4
        and reconstruction_error <= 5.0e-4
        and closure_error <= 1.0e-4
        and len(records) == INTERVENTION_RECORDS
        and all(math.isfinite(float(record["recovery"])) for record in records)
    )
    pred_b = all(
        abs(full[family]["mean_recovery"] - FULL_TARGET[family]) <= 0.03
        and full[family]["direction_fraction"] == 1.0
        for family in ("A1", "A2")
    )
    pred_c = all(
        summaries["cue"][family]["mean_recovery"] > 0.0
        and summaries["cue"][family]["mean_recovery"] >= 0.50 * full[family]["mean_recovery"]
        and summaries["cue"][family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")
    )
    pred_d = all(
        summaries[arm][family]["mean_absolute_recovery"] <= 0.35 * full[family]["mean_recovery"]
        for arm in ("prefix", "local") for family in ("A1", "A2")
    )
    pred_e = bool(
        forward_calls == MODEL_FORWARDS
        and evaluations == EXAMPLE_EVALUATIONS
        and len(records) == INTERVENTION_RECORDS
        and len({(record["arm"], record["row_id"]) for record in records})
        == INTERVENTION_RECORDS
    )
    predictions = {
        "pred_a_authority_capability_exact_partition": pred_a,
        "pred_b_head1_effect_recurrence": pred_b,
        "pred_c_literal_cue_dominates": pred_c,
        "pred_d_prefix_and_local_secondary": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else (
        "null" if pred_a and pred_b and pred_e else "invalid"
    )
    reason = {
        "screen": "block8h1_directly_transports_temporal_cue_to_subject_onset",
        "null": "block8h1_uses_contextual_source_mixture",
        "invalid": "authority_capability_reconstruction_closure_recurrence_coverage_or_price_invalid",
    }[terminal]
    result = {
        "schema": "temporal_auxiliary_will_had_block8h1_subject_source_groups_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "dryrun": dryrun,
        "instrument": {
            "native_scored_logit_max_abs": native_error,
            "source_reconstruction_max_abs": reconstruction_error,
            "all_sources_full_h1_scored_logit_max_abs": closure_error,
            "model_head_count": backend.model.config.n_head,
        },
        "summaries": summaries,
        "predictions": predictions,
        "price": {
            "model_forwards": forward_calls,
            "example_evaluations": evaluations,
            "intervention_records": len(records),
            "fitted_scalars": 0,
            "grid_evaluations": 0,
            "root_evaluations": 0,
            "transformer_backwards": 0,
            "model_updates": 0,
        },
        "intervention_records": records,
        "terminal": terminal,
        "reason": reason,
        "next_action": (
            "test the exact cue-to-subject H1 path on fresh temporal constructions and selective cue edits"
            if terminal == "screen"
            else "retain H1 identity but model its contextual source mixture before any path promotion"
        ),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        key: result[key]
        for key in (
            "candidate_id", "instrument", "summaries", "predictions", "price",
            "terminal", "reason", "next_action"
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
