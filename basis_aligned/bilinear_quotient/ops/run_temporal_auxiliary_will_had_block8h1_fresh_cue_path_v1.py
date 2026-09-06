#!/usr/bin/env python3
"""Test the identified block8H1 cue path on fresh Later/Previously text."""

# BQGATE: EXPERIMENT pred_a_fresh_native_capability_and_exact_instrument pred_b_fresh_cue_path_is_causal pred_c_head_and_source_specificity_recur pred_d_noncue_sources_are_secondary pred_e_exact_zero_fit_price
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
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_block8h1_fresh_cue_path_v1.json"
PATH_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8h1_subject_source_groups_v1_result.json"
HEAD_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8_subject_attention_heads_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
EVALUATOR = ROOT / "ops/attention_source_destination_eval.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8h1_fresh_cue_path_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.block8h1_fresh_cue_path_v1"
EXPECTED = {
    "prior": "3d8ae5b6b765122152cf05fcddb3cdb8528fbc70b6bb3fc2ae82574ce09b9b8d",
    "path_result": "8865fe22a3c12e367709706ff0b941b3c2488d1d9608ce1921a4cfa73b22c6b9",
    "head_result": "febac51eeb2bf452f6bc80f1dd4d221714da899274888fd2d0bc950e7e03e55a",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "evaluator": "3e66f65aa5ca4a84676a267f54b95f49fc2220c8efc6775cfac1a48ce972ab5b",
}
EXPECTED_ROWS_SHA256 = "c35d7f2159c47620b6f36f12c4f6c41c1630afae61a7f101ab9ff74e4e7e5e0a"
ARMS = ("full_attention", "h1_complete", "h1_all_sources", "h1_cue", "h1_noncue")
MODEL_FORWARDS = 14
EXAMPLE_EVALUATIONS = 448
INTERVENTION_RECORDS = 320


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {
        "prior": PRIOR,
        "path_result": PATH_RESULT,
        "head_result": HEAD_RESULT,
        "builder": BUILDER,
        "evaluator": EVALUATOR,
    }
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    path_result = json.loads(PATH_RESULT.read_text())
    head_result = json.loads(HEAD_RESULT.read_text())
    rows_all = candidate.build_rows()
    if candidate.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("fresh row authority changed")
    rows = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    if (
        path_result.get("terminal") != "screen"
        or head_result.get("shared_passing_heads") != [1]
        or len(rows) != 64
        or len(ARMS) != 5
    ):
        raise ExperimentError("fresh population or identified path changed")
    return rows


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
        "intervention_records": INTERVENTION_RECORDS,
        "fitted_scalars": 0,
        "transformer_backwards": 0,
        "model_updates": 0,
    }


def prediction_values(pred_a, pred_b, pred_c, pred_d, pred_e):
    return {
        "pred_a_fresh_native_capability_and_exact_instrument": bool(pred_a),
        "pred_b_fresh_cue_path_is_causal": bool(pred_b),
        "pred_c_head_and_source_specificity_recur": bool(pred_c),
        "pred_d_noncue_sources_are_secondary": bool(pred_d),
        "pred_e_exact_zero_fit_price": bool(pred_e),
    }


def capability_cells(rows, outputs_by_family):
    cells = []
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for direction in ("future_to_anterior", "anterior_to_future"):
            indices = [
                index for index, row in enumerate(family_rows)
                if row["direction_id"] == direction
            ]
            for side in ("base", "donor"):
                pairs = outputs_by_family[family][side].answer_foil
                accuracy = sum(pairs[index][0] > pairs[index][1] for index in indices) / len(indices)
                cells.append({
                    "family": family,
                    "direction": direction,
                    "side": side,
                    "count": len(indices),
                    "accuracy": accuracy,
                    "minimum_accuracy": 0.85,
                    "passed": accuracy >= 0.85,
                })
    return cells


def write_capability_stop(started_utc, started, dryrun, cells, forwards, evaluations):
    result = {
        "schema": "temporal_auxiliary_will_had_block8h1_fresh_cue_path_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "dryrun": dryrun,
        "capability_cells": cells,
        "predictions": prediction_values(False, False, False, False, False),
        "price": {
            "model_forwards": forwards,
            "example_evaluations": evaluations,
            "intervention_records": 0,
            "fitted_scalars": 0,
            "transformer_backwards": 0,
            "model_updates": 0,
        },
        "terminal": "capability_stop",
        "reason": "fresh_later_previously_behavior_not_natively_supported",
        "next_action": "retain the identified discovery path but choose a separately capable fresh temporal authority",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "capability_cells": cells,
        "terminal": result["terminal"],
        "reason": result["reason"],
    }, sort_keys=True))


def main():
    rows = validate_static()
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
    outputs_by_family = {}
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
            reconstruction_error = max(
                reconstruction_error, float(captures[side]["reconstruction_max_abs"])
            )
            forward_calls += 1
            evaluations += len(family_rows)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        batches.append((family_rows, base_batch, donor_batch, destinations, outputs, captures))
        outputs_by_family[family] = outputs

    cells = capability_cells(rows, outputs_by_family)
    if not all(cell["passed"] for cell in cells):
        write_capability_stop(started_utc, started, dryrun, cells, forward_calls, evaluations)
        return

    records = []
    arm_outputs = {}
    for arm in ARMS:
        for family_rows, base_batch, donor_batch, destinations, outputs, captures in batches:
            family = str(family_rows[0]["transform_id"])
            if arm in {"full_attention", "h1_complete"}:
                selected = tuple(range(9)) if arm == "full_attention" else (1,)
                patched = destination_source.intervene_complete_heads(
                    backend, base_batch, donor_batch, captures["donor"], destinations,
                    layer=8, selected_heads=selected,
                )
            else:
                names = {
                    "h1_all_sources": destination_source.GROUPS,
                    "h1_cue": ("cue",),
                    "h1_noncue": ("prefix", "local"),
                }[arm]
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
        max(
            abs(float(a) - float(b))
            for pair_a, pair_b in zip(
                arm_outputs[("h1_complete", family)].answer_foil,
                arm_outputs[("h1_all_sources", family)].answer_foil,
            )
            for a, b in zip(pair_a, pair_b)
        )
        for family in ("A1", "A2")
    )
    summaries = {
        arm: source_score.summarize_by_family([
            record for record in records if record["arm"] == arm
        ])
        for arm in ARMS
    }
    full_attention = summaries["full_attention"]
    h1 = summaries["h1_complete"]
    cue = summaries["h1_cue"]
    pred_a = bool(
        all(cell["passed"] for cell in cells)
        and reconstruction_error <= 5.0e-4
        and closure_error <= 1.0e-4
        and len(records) == INTERVENTION_RECORDS
        and all(math.isfinite(float(record["recovery"])) for record in records)
    )
    pred_b = bool(
        cue["A1"]["mean_recovery"] >= 0.10
        and cue["A2"]["mean_recovery"] >= 0.06
        and all(cue[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    )
    pred_c = all(
        cue[family]["mean_recovery"] >= 0.75 * h1[family]["mean_recovery"]
        and h1[family]["mean_recovery"] >= 0.75 * full_attention[family]["mean_recovery"]
        for family in ("A1", "A2")
    )
    pred_d = all(
        summaries["h1_noncue"][family]["mean_absolute_recovery"]
        <= 0.35 * h1[family]["mean_recovery"]
        for family in ("A1", "A2")
    )
    pred_e = bool(
        forward_calls == MODEL_FORWARDS
        and evaluations == EXAMPLE_EVALUATIONS
        and len(records) == INTERVENTION_RECORDS
        and len({(record["arm"], record["row_id"]) for record in records})
        == INTERVENTION_RECORDS
    )
    predictions = prediction_values(pred_a, pred_b, pred_c, pred_d, pred_e)
    terminal = "screen" if all(predictions.values()) else (
        "null" if pred_a and pred_e else "invalid"
    )
    reason = {
        "screen": "block8h1_cue_path_generalizes_to_fresh_temporal_text",
        "null": "identified_block8h1_cue_path_does_not_generalize",
        "invalid": "authority_capability_reconstruction_closure_coverage_or_price_invalid",
    }[terminal]
    result = {
        "schema": "temporal_auxiliary_will_had_block8h1_fresh_cue_path_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "dryrun": dryrun,
        "capability_cells": cells,
        "instrument": {
            "source_reconstruction_max_abs": reconstruction_error,
            "h1_all_sources_complete_scored_logit_max_abs": closure_error,
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
            "compose the identified block8H1 writer with the downstream L9H1/H4 subject reader under registered path mediation"
            if terminal == "screen"
            else "retain discovery-only path status and test a different capable temporal paraphrase"
        ),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({
        key: result[key]
        for key in (
            "candidate_id", "capability_cells", "instrument", "summaries",
            "predictions", "price", "terminal", "reason", "next_action"
        )
    }, sort_keys=True))


if __name__ == "__main__":
    main()
