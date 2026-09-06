#!/usr/bin/env python3
# BQGATE: frozen A-E disjoint suffix singleton predictions; CUDA is managed-queue only.
"""Confirm dominant single heads inside attention11 and attention15."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_battery_integration_contract as battery
import circuit_candidate_aspectual_lexical_holdout_v5 as holdout
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_attention11_15_head_compression_split_v1 as parent
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_attention11_15_single_head_confirmation_v1.json"
PARENT = ROOT / "circuits/followups/aspectual_anchor_attention11_15_head_compression_split_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_aspectual_anchor_attention11_15_head_compression_split_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_attention11_15_single_head_confirmation_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.attention11_15_single_head_confirmation_v1"
EXPECTED_PRIOR_SHA256 = "a985211f781e8553dab8f28ecfed06228d7ffa3933a06ec9e2a7ad94ee6d92e7"
EXPECTED_PARENT_SHA256 = "77c631ca20f92328ea6a76a832a5e01581d3cea328d41137a04d721b4a9ef191"
EXPECTED_PARENT_RUNNER_SHA256 = "225c822c37007445353febd133fe308230b4f47b25b572bfe2c2614cc54950f2"
EXPECTED_BUILDER_SHA256 = "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859"
EXPECTED_ROWS_SHA256 = "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
EXPECTED_CONFIRMATION_SHA256 = "ad198e745d3c2b900e097219aae918f9ec506271f159bdcdf9852db56e12e55b"
HEAD_BY_BOUNDARY = {11: 3, 15: 5}
FOUR_BY_BOUNDARY = {11: (3, 7, 2, 6), 15: (5, 1, 4, 6)}
ARMS = ("no_heads", "dominant_single", "selected_four", "all_heads")
MODEL_FORWARDS_MAX = 24
EXAMPLE_EVALUATIONS_MAX = 224


class ExperimentError(RuntimeError):
    pass


def sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    for file_path, digest in {
        PRIOR: EXPECTED_PRIOR_SHA256,
        PARENT: EXPECTED_PARENT_SHA256,
        PARENT_RUNNER: EXPECTED_PARENT_RUNNER_SHA256,
        BUILDER: EXPECTED_BUILDER_SHA256,
    }.items():
        if sha256(file_path) != digest:
            raise ExperimentError(f"authority hash changed: {file_path.name}")
    prior = json.loads(PRIOR.read_text())
    result = json.loads(PARENT.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID or result.get("terminal") != "screen":
        raise ExperimentError("prior or parent terminal changed")
    for boundary in parent.BOUNDARIES:
        selection = result["score"]["selection"][str(boundary)]
        confirmation = result["score"]["confirmation"][str(boundary)]
        if selection["ranking"][0] != f"h{HEAD_BY_BOUNDARY[boundary]}":
            raise ExperimentError("dominant selection head changed")
        if tuple(confirmation["selected_heads"]) != FOUR_BY_BOUNDARY[boundary]:
            raise ExperimentError("four-head confirmation set changed")
    rows_all = holdout.build_rows()
    if holdout.validate_rows(rows_all) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("row authority changed")
    target = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    confirmation = tuple(target[16:])
    if suffix.ids_sha256(confirmation) != EXPECTED_CONFIRMATION_SHA256:
        raise ExperimentError("confirmation split changed")
    parent_rows = parent_runner.candidate.build_rows(parent_runner.candidate.TASK_ID)
    parent_spec = parent_runner.build_spec(parent_rows)
    spec = replace(
        parent_spec,
        experiment_id="aspectual-anchor-attention11-15-single-head-confirmation-v1",
        authority_sha256=EXPECTED_ROWS_SHA256,
        expected_fit_rows=len(rows_all),
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT", forward_calls=MODEL_FORWARDS_MAX,
            example_evaluations=EXAMPLE_EVALUATIONS_MAX,
            backward_calls=0, model_updates=0, evidence_bytes=32768,
        ),
    )
    enriched_all = screen.validate_fit_authority(spec, rows_all)
    confirmation = tuple(enriched_all[str(row["row_id"])] for row in confirmation)
    if len(confirmation) != 16 or len(ARMS) != 4:
        raise ExperimentError("population or arm inventory changed")
    return confirmation, spec, result


def summarize(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("recovery is missing or nonfinite")
    return {
        "count": len(values),
        "mean_recovery": statistics.fmean(values),
        "mean_absolute_recovery": statistics.fmean(abs(value) for value in values),
        "direction_fraction": sum(value > 0.0 for value in values) / len(values),
    }


def main() -> None:
    rows, spec, parent_result = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_attention11_15_single_head_confirmation_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "row_count": len(rows),
        "head_by_boundary": HEAD_BY_BOUNDARY,
        "four_by_boundary": FOUR_BY_BOUNDARY,
        "arms": list(ARMS),
        "model_forwards_max": MODEL_FORWARDS_MAX,
        "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0,
        "model_updates": 0,
        "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = utc_now()
    started = time.perf_counter()
    backend = parent.HeadCompressionBackend.load("cuda")
    native = {}
    values = {
        boundary: {arm: {"A1": [], "A2": []} for arm in ARMS}
        for boundary in parent.BOUNDARIES
    }
    writer_values = {"A1": [], "A2": []}
    raw_records = []
    manual_base_max_abs = 0.0
    writer_tensor_error_max_abs = 0.0
    attention_projection_error_max_abs = {boundary: 0.0 for boundary in parent.BOUNDARIES}
    crossing_tensor_error_max_abs = {boundary: 0.0 for boundary in parent.BOUNDARIES}
    forward_calls = 0
    evaluations = 0

    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            base_native, base_bilinear = backend.capture_bilinear(base_batch)
            donor_native, donor_bilinear = backend.capture_bilinear(donor_batch)
            base_manual, base_capture = backend.capture_suffix_heads(base_batch)
            writer_output, hybrid_capture, writer_error = backend.capture_writer_suffix_heads(
                base_batch, donor_batch, base_bilinear, donor_bilinear
            )
            forward_calls += 4
            evaluations += 4 * len(chunk)
            writer_tensor_error_max_abs = max(writer_tensor_error_max_abs, writer_error)
            for reference, manual in zip(base_native.answer_foil, base_manual.answer_foil):
                manual_base_max_abs = max(
                    manual_base_max_abs,
                    abs(reference[0] - manual[0]), abs(reference[1] - manual[1]),
                )
            for side, output in (("base", base_native), ("donor", donor_native)):
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                        str(row["row_id"]), family, side, answer, foil
                    )
            for row, pair in zip(chunk, writer_output.answer_foil):
                answer, foil, value = suffix.recovery(row, pair, native)
                writer_values[family].append(value)
                raw_records.append({
                    "boundary": "writer", "arm_id": "writer_two_term",
                    "family": family, "row_id": str(row["row_id"]),
                    "answer_logit": answer, "foil_logit": foil, "recovery": value,
                })
            for boundary in parent.BOUNDARIES:
                projected = backend.projected_head_delta(
                    base_capture, hybrid_capture, boundary, parent.HEADS
                )
                actual_attention = (
                    hybrid_capture[f"attention{boundary}"].float()
                    - base_capture[f"attention{boundary}"].float()
                )
                attention_projection_error_max_abs[boundary] = max(
                    attention_projection_error_max_abs[boundary],
                    float((projected - actual_attention).abs().max()),
                )
                lambda0 = backend.model.transformer.h[boundary].lambdas[0]
                for i, query in enumerate(base_batch.semantic_positions):
                    reconstructed = (
                        lambda0.float() * (
                            hybrid_capture[f"resid{boundary}"][i, query].float()
                            - base_capture[f"resid{boundary}"][i, query].float()
                        )
                        + projected[i, query]
                        + hybrid_capture[f"mlp{boundary}"][i, query].float()
                        - base_capture[f"mlp{boundary}"][i, query].float()
                    )
                    direct = (
                        hybrid_capture[f"resid{boundary + 1}"][i, query].float()
                        - base_capture[f"resid{boundary + 1}"][i, query].float()
                    )
                    crossing_tensor_error_max_abs[boundary] = max(
                        crossing_tensor_error_max_abs[boundary],
                        float((reconstructed - direct).abs().max()),
                    )
                head_sets = {
                    "no_heads": (),
                    "dominant_single": (HEAD_BY_BOUNDARY[boundary],),
                    "selected_four": FOUR_BY_BOUNDARY[boundary],
                    "all_heads": parent.HEADS,
                }
                for arm, heads in head_sets.items():
                    output = backend.head_crossing(
                        base_batch, base_capture, hybrid_capture, boundary, heads
                    )
                    forward_calls += 1
                    evaluations += len(chunk)
                    for row, pair in zip(chunk, output.answer_foil):
                        answer, foil, value = suffix.recovery(row, pair, native)
                        values[boundary][arm][family].append(value)
                        raw_records.append({
                            "boundary": boundary, "arm_id": arm, "family": family,
                            "row_id": str(row["row_id"]), "answer_logit": answer,
                            "foil_logit": foil, "recovery": value,
                        })

    summaries = {}
    compression_pass = []
    family_pass = []
    control_pass = []
    for boundary in parent.BOUNDARIES:
        arm_summaries = {}
        targets = {}
        for arm in ARMS:
            families = {
                family: summarize(values[boundary][arm][family])
                for family in ("A1", "A2")
            }
            target = statistics.fmean(
                families[family]["mean_recovery"] for family in ("A1", "A2")
            )
            arm_summaries[arm] = {"families": families, "mean_target_recovery": target}
            targets[arm] = target
        singleton_increment = targets["dominant_single"] - targets["no_heads"]
        four_increment = targets["selected_four"] - targets["no_heads"]
        retained = singleton_increment / four_increment
        family_increments = {
            family: (
                arm_summaries["dominant_single"]["families"][family]["mean_recovery"]
                - arm_summaries["no_heads"]["families"][family]["mean_recovery"]
            ) for family in ("A1", "A2")
        }
        bound = parent_result["score"]["confirmation"][str(boundary)]["arms"]
        control_pass.append(all(
            abs(arm_summaries[current]["mean_target_recovery"] - bound[current]["mean_target_recovery"]) <= 1.0e-6
            for current in ("no_heads", "selected_four", "all_heads")
        ))
        compression_pass.append(retained >= 0.75)
        family_pass.append(all(value > 0.0 for value in family_increments.values()))
        summaries[str(boundary)] = {
            "head": HEAD_BY_BOUNDARY[boundary],
            "four_heads": list(FOUR_BY_BOUNDARY[boundary]),
            "arms": arm_summaries,
            "singleton_increment": singleton_increment,
            "four_head_increment": four_increment,
            "singleton_to_four_fraction": retained,
            "singleton_family_increments": family_increments,
        }
    writer_summary = {
        family: summarize(writer_values[family]) for family in ("A1", "A2")
    }
    bound_writer = parent_result["score"]["writer"]["confirmation"]
    current_capability = all(
        native[(str(row["row_id"]), side)].margin > 0.0
        for row in rows for side in ("base", "donor")
    )
    pred_a = (
        current_capability and manual_base_max_abs <= 1.0e-4
        and writer_tensor_error_max_abs <= 2.0e-3
        and all(value <= 0.04 for value in attention_projection_error_max_abs.values())
        and all(value <= 0.04 for value in crossing_tensor_error_max_abs.values())
    )
    pred_b = all(control_pass) and all(
        abs(writer_summary[family]["mean_recovery"] - bound_writer[family]["mean_recovery"]) <= 1.0e-6
        and writer_summary[family]["mean_recovery"] > 0.0
        and writer_summary[family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")
    )
    pred_c = all(compression_pass)
    pred_d = all(family_pass)
    pred_e = (
        len(raw_records) == 144 and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = (
        "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e))
        else "null" if pred_a and pred_b and pred_e
        else "invalid"
    )
    reason = {
        "screen": "dominant_single_suffix_heads_confirmed",
        "null": "one_or_both_single_head_compressions_failed",
        "invalid": "authority_split_capability_instrument_control_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_attention11_15_single_head_confirmation_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "parent_result_sha256": EXPECTED_PARENT_SHA256,
        "confirmation_row_ids_sha256": EXPECTED_CONFIRMATION_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_split_capability_and_instrument": pred_a,
            "pred_b_writer_and_control_recurrence": pred_b,
            "pred_c_single_head_compression": pred_c,
            "pred_d_single_head_family_recurrence": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "boundaries": summaries, "writer": writer_summary,
            "manual_base_scored_logit_max_abs": manual_base_max_abs,
            "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs,
            "attention_projection_error_max_abs": {
                str(key): value for key, value in attention_projection_error_max_abs.items()
            },
            "crossing_tensor_error_max_abs": {
                str(key): value for key, value in crossing_tensor_error_max_abs.items()
            },
            "forward_calls": forward_calls, "example_evaluations": evaluations,
            "raw_record_count": len(raw_records), "model_backwards": 0,
            "model_updates": 0, "fit_parameters": 0,
        },
        "intervention_logits": raw_records,
        "terminal": terminal, "reason": reason,
        "next_action": (
            "factor source terms for block11 H3 and block15 H5"
            if terminal == "screen"
            else "retain validated four-head suffix sets"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason,
        "predictions": result["predictions"],
        "compression": {
            boundary: summaries[str(boundary)]["singleton_to_four_fraction"]
            for boundary in parent.BOUNDARIES
        },
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
