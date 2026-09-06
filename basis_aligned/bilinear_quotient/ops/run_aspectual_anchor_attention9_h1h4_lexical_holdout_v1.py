#!/usr/bin/env python3
# BQGATE: frozen A-E prospective attention9 path predictions; CUDA is managed-queue only.
"""Prospective attention9 H1/H4 bank mediation on the sealed lexical holdout."""

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
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1 as discovery
import run_circuit_fast_screen_aspectual as parent_runner


ROOT = Path(__file__).resolve().parent.parent
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_attention9_h1h4_lexical_holdout_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_lexical_holdout_v5.py"
HOLDOUT_RESULT = ROOT / "circuits/followups/aspectual_anchor_explicit_path_lexical_holdout_v2_result.json"
DISCOVERY_RESULT = ROOT / "circuits/followups/aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1_result.json"
PROGRAM_RELEASE = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v1_result.json"
BACKEND_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp4_to_l9h1_h4_path_mediation_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_attention9_h1h4_lexical_holdout_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.attention9_h1h4_lexical_holdout_v1"
EXPECTED_PRIOR_SHA256 = "835244a1f848b16ef6adb3b7ca26c9c74286651c8a1d769e53378c501ec80638"
EXPECTED_BUILDER_SHA256 = "d06a4298af5ef375664d113c1528bbdd94c846c8b213ea92a6f7b75175846859"
EXPECTED_ROWS_SHA256 = "18dfe9b5e86387017f3b8a81d378cc4892b4ee5a219ea7e35bf02548cd54e493"
EXPECTED_HOLDOUT_RESULT_SHA256 = "fd1b4ae15e1d327001c8b172bcbecb0f15609d6da01bec8c8dddbf8de107549e"
EXPECTED_DISCOVERY_RESULT_SHA256 = "649cc961fd4203a9d7489344bbf169754081a288b5d575bcefcab2caf41da9ab"
EXPECTED_PROGRAM_RELEASE_SHA256 = "a2751011ac5fa02fcec433f2f83090f0911bdd5be1c84aeff0bf3ab8e3875cf1"
EXPECTED_BACKEND_RUNNER_SHA256 = "5449d45505267f2ebc92c9285b7a984fce773f834c2071e1d7e6d1a9f1949372"
ARMS = (
    "writer_two_term",
    "h1h4_complete",
    "h1h4_all_sources",
    "h1h4_last_period_determiner",
    "h1h4_cue_self",
)
MODEL_FORWARDS_MAX = 20
EXAMPLE_EVALUATIONS_MAX = 320


class ExperimentError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    expected = {
        PRIOR: EXPECTED_PRIOR_SHA256,
        BUILDER: EXPECTED_BUILDER_SHA256,
        HOLDOUT_RESULT: EXPECTED_HOLDOUT_RESULT_SHA256,
        DISCOVERY_RESULT: EXPECTED_DISCOVERY_RESULT_SHA256,
        PROGRAM_RELEASE: EXPECTED_PROGRAM_RELEASE_SHA256,
        BACKEND_RUNNER: EXPECTED_BACKEND_RUNNER_SHA256,
    }
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    holdout_result = json.loads(HOLDOUT_RESULT.read_text())
    discovery_result = json.loads(DISCOVERY_RESULT.read_text())
    release = json.loads(PROGRAM_RELEASE.read_text())
    if prior.get("candidate_id") != CANDIDATE_ID:
        raise ExperimentError("prior-art candidate changed")
    if holdout_result.get("terminal") != "screen":
        raise ExperimentError("prospective holdout authority is not a screen")
    if discovery_result.get("terminal") != "null":
        raise ExperimentError("discovery path terminal changed")
    if release.get("terminal") != "release" or not all(release["predictions"].values()):
        raise ExperimentError("typed program release changed")
    rows = holdout.build_rows()
    if holdout.validate_rows(rows) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("holdout row authority changed")
    parent_rows = parent_runner.candidate.build_rows(parent_runner.candidate.TASK_ID)
    parent_spec = parent_runner.build_spec(parent_rows)
    spec = replace(
        parent_spec,
        experiment_id="aspectual-anchor-attention9-h1h4-lexical-holdout-v1",
        authority_sha256=EXPECTED_ROWS_SHA256,
        expected_fit_rows=len(rows),
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT",
            forward_calls=MODEL_FORWARDS_MAX,
            example_evaluations=EXAMPLE_EVALUATIONS_MAX,
            backward_calls=0,
            model_updates=0,
            evidence_bytes=32768,
        ),
    )
    enriched_all = screen.validate_fit_authority(spec, rows)
    selected = tuple(
        enriched_all[str(row["row_id"])]
        for row in rows if row["transform_id"] in {"A1", "A2"}
    )
    if len(rows) != 64 or len(selected) != 32 or len(ARMS) != 5:
        raise ExperimentError("population or arm inventory changed")
    capability_cells = holdout_result["score"]["capability_cells"]
    if not capability_cells or not all(cell["passed"] for cell in capability_cells):
        raise ExperimentError("bound holdout capability changed")
    return selected, spec, holdout_result


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
    rows, spec, holdout_result = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_attention9_h1h4_lexical_holdout_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "builder_sha256": EXPECTED_BUILDER_SHA256,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "holdout_result_sha256": EXPECTED_HOLDOUT_RESULT_SHA256,
        "discovery_result_sha256": EXPECTED_DISCOVERY_RESULT_SHA256,
        "row_count": len(rows),
        "arms": list(ARMS),
        "heads": list(discovery.HEADS),
        "source_bank": list(discovery.BANK),
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
    backend = discovery.PathBackend.load("cuda")
    native = {}
    arm_values = {arm: {"A1": [], "A2": []} for arm in ARMS}
    raw_records = []
    manual_empty_max_abs = 0.0
    all_to_complete_max_abs = 0.0
    tensor_error_max_abs = 0.0
    attention_reconstruction_max_abs = 0.0
    forward_calls = 0
    evaluations = 0

    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            base_output, base_capture = backend.capture_bilinear(base_batch)
            donor_output, donor_capture = backend.capture_bilinear(donor_batch)
            forward_calls += 2
            evaluations += 2 * len(chunk)
            for side, output in (("base", base_output), ("donor", donor_output)):
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(
                        str(row["row_id"]), family, side, answer, foil
                    )

            empty_output, base_attention, tensor_error = backend.capture_writer(
                base_batch, donor_batch, base_capture, donor_capture, ()
            )
            writer_output, hybrid_attention, tensor_error_2 = backend.capture_writer(
                base_batch, donor_batch, base_capture, donor_capture, discovery.WRITER_FACTORS
            )
            forward_calls += 2
            evaluations += 2 * len(chunk)
            tensor_error_max_abs = max(tensor_error_max_abs, tensor_error, tensor_error_2)
            attention_reconstruction_max_abs = max(
                attention_reconstruction_max_abs,
                float(base_attention["reconstruction_max_abs"]),
                float(hybrid_attention["reconstruction_max_abs"]),
            )
            for reference, manual in zip(base_output.answer_foil, empty_output.answer_foil):
                manual_empty_max_abs = max(
                    manual_empty_max_abs,
                    abs(reference[0] - manual[0]),
                    abs(reference[1] - manual[1]),
                )
            outputs = {
                "writer_two_term": writer_output,
                "h1h4_complete": backend.mediate(
                    base_batch, donor_batch, base_attention, hybrid_attention, None
                ),
                "h1h4_all_sources": backend.mediate(
                    base_batch, donor_batch, base_attention, hybrid_attention, ("all",)
                ),
                "h1h4_last_period_determiner": backend.mediate(
                    base_batch, donor_batch, base_attention, hybrid_attention, discovery.BANK
                ),
                "h1h4_cue_self": backend.mediate(
                    base_batch, donor_batch, base_attention, hybrid_attention, ("cue", "self")
                ),
            }
            forward_calls += 4
            evaluations += 4 * len(chunk)
            for complete, all_sources in zip(
                outputs["h1h4_complete"].answer_foil,
                outputs["h1h4_all_sources"].answer_foil,
            ):
                all_to_complete_max_abs = max(
                    all_to_complete_max_abs,
                    abs(complete[0] - all_sources[0]),
                    abs(complete[1] - all_sources[1]),
                )
            for arm, output in outputs.items():
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    row_id = str(row["row_id"])
                    recovery = kernel.signed_pairwise_donor_recovery(
                        -native[(row_id, "base")].margin,
                        native[(row_id, "donor")].margin,
                        -(answer - foil),
                    )
                    arm_values[arm][family].append(recovery)
                    raw_records.append({
                        "arm_id": arm,
                        "family": family,
                        "row_id": row_id,
                        "answer_logit": answer,
                        "foil_logit": foil,
                        "recovery": recovery,
                    })

    current_capability = True
    for family in ("A1", "A2"):
        for direction in ("past_to_present", "present_to_past"):
            cell_rows = [
                row for row in rows
                if row["transform_id"] == family and row["direction_id"] == direction
            ]
            for side in ("base", "donor"):
                accuracy = sum(
                    native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows
                ) / len(cell_rows)
                current_capability = current_capability and accuracy >= 0.85

    summaries = {}
    targets = {}
    for arm in ARMS:
        families = {
            family: summarize(arm_values[arm][family]) for family in ("A1", "A2")
        }
        target = statistics.fmean(
            families[family]["mean_recovery"] for family in ("A1", "A2")
        )
        summaries[arm] = {"families": families, "mean_target_recovery": target}
        targets[arm] = target
    writer = targets["writer_two_term"]
    complete = targets["h1h4_complete"]
    all_sources = targets["h1h4_all_sources"]
    bank = targets["h1h4_last_period_determiner"]
    cue_self = targets["h1h4_cue_self"]
    complete_to_writer = complete / writer
    bank_to_all = bank / all_sources
    cue_self_to_all_abs = abs(cue_self) / abs(all_sources)

    bound_cells = holdout_result["score"]["capability_cells"]
    pred_a = (
        all(cell["passed"] for cell in bound_cells)
        and current_capability
        and manual_empty_max_abs <= 1.0e-4
        and tensor_error_max_abs <= 2.0e-3
        and attention_reconstruction_max_abs <= 1.0e-4
        and all_to_complete_max_abs <= 0.125
    )
    pred_b = abs(writer - 0.2835613798233539) <= 0.01 and all(
        summaries["writer_two_term"]["families"][family]["mean_recovery"] > 0.0
        and summaries["writer_two_term"]["families"][family]["direction_fraction"] >= 0.80
        for family in ("A1", "A2")
    )
    pred_c = complete > 0.0 and complete_to_writer >= 0.25 and all(
        summaries["h1h4_complete"]["families"][family]["mean_recovery"] > 0.0
        and summaries["h1h4_complete"]["families"][family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")
    )
    pred_d = bank_to_all >= 0.80 and cue_self_to_all_abs <= 0.25 and all(
        summaries["h1h4_last_period_determiner"]["families"][family]["mean_recovery"] > 0.0
        and summaries["h1h4_last_period_determiner"]["families"][family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")
    )
    pred_e = (
        len(raw_records) == len(ARMS) * len(rows)
        and forward_calls <= MODEL_FORWARDS_MAX
        and evaluations <= EXAMPLE_EVALUATIONS_MAX
    )
    terminal = (
        "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e))
        else "null" if pred_a and pred_b and pred_e
        else "invalid"
    )
    reason = {
        "screen": "prospective_attention9_h1h4_bank_transfer",
        "null": "prospective_attention9_transfer_or_specificity_failed",
        "invalid": "authority_capability_instrument_writer_or_coverage_invalid",
    }[terminal]
    result = {
        "schema": "aspectual_anchor_attention9_h1h4_lexical_holdout_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "builder_sha256": EXPECTED_BUILDER_SHA256,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "holdout_result_sha256": EXPECTED_HOLDOUT_RESULT_SHA256,
        "discovery_result_sha256": EXPECTED_DISCOVERY_RESULT_SHA256,
        "program_release_sha256": EXPECTED_PROGRAM_RELEASE_SHA256,
        "dryrun": dryrun,
        "predictions": {
            "pred_a_authority_capability_and_instrument": pred_a,
            "pred_b_writer_recurrence": pred_b,
            "pred_c_h1h4_transfer": pred_c,
            "pred_d_bank_identity_and_specificity": pred_d,
            "pred_e_exact_coverage": pred_e,
        },
        "score": {
            "manual_empty_hook_scored_logit_max_abs": manual_empty_max_abs,
            "bilinear_tensor_reconstruction_max_abs": tensor_error_max_abs,
            "attention_source_reconstruction_max_abs": attention_reconstruction_max_abs,
            "all_sources_to_complete_h1h4_scored_logit_max_abs": all_to_complete_max_abs,
            "arms": summaries,
            "complete_h1h4_to_writer_fraction": complete_to_writer,
            "bank_to_all_h1h4_fraction": bank_to_all,
            "cue_self_absolute_all_h1h4_fraction": cue_self_to_all_abs,
            "forward_calls": forward_calls,
            "example_evaluations": evaluations,
            "raw_record_count": len(raw_records),
            "model_backwards": 0,
            "model_updates": 0,
            "fit_parameters": 0,
        },
        "intervention_logits": raw_records,
        "terminal": terminal,
        "reason": reason,
        "next_action": (
            "promote downstream branch in typed program and test remaining blocks6-8 accumulation"
            if terminal == "screen"
            else "retain attention9 branch as discovery-only and factor blocks6-8"
        ),
    }
    from circuit_fast_screen_managed_runner import atomic_create_json
    atomic_create_json(OUT, result)
    print(json.dumps({
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "reason": reason,
        "predictions": result["predictions"],
        "writer": writer,
        "h1h4_complete": complete,
        "complete_to_writer": complete_to_writer,
        "bank_to_all": bank_to_all,
        "cue_self_to_all_abs": cue_self_to_all_abs,
        "result": str(OUT),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
