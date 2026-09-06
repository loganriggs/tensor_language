#!/usr/bin/env python3
"""Prospective fresh-construction transfer of complete scored-logit program v7."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_and_native_capability pred_b_writer_transfer pred_c_program_v7_transfer pred_d_program_retention pred_e_exact_coverage
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import aspectual_anchor_transparent_path_program_v7 as program
import circuit_battery_integration_contract as battery
import circuit_candidate_aspectual_fresh_construction_v2 as fresh
import circuit_fast_screen_producer as producer
import circuit_fast_screen_spec as screen
import run_aspectual_anchor_mlp11_15_bilinear_compression_split_v2 as mlp_parent
import run_aspectual_anchor_mlp12_14_bilinear_compression_split_v1 as empirical
import run_aspectual_anchor_suffix_depth_adaptive_factorial_lexical_holdout_v1 as suffix
import run_circuit_fast_screen_aspectual as parent_runner
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_program_v7_fresh_construction_transfer_v1.json"
PROGRAM = ROOT / "ops/aspectual_anchor_transparent_path_program_v7.py"
RELEASE = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v7_result.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_fresh_construction_v2.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_program_v7_fresh_construction_transfer_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.program_v7_fresh_construction_transfer_v1"
EXPECTED_PRIOR_SHA256 = "1fc8cf013974236f64350a281a84644043546bda6b72455929c1d127bb01e465"
EXPECTED = {
    PROGRAM: "492d178d11c5e461074a0e154aa52c0dca212a846761e2678d448ae3e4f02d48",
    RELEASE: "52b1f0350ecc185380fca5f141a4f96d414a3a88a94aaf3f39e9fba475be5b36",
    BUILDER: "848332a12c22bf523573e015b6f8f0a38b5865db8b77434dcbe6a176d98370ac",
}
EXPECTED_ROWS_SHA256 = "3c30019fdcc087c0e7410cd82d02458307bc6987ff9d23349dcf97d076f797d7"
MODEL_FORWARDS_MAX = 28
EXAMPLE_EVALUATIONS_MAX = 224


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict[str, object]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError("recovery missing or nonfinite")
    return {"count": len(values), "mean_recovery": statistics.fmean(values), "mean_absolute_recovery": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    release = json.loads(RELEASE.read_text())
    rows = fresh.build_rows()
    if prior.get("candidate_id") != CANDIDATE_ID or release.get("terminal") != "release" or fresh.validate_rows(rows) != EXPECTED_ROWS_SHA256:
        raise ExperimentError("authority changed")
    parent_rows = parent_runner.candidate.build_rows(parent_runner.candidate.TASK_ID)
    parent_spec = parent_runner.build_spec(parent_rows)
    spec = replace(
        parent_spec, experiment_id="aspectual-anchor-program-v7-fresh-construction-transfer-v1",
        authority_sha256=EXPECTED_ROWS_SHA256, expected_fit_rows=len(rows),
        declared_max_price=battery.ExactPhasePrice(phase="FIT", forward_calls=MODEL_FORWARDS_MAX, example_evaluations=EXAMPLE_EVALUATIONS_MAX, backward_calls=0, model_updates=0, evidence_bytes=65536),
    )
    enriched = screen.validate_fit_authority(spec, rows)
    rows = tuple(enriched[str(row["row_id"])] for row in rows)
    if len(rows) != 64 or {row["transform_id"] for row in rows} != {"A1", "A2", "P", "C"}:
        raise ExperimentError("population changed")
    return rows, spec


def main() -> None:
    rows, spec = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_program_v7_fresh_construction_transfer_dryrun_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows": len(rows), "target_rows": 32,
        "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = empirical.component_parent.utc_now(), time.perf_counter()
    backend = empirical.BilinearSuffixBackend.load("cuda")
    native = {}
    writer_values = {family: [] for family in ("A1", "A2")}
    program_values = {family: [] for family in ("A1", "A2")}
    records = []
    alignment_checks = manual_base_max_abs = writer_tensor_error_max_abs = 0
    forward_calls = evaluations = 0

    for family in ("A1", "A2", "P", "C"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in producer._chunks(family_rows, spec.batch_size):
            base_batch = producer._batch(spec, chunk, "base")
            donor_batch = producer._batch(spec, chunk, "donor")
            base_native, base_bilinear = backend.capture_bilinear(base_batch)
            donor_native, donor_bilinear = backend.capture_bilinear(donor_batch)
            forward_calls += 2
            evaluations += 2 * len(chunk)
            for side, output in (("base", base_native), ("donor", donor_native)):
                for row, pair in zip(chunk, output.answer_foil):
                    answer, foil = producer._finite_pair(pair)
                    native[(str(row["row_id"]), side)] = producer.NativeLogitEvidence(str(row["row_id"]), family, side, answer, foil)
            if family not in ("A1", "A2"):
                continue

            role_banks = backend.role_positions(base_batch, donor_batch)
            alignment_checks += len(role_banks)
            base_manual, base_capture = backend.capture_suffix_heads(base_batch)
            writer_output, hybrid_capture, writer_error = backend.capture_writer_suffix_heads(base_batch, donor_batch, base_bilinear, donor_bilinear)
            forward_calls += 2
            evaluations += 2 * len(chunk)
            writer_tensor_error_max_abs = max(writer_tensor_error_max_abs, writer_error)
            for native_pair, manual_pair in zip(base_native.answer_foil, base_manual.answer_foil):
                manual_base_max_abs = max(manual_base_max_abs, abs(native_pair[0] - manual_pair[0]), abs(native_pair[1] - manual_pair[1]))
            for row, pair in zip(chunk, writer_output.answer_foil):
                answer, foil, value = suffix.recovery(row, pair, native)
                writer_values[family].append(value)
                records.append({"arm_id": "writer_two_term", "family": family, "row_id": str(row["row_id"]), "answer_logit": answer, "foil_logit": foil, "recovery": value})

            attention_delta = {}
            for boundary in program.SUFFIX_SOURCE_BOUNDARIES:
                bp, bv, _be = backend.attention_terms(base_batch, base_capture, boundary)
                hp, hv, _he = backend.attention_terms(base_batch, hybrid_capture, boundary)
                attention_delta[boundary] = backend.projected_source_delta(base_batch, role_banks, (bp, bv), (hp, hv), boundary, mlp_parent.SOURCE_BANK_BY_BOUNDARY[boundary])
            mlp_states = {boundary: (*backend.mlp_states(base_capture, boundary), *backend.mlp_states(hybrid_capture, boundary)) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
            for i, (row, query) in enumerate(zip(chunk, base_batch.semantic_positions)):
                states = {boundary: tuple(value[i, query].float() for value in mlp_states[boundary]) for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY}
                answer_tensor, foil_tensor = program.compiled_sparse_suffix_scored_pair(
                    base_capture["resid18"][i, query].float(),
                    hybrid_capture["resid10"][i, query].float() - base_capture["resid10"][i, query].float(),
                    backend.model.lm_head, answer_id=base_batch.answer_ids[i], foil_id=base_batch.foil_ids[i],
                    lambda0_by_boundary={boundary: backend.model.transformer.h[boundary].lambdas[0].float() for boundary in program.SUFFIX_BOUNDARIES},
                    source_attention_delta_by_boundary={boundary: attention_delta[boundary][i, query] for boundary in program.SUFFIX_SOURCE_BOUNDARIES},
                    mlp_states_by_boundary=states,
                    down_weight_by_boundary={boundary: backend.model.transformer.h[boundary].mlp.Down.weight.float() for boundary in program.SUFFIX_MLP_FACTORS_BY_BOUNDARY},
                )
                answer, foil, value = suffix.recovery(row, (float(answer_tensor), float(foil_tensor)), native)
                program_values[family].append(value)
                records.append({"arm_id": "program_v7", "family": family, "row_id": str(row["row_id"]), "answer_logit": answer, "foil_logit": foil, "recovery": value})
            forward_calls += 1
            evaluations += len(chunk)

    capability_cells = []
    capability = True
    for family in ("A1", "A2", "P", "C"):
        for direction in sorted({row["direction_id"] for row in rows if row["transform_id"] == family}):
            cell_rows = [row for row in rows if row["transform_id"] == family and row["direction_id"] == direction]
            correct = sum(native[(str(row["row_id"]), side)].margin > 0.0 for row in cell_rows for side in ("base", "donor"))
            count = 2 * len(cell_rows)
            threshold = 0.75 if family == "C" else 0.85
            passed = correct / count >= threshold
            capability = capability and passed
            capability_cells.append({"family": family, "direction_id": direction, "correct_count": correct, "expected_count": count, "accuracy": correct / count, "minimum_accuracy": threshold, "passed": passed})
    writer_summary = {family: summarize(writer_values[family]) for family in ("A1", "A2")}
    program_summary = {family: summarize(program_values[family]) for family in ("A1", "A2")}
    writer_mean = statistics.fmean(writer_summary[family]["mean_recovery"] for family in ("A1", "A2"))
    program_mean = statistics.fmean(program_summary[family]["mean_recovery"] for family in ("A1", "A2"))
    retention = {family: program_summary[family]["mean_recovery"] / writer_summary[family]["mean_recovery"] for family in ("A1", "A2")}
    retention["pooled"] = program_mean / writer_mean
    pred_a = capability and alignment_checks == 32 and manual_base_max_abs <= 1.0e-4 and writer_tensor_error_max_abs <= 2.0e-3
    pred_b = all(writer_summary[family]["mean_recovery"] > 0.0 and writer_summary[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_c = all(program_summary[family]["mean_recovery"] > 0.0 and program_summary[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = all(value >= 0.75 for value in retention.values())
    pred_e = len(records) == 64 and len({(record["arm_id"], record["row_id"]) for record in records}) == 64 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX
    terminal = "screen" if all((pred_a, pred_b, pred_c, pred_d, pred_e)) else ("null" if pred_a and pred_e else "invalid")
    reason = {"screen": "program_v7_transfers_to_two_fresh_constructions", "null": "capable_fresh_panel_but_writer_or_program_transfer_fails", "invalid": "authority_alignment_capability_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_program_v7_fresh_construction_transfer_result_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": empirical.component_parent.utc_now(), "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "program_sha256": EXPECTED[PROGRAM], "rows_sha256": EXPECTED_ROWS_SHA256,
        "predictions": {"pred_a_authority_alignment_and_native_capability": pred_a, "pred_b_writer_transfer": pred_b, "pred_c_program_v7_transfer": pred_c, "pred_d_program_retention": pred_d, "pred_e_exact_coverage": pred_e},
        "score": {"capability_cells": capability_cells, "alignment_checks": alignment_checks, "writer": writer_summary, "program_v7": program_summary, "writer_mean_recovery": writer_mean, "program_mean_recovery": program_mean, "program_to_writer_retention": retention, "manual_base_scored_logit_max_abs": manual_base_max_abs, "writer_bilinear_tensor_reconstruction_max_abs": writer_tensor_error_max_abs, "forward_calls": forward_calls, "example_evaluations": evaluations, "record_count": len(records), "model_backwards": 0, "model_updates": 0, "fit_parameters": 0},
        "intervention_logits": records, "terminal": terminal, "reason": reason,
        "next_action": "release prospective new-construction scope for v7" if terminal == "screen" else "retain lexical-holdout-only scope and inspect failed capability cells",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": result["predictions"], "writer_mean": writer_mean, "program_mean": program_mean, "retention": retention, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
