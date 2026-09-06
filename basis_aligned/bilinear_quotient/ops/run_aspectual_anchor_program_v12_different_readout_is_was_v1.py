#!/usr/bin/env python3
"""Frozen cross-readout test of aspectual program v12 on is/was behavior."""

# BQGATE: EXPERIMENT pred_a_authority_population_capability_and_exact_head pred_b_frozen_cross_readout_program pred_c_is_was_A_transfer pred_d_is_was_P_generalization pred_e_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import aspectual_anchor_transparent_path_program_v12 as program
import circuit_candidate_aspectual_different_readout_is_was_v1 as fresh
import run_aspectual_anchor_base_margin_affine_carrier_actuation_v1 as affine
import run_aspectual_anchor_base_margin_affine_fresh_lexicon_v2 as prospective
import run_aspectual_anchor_resid10_margin_to_carrier_gain_v1 as upstream
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_program_v12_different_readout_is_was_v1.json"
RELEASE = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v12_result.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v1.py"
RANK1 = ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.program_v12_different_readout_is_was_v1"
EXPECTED_PRIOR_SHA256 = "6988608b9875f7f5a27c304be8dea1205d677e500903aac8be0b54efb5ed32f1"
EXPECTED = {
    RELEASE: "ed6afb3455bf5bfeea6e36f65ce33e9199290cb26540ee94da2c42accc785e7c",
    BUILDER: "9032230f40636838a458254f083b1434b455e8a1be046a6930ed76dd99b68645",
    RANK1: "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
}
EXPECTED_ROWS_SHA256 = "548a92d15a783d90582de837c009c0f25acde38dabb7399874cd413c4867aeeb"


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direction_for(row, family):
    if family in ("A1", "A2"):
        return row["direction_id"]
    return "present_to_past" if row["group_number"] % 2 == 0 else "past_to_present"


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    _old_rows, screen_spec, rank1 = prospective.validate_static()
    rows = fresh.build_rows()
    prior = json.loads(PRIOR.read_text())
    release = json.loads(RELEASE.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or release.get("terminal") != "release"
        or release.get("program_sha256") != sha(ROOT / "ops/aspectual_anchor_transparent_path_program_v12.py")
        or fresh.validate_rows(rows) != EXPECTED_ROWS_SHA256
        or len(rows) != 64
        or rank1["basis"]["sha256"] != prior["authority"]["rank1_basis_sha256"]
    ):
        raise ExperimentError("candidate, release, rows, or basis changed")
    return rows, screen_spec, rank1


def main() -> None:
    rows, screen_spec, rank1 = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_program_v12_different_readout_is_was_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "rows_sha256": EXPECTED_ROWS_SHA256,
        "rows": 64,
        "gain_readout": "fixed_has_had_at_resid10",
        "scored_readout": "is_was_at_resid18",
        "capability_policy": {"A_P": 0.85, "C": 0.75, "all_rows_retained": True},
        "counted_forwards_max": 21,
        "example_evaluations_max": 328,
        "selected_head_pair_evaluations": 224,
        "grid_evaluations": 0,
        "model_backwards": 0,
        "model_updates": 0,
        "inherited_fit_parameters": 4,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = affine.parent.scalar_parent.empirical.component_parent.utc_now(), time.perf_counter()
    backend = affine.parent.producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    q = torch.tensor(rank1["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    if q.shape != (1152,) or abs(float(q.norm()) - 1.0) > 1.0e-4 or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != rank1["basis"]["sha256"]:
        raise ExperimentError("basis reconstruction failed")
    head_ok, head_error = affine.parent.das.verify_head(backend, [r for r in rows if r["family"] == "A1"][:8], "resid:18")
    counted_forwards, evaluations, pair_evaluations = 1, 8, 0
    target_scale = float(rank1["score"]["families"]["target_scale"])
    capability_records, records = [], []

    for family in ("A1", "A2", "P", "C"):
        family_rows = [row for row in rows if row["family"] == family]
        for chunk in affine.parent.producer._chunks(family_rows, screen_spec.batch_size):
            base10, donor10, _ = affine.parent.das.capture_site(backend, chunk, "resid:10")
            base18, donor18, _ = affine.parent.das.capture_site(backend, chunk, "resid:18")
            counted_forwards += 4
            evaluations += 4 * len(chunk)
            for i, row in enumerate(chunk):
                direction = direction_for(row, family)
                source10 = donor10[i] if family == "P" else base10[i]
                source18 = donor18[i] if family == "P" else base18[i]
                contrast = program.intermediate_unembedding_contrast(source10, backend.model.lm_head, direction=direction)
                alpha = program.predict_carrier_gain(source10, backend.model.lm_head, direction=direction)
                if family in ("A1", "A2"):
                    base_pair = affine.parent.pair_logits(backend, base18[i], row["donor_answer_id"], row["base_answer_id"])
                    donor_pair = affine.parent.pair_logits(backend, donor18[i], row["donor_answer_id"], row["base_answer_id"])
                    base_margin = base_pair[0] - base_pair[1]
                    donor_margin = donor_pair[0] - donor_pair[1]
                    answer_id, foil_id = row["donor_answer_id"], row["base_answer_id"]
                    capability_records.extend([
                        {"family": family, "direction": direction, "side": "base", "correct": base_margin < 0.0},
                        {"family": family, "direction": direction, "side": "donor", "correct": donor_margin > 0.0},
                    ])
                    pair_evaluations += 4
                elif family == "P":
                    native = affine.parent.pair_logits(backend, source18, row["base_answer_id"], row["base_foil_id"])
                    native_margin = native[0] - native[1]
                    base_margin, donor_margin = -native_margin, None
                    answer_id, foil_id = row["base_foil_id"], row["base_answer_id"]
                    capability_records.append({"family": family, "direction": direction, "side": "source", "correct": native_margin > 0.0})
                    pair_evaluations += 3
                else:
                    native = affine.parent.pair_logits(backend, source18, row["base_answer_id"], row["base_foil_id"])
                    base_margin, donor_margin = native[0] - native[1], None
                    answer_id, foil_id = row["base_answer_id"], row["base_foil_id"]
                    capability_records.append({"family": family, "direction": direction, "side": "actual_base", "correct": base_margin > 0.0})
                    pair_evaluations += 3
                patched_pair = affine.parent.pair_logits(backend, source18 + alpha * q, answer_id, foil_id)
                patched_margin = patched_pair[0] - patched_pair[1]
                record = {"family": family, "row_id": str(row["row_id"]), "direction": direction, "resid10_has_had_contrast": float(contrast), "alpha": float(alpha), "base_margin": base_margin, "patched_margin": patched_margin, "gain_used_is_was_ids": False, "confirmation_resid18_margin_used_to_select_alpha": False, "confirmation_donor_activation_used_to_select_alpha": False, "outcomes_used_to_select_alpha": False}
                if family in ("A1", "A2"):
                    record["donor_reference_margin"] = donor_margin
                    record["recovery"] = (patched_margin - base_margin) / (donor_margin - base_margin)
                elif family == "P":
                    record["margin_reflection_fraction"] = (patched_margin - base_margin) / (-2.0 * base_margin)
                else:
                    record["normalized_unrelated_effect"] = abs(patched_margin - base_margin) / target_scale
                records.append(record)
            counted_forwards += 1
            evaluations += len(chunk)

    summaries = {
        "A1": upstream.summarize([r for r in records if r["family"] == "A1"], "recovery"),
        "A2": upstream.summarize([r for r in records if r["family"] == "A2"], "recovery"),
        "P": upstream.summarize([r for r in records if r["family"] == "P"], "margin_reflection_fraction"),
        "C": upstream.summarize([r for r in records if r["family"] == "C"], "normalized_unrelated_effect"),
    }
    capability_cells = []
    for family in ("A1", "A2", "P", "C"):
        for direction in sorted({r["direction"] for r in capability_records if r["family"] == family}):
            cell = [r for r in capability_records if r["family"] == family and r["direction"] == direction]
            accuracy = sum(r["correct"] for r in cell) / len(cell)
            threshold = 0.75 if family == "C" else 0.85
            capability_cells.append({"family": family, "direction": direction, "correct": sum(r["correct"] for r in cell), "total": len(cell), "accuracy": accuracy, "threshold": threshold, "passed": accuracy >= threshold})
    actuator_ok = all(math.isfinite(r["alpha"]) and math.isclose(r["alpha"], float(program.GAIN_COEFFICIENTS[r["direction"]]["intercept"] + program.GAIN_COEFFICIENTS[r["direction"]]["slope"] * r["resid10_has_had_contrast"]), rel_tol=1.0e-6, abs_tol=1.0e-3) and not r["gain_used_is_was_ids"] and not r["confirmation_resid18_margin_used_to_select_alpha"] and not r["confirmation_donor_activation_used_to_select_alpha"] and not r["outcomes_used_to_select_alpha"] for r in records)
    pred_a = all(cell["passed"] for cell in capability_cells) and head_ok and head_error <= 1.0e-3
    pred_b = actuator_ok
    pred_c = all(summaries[f]["mean_recovery"] >= 0.25 and summaries[f]["direction_fraction"] >= 0.75 for f in ("A1", "A2"))
    pred_d = summaries["P"]["mean_margin_reflection_fraction"] >= 0.25 and summaries["P"]["direction_fraction"] >= 0.75
    pred_e = summaries["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(records) == 64 and len({r["row_id"] for r in records}) == 64 and counted_forwards <= 21 and evaluations <= 328 and pair_evaluations == 224
    predictions = {"pred_a_authority_population_capability_and_exact_head": pred_a, "pred_b_frozen_cross_readout_program": pred_b, "pred_c_is_was_A_transfer": pred_c, "pred_d_is_was_P_generalization": pred_d, "pred_e_C_selectivity": pred_e, "pred_f_exact_coverage_and_price": pred_f}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "v12_aspectual_circuit_reuses_across_is_was_readout", "null": "v12_aspectual_circuit_is_has_had_readout_specific", "invalid": "authority_population_capability_head_basis_gain_or_coverage_invalid"}[terminal]
    value = {"schema": "aspectual_anchor_program_v12_different_readout_is_was_result_v1", "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": affine.parent.scalar_parent.empirical.component_parent.utc_now(), "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS_SHA256, "basis_sha256": rank1["basis"]["sha256"], "gain_readout": {"tokens": "has/had", "site": "resid:10"}, "scored_readout": {"tokens": "is/was", "site": "resid:18"}, "coefficients": program.GAIN_COEFFICIENTS, "capability_cells": capability_cells, "head_control": {"passed": head_ok, "max_abs_difference": head_error}, "predictions": predictions, "score": {"families": summaries, "control_normalization_scale": target_scale, "counted_forwards": counted_forwards, "example_evaluations": evaluations, "selected_head_pair_evaluations": pair_evaluations, "grid_evaluations": 0, "record_count": len(records), "model_backwards": 0, "model_updates": 0, "inherited_fit_parameters": 4}, "intervention_records": records, "terminal": terminal, "reason": reason, "evidence_scope": "fresh_different_auxiliary_readout_population", "next_action": "confirm cross-readout reuse on a second fresh corpus" if terminal == "screen" else "retain v12 has/had-specific scope and identify the is/was circuit independently"}
    payload = atomic_create_json(OUT, value)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "capability": capability_cells, "families": summaries, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
