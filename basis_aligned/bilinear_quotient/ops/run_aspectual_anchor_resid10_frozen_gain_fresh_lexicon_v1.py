#!/usr/bin/env python3
"""Prospective second-lexicon validation of frozen resid10-to-carrier gain."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_capability_and_local_exactness pred_b_frozen_upstream_program pred_c_new_lexicon_A_prediction pred_d_new_lexicon_P_generalization pred_e_new_lexicon_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_aspectual_fresh_lexicon_v4 as fresh
import run_aspectual_anchor_base_margin_affine_carrier_actuation_v1 as affine
import run_aspectual_anchor_base_margin_affine_fresh_lexicon_v2 as prospective
import run_aspectual_anchor_resid10_margin_to_carrier_gain_v1 as upstream
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v1.json"
FACTORY = ROOT / "ops/circuit_candidate_aspectual_lexicon_factory.py"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_fresh_lexicon_v4.py"
UPSTREAM_RESULT = ROOT / "circuits/followups/aspectual_anchor_resid10_margin_to_carrier_gain_v1_result.json"
UPSTREAM_AUDIT = ROOT / "circuits/followups/aspectual_anchor_resid10_margin_to_carrier_gain_v1_instrument_audit_result.json"
DOSE = ROOT / "circuits/followups/aspectual_anchor_affine_carrier_dose_response_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.resid10_frozen_gain_fresh_lexicon_v1"
EXPECTED_PRIOR_SHA256 = "5e0c2b8274baa09de52329ba881115aba205b87ecd2f1c1fb00a7e38f54cb74c"
EXPECTED = {
    FACTORY: "50b5ba3158d45cdf8996ac032f5b1b289aaa11a2542a16a3ab1d2c5fbccc3ccd",
    BUILDER: "b26a92c0314c15e3b0c9f48fd4278bfe2d70355c3f5fce208f6164701f0cc87c",
    UPSTREAM_RESULT: "e7a9fbb41a3330bdf7cc7f8e13545ccb3773cb803ca47250f87691f355d2839a",
    UPSTREAM_AUDIT: "4b8a8ac7e0bba304153544be3e9709bf902b59fabb7a1e2a1e0ab65e43a1ff93",
    DOSE: "4b3463aa095c03801a41453411ae663d14dad8340be6083ca42ff51264ec9a4f",
}
EXPECTED_ROWS_SHA256 = "13b81d8e9cdde30c2b31c8ffeb01d997d282269d155a124dd44b27439042e773"
COEFFICIENTS = {
    "present_to_past": {"intercept": 3049.77917349804, "slope": -1881.1152323328579},
    "past_to_present": {"intercept": 3575.0380871196844, "slope": 1181.9271936643886},
}
COUNTED_FORWARDS_MAX = 25
EXAMPLE_EVALUATIONS_MAX = 328


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manual_pair(backend, state, first_id, second_id):
    normalized = backend.F.rms_norm(state, (backend.model.config.n_embd,))
    first_raw = (normalized * backend.model.lm_head.weight[first_id]).sum()
    second_raw = (normalized * backend.model.lm_head.weight[second_id]).sum()
    return float(30.0 * backend.torch.tanh(first_raw / 30.0)), float(30.0 * backend.torch.tanh(second_raw / 30.0))


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    _old_rows, screen_spec, rank1 = prospective.validate_static()
    rows = fresh.build_rows()
    prior = json.loads(PRIOR.read_text())
    upstream_result = json.loads(UPSTREAM_RESULT.read_text())
    upstream_audit = json.loads(UPSTREAM_AUDIT.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or prior["frozen_program"]["coefficients"] != COEFFICIENTS
        or prior["frozen_program"]["fixed_token_ids"] != upstream.TOKEN_IDS
        or fresh.validate_rows(rows) != EXPECTED_ROWS_SHA256
        or len(rows) != 64
        or upstream_result.get("terminal") != "invalid"
        or upstream_audit.get("scientific_disposition") != "resid10_unembedding_contrast_to_carrier_gain_screen"
        or any(abs(upstream_result["fits"][direction][key] - COEFFICIENTS[direction][key]) > 1.0e-12 for direction in COEFFICIENTS for key in ("intercept", "slope"))
    ):
        raise ExperimentError("candidate, frozen program, rows, upstream disposition, or coefficients changed")
    return rows, screen_spec, rank1


def main() -> None:
    rows, screen_spec, rank1 = validate_static()
    dryrun = {"schema": "aspectual_anchor_resid10_frozen_gain_fresh_lexicon_dryrun_v1", "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only", "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS_SHA256, "rows": 64, "feature_site": "resid:10", "write_site": "resid:18", "coefficients": COEFFICIENTS, "fixed_token_ids": upstream.TOKEN_IDS, "confirmation_resid18_margin_used_to_select_alpha": False, "confirmation_donor_activation_used_to_select_alpha": False, "counted_forwards_max": COUNTED_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX, "selected_head_pair_evaluations": 224, "grid_evaluations": 0, "model_backwards": 0, "model_updates": 0, "inherited_fit_parameters": 4}
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
    head_ok, head_error = affine.parent.das.verify_head(backend, [row for row in rows if row["transform_id"] == "A1"][:8], "resid:18")
    counted_forwards, evaluations, pair_evaluations = 1, 8, 0
    local_error = 0.0
    target_scale = float(rank1["score"]["families"]["target_scale"])
    capability_counts = {"A_base": [0, 0], "A_donor": [0, 0], "P_source": [0, 0], "C_actual_base": [0, 0]}
    records = []

    for family in ("A1", "A2", "P", "C"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        for chunk in affine.parent.producer._chunks(family_rows, screen_spec.batch_size):
            base10, donor10, _ = affine.parent.das.capture_site(backend, chunk, "resid:10")
            base18, donor18, _ = affine.parent.das.capture_site(backend, chunk, "resid:18")
            counted_forwards += 4
            evaluations += 4 * len(chunk)
            for i, row in enumerate(chunk):
                direction = affine.direction_for(row, family)
                source10 = donor10[i] if family == "P" else base10[i]
                source18 = donor18[i] if family == "P" else base18[i]
                current_id = upstream.TOKEN_IDS["has"] if direction == "present_to_past" else upstream.TOKEN_IDS["had"]
                other_id = upstream.TOKEN_IDS["had"] if direction == "present_to_past" else upstream.TOKEN_IDS["has"]
                shared_pair = affine.parent.pair_logits(backend, source10, current_id, other_id)
                manual = manual_pair(backend, source10, current_id, other_id)
                local_error = max(local_error, abs(shared_pair[0] - manual[0]), abs(shared_pair[1] - manual[1]))
                contrast10 = shared_pair[0] - shared_pair[1]
                alpha = COEFFICIENTS[direction]["intercept"] + COEFFICIENTS[direction]["slope"] * contrast10
                if family in ("A1", "A2"):
                    base_original = affine.parent.pair_logits(backend, base18[i], current_id, other_id)
                    donor_original = affine.parent.pair_logits(backend, donor18[i], other_id, current_id)
                    base_margin = -(base_original[0] - base_original[1])
                    donor_margin = donor_original[0] - donor_original[1]
                    answer_id, foil_id = other_id, current_id
                    capability_counts["A_base"][0] += int(base_margin < 0.0)
                    capability_counts["A_base"][1] += 1
                    capability_counts["A_donor"][0] += int(donor_margin > 0.0)
                    capability_counts["A_donor"][1] += 1
                    pair_evaluations += 4
                elif family == "P":
                    original = affine.parent.pair_logits(backend, source18, current_id, other_id)
                    current_margin = original[0] - original[1]
                    base_margin, donor_margin = -current_margin, None
                    answer_id, foil_id = other_id, current_id
                    capability_counts["P_source"][0] += int(current_margin > 0.0)
                    capability_counts["P_source"][1] += 1
                    pair_evaluations += 3
                else:
                    original = affine.parent.pair_logits(backend, source18, row["base_answer_id"], row["base_foil_id"])
                    base_margin, donor_margin = original[0] - original[1], None
                    answer_id, foil_id = row["base_answer_id"], row["base_foil_id"]
                    capability_counts["C_actual_base"][0] += int(base_margin > 0.0)
                    capability_counts["C_actual_base"][1] += 1
                    pair_evaluations += 3
                patched = affine.parent.pair_logits(backend, source18 + alpha * q, answer_id, foil_id)
                patched_margin = patched[0] - patched[1]
                record = {"family": family, "row_id": str(row["row_id"]), "direction": direction, "resid10_unembedding_contrast": contrast10, "alpha": alpha, "base_margin": base_margin, "patched_margin": patched_margin, "confirmation_resid18_margin_used_to_select_alpha": False, "confirmation_donor_activation_used_to_select_alpha": False, "row_target_or_foil_used_to_select_alpha": False}
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

    by_family = {family: [record for record in records if record["family"] == family] for family in ("A1", "A2", "P", "C")}
    summaries = {"A1": upstream.summarize(by_family["A1"], "recovery"), "A2": upstream.summarize(by_family["A2"], "recovery"), "P": upstream.summarize(by_family["P"], "margin_reflection_fraction"), "C": upstream.summarize(by_family["C"], "normalized_unrelated_effect")}
    capability = all(correct == total for correct, total in capability_counts.values())
    actuator_ok = all(math.isfinite(record["alpha"]) and record["alpha"] == COEFFICIENTS[record["direction"]]["intercept"] + COEFFICIENTS[record["direction"]]["slope"] * record["resid10_unembedding_contrast"] and not record["confirmation_resid18_margin_used_to_select_alpha"] and not record["confirmation_donor_activation_used_to_select_alpha"] and not record["row_target_or_foil_used_to_select_alpha"] for record in records)
    pred_a = capability and local_error <= 1.0e-4 and head_ok and head_error <= 1.0e-3
    pred_b = actuator_ok and len(COEFFICIENTS) == 2 and sum(len(values) for values in COEFFICIENTS.values()) == 4
    pred_c = all(summaries[family]["mean_recovery"] >= 0.75 and summaries[family]["direction_fraction"] >= 0.75 for family in ("A1", "A2"))
    pred_d = summaries["P"]["mean_margin_reflection_fraction"] >= 0.75 and summaries["P"]["direction_fraction"] >= 0.75
    pred_e = summaries["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(records) == 64 and len({record["row_id"] for record in records}) == 64 and counted_forwards <= COUNTED_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX and pair_evaluations == 224
    predictions = {"pred_a_authority_novelty_capability_and_local_exactness": pred_a, "pred_b_frozen_upstream_program": pred_b, "pred_c_new_lexicon_A_prediction": pred_c, "pred_d_new_lexicon_P_generalization": pred_d, "pred_e_new_lexicon_C_selectivity": pred_e, "pred_f_exact_coverage_and_price": pred_f}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "frozen_resid10_read_compute_write_predicts_second_lexicon", "null": "frozen_resid10_program_fails_A_P_prediction_or_C_selectivity", "invalid": "authority_novelty_capability_local_exactness_head_program_or_coverage_invalid"}[terminal]
    result = {"schema": "aspectual_anchor_resid10_frozen_gain_fresh_lexicon_result_v1", "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": affine.parent.scalar_parent.empirical.component_parent.utc_now(), "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS_SHA256, "basis_sha256": rank1["basis"]["sha256"], "feature_site": "resid:10", "write_site": "resid:18", "coefficients": COEFFICIENTS, "fixed_token_ids": upstream.TOKEN_IDS, "local_unembedding_pair_max_abs_difference": local_error, "head_control": {"site": "resid:18", "passed": head_ok, "max_abs_difference": head_error}, "capability_counts": {key: {"correct": value[0], "total": value[1]} for key, value in capability_counts.items()}, "predictions": predictions, "score": {"families": summaries, "target_scale": target_scale, "counted_forwards": counted_forwards, "example_evaluations": evaluations, "selected_head_pair_evaluations": pair_evaluations, "grid_evaluations": 0, "record_count": len(records), "model_backwards": 0, "model_updates": 0, "inherited_fit_parameters": 4}, "intervention_records": records, "terminal": terminal, "reason": reason, "evidence_scope": "prospective_second_lexicon_within_tested_constructions", "next_action": "compile upstream gain map into program v12 and test different-surface or syntax OOD" if terminal == "screen" else "retain final-margin controller and add a second upstream v11 variable"}
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "families": summaries, "local_error": local_error, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
