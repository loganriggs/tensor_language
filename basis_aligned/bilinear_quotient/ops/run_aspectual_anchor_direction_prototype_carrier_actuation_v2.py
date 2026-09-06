#!/usr/bin/env python3
"""Fresh test of a two-scalar, direction-conditioned aspectual carrier actuator."""

# BQGATE: EXPERIMENT pred_a_authority_calibration_capability_and_exact_head pred_b_fixed_target_independent_actuator pred_c_fresh_A_prediction pred_d_fresh_P_generalization pred_e_fresh_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import aspectual_anchor_transparent_path_program_v9 as program
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
import run_aspectual_anchor_program_v7_fresh_construction_transfer_v1 as fresh_parent
import run_aspectual_anchor_rank1_scalar_term_compression_split_v1 as scalar_parent
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_direction_prototype_carrier_actuation_v2.json"
RANK1 = ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json"
SCALAR = ROOT / "circuits/followups/aspectual_anchor_rank1_scalar_term_compression_split_v1_result.json"
SCALAR_AUDIT = ROOT / "circuits/followups/aspectual_anchor_rank1_scalar_term_compression_v1_instrument_audit_result.json"
V10_SCREEN = ROOT / "circuits/followups/aspectual_anchor_rank1_donor_free_margin_reflection_v1_result.json"
FRESH_BUILDER = ROOT / "ops/circuit_candidate_aspectual_fresh_construction_v2.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_direction_prototype_carrier_actuation_v2_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.direction_prototype_carrier_actuation_v2"
EXPECTED_PRIOR_SHA256 = "22148654e5c3bc546d9c8687594e5416ed39e3dfeb768b07dba8e45d41b8b879"
EXPECTED = {
    RANK1: "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
    SCALAR: "4a55ef3da37b12722fabae41c9caaa7e8284fc0891ba4e15c5cfdeab40323b2d",
    SCALAR_AUDIT: "001a0ba6cc8d87fab6922b3541a1e9b4d77b38144c71fcd644f87e0dd82df130",
    V10_SCREEN: "cb1275e1b9449c52254a05efa9932aeaa916a2722c8ee72717231d9121e957fb",
    FRESH_BUILDER: "848332a12c22bf523573e015b6f8f0a38b5865db8b77434dcbe6a176d98370ac",
}
PROTOTYPES = {"present_to_past": -828.5600967407227, "past_to_present": 874.4011383056641}
MODEL_FORWARDS_MAX = 25
EXAMPLE_EVALUATIONS_MAX = 200


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(records, key):
    values = [record[key] for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError(f"missing/nonfinite {key}")
    return {"count": len(values), f"mean_{key}": statistics.fmean(values), f"mean_absolute_{key}": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    scalar = json.loads(SCALAR.read_text())
    audit = json.loads(SCALAR_AUDIT.read_text())
    v10 = json.loads(V10_SCREEN.read_text())
    lexical_rows, _lexical_spec, _fresh_target_rows, _fresh_target_spec, rank1, _reference = scalar_parent.validate_static()
    fresh_rows, fresh_spec = fresh_parent.validate_static()
    direction_by_id = {str(row["row_id"]): row["direction_id"] for row in lexical_rows}
    selection = [record for record in scalar["term_amplitudes"] if record["phase"] == "selection"]
    observed = {direction: statistics.fmean(record["total_amplitude"] for record in selection if direction_by_id[record["row_id"]] == direction) for direction in PROTOTYPES}
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or prior["frozen_design"]["prototypes"] != PROTOTYPES
        or observed != PROTOTYPES
        or scalar.get("terminal") != "invalid"
        or audit.get("terminal") != v10.get("terminal")
        or audit.get("terminal") != "screen"
        or audit.get("scientific_disposition") != "five_term_compression_null"
        or len(fresh_rows) != 64
    ):
        raise ExperimentError("candidate, calibration, terminal, disposition, or population changed")
    return fresh_rows, fresh_spec, rank1, v10


def pair_logits(backend, state, answer_id, foil_id):
    answer, foil = program.exact_scored_pair(state, backend.model.lm_head, answer_id=answer_id, foil_id=foil_id)
    return float(answer), float(foil)


def main() -> None:
    fresh_rows, fresh_spec, rank1, v10 = validate_static()
    dryrun = {
        "schema": "aspectual_anchor_direction_prototype_carrier_actuation_dryrun_v2", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows": 64, "prototypes": PROTOTYPES,
        "confirmation_donor_activation_used": False, "target_guided_alpha_search": False,
        "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
        "selected_head_evaluations": 176, "grid_evaluations": 0, "model_backwards": 0, "model_updates": 0, "fit_parameters": 2,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = scalar_parent.empirical.component_parent.utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    q = torch.tensor(rank1["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    if q.shape != (1152,) or abs(float(q.norm()) - 1.0) > 1.0e-4 or hashlib.sha256(q.cpu().numpy().tobytes()).hexdigest() != rank1["basis"]["sha256"]:
        raise ExperimentError("basis reconstruction failed")
    a1_rows = [row for row in fresh_rows if row["transform_id"] == "A1"]
    head_ok, head_error = das.verify_head(backend, a1_rows[:8], "resid:18")
    forward_calls, evaluations, head_evaluations = 1, 8, 0
    target_scale = float(rank1["score"]["families"]["target_scale"])
    records, capability = [], True

    for family in ("A1", "A2", "P", "C"):
        rows = [row for row in fresh_rows if row["transform_id"] == family]
        for chunk in producer._chunks(rows, fresh_spec.batch_size):
            base, donor, _ = das.capture_site(backend, chunk, "resid:18")
            forward_calls += 2
            evaluations += 2 * len(chunk)
            for i, row in enumerate(chunk):
                direction = row["direction_id"] if family in ("A1", "A2") else ("present_to_past" if row["group_number"] % 2 == 0 else "past_to_present")
                alpha = PROTOTYPES[direction]
                if family in ("A1", "A2"):
                    source = base[i]
                    answer_id, foil_id = row["donor_answer_id"], row["donor_foil_id"]
                    base_original = pair_logits(backend, base[i], row["base_answer_id"], row["base_foil_id"])
                    donor_original = pair_logits(backend, donor[i], row["donor_answer_id"], row["donor_foil_id"])
                    capability = capability and base_original[0] > base_original[1] and donor_original[0] > donor_original[1]
                    base_target = (base_original[1], base_original[0])
                    donor_target = donor_original
                    head_evaluations += 3
                elif family == "P":
                    source = donor[i]
                    answer_id, foil_id = row["base_foil_id"], row["base_answer_id"]
                    original = pair_logits(backend, source, foil_id, answer_id)
                    capability = capability and original[0] > original[1]
                    base_target, donor_target = (original[1], original[0]), None
                    head_evaluations += 2
                else:
                    source = base[i]
                    answer_id, foil_id = row["base_answer_id"], row["base_foil_id"]
                    original = pair_logits(backend, source, answer_id, foil_id)
                    donor_original = pair_logits(backend, donor[i], row["donor_answer_id"], row["donor_foil_id"])
                    capability = capability and original[0] > original[1] and donor_original[0] > donor_original[1]
                    base_target, donor_target = original, None
                    head_evaluations += 3
                patched = pair_logits(backend, source + alpha * q, answer_id, foil_id)
                base_margin, patched_margin = base_target[0] - base_target[1], patched[0] - patched[1]
                record = {"family": family, "row_id": str(row["row_id"]), "direction": direction, "alpha": alpha, "base_margin": base_margin, "patched_margin": patched_margin, "confirmation_donor_activation_used_by_actuator": False, "target_or_foil_used_to_select_alpha": False}
                if family in ("A1", "A2"):
                    donor_margin = donor_target[0] - donor_target[1]
                    record["donor_reference_margin"] = donor_margin
                    record["recovery"] = (patched_margin - base_margin) / (donor_margin - base_margin)
                elif family == "P":
                    record["margin_reflection_fraction"] = (patched_margin - base_margin) / (-2.0 * base_margin)
                else:
                    record["normalized_unrelated_effect"] = abs(patched_margin - base_margin) / target_scale
                records.append(record)
            forward_calls += 1
            evaluations += len(chunk)

    by_family = {family: [record for record in records if record["family"] == family] for family in ("A1", "A2", "P", "C")}
    summaries = {"A1": summarize(by_family["A1"], "recovery"), "A2": summarize(by_family["A2"], "recovery"), "P": summarize(by_family["P"], "margin_reflection_fraction"), "C": summarize(by_family["C"], "normalized_unrelated_effect")}
    v10_means = {"A1": v10["score"]["panels"]["fresh_A1"]["mean_recovery"], "A2": v10["score"]["panels"]["fresh_A2"]["mean_recovery"]}
    versus_v10 = {family: summaries[family]["mean_recovery"] / v10_means[family] for family in ("A1", "A2")}
    pred_a = capability and head_ok and head_error <= 1.0e-3
    pred_b = len(PROTOTYPES) == 2 and all(math.isfinite(value) and value != 0.0 for value in PROTOTYPES.values()) and all(record["confirmation_donor_activation_used_by_actuator"] is False and record["target_or_foil_used_to_select_alpha"] is False and record["alpha"] == PROTOTYPES[record["direction"]] for record in records)
    pred_c = all(summaries[family]["mean_recovery"] >= 0.50 and summaries[family]["direction_fraction"] >= 0.75 and versus_v10[family] >= 0.50 for family in ("A1", "A2"))
    pred_d = summaries["P"]["mean_margin_reflection_fraction"] >= 0.50 and summaries["P"]["direction_fraction"] >= 0.75
    pred_e = summaries["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_f = len(records) == 64 and len({record["row_id"] for record in records}) == 64 and forward_calls <= MODEL_FORWARDS_MAX and evaluations <= EXAMPLE_EVALUATIONS_MAX and head_evaluations == 176
    predictions = {"pred_a_authority_calibration_capability_and_exact_head": pred_a, "pred_b_fixed_target_independent_actuator": pred_b, "pred_c_fresh_A_prediction": pred_c, "pred_d_fresh_P_generalization": pred_d, "pred_e_fresh_C_selectivity": pred_e, "pred_f_exact_coverage_and_price": pred_f}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "two_direction_scalars_predict_fresh_selective_carrier_actuation", "null": "fixed_direction_prototypes_fail_fresh_prediction_or_selectivity", "invalid": "authority_calibration_capability_head_actuator_or_coverage_invalid"}[terminal]
    result = {"schema": "aspectual_anchor_direction_prototype_carrier_actuation_result_v2", "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": scalar_parent.empirical.component_parent.utc_now(), "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256, "basis_sha256": rank1["basis"]["sha256"], "prototypes": PROTOTYPES, "head_control": {"passed": head_ok, "max_abs_difference": head_error}, "predictions": predictions, "score": {"families": summaries, "recovery_fraction_vs_v10": versus_v10, "target_scale": target_scale, "forward_calls": forward_calls, "example_evaluations": evaluations, "selected_head_evaluations": head_evaluations, "grid_evaluations": 0, "record_count": len(records), "model_backwards": 0, "model_updates": 0, "fit_parameters": 2}, "intervention_records": records, "terminal": terminal, "reason": reason, "next_action": "compile two-scalar target-independent actuator into program v12" if terminal == "screen" else "retain v10 target-guided actuator and test upstream variable prototypes"}
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "families": summaries, "versus_v10": versus_v10, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
