#!/usr/bin/env python3
"""Opposing-command composition test for released q_has and q_is programs."""

# BQGATE: EXPERIMENT pred_a_authority_basis_capability_and_exact_heads pred_b_frozen_opposed_program_identity_and_own_route pred_c_opposed_has_had_program_preservation pred_d_opposed_is_was_program_preservation pred_e_additive_law_and_live_opposition pred_f_exact_coverage_and_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import aspectual_anchor_transparent_path_program_v12 as has_program
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_rank1_donor_free_margin_reflection_v1 as head
import run_aspectual_tense_joint_upstream_program_composition_v1 as aligned
import tense_auxiliary_is_was_transparent_path_program_v1 as is_program


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_opposed_command_program_composition_v1.json"
ALIGNED_RESULT = ROOT / "circuits/followups/aspectual_tense_joint_upstream_program_composition_v1_result.json"
ALIGNED_RUNNER = ROOT / "ops/run_aspectual_tense_joint_upstream_program_composition_v1.py"
OUT = ROOT / "circuits/followups/aspectual_tense_opposed_command_program_composition_v1_result.json"
CANDIDATE_ID = "aspectual_tense.opposed_command_program_composition_v1"
EXPECTED_PRIOR_SHA256 = "fc146ebb281ffe9925bdc51d3d1b5e18598b7632c4e3bb2d78b0a16312ab8e2c"
EXPECTED_AUTHORITIES = {
    "aligned_joint_result_sha256": "46479986f81751af6141e8fcbaf19d4413198b119171711715414d2869f43e08",
    "aligned_joint_runner_sha256": "60feb93111876d8075b6baadb6411693d57540c31fe93f780bd892f6cd419659",
    "q_has_program_release_sha256": "ed6afb3455bf5bfeea6e36f65ce33e9199290cb26540ee94da2c42accc785e7c",
    "q_is_program_release_sha256": "9804a0d0f047f194f6cce3490828c3a6e9525940f8c2b822467bc52176957e98",
    "q_has_prospective_screen_sha256": "a4cee1818acd6a28e999eda26d0447f94d080b913f4061d0e6dab4914cb3802c",
    "q_is_prospective_screen_sha256": "dad39b298a0e89e5e0271149574012ebf7f75e995b83f6ede12ebe3b1aa746e8",
}


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def opposite(direction):
    if direction == "present_to_past":
        return "past_to_present"
    if direction == "past_to_present":
        return "present_to_past"
    raise ExperimentError(f"unknown direction: {direction}")


def summarize(records, key):
    values = [record[key] for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError(f"missing/nonfinite {key}")
    return {"count": len(values), f"mean_{key}": statistics.fmean(values), f"mean_absolute_{key}": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    prior = json.loads(PRIOR.read_text())
    observed = {
        "aligned_joint_result_sha256": sha(ALIGNED_RESULT),
        "aligned_joint_runner_sha256": sha(ALIGNED_RUNNER),
        "q_has_program_release_sha256": sha(aligned.PATHS["q_has_program_release"]),
        "q_is_program_release_sha256": sha(aligned.PATHS["q_is_program_release"]),
        "q_has_prospective_screen_sha256": sha(aligned.PATHS["q_has_prospective_screen"]),
        "q_is_prospective_screen_sha256": sha(aligned.PATHS["q_is_prospective_screen"]),
    }
    rows, loaded = aligned.validate_static()
    aligned_result = json.loads(ALIGNED_RESULT.read_text())
    parent_required = tuple("pred_" + suffix for suffix in (
        "a_authority_capability_basis_and_exact_heads", "b_frozen_dual_program_identity_and_own_route",
        "c_joint_has_had_program_preservation", "d_joint_is_was_program_preservation", "f_exact_coverage_and_price",
    ))
    if prior.get("candidate_id") != CANDIDATE_ID or prior["authorities"] != observed or observed != EXPECTED_AUTHORITIES or aligned_result.get("terminal") != "null" or not all(aligned_result["predictions"][key] for key in parent_required):
        raise ExperimentError("candidate, authorities, aligned result, or parent validation changed")
    return rows, loaded, aligned_result


def main():
    rows_by_bank, loaded, aligned_result = validate_static()
    plan = {
        "schema": "aspectual_tense_opposed_command_program_composition_dryrun_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": aligned.EXPECTED_ROWS, "rows": 128,
        "model_forwards_exact": 4, "example_evaluations_exact": 256, "inherited_gain_scalars": 8,
        "basis_scalars": 2304, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    q_has = torch.as_tensor(loaded["q_has_basis_result"]["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    q_is = torch.as_tensor(loaded["q_is_basis_result"]["basis"]["values_column_major"], device=backend.device, dtype=torch.float32)
    basis_ok = q_has.shape == q_is.shape == (1152,) and hashlib.sha256(q_has.cpu().numpy().tobytes()).hexdigest() == has_program.program_manifest()["rank1_basis_sha256"] and hashlib.sha256(q_is.cpu().numpy().tobytes()).hexdigest() == is_program.BASIS_SHA256
    aligned_by_row = {(record["bank"], record["row_id"]): record for record in aligned_result["intervention_records"]}
    outputs_by_bank, cells_by_bank = {}, {}
    capability_match = True
    for bank, rows in rows_by_bank.items():
        outputs = {side: backend.native(das._batch(backend, rows, side=side), capture=True) for side in ("base", "donor")}
        outputs_by_bank[bank] = outputs
        cells = aligned.capability_cells(bank, rows, outputs)
        cells_by_bank[bank] = cells
        expected_cells = loaded[f"q_{'has' if bank == 'has_had' else 'is'}_prospective_screen"]["capability_cells"]
        capability_match = capability_match and cells == expected_cells and all(cell["passed"] for cell in cells)

    scales = {"has_had": float(loaded["q_has_prospective_screen"]["score"]["target_scale"]), "is_was": float(loaded["q_is_basis_result"]["score"]["families"]["target_scale"])}
    records = []
    local_error = 0.0
    final_error = 0.0
    for bank, rows in rows_by_bank.items():
        outputs = outputs_by_bank[bank]
        for i, row in enumerate(rows):
            family, direction = row["family"], aligned.direction_for(row)
            partner_direction = opposite(direction)
            side = "donor" if family == "P" else "base"
            source10 = torch.as_tensor(outputs[side].captured[(row["row_id"], "resid:10")], device=backend.device).float()
            source18 = torch.as_tensor(outputs[side].captured[(row["row_id"], "resid:18")], device=backend.device).float()
            donor18 = torch.as_tensor(outputs["donor"].captured[(row["row_id"], "resid:18")], device=backend.device).float()
            own_program = has_program if bank == "has_had" else is_program
            partner_program = is_program if bank == "has_had" else has_program
            own_q, partner_q = (q_has, q_is) if bank == "has_had" else (q_is, q_has)
            own_gain_fn = has_program.predict_carrier_gain if bank == "has_had" else is_program.predict_writer_gain
            partner_gain_fn = is_program.predict_writer_gain if bank == "has_had" else has_program.predict_carrier_gain
            own_alpha = own_gain_fn(source10, backend.model.lm_head, direction=direction)
            partner_alpha = partner_gain_fn(source10, backend.model.lm_head, direction=partner_direction)
            own_contrast = own_program.intermediate_unembedding_contrast(source10, backend.model.lm_head, direction=direction)
            partner_contrast = partner_program.intermediate_unembedding_contrast(source10, backend.model.lm_head, direction=partner_direction)
            full10 = das.head_logits(backend, source10[None, :])
            for program_name, contrast, tokens, command in (("has" if bank == "has_had" else "is", own_contrast, own_program.TOKEN_IDS, direction), ("is" if bank == "has_had" else "has", partner_contrast, partner_program.TOKEN_IDS, partner_direction)):
                present, past = (tokens["has"], tokens["had"]) if program_name == "has" else (tokens["is"], tokens["was"])
                current, other = (present, past) if command == "present_to_past" else (past, present)
                local_error = max(local_error, abs(float(contrast) - float(full10[0, current] - full10[0, other])))
            own_state = source18 + own_alpha * own_q
            partner_state = source18 + partner_alpha * partner_q
            joint_state = own_state + partner_alpha * partner_q
            if family == "C":
                target, foil = row["base_answer_id"], row["base_foil_id"]
            else:
                target, foil = aligned.requested_ids(bank, direction)
            margins = {name: float(head.selected_margin(backend, state[None, :], [target], [foil])[0]) for name, state in (("base", source18), ("own", own_state), ("partner", partner_state), ("joint", joint_state))}
            full18 = das.head_logits(backend, source18[None, :])
            final_error = max(final_error, abs(margins["base"] - float(full18[0, target] - full18[0, foil])))
            scale = scales[bank]
            aligned_record = aligned_by_row[(bank, str(row["row_id"]))]
            aligned_partner_delta = aligned_record["margins"]["partner"] - aligned_record["margins"]["base"]
            opposed_partner_delta = margins["partner"] - margins["base"]
            record = {
                "bank": bank, "family": family, "row_id": str(row["row_id"]), "own_direction": direction, "partner_direction": partner_direction,
                "own_alpha": float(own_alpha), "partner_alpha": float(partner_alpha), "own_contrast": float(own_contrast), "partner_contrast": float(partner_contrast),
                "margins": margins, "aligned_partner_delta": aligned_partner_delta, "opposed_partner_delta": opposed_partner_delta,
                "partner_effect_reversed": aligned_partner_delta * opposed_partner_delta < 0.0,
                "additive_residual_normalized": abs((margins["joint"] - margins["base"]) - (margins["own"] - margins["base"]) - opposed_partner_delta) / scale,
                "gain_refit": False, "basis_changed": False, "donor_or_outcome_used_to_select_gain": False,
            }
            if family in ("A1", "A2"):
                donor_reference = float(head.selected_margin(backend, donor18[None, :], [target], [foil])[0])
                denominator = donor_reference - margins["base"]
                record["donor_reference_margin"] = donor_reference
                record["own_recovery"] = (margins["own"] - margins["base"]) / denominator
                record["joint_recovery"] = (margins["joint"] - margins["base"]) / denominator
            elif family == "P":
                record["own_margin_reflection_fraction"] = (margins["own"] - margins["base"]) / (-2.0 * margins["base"])
                record["joint_margin_reflection_fraction"] = (margins["joint"] - margins["base"]) / (-2.0 * margins["base"])
            else:
                record["own_normalized_unrelated_effect"] = abs(margins["own"] - margins["base"]) / scale
                record["joint_normalized_unrelated_effect"] = abs(margins["joint"] - margins["base"]) / scale
            records.append(record)

    summaries = {}
    for bank in ("has_had", "is_was"):
        summaries[bank] = {}
        for family in ("A1", "A2", "P", "C"):
            selected = [record for record in records if record["bank"] == bank and record["family"] == family]
            if family in ("A1", "A2"):
                own, joint = summarize(selected, "own_recovery"), summarize(selected, "joint_recovery")
            elif family == "P":
                own, joint = summarize(selected, "own_margin_reflection_fraction"), summarize(selected, "joint_margin_reflection_fraction")
            else:
                own, joint = summarize(selected, "own_normalized_unrelated_effect"), summarize(selected, "joint_normalized_unrelated_effect")
            summaries[bank][family] = {"own": own, "joint": joint, "mean_additive_residual_normalized": statistics.fmean(record["additive_residual_normalized"] for record in selected), "partner_reversal_fraction": sum(record["partner_effect_reversed"] for record in selected) / len(selected), "mean_opposed_partner_delta_normalized": statistics.fmean(record["opposed_partner_delta"] / scales[bank] for record in selected)}

    own_route_ok = True
    for bank, receipt_key in (("has_had", "q_has_prospective_screen"), ("is_was", "q_is_prospective_screen")):
        expected_families = loaded[receipt_key]["score"]["families"]
        for family in ("A1", "A2", "P", "C"):
            metric = "recovery" if family in ("A1", "A2") else ("margin_reflection_fraction" if family == "P" else "normalized_unrelated_effect")
            own, expected = summaries[bank][family]["own"], expected_families[family]
            own_route_ok = own_route_ok and abs(own[f"mean_own_{metric}"] - expected[f"mean_{metric}"]) <= 1.0e-5 and abs(own[f"mean_absolute_own_{metric}"] - expected[f"mean_absolute_{metric}"]) <= 1.0e-5 and own["direction_fraction"] == expected["direction_fraction"]

    def bank_passes(bank):
        family = summaries[bank]
        return all(family[name]["joint"]["mean_joint_recovery"] >= 0.75 and family[name]["joint"]["direction_fraction"] >= 0.75 for name in ("A1", "A2")) and family["P"]["joint"]["mean_joint_margin_reflection_fraction"] >= 0.75 and family["P"]["joint"]["direction_fraction"] >= 0.75 and family["C"]["joint"]["mean_joint_normalized_unrelated_effect"] <= 0.20

    identity_ok = all(math.isfinite(record["own_alpha"]) and math.isfinite(record["partner_alpha"]) and record["partner_direction"] == opposite(record["own_direction"]) and not record["gain_refit"] and not record["basis_changed"] and not record["donor_or_outcome_used_to_select_gain"] for record in records)
    pred_a = basis_ok and capability_match and local_error <= 1.0e-4 and final_error <= 1.0e-4
    pred_b = identity_ok and own_route_ok
    pred_c = bank_passes("has_had")
    pred_d = bank_passes("is_was")
    pred_e = all(cell["mean_additive_residual_normalized"] <= 0.20 for bank in summaries.values() for cell in bank.values()) and all(summaries[bank][family]["partner_reversal_fraction"] >= 0.75 for bank in summaries for family in ("A1", "A2", "P"))
    pred_f = len(records) == 128 and len({(record["bank"], record["row_id"]) for record in records}) == 128
    predictions = {"pred_a_authority_basis_capability_and_exact_heads": pred_a, "pred_b_frozen_opposed_program_identity_and_own_route": pred_b, "pred_c_opposed_has_had_program_preservation": pred_c, "pred_d_opposed_is_was_program_preservation": pred_d, "pred_e_additive_law_and_live_opposition": pred_e, "pred_f_exact_coverage_and_price": pred_f}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e and pred_f else "invalid")
    reason = {"screen": "released_programs_remain_independently_manipulable_under_opposed_commands", "null": "opposed_partner_command_disrupts_own_program", "invalid": "authority_basis_capability_head_identity_replay_opposition_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_tense_opposed_command_program_composition_result_v1", "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": aligned.EXPECTED_ROWS, "aligned_result_sha256": EXPECTED_AUTHORITIES["aligned_joint_result_sha256"],
        "basis": {"q_has_sha256": has_program.program_manifest()["rank1_basis_sha256"], "q_is_sha256": is_program.BASIS_SHA256, "absolute_cosine": abs(float(q_has @ q_is))},
        "capability_cells": cells_by_bank, "head_controls": {"resid10_local_max_abs_difference": local_error, "resid18_local_max_abs_difference": final_error},
        "summaries": summaries, "intervention_records": records, "predictions": predictions,
        "price": {"model_forwards": 4, "example_evaluations": 256, "rows": len(records), "inherited_gain_scalars": 8, "basis_scalars": 2304, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "terminal": terminal, "reason": reason,
        "next_action": "compile independently composable dual program" if terminal == "screen" else "bound which writer/output family prevents independent commands; do not change rank or gains",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "summaries": summaries, "price": result["price"], "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
