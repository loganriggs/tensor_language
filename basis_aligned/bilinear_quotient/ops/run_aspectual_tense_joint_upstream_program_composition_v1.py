#!/usr/bin/env python3
"""Joint composition test for the released q_has and q_is programs."""

# BQGATE: EXPERIMENT pred_a_authority_capability_basis_and_exact_heads pred_b_frozen_dual_program_identity_and_own_route pred_c_joint_has_had_program_preservation pred_d_joint_is_was_program_preservation pred_e_additive_composition_and_bounded_partner_surcharge pred_f_exact_coverage_and_price
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
import circuit_candidate_aspectual_fresh_lexicon_v5 as has_rows_builder
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6 as is_rows_builder
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_rank1_donor_free_margin_reflection_v1 as head
import tense_auxiliary_is_was_transparent_path_program_v1 as is_program


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_joint_upstream_program_composition_v1.json"
OUT = ROOT / "circuits/followups/aspectual_tense_joint_upstream_program_composition_v1_result.json"
CANDIDATE_ID = "aspectual_tense.joint_upstream_program_composition_v1"
PATHS = {
    "q_has_program_release": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v12_result.json",
    "q_is_program_release": ROOT / "circuits/followups/tense_auxiliary_is_was_transparent_path_program_release_v1_result.json",
    "q_has_prospective_screen": ROOT / "circuits/followups/aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v2_result.json",
    "q_is_prospective_screen": ROOT / "circuits/followups/tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_v2_result.json",
    "q_has_basis_result": ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json",
    "q_is_basis_result": ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json",
    "projected_response_factor_audit": ROOT / "circuits/followups/aspectual_tense_projected_response_factor_audit_v1_result.json",
    "q_has_builder": ROOT / "ops/circuit_candidate_aspectual_fresh_lexicon_v5.py",
    "q_is_builder": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6.py",
}
EXPECTED_PRIOR_SHA256 = "ff2f69035a1d5e6eaea29062cd17b001140368fbf8af2e2567d1fd567d2c6d62"
EXPECTED = {
    "q_has_program_release": "ed6afb3455bf5bfeea6e36f65ce33e9199290cb26540ee94da2c42accc785e7c",
    "q_is_program_release": "9804a0d0f047f194f6cce3490828c3a6e9525940f8c2b822467bc52176957e98",
    "q_has_prospective_screen": "a4cee1818acd6a28e999eda26d0447f94d080b913f4061d0e6dab4914cb3802c",
    "q_is_prospective_screen": "dad39b298a0e89e5e0271149574012ebf7f75e995b83f6ede12ebe3b1aa746e8",
    "q_has_basis_result": "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
    "q_is_basis_result": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "projected_response_factor_audit": "28a9a49db15e2676d7c2a0a1c6b3ca039dc359fb91fdb47994cda2eb6a85bfc0",
    "q_has_builder": "ae624913c5adfe07cf028acf6549cd5fe2debd4b090c71659218fe158089fe2c",
    "q_is_builder": "b8541360334bd2793a02fae525a94dda05ce600fd4de5b6c3d953063d4c6b0ae",
}
EXPECTED_ROWS = {"has_had": "296c2186f477a6d450bbbb87fda5ba89b999eb4d3ac0dc18e31496ca47d5caf7", "is_was": "4eee90d9f39f6997c4926a0e7f6baecc4134c06535fe307d0a38f936b75defd5"}


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direction_for(row):
    return row["direction_id"] if row["family"] in ("A1", "A2") else ("present_to_past" if row["group_number"] % 2 == 0 else "past_to_present")


def requested_ids(bank, direction):
    tokens = has_program.TOKEN_IDS if bank == "has_had" else is_program.TOKEN_IDS
    present, past = (tokens["has"], tokens["had"]) if bank == "has_had" else (tokens["is"], tokens["was"])
    return (past, present) if direction == "present_to_past" else (present, past)


def summarize(records, key):
    values = [record[key] for record in records]
    if not values or any(not math.isfinite(value) for value in values):
        raise ExperimentError(f"missing/nonfinite {key}")
    return {"count": len(values), f"mean_{key}": statistics.fmean(values), f"mean_absolute_{key}": statistics.fmean(abs(value) for value in values), "direction_fraction": sum(value > 0.0 for value in values) / len(values)}


def capability_cells(bank, rows, outputs):
    records = []
    for i, row in enumerate(rows):
        direction = direction_for(row)
        sides = ("base", "donor") if bank == "is_was" or row["family"] in ("A1", "A2") else (("donor",) if row["family"] == "P" else ("base",))
        for side in sides:
            answer, foil = outputs[side].answer_foil[i]
            records.append({"family": row["family"], "direction": direction, "side": side, "correct": float(answer) > float(foil)})
    cells = []
    for family in ("A1", "A2", "P", "C"):
        for direction in ("past_to_present", "present_to_past"):
            selected = [record for record in records if record["family"] == family and record["direction"] == direction]
            correct = sum(record["correct"] for record in selected)
            threshold = 0.75 if family == "C" else 0.85
            accuracy = correct / len(selected)
            cells.append({"family": family, "direction": direction, "correct": correct, "total": len(selected), "accuracy": accuracy, "threshold": threshold, "passed": accuracy >= threshold})
    return cells


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    observed = {name: sha(path) for name, path in PATHS.items()}
    if observed != EXPECTED:
        raise ExperimentError("authority hash changed")
    prior = json.loads(PRIOR.read_text())
    loaded = {name: json.loads(path.read_text()) for name, path in PATHS.items() if "builder" not in name}
    rows = {"has_had": has_rows_builder.build_rows(), "is_was": is_rows_builder.build_rows()}
    ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and prior["authorities"] == {
            **{f"{key}_sha256": value for key, value in EXPECTED.items()},
            "q_has_rows_sha256": EXPECTED_ROWS["has_had"],
            "q_is_rows_sha256": EXPECTED_ROWS["is_was"],
        }
        and has_rows_builder.validate_rows(rows["has_had"]) == EXPECTED_ROWS["has_had"]
        and is_rows_builder.validate_rows(rows["is_was"]) == EXPECTED_ROWS["is_was"]
        and all(len(bank_rows) == 64 for bank_rows in rows.values())
        and loaded["q_has_program_release"].get("terminal") == "release"
        and loaded["q_is_program_release"].get("terminal") == "release"
        and loaded["q_has_prospective_screen"].get("terminal") == "screen"
        and loaded["q_is_prospective_screen"].get("terminal") == "screen"
        and loaded["q_has_basis_result"]["basis"]["sha256"] == has_program.program_manifest()["rank1_basis_sha256"]
        and loaded["q_is_basis_result"]["basis"]["sha256"] == is_program.BASIS_SHA256
    )
    if not ok:
        raise ExperimentError("candidate, rows, releases, screens, or bases changed")
    return rows, loaded


def main():
    rows_by_bank, loaded = validate_static()
    plan = {
        "schema": "aspectual_tense_joint_upstream_program_composition_dryrun_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS, "rows": 128,
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
    basis_ok = (
        q_has.shape == q_is.shape == (1152,)
        and hashlib.sha256(q_has.cpu().numpy().tobytes()).hexdigest() == has_program.program_manifest()["rank1_basis_sha256"]
        and hashlib.sha256(q_is.cpu().numpy().tobytes()).hexdigest() == is_program.BASIS_SHA256
    )
    outputs_by_bank = {}
    cells_by_bank = {}
    capability_match = True
    for bank, rows in rows_by_bank.items():
        outputs = {side: backend.native(das._batch(backend, rows, side=side), capture=True) for side in ("base", "donor")}
        outputs_by_bank[bank] = outputs
        cells = capability_cells(bank, rows, outputs)
        cells_by_bank[bank] = cells
        capability_match = capability_match and cells == loaded[f"q_{'has' if bank == 'has_had' else 'is'}_prospective_screen"]["capability_cells"] and all(cell["passed"] for cell in cells)

    records = []
    local_error = 0.0
    final_error = 0.0
    scales = {
        "has_had": float(loaded["q_has_prospective_screen"]["score"]["target_scale"]),
        "is_was": float(loaded["q_is_basis_result"]["score"]["families"]["target_scale"]),
    }
    for bank, rows in rows_by_bank.items():
        outputs = outputs_by_bank[bank]
        for i, row in enumerate(rows):
            family, direction = row["family"], direction_for(row)
            side = "donor" if family == "P" else "base"
            source10 = torch.as_tensor(outputs[side].captured[(row["row_id"], "resid:10")], device=backend.device).float()
            source18 = torch.as_tensor(outputs[side].captured[(row["row_id"], "resid:18")], device=backend.device).float()
            donor18 = torch.as_tensor(outputs["donor"].captured[(row["row_id"], "resid:18")], device=backend.device).float()
            alpha_has = has_program.predict_carrier_gain(source10, backend.model.lm_head, direction=direction)
            alpha_is = is_program.predict_writer_gain(source10, backend.model.lm_head, direction=direction)
            has_contrast = has_program.intermediate_unembedding_contrast(source10, backend.model.lm_head, direction=direction)
            is_contrast = is_program.intermediate_unembedding_contrast(source10, backend.model.lm_head, direction=direction)
            full10 = das.head_logits(backend, source10[None, :])
            for program_name, contrast, tokens in (("has", has_contrast, has_program.TOKEN_IDS), ("is", is_contrast, is_program.TOKEN_IDS)):
                present, past = (tokens["has"], tokens["had"]) if program_name == "has" else (tokens["is"], tokens["was"])
                current, other = (present, past) if direction == "present_to_past" else (past, present)
                local_error = max(local_error, abs(float(contrast) - float(full10[0, current] - full10[0, other])))
            own_q, own_alpha = (q_has, alpha_has) if bank == "has_had" else (q_is, alpha_is)
            partner_q, partner_alpha = (q_is, alpha_is) if bank == "has_had" else (q_has, alpha_has)
            own_state = source18 + own_alpha * own_q
            partner_state = source18 + partner_alpha * partner_q
            joint_state = own_state + partner_alpha * partner_q
            if family == "C":
                target, foil = row["base_answer_id"], row["base_foil_id"]
            else:
                target, foil = requested_ids(bank, direction)
            margins = {
                "base": float(head.selected_margin(backend, source18[None, :], [target], [foil])[0]),
                "own": float(head.selected_margin(backend, own_state[None, :], [target], [foil])[0]),
                "partner": float(head.selected_margin(backend, partner_state[None, :], [target], [foil])[0]),
                "joint": float(head.selected_margin(backend, joint_state[None, :], [target], [foil])[0]),
            }
            full18 = das.head_logits(backend, source18[None, :])
            final_error = max(final_error, abs(margins["base"] - float(full18[0, target] - full18[0, foil])))
            scale = scales[bank]
            record = {
                "bank": bank, "family": family, "row_id": str(row["row_id"]), "direction": direction,
                "alpha_has": float(alpha_has), "alpha_is": float(alpha_is), "has_contrast": float(has_contrast), "is_contrast": float(is_contrast),
                "margins": margins,
                "additive_residual_normalized": abs((margins["joint"] - margins["base"]) - (margins["own"] - margins["base"]) - (margins["partner"] - margins["base"])) / scale,
                "partner_surcharge_normalized": abs(margins["joint"] - margins["own"]) / scale,
                "partner_signed_response_normalized": (margins["partner"] - margins["base"]) / scale,
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
                own = summarize(selected, "own_recovery")
                joint = summarize(selected, "joint_recovery")
            elif family == "P":
                own = summarize(selected, "own_margin_reflection_fraction")
                joint = summarize(selected, "joint_margin_reflection_fraction")
            else:
                own = summarize(selected, "own_normalized_unrelated_effect")
                joint = summarize(selected, "joint_normalized_unrelated_effect")
            summaries[bank][family] = {
                "own": own, "joint": joint,
                "mean_additive_residual_normalized": statistics.fmean(record["additive_residual_normalized"] for record in selected),
                "mean_partner_surcharge_normalized": statistics.fmean(record["partner_surcharge_normalized"] for record in selected),
                "mean_partner_signed_response_normalized": statistics.fmean(record["partner_signed_response_normalized"] for record in selected),
            }

    own_route_ok = True
    for bank, receipt_key in (("has_had", "q_has_prospective_screen"), ("is_was", "q_is_prospective_screen")):
        expected_families = loaded[receipt_key]["score"]["families"]
        for family in ("A1", "A2", "P", "C"):
            own = summaries[bank][family]["own"]
            expected = expected_families[family]
            metric = "recovery" if family in ("A1", "A2") else ("margin_reflection_fraction" if family == "P" else "normalized_unrelated_effect")
            own_route_ok = own_route_ok and abs(own[f"mean_own_{metric}"] - expected[f"mean_{metric}"]) <= 1.0e-5 and abs(own[f"mean_absolute_own_{metric}"] - expected[f"mean_absolute_{metric}"]) <= 1.0e-5 and own["direction_fraction"] == expected["direction_fraction"]

    identity_ok = all(
        math.isfinite(record["alpha_has"]) and math.isfinite(record["alpha_is"])
        and not record["gain_refit"] and not record["basis_changed"] and not record["donor_or_outcome_used_to_select_gain"]
        for record in records
    )
    def joint_bank_passes(bank):
        family = summaries[bank]
        return (
            all(family[name]["joint"]["mean_joint_recovery"] >= 0.75 and family[name]["joint"]["direction_fraction"] >= 0.75 for name in ("A1", "A2"))
            and family["P"]["joint"]["mean_joint_margin_reflection_fraction"] >= 0.75
            and family["P"]["joint"]["direction_fraction"] >= 0.75
            and family["C"]["joint"]["mean_joint_normalized_unrelated_effect"] <= 0.20
        )
    pred_a = basis_ok and capability_match and local_error <= 1.0e-4 and final_error <= 1.0e-4
    pred_b = identity_ok and own_route_ok
    pred_c = joint_bank_passes("has_had")
    pred_d = joint_bank_passes("is_was")
    pred_e = all(cell["mean_additive_residual_normalized"] <= 0.20 and cell["mean_partner_surcharge_normalized"] <= 0.50 for bank in summaries.values() for cell in bank.values())
    pred_f = len(records) == 128 and len({(record["bank"], record["row_id"]) for record in records}) == 128
    predictions = {
        "pred_a_authority_capability_basis_and_exact_heads": pred_a,
        "pred_b_frozen_dual_program_identity_and_own_route": pred_b,
        "pred_c_joint_has_had_program_preservation": pred_c,
        "pred_d_joint_is_was_program_preservation": pred_d,
        "pred_e_additive_composition_and_bounded_partner_surcharge": pred_e,
        "pred_f_exact_coverage_and_price": pred_f,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_f else "invalid")
    reason = {"screen": "released_q_has_q_is_programs_compose_predictably", "null": "joint_preservation_or_composition_law_misses", "invalid": "authority_capability_basis_head_identity_replay_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_tense_joint_upstream_program_composition_result_v1", "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "serial_seconds": time.perf_counter() - started,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS,
        "basis": {"q_has_sha256": has_program.program_manifest()["rank1_basis_sha256"], "q_is_sha256": is_program.BASIS_SHA256, "absolute_cosine": abs(float(q_has @ q_is))},
        "capability_cells": cells_by_bank, "head_controls": {"resid10_local_max_abs_difference": local_error, "resid18_local_max_abs_difference": final_error},
        "summaries": summaries, "intervention_records": records, "predictions": predictions,
        "price": {"model_forwards": 4, "example_evaluations": 256, "rows": len(records), "inherited_gain_scalars": 8, "basis_scalars": 2304, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "terminal": terminal, "reason": reason,
        "next_action": "test joint selective removals and compile a dual-program composition manifest" if terminal == "screen" else "classify the family and direction of interference without changing rank or gains",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "basis_cosine": result["basis"]["absolute_cosine"], "summaries": summaries, "price": result["price"], "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
