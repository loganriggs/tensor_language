#!/usr/bin/env python3
"""Prospective matched-lexicon validation of the released raw-text dual program."""

# BQGATE: EXPERIMENT pred_a_authority_release_basis_and_rows pred_b_native_capability_first pred_c_automatic_selector_identity pred_d_prospective_has_had_preservation pred_e_prospective_is_was_preservation pred_f_exact_coverage_and_price
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
import aspectual_tense_raw_text_dual_program_v1 as dual
import circuit_candidate_aspectual_fresh_lexicon_v6 as has_builder
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v7 as is_builder
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_rank1_donor_free_margin_reflection_v1 as head
import tense_auxiliary_is_was_transparent_path_program_v1 as is_program


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_raw_text_dual_program_fresh_lexicon_v1.json"
OUT = ROOT / "circuits/followups/aspectual_tense_raw_text_dual_program_fresh_lexicon_v1_result.json"
CANDIDATE_ID = "aspectual_tense.raw_text_dual_program_fresh_lexicon_v1"
PATHS = {
    "dual_program_result": ROOT / "circuits/followups/aspectual_tense_raw_text_dual_program_v1_result.json",
    "dual_program": ROOT / "ops/aspectual_tense_raw_text_dual_program_v1.py",
    "q_has_basis_result": ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json",
    "q_is_basis_result": ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json",
    "q_has_prospective_result": ROOT / "circuits/followups/aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v2_result.json",
    "q_is_prospective_result": ROOT / "circuits/followups/tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_v2_result.json",
    "q_has_release": ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v12_result.json",
    "q_is_release": ROOT / "circuits/followups/tense_auxiliary_is_was_transparent_path_program_release_v1_result.json",
    "q_has_v6_builder": ROOT / "ops/circuit_candidate_aspectual_fresh_lexicon_v6.py",
    "q_is_v7_builder": ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v7.py",
}
EXPECTED_PRIOR_SHA256 = "bfb62286c6a5fec5f005f14b4d83924743d81277177b31775eda645c27ddf2d8"
EXPECTED = {
    "dual_program_result": "0fb9d071d1ccad8662ab10958ea30db55b3293e1766d3c08cb1296ec03a37336",
    "dual_program": "a756bfbeddaad7db2bb0c7feec1f3a6bd976b05fec36d231865b69e4813a976c",
    "q_has_basis_result": "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
    "q_is_basis_result": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "q_has_prospective_result": "a4cee1818acd6a28e999eda26d0447f94d080b913f4061d0e6dab4914cb3802c",
    "q_is_prospective_result": "dad39b298a0e89e5e0271149574012ebf7f75e995b83f6ede12ebe3b1aa746e8",
    "q_has_release": "ed6afb3455bf5bfeea6e36f65ce33e9199290cb26540ee94da2c42accc785e7c",
    "q_is_release": "9804a0d0f047f194f6cce3490828c3a6e9525940f8c2b822467bc52176957e98",
    "q_has_v6_builder": "9ebc6613df73b9470e562be7e91b4f5b27f7f1e9c111aacf5470a2a7e07377a5",
    "q_is_v7_builder": "985e8e6647b35c29a9e02a6291fd11ea44e49e4113aaece3d029db12f65bb6c0",
}
EXPECTED_ROWS = {"has_had": "a7592b57b40beb3fc75c9e6ef6385367dbbc5d3d3462f93df2cc6f6b79fd9fc2", "is_was": "05847e77b62ba1321256dc3de058c6b05a52344a2d036f53753cc48f05f52f0d"}


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
    observations = []
    for index, row in enumerate(rows):
        for side in ("base", "donor"):
            answer, foil = outputs[side].answer_foil[index]
            observations.append({"family": row["family"], "direction": direction_for(row), "correct": float(answer) > float(foil)})
    cells = []
    for family in ("A1", "A2", "P", "C"):
        for direction in ("past_to_present", "present_to_past"):
            selected = [item for item in observations if item["family"] == family and item["direction"] == direction]
            correct = sum(item["correct"] for item in selected)
            threshold = 0.75 if family == "C" else 0.85
            accuracy = correct / len(selected)
            cells.append({"bank": bank, "family": family, "direction": direction, "correct": correct, "total": len(selected), "accuracy": accuracy, "threshold": threshold, "passed": accuracy >= threshold})
    return cells


def prediction_record(a, b, c, d, e, f):
    return {
        "pred_a_authority_release_basis_and_rows": a, "pred_b_native_capability_first": b,
        "pred_c_automatic_selector_identity": c, "pred_d_prospective_has_had_preservation": d,
        "pred_e_prospective_is_was_preservation": e, "pred_f_exact_coverage_and_price": f,
    }


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    prior = json.loads(PRIOR.read_text())
    loaded = {name: json.loads(path.read_text()) for name, path in PATHS.items() if name not in ("dual_program", "q_has_v6_builder", "q_is_v7_builder")}
    rows = {"has_had": has_builder.build_rows(), "is_was": is_builder.build_rows()}
    row_hashes = {"has_had": has_builder.validate_rows(rows["has_had"]), "is_was": is_builder.validate_rows(rows["is_was"])}
    expected_authorities = {**{f"{name}_sha256": digest for name, digest in EXPECTED.items()}, "q_has_v6_rows_sha256": EXPECTED_ROWS["has_had"], "q_is_v7_rows_sha256": EXPECTED_ROWS["is_was"]}
    ok = (
        prior.get("candidate_id") == CANDIDATE_ID and prior.get("authorities") == expected_authorities
        and row_hashes == EXPECTED_ROWS and all(len(value) == 64 for value in rows.values())
        and set(has_builder._AGENTS) == set(is_builder._AGENTS)
        and loaded["dual_program_result"].get("terminal") == "release"
        and loaded["q_has_release"].get("terminal") == "release" and loaded["q_is_release"].get("terminal") == "release"
        and loaded["q_has_prospective_result"].get("terminal") == "screen" and loaded["q_is_prospective_result"].get("terminal") == "screen"
        and loaded["q_has_basis_result"]["basis"]["sha256"] == has_program.program_manifest()["rank1_basis_sha256"]
        and loaded["q_is_basis_result"]["basis"]["sha256"] == is_program.BASIS_SHA256
    )
    if not ok:
        raise ExperimentError("candidate, release, basis, matched lexicon, or row authority changed")
    return rows, loaded


def main():
    rows_by_bank, loaded = validate_static()
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False,
        "rows": 128, "model_forwards_exact": 4, "example_evaluations_exact": 256,
        "inherited_gain_scalars": 8, "basis_scalars": 2304, "fitted_scalars": 0,
        "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch_module = backend.torch
    outputs_by_bank = {}
    capability_by_bank = {}
    for bank, rows in rows_by_bank.items():
        outputs = {side: backend.native(das._batch(backend, rows, side=side), capture=True) for side in ("base", "donor")}
        outputs_by_bank[bank] = outputs
        capability_by_bank[bank] = capability_cells(bank, rows, outputs)
    capability_pass = all(cell["passed"] for cells in capability_by_bank.values() for cell in cells)
    if not capability_pass:
        predictions = prediction_record(True, False, False, False, False, True)
        result = {
            "schema": "aspectual_tense_raw_text_dual_program_fresh_lexicon_result_v1", "candidate_id": CANDIDATE_ID,
            "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS, "capability_cells": capability_by_bank,
            "causal_outcomes_opened": False, "intervention_records": [], "predictions": predictions,
            "price": {"model_forwards": 4, "example_evaluations": 256, "rows": 128, "intervention_records": 0, "inherited_gain_scalars": 8, "basis_scalars": 2304, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
            "terminal": "invalid", "reason": "native_capability_gate_failed_before_causal_outcomes", "serial_seconds": time.perf_counter() - started,
            "next_action": "retain the invalid lexical authority without inspecting or changing the frozen dual program",
        }
        atomic_create_json(OUT, result)
        print(json.dumps(result, sort_keys=True))
        return

    bases = {
        "has_had": torch_module.as_tensor(loaded["q_has_basis_result"]["basis"]["values_column_major"], device=backend.device, dtype=torch_module.float32),
        "is_was": torch_module.as_tensor(loaded["q_is_basis_result"]["basis"]["values_column_major"], device=backend.device, dtype=torch_module.float32),
    }
    basis_ok = all(basis.shape == (1152,) and abs(float(basis.norm()) - 1.0) <= 1e-4 for basis in bases.values())
    basis_ok = basis_ok and hashlib.sha256(bases["has_had"].cpu().numpy().tobytes()).hexdigest() == has_program.program_manifest()["rank1_basis_sha256"]
    basis_ok = basis_ok and hashlib.sha256(bases["is_was"].cpu().numpy().tobytes()).hexdigest() == is_program.BASIS_SHA256
    scales = {"has_had": float(loaded["q_has_prospective_result"]["score"]["target_scale"]), "is_was": float(loaded["q_is_basis_result"]["score"]["families"]["target_scale"])}
    records = []
    selector_ok, local_error, final_error = True, 0.0, 0.0
    for bank, rows in rows_by_bank.items():
        outputs = outputs_by_bank[bank]
        for index, row in enumerate(rows):
            family, direction = row["family"], direction_for(row)
            side = "donor" if family == "P" else "base"
            source10 = torch_module.as_tensor(outputs[side].captured[(row["row_id"], "resid:10")], device=backend.device).float()
            source18 = torch_module.as_tensor(outputs[side].captured[(row["row_id"], "resid:18")], device=backend.device).float()
            dispatched = dual.actuate(source10, source18, bases, backend.model.lm_head, text=row[f"{side}_text"])
            expected_bank, expected_direction = ("abstain", None) if family == "C" else (bank, direction)
            selector_ok = selector_ok and dispatched["bank"] == expected_bank and dispatched["direction"] == expected_direction
            if family == "C":
                target, foil = row["base_answer_id"], row["base_foil_id"]
            else:
                target, foil = requested_ids(bank, direction)
                tokens = has_program.TOKEN_IDS if bank == "has_had" else is_program.TOKEN_IDS
                present, past = (tokens["has"], tokens["had"]) if bank == "has_had" else (tokens["is"], tokens["was"])
                current, other = (present, past) if direction == "present_to_past" else (past, present)
                full10 = das.head_logits(backend, source10[None, :])
                local_error = max(local_error, abs(float(dispatched["resid10_unembedding_contrast"]) - float(full10[0, current] - full10[0, other])))
            base_margin = float(head.selected_margin(backend, source18[None, :], [target], [foil])[0])
            patched_margin = float(head.selected_margin(backend, dispatched["patched_resid18"][None, :], [target], [foil])[0])
            full18 = das.head_logits(backend, source18[None, :])
            final_error = max(final_error, abs(base_margin - float(full18[0, target] - full18[0, foil])))
            record = {
                "bank": bank, "family": family, "direction": direction, "row_id": str(row["row_id"]), "source_side": side,
                "selected_bank": dispatched["bank"], "selected_direction": dispatched["direction"],
                "base_margin": base_margin, "patched_margin": patched_margin,
                "gain_refit": False, "basis_changed": False, "donor_or_outcome_used_to_select_gain": False,
            }
            if family in ("A1", "A2"):
                donor18 = torch_module.as_tensor(outputs["donor"].captured[(row["row_id"], "resid:18")], device=backend.device).float()
                donor_margin = float(head.selected_margin(backend, donor18[None, :], [target], [foil])[0])
                denominator = donor_margin - base_margin
                if denominator == 0.0:
                    raise ExperimentError("zero A recovery denominator")
                record["donor_reference_margin"] = donor_margin
                record["recovery"] = (patched_margin - base_margin) / denominator
            elif family == "P":
                if base_margin == 0.0:
                    raise ExperimentError("zero P reflection denominator")
                record["margin_reflection_fraction"] = (patched_margin - base_margin) / (-2.0 * base_margin)
            else:
                record["normalized_unrelated_effect"] = abs(patched_margin - base_margin) / scales[bank]
            records.append(record)

    summaries = {}
    for bank in ("has_had", "is_was"):
        summaries[bank] = {}
        for family in ("A1", "A2", "P", "C"):
            selected = [record for record in records if record["bank"] == bank and record["family"] == family]
            key = "recovery" if family in ("A1", "A2") else ("margin_reflection_fraction" if family == "P" else "normalized_unrelated_effect")
            summaries[bank][family] = summarize(selected, key)

    has_pass = all(summaries["has_had"][family][f"mean_{'recovery' if family in ('A1', 'A2') else 'margin_reflection_fraction'}"] >= 0.75 and summaries["has_had"][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2", "P")) and summaries["has_had"]["C"]["mean_normalized_unrelated_effect"] <= 0.20
    is_pass = all(summaries["is_was"][family][f"mean_{'recovery' if family in ('A1', 'A2') else 'margin_reflection_fraction'}"] >= 0.75 and summaries["is_was"][family]["direction_fraction"] >= 0.75 for family in ("A1", "A2", "P")) and summaries["is_was"]["C"]["mean_normalized_unrelated_effect"] <= 0.20
    pred_a, pred_b = basis_ok, capability_pass
    pred_c = selector_ok and local_error <= 1e-4 and final_error <= 1e-4
    pred_d, pred_e = has_pass, is_pass
    pred_f = len(records) == 128 and all(cell["total"] == 16 for cells in capability_by_bank.values() for cell in cells)
    predictions = prediction_record(pred_a, pred_b, pred_c, pred_d, pred_e, pred_f)
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_c and pred_f else "invalid")
    reason = {"screen": "prospective_matched_lexicon_dual_program_supported", "null": "prospective_dual_program_causal_bar_misses", "invalid": "authority_capability_selector_identity_coverage_or_price_invalid"}[terminal]
    result = {
        "schema": "aspectual_tense_raw_text_dual_program_fresh_lexicon_result_v1", "candidate_id": CANDIDATE_ID,
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS, "capability_cells": capability_by_bank,
        "causal_outcomes_opened": True, "basis": {"q_has_sha256": has_program.program_manifest()["rank1_basis_sha256"], "q_is_sha256": is_program.BASIS_SHA256},
        "head_controls": {"local_max_abs_error": local_error, "final_max_abs_error": final_error},
        "summaries": summaries, "intervention_records": records, "predictions": predictions,
        "price": {"model_forwards": 4, "example_evaluations": 256, "rows": 128, "intervention_records": 128, "inherited_gain_scalars": 8, "basis_scalars": 2304, "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "terminal": terminal, "reason": reason, "serial_seconds": time.perf_counter() - started,
        "next_action": "localize an internal task branch while retaining this prospective interface as the simple executable baseline" if terminal == "screen" else "retain the released single-bank programs and do not rescue this matched lexical population",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability_cells", "causal_outcomes_opened", "head_controls", "summaries", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
