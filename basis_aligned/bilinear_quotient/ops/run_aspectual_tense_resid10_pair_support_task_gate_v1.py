#!/usr/bin/env python3
"""Prospective fit-free resid:10 answer-pair support task-gate screen."""

# BQGATE: EXPERIMENT pred_a_authority_basis_rows_and_shared_contract pred_b_native_capability_first pred_c_fixed_pair_support_task_routing pred_d_affinity_dispatched_causal_preservation pred_e_raw_text_control_abstention pred_f_exact_identity_coverage_and_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import aspectual_anchor_transparent_path_program_v12 as has_program
import aspectual_tense_dual_eval as evaluator
import aspectual_tense_raw_text_dual_program_v1 as raw_program
import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2 as rows_builder
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_aspectual_anchor_rank1_donor_free_margin_reflection_v1 as head
import tense_auxiliary_is_was_transparent_path_program_v1 as is_program


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_resid10_pair_support_task_gate_v1.json"
OUT = ROOT / "circuits/followups/aspectual_tense_resid10_pair_support_task_gate_v1_result.json"
CANDIDATE_ID = "aspectual_tense.resid10_pair_support_task_gate_v1"
PATHS = {
    "shared_eval": ROOT / "ops/aspectual_tense_dual_eval.py",
    "matched_v2_builder": ROOT / "ops/circuit_candidate_aspectual_tense_matched_fresh_lexicon_v2.py",
    "raw_text_dual_program": ROOT / "ops/aspectual_tense_raw_text_dual_program_v1.py",
    "prospective_dual_screen": ROOT / "circuits/followups/aspectual_tense_raw_text_dual_program_fresh_lexicon_v1_result.json",
    "q_has_basis_result": ROOT / "circuits/followups/aspectual_anchor_das_resid18_rank1_transfer_v1_result.json",
    "q_is_basis_result": ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json",
    "q_has_prospective_result": ROOT / "circuits/followups/aspectual_anchor_resid10_frozen_gain_fresh_lexicon_v2_result.json",
    "q_is_prospective_result": ROOT / "circuits/followups/tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_v2_result.json",
}
EXPECTED_PRIOR_SHA256 = "0af300d9415ab02e89df918e1db3704a88e9f96ec23bec6c698d4b0ebe9d330c"
EXPECTED = {
    "shared_eval": "0e6dc9e7b8349bc4795f3ed65bc326e175a0a164ede19f4c6ae7ff10f56141ce",
    "matched_v2_builder": "1f4b29bda3e26af3ee0102316ab0af166e317d1646e8b0b51332061245e606d6",
    "raw_text_dual_program": "a756bfbeddaad7db2bb0c7feec1f3a6bd976b05fec36d231865b69e4813a976c",
    "prospective_dual_screen": "36d54f861bd6dd70a493e306480a812b1fb9009e4e35c26fd77df5ab22d59ca7",
    "q_has_basis_result": "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
    "q_is_basis_result": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "q_has_prospective_result": "a4cee1818acd6a28e999eda26d0447f94d080b913f4061d0e6dab4914cb3802c",
    "q_is_prospective_result": "dad39b298a0e89e5e0271149574012ebf7f75e995b83f6ede12ebe3b1aa746e8",
}
EXPECTED_ROWS = {"has_had": "7c2341ea65eb5915114ac4def7c3e7433d063e4cb3c988e518c91f1ff8e2b0ff", "is_was": "2efd47b9a89d0f092688a96d75bbc33e5b89991a8e5de28723c714319b9ccceb"}
TOKEN_MAPS = {"has_had": has_program.TOKEN_IDS, "is_was": is_program.TOKEN_IDS}


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_record(a, b, c, d, e, f):
    return {
        "pred_a_authority_basis_rows_and_shared_contract": a, "pred_b_native_capability_first": b,
        "pred_c_fixed_pair_support_task_routing": c, "pred_d_affinity_dispatched_causal_preservation": d,
        "pred_e_raw_text_control_abstention": e, "pred_f_exact_identity_coverage_and_price": f,
    }


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or {name: sha(path) for name, path in PATHS.items()} != EXPECTED:
        raise ExperimentError("prior or authority hash changed")
    prior = json.loads(PRIOR.read_text())
    loaded = {name: json.loads(path.read_text()) for name, path in PATHS.items() if name not in ("shared_eval", "matched_v2_builder", "raw_text_dual_program")}
    rows = rows_builder.build_rows_by_bank()
    rows_sha = rows_builder.validate_rows_by_bank(rows)
    expected_authorities = {
        "shared_eval_sha256": EXPECTED["shared_eval"], "matched_v2_builder_sha256": EXPECTED["matched_v2_builder"],
        "matched_v2_has_rows_sha256": EXPECTED_ROWS["has_had"], "matched_v2_is_rows_sha256": EXPECTED_ROWS["is_was"],
        "raw_text_dual_program_sha256": EXPECTED["raw_text_dual_program"], "prospective_dual_screen_sha256": EXPECTED["prospective_dual_screen"],
        "q_has_basis_result_sha256": EXPECTED["q_has_basis_result"], "q_is_basis_result_sha256": EXPECTED["q_is_basis_result"],
        "q_has_prospective_result_sha256": EXPECTED["q_has_prospective_result"], "q_is_prospective_result_sha256": EXPECTED["q_is_prospective_result"],
    }
    ok = (
        prior.get("candidate_id") == CANDIDATE_ID and prior.get("authorities") == expected_authorities
        and rows_sha == EXPECTED_ROWS and all(len(bank_rows) == 64 for bank_rows in rows.values())
        and evaluator.verify_contract() and loaded["prospective_dual_screen"].get("terminal") == "screen"
        and loaded["q_has_prospective_result"].get("terminal") == "screen" and loaded["q_is_prospective_result"].get("terminal") == "screen"
        and loaded["q_has_basis_result"]["basis"]["sha256"] == has_program.program_manifest()["rank1_basis_sha256"]
        and loaded["q_is_basis_result"]["basis"]["sha256"] == is_program.BASIS_SHA256
    )
    if not ok:
        raise ExperimentError("candidate, shared contract, prior screen, basis, or row authority changed")
    return rows, loaded


def main():
    rows_by_bank, loaded = validate_static()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False, "model_loaded": False, **evaluator.exact_price(rows=128, forwards=4, intervention_records=128, fitted_scalars=0)}, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    outputs_by_bank, capability = {}, {}
    for bank, rows in rows_by_bank.items():
        outputs = {side: backend.native(das._batch(backend, rows, side=side), capture=True) for side in ("base", "donor")}
        outputs_by_bank[bank] = outputs
        capability[bank] = evaluator.capability_cells(bank, rows, outputs, full_two_sided=True)
    capability_pass = all(cell["passed"] for cells in capability.values() for cell in cells)
    price = evaluator.exact_price(rows=128, forwards=4, intervention_records=128 if capability_pass else 0, fitted_scalars=0)
    if not capability_pass:
        predictions = prediction_record(True, False, False, False, False, True)
        result = {
            "schema": "aspectual_tense_resid10_pair_support_task_gate_result_v1", "candidate_id": CANDIDATE_ID,
            "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS, "capability_cells": capability,
            "causal_outcomes_opened": False, "task_affinity_records": [], "intervention_records": [], "predictions": predictions,
            "price": price, "terminal": "invalid", "reason": "native_capability_gate_failed_before_task_affinity_or_causal_outcomes",
            "serial_seconds": time.perf_counter() - started, "next_action": "retain the failed population without inspecting or changing the frozen affinity gate",
        }
        atomic_create_json(OUT, result)
        print(json.dumps(result, sort_keys=True))
        return

    bases = {
        "has_had": torch.as_tensor(loaded["q_has_basis_result"]["basis"]["values_column_major"], device=backend.device, dtype=torch.float32),
        "is_was": torch.as_tensor(loaded["q_is_basis_result"]["basis"]["values_column_major"], device=backend.device, dtype=torch.float32),
    }
    basis_ok = all(basis.shape == (1152,) and abs(float(basis.norm()) - 1.0) <= 1e-4 for basis in bases.values())
    basis_ok = basis_ok and hashlib.sha256(bases["has_had"].cpu().numpy().tobytes()).hexdigest() == has_program.program_manifest()["rank1_basis_sha256"]
    basis_ok = basis_ok and hashlib.sha256(bases["is_was"].cpu().numpy().tobytes()).hexdigest() == is_program.BASIS_SHA256
    records, affinity_records = [], []
    direction_ok, control_ok, local_error, final_error, no_ties = True, True, 0.0, 0.0, True
    for bank, rows in rows_by_bank.items():
        outputs = outputs_by_bank[bank]
        for index, row in enumerate(rows):
            family, direction = row["family"], evaluator.direction_for(row)
            side = evaluator.source_side(family)
            text_command = raw_program.select_command(row[f"{side}_text"])
            source10 = torch.as_tensor(outputs[side].captured[(row["row_id"], "resid:10")], device=backend.device).float()
            source18 = torch.as_tensor(outputs[side].captured[(row["row_id"], "resid:18")], device=backend.device).float()
            if family == "C":
                control_ok = control_ok and text_command["bank"] == "abstain" and text_command["direction"] is None
                predicted_bank, affinity = "abstain", None
                patched18, contrast = source18, None
                target, foil = row["base_answer_id"], row["base_foil_id"]
            else:
                direction_ok = direction_ok and text_command["bank"] == bank and text_command["direction"] == direction
                logits10 = das.head_logits(backend, source10[None, :])[0]
                has_support = max(float(logits10[has_program.TOKEN_IDS["has"]]), float(logits10[has_program.TOKEN_IDS["had"]]))
                is_support = max(float(logits10[is_program.TOKEN_IDS["is"]]), float(logits10[is_program.TOKEN_IDS["was"]]))
                affinity = has_support - is_support
                no_ties = no_ties and math.isfinite(affinity) and affinity != 0.0
                predicted_bank = "has_had" if affinity > 0.0 else "is_was"
                if predicted_bank == "has_had":
                    dispatched = has_program.upstream_carrier_actuation(source10, source18, bases[predicted_bank], backend.model.lm_head, direction=direction)
                    tokens = has_program.TOKEN_IDS
                else:
                    dispatched = is_program.upstream_writer_actuation(source10, source18, bases[predicted_bank], backend.model.lm_head, direction=direction)
                    tokens = is_program.TOKEN_IDS
                patched18, contrast = dispatched["patched_resid18"], dispatched["resid10_unembedding_contrast"]
                present_name, past_name = (("has", "had") if predicted_bank == "has_had" else ("is", "was"))
                current, other = (tokens[present_name], tokens[past_name]) if direction == "present_to_past" else (tokens[past_name], tokens[present_name])
                local_error = max(local_error, abs(float(contrast) - float(logits10[current] - logits10[other])))
                target, foil = evaluator.requested_token_ids(bank, direction, TOKEN_MAPS)
                affinity_records.append({"bank": bank, "family": family, "direction": direction, "row_id": str(row["row_id"]), "has_support": has_support, "is_support": is_support, "affinity": affinity, "predicted_bank": predicted_bank, "correct": predicted_bank == bank})
            base_margin = float(head.selected_margin(backend, source18[None, :], [target], [foil])[0])
            patched_margin = float(head.selected_margin(backend, patched18[None, :], [target], [foil])[0])
            full18 = das.head_logits(backend, source18[None, :])
            final_error = max(final_error, abs(base_margin - float(full18[0, target] - full18[0, foil])))
            record = {"bank": bank, "family": family, "direction": direction, "row_id": str(row["row_id"]), "source_side": side, "predicted_bank": predicted_bank, "affinity": affinity, "base_margin": base_margin, "patched_margin": patched_margin, "gain_refit": False, "basis_changed": False}
            if family in ("A1", "A2"):
                donor18 = torch.as_tensor(outputs["donor"].captured[(row["row_id"], "resid:18")], device=backend.device).float()
                donor_margin = float(head.selected_margin(backend, donor18[None, :], [target], [foil])[0])
                denominator = donor_margin - base_margin
                if denominator == 0.0:
                    raise ExperimentError("zero A recovery denominator")
                record["recovery"] = (patched_margin - base_margin) / denominator
            elif family == "P":
                if base_margin == 0.0:
                    raise ExperimentError("zero P reflection denominator")
                record["margin_reflection_fraction"] = (patched_margin - base_margin) / (-2.0 * base_margin)
            else:
                record["normalized_unrelated_effect"] = 0.0 if patched_margin == base_margin else math.inf
                control_ok = control_ok and patched18 is source18 and patched_margin == base_margin
            records.append(record)

    routing = []
    for bank in evaluator.BANKS:
        for family in ("A1", "A2", "P"):
            for direction in evaluator.DIRECTIONS:
                selected = [record for record in affinity_records if record["bank"] == bank and record["family"] == family and record["direction"] == direction]
                routing.append({"bank": bank, "family": family, "direction": direction, "count": len(selected), "accuracy": sum(record["correct"] for record in selected) / len(selected), "passed": sum(record["correct"] for record in selected) / len(selected) >= 0.75})
    summaries = evaluator.summarize_program_records(records)
    pred_a, pred_b = basis_ok, capability_pass
    pred_c = no_ties and len(affinity_records) == 96 and all(cell["count"] == 8 and cell["passed"] for cell in routing)
    pred_d = all(evaluator.program_bars_pass(summaries[bank]) for bank in evaluator.BANKS)
    pred_e = control_ok and all(record["predicted_bank"] == "abstain" and record["normalized_unrelated_effect"] == 0.0 for record in records if record["family"] == "C")
    pred_f = direction_ok and local_error <= 1e-4 and final_error <= 1e-4 and len(records) == 128 and price == evaluator.exact_price(rows=128, forwards=4, intervention_records=128, fitted_scalars=0)
    predictions = prediction_record(pred_a, pred_b, pred_c, pred_d, pred_e, pred_f)
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e and pred_f else "invalid")
    reason = {"screen": "resid10_pair_support_task_gate_supported", "null": "resid10_pair_support_routing_or_causal_bar_misses", "invalid": "authority_capability_tie_identity_coverage_finiteness_or_price_invalid"}[terminal]
    result = {
        "schema": "aspectual_tense_resid10_pair_support_task_gate_result_v1", "candidate_id": CANDIDATE_ID,
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS, "capability_cells": capability,
        "causal_outcomes_opened": True, "basis": {"q_has_sha256": has_program.program_manifest()["rank1_basis_sha256"], "q_is_sha256": is_program.BASIS_SHA256},
        "head_controls": {"local_max_abs_error": local_error, "final_max_abs_error": final_error},
        "routing_cells": routing, "task_affinity_records": affinity_records, "summaries": summaries, "intervention_records": records,
        "predictions": predictions, "price": price, "terminal": terminal, "reason": reason, "serial_seconds": time.perf_counter() - started,
        "next_action": "localize upstream production of the task-affinity branch scalar" if terminal == "screen" else "kill the fixed pair-support gate without threshold or feature rescue; retain the prospective raw-text interface",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability_cells", "causal_outcomes_opened", "head_controls", "routing_cells", "summaries", "predictions", "price", "terminal", "reason", "next_action")}, sort_keys=True))


if __name__ == "__main__":
    main()
