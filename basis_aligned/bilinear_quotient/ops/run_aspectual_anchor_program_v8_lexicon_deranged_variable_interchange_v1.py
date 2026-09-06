#!/usr/bin/env python3
"""Lexicon-deranged operational interchange of explicit v8 variable groups."""

# BQGATE: EXPERIMENT pred_a_authority_derangement_capability_and_controls pred_b_target_program_effect pred_c_whole_lexicon_transfer pred_d_groupwise_lexicon_invariance pred_e_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import aspectual_anchor_variable_interchange_engine as engine
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_program_v8_lexicon_deranged_variable_interchange_v1.json"
ENGINE = ROOT / "ops/aspectual_anchor_variable_interchange_engine.py"
PROGRAM = ROOT / "ops/aspectual_anchor_transparent_path_program_v8.py"
FAMILY_RESULT = ROOT / "circuits/followups/aspectual_anchor_program_v8_cross_family_variable_interchange_v1_result.json"
PAIRING_AUDIT = ROOT / "circuits/followups/aspectual_anchor_program_v8_lexicon_derangement_pairing_audit_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_program_v8_lexicon_deranged_variable_interchange_v1_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.program_v8_lexicon_deranged_variable_interchange_v1"
EXPECTED_PRIOR_SHA256 = "9183760625e2de4165368966b4314e16b5f2abc6bdcabac0b884595f9c73fe3f"
EXPECTED = {
    ENGINE: "c7158f4e0313d7eaa5a3c3010745182a31cea2c16fc31a87dd22af8449bc9c92",
    PROGRAM: "87eb67f3a96904534c8d3ddca5e1df59fa14efd88d1174e1cc2805435346bb57",
    FAMILY_RESULT: "b10b919f4c3cf9d53d1a397326f5d46a195ecd710f3df24fb8acff35966ea031",
    PAIRING_AUDIT: "ef01c0107f84e6f50a3200e81c74f0875e12e06f709a3935bcc720a0ecab671b",
}
MODEL_FORWARDS_MAX = 72
EXAMPLE_EVALUATIONS_MAX = 576


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_selector(_panel, target_key, _components):
    family, group_number = target_key
    return family, (group_number + 6) % 16


def validate_static() -> None:
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    family = json.loads(FAMILY_RESULT.read_text())
    pairing = json.loads(PAIRING_AUDIT.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or family.get("terminal") != pairing.get("terminal")
        or family.get("terminal") != "screen"
        or pairing.get("pair_count") != 64
        or not all(pairing.get("predictions", {}).values())
    ):
        raise ExperimentError("candidate, parent terminal, or pairing audit changed")


def main() -> None:
    validate_static()
    dryrun = {
        "schema": "aspectual_anchor_program_v8_lexicon_deranged_variable_interchange_dryrun_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "target_rows": 64,
        "source_offset": 6,
        "arms": list(engine.ARMS),
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

    measured = engine.run_interchange(source_selector)
    panels = measured["panels"]
    ratios = measured["recovery_fraction_vs_target_full"]
    pairs = measured["pair_records"]
    pred_a = (
        measured["capability"]
        and measured["manual_base_scored_logit_max_abs"] <= 1.0e-4
        and measured["writer_bilinear_tensor_reconstruction_max_abs"] <= 2.0e-3
        and measured["mlp_bilinear_tensor_reconstruction_max_abs"] <= 5.0e-3
        and len(pairs) == 64
        and all(
            pair["source_group"] == (pair["target_group"] + 6) % 16
            and pair["source_family"] == pair["target_family"]
            and pair["same_direction"]
            and pair["different_reporter"]
            and pair["different_period"]
            for pair in pairs
        )
    )
    pred_b = all(panels[panel]["target_full"]["mean_recovery"] > 0.0 and panels[panel]["target_full"]["direction_fraction"] >= 0.75 for panel in panels)
    pred_c = all(ratios[panel]["source_full"] >= 0.50 and panels[panel]["source_full"]["direction_fraction"] >= 0.75 for panel in panels)
    pred_d = all(ratios[panel][arm] >= 0.75 and panels[panel][arm]["direction_fraction"] >= 0.75 for panel in panels for arm in ("swap_initial", "swap_attention", "swap_mlp"))
    records = measured["intervention_records"]
    pred_e = (
        len(records) == 320
        and len({(record["panel"], record["arm_id"], record["target_row_id"]) for record in records}) == 320
        and measured["forward_calls"] <= MODEL_FORWARDS_MAX
        and measured["example_evaluations"] <= EXAMPLE_EVALUATIONS_MAX
    )
    predictions = {
        "pred_a_authority_derangement_capability_and_controls": pred_a,
        "pred_b_target_program_effect": pred_b,
        "pred_c_whole_lexicon_transfer": pred_c,
        "pred_d_groupwise_lexicon_invariance": pred_d,
        "pred_e_exact_coverage_and_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_e else "invalid")
    reason = {"screen": "explicit_v8_variables_define_tested_lexicon_invariant_quotient", "null": "whole_or_groupwise_variables_remain_lexically_bound", "invalid": "authority_derangement_capability_control_target_or_coverage_invalid"}[terminal]
    result = {
        "schema": "aspectual_anchor_program_v8_lexicon_deranged_variable_interchange_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": measured["started_utc"],
        "finished_utc": measured["finished_utc"],
        "serial_seconds": measured["serial_seconds"],
        "prior_art_sha256": EXPECTED_PRIOR_SHA256,
        "engine_sha256": EXPECTED[ENGINE],
        "program_sha256": EXPECTED[PROGRAM],
        "predictions": predictions,
        "score": {
            "panels": panels,
            "recovery_fraction_vs_target_full": ratios,
            "manual_base_scored_logit_max_abs": measured["manual_base_scored_logit_max_abs"],
            "writer_bilinear_tensor_reconstruction_max_abs": measured["writer_bilinear_tensor_reconstruction_max_abs"],
            "mlp_bilinear_tensor_reconstruction_max_abs": measured["mlp_bilinear_tensor_reconstruction_max_abs"],
            "forward_calls": measured["forward_calls"],
            "example_evaluations": measured["example_evaluations"],
            "record_count": len(records),
            "model_backwards": 0,
            "model_updates": 0,
            "fit_parameters": 0,
        },
        "pair_records": pairs,
        "intervention_records": records,
        "terminal": terminal,
        "reason": reason,
        "next_action": "compile construction-family-lexicon quotient scope into program v11" if terminal == "screen" else "retain the first failing variable group as lexically bound and localize its downstream reader",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "ratios": ratios, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
