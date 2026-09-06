#!/usr/bin/env python3
"""Exhaustively audit the typed composition contract around the v4 actuator."""

# BQGATE: EXPERIMENT pred_a_authority_and_complete_domain pred_b_identity_and_idempotence pred_c_last_write_wins pred_d_independent_commutativity pred_e_no_hidden_additivity_or_price
# BQLANE: cpu
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import task14_bracket_margin_actuator as actuator
import task14_bracket_margin_composition as composition

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/task14_bracket_margin_actuator_composition_contract_v5.json"
V4 = ROOT / "circuits/followups/task14_bracket_counterfactual_margin_actuator_v4_artifact.json"
V4_RESULT = ROOT / "circuits/followups/task14_bracket_counterfactual_margin_actuator_v4_result.json"
ARTIFACT = ROOT / "circuits/followups/task14_bracket_margin_actuator_composition_contract_v5_artifact.json"
OUT = ROOT / "circuits/followups/task14_bracket_margin_actuator_composition_contract_v5_result.json"
CANDIDATE_ID = "cross_behavior.task14_bracket_margin_actuator_composition_contract_v5"
EXPECTED = {
    PRIOR: "02ed75ec5118c39461d670381a28b99c439747c41a49ccb9852e38410ded99eb",
    V4: "85c5cc0549421fc1575d96ce621d0677ea4b0cc2d154b2c0bf7af90f4148bd4c",
    V4_RESULT: "fb3520b7bc77720adcda6d8a87066061878462136a60bc7d866bd2de056f2311",
}
BASELINES = (-10.0, -1.0, 0.0, 1.0, 10.0)
TASK_EDITS = tuple((recipient, donor, cardinality) for recipient, donor in (("singular", "plural"), ("plural", "singular")) for cardinality in range(5))
BRACKET_EDITS = tuple((recipient, donor) for recipient in (1, 8, 60) for donor in (1, 8, 60))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> dict:
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"immutable input changed: {path}")
    artifact = json.loads(V4.read_text())
    result = json.loads(V4_RESULT.read_text())
    if result["terminal"] != "screen" or artifact["candidate_id"] != "cross_behavior.task14_bracket_counterfactual_margin_actuator_v4":
        raise ValueError("v4 authority status invalid")
    return artifact


def compile_plan() -> dict:
    load()
    return {
        "schema": "task14_bracket_margin_actuator_composition_contract_plan_v5",
        "candidate_id": CANDIDATE_ID,
        "prior_art_sha256": EXPECTED[PRIOR],
        "domain": {"task14_edits": len(TASK_EDITS), "task14_no_edits": 1, "bracket_edits_including_self_identity": len(BRACKET_EDITS)},
        "price": {"new_learned_scalars": 0, "model_forwards": 0, "example_evaluations": 0, "backwards": 0, "fits": 0, "parameter_updates": 0},
    }


def build() -> dict:
    return {
        "schema": "task14_bracket_margin_actuator_composition_contract_artifact_v5",
        "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "state": ["slot_id", "behavior", "immutable_native_baseline_margin", "current_edit_or_none"],
        "evaluation": "immutable_native_baseline_margin + effect(current_edit_or_none)",
        "same_slot_composition": "right_biased_overwrite",
        "different_slot_composition": "commuting_product_update",
        "identity": "edit_none_or_bracket_self_pair",
        "forbidden": ["sum_effect_history", "mutate_native_baseline", "cross_typed_edit"],
        "new_learned_scalars": 0,
    }


def evaluate(contract: dict, artifact: dict) -> dict:
    domain_ok = len(artifact["effects"]["task14"]) == len(TASK_EDITS) == 10 and len(artifact["effects"]["bracket"]) == 6 and len(BRACKET_EDITS) == 9
    identity_cases = idempotence_cases = overwrite_cases = commute_cases = 0
    identity_ok = idempotence_ok = overwrite_ok = commute_ok = True
    for behavior, edits in (("task14", TASK_EDITS), ("bracket", BRACKET_EDITS)):
        for baseline in BASELINES:
            base = composition.SlotState("x", behavior, baseline)
            identity_ok &= composition.evaluate(artifact, base) == baseline and composition.overwrite(artifact, base, None) == base
            identity_cases += 2
            for edit in edits:
                once = composition.overwrite(artifact, base, edit)
                if behavior == "bracket" and edit[0] == edit[1]:
                    identity_ok &= composition.evaluate(artifact, once) == baseline
                    identity_cases += 1
                twice = composition.overwrite(artifact, once, edit)
                idempotence_ok &= once == twice and composition.evaluate(artifact, once) == composition.evaluate(artifact, twice)
                idempotence_cases += 1
                for second in edits:
                    sequential = composition.overwrite(artifact, once, second)
                    direct = composition.overwrite(artifact, base, second)
                    overwrite_ok &= sequential == direct and composition.evaluate(artifact, sequential) == composition.evaluate(artifact, direct)
                    overwrite_cases += 1
    for baseline_task in BASELINES:
        for baseline_bracket in BASELINES:
            initial = {
                "task": composition.SlotState("task", "task14", baseline_task),
                "bracket": composition.SlotState("bracket", "bracket", baseline_bracket),
            }
            for task_edit in TASK_EDITS:
                for bracket_edit in BRACKET_EDITS:
                    task_then_bracket = composition.update_slot(artifact, composition.update_slot(artifact, initial, "task", task_edit), "bracket", bracket_edit)
                    bracket_then_task = composition.update_slot(artifact, composition.update_slot(artifact, initial, "bracket", bracket_edit), "task", task_edit)
                    commute_ok &= task_then_bracket == bracket_then_task and {key: composition.evaluate(artifact, value) for key, value in task_then_bracket.items()} == {key: composition.evaluate(artifact, value) for key, value in bracket_then_task.items()}
                    commute_cases += 1
    rejection_ok = True
    malformed = (
        (composition.SlotState("x", "task14", 0.0), (1, 8)),
        (composition.SlotState("x", "bracket", 0.0), ("singular", "plural", 0)),
        (composition.SlotState("x", "unknown", 0.0), None),
    )
    for state, edit in malformed:
        try:
            composition.overwrite(artifact, state, edit)
            if state.behavior == "unknown":
                composition.effect(artifact, state)
        except actuator.ActuatorError:
            continue
        rejection_ok = False
    no_hidden = contract["forbidden"] == ["sum_effect_history", "mutate_native_baseline", "cross_typed_edit"] and contract["new_learned_scalars"] == 0 and artifact["runtime"] == {"table_lookups": 1, "scalar_additions": 1, "model_forwards_after_input": 0}
    predictions = {
        "pred_a_authority_and_complete_domain": domain_ok,
        "pred_b_identity_and_idempotence": identity_ok and idempotence_ok,
        "pred_c_last_write_wins": overwrite_ok,
        "pred_d_independent_commutativity": commute_ok,
        "pred_e_no_hidden_additivity_or_price": no_hidden and rejection_ok,
    }
    return {
        "exhaustive_cases": {"identity": identity_cases, "idempotence": idempotence_cases, "same_slot_ordered_overwrite": overwrite_cases, "independent_slot_commutativity": commute_cases},
        "semantics": {"same_slot": "right_biased_overwrite_from_immutable_baseline", "different_slots": "commuting_product_update", "effect_history_summed": False},
        "new_learned_scalars": 0,
        "classification": "exact_composition_contract_around_empirically_screened_margin_actuator",
        "predictions": predictions,
        "terminal": "screen" if all(predictions.values()) else "null",
    }


def main() -> None:
    plan = compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, sort_keys=True)); return
    if ARTIFACT.exists() or OUT.exists():
        raise ValueError("refusing overwrite")
    artifact = load(); contract = build()
    contract_bytes = managed.atomic_create_json(ARTIFACT, contract)
    result = evaluate(contract, artifact)
    payload = managed.atomic_create_json(OUT, {"schema": "task14_bracket_margin_actuator_composition_contract_result_v5", "candidate_id": CANDIDATE_ID, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "plan": plan, "artifact_sha256": hashlib.sha256(contract_bytes).hexdigest(), "score": result, "terminal": result["terminal"]})
    print(json.dumps({"terminal": result["terminal"], "predictions": result["predictions"], "artifact_sha256": hashlib.sha256(contract_bytes).hexdigest(), "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
