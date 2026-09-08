#!/usr/bin/env python3
"""Capability-only gate for the aligned four-head-confirmation v23 bank."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
import json
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v23_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23.py"
V22_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v22_capability_v1_result.json"
V22_AUDIT = ROOT / "circuits/followups/temporal_iswas_v22_reader_contracted_writer_effect_game_v1_instrument_audit.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v23_capability_v1_result.json"
EXPECTED = {
    "prior": "bee191ff56f93f72b11b14cefe6439db578ce495bad800d6f5177f5a15a6f2d2",
    "builder": "a4830fd110b8cd854a5f28bfae776f697a15d4f791355990e02ea030fdca4c05",
    "v22_result": "692914652431d33389f88818532c6d29aa16b3f1d151ed4728eb7aece7c3b437",
    "v22_audit": "e49fa3db73811cdfc7481357378c89a46d562304a5820b46b5f145c8c236a8c2",
    "base_runner": "33e8afdcfa02379539bc6f018d662af5f13901d07676458414c9a25aecee8a3e",
}
ROWS_SHA256 = "46e9e493a4b2b42ce0c121b30978f08208537300363fa91902184c8f8d6565c7"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v23_capability_v1"
RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v23_capability_result_v1"
JOINT_BAR = 12
PREDICTION_KEYS = (
    "pred_a_authority_novelty_and_exact_population",
    "pred_b_native_a_panel_capability",
    "pred_c_joint_capable_population",
    "pred_d_no_causal_outcome_access_and_exact_price",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "builder": BUILDER, "v22_result": V22_RESULT,
             "v22_audit": V22_AUDIT, "base_runner": BASE_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"v23 capability authority changed: {observed}")
    ancestry, audit = json.loads(V22_RESULT.read_text()), json.loads(V22_AUDIT.read_text())
    if (ancestry.get("terminal") != "screen" or not all(ancestry.get("predictions", {}).values())
            or ancestry.get("causal_outcomes_opened") is not False
            or audit.get("verdict") != "target_and_P_game_valid_four_head_confirmation; global_A_and_C_selectivity_not_admissible"):
        raise RuntimeError("v22 capability/audit ancestry changed")
    rows = fresh.build_rows()
    if any(row["base_semantic_position"] != row["donor_semantic_position"] for row in rows):
        raise RuntimeError("v23 alignment invariant changed")
    experiment.fresh, experiment.PRIOR, experiment.BUILDER = fresh, PRIOR, BUILDER
    experiment.OUT, experiment.CANDIDATE_ID = OUT, CANDIDATE_ID
    experiment.RESULT_SCHEMA, experiment.ROWS_SHA256 = RESULT_SCHEMA, ROWS_SHA256
    experiment.EXPECTED = {"prior": EXPECTED["prior"], "builder": EXPECTED["builder"],
                           "head_atlas": experiment.EXPECTED["head_atlas"]}
    original_write = experiment.atomic_create_json

    def write_result(_path, result):
        if set(result["predictions"]) != set(PREDICTION_KEYS):
            raise RuntimeError("v23 prediction inventory changed")
        if result["price"] != {"model_forwards": 2, "example_evaluations": 128,
                "interventions": 0, "transformer_backwards": 0, "model_updates": 0}:
            raise RuntimeError("v23 capability price changed")
        if any(len(result["jointly_capable_row_ids"][panel]) < JOINT_BAR
               for panel in ("A1", "A2")) != (result["predictions"][PREDICTION_KEYS[2]] is False):
            raise RuntimeError("v23 joint-capability predicate changed")
        result["authority_sha256"] = EXPECTED
        result["position_alignment"] = {
            family: all(row["base_semantic_position"] == row["donor_semantic_position"]
                        for row in rows if row["family"] == family)
            for family in ("A1", "A2", "P", "C")
        }
        result["causal_outcomes_opened"] = False
        original_write(OUT, result)

    experiment.atomic_create_json = write_result
    experiment.main()


if __name__ == "__main__":
    main()
