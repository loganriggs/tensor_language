#!/usr/bin/env python3
"""Capability-only gate for the frozen-tensor transfer v18 bank."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
import json
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v18 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v18_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v18.py"
V17_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v17_capability_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v18_capability_v1_result.json"
EXPECTED = {
    "prior": "582e780923b6ba8860797ad3ea45011a227421667e182ea826ad6b5ada8bae0a",
    "builder": "44a51b3370bcf59f9cb5b17e5d63ac1cfdba3e883072112a49ec2f92ba78ae2a",
    "v17_result": "53e89054f456b65853d7ef5f0e566d1bd3993a6a55c6ecdc6615269f28d23e78",
    "base_runner": "33e8afdcfa02379539bc6f018d662af5f13901d07676458414c9a25aecee8a3e",
}
ROWS_SHA256 = "ef28c66256fd0800d70c62744a549f66e1a6ce35d67acd88569a5d552aa7738f"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v18_capability_v1"
RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v18_capability_result_v1"
JOINT_BAR = 12
PREDICTION_KEYS = (
    "pred_a_authority_novelty_and_exact_population",
    "pred_b_native_a_panel_capability",
    "pred_c_joint_capable_population",
    "pred_d_no_causal_outcome_access_and_exact_price",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "builder": BUILDER, "v17_result": V17_RESULT,
             "base_runner": BASE_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"v18 capability authority changed: {observed}")
    ancestry = json.loads(V17_RESULT.read_text())
    if (
        ancestry.get("terminal") != "screen"
        or not all(ancestry.get("predictions", {}).values())
        or ancestry.get("causal_outcomes_opened") is not False
        or any(len(ancestry["jointly_capable_row_ids"][panel]) < JOINT_BAR
               for panel in ("A1", "A2"))
    ):
        raise RuntimeError("v17 capability-only ancestry changed")

    experiment.fresh = fresh
    experiment.PRIOR = PRIOR
    experiment.BUILDER = BUILDER
    experiment.OUT = OUT
    experiment.CANDIDATE_ID = CANDIDATE_ID
    experiment.RESULT_SCHEMA = RESULT_SCHEMA
    experiment.ROWS_SHA256 = ROWS_SHA256
    experiment.EXPECTED = {
        "prior": EXPECTED["prior"],
        "builder": EXPECTED["builder"],
        "head_atlas": experiment.EXPECTED["head_atlas"],
    }
    original_write = experiment.atomic_create_json

    def write_result(_path, result):
        if set(result["predictions"]) != set(PREDICTION_KEYS):
            raise RuntimeError("v18 prediction inventory changed")
        if result["price"] != {
            "model_forwards": 2,
            "example_evaluations": 128,
            "interventions": 0,
            "transformer_backwards": 0,
            "model_updates": 0,
        }:
            raise RuntimeError("v18 capability price changed")
        if any(len(result["jointly_capable_row_ids"][panel]) < JOINT_BAR
               for panel in ("A1", "A2")) != (
                   result["predictions"][PREDICTION_KEYS[2]] is False):
            raise RuntimeError("v18 joint-capability predicate changed")
        result["authority_sha256"] = EXPECTED
        original_write(OUT, result)

    experiment.atomic_create_json = write_result
    experiment.main()


if __name__ == "__main__":
    main()
