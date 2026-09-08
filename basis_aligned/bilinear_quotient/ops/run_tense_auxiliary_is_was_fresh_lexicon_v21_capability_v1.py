#!/usr/bin/env python3
"""Capability-only gate for the pristine read/write-confirmation v21 bank."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
import json
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v21 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v21_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v21.py"
V20_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v20_capability_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v21_capability_v1_result.json"
EXPECTED = {
    "prior": "78bcc4429d65b580cf01adbf7136b083c6c22a405da1a6fe1ac09499bfe7a713",
    "builder": "4289795a9886c3c36561f47ca202cb790072dd05c324d96ca9fbd5e837a23dc8",
    "v20_result": "d03534d18f9d3662e5765f7d79294f9ea5f45d09f59d891a093c1739bd20c2ef",
    "base_runner": "33e8afdcfa02379539bc6f018d662af5f13901d07676458414c9a25aecee8a3e",
}
ROWS_SHA256 = "a6ef4ad752ccf82365f202709091334249335ec2809fc8c1b33b71c2bee6b6c5"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v21_capability_v1"
RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v21_capability_result_v1"
JOINT_BAR = 12
PREDICTION_KEYS = (
    "pred_a_authority_novelty_and_exact_population",
    "pred_b_native_a_panel_capability",
    "pred_c_joint_capable_population",
    "pred_d_no_causal_outcome_access_and_exact_price",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "builder": BUILDER, "v20_result": V20_RESULT,
             "base_runner": BASE_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"v21 capability authority changed: {observed}")
    ancestry = json.loads(V20_RESULT.read_text())
    if (ancestry.get("terminal") != "screen" or not all(ancestry.get("predictions", {}).values())
            or ancestry.get("causal_outcomes_opened") is not False):
        raise RuntimeError("v20 capability-only ancestry changed")
    experiment.fresh, experiment.PRIOR, experiment.BUILDER = fresh, PRIOR, BUILDER
    experiment.OUT, experiment.CANDIDATE_ID = OUT, CANDIDATE_ID
    experiment.RESULT_SCHEMA, experiment.ROWS_SHA256 = RESULT_SCHEMA, ROWS_SHA256
    experiment.EXPECTED = {"prior": EXPECTED["prior"], "builder": EXPECTED["builder"],
                           "head_atlas": experiment.EXPECTED["head_atlas"]}
    original_write = experiment.atomic_create_json
    def write_result(_path, result):
        if set(result["predictions"]) != set(PREDICTION_KEYS):
            raise RuntimeError("v21 prediction inventory changed")
        if result["price"] != {"model_forwards": 2, "example_evaluations": 128,
                "interventions": 0, "transformer_backwards": 0, "model_updates": 0}:
            raise RuntimeError("v21 capability price changed")
        if any(len(result["jointly_capable_row_ids"][panel]) < JOINT_BAR
               for panel in ("A1", "A2")) != (
                   result["predictions"][PREDICTION_KEYS[2]] is False):
            raise RuntimeError("v21 joint-capability predicate changed")
        result["authority_sha256"] = EXPECTED
        original_write(OUT, result)
    experiment.atomic_create_json = write_result
    experiment.main()


if __name__ == "__main__": main()
