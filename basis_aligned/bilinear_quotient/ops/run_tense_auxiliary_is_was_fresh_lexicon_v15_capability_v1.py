#!/usr/bin/env python3
"""Capability-only gate for the cue-strong, text-disjoint is/was v15 bank."""
# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
import json
import os
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v15_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15.py"
V13_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v13_capability_v1_result.json"
V14_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v14_capability_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
EXPECTED = {
    "prior": "4fe19cddb27342f605058c02b673a91255ad866e503a82f953e4fcd192d4ac21",
    "builder": "e5774cd5e93c564ad3bdb3169c482c2e1ddb295b85cc7c9885a7b077493a8343",
    "v13_capability": "2c500e5043d24501d9633ff8015ec58374f8e2088362c74857c891fa96806fa8",
    "v14_capability": "a0b1a5da58a5d041d29debbbe4fd5a02d4de4f81223d312fc6178e1ce17e50f9",
    "base_runner": "33e8afdcfa02379539bc6f018d662af5f13901d07676458414c9a25aecee8a3e",
}
ROWS_SHA256 = "9e6848a8bfe05be59ecd02d256502597e5e8b6b48fadd2b9f8e1b830b9e50c46"
PREDICTION_KEYS = (
    "pred_a_authority_novelty_and_exact_population",
    "pred_b_native_a_panel_capability",
    "pred_c_joint_capable_population",
    "pred_d_no_causal_outcome_access_and_exact_price",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {
        "prior": PRIOR, "builder": BUILDER, "v13_capability": V13_CAPABILITY,
        "v14_capability": V14_CAPABILITY, "base_runner": BASE_RUNNER,
    }
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"v15 capability authority changed: {observed}")
    if json.loads(V13_CAPABILITY.read_text())["terminal"] != "null" or json.loads(V14_CAPABILITY.read_text())["terminal"] != "null":
        raise RuntimeError("v13/v14 capability-null ancestry changed")

    experiment.fresh = fresh
    experiment.PRIOR = PRIOR
    experiment.BUILDER = BUILDER
    experiment.OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v15_capability_v1_result.json"
    experiment.CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v15_capability_v1"
    experiment.RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v15_capability_result_v1"
    experiment.ROWS_SHA256 = ROWS_SHA256
    experiment.EXPECTED = {
        "prior": EXPECTED["prior"], "builder": EXPECTED["builder"],
        "head_atlas": experiment.EXPECTED["head_atlas"],
    }
    original_write = experiment.atomic_create_json

    def write_result(_path, result):
        result["authority_sha256"] = EXPECTED
        result["predictions"]["pred_c_joint_capable_population"] = all(
            len(result["jointly_capable_row_ids"][panel]) >= 28 for panel in ("A1", "A2")
        )
        result["terminal"] = (
            "invalid" if not result["predictions"]["pred_a_authority_novelty_and_exact_population"]
            or not result["predictions"]["pred_d_no_causal_outcome_access_and_exact_price"]
            else "screen" if all(result["predictions"].values()) else "null"
        )
        original_write(experiment.OUT, result)

    experiment.atomic_create_json = write_result
    experiment.main()


if __name__ == "__main__":
    main()
