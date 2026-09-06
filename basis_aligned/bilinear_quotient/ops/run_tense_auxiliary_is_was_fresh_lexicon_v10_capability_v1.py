#!/usr/bin/env python3
"""Capability-only gate for hybrid is/was lexicon v10."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
from pathlib import Path
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v10.py"
V8_NULL = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1_result.json"
V9_NULL = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v9_capability_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
EXPECTED = {"prior": "77aa5a1d5c83ebbdff7cdd417cd13ba90b928c2604b0b3ca14fa870dfd180b7b",
    "builder": "13e7954cde01b6b7d826915fc2ae02d4b9e16975150cf73aa5e0a1f906c1b757",
    "v8_null": "a2f89f5ef90ae52b6e76fbccf14daf4fef5873bd5c07766ef03070836e93a0bf",
    "v9_null": "ae0be913b0d2389048ee40aa4fd1999ab2dd684916c05bd506c173b76a068bd9",
    "base_runner": "33e8afdcfa02379539bc6f018d662af5f13901d07676458414c9a25aecee8a3e"}
PREDICTION_KEYS = ("pred_a_authority_novelty_and_exact_population",
    "pred_b_native_a_panel_capability", "pred_c_joint_capable_population",
    "pred_d_no_causal_outcome_access_and_exact_price")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "builder": BUILDER, "v8_null": V8_NULL,
             "v9_null": V9_NULL, "base_runner": BASE_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("v10 hybrid capability authority changed")
    experiment.fresh = fresh
    experiment.PRIOR = PRIOR
    experiment.BUILDER = BUILDER
    experiment.OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1_result.json"
    experiment.CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v10_capability_v1"
    experiment.RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v10_capability_result_v1"
    experiment.ROWS_SHA256 = "e659ebc13bf5ef40cfb6693b8a694b46e533129dc0b156a76bc4456de4a0dfcd"
    experiment.EXPECTED = {"prior": EXPECTED["prior"], "builder": EXPECTED["builder"],
                           "head_atlas": experiment.EXPECTED["head_atlas"]}
    experiment.main()


if __name__ == "__main__":
    main()
