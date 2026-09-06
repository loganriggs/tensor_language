#!/usr/bin/env python3
"""Capability-only gate for lexical-transfer is/was lexicon v12."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
V11_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
SOURCE_SCREEN = ROOT / "circuits/followups/iswas_mlp8_auxiliary_value_source_subset_atlas_v3_result.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
EXPECTED = {
    "prior": "3f2d08d15887c2ba7ec6f68a89b3e3eb194ded0f1c3ab08275002bfb8039aa5c",
    "builder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2",
    "v11_capability": "6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2",
    "source_screen": "e389c30a9d6d9f11e62c2a2a488d9ff5150433682f0716dc10f1b7c18d3b9f53",
    "base_runner": "33e8afdcfa02379539bc6f018d662af5f13901d07676458414c9a25aecee8a3e",
}
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
        "prior": PRIOR,
        "builder": BUILDER,
        "v11_capability": V11_CAPABILITY,
        "source_screen": SOURCE_SCREEN,
        "base_runner": BASE_RUNNER,
    }
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("v12 capability authority changed")
    experiment.fresh = fresh
    experiment.PRIOR = PRIOR
    experiment.BUILDER = BUILDER
    experiment.OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
    experiment.CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v12_capability_v1"
    experiment.RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v12_capability_result_v1"
    experiment.ROWS_SHA256 = "5aa01e53247320889efc95a5472a577765f49e7879a861bf0c4230e86a4c8272"
    experiment.EXPECTED = {
        "prior": EXPECTED["prior"],
        "builder": EXPECTED["builder"],
        "head_atlas": experiment.EXPECTED["head_atlas"],
    }
    experiment.main()


if __name__ == "__main__":
    main()
