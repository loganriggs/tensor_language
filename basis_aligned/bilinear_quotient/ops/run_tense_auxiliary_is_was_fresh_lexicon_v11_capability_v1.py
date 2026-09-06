#!/usr/bin/env python3
"""Capability-only gate for construction-shifted is/was lexicon v11."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11.py"
V10_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v10_capability_v1_result.json"
INVALID_SOURCE_ATLAS = ROOT / "circuits/followups/iswas_mlp8_auxiliary_value_source_subset_atlas_v2_result.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
EXPECTED = {
    "prior": "38d8601d75d227621c283a2c6c03a7f98a59a32fdb54c0001c28e42c4a764245",
    "builder": "fbd47713fafcb87fc30ba339d175f7d06770ce36b93b6035e8848455529344ec",
    "v10_capability": "77e3c5b6e47dc9416133643f422c130427f90fed0746d26f211f54b681718b3d",
    "invalid_source_atlas": "a547d8f1e1fbafef58dbf37aa1835b362d19be444ecf54e62c9b3f9c4585f436",
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
        "v10_capability": V10_CAPABILITY,
        "invalid_source_atlas": INVALID_SOURCE_ATLAS,
        "base_runner": BASE_RUNNER,
    }
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("v11 capability authority changed")
    experiment.fresh = fresh
    experiment.PRIOR = PRIOR
    experiment.BUILDER = BUILDER
    experiment.OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
    experiment.CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v11_capability_v1"
    experiment.RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v11_capability_result_v1"
    experiment.ROWS_SHA256 = "70dd2a100a385c077751d2aae3cafdb731eccab3aac845a88560f146937238ff"
    experiment.EXPECTED = {
        "prior": EXPECTED["prior"],
        "builder": EXPECTED["builder"],
        "head_atlas": experiment.EXPECTED["head_atlas"],
    }
    experiment.main()


if __name__ == "__main__":
    main()
