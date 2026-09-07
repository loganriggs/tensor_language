#!/usr/bin/env python3
"""Capability-only gate for construction/lexicon-disjoint is/was v13."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v13 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v13_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v13.py"
V12_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
SELECTION = ROOT / "circuits/followups/temporal_iswas_v12_omitted_response_augmentation_cube_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
EXPECTED = {
    "prior": "4a0664f5a8dc202ab2f882714d0e2a90238931fad3475be5b801e64d08bd5f44",
    "builder": "9b2dbe9a7f0339131ed7c0ed486b0467b44d0cbb9bb5c9dfef1ed1e987ee730a",
    "v12_capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
    "selection": "144e1c001a9c186daa7e2fa426b702a390f938b6fb2f2cf27880812d366301ed",
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
        "prior": PRIOR, "builder": BUILDER, "v12_capability": V12_CAPABILITY,
        "selection": SELECTION, "base_runner": BASE_RUNNER,
    }
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("v13 capability authority changed")
    experiment.fresh = fresh
    experiment.PRIOR = PRIOR
    experiment.BUILDER = BUILDER
    experiment.OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v13_capability_v1_result.json"
    experiment.CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v13_capability_v1"
    experiment.RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v13_capability_result_v1"
    experiment.ROWS_SHA256 = "d495ba5b993f66e277e17e2b31d73b2b0e1064c1cf3ca88c761c4fb47bcda3b0"
    experiment.EXPECTED = {
        "prior": EXPECTED["prior"],
        "builder": EXPECTED["builder"],
        "head_atlas": experiment.EXPECTED["head_atlas"],
    }
    experiment.main()


if __name__ == "__main__":
    main()
