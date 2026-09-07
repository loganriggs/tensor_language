#!/usr/bin/env python3
"""Capability-only gate for simple-frame, lexicon-disjoint is/was v14."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v14 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v14_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v14.py"
V13_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v13_capability_v1_result.json"
SELECTION = ROOT / "circuits/followups/temporal_iswas_v12_omitted_response_augmentation_cube_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
EXPECTED = {
    "prior": "03be21e66654abfb1f4d8887f25110479ef00e7468c423d56927a9f7d19d0c65",
    "builder": "ca414597f76c5621b4527ac8f1920cb5c09b6df08346d0b0ce988e46991e331f",
    "v13_capability": "2c500e5043d24501d9633ff8015ec58374f8e2088362c74857c891fa96806fa8",
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
        "prior": PRIOR, "builder": BUILDER, "v13_capability": V13_CAPABILITY,
        "selection": SELECTION, "base_runner": BASE_RUNNER,
    }
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("v14 capability authority changed")
    experiment.fresh = fresh
    experiment.PRIOR = PRIOR
    experiment.BUILDER = BUILDER
    experiment.OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v14_capability_v1_result.json"
    experiment.CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v14_capability_v1"
    experiment.RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v14_capability_result_v1"
    experiment.ROWS_SHA256 = "e70775d758e647af2ed541ddd93a1a4a8050d121c2c023df19198d68e8322103"
    experiment.EXPECTED = {
        "prior": EXPECTED["prior"],
        "builder": EXPECTED["builder"],
        "head_atlas": experiment.EXPECTED["head_atlas"],
    }
    experiment.main()


if __name__ == "__main__":
    main()
