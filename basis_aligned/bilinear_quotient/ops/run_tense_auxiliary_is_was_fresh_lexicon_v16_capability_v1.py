#!/usr/bin/env python3
"""Capability-only gate for the third-construction is/was v16 bank."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
import json
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16.py"
V15_AUDIT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v15_capability_v2_audit_result.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
EXPECTED = {
    "prior": "feea0b786aa23135daf6cfc7f818365e80bebcccfdd20e3e8e47af1b039c904b",
    "builder": "5b1cb38cc62b5682c03505a5090e962d4773015efb44f8fad145515f5066b549",
    "v15_audit": "fbe395731de4fa7850999864f67bd45ec709e5da3e3e237e9dcae352c1211c2f",
    "base_runner": "33e8afdcfa02379539bc6f018d662af5f13901d07676458414c9a25aecee8a3e",
}
ROWS_SHA256 = "4c5dfaee126c04ac7ea6ef5f53d6ad62a24806fa7133a0c773471a90e4d2e468"
PREDICTION_KEYS = (
    "pred_a_authority_novelty_and_exact_population",
    "pred_b_native_a_panel_capability",
    "pred_c_joint_capable_population",
    "pred_d_no_causal_outcome_access_and_exact_price",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "builder": BUILDER, "v15_audit": V15_AUDIT,
             "base_runner": BASE_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"v16 capability authority changed: {observed}")
    audit = json.loads(V15_AUDIT.read_text())
    if not all(audit["predictions"].values()) or audit["causal_outcomes_opened"]:
        raise RuntimeError("v15 capability-only ancestry changed")

    experiment.fresh = fresh
    experiment.PRIOR = PRIOR
    experiment.BUILDER = BUILDER
    experiment.OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v1_result.json"
    experiment.CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v16_capability_v1"
    experiment.RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v16_capability_result_v1"
    experiment.ROWS_SHA256 = ROWS_SHA256
    experiment.EXPECTED = {
        "prior": EXPECTED["prior"], "builder": EXPECTED["builder"],
        "head_atlas": experiment.EXPECTED["head_atlas"],
    }
    original_write = experiment.atomic_create_json

    def write_result(_path, result):
        result["authority_sha256"] = EXPECTED
        result["predictions"]["pred_c_joint_capable_population"] = all(
            len(result["jointly_capable_row_ids"][panel]) >= 24 for panel in ("A1", "A2")
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
