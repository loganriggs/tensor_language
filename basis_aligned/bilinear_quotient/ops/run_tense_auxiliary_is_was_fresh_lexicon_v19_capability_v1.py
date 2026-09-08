#!/usr/bin/env python3
"""Capability-only gate for the pristine factor-confirmation v19 bank."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
import json
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v19 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v19_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v19.py"
V18_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v18_capability_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v19_capability_v1_result.json"
EXPECTED = {
    "prior": "7c0a3cd628c5fc23c38c2be8b3a0fa5b62b1d012c6ed5af30595ebd0671cfef7",
    "builder": "67916a1c83a019e037fc14415932560ba98cc5dd46a623216b43fd36d267d8b6",
    "v18_result": "e2f5a4368303a867646e6df7f67e3c64e98a3b3eb9a742ea7714ab413678020b",
    "base_runner": "33e8afdcfa02379539bc6f018d662af5f13901d07676458414c9a25aecee8a3e",
}
ROWS_SHA256 = "5d03c568c37c2d5a9e1e60ac99a98a9f92228d25546e98fc9632b9f240758e4f"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v19_capability_v1"
RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v19_capability_result_v1"
JOINT_BAR = 12
PREDICTION_KEYS = (
    "pred_a_authority_novelty_and_exact_population",
    "pred_b_native_a_panel_capability",
    "pred_c_joint_capable_population",
    "pred_d_no_causal_outcome_access_and_exact_price",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "builder": BUILDER, "v18_result": V18_RESULT,
             "base_runner": BASE_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"v19 capability authority changed: {observed}")
    ancestry = json.loads(V18_RESULT.read_text())
    if (ancestry.get("terminal") != "screen" or not all(ancestry.get("predictions", {}).values())
            or ancestry.get("causal_outcomes_opened") is not False):
        raise RuntimeError("v18 capability-only ancestry changed")
    experiment.fresh, experiment.PRIOR, experiment.BUILDER = fresh, PRIOR, BUILDER
    experiment.OUT, experiment.CANDIDATE_ID = OUT, CANDIDATE_ID
    experiment.RESULT_SCHEMA, experiment.ROWS_SHA256 = RESULT_SCHEMA, ROWS_SHA256
    experiment.EXPECTED = {"prior": EXPECTED["prior"], "builder": EXPECTED["builder"],
                           "head_atlas": experiment.EXPECTED["head_atlas"]}
    original_write = experiment.atomic_create_json
    def write_result(_path, result):
        if set(result["predictions"]) != set(PREDICTION_KEYS):
            raise RuntimeError("v19 prediction inventory changed")
        if result["price"] != {"model_forwards": 2, "example_evaluations": 128,
                "interventions": 0, "transformer_backwards": 0, "model_updates": 0}:
            raise RuntimeError("v19 capability price changed")
        if any(len(result["jointly_capable_row_ids"][panel]) < JOINT_BAR
               for panel in ("A1", "A2")) != (
                   result["predictions"][PREDICTION_KEYS[2]] is False):
            raise RuntimeError("v19 joint-capability predicate changed")
        result["authority_sha256"] = EXPECTED
        original_write(OUT, result)
    experiment.atomic_create_json = write_result
    experiment.main()


if __name__ == "__main__": main()
