#!/usr/bin/env python3
"""Capability-only gate for the pristine fixed-gain-confirmation v20 bank."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
import json
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v20 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v20_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v20.py"
V19_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v19_capability_v1_result.json"
GAIN_RESULT = ROOT / "circuits/followups/temporal_iswas_top16_fixed_gain_redundancy_compiler_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v20_capability_v1_result.json"
EXPECTED = {
    "prior": "80e23a3d8c1b867d9f9784431f49101d5a7d2a7472ac1599aa1e77cde97a0b82",
    "builder": "cafe9120c4ba8fdbeff27c5a2d05a247c2000641fd401529a4ca0a089137d1d9",
    "v19_result": "28f8f134d78279dd82cd7cca94f76e9cf35815bd1a59430ad34601fd0294fdd4",
    "gain_result": "a5b493763cea7fb0270f179a2f7e4cf5e8ca45354c037844d9d77ff6859d9b0e",
    "base_runner": "33e8afdcfa02379539bc6f018d662af5f13901d07676458414c9a25aecee8a3e",
}
ROWS_SHA256 = "9f9e07de8d35a04496f1e0e70326fb10c891a7379425d1d69188771c375e1331"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v20_capability_v1"
RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v20_capability_result_v1"
JOINT_BAR = 12
PREDICTION_KEYS = (
    "pred_a_authority_novelty_and_exact_population",
    "pred_b_native_a_panel_capability",
    "pred_c_joint_capable_population",
    "pred_d_no_causal_outcome_access_and_exact_price",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "builder": BUILDER, "v19_result": V19_RESULT,
             "gain_result": GAIN_RESULT, "base_runner": BASE_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"v20 capability authority changed: {observed}")
    ancestry = json.loads(V19_RESULT.read_text())
    calibration = json.loads(GAIN_RESULT.read_text())
    if (ancestry.get("terminal") != "screen" or not all(ancestry.get("predictions", {}).values())
            or ancestry.get("causal_outcomes_opened") is not False):
        raise RuntimeError("v19 capability-only ancestry changed")
    if (calibration.get("terminal") != "fixed_gain_calibration_screen"
            or not all(calibration.get("predictions", {}).values())):
        raise RuntimeError("fixed-gain calibration ancestry changed")
    experiment.fresh, experiment.PRIOR, experiment.BUILDER = fresh, PRIOR, BUILDER
    experiment.OUT, experiment.CANDIDATE_ID = OUT, CANDIDATE_ID
    experiment.RESULT_SCHEMA, experiment.ROWS_SHA256 = RESULT_SCHEMA, ROWS_SHA256
    experiment.EXPECTED = {"prior": EXPECTED["prior"], "builder": EXPECTED["builder"],
                           "head_atlas": experiment.EXPECTED["head_atlas"]}
    original_write = experiment.atomic_create_json
    def write_result(_path, result):
        if set(result["predictions"]) != set(PREDICTION_KEYS):
            raise RuntimeError("v20 prediction inventory changed")
        if result["price"] != {"model_forwards": 2, "example_evaluations": 128,
                "interventions": 0, "transformer_backwards": 0, "model_updates": 0}:
            raise RuntimeError("v20 capability price changed")
        if any(len(result["jointly_capable_row_ids"][panel]) < JOINT_BAR
               for panel in ("A1", "A2")) != (
                   result["predictions"][PREDICTION_KEYS[2]] is False):
            raise RuntimeError("v20 joint-capability predicate changed")
        result["authority_sha256"] = EXPECTED
        original_write(OUT, result)
    experiment.atomic_create_json = write_result
    experiment.main()


if __name__ == "__main__": main()
