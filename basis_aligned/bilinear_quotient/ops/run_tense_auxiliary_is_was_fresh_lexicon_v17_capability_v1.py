#!/usr/bin/env python3
"""Capability-only gate for the sealed A11/M11 edge-test v17 bank."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
import json
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v17 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v17_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v17.py"
V16_AUDIT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v2_audit_result.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v17_capability_v1_result.json"
EXPECTED = {
    "prior": "0fb40b0d3d266024d7c7fc02ee8dbe57126a3ddb881b9c2b2337e92f9f6de581",
    "builder": "7e59800324e5b6a4a9c5af564cbe8fb5cf642cfd6e702e1e97c32bc0cc54fe9e",
    "v16_audit": "a1c2baf0bd9548e189ccc4ba4d11c4905f85af1ff7013d408c143f6f2a6434e3",
    "base_runner": "33e8afdcfa02379539bc6f018d662af5f13901d07676458414c9a25aecee8a3e",
}
ROWS_SHA256 = "287e744a0fcdfd609149ee99ef2f280b485eeae2e08c20b1a10d269ef17b10e3"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v17_capability_v1"
RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v17_capability_result_v1"
JOINT_BAR = 12
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
        "v16_audit": V16_AUDIT,
        "base_runner": BASE_RUNNER,
    }
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"v17 capability authority changed: {observed}")
    audit = json.loads(V16_AUDIT.read_text())
    if (
        audit.get("terminal") != "manifest"
        or not all(audit.get("predictions", {}).values())
        or audit.get("causal_outcomes_opened") is not False
        or audit.get("corrected_joint_bar") != JOINT_BAR
    ):
        raise RuntimeError("v16 capability-only ancestry changed")

    experiment.fresh = fresh
    experiment.PRIOR = PRIOR
    experiment.BUILDER = BUILDER
    experiment.OUT = OUT
    experiment.CANDIDATE_ID = CANDIDATE_ID
    experiment.RESULT_SCHEMA = RESULT_SCHEMA
    experiment.ROWS_SHA256 = ROWS_SHA256
    experiment.EXPECTED = {
        "prior": EXPECTED["prior"],
        "builder": EXPECTED["builder"],
        "head_atlas": experiment.EXPECTED["head_atlas"],
    }
    original_write = experiment.atomic_create_json

    def write_result(_path, result):
        if tuple(result["predictions"]) != PREDICTION_KEYS:
            raise RuntimeError("v17 prediction inventory changed")
        if result["price"] != {
            "model_forwards": 2,
            "example_evaluations": 128,
            "interventions": 0,
            "transformer_backwards": 0,
            "model_updates": 0,
        }:
            raise RuntimeError("v17 capability price changed")
        if any(
            len(result["jointly_capable_row_ids"][panel]) < JOINT_BAR
            for panel in ("A1", "A2")
        ) != (result["predictions"][PREDICTION_KEYS[2]] is False):
            raise RuntimeError("v17 joint-capability predicate changed")
        result["authority_sha256"] = EXPECTED
        original_write(OUT, result)

    experiment.atomic_create_json = write_result
    experiment.main()


if __name__ == "__main__":
    main()
