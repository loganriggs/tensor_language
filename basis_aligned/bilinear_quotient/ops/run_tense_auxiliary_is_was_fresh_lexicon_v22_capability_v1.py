#!/usr/bin/env python3
"""Capability-only gate for the untouched reader-contracted-confirmation v22 bank."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
import hashlib
import json
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v22 as fresh
import run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v22_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v22.py"
V21_RESULT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v21_capability_v1_result.json"
ATLAS_AUDIT = ROOT / "circuits/followups/temporal_iswas_v21_m11_complete_upstream_writer_module_head_atlas_v1_instrument_audit.json"
BASE_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v22_capability_v1_result.json"
EXPECTED = {
    "prior": "9bad181ec60aa94b811757ac810c324baa4e90edfa3003a3602d17376fedb032",
    "builder": "d84f50b334c6a2b2af47273da7668fdea8269bbc615378985fd49547f2f65499",
    "v21_result": "9819224e3d83cc99039bbbe64f45e725050aa2381535b2ec655b08ca772a79d7",
    "atlas_audit": "0fe6ed8b69498cbe68d00457cd797b2ecdb2b8ed06d02855b6e27b1a3cbf2b69",
    "base_runner": "33e8afdcfa02379539bc6f018d662af5f13901d07676458414c9a25aecee8a3e",
}
ROWS_SHA256 = "7e205a0eb515f53bcb1287ff3c20c04c4b7479772bfdf1709ace1f75c763153a"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v22_capability_v1"
RESULT_SCHEMA = "tense_auxiliary_is_was_fresh_lexicon_v22_capability_result_v1"
JOINT_BAR = 12
PREDICTION_KEYS = (
    "pred_a_authority_novelty_and_exact_population",
    "pred_b_native_a_panel_capability",
    "pred_c_joint_capable_population",
    "pred_d_no_causal_outcome_access_and_exact_price",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "builder": BUILDER, "v21_result": V21_RESULT,
             "atlas_audit": ATLAS_AUDIT, "base_runner": BASE_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"v22 capability authority changed: {observed}")
    ancestry, audit = json.loads(V21_RESULT.read_text()), json.loads(ATLAS_AUDIT.read_text())
    if (ancestry.get("terminal") != "screen" or not all(ancestry.get("predictions", {}).values())
            or ancestry.get("causal_outcomes_opened") is not False
            or not audit.get("verdict", "").startswith("registered_union_selectivity_test_is_self_contradictory")):
        raise RuntimeError("v21 capability/audit ancestry changed")
    experiment.fresh, experiment.PRIOR, experiment.BUILDER = fresh, PRIOR, BUILDER
    experiment.OUT, experiment.CANDIDATE_ID = OUT, CANDIDATE_ID
    experiment.RESULT_SCHEMA, experiment.ROWS_SHA256 = RESULT_SCHEMA, ROWS_SHA256
    experiment.EXPECTED = {"prior": EXPECTED["prior"], "builder": EXPECTED["builder"],
                           "head_atlas": experiment.EXPECTED["head_atlas"]}
    original_write = experiment.atomic_create_json
    def write_result(_path, result):
        if set(result["predictions"]) != set(PREDICTION_KEYS):
            raise RuntimeError("v22 prediction inventory changed")
        if result["price"] != {"model_forwards": 2, "example_evaluations": 128,
                "interventions": 0, "transformer_backwards": 0, "model_updates": 0}:
            raise RuntimeError("v22 capability price changed")
        if any(len(result["jointly_capable_row_ids"][panel]) < JOINT_BAR
               for panel in ("A1", "A2")) != (result["predictions"][PREDICTION_KEYS[2]] is False):
            raise RuntimeError("v22 joint-capability predicate changed")
        result["authority_sha256"] = EXPECTED; result["causal_outcomes_opened"] = False
        original_write(OUT, result)
    experiment.atomic_create_json = write_result
    experiment.main()


if __name__ == "__main__": main()
