#!/usr/bin/env python3
"""Zero-fit Later/Previously confirmation of the v4-selected L9H1/H4/H7 edge."""

# BQGATE: EXPERIMENT pred_a_authority_capture_self_clamp_and_price pred_b_complete_l9_removal_is_material pred_c_greedy_set_is_necessary_for_h3_response pred_d_complement_is_unnecessary_for_h3_response pred_e_greedy_set_is_necessary_for_behavior
import hashlib
import json
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fresh
import run_temporal_auxiliary_will_had_l9_subject_to_h3_greedy_necessity_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_l9_subject_to_h3_greedy_crosscue_v1.json"
DISCOVERY = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_greedy_necessity_v1_result.json"
DISCOVERY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_l9_subject_to_h3_greedy_necessity_v1.py"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_capability_manifest_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_greedy_crosscue_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.l9_subject_to_h3_greedy_crosscue_v1"
EXPECTED = {
    "prior": "eb268f9113f50b12cf2a4d81f559e2561f444de03c68aae576244395806c6a94",
    "discovery": "4c93e024299e6fff6fcd894ca7bef1b7365c0a56e1d3d9cb6daf996284306d32",
    "discovery_runner": "1d4be0b54e81e9f272622739aee4d623011cedafdef290e53442b4f9462893ca",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "capability": "d59fdc0659f7db4632607a5fae860887bdf0f69f03af659b02ffcd6cc8c3be59",
}
REGISTERED_PREDICTIONS = (
    "pred_a_authority_capture_self_clamp_and_price",
    "pred_b_complete_l9_removal_is_material",
    "pred_c_greedy_set_is_necessary_for_h3_response",
    "pred_d_complement_is_unnecessary_for_h3_response",
    "pred_e_greedy_set_is_necessary_for_behavior",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure():
    observed = {"prior": sha(PRIOR), "discovery": sha(DISCOVERY),
        "discovery_runner": sha(DISCOVERY_RUNNER), "builder": sha(Path(fresh.__file__)),
        "capability": sha(CAPABILITY)}
    if observed != EXPECTED:
        raise RuntimeError(f"cross-cue authority changed: {observed}")
    manifest = json.loads(CAPABILITY.read_text())
    allowed = {row_id for ids in manifest["jointly_capable_row_ids"].values() for row_id in ids}

    class FrozenPopulation:
        @staticmethod
        def build_rows():
            return [row for row in fresh.build_rows()
                    if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]

    parent.candidate = FrozenPopulation
    parent.PRIOR = PRIOR
    parent.GREEDY = DISCOVERY
    parent.GREEDY_RUNNER = DISCOVERY_RUNNER
    parent.CAPABILITY = CAPABILITY
    parent.BUILDER = Path(fresh.__file__)
    parent.OUT = OUT
    parent.CANDIDATE_ID = CANDIDATE_ID
    parent.RECORDS = 295
    parent.EXPECTED = {**parent.EXPECTED, "prior": EXPECTED["prior"],
        "greedy": EXPECTED["discovery"], "greedy_runner": EXPECTED["discovery_runner"],
        "capability": EXPECTED["capability"], "builder": EXPECTED["builder"]}
    original_write = parent.atomic_create_json

    def write_result(_path, result):
        result["schema"] = "temporal_auxiliary_l9_subject_to_h3_greedy_crosscue_result_v1"
        result["candidate_id"] = CANDIDATE_ID
        result["dryrun"]["candidate_id"] = CANDIDATE_ID
        result["crosscue_scope"] = {
            "cue_bank": "Later/Previously",
            "jointly_capable_rows": {"A1": 29, "A2": 30},
            "head_set_ood": True,
            "subspace_ood": False,
        }
        if result["terminal"] == "screen":
            result["terminal"] = "confirmation"
        original_write(OUT, result)

    parent.atomic_create_json = write_result


def main():
    configure()
    parent.main()


if __name__ == "__main__":
    main()
