#!/usr/bin/env python3
"""Paired v8 necessity confirmation of the frozen L9H1/H4/H7 core."""

# BQGATE: EXPERIMENT pred_a_authority_capture_self_clamp_and_price pred_b_complete_l9_removal_is_material pred_c_greedy_set_is_necessary_for_h3_response pred_d_complement_is_unnecessary_for_h3_response pred_e_greedy_set_is_necessary_for_behavior
import hashlib
import json
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v8 as fresh
import run_temporal_auxiliary_will_had_l9_subject_to_h3_greedy_necessity_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_l9_subject_to_h3_triple_v8_necessity_v1.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v8_capability_v1_result.json"
SUFFICIENCY = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_quad_v8_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_triple_v8_necessity_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.l9_subject_to_h3_triple_v8_necessity_v1"
EXPECTED = {
    "prior": "940ca29ed52d41e29620aac84c10e3154924bab6fb130c9a33009e0547224a0a",
    "capability": "fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff",
    "builder": "13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf",
    "sufficiency": "199a534dc2aea9607aeae4a4033f76b27a515e8b884cbdab057f236f2d59daac",
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
    observed = {"prior": sha(PRIOR), "capability": sha(CAPABILITY),
        "builder": sha(Path(fresh.__file__)), "sufficiency": sha(SUFFICIENCY)}
    if observed != EXPECTED:
        raise RuntimeError(f"v8 necessity authority changed: {observed}")
    manifest = json.loads(CAPABILITY.read_text())
    allowed = {row_id for ids in manifest["jointly_capable_row_ids"].values() for row_id in ids}

    class FrozenPopulation:
        @staticmethod
        def build_rows():
            return [row for row in fresh.build_rows()
                    if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]

    parent.candidate = FrozenPopulation
    parent.PRIOR = PRIOR
    parent.CAPABILITY = CAPABILITY
    parent.BUILDER = Path(fresh.__file__)
    parent.OUT = OUT
    parent.CANDIDATE_ID = CANDIDATE_ID
    parent.RECORDS = 300
    parent.EXPECTED = {**parent.EXPECTED, "prior": EXPECTED["prior"],
        "capability": EXPECTED["capability"], "builder": EXPECTED["builder"]}
    original_write = parent.atomic_create_json

    def write_result(_path, result):
        result["schema"] = "temporal_auxiliary_l9_subject_to_h3_triple_v8_necessity_result_v1"
        result["candidate_id"] = CANDIDATE_ID
        result["dryrun"]["candidate_id"] = CANDIDATE_ID
        result["paired_sufficiency_sha256"] = EXPECTED["sufficiency"]
        if result["terminal"] == "screen":
            result["terminal"] = "paired_confirmation"
        original_write(OUT, result)

    parent.atomic_create_json = write_result


def main():
    configure()
    parent.main()


if __name__ == "__main__":
    main()
