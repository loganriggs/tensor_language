#!/usr/bin/env python3
"""Frozen singleton atlas for the Later/Previously L9-to-H3 remainder."""

# BQGATE: EXPERIMENT pred_a_authority_capture_closure_and_price pred_b_writer_and_full_l9_subject_response_are_material pred_c_frozen_triple_remains_dominant pred_d_h3_is_largest_complement_singleton pred_e_h3_explains_material_a2_remainder
import hashlib
import json
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as fresh
import run_temporal_auxiliary_will_had_l9_subject_to_h3_weight_guided_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_l9_subject_to_h3_crosscue_remainder_v1.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_capability_manifest_v1_result.json"
CROSSCUE = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_greedy_crosscue_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_crosscue_remainder_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.l9_subject_to_h3_crosscue_remainder_v1"
EXPECTED = {
    "prior": "99e4ab3a0572e953c209a517c8e83dfc83dd1af1f6f2ced5ab37dddeca659331",
    "capability": "d59fdc0659f7db4632607a5fae860887bdf0f69f03af659b02ffcd6cc8c3be59",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "crosscue": "3c904aaec38bcc2577ae2c549fe5ad9eb818d3c664eee63210476f765b953479",
}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure():
    observed = {"prior": sha(PRIOR), "capability": sha(CAPABILITY),
        "builder": sha(Path(fresh.__file__)), "crosscue": sha(CROSSCUE)}
    if observed != EXPECTED:
        raise RuntimeError(f"remainder authority changed: {observed}")
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
    parent.ARM_HEADS = {**{f"head:{head}": (head,) for head in parent.HEADS},
        "weight_pair_h1_h4": (1, 4, 7), "pair_complement": (0, 2, 3, 5, 6, 8),
        "all_heads": parent.HEADS}
    parent.EVALUATIONS = 1062
    parent.RECORDS = 708
    parent.EXPECTED = {**parent.EXPECTED, "prior": EXPECTED["prior"],
        "capability": EXPECTED["capability"], "builder": EXPECTED["builder"]}
    original_write = parent.atomic_create_json

    def write_result(_path, result):
        fractions = result["projected_fraction_of_all_heads"]
        a = result["predictions"]["pred_a_authority_capability_capture_closure_and_price"]
        b = result["predictions"]["pred_b_writer_and_full_l9_subject_response_are_material"]
        c = all(fractions[p]["weight_pair_h1_h4"] >= 0.85 for p in ("A1", "A2"))
        complement = (0, 2, 3, 5, 6, 8)
        d = max(complement, key=lambda h: fractions["A2"][f"head:{h}"]) == 3
        e = fractions["A2"]["head:3"] >= 0.05
        result["schema"] = "temporal_auxiliary_l9_subject_to_h3_crosscue_remainder_result_v1"
        result["candidate_id"] = CANDIDATE_ID
        result["dryrun"]["candidate_id"] = CANDIDATE_ID
        result["crosscue_authority_sha256"] = EXPECTED["crosscue"]
        result["predictions"] = {
            "pred_a_authority_capture_closure_and_price": a,
            "pred_b_writer_and_full_l9_subject_response_are_material": b,
            "pred_c_frozen_triple_remains_dominant": c,
            "pred_d_h3_is_largest_complement_singleton": d,
            "pred_e_h3_explains_material_a2_remainder": e,
        }
        result["terminal"] = ("invalid" if not a else "screen" if all(result["predictions"].values())
            else "alternate_remainder" if b and c else "null" if not b else "distributed")
        original_write(OUT, result)

    parent.atomic_create_json = write_result


def main():
    configure()
    parent.main()


if __name__ == "__main__":
    main()
