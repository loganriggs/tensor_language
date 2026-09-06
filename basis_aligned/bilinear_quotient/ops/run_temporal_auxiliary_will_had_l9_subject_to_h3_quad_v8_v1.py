#!/usr/bin/env python3
"""Frozen v8 test of H0 as a reusable fourth L9-to-H3 component."""

# BQGATE: EXPERIMENT pred_a_authority_capability_capture_closure_and_price pred_b_writer_and_full_l9_subject_response_are_material pred_c_frozen_triple_remains_dominant pred_d_h0_is_reusable_remainder pred_e_quad_closes_selectively_and_behaviorally
import hashlib
import json
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v8 as fresh
import run_temporal_auxiliary_will_had_l9_subject_to_h3_weight_guided_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_l9_subject_to_h3_quad_v8_v1.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v8_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_quad_v8_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.l9_subject_to_h3_quad_v8_v1"
EXPECTED = {
    "prior": "345a4957704286d93b62387b65c53bfc4bbec9e65937ff42310de26db6d3383f",
    "capability": "fe9255aa8221fe68331bc49c43f1b59cf5909c599f96ff5f36bf653ea1162cff",
    "builder": "13c0ae6424cc936dfa4ccb6ec89cd696e4b2c1267c2a4ebeeebd1a313bb443cf",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure():
    observed = {"prior": sha(PRIOR), "capability": sha(CAPABILITY),
                "builder": sha(Path(fresh.__file__))}
    if observed != EXPECTED:
        raise RuntimeError(f"v8 quad authority changed: {observed}")
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
        "weight_pair_h1_h4": (1, 4, 7), "quad_h0_h1_h4_h7": (0, 1, 4, 7),
        "pair_complement": (2, 3, 5, 6, 8), "all_heads": parent.HEADS}
    parent.FORWARDS = 38
    parent.EVALUATIONS = 1140
    parent.RECORDS = 780
    parent.EXPECTED = {**parent.EXPECTED, "prior": EXPECTED["prior"],
        "capability": EXPECTED["capability"], "builder": EXPECTED["builder"]}
    original_write = parent.atomic_create_json

    def write_result(_path, result):
        fractions = result["projected_fraction_of_all_heads"]
        behavior = result["behavior_summaries"]
        a = result["predictions"]["pred_a_authority_capability_capture_closure_and_price"]
        b = result["predictions"]["pred_b_writer_and_full_l9_subject_response_are_material"]
        c = all(fractions[p]["weight_pair_h1_h4"] >= 0.85 for p in ("A1", "A2"))
        d = (fractions["A2"]["head:0"] >= 0.05
             and fractions["A2"]["quad_h0_h1_h4_h7"]
                 - fractions["A2"]["weight_pair_h1_h4"] >= 0.03)
        e = all(fractions[p]["quad_h0_h1_h4_h7"] >= 0.90
            and fractions[p]["pair_complement"] <= 0.15
            and abs(behavior[p]["quad_h0_h1_h4_h7"]["mean_recovery"]
                    - behavior[p]["all_heads"]["mean_recovery"])
                <= 0.20 * abs(behavior[p]["all_heads"]["mean_recovery"])
            for p in ("A1", "A2"))
        result["schema"] = "temporal_auxiliary_l9_subject_to_h3_quad_v8_result_v1"
        result["candidate_id"] = CANDIDATE_ID
        result["dryrun"]["candidate_id"] = CANDIDATE_ID
        result["predictions"] = {
            "pred_a_authority_capability_capture_closure_and_price": a,
            "pred_b_writer_and_full_l9_subject_response_are_material": b,
            "pred_c_frozen_triple_remains_dominant": c,
            "pred_d_h0_is_reusable_remainder": d,
            "pred_e_quad_closes_selectively_and_behaviorally": e,
        }
        result["terminal"] = ("invalid" if not a else "confirmation"
            if all(result["predictions"].values()) else "null" if not b
            else "unstable_remainder" if c else "distributed")
        original_write(OUT, result)

    parent.atomic_create_json = write_result


def main():
    configure()
    parent.main()


if __name__ == "__main__":
    main()
