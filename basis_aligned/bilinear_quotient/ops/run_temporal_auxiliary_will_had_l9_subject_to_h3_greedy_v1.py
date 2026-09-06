#!/usr/bin/env python3
"""Frozen greedy H1/H4/H7 closure of the L9 subject write into H3."""

# BQGATE: EXPERIMENT pred_a_authority_capability_capture_closure_and_price pred_b_writer_and_full_l9_subject_response_are_material pred_c_greedy_triple_closes_h3_response pred_d_six_head_complement_is_inert pred_e_greedy_triple_closes_behavior
import hashlib
from pathlib import Path

import run_temporal_auxiliary_will_had_l9_subject_to_h3_weight_guided_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_l9_subject_to_h3_greedy_v1.json"
PARENT_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_weight_guided_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_l9_subject_to_h3_weight_guided_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_greedy_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.l9_subject_to_h3_greedy_v1"
EXPECTED = {
    "prior": "1d31f7a2bc15eebfa5d6b5d8ddf3d2c1b720a823e861276718f9654681e0ba90",
    "parent_result": "65bbcf6be81bdba943ae0bc0cacf9570d4530b3b4a578e52160c7c8a39998a4f",
    "parent_runner": "b3bc00017d1eef6fdd73d9f77a909812974c187b1ca2e23ab38fb2699d2f27fa",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rename(mapping):
    mapping["greedy_h1_h4_h7"] = mapping.pop("weight_pair_h1_h4")
    mapping["greedy_complement"] = mapping.pop("pair_complement")


def configure():
    if {"prior": sha(PRIOR), "parent_result": sha(PARENT_RESULT),
            "parent_runner": sha(PARENT_RUNNER)} != EXPECTED:
        raise RuntimeError("greedy authority changed")
    parent.PRIOR = PRIOR
    parent.OUT = OUT
    parent.CANDIDATE_ID = CANDIDATE_ID
    parent.EXPECTED = {**parent.EXPECTED, "prior": EXPECTED["prior"]}
    parent.ARM_HEADS = {**{f"head:{head}": (head,) for head in parent.HEADS},
                        "weight_pair_h1_h4": (1, 4, 7),
                        "pair_complement": (0, 2, 3, 5, 6, 8),
                        "all_heads": parent.HEADS}
    original_write = parent.atomic_create_json

    def write_result(_path, result):
        fractions = result["projected_fraction_of_all_heads"]
        behavior = result["behavior_summaries"]
        pred_a = result["predictions"]["pred_a_authority_capability_capture_closure_and_price"]
        pred_b = result["predictions"]["pred_b_writer_and_full_l9_subject_response_are_material"]
        pred_c = all(fractions[panel]["weight_pair_h1_h4"] >= 0.90
                     for panel in ("A1", "A2"))
        pred_d = all(fractions[panel]["pair_complement"] <= 0.15
                     for panel in ("A1", "A2"))
        pred_e = all(abs(behavior[panel]["weight_pair_h1_h4"]["mean_recovery"]
                         - behavior[panel]["all_heads"]["mean_recovery"])
                     <= 0.20 * abs(behavior[panel]["all_heads"]["mean_recovery"])
                     for panel in ("A1", "A2"))
        result["schema"] = "temporal_auxiliary_l9_subject_to_h3_greedy_result_v1"
        result["candidate_id"] = CANDIDATE_ID
        result["dryrun"]["candidate_id"] = CANDIDATE_ID
        result["dryrun"]["arms"] = [key.replace("weight_pair_h1_h4", "greedy_h1_h4_h7")
            .replace("pair_complement", "greedy_complement") for key in result["dryrun"]["arms"]]
        result["predictions"] = {
            "pred_a_authority_capability_capture_closure_and_price": pred_a,
            "pred_b_writer_and_full_l9_subject_response_are_material": pred_b,
            "pred_c_greedy_triple_closes_h3_response": pred_c,
            "pred_d_six_head_complement_is_inert": pred_d,
            "pred_e_greedy_triple_closes_behavior": pred_e,
        }
        result["terminal"] = ("invalid" if not pred_a else "screen"
                              if all(result["predictions"].values()) else "distributed"
                              if pred_b else "null")
        for panel in ("A1", "A2"):
            rename(result["behavior_summaries"][panel])
            rename(result["projected_h3_norm_means"][panel])
            rename(result["projected_fraction_of_all_heads"][panel])
        original_write(OUT, result)

    parent.atomic_create_json = write_result


def main():
    configure()
    parent.main()


if __name__ == "__main__":
    main()
