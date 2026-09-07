#!/usr/bin/env python3
"""Capability-only wrapper for the prospective v14 temporal authority."""
# BQGATE: EXPERIMENT pred_a_authority pred_b_joint_capability pred_c_exact_finite_coverage_and_price
import hashlib
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v14 as fresh
import run_temporal_auxiliary_will_had_fresh_v5_capability_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_v14_capability_v1.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v14_capability_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_v14_capability_v1"
EXPECTED = {
    "prior": "e782ee79673d0d7bf50517e7e96d079e6a00c7e035b5e710e19378db39fff84b",
    "builder": "3d59fbee0f7700d2c96a34e23337836808806fafac9600c6d2b12ea5135d7af9",
}
REGISTERED_PREDICTIONS = (
    "pred_a_authority",
    "pred_b_joint_capability",
    "pred_c_exact_finite_coverage_and_price",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if {"prior": sha(PRIOR), "builder": sha(Path(fresh.__file__))} != EXPECTED:
        raise RuntimeError("v14 capability authority changed")
    parent.candidate = fresh
    parent.PRIOR = PRIOR
    parent.BUILDER = Path(fresh.__file__)
    parent.OUT = OUT
    parent.CANDIDATE_ID = CANDIDATE_ID
    parent.EXPECTED = EXPECTED
    original_write = parent.atomic_create_json

    def write_result(_path, result):
        result["schema"] = "temporal_auxiliary_fresh_v14_capability_result_v1"
        result["candidate_id"] = CANDIDATE_ID
        result["dryrun"]["candidate_id"] = CANDIDATE_ID
        original_write(OUT, result)

    parent.atomic_create_json = write_result
    parent.main()


if __name__ == "__main__":
    main()
