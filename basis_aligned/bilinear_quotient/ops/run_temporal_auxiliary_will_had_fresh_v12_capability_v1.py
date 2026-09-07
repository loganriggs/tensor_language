#!/usr/bin/env python3
"""Capability-only wrapper for the sealed v12 temporal authority."""

# BQGATE: EXPERIMENT pred_a_authority pred_b_joint_capability pred_c_exact_finite_coverage_and_price
import hashlib
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v12 as fresh
import run_temporal_auxiliary_will_had_fresh_v5_capability_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_v12_capability_v1.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_v12_capability_v1"
EXPECTED = {
    "prior": "a2072b68a1854b8b2e82b3158b72aaf53f51e89dcb24532955ffe21b0a1f45b5",
    "builder": "4cf4624361ff2bd3c87cd987a7c4d16a1eeefb1c7f78287b78319862ab8d8de9",
}
REGISTERED_PREDICTIONS = (
    "pred_a_authority",
    "pred_b_joint_capability",
    "pred_c_exact_finite_coverage_and_price",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure():
    if {"prior": sha(PRIOR), "builder": sha(Path(fresh.__file__))} != EXPECTED:
        raise RuntimeError("v12 capability authority changed")
    parent.candidate = fresh
    parent.PRIOR = PRIOR
    parent.BUILDER = Path(fresh.__file__)
    parent.OUT = OUT
    parent.CANDIDATE_ID = CANDIDATE_ID
    parent.EXPECTED = EXPECTED
    original_write = parent.atomic_create_json

    def write_result(_path, result):
        result["schema"] = "temporal_auxiliary_fresh_v12_capability_result_v1"
        result["candidate_id"] = CANDIDATE_ID
        result["dryrun"]["candidate_id"] = CANDIDATE_ID
        original_write(OUT, result)

    parent.atomic_create_json = write_result


def main():
    configure()
    parent.main()


if __name__ == "__main__":
    main()
