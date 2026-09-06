#!/usr/bin/env python3
"""Frozen nine-head inventory repair for the block-8 subject attention screen."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_head_instrument pred_b_material_attention_branch_recurrence pred_c_shared_single_head_concentration pred_d_complete_head_ranking_reported pred_e_exact_zero_fit_price
from __future__ import annotations

import json
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import run_temporal_auxiliary_will_had_block8_subject_attention_heads_v1 as v1


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_block8_subject_attention_heads_v2.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8_subject_attention_heads_v2_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.block8_subject_attention_heads_v2"
EXPECTED_PRIOR_SHA256 = "f7219da9c2ebf7e147b07e7d93e233de3f278a6de856efdd9c0e7aa570fad4d0"
HEADS = tuple(range(9))
ARMS = tuple(f"head:{head:02d}" for head in HEADS) + ("full_heads", "direct_output")
PREDICATES = (
    "pred_a_authority_capability_exact_head_instrument",
    "pred_b_material_attention_branch_recurrence",
    "pred_c_shared_single_head_concentration",
    "pred_d_complete_head_ranking_reported",
    "pred_e_exact_zero_fit_price",
)


def configure_v2():
    if v1.sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise v1.ExperimentError("v2 prior-art hash changed")
    v1.PRIOR = PRIOR
    v1.OUT = OUT
    v1.CANDIDATE_ID = CANDIDATE_ID
    v1.EXPECTED = {**v1.EXPECTED, "prior": EXPECTED_PRIOR_SHA256}
    v1.HEADS = HEADS
    v1.ARMS = ARMS
    v1.MODEL_FORWARDS = 26
    v1.EXAMPLE_EVALUATIONS = 832
    v1.INTERVENTION_RECORDS = 704

    original_load = v1.producer.Bilin18TorchBackend.load

    def load_checked(device):
        backend = original_load(device)
        if backend.model.config.n_head != len(HEADS):
            raise v1.ExperimentError(
                f"frozen head inventory changed: {backend.model.config.n_head} != {len(HEADS)}"
            )
        return backend

    original_write = managed.atomic_create_json

    def write_v2(_path, result):
        result["schema"] = "temporal_auxiliary_will_had_block8_subject_attention_heads_result_v2"
        result["candidate_id"] = CANDIDATE_ID
        result["instrument"]["model_head_count"] = len(HEADS)
        original_write(OUT, result)

    v1.producer.Bilin18TorchBackend.load = load_checked
    v1.atomic_create_json = write_v2


def main():
    configure_v2()
    v1.main()


if __name__ == "__main__":
    main()
