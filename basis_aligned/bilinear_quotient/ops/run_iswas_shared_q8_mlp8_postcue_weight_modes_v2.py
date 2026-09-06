#!/usr/bin/env python3
"""Numeric-only rerun of the MLP8 post-cue Q8 Down/product mode test."""

# BQGATE: EXPERIMENT pred_a_authority_exact_factor_weight_closure_finiteness_and_price pred_b_rank8_weight_modes_preserve_the_postcue_writer pred_c_rank8_complement_is_secondary pred_d_left_right_terms_explain_the_weight_mode_write pred_e_weight_interface_is_compressive_and_zero_fit
import hashlib
from pathlib import Path

import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as experiment

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_shared_q8_mlp8_postcue_weight_modes_v2.json"
V1_RESULT = ROOT / "circuits/followups/iswas_shared_q8_mlp8_postcue_weight_modes_v1_result.json"
BASE_RUNNER = ROOT / "ops/run_iswas_shared_q8_mlp8_postcue_weight_modes_v1.py"
EXPECTED = {
    "prior": "a33dcfd31490de0a5a566299de50bbbde5da4f737f50329abdd51e46029731e7",
    "v1_result": "89281f37a46514b0dcf024acfc6e09be8dc8537f298620c1787299f481f93470",
    "base_runner": "d82035e45f8818a7de73024500592f1677f887e8be5d8a2e617c28afc6c1238c",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if {"prior": sha(PRIOR), "v1_result": sha(V1_RESULT),
            "base_runner": sha(BASE_RUNNER)} != EXPECTED:
        raise RuntimeError("v2 numeric-repair authority changed")
    experiment.PRIOR = PRIOR
    experiment.OUT = ROOT / "circuits/followups/iswas_shared_q8_mlp8_postcue_weight_modes_v2_result.json"
    experiment.CANDIDATE_ID = "cross_task.iswas_shared_q8_mlp8_postcue_weight_modes_v2"
    experiment.RESULT_SCHEMA = "iswas_shared_q8_mlp8_postcue_weight_modes_result_v2"
    experiment.DIRECT_Q8_TOLERANCE = 1e-3
    experiment.EXPECTED = dict(experiment.EXPECTED, prior=EXPECTED["prior"])
    experiment.main()


if __name__ == "__main__":
    main()
