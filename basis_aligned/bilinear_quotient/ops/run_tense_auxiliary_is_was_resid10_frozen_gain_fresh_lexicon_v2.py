#!/usr/bin/env python3
"""Capability-authorized v6 use of the frozen q_is controller executor."""

# BQGATE: EXPERIMENT pred_a_authority_population_capability_and_exact_heads pred_b_frozen_upstream_program pred_c_new_lexicon_A_prediction pred_d_new_lexicon_P_generalization pred_e_new_lexicon_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6 as fresh
import run_tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_v1 as executor


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_v2.json"
UPSTREAM = ROOT / "circuits/followups/tense_auxiliary_is_was_resid10_margin_to_root_gain_v1_result.json"
Q_IS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v6_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v6.py"
EXECUTOR = ROOT / "ops/run_tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_v1.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_resid10_frozen_gain_fresh_lexicon_v2_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.resid10_frozen_gain_fresh_lexicon_v2"
EXPECTED_PRIOR_SHA256 = "8ecf27abecf158081af901c753802b0367e41c89efe2a01b1a3026e20d7987f6"
EXPECTED_ROWS_SHA256 = "4eee90d9f39f6997c4926a0e7f6baecc4134c06535fe307d0a38f936b75defd5"
EXPECTED = {
    UPSTREAM: "0c52305cf9ec3bba5bdc2f9ceedf2ec4ab047ab68a5accca4a53b3a6071f60f6",
    Q_IS: "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    CAPABILITY: "86ec66fa81346e61382c951e46899236ee1b7b7ec32c16948936fd9de6f77940",
    BUILDER: "b8541360334bd2793a02fae525a94dda05ce600fd4de5b6c3d953063d4c6b0ae",
    EXECUTOR: "9675ddd115e69161c135ef65d25808cce805d708825609a20aaedf1a5af1d4d4",
}
PREDICATE_KEYS = (
    "pred_a_authority_population_capability_and_exact_heads",
    "pred_b_frozen_upstream_program",
    "pred_c_new_lexicon_A_prediction",
    "pred_d_new_lexicon_P_generalization",
    "pred_e_new_lexicon_C_selectivity",
    "pred_f_exact_coverage_and_price",
)


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    upstream, qi, capability = json.loads(UPSTREAM.read_text()), json.loads(Q_IS.read_text()), json.loads(CAPABILITY.read_text())
    rows = fresh.build_rows()
    ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and prior["frozen_program"]["coefficients"] == executor.COEFFICIENTS
        and upstream.get("terminal") == "screen"
        and qi.get("terminal") == "screen"
        and capability.get("terminal") == "screen"
        and capability.get("causal_outcomes_opened") is False
        and all(cell["passed"] for cell in capability["capability_cells"])
        and upstream["basis_sha256"] == qi["basis"]["sha256"]
        and fresh.validate_rows(rows) == EXPECTED_ROWS_SHA256
        and capability["rows_sha256"] == EXPECTED_ROWS_SHA256
    )
    if not ok:
        raise ExperimentError("candidate, frozen coefficients, capability, basis, or v6 rows changed")
    return rows, qi


def configure_executor():
    executor.PRIOR = PRIOR
    executor.UPSTREAM = UPSTREAM
    executor.Q_IS = Q_IS
    executor.BUILDER = BUILDER
    executor.OUT = OUT
    executor.CANDIDATE_ID = CANDIDATE_ID
    executor.EXPECTED_PRIOR_SHA256 = EXPECTED_PRIOR_SHA256
    executor.EXPECTED_ROWS_SHA256 = EXPECTED_ROWS_SHA256
    executor.fresh = fresh
    executor.validate_static = validate_static


def main():
    configure_executor()
    executor.main()


if __name__ == "__main__":
    main()
