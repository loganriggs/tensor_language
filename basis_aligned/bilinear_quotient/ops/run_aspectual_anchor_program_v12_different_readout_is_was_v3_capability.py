#!/usr/bin/env python3
"""Second-lexicon capability gate using the shared v2 capability executor."""

# BQGATE: EXPERIMENT pred_a_authority_and_exact_head pred_b_population_capability pred_c_exact_coverage
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import circuit_candidate_aspectual_different_readout_is_was_v3 as fresh
import run_aspectual_anchor_program_v12_different_readout_is_was_v2_capability as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_program_v12_different_readout_is_was_v3_capability.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v3.py"
V2 = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v2_result.json"
V12 = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v12_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v3_capability_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.program_v12_different_readout_is_was_v3_capability"
EXPECTED_PRIOR_SHA256 = "61538c3fc84d5bfa2bb50eae1b0b842586bc1bb102348e62732ab39b21dea53c"
EXPECTED_BUILDER_SHA256 = "ac240f13478168948814f59c0425270dd345ba03053332b72b34857e9a022638"
EXPECTED_ROWS_SHA256 = "35dc18a4e95764bc2126fc18d672d2f7666e4bc1c84a7bcc06d1299b820d7ad2"
EXPECTED_V2_SHA256 = "946131422a9d155551627617b3e5f4bb2f4331c5eb792aef96c996e07ab2840a"
EXPECTED_V12_SHA256 = "ed6afb3455bf5bfeea6e36f65ce33e9199290cb26540ee94da2c42accc785e7c"
REGISTERED_PREDICTIONS = {
    "pred_a_authority_and_exact_head": "delegated unchanged to capability executor",
    "pred_b_population_capability": "delegated unchanged to capability executor",
    "pred_c_exact_coverage": "delegated unchanged to capability executor",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_v3():
    prior = json.loads(PRIOR.read_text())
    v2 = json.loads(V2.read_text())
    v12 = json.loads(V12.read_text())
    rows = fresh.build_rows()
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or sha(BUILDER) != EXPECTED_BUILDER_SHA256 or sha(V2) != EXPECTED_V2_SHA256 or sha(V12) != EXPECTED_V12_SHA256:
        raise parent.ExperimentError("v3 authority hash changed")
    if prior.get("candidate_id") != CANDIDATE_ID or v2.get("terminal") != "screen" or v12.get("terminal") != "release" or fresh.validate_rows(rows) != EXPECTED_ROWS_SHA256 or len(rows) != 64:
        raise parent.ExperimentError("v3 candidate, parent screen, release, or rows changed")
    return rows


def configure() -> None:
    parent.PRIOR = PRIOR
    parent.BUILDER = BUILDER
    parent.OUT = OUT
    parent.CANDIDATE_ID = CANDIDATE_ID
    parent.EXPECTED_PRIOR_SHA256 = EXPECTED_PRIOR_SHA256
    parent.EXPECTED_ROWS_SHA256 = EXPECTED_ROWS_SHA256
    parent.fresh = fresh
    parent.validate_static = validate_v3


if __name__ == "__main__":
    configure()
    parent.main()
