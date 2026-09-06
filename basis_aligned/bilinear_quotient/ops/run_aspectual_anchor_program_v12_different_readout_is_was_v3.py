#!/usr/bin/env python3
"""Second-lexicon confirmation through the shared frozen cross-readout executor."""

# BQGATE: EXPERIMENT pred_a_authority_population_capability_and_exact_head pred_b_frozen_cross_readout_program pred_c_is_was_A_transfer pred_d_is_was_P_generalization pred_e_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import circuit_candidate_aspectual_different_readout_is_was_v3 as fresh
import run_aspectual_anchor_program_v12_different_readout_is_was_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_program_v12_different_readout_is_was_v3.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v3.py"
CAPABILITY = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v3_capability_result.json"
V2 = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v2_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v3_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.program_v12_different_readout_is_was_v3"
EXPECTED_PRIOR_SHA256 = "264342a0dc9316bfc00b9ac32c73b908e54babb594e81d0361c0285d36a27d16"
EXPECTED_CAPABILITY_SHA256 = "744d2fd3c8200ca00005357961df3d435a7a13dbdbd7c3a51a487daee76acec3"
EXPECTED_V2_SHA256 = "946131422a9d155551627617b3e5f4bb2f4331c5eb792aef96c996e07ab2840a"
EXPECTED_BUILDER_SHA256 = "ac240f13478168948814f59c0425270dd345ba03053332b72b34857e9a022638"
EXPECTED_ROWS_SHA256 = "35dc18a4e95764bc2126fc18d672d2f7666e4bc1c84a7bcc06d1299b820d7ad2"
REGISTERED_PREDICTIONS = {
    "pred_a_authority_population_capability_and_exact_head": "delegated unchanged to shared executor",
    "pred_b_frozen_cross_readout_program": "delegated unchanged to shared executor",
    "pred_c_is_was_A_transfer": "delegated unchanged to shared executor",
    "pred_d_is_was_P_generalization": "delegated unchanged to shared executor",
    "pred_e_C_selectivity": "delegated unchanged to shared executor",
    "pred_f_exact_coverage_and_price": "delegated unchanged to shared executor",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure() -> None:
    prior = json.loads(PRIOR.read_text())
    capability = json.loads(CAPABILITY.read_text())
    v2 = json.loads(V2.read_text())
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or sha(CAPABILITY) != EXPECTED_CAPABILITY_SHA256 or sha(V2) != EXPECTED_V2_SHA256 or sha(BUILDER) != EXPECTED_BUILDER_SHA256:
        raise parent.ExperimentError("v3 authority hash changed")
    if prior.get("candidate_id") != CANDIDATE_ID or capability.get("terminal") != "screen" or capability.get("causal_outcomes_opened") is not False or not all(capability["predictions"].values()) or not all(cell["passed"] for cell in capability["capability_cells"]) or v2.get("terminal") != "screen":
        raise parent.ExperimentError("v3 capability or v2 screen does not authorize confirmation")
    parent.PRIOR = PRIOR
    parent.BUILDER = BUILDER
    parent.OUT = OUT
    parent.CANDIDATE_ID = CANDIDATE_ID
    parent.EXPECTED_PRIOR_SHA256 = EXPECTED_PRIOR_SHA256
    parent.EXPECTED_ROWS_SHA256 = EXPECTED_ROWS_SHA256
    parent.EXPECTED = {
        parent.RELEASE: "ed6afb3455bf5bfeea6e36f65ce33e9199290cb26540ee94da2c42accc785e7c",
        BUILDER: EXPECTED_BUILDER_SHA256,
        parent.RANK1: "58b83a2714ae8d53cc799d5e6ae96c61cc476f22e09019e6e1f620581ff9a278",
    }
    parent.fresh = fresh


if __name__ == "__main__":
    configure()
    parent.main()
