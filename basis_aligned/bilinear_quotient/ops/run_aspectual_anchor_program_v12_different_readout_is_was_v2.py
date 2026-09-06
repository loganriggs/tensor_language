#!/usr/bin/env python3
"""Capability-authorized v2 wrapper around the frozen cross-readout executor."""

# BQGATE: EXPERIMENT pred_a_authority_population_capability_and_exact_head pred_b_frozen_cross_readout_program pred_c_is_was_A_transfer pred_d_is_was_P_generalization pred_e_C_selectivity pred_f_exact_coverage_and_price
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import circuit_candidate_aspectual_different_readout_is_was_v2 as fresh
import run_aspectual_anchor_program_v12_different_readout_is_was_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_program_v12_different_readout_is_was_v2.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v2.py"
CAPABILITY = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v2_capability_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v2_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.program_v12_different_readout_is_was_v2"
EXPECTED_PRIOR_SHA256 = "2eea31884bf3705001168fe52adddb0e3a3cac9182d24dd1d3fe0e9df4ff38a2"
EXPECTED_CAPABILITY_SHA256 = "f76fbcd6174cc9e8e3f77352ee5461815156cdae421ecc43a0e0c3576b63af7e"
EXPECTED_BUILDER_SHA256 = "4ed62d06e2ffe5c471efc70eeaa35c7524a66923f19c64c803faef7200d4a62f"
EXPECTED_ROWS_SHA256 = "fef8174cd57a1382d87ae47ddb06ddebd8059e96e4f71472525a066a3048f911"
REGISTERED_PREDICTIONS = {
    "pred_a_authority_population_capability_and_exact_head": "delegated unchanged to the audited v1 executor",
    "pred_b_frozen_cross_readout_program": "delegated unchanged to the audited v1 executor",
    "pred_c_is_was_A_transfer": "delegated unchanged to the audited v1 executor",
    "pred_d_is_was_P_generalization": "delegated unchanged to the audited v1 executor",
    "pred_e_C_selectivity": "delegated unchanged to the audited v1 executor",
    "pred_f_exact_coverage_and_price": "delegated unchanged to the audited v1 executor",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure() -> None:
    prior = json.loads(PRIOR.read_text())
    capability = json.loads(CAPABILITY.read_text())
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256 or sha(CAPABILITY) != EXPECTED_CAPABILITY_SHA256 or sha(BUILDER) != EXPECTED_BUILDER_SHA256:
        raise parent.ExperimentError("v2 authority hash changed")
    if prior.get("candidate_id") != CANDIDATE_ID or capability.get("terminal") != "screen" or capability.get("causal_outcomes_opened") is not False or not all(capability["predictions"].values()) or not all(cell["passed"] for cell in capability["capability_cells"]):
        raise parent.ExperimentError("v2 capability authority does not authorize causal work")
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
