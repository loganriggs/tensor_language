#!/usr/bin/env python3
"""Independent is/was carrier fit through the audited rank-one DAS executor."""

# BQGATE: EXPERIMENT pred_a_authority_head_and_rank pred_b_heldout_lexical_a1 pred_c_cross_construction_lexical_a2 pred_d_prospective_construction_transfer pred_e_same_answer_selectivity pred_f_exact_coverage
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import circuit_candidate_aspectual_different_readout_is_was_v2 as v2
import circuit_candidate_aspectual_different_readout_is_was_v3 as v3
import run_aspectual_anchor_das_resid18_rank1_transfer_v1 as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_das_resid18_rank1_transfer_v1.json"
V2 = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v2_result.json"
V3 = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v3_result.json"
V2_CAP = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v2_capability_result.json"
V3_CAP = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v3_capability_result.json"
V2_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v2.py"
V3_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v3.py"
DAS = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_das_resid18_rank1_transfer_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.das_resid18_rank1_transfer_v1"
EXPECTED_PRIOR_SHA256 = "d3a73799106abfb6e6c8aed4f309159494b600603abaa609ede6ebc6847ccfec"
EXPECTED = {
    V2: "946131422a9d155551627617b3e5f4bb2f4331c5eb792aef96c996e07ab2840a",
    V3: "b64f84c94021581e3c6d0ba5e35193a009faf064772a4e7c74639137d2be959b",
    V2_CAP: "f76fbcd6174cc9e8e3f77352ee5461815156cdae421ecc43a0e0c3576b63af7e",
    V3_CAP: "744d2fd3c8200ca00005357961df3d435a7a13dbdbd7c3a51a487daee76acec3",
    V2_BUILDER: "4ed62d06e2ffe5c471efc70eeaa35c7524a66923f19c64c803faef7200d4a62f",
    V3_BUILDER: "ac240f13478168948814f59c0425270dd345ba03053332b72b34857e9a022638",
    DAS: "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f",
}
REGISTERED_PREDICTIONS = {
    "pred_a_authority_head_and_rank": "delegated unchanged to audited DAS executor",
    "pred_b_heldout_lexical_a1": "delegated as heldout lexical A1",
    "pred_c_cross_construction_lexical_a2": "delegated as lexical A2",
    "pred_d_prospective_construction_transfer": "delegated as fresh A1/A2",
    "pred_e_same_answer_selectivity": "delegated unchanged",
    "pred_f_exact_coverage": "delegated unchanged",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_is_was():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise parent.ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise parent.ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    results = [json.loads(path.read_text()) for path in (V2, V3, V2_CAP, V3_CAP)]
    rows2, rows3 = v2.build_rows(), v3.build_rows()
    if prior.get("candidate_id") != CANDIDATE_ID or prior["frozen_design"]["rank"] != 1 or not all(result.get("terminal") == "screen" for result in results) or not all(all(cell["passed"] for cell in result["capability_cells"]) for result in results[2:]) or v2.validate_rows(rows2) != "fef8174cd57a1382d87ae47ddb06ddebd8059e96e4f71472525a066a3048f911" or v3.validate_rows(rows3) != "35dc18a4e95764bc2126fc18d672d2f7666e4bc1c84a7bcc06d1299b820d7ad2":
        raise parent.ExperimentError("candidate, rank, screens, capability, or rows changed")
    return rows2, rows3


def configure() -> None:
    parent.PRIOR = PRIOR
    parent.OUT = OUT
    parent.CANDIDATE_ID = CANDIDATE_ID
    parent.EXPECTED_PRIOR_SHA256 = EXPECTED_PRIOR_SHA256
    parent.validate_static = validate_is_was


if __name__ == "__main__":
    configure()
    parent.main()
