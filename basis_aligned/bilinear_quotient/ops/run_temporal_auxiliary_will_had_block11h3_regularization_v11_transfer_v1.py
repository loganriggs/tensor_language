#!/usr/bin/env python3
"""Zero-refit v11 transfer test of previously frozen rank-one DAS regularizers."""

# BQGATE: EXPERIMENT pred_a_authority_capability_closure_and_price pred_b_regularization_reduces_unregularized_overfit pred_c_aligned_beats_dim_out_of_task pred_d_aligned_preserves_behavioral_usefulness pred_e_regularization_ranking_is_construction_stable
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v11 as fresh
import run_temporal_auxiliary_will_had_block11h3_regularization_fresh_transfer_v1 as experiment


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_block11h3_regularization_v11_transfer_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v11.py"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v11_capability_v1_result.json"
PARENT = ROOT / "ops/run_temporal_auxiliary_will_had_block11h3_regularization_fresh_transfer_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_regularization_v11_transfer_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.block11h3_regularization_v11_transfer_v1"
REGISTERED_PREDICTIONS = (
    "pred_a_authority_capability_closure_and_price",
    "pred_b_regularization_reduces_unregularized_overfit",
    "pred_c_aligned_beats_dim_out_of_task",
    "pred_d_aligned_preserves_behavioral_usefulness",
    "pred_e_regularization_ranking_is_construction_stable",
)
EXPECTED_WRAPPER = {
    "prior": "8b5b667e494a8693e45de159c38696380c245e491021444550be3b7422494103",
    "builder": "f75b17669a5fc5299d21f5b44e91530c03c71d75181683c7b6728cb95c862450",
    "capability": "0330dc5a4f85bc68c4da6f98af2f4208335e65c644ddedd5d8cc487368091026",
    "parent": "104ae7f3ca5e0c998c0218c5b96579278f2023199fbba67cd94f64c9c5d60625",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure() -> None:
    paths = {"prior": PRIOR, "builder": BUILDER, "capability": CAPABILITY, "parent": PARENT}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED_WRAPPER:
        raise RuntimeError("v11 regularization-transfer authority changed")
    capability = json.loads(CAPABILITY.read_text())
    if (capability.get("terminal") != "manifest"
            or not all(capability.get("predictions", {}).values())):
        raise RuntimeError("v11 capability receipt is not valid")

    experiment.candidate = fresh
    experiment.PRIOR = PRIOR
    experiment.BUILDER = BUILDER
    experiment.OUT = OUT
    experiment.CANDIDATE_ID = CANDIDATE_ID
    experiment.EXPECTED = {
        "prior": EXPECTED_WRAPPER["prior"],
        "builder": EXPECTED_WRAPPER["builder"],
        "discovery_builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
        "aligned": "3aea84323bae1c2e46a430ef5f08b838826504693e6b1ba8a05027ca065b379d",
        "regularized": "f7d53dd6530dbdbebba7610236adc862b3c595bd83fb6c1b24d8fd4365543163",
        "scalar_axis": "4f345907c41222ebeec33c3a860052a0a8f39166655da0354f69e79a1c577fc5",
        "evaluator": "966fc3b4bafba272ca5702a934635f6ae033abc8c1575cefd1390fda2b1cdc11",
        "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
        "unit_lib": "079780fef43db5e2a2c6bd2ef3a5ab18833bbd5cd5c6de5c42d4ea7a5b246f80",
    }
    original_write = experiment.atomic_create_json

    def write_result(_path, result):
        result["schema"] = "temporal_auxiliary_block11h3_regularization_v11_transfer_result_v1"
        original_write(OUT, result)

    experiment.atomic_create_json = write_result


def main() -> None:
    configure()
    experiment.main()


if __name__ == "__main__":
    main()
