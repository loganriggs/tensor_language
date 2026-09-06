#!/usr/bin/env python3
"""Post-outcome repair replication of the sparse-suffix missing-block screen."""

# BQGATE: EXPERIMENT pred_a_authority_split_capability_and_instruments pred_b_positive_selection_missing_signal pred_c_two_block_increment_compression pred_d_total_program_sufficiency pred_e_exact_coverage
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_sparse_suffix_missing_block_compression_split_v2.json"
AUDIT = ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_missing_block_v1_design_audit_result.json"
OLD_RUNNER = ROOT / "ops/run_aspectual_anchor_sparse_suffix_missing_block_compression_split_v1.py"
OUT = ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_missing_block_compression_split_v2_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.sparse_suffix_missing_block_compression_split_v2"
EXPECTED_PRIOR_SHA256 = "666ade424a23df52bbac201f0d7f122956a0419876c049c193eeeb01e05db562"
EXPECTED_AUDIT_SHA256 = "8537275c7ac7c3d30bc1c15887f1aa82765b0f436c7cfd843e5803c5cec3ff38"
EXPECTED_OLD_RUNNER_SHA256 = "c8a20f6648078863f313329b473b3b246db06313f23a6e1e03c3394b35e34197"
PREDICTION_SCHEMA = {
    "pred_a_authority_split_capability_and_instruments": None,
    "pred_b_positive_selection_missing_signal": None,
    "pred_c_two_block_increment_compression": None,
    "pred_d_total_program_sufficiency": None,
    "pred_e_exact_coverage": None,
}


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ExperimentError(f"expected exactly one immutable-engine match: {old[:60]!r}")
    return source.replace(old, new)


def main() -> None:
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("repair prior hash changed")
    if sha(AUDIT) != EXPECTED_AUDIT_SHA256:
        raise ExperimentError("design audit hash changed")
    if sha(OLD_RUNNER) != EXPECTED_OLD_RUNNER_SHA256:
        raise ExperimentError("v1 engine hash changed")
    prior = json.loads(PRIOR.read_text())
    audit = json.loads(AUDIT.read_text())
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or prior.get("evidence_class", "").split(";")[0] != "post_outcome_repair_replication"
        or audit.get("terminal") != "screen"
        or audit.get("scientific_disposition") != "v1_invalid_control_incommensurate"
    ):
        raise ExperimentError("repair authority changed")

    source = OLD_RUNNER.read_text()
    source = replace_once(source,
        'PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_sparse_suffix_missing_block_compression_split_v1.json"',
        'PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_sparse_suffix_missing_block_compression_split_v2.json"')
    source = replace_once(source,
        'OUT = ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_missing_block_compression_split_v1_result.json"',
        'OUT = ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_missing_block_compression_split_v2_result.json"')
    source = replace_once(source,
        'CANDIDATE_ID = "aspectual_anchor.has_vs_had.sparse_suffix_missing_block_compression_split_v1"',
        f'CANDIDATE_ID = "{CANDIDATE_ID}"')
    source = replace_once(source,
        'EXPECTED_PRIOR_SHA256 = "2818b4116295a987d8ff4c5a5ce487bb730053230458146e5688f4deebb2649f"',
        f'EXPECTED_PRIOR_SHA256 = "{EXPECTED_PRIOR_SHA256}"')
    source = replace_once(source,
        'experiment_id="aspectual-anchor-sparse-suffix-missing-block-compression-split-v1"',
        'experiment_id="aspectual-anchor-sparse-suffix-missing-block-compression-split-v2"')
    source = replace_once(source,
        '"schema": "aspectual_anchor_sparse_suffix_missing_block_compression_split_dryrun_v1",',
        '"schema": "aspectual_anchor_sparse_suffix_missing_block_compression_split_dryrun_v2",')
    source = replace_once(source,
        '"schema": "aspectual_anchor_sparse_suffix_missing_block_compression_split_result_v1",',
        '"schema": "aspectual_anchor_sparse_suffix_missing_block_compression_split_result_v2",')
    source = replace_once(source,
        'and writer_tensor_error_max_abs <= 2.0e-3\n        and all_omitted_writer_logit_max_abs <= 0.125',
        'and writer_tensor_error_max_abs <= 2.0e-3')
    source = replace_once(source,
        '"' + "pred_a_authority_split_capability_and_dense_control" + '": pred_a,',
        f'"{tuple(PREDICTION_SCHEMA)[0]}": pred_a,')
    source = replace_once(source,
        '"' + "pred_c_disjoint_two_block_increment_compression" + '": pred_c,',
        f'"{tuple(PREDICTION_SCHEMA)[2]}": pred_c,')
    source = replace_once(source,
        '"' + "pred_d_disjoint_total_program_sufficiency" + '": pred_d,',
        f'"{tuple(PREDICTION_SCHEMA)[3]}": pred_d,')
    source = replace_once(source,
        '"sparse_null_sha256": EXPECTED_SPARSE_RESULT_SHA256,',
        '"sparse_null_sha256": EXPECTED_SPARSE_RESULT_SHA256,\n        "repair_design_audit_sha256": "8537275c7ac7c3d30bc1c15887f1aa82765b0f436c7cfd843e5803c5cec3ff38",\n        "evidence_class": "post_outcome_repair_replication",')

    namespace = {"__name__": "aspectual_anchor_sparse_suffix_missing_block_compression_split_v2_engine", "__file__": str(OLD_RUNNER)}
    exec(compile(source, str(OLD_RUNNER), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
