#!/usr/bin/env python3
"""Fail-closed audit of the v1 sparse-suffix dense-control semantics."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_authority pred_b_nonidentical_programs pred_c_failure_localized
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_sparse_suffix_missing_block_v1_design_audit.json"
SCIENCE_PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_sparse_suffix_missing_block_compression_split_v1.json"
SCIENCE_RUNNER = ROOT / "ops/run_aspectual_anchor_sparse_suffix_missing_block_compression_split_v1.py"
SCIENCE_RESULT = ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_missing_block_compression_split_v1_result.json"
SPARSE_PARENT = ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_recurrence_confirmation_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_sparse_suffix_missing_block_v1_design_audit_result.json"
EXPECTED = {
    SCIENCE_PRIOR: "2818b4116295a987d8ff4c5a5ce487bb730053230458146e5688f4deebb2649f",
    SCIENCE_RUNNER: "c8a20f6648078863f313329b473b3b246db06313f23a6e1e03c3394b35e34197",
    SCIENCE_RESULT: "cebdd3b1b15fb0117c86197d61b5cbc2e344acf748cbdc78970f3f6632963cca",
    SPARSE_PARENT: "db666e5e006d5ecb3300806399c682441ea92b990cb1155eaa76479899326ef2",
}
CANDIDATE_ID = "aspectual_anchor.has_vs_had.sparse_suffix_missing_block_v1_design_audit"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "model_forwards": 0, "example_evaluations": 0}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    result = json.loads(SCIENCE_RESULT.read_text())
    parent = json.loads(SPARSE_PARENT.read_text())
    source = SCIENCE_RUNNER.read_text()
    observed = {str(path.relative_to(ROOT)): sha(path) for path in EXPECTED}
    pred_a = (
        sha(PRIOR) == "76967c2fae2efc43e422909d52aaef0fec077579abc236af8bcd613abb9ced18"
        and all(observed[str(path.relative_to(ROOT))] == digest for path, digest in EXPECTED.items())
        and prior.get("candidate_id") == CANDIDATE_ID
        and result.get("terminal") == "invalid" and parent.get("terminal") == "null"
    )
    pred_b = (
        "if boundary in (11, 15):" in source
        and "projected_attention = self.projected_source_delta(" in source
        and "for factor in sparse_parent.SELECTED_MLP[boundary]:" in source
        and "elif boundary in selected_blocks:" in source
        and "writer_output, hybrid_capture" in source
        and parent["score"]["dense_to_writer_logit_max_abs"] <= 0.125
    )
    predictions_v1 = result.get("predictions", {})
    pred_c = (
        result["score"]["all_omitted_to_writer_logit_max_abs"] > 0.125
        and predictions_v1.get("pred_a_authority_split_capability_and_dense_control") is False
        and len(predictions_v1) == 5
        and sum(value is True for value in predictions_v1.values()) == 4
    )
    predictions = {
        "pred_a_authority": pred_a,
        "pred_b_nonidentical_programs": pred_b,
        "pred_c_failure_localized": pred_c,
    }
    terminal = "screen" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_sparse_suffix_missing_block_v1_design_audit_result_v1",
        "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": sha(PRIOR), "authority_sha256": observed,
        "predictions": predictions, "terminal": terminal,
        "scientific_disposition": "v1_invalid_control_incommensurate" if terminal == "screen" else "withhold",
        "reason": "compressed_block11_15_arm_compared_to_full_block11_15_writer" if terminal == "screen" else "audit_failed",
        "price": prior["price"],
    }
    atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
