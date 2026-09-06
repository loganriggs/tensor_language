#!/usr/bin/env python3
"""Fail-closed postexecution design audit for suffix-MLP compression v1."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_authority pred_b_design_mismatch pred_c_fail_closed
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit.json"
SCIENCE_PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp11_15_bilinear_compression_split_v1.json"
RUNNER = ROOT / "ops/run_aspectual_anchor_mlp11_15_bilinear_compression_split_v1.py"
RESULT = ROOT / "circuits/followups/aspectual_anchor_mlp11_15_bilinear_compression_split_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit_result.json"
EXPECTED = {
    PRIOR: "b75fe69aaba7d59ff5ca5596e15de9f2a62122274978d18b40ad7bd39d3ae77c",
    SCIENCE_PRIOR: "e0f6f263fce698c93378bbf9ecfda8274ac113119d5e9fe0efa4a0f8dc254622",
    RUNNER: "4114c79ca08e1fb5e293481221a4a4249bc18c7f3dc7018b393d751be5f18b84",
    RESULT: "960bde3deb6b416212ff1be616887cc1b68f546fc31c0619a89be3aa6bef3bdf",
}
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp11_15_bilinear_compression_v1_design_audit"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "model_forwards": 0, "example_evaluations": 0}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    observed = {str(path.relative_to(ROOT)): sha(path) for path in EXPECTED}
    audit_prior = json.loads(PRIOR.read_text())
    science_prior = json.loads(SCIENCE_PRIOR.read_text())
    result = json.loads(RESULT.read_text())
    source = RUNNER.read_text()
    pred_a = (
        all(observed[str(path.relative_to(ROOT))] == digest for path, digest in EXPECTED.items())
        and audit_prior.get("candidate_id") == CANDIDATE_ID
        and result.get("terminal") == "screen"
    )
    frozen_banks = science_prior["frozen_design"]["source_bank_by_boundary"]
    pred_b = (
        frozen_banks == {
            "11": ["determiner", "period", "self"],
            "15": ["period", "determiner", "self"],
        }
        and "tuple(source_parent.ROLES[index] for index in range(len(source_parent.ROLES)))" in source
        and '"source_roles": ["cue", "last", "period", "determiner", "self", "other"]' in SOURCE_RELEASE_TEXT
    )
    pred_c = (
        audit_prior["price"] == {"model_forwards": 0, "example_evaluations": 0, "fit_parameters": 0}
        and result.get("terminal") == "screen"
        and audit_prior["terminal"]["screen"].endswith("corrected v2.")
    )
    predictions = {
        "pred_a_authority": pred_a,
        "pred_b_design_mismatch": pred_b,
        "pred_c_fail_closed": pred_c,
    }
    terminal = "screen" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit_result_v1",
        "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": sha(PRIOR), "authority_sha256": observed,
        "predictions": predictions, "terminal": terminal,
        "diagnosis": "v1 passed all six source roles instead of each released three-role bank" if terminal == "screen" else "diagnosis_not_reproduced",
        "scientific_disposition": "supersede_v1_as_invalid_and_run_corrected_v2" if terminal == "screen" else "withhold",
        "price": audit_prior["price"],
    }
    atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "result": str(OUT)}, sort_keys=True))


SOURCE_RELEASE_TEXT = json.dumps({
    "source_roles": ["cue", "last", "period", "determiner", "self", "other"]
})


if __name__ == "__main__":
    main()
