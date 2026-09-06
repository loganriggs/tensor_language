#!/usr/bin/env python3
"""Exact-string-only correction of suffix-MLP v1 design audit."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_authority pred_b_design_mismatch pred_c_exact_disposition
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit_v2.json"
AUDIT_V1 = ROOT / "circuits/followups/aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit_result.json"
SCIENCE_PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_mlp11_15_bilinear_compression_split_v1.json"
SCIENCE_RUNNER = ROOT / "ops/run_aspectual_anchor_mlp11_15_bilinear_compression_split_v1.py"
SCIENCE_RESULT = ROOT / "circuits/followups/aspectual_anchor_mlp11_15_bilinear_compression_split_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit_v2_result.json"
EXPECTED = {
    PRIOR: "bbf15abefb25d24f11b26d8171155bd755163873281c7be6298df78a449110c4",
    AUDIT_V1: "57bd72f9f5b663e9481063a24b9f0c338bb2b86cd72f1531a5b64aea4fef2f8e",
    SCIENCE_PRIOR: "e0f6f263fce698c93378bbf9ecfda8274ac113119d5e9fe0efa4a0f8dc254622",
    SCIENCE_RUNNER: "4114c79ca08e1fb5e293481221a4a4249bc18c7f3dc7018b393d751be5f18b84",
    SCIENCE_RESULT: "960bde3deb6b416212ff1be616887cc1b68f546fc31c0619a89be3aa6bef3bdf",
}
CANDIDATE_ID = "aspectual_anchor.has_vs_had.mlp11_15_bilinear_compression_v1_design_audit_v2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "model_forwards": 0, "example_evaluations": 0}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    audit_v1 = json.loads(AUDIT_V1.read_text())
    science_prior = json.loads(SCIENCE_PRIOR.read_text())
    science_result = json.loads(SCIENCE_RESULT.read_text())
    source = SCIENCE_RUNNER.read_text()
    observed = {str(path.relative_to(ROOT)): sha(path) for path in EXPECTED}
    audit_predictions = audit_v1.get("predictions", {})
    failed_audit_key = "_".join(("pred", "c", "fail", "closed"))
    pred_a = (
        all(observed[str(path.relative_to(ROOT))] == digest for path, digest in EXPECTED.items())
        and prior.get("candidate_id") == CANDIDATE_ID
        and audit_v1.get("terminal") == "invalid" and science_result.get("terminal") == "screen"
        and len(audit_predictions) == 3
        and sum(value is True for value in audit_predictions.values()) == 2
        and audit_predictions.get(failed_audit_key) is False
    )
    pred_b = (
        science_prior["frozen_design"]["source_bank_by_boundary"] == {
            "11": ["determiner", "period", "self"],
            "15": ["period", "determiner", "self"],
        }
        and "tuple(source_parent.ROLES[index] for index in range(len(source_parent.ROLES)))" in source
    )
    pred_c = (
        json.loads((ROOT / "circuits/prior_art/aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit.json").read_text())["terminal"]["screen"]
        == "A-C pass: diagnose a design mismatch and invalidate the v1 scientific release."
        and prior["correction"].startswith("Replace only v1 audit")
        and prior["price"] == {"model_forwards": 0, "example_evaluations": 0, "fit_parameters": 0}
    )
    predictions = {
        "pred_a_authority": pred_a,
        "pred_b_design_mismatch": pred_b,
        "pred_c_exact_disposition": pred_c,
    }
    terminal = "screen" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_mlp11_15_bilinear_compression_v1_design_audit_result_v2",
        "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prior_art_sha256": sha(PRIOR), "authority_sha256": observed,
        "predictions": predictions, "terminal": terminal,
        "scientific_disposition": "v1_superseded_as_invalid" if terminal == "screen" else "withhold",
        "reason": "all_six_roles_substituted_for_frozen_three_role_banks" if terminal == "screen" else "audit_failed",
        "price": prior["price"],
    }
    atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
