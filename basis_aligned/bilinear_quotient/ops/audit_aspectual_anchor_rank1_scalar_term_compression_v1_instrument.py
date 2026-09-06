#!/usr/bin/env python3
"""Post-outcome, zero-forward audit of scalar-compression v1's float32 closure gate."""

# BQLANE: cpu
# BQGATE: AUDIT pred_a_authority pred_b_float32_noise_localized pred_c_scientific_disposition_unchanged
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_rank1_scalar_term_compression_v1_instrument_audit.json"
RELEASE = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v9_result.json"
MEDIATION = ROOT / "circuits/followups/aspectual_anchor_program_v8_rank1_carrier_mediation_v1_result.json"
SCIENCE_PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_rank1_scalar_term_compression_split_v1.json"
SCIENCE_RUNNER = ROOT / "ops/run_aspectual_anchor_rank1_scalar_term_compression_split_v1.py"
SCIENCE_RESULT = ROOT / "circuits/followups/aspectual_anchor_rank1_scalar_term_compression_split_v1_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_rank1_scalar_term_compression_v1_instrument_audit_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.rank1_scalar_term_compression_v1_instrument_audit"
EXPECTED = {
    RELEASE: "dc2cd67daa2fbed6ae9ddde33e9c44fae11b694e4f351b596b541878df9a9106",
    MEDIATION: "313ccce304a18f8b0d63547bb973964f0b6f93a506765e4bb87c31e40c128aa9",
    SCIENCE_PRIOR: "2a7d57dafa5f87f8405318cad35f6c47bda93116e8cf70948ba27202a10fe982",
    SCIENCE_RUNNER: "cf236cec36b8f6293477c5b0041d36ce537cace7bb55f38228cdd169ff66cdac",
    SCIENCE_RESULT: "4a55ef3da37b12722fabae41c9caaa7e8284fc0891ba4e15c5cfdeab40323b2d",
}
EXPECTED_PRIOR_SHA256 = "0966561f30808a4aed7769454e1fe8cf2ea2dcd2be053c4cd023c0ce87412ca3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps({"dryrun": True, "model_forwards": 0, "example_evaluations": 0}))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    prior = json.loads(PRIOR.read_text())
    release = json.loads(RELEASE.read_text())
    mediation = json.loads(MEDIATION.read_text())
    result = json.loads(SCIENCE_RESULT.read_text())
    observed = {str(path.relative_to(ROOT)): sha(path) for path in EXPECTED}
    pred_a = (
        sha(PRIOR) == EXPECTED_PRIOR_SHA256
        and all(observed[str(path.relative_to(ROOT))] == digest for path, digest in EXPECTED.items())
        and prior.get("candidate_id") == CANDIDATE_ID
        and release.get("terminal") == "release"
        and mediation.get("terminal") == "screen"
        and result.get("terminal") == "invalid"
    )
    score = result["score"]
    pred_b = (
        score["full_rank1_reference_logit_max_abs"] == 0.0
        and score["manual_base_scored_logit_max_abs"] == 0.0
        and score["writer_bilinear_tensor_reconstruction_max_abs"] <= 2.0e-3
        and score["mlp_bilinear_tensor_reconstruction_max_abs"] <= 5.0e-3
        and 1.0e-4 < score["tensor_closure_max_abs"] <= 5.0e-4
        and 1.0e-4 < score["scalar_closure_max_abs"] <= 5.0e-4
    )
    scalar = score["fresh_scalar_fraction"]
    scored = score["fresh_scored_fraction"]
    pred_c = (
        result["predictions"]["pred_c_fresh_scalar_compression"] is False
        and scalar["A1"] < 0.80
        and scalar["A2"] >= 0.80
        and scored["A1"] >= 0.80
        and scored["A2"] >= 0.80
        and score["selected_terms"] == [
            "carried_resid10", "attention11", "mlp11_left_change",
            "mlp11_right_change", "mlp15_left_change",
        ]
    )
    predictions = {
        "pred_a_authority": pred_a,
        "pred_b_float32_noise_localized": pred_b,
        "pred_c_scientific_disposition_unchanged": pred_c,
    }
    terminal = "screen" if all(predictions.values()) else "invalid"
    value = {
        "schema": "aspectual_anchor_rank1_scalar_term_compression_v1_instrument_audit_result_v1",
        "candidate_id": CANDIDATE_ID,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evidence_class": "post_outcome_instrument_audit",
        "prior_art_sha256": sha(PRIOR),
        "authority_sha256": observed,
        "predictions": predictions,
        "terminal": terminal,
        "scientific_disposition": "five_term_compression_null" if terminal == "screen" else "withhold",
        "reason": "float32_closure_gate_invalid_but_frozen_fresh_A1_scalar_gate_misses" if terminal == "screen" else "audit_failed",
        "unchanged_scientific_measurements": {
            "selected_terms": score["selected_terms"],
            "fresh_scalar_fraction": scalar,
            "fresh_scalar_alignment": score["fresh_scalar_alignment"],
            "fresh_scored_fraction": scored,
        },
        "price": prior["price"],
    }
    atomic_create_json(OUT, value)
    print(json.dumps({"terminal": terminal, "scientific_disposition": value["scientific_disposition"], "predictions": predictions, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
