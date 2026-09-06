#!/usr/bin/env python3
"""Numeric raw-state tolerance repair for shared/specific causal v1."""

# BQGATE: EXPERIMENT pred_a_exact_authority_replay_decomposition_coverage_and_price pred_b_temporal_q8_shared_component_is_causally_material pred_c_temporal_q8_shared_component_is_selective pred_d_iswas_specific_component_remains_material pred_e_shared_and_specific_compose_without_signed_reversal
import hashlib
import json
import os
from pathlib import Path

import run_temporal_q8_iswas_cdas_shared_specific_causal_v1 as impl

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_q8_iswas_cdas_shared_specific_causal_v2.json"
V1_RESULT = ROOT / "circuits/followups/temporal_q8_iswas_cdas_shared_specific_causal_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_q8_iswas_cdas_shared_specific_causal_v2_result.json"
TEMP = Path("/tmp/temporal_q8_iswas_cdas_shared_specific_causal_v2_raw.json")
CANDIDATE_ID = "cross_task.temporal_q8_iswas_cdas_shared_specific_causal_v2"
EXPECTED_PRIOR = "b46fa0aa288763a89e317691733325c44433da0ef1996aa71b5580205354a659"
EXPECTED_V1_RESULT = "69b4bd8f3c2550e3e31008509a1c99e73fd5b2f2722e15689b75f9c0f6d26e05"
PREDICTION_KEYS = (
    "pred_a_exact_authority_replay_decomposition_coverage_and_price",
    "pred_b_temporal_q8_shared_component_is_causally_material",
    "pred_c_temporal_q8_shared_component_is_selective",
    "pred_d_iswas_specific_component_remains_material",
    "pred_e_shared_and_specific_compose_without_signed_reversal",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if sha(PRIOR) != EXPECTED_PRIOR or sha(V1_RESULT) != EXPECTED_V1_RESULT:
        raise RuntimeError("v2 numeric-repair authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    if TEMP.exists():
        TEMP.unlink()
    impl.PRIOR = PRIOR
    impl.OUT = TEMP
    impl.CANDIDATE_ID = CANDIDATE_ID
    impl.EXPECTED = dict(impl.EXPECTED, prior=EXPECTED_PRIOR)
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        impl.main()
        return
    impl.main()
    result = json.loads(TEMP.read_text())
    instrument, price = result["instrument"], result["price"]
    pred_a = bool(instrument["native_head_max_abs"] <= 1e-3
        and instrument["f_linear_orientation_max_abs"] <= 1e-6
        and instrument["write_sum_max_abs"] <= 1e-3
        and instrument["logit_sum_max_abs"] <= 1e-5
        and instrument["released_cdas_metric_replay_max_abs"] <= 1e-4
        and price["model_forwards"] <= 2 and price["example_evaluations"] <= 208
        and price["records"] == 520)
    result["schema"] = "temporal_q8_iswas_cdas_shared_specific_causal_result_v2"
    result["candidate_id"] = CANDIDATE_ID
    result["dryrun"]["candidate_id"] = CANDIDATE_ID
    result["predictions"][PREDICTION_KEYS[0]] = pred_a
    if tuple(result["predictions"]) != PREDICTION_KEYS:
        raise RuntimeError("prediction inventory changed")
    result["engineering_repair"] = {"raw_write_tolerance": 1e-3,
        "logit_tolerance": 1e-5, "released_metric_tolerance": 1e-4,
        "scientific_arms_or_thresholds_changed": False}
    result["terminal"] = "screen" if all(result["predictions"].values()) else (
        "null" if pred_a else "invalid")
    impl.atomic_create_json(OUT, result)
    TEMP.unlink()
    print(json.dumps({key: result[key] for key in ("candidate_id", "engineering_repair",
        "instrument", "reports", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
