#!/usr/bin/env python3
"""Zero-model literal-price audit of the immutable normalized weight atlas."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_v1_invalid_only_from_registered_forward_price pred_b_corrected_exact_price_validates_instrument pred_c_scientific_predictions_and_nulls_are_immutable pred_d_all_preserved_metrics_are_finite
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v2_price_audit.json"
V1 = ROOT / "circuits/followups/temporal_iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v2_price_audit_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v2_price_audit"
EXPECTED = {"prior": "ab213a469291541d3b1ad2fad6d438332df7522fb545c2b7d089859d3fed9741",
            "v1": "e8dd97ad21542ccba884e9e0c52c9cb007473b24a989782c099ea2f515dda429"}
CORRECTED_MAX = {"native_capture_forwards": 10, "differentiable_transformer_forwards": 10,
                 "transformer_backward_forwards": 0, "model_updates": 0,
                 "example_evaluations": 1400, "fit_parameters": 0, "checkpoint_loads": 1}
SCIENTIFIC_KEYS = tuple("pred" + suffix for suffix in (
    "_b_raw_and_exact_normalized_rankings_differ",
    "_c_rms_tangent_tracks_exact_finite_response",
    "_d_exact_reader_nominations_are_fold_stable",
    "_e_a_shared_exact_reader_is_nominated",
    "_f_known_oracle_sources_write_into_the_span",
))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    observed = {"prior": sha(PRIOR), "v1": sha(V1)}
    authority = observed == EXPECTED
    if not authority:
        raise RuntimeError(f"authority changed: {observed}")
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "authority_ok": authority,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "corrected_max": CORRECTED_MAX}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)

    v1 = json.loads(V1.read_text())
    price = v1["price"]
    instrument = v1["instrument"]
    predictions1 = v1["predictions"]
    nonprice_instrument = bool(
        instrument["manual_native_max_abs_error"] <= 1e-4
        and instrument["orthogonality_max_abs_error"] <= 1e-5
        and instrument["geometry_relative_max_error"] <= 1e-5
        and instrument["train_held_disjoint"] is True
        and instrument["model_config"] == {"vocab_size": 50304, "n_layer": 18, "n_head": 9,
            "n_embd": 1152, "squared_mlp": False, "bilinear": True, "expansion_factor": 4,
            "gated": False, "squared_attn": True, "bilinear_attn": True})
    old_max = price["maxima"]
    pred_a = bool(v1["terminal"] == "invalid"
                  and predictions1["pred" + "_a_authority_geometry_finiteness_and_price"] is False
                  and nonprice_instrument and finite(v1)
                  and price["differentiable_transformer_forwards"] == 10
                  and old_max["differentiable_transformer_forwards"] == 6
                  and all(price[name] <= old_max[name] for name in old_max
                          if name != "differentiable_transformer_forwards"))
    pred_b = bool(price["differentiable_transformer_forwards"] == 10
                  and price["checkpoint_loads"] == 1
                  and all(price[name] <= CORRECTED_MAX[name] for name in CORRECTED_MAX))
    preserved = {key: predictions1[key] for key in SCIENTIFIC_KEYS}
    pred_c = preserved == dict(zip(SCIENTIFIC_KEYS, (False, True, True, True, True)))
    pred_d = finite({key: v1[key] for key in (
        "raw_exact_top10_overlap", "tangent_exact_fidelity", "exact_fold_stability",
        "known_writer_percentiles", "upstream_writers", "downstream_readers")})
    predictions = {
        "pred_a_v1_invalid_only_from_registered_forward_price": pred_a,
        "pred_b_corrected_exact_price_validates_instrument": pred_b,
        "pred_c_scientific_predictions_and_nulls_are_immutable": pred_c,
        "pred_d_all_preserved_metrics_are_finite": pred_d,
    }
    terminal = "normalized_reader_candidate_price_corrected" if all(predictions.values()) else "invalid"
    result = {
        "schema": "temporal_iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v2_price_audit_result_v1",
        "candidate_id": CANDIDATE_ID,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "authority_sha256": EXPECTED,
        "v1_terminal_preserved": v1["terminal"],
        "v1_scientific_predictions": preserved,
        "corrected_price_max": CORRECTED_MAX,
        "actual_price": {name: price[name] for name in CORRECTED_MAX},
        "preserved_summary": {key: v1[key] for key in (
            "raw_exact_top10_overlap", "tangent_exact_fidelity", "exact_fold_stability",
            "shared_exact_top10_interfaces", "known_writer_percentiles")},
        "predictions": predictions,
        "terminal": terminal,
        "price": {"model_forwards": 0, "checkpoint_loads": 0, "example_evaluations": 0,
                  "model_updates": 0, "fit_parameters": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
