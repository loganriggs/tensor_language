#!/usr/bin/env python3
"""Zero-model correction of the immutable v16 capability panel denominator."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_hashes_inventory_and_v1_null pred_b_native_cells_and_corrected_joint_bar pred_c_no_causal_access_and_exact_original_price pred_d_all_records_and_metrics_finite
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16 as fresh
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v2_audit.json"
V1 = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v1_result.json"
V1_PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v1.json"
V1_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v16_capability_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v16_capability_v2_audit_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v16_capability_v2_audit"
EXPECTED = {
    "prior": "716fc8e0475cf2821aca4f373f66b87220c57130254b87ad7815cdb232e3dba3",
    "v1_result": "921cc7737f0986c1a897352c0e66171800b17660d6871742c4ec35741bbd30cf",
    "v1_prior": "feea0b786aa23135daf6cfc7f818365e80bebcccfdd20e3e8e47af1b039c904b",
    "v1_runner": "333c5398566539d8ba1f8940ab894d2ed8db9f1449caf73d5fd8d0156d40c34f",
    "builder": "5b1cb38cc62b5682c03505a5090e962d4773015efb44f8fad145515f5066b549",
}
ROWS_SHA256 = "4c5dfaee126c04ac7ea6ef5f53d6ad62a24806fa7133a0c773471a90e4d2e468"
CORRECTED_JOINT_BAR = 12
PRICE = {"model_forwards": 0, "checkpoint_loads": 0, "example_evaluations": 0,
         "model_updates": 0, "transformer_backwards": 0}
V1_KEYS = tuple("pred" + suffix for suffix in (
    "_a_authority_novelty_and_exact_population",
    "_b_native_a_panel_capability",
    "_c_joint_capable_population",
    "_d_no_causal_outcome_access_and_exact_price",
))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {"prior": PRIOR, "v1_result": V1, "v1_prior": V1_PRIOR,
             "v1_runner": V1_RUNNER, "builder": BUILDER}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"v16 audit authority changed: {observed}")
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False,
        "corrected_bar": "12/16 per A panel", **PRICE,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    v1 = json.loads(V1.read_text())
    rows = fresh.build_rows()
    panels = ("A1", "A2")
    panel_counts = {panel: sum(row["transform_id"] == panel for row in rows) for panel in panels}
    joint_counts = {panel: len(v1["jointly_capable_row_ids"][panel]) for panel in panels}
    predictions1 = v1.get("predictions", {})
    pred_a = bool(
        v1.get("terminal") == "null"
        and predictions1.get(V1_KEYS[0]) is True
        and predictions1.get(V1_KEYS[1]) is True
        and predictions1.get(V1_KEYS[2]) is False
        and predictions1.get(V1_KEYS[3]) is True
        and fresh.authority_sha256() == ROWS_SHA256 and len(rows) == 64
        and panel_counts == {"A1": 16, "A2": 16})
    pred_b = bool(all(
        cell["passed"] and math.isfinite(float(cell["accuracy"])) and cell["accuracy"] >= .75
        for cell in v1["capability_cells"])
        and all(joint_counts[panel] >= CORRECTED_JOINT_BAR for panel in panels))
    pred_c = bool(
        not v1["causal_outcomes_opened"] and v1["price"]["model_forwards"] == 2
        and v1["price"]["example_evaluations"] == 128
        and v1["price"]["interventions"] == 0)
    pred_d = bool(len(v1["native_records"]) == 128 and all(
        math.isfinite(float(cell["accuracy"])) for cell in v1["capability_cells"]))
    predictions = {
        "pred_a_hashes_inventory_and_v1_null": pred_a,
        "pred_b_native_cells_and_corrected_joint_bar": pred_b,
        "pred_c_no_causal_access_and_exact_original_price": pred_c,
        "pred_d_all_records_and_metrics_finite": pred_d,
    }
    terminal = "manifest" if all(predictions.values()) else \
        "null" if pred_a and pred_c and pred_d else "invalid"
    result = {
        "schema": "tense_auxiliary_is_was_fresh_lexicon_v16_capability_v2_audit_result_v1",
        "candidate_id": CANDIDATE_ID,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "authority_sha256": EXPECTED, "rows_sha256": ROWS_SHA256,
        "preserved_v1_terminal": v1["terminal"], "panel_counts": panel_counts,
        "jointly_capable_counts": joint_counts, "corrected_joint_bar": CORRECTED_JOINT_BAR,
        "capability_cells": v1["capability_cells"], "causal_outcomes_opened": False,
        "predictions": predictions, "terminal": terminal, "price": PRICE,
    }
    atomic_create_json(OUT, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
