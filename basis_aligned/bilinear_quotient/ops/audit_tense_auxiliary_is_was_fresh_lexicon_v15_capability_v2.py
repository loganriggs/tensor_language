#!/usr/bin/env python3
"""Zero-forward correction of the v15 capability panel denominator."""
# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_hashes_inventory_and_v1_null pred_b_native_cells_and_corrected_joint_bar pred_c_no_causal_access_and_exact_original_price pred_d_all_records_and_metrics_finite
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15 as fresh
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v15_capability_v2_audit.json"
V1 = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v15_capability_v1_result.json"
V1_PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v15_capability_v1.json"
V1_RUNNER = ROOT / "ops/run_tense_auxiliary_is_was_fresh_lexicon_v15_capability_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15.py"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v15_capability_v2_audit_result.json"
EXPECTED = {
    "prior": "4aa288fea8bb78cf8297926d86bdb7927b6d7c5bc6f8f0f8da9335314693306b",
    "v1_result": "ecc34f5d089f31269de29250d8d63843ce49c622f5cc0161cbea3afb9042d7f0",
    "v1_prior": "4fe19cddb27342f605058c02b673a91255ad866e503a82f953e4fcd192d4ac21",
    "v1_runner": "8277df9126755bf619c18668864e31ba16d7d07f773445202707cadc43a1bec3",
    "builder": "e5774cd5e93c564ad3bdb3169c482c2e1ddb295b85cc7c9885a7b077493a8343",
}
ROWS_SHA256 = "9e6848a8bfe05be59ecd02d256502597e5e8b6b48fadd2b9f8e1b830b9e50c46"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    candidate_id = "tense_auxiliary.is_vs_was.fresh_lexicon_v15_capability_v2_audit"
    dry = {
        "candidate_id": candidate_id, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "model_forwards_exact": 0,
        "model_updates": 0, "transformer_backwards": 0, "corrected_bar": "14/16 per A panel",
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    observed = {
        "prior": sha(PRIOR), "v1_result": sha(V1), "v1_prior": sha(V1_PRIOR),
        "v1_runner": sha(V1_RUNNER), "builder": sha(BUILDER),
    }
    v1 = json.loads(V1.read_text())
    rows = fresh.build_rows()
    panel_counts = {panel: sum(row["transform_id"] == panel for row in rows) for panel in ("A1", "A2")}
    joint_counts = {panel: len(v1["jointly_capable_row_ids"][panel]) for panel in ("A1", "A2")}
    pred_a = bool(
        observed == EXPECTED and v1["terminal"] == "null"
        and v1["predictions"]["pred_a_authority_novelty_and_exact_population"]
        and v1["predictions"]["pred_b_native_a_panel_capability"]
        and not v1["predictions"]["pred_c_joint_capable_population"]
        and fresh.authority_sha256() == ROWS_SHA256 and len(rows) == 64
        and panel_counts == {"A1": 16, "A2": 16}
    )
    pred_b = all(cell["passed"] and cell["accuracy"] >= .75 for cell in v1["capability_cells"]) and all(
        joint_counts[panel] >= 14 for panel in ("A1", "A2")
    )
    pred_c = bool(
        not v1["causal_outcomes_opened"] and v1["price"]["model_forwards"] == 2
        and v1["price"]["example_evaluations"] == 128
    )
    pred_d = len(v1["native_records"]) == 128 and all(
        math.isfinite(float(cell["accuracy"])) for cell in v1["capability_cells"]
    )
    predictions = {
        "pred_a_hashes_inventory_and_v1_null": pred_a,
        "pred_b_native_cells_and_corrected_joint_bar": pred_b,
        "pred_c_no_causal_access_and_exact_original_price": pred_c,
        "pred_d_all_records_and_metrics_finite": pred_d,
    }
    terminal = "manifest" if all(predictions.values()) else "null" if pred_a and pred_c and pred_d else "invalid"
    result = {
        "schema": "tense_auxiliary_is_was_fresh_lexicon_v15_capability_v2_audit_result_v1",
        "candidate_id": candidate_id, "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "authority_sha256": EXPECTED, "rows_sha256": ROWS_SHA256,
        "preserved_v1_terminal": v1["terminal"], "panel_counts": panel_counts,
        "jointly_capable_counts": joint_counts, "corrected_joint_bar": 14,
        "capability_cells": v1["capability_cells"], "causal_outcomes_opened": False,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": 0, "model_updates": 0, "transformer_backwards": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
