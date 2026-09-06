#!/usr/bin/env python3
"""Capability-only gate for prospective is/was lexicon v8."""

# BQGATE: EXPERIMENT pred_a_authority_novelty_and_exact_population pred_b_native_a_panel_capability pred_c_joint_capable_population pred_d_no_causal_outcome_access_and_exact_price
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v8 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v8.py"
HEAD_ATLAS = ROOT / "circuits/followups/iswas_mlp8_complement_attn9_head_converter_atlas_v1_result.json"
OUT = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v8_capability_v1_result.json"
CANDIDATE_ID = "tense_auxiliary.is_vs_was.fresh_lexicon_v8_capability_v1"
EXPECTED = {"prior": "936ea7b2e2520a6dbd18be5d453a4b81d854610d822a8c39fb7e3f1cc44ec502",
    "builder": "0d9e306face4125dcaa8ee5d89edff1731e30604109924ec81551cccd23671e2",
    "head_atlas": "2c7e19d53f6123491de20882c04f3781ac5c81531f6dfc8ca9137c53b96a01a8"}
ROWS_SHA256 = "962ab179e34462266bff68b62e705fb4b94b4baca1e41149a94958add059d96d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if {"prior": sha(PRIOR), "builder": sha(BUILDER), "head_atlas": sha(HEAD_ATLAS)} != EXPECTED:
        raise RuntimeError("v8 capability authority changed")
    prior, head_atlas = json.loads(PRIOR.read_text()), json.loads(HEAD_ATLAS.read_text())
    rows = fresh.build_rows()
    if (prior.get("candidate_id") != CANDIDATE_ID or fresh.validate_rows(rows) != ROWS_SHA256
            or len(rows) != 64 or head_atlas.get("stable_material_heads") != [1, 4]):
        raise RuntimeError("v8 population or frozen head union changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": 64, "native_sides": 128,
        "model_forwards_exact": 2, "example_evaluations_exact": 128,
        "interventions": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    outputs = {side: backend.native(das._batch(backend, rows, side=side), capture=False)
               for side in ("base", "donor")}
    records = []
    for i, row in enumerate(rows):
        for side in ("base", "donor"):
            answer, foil = outputs[side].answer_foil[i]
            records.append({"row_id": row["row_id"], "family": row["family"],
                "direction": row["direction_id"], "side": side, "correct": float(answer) > float(foil)})
    cells = []
    for family in ("A1", "A2"):
        for direction in ("present_to_past", "past_to_present"):
            for side in ("base", "donor"):
                selected = [record for record in records if record["family"] == family
                            and record["direction"] == direction and record["side"] == side]
                accuracy = sum(record["correct"] for record in selected)/len(selected)
                cells.append({"family": family, "direction": direction, "side": side,
                    "count": len(selected), "accuracy": accuracy, "threshold": .75,
                    "passed": accuracy >= .75})
    joint = {family: [row["row_id"] for row in rows if row["family"] == family and all(
        next(record["correct"] for record in records if record["row_id"] == row["row_id"]
             and record["side"] == side) for side in ("base", "donor"))] for family in ("A1", "A2")}
    pred_a = fresh.authority_sha256() == ROWS_SHA256 and len({row["row_id"] for row in rows}) == 64
    pred_b = all(cell["passed"] for cell in cells)
    pred_c = all(len(joint[family]) >= 12 for family in ("A1", "A2"))
    pred_d = len(records) == 128
    predictions = {"pred_a_authority_novelty_and_exact_population": pred_a,
        "pred_b_native_a_panel_capability": pred_b,
        "pred_c_joint_capable_population": pred_c,
        "pred_d_no_causal_outcome_access_and_exact_price": pred_d}
    terminal = "invalid" if not pred_a or not pred_d else "screen" if pred_b and pred_c else "null"
    result = {"schema": "tense_auxiliary_is_was_fresh_lexicon_v8_capability_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only_capability_only",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "rows_sha256": ROWS_SHA256, "dryrun": dryrun, "capability_cells": cells,
        "jointly_capable_row_ids": joint, "native_records": records,
        "causal_outcomes_opened": False, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": 2, "example_evaluations": 128, "interventions": 0,
            "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability_cells",
        "jointly_capable_row_ids", "causal_outcomes_opened", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__":
    main()
