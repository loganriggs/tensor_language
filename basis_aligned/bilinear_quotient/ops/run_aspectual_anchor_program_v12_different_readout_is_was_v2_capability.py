#!/usr/bin/env python3
"""Capability-only gate before a second is/was cross-readout causal test."""

# BQGATE: EXPERIMENT pred_a_authority_and_exact_head pred_b_population_capability pred_c_exact_coverage
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import circuit_candidate_aspectual_different_readout_is_was_v2 as fresh
import run_aspectual_anchor_base_margin_affine_carrier_actuation_v1 as affine
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_anchor_program_v12_different_readout_is_was_v2_capability.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v2.py"
V1 = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v1_result.json"
V12 = ROOT / "circuits/followups/aspectual_anchor_transparent_path_program_release_v12_result.json"
OUT = ROOT / "circuits/followups/aspectual_anchor_program_v12_different_readout_is_was_v2_capability_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.program_v12_different_readout_is_was_v2_capability"
EXPECTED_PRIOR_SHA256 = "bde8dfd9adfa0ce8aa2b54e4eb0d025f3ee167885216216e8531c2f5a53a5b8e"
EXPECTED = {
    BUILDER: "4ed62d06e2ffe5c471efc70eeaa35c7524a66923f19c64c803faef7200d4a62f",
    V1: "574518c76300a7633c438f94e72c656cfae167135c292407770ec41656131b0b",
    V12: "ed6afb3455bf5bfeea6e36f65ce33e9199290cb26540ee94da2c42accc785e7c",
}
EXPECTED_ROWS_SHA256 = "fef8174cd57a1382d87ae47ddb06ddebd8059e96e4f71472525a066a3048f911"


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direction_for(row, family):
    return row["direction_id"] if family in ("A1", "A2") else ("present_to_past" if row["group_number"] % 2 == 0 else "past_to_present")


def validate_static():
    if sha(PRIOR) != EXPECTED_PRIOR_SHA256:
        raise ExperimentError("prior hash changed")
    for path, digest in EXPECTED.items():
        if sha(path) != digest:
            raise ExperimentError(f"authority hash changed: {path.name}")
    prior = json.loads(PRIOR.read_text())
    v1 = json.loads(V1.read_text())
    v12 = json.loads(V12.read_text())
    rows = fresh.build_rows()
    if prior.get("candidate_id") != CANDIDATE_ID or v1.get("terminal") != "invalid" or v12.get("terminal") != "release" or fresh.validate_rows(rows) != EXPECTED_ROWS_SHA256 or len(rows) != 64:
        raise ExperimentError("candidate, boundary, release, or rows changed")
    return rows


def main() -> None:
    rows = validate_static()
    dryrun = {"schema": "aspectual_anchor_program_v12_different_readout_is_was_v2_capability_dryrun", "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only", "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS_SHA256, "rows": 64, "counted_forwards_max": 9, "example_evaluations_max": 136, "scored_native_sides": 128, "interventions": 0, "grid_evaluations": 0, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = affine.parent.scalar_parent.empirical.component_parent.utc_now(), time.perf_counter()
    backend = affine.parent.producer.Bilin18TorchBackend.load("cuda")
    head_ok, head_error = affine.parent.das.verify_head(backend, [r for r in rows if r["family"] == "A1"][:8], "resid:18")
    forwards, evaluations = 1, 8
    records = []
    for family in ("A1", "A2", "P", "C"):
        family_rows = [row for row in rows if row["family"] == family]
        for chunk in affine.parent.producer._chunks(family_rows, 64):
            base, donor, _ = affine.parent.das.capture_site(backend, chunk, "resid:18")
            forwards += 2
            evaluations += 2 * len(chunk)
            for i, row in enumerate(chunk):
                direction = direction_for(row, family)
                for side, state, answer_id, foil_id in (("base", base[i], row["base_answer_id"], row["base_foil_id"]), ("donor", donor[i], row["donor_answer_id"], row["donor_foil_id"])):
                    pair = affine.parent.pair_logits(backend, state, answer_id, foil_id)
                    records.append({"family": family, "direction": direction, "row_id": str(row["row_id"]), "side": side, "native_margin": pair[0] - pair[1], "correct": pair[0] > pair[1]})
    cells = []
    for family in ("A1", "A2", "P", "C"):
        for direction in ("past_to_present", "present_to_past"):
            selected = [r for r in records if r["family"] == family and r["direction"] == direction]
            accuracy = sum(r["correct"] for r in selected) / len(selected)
            threshold = 0.75 if family == "C" else 0.85
            cells.append({"family": family, "direction": direction, "correct": sum(r["correct"] for r in selected), "total": len(selected), "accuracy": accuracy, "threshold": threshold, "passed": accuracy >= threshold})
    pred_a = head_ok and head_error <= 1.0e-3
    pred_b = all(cell["passed"] for cell in cells)
    pred_c = len(records) == 128 and len({(r["row_id"], r["side"]) for r in records}) == 128 and forwards <= 9 and evaluations <= 136
    predictions = {"pred_a_authority_and_exact_head": pred_a, "pred_b_population_capability": pred_b, "pred_c_exact_coverage": pred_c}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_c else "invalid")
    reason = {"screen": "v2_population_capability_authorizes_frozen_causal_followup", "null": "v2_population_capability_fails_before_causal_work", "invalid": "authority_head_or_coverage_invalid"}[terminal]
    value = {"schema": "aspectual_anchor_program_v12_different_readout_is_was_v2_capability_result", "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only", "started_utc": started_utc, "finished_utc": affine.parent.scalar_parent.empirical.component_parent.utc_now(), "serial_seconds": time.perf_counter() - started, "prior_art_sha256": EXPECTED_PRIOR_SHA256, "rows_sha256": EXPECTED_ROWS_SHA256, "head_control": {"passed": head_ok, "max_abs_difference": head_error}, "capability_cells": cells, "predictions": predictions, "score": {"counted_forwards": forwards, "example_evaluations": evaluations, "scored_native_sides": len(records), "interventions": 0, "grid_evaluations": 0, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0}, "native_records": records, "causal_outcomes_opened": False, "terminal": terminal, "reason": reason, "next_action": "run frozen v12 causal followup only if screen"}
    payload = atomic_create_json(OUT, value)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "capability_cells": cells, "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
