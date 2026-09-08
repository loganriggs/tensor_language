#!/usr/bin/env python3
"""Repair v1's direct-route equality check to the registered query footprint."""

# BQGATE: EXPERIMENT pred_a_authority_scoped_endpoint_capture_state_replay_finiteness_and_exact_price pred_b_scoped_direct_add_matches_dynamic_identity_carriage pred_c_scoped_direct_remove_matches_dynamic_module_response pred_d_frozen_identity_route_predicts_ood_writer_behavior pred_e_identity_carriage_is_selectively_manipulable
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_selected_writer_block10_direct_residual_add_remove_v1 as v1


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_selected_writer_block10_direct_residual_add_remove_v2_scoped_logit_audit.json"
V1_RESULT = ROOT / "circuits/followups/temporal_iswas_selected_writer_block10_direct_residual_add_remove_v1_result.json"
V1_RUNNER = ROOT / "ops/run_temporal_iswas_selected_writer_block10_direct_residual_add_remove_v1.py"
V1_BINDING = ROOT / "circuits/prior_art/temporal_iswas_selected_writer_block10_direct_residual_add_remove_v1_authority_binding.json"
OUT = ROOT / "circuits/followups/temporal_iswas_selected_writer_block10_direct_residual_add_remove_v2_scoped_logit_audit_result.json"
EXPECTED = {
    "prior": "3d9e27b33007b87772a29508a1daae510ad8a598919c19d52bd522f50ed0ee22",
    "v1_result": "8c17e986b9c4f3f637cbd3a463f006b9c759aff4da0dbde9183df115c92b42c5",
    "v1_runner": "bd3d230c635892a4c174d0561d30c1c43fdadfadc217a455e07d59a1cd3c0a7f",
    "v1_binding": "ffdd22f28ccdb2a735c877a631d2645f72c9d94cb828e981905b87f66b6e8fcd",
}
FILES = {"prior": PRIOR, "v1_result": V1_RESULT, "v1_runner": V1_RUNNER,
         "v1_binding": V1_BINDING}
PRICE = dict(v1.PRICE)
BARS = dict(v1.BARS)
PREDICTION_KEYS = (
    "pred_a_authority_scoped_endpoint_capture_state_replay_finiteness_and_exact_price",
    "pred_b_scoped_direct_add_matches_dynamic_identity_carriage",
    "pred_c_scoped_direct_remove_matches_dynamic_module_response",
    "pred_d_frozen_identity_route_predicts_ood_writer_behavior",
    "pred_e_identity_carriage_is_selectively_manipulable",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_logit_comparator(authority):
    rows = authority.build_rows()
    endpoints, _lookup = v1.factorial.atlas.mediation.parent.loc.parent.endpoint_bank(rows)
    original = v1.factorial.atlas.mediation.parent.loc.original
    positive, negative = original._single(" is"), original._single(" was")
    positions = [endpoint["iswas_position"] - 1 for _row, _cell, endpoint in endpoints]

    def compare(left, right):
        if left.shape != right.shape or left.ndim != 3 or left.shape[0] != len(endpoints):
            raise RuntimeError("selected-logit comparison shape changed")
        batch = torch.arange(len(endpoints), device=left.device)[:, None]
        query = torch.tensor(positions, device=left.device)[:, None]
        vocabulary = torch.tensor([positive, negative], device=left.device)[None, :]
        return float((left[batch, query, vocabulary] - right[batch, query, vocabulary]).abs().max())

    return compare


def run_scoped_population(backend, authority, population, selected_heads):
    original = v1.maxdiff
    v1.maxdiff = selected_logit_comparator(authority)
    try:
        return v1.run_population(backend, authority, population, selected_heads)
    finally:
        v1.maxdiff = original


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior = json.loads(PRIOR.read_text())
    source = json.loads(V1_RESULT.read_text())
    authority_ok = bool(observed == EXPECTED and prior["price"] == PRICE
        and source.get("terminal") == "arithmetic_or_hook_failure"
        and tuple(source.get("predictions", {}).values()) == (True, False, False, False, False)
        and all(panel["instrumentation"]["direct_add_vs_R1M0_state_relative_l2"] <= 3.7e-8
                and panel["instrumentation"]["direct_remove_vs_R0M1_state_relative_l2"] <= 3.7e-8
                for panel in source["panels"].values()))
    dry = {"candidate_id": prior["candidate_id"], "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "authority_ok": authority_ok, "comparison_scope": "iswas_query_answer_and_foil",
           "arms": list(v1.ARMS), "populations": ["original", "ood"], "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if not authority_ok:
        raise RuntimeError("v2 scoped-logit authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = v1.factorial.atlas.mediation.parent.loc.producer.Bilin18TorchBackend.load("cuda")
    greedy = json.loads(v1.GREEDY_RESULT.read_text())
    selected_heads = tuple(greedy["selected_prefixes"]["iswas"]["heads"])
    if selected_heads != ("L07H07", "L09H04"):
        raise RuntimeError("selected compact writer changed")
    loc = v1.factorial.atlas.mediation.parent.loc
    panels = {
        "original": run_scoped_population(backend, loc.original, "original", selected_heads),
        "ood": run_scoped_population(backend, loc.ood, "ood", selected_heads),
    }
    by = v1.report_index(panels)
    instruments = [panel["instrumentation"] for panel in panels.values()]
    A = bool(authority_ok and sum(panel["row_count"] * 7 for panel in panels.values())
        == PRICE["sequence_evaluations"] and all(item["pairing_ok"] and item["positions_ok"]
        and item["native_self_patch_max_abs_logit_error"] <= BARS["self_patch_logit"]
        and item["native_self_patch_raw_state_relative_l2"] <= BARS["state_relative_l2"]
        and item["direct_add_vs_R1M0_state_relative_l2"] <= BARS["state_relative_l2"]
        and item["direct_remove_vs_R0M1_state_relative_l2"] <= BARS["state_relative_l2"]
        and item["native_writer_capture_calls"]["x10"] == [1, 1]
        and item["native_writer_capture_calls"]["x18"] == [1, 1]
        and set(item["native_writer_capture_calls"]["native_modules"].values()) == {1}
        and set(item["native_writer_capture_calls"]["writer_modules"].values()) == {1}
        and set(item["final_patch_calls"].values()) == {1}
        and all(cell["entry_calls"] == 1 and cell["final_calls"] == 1
                and set(cell["module_calls"].values()) == {1}
                for cell in item["dynamic_coverage"].values()) for item in instruments)
        and v1.finite(panels))
    B = all(item["direct_add_vs_R1M0_max_abs_logit_error"] <= BARS["direct_logit"]
            for item in instruments)
    C = all(item["direct_remove_vs_R0M1_max_abs_logit_error"] <= BARS["direct_logit"]
            for item in instruments)
    D = all(by[(population, phase, "direct_add")]["signed_recovery"] >= BARS["recovery"]
            and by[(population, phase, "direct_add")]["cosine"] >= BARS["cosine"]
            and by[(population, phase, "direct_add")]["direction_agreement"] >= BARS["direction"]
            for population, phase in v1.VALIDATION_CELLS)
    E = bool(all(by[(population, phase, "direct_remove")]["identity_removal_signed_recovery"]
                 >= BARS["removal_fraction"]
                 and by[(population, phase, "direct_remove")]["identity_removal_direction_agreement"]
                 >= BARS["removal_direction"] for population, phase in v1.VALIDATION_CELLS)
        and all(row["temporal_command_gold_collateral"] <= BARS["collateral"]
                for panel in panels.values() for row in panel["reports"]))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid" if not A else "physical_route_failure" if not (B and C)
                else "ood_instability" if not D else "nonselective_or_redundant" if not E
                else "direct_residual_identified")
    off_footprint = {name: {
        "direct_add_vs_R1M0_all_sequence_max_abs_logit_error": panel["instrumentation"][
            "direct_add_vs_R1M0_max_abs_logit_error"],
        "direct_remove_vs_R0M1_all_sequence_max_abs_logit_error": panel["instrumentation"][
            "direct_remove_vs_R0M1_max_abs_logit_error"]}
        for name, panel in source["panels"].items()}
    result = {"schema": "temporal_iswas_selected_writer_block10_direct_residual_add_remove_scoped_logit_audit_result_v2",
        "candidate_id": prior["candidate_id"],
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "selected_writer_heads": list(selected_heads), "comparison_scope":
            "answer_and_foil_logits_at_each_registered_iswas_query",
        "v1_off_footprint_diagnostics": off_footprint, "panels": panels,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({"predictions": predictions, "terminal": terminal,
        "selected_logit_errors": {name: {key: value for key, value in panel["instrumentation"].items()
            if "max_abs_logit_error" in key} for name, panel in panels.items()},
        "v1_off_footprint_diagnostics": off_footprint,
        "validation": [row for panel in panels.values() for row in panel["reports"]
            if (row["population"], row["phase"]) in v1.VALIDATION_CELLS],
        "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
