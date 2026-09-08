#!/usr/bin/env python3
"""Aligned fresh-bank confirmation of the frozen four-head is/was writer program."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_reader_hooks_closures_finiteness_and_exact_price pred_b_four_head_union_selectively_recovers_behavior_and_q pred_c_at_least_three_frozen_heads_replicate_as_selective_singletons pred_d_effect_is_distributed_and_split_stable
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1 as shared

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v23_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23.py"
V22_RESULT = ROOT / "circuits/followups/temporal_iswas_v22_reader_contracted_writer_effect_game_v1_result.json"
V22_AUDIT = ROOT / "circuits/followups/temporal_iswas_v22_reader_contracted_writer_effect_game_v1_instrument_audit.json"
SHARED = ROOT / "ops/run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
DAS = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
EXPECTED = {
    "prior": "3defdb0612e552f1becc503d9d22e8fc611adc0cf8b5efb3fdb817b6f87450cb",
    "capability": "35e6cb352c276a4e1b3643cc2f5795bde667c06b8d7987352b198ccce10a4856",
    "builder": "a4830fd110b8cd854a5f28bfae776f697a15d4f791355990e02ea030fdca4c05",
    "v22_result": "b0c8ea135e1a84ebe148445203b3ba7be82994cfb6fc1f82ad625f90ab13f75f",
    "v22_audit": "e49fa3db73811cdfc7481357378c89a46d562304a5820b46b5f145c8c236a8c2",
    "shared": "9ab2a9edb60f4e3e4befebf11f55225659559a2a504075567218eab9bf903d06",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f"
}
ROWS_SHA256 = "46e9e493a4b2b42ce0c121b30978f08208537300363fa91902184c8f8d6565c7"
CANDIDATE_ID = "cross_task.temporal_iswas.v23_aligned_four_head_reader_contracted_confirmation_v1"
ROUTES = ("L9H1", "L9H4", "L8H1", "L11H3")
PANELS = ("A1", "A2", "P", "C")
BARS = {"closure": 1e-4, "hidden_closure": 1e-4, "shapley_efficiency": 1e-8,
        "union_recovery": .50, "union_cosine": .90, "union_direction": .90,
        "union_control": .15, "singleton_behavior": .15, "singleton_q": .05,
        "singleton_direction": .75, "singleton_control": .10,
        "singleton_count": 3, "distributed_allocation": .05, "distributed_count": 3}
PRICE = {"checkpoint_loads": 1, "model_forwards_exact": 19,
         "sequence_evaluations_exact": 1216, "transformer_backwards": 1,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_alignment_reader_hooks_closures_finiteness_and_exact_price",
    "pred_b_four_head_union_selectively_recovers_behavior_and_q",
    "pred_c_at_least_three_frozen_heads_replicate_as_selective_singletons",
    "pred_d_effect_is_distributed_and_split_stable",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def shapley(values):
    values = np.asarray(values, dtype=np.float64); n = len(ROUTES); phi = np.zeros(n)
    for i in range(n):
        bit = 1 << i
        for mask in range(1 << n):
            if mask & bit: continue
            size = mask.bit_count()
            phi[i] += (math.factorial(size) * math.factorial(n-size-1) / math.factorial(n)) * (values[mask | bit] - values[mask])
    interactions = {}
    for i, j in itertools.combinations(range(n), 2):
        total = 0.0; bi, bj = 1 << i, 1 << j
        for mask in range(1 << n):
            if mask & (bi | bj): continue
            size = mask.bit_count()
            total += (math.factorial(size) * math.factorial(n-size-2) / math.factorial(n-1)) * (values[mask | bi | bj] - values[mask | bi] - values[mask | bj] + values[mask])
        interactions[f"{ROUTES[i]}|{ROUTES[j]}"] = float(total)
    return {"allocations": {ROUTES[i]: float(phi[i]) for i in range(n)},
            "pair_interactions": interactions,
            "efficiency_residual": float(abs(phi.sum() - (values[-1] - values[0])))}


def main():
    paths = {"prior": PRIOR, "capability": CAPABILITY, "builder": BUILDER,
             "v22_result": V22_RESULT, "v22_audit": V22_AUDIT, "shared": SHARED,
             "producer": PRODUCER, "das": DAS}
    observed = {name: sha(path) for name, path in paths.items()}
    capability = json.loads(CAPABILITY.read_text()); audit = json.loads(V22_AUDIT.read_text())
    rows = fresh.build_rows(); counts = {p: sum(r["family"] == p for r in rows) for p in PANELS}
    capable_ids = set(capability["jointly_capable_row_ids"]["A1"] + capability["jointly_capable_row_ids"]["A2"])
    target_rows = [i for i, row in enumerate(rows) if row["row_id"] in capable_ids]
    alignment = all(row["base_semantic_position"] == row["donor_semantic_position"]
                    and len(row["base_ids"]) == len(row["donor_ids"]) for row in rows)
    authority_ok = bool(observed == EXPECTED and fresh.authority_sha256() == ROWS_SHA256
        and counts == {p: 16 for p in PANELS} and alignment and capability.get("terminal") == "screen"
        and all(capability.get("predictions", {}).values()) and len(target_rows) == 30
        and {k: len(v) for k, v in capability["jointly_capable_row_ids"].items()} == {"A1": 16, "A2": 14}
        and audit.get("verdict") == "target_and_P_game_valid_four_head_confirmation; global_A_and_C_selectivity_not_admissible")
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "rows": counts, "target_rows": len(target_rows), "routes": ROUTES,
           "coalitions": 1 << len(ROUTES), "bars": BARS, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("v23 aligned confirmation authority invalid")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    for parameter in backend.model.parameters(): parameter.requires_grad_(False)
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_logits, lengths, base_final, base_cache, base_saved, _ = shared.capture_native(backend, base_batch, True)
    base_margin = shared.margin(torch, base_logits, lengths, rows, backend.device); base_margin.sum().backward()
    if base_saved["leaf"].grad is None: raise RuntimeError("frozen native reader gradient missing")
    reader = base_saved["leaf"].grad.detach().float() @ backend.model.transformer.h[11].mlp.Down.weight.detach().float()
    with torch.no_grad():
        donor_logits, donor_lengths, donor_final, donor_cache, donor_saved, _ = shared.capture_native(backend, donor_batch, False)
        donor_margin = shared.margin(torch, donor_logits, donor_lengths, rows, backend.device)
        base_h, donor_h = shared.hidden(backend.model, base_saved["x"]), shared.hidden(backend.model, donor_saved["x"])
        mask = torch.zeros(base_h.shape[:2], dtype=torch.bool, device=backend.device)
        for i, query in enumerate(base_batch.semantic_positions): mask[i, :int(query)+1] = True
        native_q = ((donor_h-base_h) * reader * mask.unsqueeze(-1)).sum(dim=(1, 2))
        native_behavior = donor_margin-base_margin.detach()
        target_idx = torch.as_tensor(target_rows, device=backend.device)
        target_behavior, target_q = native_behavior[target_idx], native_q[target_idx]
        control_idx = {p: torch.as_tensor([i for i, r in enumerate(rows) if r["family"] == p], device=backend.device)
                       for p in ("P", "C")}
        half_idx = {name: torch.as_tensor([i for i in target_rows if int(rows[i]["group_number"]) % 4 in residues], device=backend.device)
                    for name, residues in (("first", (0, 1)), ("second", (2, 3)))}

        def report(logits, run_lengths, x):
            behavior = shared.margin(torch, logits, run_lengths, rows, backend.device)-base_margin.detach()
            q = ((shared.hidden(backend.model, x)-base_h) * reader * mask.unsqueeze(-1)).sum(dim=(1, 2))
            result = {"target": {"behavior": shared.metrics(torch, behavior[target_idx], target_behavior),
                                   "q": shared.metrics(torch, q[target_idx], target_q)},
                      "controls": {}, "halves": {}}
            for panel, selected in control_idx.items():
                result["controls"][panel] = {"behavior_leak_ratio": shared.rms_ratio(behavior[selected], target_behavior),
                                              "q_leak_ratio": shared.rms_ratio(q[selected], target_q)}
            for name, selected in half_idx.items():
                result["halves"][name] = {"behavior": shared.metrics(torch, behavior[selected], native_behavior[selected]),
                                           "q": shared.metrics(torch, q[selected], native_q[selected])}
            return result

        self_logits, self_lengths, self_final, self_x = shared.run_patch(backend, base_batch, base_cache, all_sites=True)
        donor_patch_logits, donor_patch_lengths, donor_patch_final, donor_patch_x = shared.run_patch(backend, base_batch, donor_cache, all_sites=True)
        self_error = max(float((self_final-base_final.detach()).abs().max()),
                         float((shared.margin(torch, self_logits, self_lengths, rows, backend.device)-base_margin.detach()).abs().max()),
                         float((self_x-base_saved["x"]).abs().max()))
        donor_error = max(float((donor_patch_final-donor_final).abs().max()),
                          float((shared.margin(torch, donor_patch_logits, donor_patch_lengths, rows, backend.device)-donor_margin).abs().max()),
                          float((donor_patch_x-donor_saved["x"]).abs().max()))
        hidden_error = float(torch.linalg.vector_norm(shared.hidden(backend.model, donor_patch_x)-donor_h)
                             / torch.linalg.vector_norm(donor_h).clamp_min(1e-30))
        reports = {0: {"target": {"behavior": shared.metrics(torch, torch.zeros_like(target_behavior), target_behavior),
                                   "q": shared.metrics(torch, torch.zeros_like(target_q), target_q)},
                       "controls": {p: {"behavior_leak_ratio": 0.0, "q_leak_ratio": 0.0} for p in ("P", "C")},
                       "halves": {}}}
        games = {key: np.zeros(1 << len(ROUTES)) for key in ("behavior", "q", "first_behavior", "first_q", "second_behavior", "second_q")}
        forwards = 4
        for subset in range(1, 1 << len(ROUTES)):
            selected = tuple(ROUTES[i] for i in range(len(ROUTES)) if subset & (1 << i))
            logits, run_lengths, _, x = shared.run_patch(backend, base_batch, donor_cache, selected, False)
            arm = report(logits, run_lengths, x); reports[subset] = arm; forwards += 1
            games["behavior"][subset] = arm["target"]["behavior"]["signed_projection"]
            games["q"][subset] = arm["target"]["q"]["signed_projection"]
            for half in ("first", "second"):
                for kind in ("behavior", "q"):
                    games[f"{half}_{kind}"][subset] = arm["halves"][half][kind]["signed_projection"]
    game_reports = {name: shapley(values) for name, values in games.items()}
    max_efficiency = max(game["efficiency_residual"] for game in game_reports.values())
    union = reports[(1 << len(ROUTES))-1]
    singletons = {route: reports[1 << i] for i, route in enumerate(ROUTES)}

    def selective(value, recovery, control):
        return bool(value["target"]["behavior"]["signed_projection"] >= recovery
            and value["target"]["q"]["signed_projection"] >= recovery
            and value["target"]["behavior"]["cosine"] >= BARS["union_cosine"]
            and value["target"]["q"]["cosine"] >= BARS["union_cosine"]
            and value["target"]["behavior"]["direction_fraction"] >= BARS["union_direction"]
            and value["target"]["q"]["direction_fraction"] >= BARS["union_direction"]
            and all(value["controls"][p][k] <= control for p in ("P", "C")
                    for k in ("behavior_leak_ratio", "q_leak_ratio")))

    singleton_pass = {route: bool(value["target"]["behavior"]["signed_projection"] >= BARS["singleton_behavior"]
        and value["target"]["q"]["signed_projection"] >= BARS["singleton_q"]
        and value["target"]["behavior"]["direction_fraction"] >= BARS["singleton_direction"]
        and value["target"]["q"]["direction_fraction"] >= BARS["singleton_direction"]
        and all(value["controls"][p][k] <= BARS["singleton_control"] for p in ("P", "C")
                for k in ("behavior_leak_ratio", "q_leak_ratio"))) for route, value in singletons.items()}
    distributed = [route for route in ROUTES if game_reports["behavior"]["allocations"][route] >= BARS["distributed_allocation"]
        and game_reports["q"]["allocations"][route] >= BARS["distributed_allocation"]
        and all(game_reports[f"{half}_{kind}"]["allocations"][route] > 0
                for half in ("first", "second") for kind in ("behavior", "q"))]
    A = bool(authority_ok and float(reader.abs().max()) > 0 and self_error <= BARS["closure"]
        and donor_error <= BARS["closure"] and hidden_error <= BARS["hidden_closure"]
        and max_efficiency <= BARS["shapley_efficiency"] and shared.finite([reports, game_reports])
        and forwards == PRICE["model_forwards_exact"] and forwards*len(rows) == PRICE["sequence_evaluations_exact"])
    B = selective(union, BARS["union_recovery"], BARS["union_control"])
    C = sum(singleton_pass.values()) >= BARS["singleton_count"]
    D = len(distributed) >= BARS["distributed_count"]
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = "invalid_instrument" if not A else "confirmed_selective_four_head_writer_program" if all((B, C, D)) else "valid_four_head_confirmation_failure"
    result = {"schema": "temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": observed,
        "population": {"counts": counts, "target_jointly_capable_rows": len(target_rows),
                       "target_row_ids": [rows[i]["row_id"] for i in target_rows]},
        "alignment_all_rows": alignment, "routes": list(ROUTES), "reader_gradient_max_abs": float(reader.abs().max()),
        "native_target_q_rms": float(target_q.float().square().mean().sqrt()),
        "closures": {"self_max_abs": self_error, "donor_max_abs": donor_error, "donor_hidden_relative": hidden_error},
        "union_report": union, "singleton_reports": singletons, "singleton_pass": singleton_pass,
        "game_reports": game_reports, "distributed_routes": distributed,
        "max_shapley_efficiency_residual": max_efficiency, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE, "model_forwards": forwards,
        "v23_selected_subset": None, "v23_tuned_rank_or_dose": False}
    atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("closures", "union_report", "singleton_pass",
        "distributed_routes", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
