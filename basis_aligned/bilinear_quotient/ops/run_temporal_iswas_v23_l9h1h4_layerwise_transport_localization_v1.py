#!/usr/bin/env python3
"""Locate where L9H1/H4 writes acquire the occupied M11 mode orientation."""

# BQGATE: EXPERIMENT pred_a_authority_hooks_boundaries_closures_finiteness_and_exact_price pred_b_both_L9_heads_localize_before_M11_normalization pred_c_block11_input_is_a_sufficient_transport_boundary pred_d_localized_boundary_is_split_stable_and_selective
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
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
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_l9h1h4_layerwise_transport_localization_v1.json"
L11_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_l11h3_exact_rmsnorm_transport_v1_result.json"
OCCUPANCY = ROOT / "circuits/followups/temporal_iswas_v23_four_head_weight_mode_occupancy_transport_v1_result.json"
WEIGHT_NPZ = ROOT / "circuits/followups/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1.npz"
V23_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23.py"
SHARED = ROOT / "ops/run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
DAS = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_l9h1h4_layerwise_transport_localization_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v23_l9h1h4_layerwise_transport_localization_v1"
EXPECTED = {"prior": "8fee0bf207035502710dbc4d7663a7b4306033551d6d1709526e62e9f5f0b59a",
    "l11_result": "3d1c87d9dfe81985497c045db9a1cb567e5c0f88d518f4ef880fadba6d6f950b",
    "occupancy": "2aae7e4bb5eb5092c64cc0248f5fa789a418ff2d78549e4d2acf438b8eef589b",
    "weight_npz": "9b553f42e708f1612fd90a934e0d26783b935457156f4856bc4b69f60df37892",
    "v23_result": "db850d5e9b86f76cb4381a12fc83f91cd2aac3544029fabd18bad138d34fed92",
    "builder": "a4830fd110b8cd854a5f28bfae776f697a15d4f791355990e02ea030fdca4c05",
    "shared": "9ab2a9edb60f4e3e4befebf11f55225659559a2a504075567218eab9bf903d06",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f"}
ROUTES = ("L9H1", "L9H4")
BOUNDARIES = ("block10_input_after_block9", "block11_input_after_block10",
              "block11_post_attention_raw", "M11_normalized_input")
BARS = {"self": 1e-4, "closure": .002, "cosine": .90, "residual": .25,
        "half_cosine": .90}
PRICE = {"checkpoint_loads": 1, "model_forwards_exact": 5,
         "sequence_evaluations_exact": 320, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = ("pred_a_authority_hooks_boundaries_closures_finiteness_and_exact_price",
    "pred_b_both_L9_heads_localize_before_M11_normalization",
    "pred_c_block11_input_is_a_sufficient_transport_boundary",
    "pred_d_localized_boundary_is_split_stable_and_selective")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def metrics(torch, predicted, actual, selected):
    x, y = predicted[selected].reshape(-1).double(), actual[selected].reshape(-1).double()
    xx, yy, xy = float(x@x), float(y@y), float(x@y)
    return {"cosine": xy/math.sqrt(max(xx*yy, 1e-30)), "signed_projection": xy/max(yy, 1e-30),
            "relative_residual": math.sqrt(float((x-y)@(x-y))/max(yy, 1e-30)),
            "norm_ratio": math.sqrt(xx/max(yy, 1e-30))}


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, (int, float)) or isinstance(value, bool) or math.isfinite(float(value))


def with_boundaries(backend, execute):
    saved, calls, handles = {}, {"block10": 0, "block11": 0, "attn11": 0, "mlp11": 0}, []
    for layer in (10, 11):
        def block_pre(_module, args, layer=layer):
            calls[f"block{layer}"] += 1
            saved[f"block{layer}_input"] = args[0].detach().clone()
            if layer == 11: saved["block11_x0"] = args[2].detach().clone()
        handles.append(backend.model.transformer.h[layer].register_forward_pre_hook(block_pre))
    def attn_out(_module, _args, output):
        calls["attn11"] += 1; saved["attn11_write"] = output.detach().clone()
    handles.append(backend.model.transformer.h[11].attn.c_proj.register_forward_hook(attn_out))
    def mlp_pre(_module, args):
        calls["mlp11"] += 1; saved["M11_input"] = args[0].detach().clone()
    handles.append(backend.model.transformer.h[11].mlp.register_forward_pre_hook(mlp_pre))
    try: result = execute()
    finally:
        for handle in handles: handle.remove()
    if any(value != 1 for value in calls.values()): raise RuntimeError(f"boundary hook counts changed: {calls}")
    return result, saved, calls


def raw11(backend, state):
    block = backend.model.transformer.h[11]
    return (block.lambdas[0]*state["block11_input"] + block.lambdas[1]*state["block11_x0"]
            + state["attn11_write"])


def main():
    paths = {"prior": PRIOR, "l11_result": L11_RESULT, "occupancy": OCCUPANCY,
             "weight_npz": WEIGHT_NPZ, "v23_result": V23_RESULT, "builder": BUILDER,
             "shared": SHARED, "producer": PRODUCER, "das": DAS}
    observed = {name: sha(path) for name, path in paths.items()}
    l11, v23 = json.loads(L11_RESULT.read_text()), json.loads(V23_RESULT.read_text())
    rows = fresh.build_rows(); target_ids = set(v23["population"]["target_row_ids"])
    authority_ok = bool(observed == EXPECTED and l11.get("terminal") == "rms_tangent_transport_program"
        and all(l11.get("predictions", {}).values()) and v23.get("terminal") == "confirmed_selective_four_head_writer_program"
        and sum(row["row_id"] in target_ids for row in rows) == 30)
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "routes": ROUTES, "boundaries": BOUNDARIES, "rows": len(rows), "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("L9 layerwise transport authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = now(), time.perf_counter()
    frozen = np.load(WEIGHT_NPZ); backend = producer.Bilin18TorchBackend.load("cuda"); torch, F = backend.torch, backend.F
    reader_basis = torch.as_tensor(frozen["reader_basis"], device=backend.device).float()
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    with torch.no_grad():
        base_bundle, base_state, base_calls = with_boundaries(backend, lambda: shared.capture_native(backend, base_batch, False))
        base_logits, _, base_final, base_cache, base_saved, _ = base_bundle
        donor_bundle, _, donor_calls = with_boundaries(backend, lambda: shared.capture_native(backend, donor_batch, False))
        _, _, _, donor_cache, _, _ = donor_bundle
        self_bundle, _, self_calls = with_boundaries(backend,
            lambda: shared.run_patch(backend, base_batch, base_cache, ROUTES, False))
        self_logits, _, self_final, self_x = self_bundle
        self_error = max(float((self_logits-base_logits).abs().max()), float((self_final-base_final).abs().max()),
                         float((self_x-base_saved["x"]).abs().max()))
        dimension = int(backend.model.config.n_embd); eps = torch.finfo(raw11(backend, base_state).dtype).eps
        base_raw = raw11(backend, base_state); native_norm = F.rms_norm(base_raw, (dimension,), eps=eps)
        native_closure = float((native_norm-base_state["M11_input"]).abs().max())
        base_h = shared.hidden(backend.model, base_state["M11_input"])
        valid = torch.zeros(base_h.shape[:2], dtype=torch.bool, device=backend.device)
        for i, query in enumerate(base_batch.semantic_positions): valid[i, :int(query)+1] = True
        target = valid & torch.as_tensor([row["row_id"] in target_ids for row in rows], device=backend.device)[:, None]
        controls = {panel: valid & torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)[:, None]
                    for panel in ("P", "C")}
        halves = {name: target & torch.as_tensor([int(row["group_number"]) % 4 in residues for row in rows], device=backend.device)[:, None]
                  for name, residues in (("first", (0, 1)), ("second", (2, 3)))}
        route_reports, closure_errors = {}, []
        for route in ROUTES:
            patch_bundle, state, calls = with_boundaries(backend,
                lambda route=route: shared.run_patch(backend, base_batch, donor_cache, (route,), False))
            _, _, _, patched_x = patch_bundle
            patch_raw = raw11(backend, state); reconstructed = F.rms_norm(patch_raw, (dimension,), eps=eps)
            closure_errors.append(float((reconstructed-patched_x).abs().max()))
            actual = torch.einsum("an,bsn->bsa", reader_basis,
                                  shared.hidden(backend.model, patched_x)-base_h)
            deltas = {"block10_input_after_block9": state["block10_input"]-base_state["block10_input"],
                      "block11_input_after_block10": state["block11_input"]-base_state["block11_input"],
                      "block11_post_attention_raw": patch_raw-base_raw,
                      "M11_normalized_input": patched_x-base_state["M11_input"]}
            reports = {}
            for boundary, delta in deltas.items():
                predicted_x = (base_state["M11_input"]+delta if boundary == "M11_normalized_input"
                               else F.rms_norm(base_raw+delta, (dimension,), eps=eps))
                predicted = torch.einsum("an,bsn->bsa", reader_basis,
                    shared.hidden(backend.model, predicted_x)-base_h)
                reports[boundary] = {"target": metrics(torch, predicted, actual, target),
                    "halves": {name: metrics(torch, predicted, actual, selected) for name, selected in halves.items()}}
            passing = [boundary for boundary in BOUNDARIES[:-1]
                       if reports[boundary]["target"]["cosine"] >= BARS["cosine"]
                       and reports[boundary]["target"]["relative_residual"] <= BARS["residual"]]
            actual_rms = {panel: float(actual[selected].square().mean().sqrt())
                          for panel, selected in (("target", target), ("P", controls["P"]), ("C", controls["C"]))}
            route_reports[route] = {"boundaries": reports, "passing_pre_normalization_boundaries": passing,
                                    "earliest_passing_boundary": passing[0] if passing else None,
                                    "actual_response_rms": actual_rms, "hook_calls": calls}
    max_closure = max([native_closure] + closure_errors)
    A = bool(authority_ok and self_error <= BARS["self"] and max_closure <= BARS["closure"]
        and finite(route_reports) and all(value == 1 for calls in (base_calls, donor_calls, self_calls)
            for value in calls.values()) and PRICE["model_forwards_exact"] == 5
        and PRICE["sequence_evaluations_exact"] == 5*len(rows))
    B = all(route_reports[route]["earliest_passing_boundary"] is not None for route in ROUTES)
    C = all("block11_input_after_block10" in route_reports[route]["passing_pre_normalization_boundaries"] for route in ROUTES)
    D = all(route_reports[route]["earliest_passing_boundary"] is not None
        and min(route_reports[route]["boundaries"][route_reports[route]["earliest_passing_boundary"]]["halves"][half]["cosine"]
                for half in ("first", "second")) >= BARS["half_cosine"]
        and route_reports[route]["actual_response_rms"]["P"] < route_reports[route]["actual_response_rms"]["target"]
        and route_reports[route]["actual_response_rms"]["C"] < route_reports[route]["actual_response_rms"]["target"] for route in ROUTES)
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A else "block11_input_transport" if B and C and D
                else "later_block11_transport" if B and D else "distributed_nonlinear_transport")
    result = {"schema": "temporal_iswas_v23_l9h1h4_layerwise_transport_localization_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": observed,
        "hook_calls": {"base": base_calls, "donor": donor_calls, "self": self_calls},
        "self_patch_max_abs": self_error, "max_raw_normalized_closure_abs": max_closure,
        "route_reports": route_reports, "predictions": predictions, "terminal": terminal,
        "bars": BARS, "price": PRICE, "model_forwards": 5, "selected_boundary_after_outcome": None}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("self_patch_max_abs", "max_raw_normalized_closure_abs",
        "route_reports", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
