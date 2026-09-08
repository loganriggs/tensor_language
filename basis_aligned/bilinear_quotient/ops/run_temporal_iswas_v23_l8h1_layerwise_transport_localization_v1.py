#!/usr/bin/env python3
"""Locate where the confirmed L8H1 write becomes the occupied M11 response."""

# BQGATE: EXPERIMENT pred_a_authority_hooks_boundaries_closures_finiteness_and_exact_price pred_b_L8H1_localizes_before_block11_attention pred_c_block10_or_earlier_is_sufficient pred_d_earliest_boundary_is_split_stable_and_selective
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
import transport_boundary_capture as boundary

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_l8h1_layerwise_transport_localization_v1.json"
L9_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_l9h1h4_layerwise_transport_localization_v1_result.json"
L11_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_l11h3_exact_rmsnorm_transport_v1_result.json"
WEIGHT_NPZ = ROOT / "circuits/followups/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1.npz"
V23_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23.py"
BOUNDARY = ROOT / "ops/transport_boundary_capture.py"
SHARED = ROOT / "ops/run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
DAS = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_l8h1_layerwise_transport_localization_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v23_l8h1_layerwise_transport_localization_v1"
EXPECTED = {"prior": "bceb8f5a0e25dcc4e09d112371639c579b89bc1402c9fe9d75ad0b581047d3a3",
    "l9_result": "e5adbee64bd46f58419ed95863f6c0466d46a9460e889435e0b2ee2807c80a70",
    "l11_result": "3d1c87d9dfe81985497c045db9a1cb567e5c0f88d518f4ef880fadba6d6f950b",
    "weight_npz": "9b553f42e708f1612fd90a934e0d26783b935457156f4856bc4b69f60df37892",
    "v23_result": "db850d5e9b86f76cb4381a12fc83f91cd2aac3544029fabd18bad138d34fed92",
    "builder": "a4830fd110b8cd854a5f28bfae776f697a15d4f791355990e02ea030fdca4c05",
    "boundary": "d027438fbd9f65b336793cd628c8d55f41510adcf36266198b43449d36cdc8b9",
    "shared": "9ab2a9edb60f4e3e4befebf11f55225659559a2a504075567218eab9bf903d06",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f"}
ROUTE = "L8H1"
BOUNDARIES = ("block9_input_after_block8", "block10_input_after_block9",
              "block11_input_after_block10", "block11_post_attention_raw",
              "M11_normalized_input")
BARS = {"self": 1e-4, "closure": .002, "cosine": .90, "residual": .25,
        "half_cosine": .90}
PRICE = {"checkpoint_loads": 1, "model_forwards_exact": 4,
         "sequence_evaluations_exact": 256, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = ("pred_a_authority_hooks_boundaries_closures_finiteness_and_exact_price",
    "pred_b_L8H1_localizes_before_block11_attention", "pred_c_block10_or_earlier_is_sufficient",
    "pred_d_earliest_boundary_is_split_stable_and_selective")


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


def capture(backend, execute):
    return boundary.execute_with_boundaries(backend.model, execute,
        block_layers=(9, 10, 11), raw_attention_layer=11, mlp_input_layer=11)


def main():
    paths = {"prior": PRIOR, "l9_result": L9_RESULT, "l11_result": L11_RESULT,
             "weight_npz": WEIGHT_NPZ, "v23_result": V23_RESULT, "builder": BUILDER,
             "boundary": BOUNDARY, "shared": SHARED, "producer": PRODUCER, "das": DAS}
    observed = {name: sha(path) for name, path in paths.items()}
    l9, l11, v23 = (json.loads(path.read_text()) for path in (L9_RESULT, L11_RESULT, V23_RESULT))
    rows = fresh.build_rows(); target_ids = set(v23["population"]["target_row_ids"])
    authority_ok = bool(observed == EXPECTED and l9.get("terminal") == "later_block11_transport"
        and l11.get("terminal") == "rms_tangent_transport_program"
        and v23.get("terminal") == "confirmed_selective_four_head_writer_program"
        and sum(row["row_id"] in target_ids for row in rows) == 30)
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "route": ROUTE, "boundaries": BOUNDARIES, "rows": len(rows), "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("L8H1 layerwise transport authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = now(), time.perf_counter()
    frozen = np.load(WEIGHT_NPZ); backend = producer.Bilin18TorchBackend.load("cuda"); torch, F = backend.torch, backend.F
    reader_basis = torch.as_tensor(frozen["reader_basis"], device=backend.device).float()
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    with torch.no_grad():
        base_bundle, base_state, base_calls = capture(backend, lambda: shared.capture_native(backend, base_batch, False))
        base_logits, _, base_final, base_cache, base_saved, _ = base_bundle
        donor_bundle, _, donor_calls = capture(backend, lambda: shared.capture_native(backend, donor_batch, False))
        _, _, _, donor_cache, _, _ = donor_bundle
        self_bundle, _, self_calls = capture(backend,
            lambda: shared.run_patch(backend, base_batch, base_cache, (ROUTE,), False))
        self_logits, _, self_final, self_x = self_bundle
        patch_bundle, patch_state, patch_calls = capture(backend,
            lambda: shared.run_patch(backend, base_batch, donor_cache, (ROUTE,), False))
        _, _, _, patched_x = patch_bundle
        self_error = max(float((self_logits-base_logits).abs().max()), float((self_final-base_final).abs().max()),
                         float((self_x-base_saved["x"]).abs().max()))
        base_raw = boundary.pre_mlp_raw_state(backend.model, base_state, layer=11)
        patch_raw = boundary.pre_mlp_raw_state(backend.model, patch_state, layer=11)
        dimension = int(backend.model.config.n_embd); eps = torch.finfo(base_raw.dtype).eps
        native_closure = float((F.rms_norm(base_raw, (dimension,), eps=eps)-base_state["M11_input"]).abs().max())
        patch_closure = float((F.rms_norm(patch_raw, (dimension,), eps=eps)-patched_x).abs().max())
        base_h = shared.hidden(backend.model, base_state["M11_input"])
        actual = torch.einsum("an,bsn->bsa", reader_basis, shared.hidden(backend.model, patched_x)-base_h)
        valid = torch.zeros(base_h.shape[:2], dtype=torch.bool, device=backend.device)
        for i, query in enumerate(base_batch.semantic_positions): valid[i, :int(query)+1] = True
        target = valid & torch.as_tensor([row["row_id"] in target_ids for row in rows], device=backend.device)[:, None]
        controls = {panel: valid & torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)[:, None]
                    for panel in ("P", "C")}
        halves = {name: target & torch.as_tensor([int(row["group_number"]) % 4 in residues for row in rows], device=backend.device)[:, None]
                  for name, residues in (("first", (0, 1)), ("second", (2, 3)))}
        deltas = {"block9_input_after_block8": patch_state["block9_input"]-base_state["block9_input"],
                  "block10_input_after_block9": patch_state["block10_input"]-base_state["block10_input"],
                  "block11_input_after_block10": patch_state["block11_input"]-base_state["block11_input"],
                  "block11_post_attention_raw": patch_raw-base_raw,
                  "M11_normalized_input": patched_x-base_state["M11_input"]}
        reports = {}
        for name, delta in deltas.items():
            predicted_x = (base_state["M11_input"]+delta if name == "M11_normalized_input"
                           else F.rms_norm(base_raw+delta, (dimension,), eps=eps))
            predicted = torch.einsum("an,bsn->bsa", reader_basis,
                shared.hidden(backend.model, predicted_x)-base_h)
            reports[name] = {"target": metrics(torch, predicted, actual, target),
                             "halves": {half: metrics(torch, predicted, actual, selected)
                                        for half, selected in halves.items()}}
        passing = [name for name in BOUNDARIES[:-1] if reports[name]["target"]["cosine"] >= BARS["cosine"]
                   and reports[name]["target"]["relative_residual"] <= BARS["residual"]]
        earliest = passing[0] if passing else None
        actual_rms = {panel: float(actual[selected].square().mean().sqrt())
                      for panel, selected in (("target", target), ("P", controls["P"]), ("C", controls["C"]))}
    closure = max(native_closure, patch_closure)
    A = bool(authority_ok and self_error <= BARS["self"] and closure <= BARS["closure"]
        and finite([reports, actual_rms]) and all(value == 1 for calls in (base_calls, donor_calls, self_calls, patch_calls)
            for value in calls.values()) and PRICE["model_forwards_exact"] == 4
        and PRICE["sequence_evaluations_exact"] == 4*len(rows))
    B = any(name in passing for name in BOUNDARIES[:3])
    C = any(name in passing for name in BOUNDARIES[:2])
    D = bool(earliest is not None and min(reports[earliest]["halves"][half]["cosine"] for half in ("first", "second")) >= BARS["half_cosine"]
             and actual_rms["P"] < actual_rms["target"] and actual_rms["C"] < actual_rms["target"])
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A else "block8_write_ready" if C and earliest == BOUNDARIES[0] and D
                else "block9_converter" if C and earliest == BOUNDARIES[1] and D
                else "block10_or_11_converter" if B and D else "distributed_transport")
    result = {"schema": "temporal_iswas_v23_l8h1_layerwise_transport_localization_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": observed,
        "hook_calls": {"base": base_calls, "donor": donor_calls, "self": self_calls, "patch": patch_calls},
        "self_patch_max_abs": self_error, "max_raw_normalized_closure_abs": closure,
        "boundary_reports": reports, "passing_pre_normalization_boundaries": passing,
        "earliest_passing_boundary": earliest, "actual_response_rms": actual_rms,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE,
        "model_forwards": 4, "selected_boundary_after_outcome": None}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("self_patch_max_abs", "max_raw_normalized_closure_abs",
        "boundary_reports", "passing_pre_normalization_boundaries", "earliest_passing_boundary",
        "actual_response_rms", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
