#!/usr/bin/env python3
"""Reverse four-head necessity plus full-factor and occupied-mode M11 rescue."""

# BQGATE: EXPERIMENT pred_a_authority_hooks_self_patch_finiteness_and_exact_price pred_b_four_head_union_is_selectively_necessary_in_reverse pred_c_occupied_mode_rescues_most_of_complete_M11_factor_mediation pred_d_occupied_mode_is_directly_necessary
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
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_four_head_necessity_occupied_mode_rescue_v1.json"
V23_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
WEIGHT_NPZ = ROOT / "circuits/followups/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1.npz"
OCCUPANCY = ROOT / "circuits/followups/temporal_iswas_v23_four_head_weight_mode_occupancy_transport_v1_result.json"
L11_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_l11h3_exact_rmsnorm_transport_v1_result.json"
L9_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_l9h1h4_layerwise_transport_localization_v1_result.json"
L8_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_l8h1_layerwise_transport_localization_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23.py"
SHARED = ROOT / "ops/run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
DAS = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_four_head_necessity_occupied_mode_rescue_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v23_four_head_necessity_occupied_mode_rescue_v1"
EXPECTED = {"prior": "27c7f4a9f75fec273385a6875299b1214156650940e4d67b4d71875a5350f387",
    "v23_result": "db850d5e9b86f76cb4381a12fc83f91cd2aac3544029fabd18bad138d34fed92",
    "weight_npz": "9b553f42e708f1612fd90a934e0d26783b935457156f4856bc4b69f60df37892",
    "occupancy": "2aae7e4bb5eb5092c64cc0248f5fa789a418ff2d78549e4d2acf438b8eef589b",
    "l11_result": "3d1c87d9dfe81985497c045db9a1cb567e5c0f88d518f4ef880fadba6d6f950b",
    "l9_result": "e5adbee64bd46f58419ed95863f6c0466d46a9460e889435e0b2ee2807c80a70",
    "l8_result": "b2763ed635ad481d450bbb37bccfdf9ee6d3b452b4c1977606a08ce277b326a3",
    "builder": "a4830fd110b8cd854a5f28bfae776f697a15d4f791355990e02ea030fdca4c05",
    "shared": "9ab2a9edb60f4e3e4befebf11f55225659559a2a504075567218eab9bf903d06",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f"}
ROUTES = ("L9H1", "L9H4", "L8H1", "L11H3")
BARS = {"self": 1e-4, "head_recovery": .50, "direction": .90, "control": .15,
        "full_rescue": .50, "mode_rescue_cosine": .95, "mode_rescue_projection": .70,
        "mode_removal_recovery": .40}
PRICE = {"checkpoint_loads": 1, "model_forwards_exact": 7,
         "sequence_evaluations_exact": 448, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = ("pred_a_authority_hooks_self_patch_finiteness_and_exact_price",
    "pred_b_four_head_union_is_selectively_necessary_in_reverse",
    "pred_c_occupied_mode_rescues_most_of_complete_M11_factor_mediation",
    "pred_d_occupied_mode_is_directly_necessary")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def metrics(torch, value, reference, selected):
    x, y = value[selected].reshape(-1).double(), reference[selected].reshape(-1).double()
    xx, yy, xy = float(x@x), float(y@y), float(x@y)
    return {"cosine": xy/math.sqrt(max(xx*yy, 1e-30)), "signed_projection": xy/max(yy, 1e-30),
            "relative_residual": math.sqrt(float((x-y)@(x-y))/max(yy, 1e-30)),
            "norm_ratio": math.sqrt(xx/max(yy, 1e-30)),
            "direction_fraction": float(((value[selected]*reference[selected]) > 0).float().mean())}


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, (int, float)) or isinstance(value, bool) or math.isfinite(float(value))


def corrected_forward(backend, batch, correction):
    calls = {"mlp11": 0}
    def hook(_module, _args, output):
        calls["mlp11"] += 1
        if output.shape != correction.shape: raise RuntimeError("M11 correction shape changed")
        return output + correction.to(output)
    handle = backend.model.transformer.h[11].mlp.register_forward_hook(hook)
    try: logits, lengths, final = shared.full_forward(backend, batch)
    finally: handle.remove()
    if calls["mlp11"] != 1: raise RuntimeError("M11 correction hook count changed")
    return logits, lengths, final, calls


def main():
    paths = {"prior": PRIOR, "v23_result": V23_RESULT, "weight_npz": WEIGHT_NPZ,
             "occupancy": OCCUPANCY, "l11_result": L11_RESULT, "l9_result": L9_RESULT,
             "l8_result": L8_RESULT, "builder": BUILDER, "shared": SHARED,
             "producer": PRODUCER, "das": DAS}
    observed = {name: sha(path) for name, path in paths.items()}
    v23, occupancy, l11, l9, l8 = (json.loads(path.read_text()) for path in
        (V23_RESULT, OCCUPANCY, L11_RESULT, L9_RESULT, L8_RESULT))
    rows = fresh.build_rows(); target_ids = set(v23["population"]["target_row_ids"])
    authority_ok = bool(observed == EXPECTED and v23.get("terminal") == "confirmed_selective_four_head_writer_program"
        and occupancy.get("predictions", {}).get("pred_d_realized_target_use_is_sparser_and_split_stable") is True
        and l11.get("terminal") == "rms_tangent_transport_program"
        and l9.get("terminal") == "later_block11_transport" and l8.get("terminal") == "distributed_transport"
        and sum(row["row_id"] in target_ids for row in rows) == 30)
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "routes": ROUTES, "rows": len(rows), "target_rows": len(target_ids), "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("necessity/mode-rescue authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = now(), time.perf_counter()
    frozen = np.load(WEIGHT_NPZ); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    cross = torch.as_tensor(frozen["cross"], device=backend.device).float()
    reader_basis = torch.as_tensor(frozen["reader_basis"], device=backend.device).float()
    reader_unfolding = cross.movedim(1, 0).reshape(4, -1)
    reader_modes, singular, _ = torch.linalg.svd(reader_unfolding, full_matrices=False)
    factor_mode = reader_modes[:, 0] @ reader_basis
    factor_mode = factor_mode/factor_mode.norm().clamp_min(1e-30)
    mode_norm_error = abs(float(factor_mode.norm())-1.0)
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    with torch.no_grad():
        base_logits, base_lengths, _, base_cache, base_saved, _ = shared.capture_native(backend, base_batch, False)
        donor_logits, donor_lengths, donor_final, donor_cache, donor_saved, _ = shared.capture_native(backend, donor_batch, False)
        self_logits, self_lengths, self_final, self_x = shared.run_patch(backend, donor_batch, donor_cache, ROUTES, False)
        removed_logits, removed_lengths, _, removed_x = shared.run_patch(backend, donor_batch, base_cache, ROUTES, False)
        self_error = max(float((self_logits-donor_logits).abs().max()), float((self_final-donor_final).abs().max()),
                         float((self_x-donor_saved["x"]).abs().max()))
        base_margin = shared.margin(torch, base_logits, base_lengths, rows, backend.device)
        donor_margin = shared.margin(torch, donor_logits, donor_lengths, rows, backend.device)
        removed_margin = shared.margin(torch, removed_logits, removed_lengths, rows, backend.device)
        base_h, donor_h = shared.hidden(backend.model, base_saved["x"]), shared.hidden(backend.model, donor_saved["x"])
        removed_h = shared.hidden(backend.model, removed_x)
        valid = torch.zeros(donor_h.shape[:2], dtype=torch.bool, device=backend.device)
        for i, query in enumerate(donor_batch.semantic_positions): valid[i, :int(query)+1] = True
        factor_delta = (donor_h-removed_h)*valid.unsqueeze(-1)
        full_factor = factor_delta
        coefficient = torch.einsum("bsn,n->bs", factor_delta, factor_mode)
        occupied_factor = coefficient.unsqueeze(-1)*factor_mode
        paired_delta = (donor_h-base_h)*valid.unsqueeze(-1)
        paired_coefficient = torch.einsum("bsn,n->bs", paired_delta, factor_mode)
        paired_occupied_factor = paired_coefficient.unsqueeze(-1)*factor_mode
        down = backend.model.transformer.h[11].mlp.Down.weight.detach().float()
        full_correction = full_factor @ down.T
        occupied_correction = occupied_factor @ down.T
        removal_correction = -(paired_occupied_factor @ down.T)
        # The rescue arms repeat the same head removal and add the M11 correction.
        full_handle_calls = {"mlp11": 0}
        def full_hook(_m, _a, output):
            full_handle_calls["mlp11"] += 1; return output + full_correction.to(output)
        handle = backend.model.transformer.h[11].mlp.register_forward_hook(full_hook)
        try: full_logits, full_lengths, _, _ = shared.run_patch(backend, donor_batch, base_cache, ROUTES, False)
        finally: handle.remove()
        occupied_handle_calls = {"mlp11": 0}
        def occupied_hook(_m, _a, output):
            occupied_handle_calls["mlp11"] += 1; return output + occupied_correction.to(output)
        handle = backend.model.transformer.h[11].mlp.register_forward_hook(occupied_hook)
        try: occupied_logits, occupied_lengths, _, _ = shared.run_patch(backend, donor_batch, base_cache, ROUTES, False)
        finally: handle.remove()
        mode_removed_logits, mode_removed_lengths, _, removal_calls = corrected_forward(backend, donor_batch, removal_correction)
        full_margin = shared.margin(torch, full_logits, full_lengths, rows, backend.device)
        occupied_margin = shared.margin(torch, occupied_logits, occupied_lengths, rows, backend.device)
        mode_removed_margin = shared.margin(torch, mode_removed_logits, mode_removed_lengths, rows, backend.device)
    target = torch.as_tensor([row["row_id"] in target_ids for row in rows], device=backend.device)
    panels = {panel: torch.as_tensor([row["family"] == panel for row in rows], device=backend.device) for panel in ("P", "C")}
    halves = {name: target & torch.as_tensor([int(row["group_number"]) % 4 in residues for row in rows], device=backend.device)
              for name, residues in (("first", (0, 1)), ("second", (2, 3)))}
    native_delta = donor_margin-base_margin; head_removal = donor_margin-removed_margin
    full_rescue = full_margin-removed_margin; occupied_rescue = occupied_margin-removed_margin
    mode_removal = donor_margin-mode_removed_margin
    target_rms = float(native_delta[target].square().mean().sqrt())
    leak = lambda effect, panel: float(effect[panels[panel]].square().mean().sqrt())/max(target_rms, 1e-30)
    head_report = metrics(torch, head_removal, native_delta, target)
    head_report["controls"] = {panel: leak(head_removal, panel) for panel in ("P", "C")}
    head_halves = {name: metrics(torch, head_removal, native_delta, selected) for name, selected in halves.items()}
    full_report = metrics(torch, full_rescue, head_removal, target)
    occupied_to_full = metrics(torch, occupied_rescue, full_rescue, target)
    occupied_halves = {name: metrics(torch, occupied_rescue, full_rescue, selected) for name, selected in halves.items()}
    mode_removal_report = metrics(torch, mode_removal, head_removal, target)
    mode_removal_report["controls"] = {panel: leak(mode_removal, panel) for panel in ("P", "C")}
    mode_removal_halves = {name: metrics(torch, mode_removal, head_removal, selected) for name, selected in halves.items()}
    correction_calls_ok = bool(full_handle_calls["mlp11"] == 1
                               and occupied_handle_calls["mlp11"] == 1 and removal_calls["mlp11"] == 1)
    A = bool(authority_ok and mode_norm_error <= 1e-6 and self_error <= BARS["self"]
        and correction_calls_ok and finite([head_report, head_halves, full_report, occupied_to_full,
            occupied_halves, mode_removal_report, mode_removal_halves])
        and PRICE["model_forwards_exact"] == 7 and PRICE["sequence_evaluations_exact"] == 7*len(rows))
    B = bool(head_report["signed_projection"] >= BARS["head_recovery"]
        and head_report["direction_fraction"] >= BARS["direction"]
        and all(value <= BARS["control"] for value in head_report["controls"].values())
        and all(value["signed_projection"] > 0 for value in head_halves.values()))
    C = bool(full_report["signed_projection"] >= BARS["full_rescue"]
        and occupied_to_full["cosine"] >= BARS["mode_rescue_cosine"]
        and occupied_to_full["signed_projection"] >= BARS["mode_rescue_projection"]
        and all(value["signed_projection"] > 0 for value in occupied_halves.values()))
    D = bool(mode_removal_report["signed_projection"] >= BARS["mode_removal_recovery"]
        and all(value <= BARS["control"] for value in mode_removal_report["controls"].values())
        and all(value["signed_projection"] > 0 for value in mode_removal_halves.values()))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A else "selective_occupied_mode_mediator" if B and C and D
                else "four_head_necessary_multidimensional_mediator" if B else "reverse_necessity_failure")
    result = {"schema": "temporal_iswas_v23_four_head_necessity_occupied_mode_rescue_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": observed,
        "mode": {"factor_norm_error": mode_norm_error, "weight_reader_singular_values": singular.tolist()},
        "self_patch_max_abs": self_error, "correction_hook_calls": {
            "full_rescue": full_handle_calls, "occupied_rescue": occupied_handle_calls, "mode_removal": removal_calls},
        "head_removal_report": head_report, "head_removal_half_reports": head_halves,
        "full_factor_rescue_report": full_report, "occupied_to_full_rescue_report": occupied_to_full,
        "occupied_rescue_half_reports": occupied_halves, "mode_removal_report": mode_removal_report,
        "mode_removal_half_reports": mode_removal_halves, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE, "model_forwards": 7,
        "selected_mode_or_dose_after_outcome": None}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("self_patch_max_abs", "head_removal_report",
        "full_factor_rescue_report", "occupied_to_full_rescue_report", "mode_removal_report",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
