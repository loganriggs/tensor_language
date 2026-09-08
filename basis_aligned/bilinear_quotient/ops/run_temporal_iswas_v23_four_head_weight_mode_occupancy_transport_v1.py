#!/usr/bin/env python3
"""Overlay v23 occupancy and causal transport on frozen four-head weight modes."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_hooks_self_capture_finiteness_and_exact_price pred_b_adjacent_L11H3_preserves_direct_weight_orientation pred_c_at_least_two_early_heads_preserve_direct_weight_orientation pred_d_realized_target_use_is_sparser_and_split_stable
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
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_four_head_weight_mode_occupancy_transport_v1.json"
WEIGHT_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1_result.json"
WEIGHT_NPZ = ROOT / "circuits/followups/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1.npz"
V23_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23.py"
SHARED = ROOT / "ops/run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
DAS = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_four_head_weight_mode_occupancy_transport_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v23_four_head_weight_mode_occupancy_transport_v1"
EXPECTED = {"prior": "a005a91a211ebdde81389703c431d0dff79ba4127242623e38cd1fa5a333237e",
    "weight_result": "8065c2403bd583b71adca4adcc5e41db9ce4247e1360fd2a75ca6a34980f5393",
    "weight_npz": "9b553f42e708f1612fd90a934e0d26783b935457156f4856bc4b69f60df37892",
    "v23_result": "db850d5e9b86f76cb4381a12fc83f91cd2aac3544029fabd18bad138d34fed92",
    "builder": "a4830fd110b8cd854a5f28bfae776f697a15d4f791355990e02ea030fdca4c05",
    "shared": "9ab2a9edb60f4e3e4befebf11f55225659559a2a504075567218eab9bf903d06",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f"}
ROUTES = ("L9H1", "L9H4", "L8H1", "L11H3")
BARS = {"self": 1e-4, "adjacent_cosine": .70, "early_cosine": .70,
        "early_count": 2, "top_mode_energy": .60, "half_profile_cosine": .90}
PRICE = {"checkpoint_loads": 1, "model_forwards_exact": 7,
         "sequence_evaluations_exact": 448, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = ("pred_a_authority_alignment_hooks_self_capture_finiteness_and_exact_price",
    "pred_b_adjacent_L11H3_preserves_direct_weight_orientation",
    "pred_c_at_least_two_early_heads_preserve_direct_weight_orientation",
    "pred_d_realized_target_use_is_sparser_and_split_stable")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def vector_metrics(torch, predicted, actual, selected):
    x, y = predicted[selected].reshape(-1).double(), actual[selected].reshape(-1).double()
    xx, yy, xy = float(x@x), float(y@y), float(x@y)
    return {"cosine": xy/math.sqrt(max(xx*yy, 1e-30)), "signed_projection": xy/max(yy, 1e-30),
            "relative_residual": math.sqrt(float((x-y)@(x-y))/max(yy, 1e-30)),
            "predicted_to_actual_norm_ratio": math.sqrt(xx/max(yy, 1e-30))}


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, (int, float)) or isinstance(value, bool) or math.isfinite(float(value))


def main():
    paths = {"prior": PRIOR, "weight_result": WEIGHT_RESULT, "weight_npz": WEIGHT_NPZ,
             "v23_result": V23_RESULT, "builder": BUILDER, "shared": SHARED,
             "producer": PRODUCER, "das": DAS}
    observed = {name: sha(path) for name, path in paths.items()}
    weight_result, v23 = json.loads(WEIGHT_RESULT.read_text()), json.loads(V23_RESULT.read_text())
    rows = fresh.build_rows(); capable = set(v23["population"]["target_row_ids"])
    target_rows = [i for i, row in enumerate(rows) if row["row_id"] in capable]
    authority_ok = bool(observed == EXPECTED and weight_result.get("terminal") == "head_private_weight_capability"
        and weight_result.get("activation_occupancy_opened") is False
        and v23.get("terminal") == "confirmed_selective_four_head_writer_program"
        and len(target_rows) == 30 and all(len(row["base_ids"]) == len(row["donor_ids"])
            and row["base_semantic_position"] == row["donor_semantic_position"] for row in rows))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "rows": len(rows), "target_rows": len(target_rows), "routes": ROUTES, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("weight-mode occupancy authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = now(), time.perf_counter()
    frozen = np.load(WEIGHT_NPZ); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    cross = torch.as_tensor(frozen["cross"], device=backend.device).float()
    self_term = torch.as_tensor(frozen["self"], device=backend.device).float()
    reader_basis = torch.as_tensor(frozen["reader_basis"], device=backend.device).float()
    reader_unfolding = cross.movedim(1, 0).reshape(4, -1)
    reader_modes, reader_singular, _ = torch.linalg.svd(reader_unfolding, full_matrices=False)
    weight_mode_energy = reader_singular.square()/reader_singular.square().sum()
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    with torch.no_grad():
        base_logits, lengths, base_final, base_cache, base_saved, _ = shared.capture_native(backend, base_batch, False)
        _, _, _, donor_cache, _, _ = shared.capture_native(backend, donor_batch, False)
        self_logits, self_lengths, self_final, self_x = shared.run_patch(backend, base_batch, base_cache, ROUTES, False)
        self_error = max(float((self_x-base_saved["x"]).abs().max()), float((self_final-base_final).abs().max()),
                         float((self_logits-base_logits).abs().max()))
        base_h = shared.hidden(backend.model, base_saved["x"])
        mask = torch.zeros(base_h.shape[:2], dtype=torch.bool, device=backend.device)
        for i, query in enumerate(base_batch.semantic_positions): mask[i, :int(query)+1] = True
        target_mask = mask.clone(); target_members = torch.zeros(len(rows), dtype=torch.bool, device=backend.device)
        target_members[target_rows] = True; target_mask &= target_members[:, None]
        panel_masks = {panel: mask & torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)[:, None]
                       for panel in ("P", "C")}
        half_masks = {name: target_mask & torch.as_tensor([int(row["group_number"]) % 4 in residues for row in rows], device=backend.device)[:, None]
                      for name, residues in (("first", (0, 1)), ("second", (2, 3)))}
        reports, actuals = {}, []
        for route_index, route in enumerate(ROUTES):
            layer, head = shared.parse_head(route); width = int(backend.model.config.n_embd//backend.model.config.n_head)
            dz = (donor_cache[f"attn:{layer}"][..., head*width:(head+1)*width]
                  - base_cache[f"attn:{layer}"][..., head*width:(head+1)*width])
            x = base_saved["x"].float()
            predicted = (torch.einsum("aik,bsi,bsk->bsa", cross[route_index], x, dz)
                         + torch.einsum("akl,bsk,bsl->bsa", self_term[route_index], dz, dz))
            _, _, _, patched_x = shared.run_patch(backend, base_batch, donor_cache, (route,), False)
            actual = torch.einsum("an,bsn->bsa", reader_basis, shared.hidden(backend.model, patched_x)-base_h)
            actuals.append(actual)
            reports[route] = {"target_direct_vs_actual": vector_metrics(torch, predicted, actual, target_mask),
                              "P_direct_vs_actual": vector_metrics(torch, predicted, actual, panel_masks["P"]),
                              "C_direct_vs_actual": vector_metrics(torch, predicted, actual, panel_masks["C"]),
                              "actual_rms": {panel: float(actual[selected].square().mean().sqrt())
                                             for panel, selected in (("target", target_mask), ("P", panel_masks["P"]), ("C", panel_masks["C"]))}}
    summed_actual = torch.stack(actuals).sum(dim=0)
    mode_values = torch.einsum("bsa,am->bsm", summed_actual, reader_modes)
    target_energy = mode_values[target_mask].square().sum(dim=0); target_profile = target_energy/target_energy.sum().clamp_min(1e-30)
    half_profiles = {}
    for name, selected in half_masks.items():
        energy = mode_values[selected].square().sum(dim=0); half_profiles[name] = energy/energy.sum().clamp_min(1e-30)
    half_cosine = float((half_profiles["first"] @ half_profiles["second"])
                        / (half_profiles["first"].norm()*half_profiles["second"].norm()).clamp_min(1e-30))
    early = [route for route in ROUTES if route != "L11H3" and reports[route]["target_direct_vs_actual"]["cosine"] >= BARS["early_cosine"]]
    A = bool(authority_ok and self_error <= BARS["self"] and finite(reports)
             and PRICE["model_forwards_exact"] == 7 and PRICE["sequence_evaluations_exact"] == 7*len(rows))
    B = reports["L11H3"]["target_direct_vs_actual"]["cosine"] >= BARS["adjacent_cosine"]
    C = len(early) >= BARS["early_count"]
    D = float(target_profile.max()) >= BARS["top_mode_energy"] and half_cosine >= BARS["half_profile_cosine"]
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A else "direct_transport_sparse_occupancy" if B and C and D
                else "nonlinear_transport_required" if not B or not C else "multimode_occupancy")
    result = {"schema": "temporal_iswas_v23_four_head_weight_mode_occupancy_transport_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": observed,
        "population": {"rows": len(rows), "target_jointly_capable_rows": len(target_rows)},
        "self_patch_max_abs": self_error, "route_reports": reports, "early_orientation_preserved": early,
        "weight_reader_mode_energy_fraction": weight_mode_energy.tolist(),
        "realized_target_reader_mode_energy_fraction": target_profile.tolist(),
        "realized_half_reader_mode_energy_fraction": {name: value.tolist() for name, value in half_profiles.items()},
        "half_mode_energy_cosine": half_cosine, "weight_basis_refit_on_activations": False,
        "activation_pca_or_sae_fit": False, "predictions": predictions, "terminal": terminal,
        "bars": BARS, "price": PRICE, "model_forwards": 7}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("self_patch_max_abs", "route_reports",
        "early_orientation_preserved", "weight_reader_mode_energy_fraction",
        "realized_target_reader_mode_energy_fraction", "half_mode_energy_cosine",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
