#!/usr/bin/env python3
"""Exact and tangent RMS-normalization transport from L11H3 into M11."""

# BQGATE: EXPERIMENT pred_a_authority_hooks_raw_state_self_patch_finiteness_and_exact_price pred_b_exact_rmsnorm_transport_closes_L11H3_to_M11 pred_c_rmsnorm_tangent_is_sufficient_at_finite_dose pred_d_finite_remainder_is_selective
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import rmsnorm_transport as rms
import run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1 as shared

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_l11h3_exact_rmsnorm_transport_v1.json"
OCCUPANCY = ROOT / "circuits/followups/temporal_iswas_v23_four_head_weight_mode_occupancy_transport_v1_result.json"
WEIGHT_NPZ = ROOT / "circuits/followups/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1.npz"
V23_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23.py"
RMS = ROOT / "ops/rmsnorm_transport.py"
SHARED = ROOT / "ops/run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
DAS = ROOT / "ops/circuit_das_subspace.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_l11h3_exact_rmsnorm_transport_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas.v23_l11h3_exact_rmsnorm_transport_v1"
EXPECTED = {"prior": "3a9a503cffb9c6a1a87d9a8e7594877b9a8d3a759968ac6dc93047b3578986e5",
    "occupancy": "2aae7e4bb5eb5092c64cc0248f5fa789a418ff2d78549e4d2acf438b8eef589b",
    "weight_npz": "9b553f42e708f1612fd90a934e0d26783b935457156f4856bc4b69f60df37892",
    "v23_result": "db850d5e9b86f76cb4381a12fc83f91cd2aac3544029fabd18bad138d34fed92",
    "builder": "a4830fd110b8cd854a5f28bfae776f697a15d4f791355990e02ea030fdca4c05",
    "rms": "6145377fc6492d2b254967f4ddc3519d8d980d3062965251fde76775b70ecfbc",
    "shared": "9ab2a9edb60f4e3e4befebf11f55225659559a2a504075567218eab9bf903d06",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
    "das": "49d67620b09c80edd1c999476ea9cfddb375f41016443f58cb6cc96111809d3f"}
BARS = {"self": 1e-4, "input": .002, "exact_cosine": .999, "exact_residual": .01,
        "tangent_cosine": .90, "tangent_residual": .25, "half_cosine": .90}
PRICE = {"checkpoint_loads": 1, "model_forwards_exact": 4, "sequence_evaluations_exact": 256,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = ("pred_a_authority_hooks_raw_state_self_patch_finiteness_and_exact_price",
    "pred_b_exact_rmsnorm_transport_closes_L11H3_to_M11",
    "pred_c_rmsnorm_tangent_is_sufficient_at_finite_dose",
    "pred_d_finite_remainder_is_selective")


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


def native_with_block_state(backend, batch):
    saved = {}; calls = {"block": 0}
    def hook(_module, args):
        calls["block"] += 1
        saved["block_x"], saved["x0"] = args[0].detach().clone(), args[2].detach().clone()
    handle = backend.model.transformer.h[11].register_forward_pre_hook(hook)
    try: result = shared.capture_native(backend, batch, False)
    finally: handle.remove()
    if calls["block"] != 1: raise RuntimeError("block11 native hook count changed")
    return result, saved, calls


def patch_with_block_state(backend, batch, cache, routes):
    saved = {}; calls = {"block": 0}
    def hook(_module, args):
        calls["block"] += 1
        saved["block_x"], saved["x0"] = args[0].detach().clone(), args[2].detach().clone()
    handle = backend.model.transformer.h[11].register_forward_pre_hook(hook)
    try: result = shared.run_patch(backend, batch, cache, routes, False)
    finally: handle.remove()
    if calls["block"] != 1: raise RuntimeError("block11 patch hook count changed")
    return result, saved, calls


def main():
    paths = {"prior": PRIOR, "occupancy": OCCUPANCY, "weight_npz": WEIGHT_NPZ,
             "v23_result": V23_RESULT, "builder": BUILDER, "rms": RMS,
             "shared": SHARED, "producer": PRODUCER, "das": DAS}
    observed = {name: sha(path) for name, path in paths.items()}
    occupancy, v23 = json.loads(OCCUPANCY.read_text()), json.loads(V23_RESULT.read_text())
    rows = fresh.build_rows(); target_ids = set(v23["population"]["target_row_ids"])
    authority_ok = bool(observed == EXPECTED and occupancy.get("terminal") == "nonlinear_transport_required"
        and occupancy.get("predictions", {}).get("pred_d_realized_target_use_is_sparser_and_split_stable") is True
        and v23.get("terminal") == "confirmed_selective_four_head_writer_program"
        and sum(row["row_id"] in target_ids for row in rows) == 30)
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "rows": len(rows), "target_rows": len(target_ids), "route": "L11H3", "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("L11H3 RMS transport authority changed")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = now(), time.perf_counter()
    import numpy as np
    frozen = np.load(WEIGHT_NPZ); backend = producer.Bilin18TorchBackend.load("cuda"); torch, F = backend.torch, backend.F
    reader_basis = torch.as_tensor(frozen["reader_basis"], device=backend.device).float()
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    with torch.no_grad():
        (base_bundle, base_block, base_calls) = native_with_block_state(backend, base_batch)
        base_logits, lengths, base_final, base_cache, base_saved, _ = base_bundle
        donor_bundle, _, donor_calls = native_with_block_state(backend, donor_batch)
        _, _, _, donor_cache, _, _ = donor_bundle
        (self_bundle, _, self_calls) = patch_with_block_state(backend, base_batch, base_cache, ("L11H3",))
        self_logits, self_lengths, self_final, self_x = self_bundle
        (patch_bundle, _, patch_calls) = patch_with_block_state(backend, base_batch, donor_cache, ("L11H3",))
        _, _, _, patched_x = patch_bundle
        self_error = max(float((self_x-base_saved["x"]).abs().max()), float((self_final-base_final).abs().max()),
                         float((self_logits-base_logits).abs().max()))
        block = backend.model.transformer.h[11]; attention = block.attn; dimension = int(backend.model.config.n_embd)
        mixed = block.lambdas[0]*base_block["block_x"] + block.lambdas[1]*base_block["x0"]
        native_write = attention.c_proj(base_cache["attn:11"]); raw = mixed + native_write
        eps = torch.finfo(raw.dtype).eps
        native_predicted_x = F.rms_norm(raw, (dimension,), eps=eps)
        native_input_error = float((native_predicted_x-base_saved["x"]).abs().max())
        width, head = dimension//int(backend.model.config.n_head), 3
        z_patch = base_cache["attn:11"].clone()
        for i, query in enumerate(base_batch.semantic_positions):
            z_patch[i, :int(query)+1, head*width:(head+1)*width] = donor_cache["attn:11"][i, :int(query)+1, head*width:(head+1)*width]
        patch_write = attention.c_proj(z_patch); write_delta = patch_write-native_write
        exact_predicted_x = F.rms_norm(raw+write_delta, (dimension,), eps=eps)
        patched_input_error = float((exact_predicted_x-patched_x).abs().max())
        tangent_delta = rms.tangent_delta(raw.float(), write_delta.float(), float(eps))
        base_h = shared.hidden(backend.model, base_saved["x"])
        actual = torch.einsum("an,bsn->bsa", reader_basis, shared.hidden(backend.model, patched_x)-base_h)
        exact = torch.einsum("an,bsn->bsa", reader_basis, shared.hidden(backend.model, exact_predicted_x)-base_h)
        tangent = torch.einsum("an,bsn->bsa", reader_basis,
            shared.hidden(backend.model, base_saved["x"].float()+tangent_delta)-base_h)
        remainder = exact-tangent
        valid = torch.zeros(base_h.shape[:2], dtype=torch.bool, device=backend.device)
        for i, query in enumerate(base_batch.semantic_positions): valid[i, :int(query)+1] = True
        target = valid & torch.as_tensor([row["row_id"] in target_ids for row in rows], device=backend.device)[:, None]
        controls = {panel: valid & torch.as_tensor([row["family"] == panel for row in rows], device=backend.device)[:, None]
                    for panel in ("P", "C")}
        halves = {name: target & torch.as_tensor([int(row["group_number"]) % 4 in residues for row in rows], device=backend.device)[:, None]
                  for name, residues in (("first", (0, 1)), ("second", (2, 3)))}
        exact_report, tangent_report = metrics(torch, exact, actual, target), metrics(torch, tangent, actual, target)
        half_reports = {name: metrics(torch, tangent, actual, selected) for name, selected in halves.items()}
        remainder_rms = {panel: float(remainder[selected].square().mean().sqrt())
                         for panel, selected in (("target", target), ("P", controls["P"]), ("C", controls["C"]))}
    A = bool(authority_ok and self_error <= BARS["self"] and native_input_error <= BARS["input"]
        and finite([exact_report, tangent_report, half_reports, remainder_rms])
        and all(item["block"] == 1 for item in (base_calls, donor_calls, self_calls, patch_calls))
        and PRICE["model_forwards_exact"] == 4 and PRICE["sequence_evaluations_exact"] == 4*len(rows))
    B = bool(patched_input_error <= BARS["input"] and exact_report["cosine"] >= BARS["exact_cosine"]
             and exact_report["relative_residual"] <= BARS["exact_residual"])
    C = bool(tangent_report["cosine"] >= BARS["tangent_cosine"]
        and tangent_report["relative_residual"] <= BARS["tangent_residual"]
        and min(value["cosine"] for value in half_reports.values()) >= BARS["half_cosine"])
    D = remainder_rms["P"] <= remainder_rms["target"] and remainder_rms["C"] <= remainder_rms["target"]
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A or not B else "rms_tangent_transport_program" if C and D
                else "finite_rms_transport_program" if D else "control_sensitive_finite_rms_remainder")
    result = {"schema": "temporal_iswas_v23_l11h3_exact_rmsnorm_transport_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": observed,
        "hook_calls": {"base": base_calls, "donor": donor_calls, "self": self_calls, "patch": patch_calls},
        "self_patch_max_abs": self_error, "rms_eps": float(eps),
        "native_input_reconstruction_max_abs": native_input_error,
        "patched_input_reconstruction_max_abs": patched_input_error,
        "exact_projected_hidden_report": exact_report, "tangent_projected_hidden_report": tangent_report,
        "tangent_half_reports": half_reports, "finite_remainder_rms": remainder_rms,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE,
        "model_forwards": 4, "selected_tangent_rank_or_dose": None}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("self_patch_max_abs", "native_input_reconstruction_max_abs",
        "patched_input_reconstruction_max_abs", "exact_projected_hidden_report",
        "tangent_projected_hidden_report", "tangent_half_reports", "finite_remainder_rms",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
