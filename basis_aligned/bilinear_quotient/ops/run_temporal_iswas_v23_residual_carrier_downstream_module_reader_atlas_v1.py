#!/usr/bin/env python3
"""Reciprocal complete-module reader atlas for the exact block-11 residual carrier."""

# BQGATE: EXPERIMENT pred_a_authority_residual_replay_hooks_finiteness_and_exact_price pred_b_no_single_complete_downstream_module_is_a_dominant_reader pred_c_any_fit_candidate_validates_without_reselection pred_d_any_validated_candidate_is_selective pred_e_residual_branch_remains_reporter_split_stable
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
import residual_reader_module_atlas as atlas
import run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1 as shared
import run_temporal_iswas_v23_block11_residual_mlp_factorial_rescue_v1 as component
import transport_boundary_capture as boundary

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_residual_carrier_downstream_module_reader_atlas_v1.json"
FACTORIAL_PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_block11_residual_mlp_factorial_rescue_v1.json"
FACTORIAL_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_block11_residual_mlp_factorial_rescue_v1_result.json"
BINDING = ROOT / "circuits/bindings/temporal_iswas_v23_residual_carrier_downstream_module_reader_atlas_v1.json"
PRECISION_AUDIT = ROOT / "circuits/audits/temporal_iswas_v23_block11_factorial_precision_audit_v1.json"
NECESSITY = ROOT / "circuits/followups/temporal_iswas_v23_four_head_necessity_occupied_mode_rescue_v1_result.json"
CONFIRMATION = ROOT / "circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json"
HELPER = ROOT / "ops/residual_reader_module_atlas.py"
COMPONENT = ROOT / "ops/run_temporal_iswas_v23_block11_residual_mlp_factorial_rescue_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v23.py"
SHARED = ROOT / "ops/run_temporal_iswas_v22_reader_contracted_writer_effect_game_v1.py"
BOUNDARY = ROOT / "ops/transport_boundary_capture.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_residual_carrier_downstream_module_reader_atlas_v1_result.json"
EXPECTED = {"prior": "26523fe81959412e9e959183b7eda6fcaec8412114bde4f0ccb98747adb35751",
    "factorial_prior": "564c59004f32c26632fbc54973ed86cdb9c8dfdd7f19207f3922c4ad57ce6e9e",
    "necessity": "284ff3e05dddd14a979c88c6aa9e662deac4721e678a077055021f919f7a943f",
    "confirmation": "db850d5e9b86f76cb4381a12fc83f91cd2aac3544029fabd18bad138d34fed92",
    "helper": "e107bf54cd860a3f82aafb2f26a9a5ed1e69759e5af1b33d739f7de3c354745c",
    "component": "7aac1b15c2090e4812bd4de6f0adb78a58ee34e4c7598ecc77ba60ebe012bff7",
    "builder": "a4830fd110b8cd854a5f28bfae776f697a15d4f791355990e02ea030fdca4c05",
    "shared": "9ab2a9edb60f4e3e4befebf11f55225659559a2a504075567218eab9bf903d06",
    "boundary": "d027438fbd9f65b336793cd628c8d55f41510adcf36266198b43449d36cdc8b9"}
EXPECTED_PRECISION_AUDIT_SHA256 = "9b8ddd4ea12c03dd91f3affbe1e343f5de2f16e2c2f1d1ab3302e082cfa0faa3"
ROUTES = ("L8H1", "L9H1", "L9H4", "L11H3")
MODULES = tuple(name for layer in range(12, 18) for name in (f"A{layer}", f"M{layer}"))
BARS = {"replay": 1e-4, "fit_projection": .15, "fit_cosine": .85,
        "holdout_projection": .08, "holdout_cosine": .75, "direction": .75,
        "control": .15, "maximum": 4, "half_direction": .90}
PRICE = {"checkpoint_loads": 1, "model_forwards_exact": 28,
         "sequence_evaluations_exact": 1792, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = ("pred_a_authority_residual_replay_hooks_finiteness_and_exact_price",
    "pred_b_no_single_complete_downstream_module_is_a_dominant_reader",
    "pred_c_any_fit_candidate_validates_without_reselection",
    "pred_d_any_validated_candidate_is_selective",
    "pred_e_residual_branch_remains_reporter_split_stable")

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

def eligibility(binding, predecessor):
    """Require a result-conditioned, runner-bound license for this causal successor."""
    required = (component.PREDICTION_KEYS[0], component.PREDICTION_KEYS[1])
    audit = json.loads(PRECISION_AUDIT.read_text()) if PRECISION_AUDIT.exists() else {}
    return bool(
        binding.get("schema") == "temporal_iswas_v23_residual_reader_atlas_binding_v2"
        and binding.get("factorial_result_sha256") == sha(FACTORIAL_RESULT)
        and binding.get("precision_audit_sha256") == EXPECTED_PRECISION_AUDIT_SHA256
        and sha(PRECISION_AUDIT) == EXPECTED_PRECISION_AUDIT_SHA256
        and audit.get("downstream_reader_license") is True
        and audit.get("original_factorial_relabelled") is False
        and binding.get("factorial_runner_sha256") == EXPECTED["component"]
        and binding.get("atlas_runner_sha256") == sha(SELF)
        and all(predecessor.get("predictions", {}).get(key) is True for key in required)
    )

def add_block_correction(model, execute, correction):
    calls = {"block11": 0}
    def hook(_module, _args, output):
        calls["block11"] += 1; tensor = component.first_tensor(output)
        if tensor.shape != correction.shape: raise RuntimeError("residual correction shape changed")
        return component.replace_first(output, tensor + correction.to(tensor))
    handle = model.transformer.h[11].register_forward_hook(hook)
    try: result = execute()
    finally: handle.remove()
    if calls["block11"] != 1: raise RuntimeError("residual correction hook count changed")
    return result, calls

def run_path(backend, batch, base_cache, *, correction=None, capture=False,
             patch_label=None, replacement=None):
    execute = lambda: shared.run_patch(backend, batch, base_cache, ROUTES, False)
    patch_calls = {}
    if patch_label is not None:
        inner = execute
        def execute():
            nonlocal patch_calls
            result, patch_calls = atlas.execute_with_module_patch(
                backend.model, inner, label=patch_label, replacement=replacement,
                semantic_positions=batch.semantic_positions)
            return result
    capture_calls, saved = {}, None
    if capture:
        inner = execute
        def execute():
            nonlocal capture_calls, saved
            result, saved, capture_calls = atlas.capture_complete_modules(backend.model, inner)
            return result
    correction_calls = {}
    if correction is not None:
        result, correction_calls = add_block_correction(backend.model, execute, correction)
    else:
        result = execute()
    return result, saved, {"module_patch": patch_calls, "module_capture": capture_calls,
                            "residual_correction": correction_calls}

def main():
    base_paths = {"prior": PRIOR, "factorial_prior": FACTORIAL_PRIOR, "necessity": NECESSITY,
        "confirmation": CONFIRMATION, "helper": HELPER, "component": COMPONENT,
        "builder": BUILDER, "shared": SHARED, "boundary": BOUNDARY}
    observed = {name: sha(path) for name, path in base_paths.items()}
    predecessor_ready = bool(FACTORIAL_RESULT.exists() and BINDING.exists())
    predecessor = json.loads(FACTORIAL_RESULT.read_text()) if FACTORIAL_RESULT.exists() else {}
    binding = json.loads(BINDING.read_text()) if BINDING.exists() else {}
    predecessor_hash_ok = bool(predecessor_ready and eligibility(binding, predecessor))
    dry = {"candidate_id": "cross_task.temporal_iswas.v23_residual_carrier_downstream_module_reader_atlas_v1",
        "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
        "base_authority_ok": observed == EXPECTED, "predecessor_ready": predecessor_ready,
        "predecessor_hash_ok": predecessor_hash_ok, "modules": MODULES, "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dry, sort_keys=True)); return
    if observed != EXPECTED or not predecessor_hash_ok:
        raise RuntimeError("downstream reader atlas is not bound to a valid predecessor")
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = now(), time.perf_counter(); rows = fresh.build_rows()
    confirmation = json.loads(CONFIRMATION.read_text()); target_ids = set(confirmation["population"]["target_row_ids"])
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    base_batch = das._batch(backend, rows, side="base"); donor_batch = das._batch(backend, rows, side="donor")
    with torch.no_grad():
        base_logits, base_lengths, _, base_cache, _, _ = shared.capture_native(backend, base_batch, False)
        donor_pack, donor_state, _donor_output, donor_calls = component.execute_capture(
            backend.model, lambda: shared.capture_native(backend, donor_batch, False))
        donor_logits, donor_lengths, _, _, _, _ = donor_pack
        removed_outer, removed_state, _removed_output, removed_boundary_calls = component.execute_capture(
            backend.model, lambda: run_path(backend, donor_batch, base_cache, capture=True))
        removed_pack, removed_modules, removed_path_calls = removed_outer
        removed_logits, removed_lengths, _, _ = removed_pack
        delta_y = (boundary.pre_mlp_raw_state(backend.model, donor_state, layer=11)
                   - boundary.pre_mlp_raw_state(backend.model, removed_state, layer=11))
        rescued_pack, rescued_modules, rescued_calls = run_path(
            backend, donor_batch, base_cache, correction=delta_y, capture=True)
        rescued_logits, rescued_lengths, _, _ = rescued_pack
        transfer_logits, reset_logits, arm_calls = {}, {}, {}
        for label in MODULES:
            transfer_pack, _, transfer_calls = run_path(
                backend, donor_batch, base_cache, patch_label=label, replacement=rescued_modules[label])
            reset_pack, _, reset_calls = run_path(
                backend, donor_batch, base_cache, correction=delta_y,
                patch_label=label, replacement=removed_modules[label])
            transfer_logits[label] = transfer_pack[:2]; reset_logits[label] = reset_pack[:2]
            arm_calls[label] = {"transfer": transfer_calls, "reset": reset_calls}
        base_margin = shared.margin(torch, base_logits, base_lengths, rows, backend.device)
        donor_margin = shared.margin(torch, donor_logits, donor_lengths, rows, backend.device)
        removed_margin = shared.margin(torch, removed_logits, removed_lengths, rows, backend.device)
        rescued_margin = shared.margin(torch, rescued_logits, rescued_lengths, rows, backend.device)
        transfer_margins = {label: shared.margin(torch, *value, rows, backend.device)
                            for label, value in transfer_logits.items()}
        reset_margins = {label: shared.margin(torch, *value, rows, backend.device)
                         for label, value in reset_logits.items()}
    target = torch.as_tensor([row["row_id"] in target_ids for row in rows], device=backend.device)
    phases = {"FIT": target & torch.as_tensor([int(row["group_number"])%4 in (0,1) for row in rows], device=backend.device),
              "HOLDOUT": target & torch.as_tensor([int(row["group_number"])%4 in (2,3) for row in rows], device=backend.device)}
    panels = {name: torch.as_tensor([row["family"] == name for row in rows], device=backend.device) for name in ("P", "C")}
    residual_effect = rescued_margin-removed_margin; native_delta = donor_margin-base_margin
    target_rms = float(native_delta[target].square().mean().sqrt())
    reports = {}
    for label in MODULES:
        transfer_effect = transfer_margins[label]-removed_margin
        reset_loss = rescued_margin-reset_margins[label]
        reports[label] = {}
        for phase, mask in phases.items():
            reports[label][phase] = {"transfer": metrics(torch, transfer_effect, residual_effect, mask),
                                     "reset": metrics(torch, reset_loss, residual_effect, mask)}
        reports[label]["controls"] = {kind: {panel: float(effect[mask].square().mean().sqrt())/max(target_rms,1e-30)
            for panel, mask in panels.items()} for kind, effect in (("transfer",transfer_effect),("reset",reset_loss))}
    fit_reports = {label: reports[label]["FIT"] for label in MODULES}
    selected = atlas.select_reciprocal_candidates(fit_reports, fit_projection=BARS["fit_projection"],
                                                   fit_cosine=BARS["fit_cosine"], maximum=BARS["maximum"])
    residual_report = metrics(torch, residual_effect, donor_margin-removed_margin, target)
    half_reports = {phase: metrics(torch, residual_effect, donor_margin-removed_margin, mask) for phase,mask in phases.items()}
    replay_fields = ("cosine", "signed_projection", "relative_residual", "norm_ratio", "direction_fraction")
    replay_error = max(abs(residual_report[key]-predecessor["reports"]["residual"][key]) for key in replay_fields)
    hook_values = list(donor_calls.values()) + list(removed_boundary_calls.values())
    hook_values += list(removed_path_calls["module_capture"].values()) + list(rescued_calls["module_capture"].values())
    hook_values += list(rescued_calls["residual_correction"].values())
    for calls in arm_calls.values():
        for direction in calls.values():
            hook_values += list(direction["module_patch"].values()) + list(direction["residual_correction"].values())
    A = bool(replay_error <= BARS["replay"] and set(hook_values) == {1} and finite([reports, residual_report, half_reports])
        and PRICE["model_forwards_exact"] == 28 and PRICE["sequence_evaluations_exact"] == 28*len(rows))
    B = not selected
    C = None if B else bool(all(all(reports[label]["HOLDOUT"][kind]["signed_projection"] >= BARS["holdout_projection"]
        and reports[label]["HOLDOUT"][kind]["cosine"] >= BARS["holdout_cosine"]
        and reports[label]["HOLDOUT"][kind]["direction_fraction"] >= BARS["direction"] for kind in ("transfer","reset"))
        for label in selected))
    D = None if B else bool(all(all(value <= BARS["control"] for kind in ("transfer","reset")
        for value in reports[label]["controls"][kind].values()) for label in selected))
    E = bool(all(value["signed_projection"] > 0 and value["direction_fraction"] >= BARS["half_direction"]
                 for value in half_reports.values()))
    predictions = dict(zip(PREDICTION_KEYS, (A, B, C, D, E)))
    terminal = ("invalid_instrument" if not A else "direct_residual_readout_candidate" if B and E else
                "validated_selective_module_readers" if C and D and E else "distributed_or_unstable_module_readers")
    result = {"schema": "temporal_iswas_v23_residual_carrier_downstream_module_reader_atlas_result_v1",
        "candidate_id": dry["candidate_id"], "started_utc": started_utc, "finished_utc": now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": {**observed,
            "factorial_result": sha(FACTORIAL_RESULT), "binding": sha(BINDING)},
        "residual_replay_max_abs": replay_error,
        "residual_report": residual_report, "residual_half_reports": half_reports,
        "reports": reports, "selected_modules": selected, "hook_count_values": sorted(set(hook_values)),
        "predictions": predictions, "prediction_applicability": {PREDICTION_KEYS[2]: not B, PREDICTION_KEYS[3]: not B},
        "terminal": terminal, "bars": BARS, "price": PRICE, "model_forwards": 28,
        "selected_module_or_threshold_after_outcome": None}
    atomic_create_json(OUT, result)
    print(json.dumps({"residual_replay_max_abs": replay_error, "residual_report": residual_report,
        "selected_modules": selected, "reports": reports, "predictions": predictions,
        "terminal": terminal, "price": PRICE}, sort_keys=True))

if __name__ == "__main__": main()
