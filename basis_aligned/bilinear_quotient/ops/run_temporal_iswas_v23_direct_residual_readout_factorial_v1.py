#!/usr/bin/env python3
"""Exact residual-identity versus downstream-response telescope for v23."""

# BQGATE: EXPERIMENT pred_a_hash_hooks_state_closure_logit_replay_finiteness_and_exact_price pred_b_direct_identity_carry_is_dominant_selective_and_split_stable pred_c_downstream_response_is_a_real_sign_stable_correction pred_d_final_decoder_direct_response_interaction_is_small pred_e_analytic_carry_is_exact_finite_and_nonzero
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v23_residual_carrier_downstream_module_reader_atlas_v1 as reader


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT.parent / "polynomial_causal/TEMPORAL_ISWAS_V23_DIRECT_RESIDUAL_READOUT_FACTORIAL_V1_PREREGISTRATION.md"
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_direct_residual_readout_factorial_v1.json"
PARENT = ROOT / "circuits/followups/temporal_iswas_v23_residual_carrier_downstream_module_reader_atlas_v1_result.json"
READER = Path(reader.__file__).resolve()
COMPONENT = reader.COMPONENT
SHARED = reader.SHARED
OUT = ROOT / "circuits/followups/temporal_iswas_v23_direct_residual_readout_factorial_v1_result.json"
EXPECTED = {
    "authority": "8cb689a7ce59a67c6ab953a71bd7fe22aa23d0b3caec38a8ca8897946c4b922a",
    "prior": "0c2e00ec0a0b81e2cb5e1bff99b69c91d004c357077867e42cb0314262b96d85",
    "parent": "7dcf47112e757e3e150684622b7693863b28dc6aa51859ef9bf7428c97a26d53",
    "reader": "421bbf43d5a93df00785820a5efc02e7ce00d05e65868a6de70b729a0f2ceacb",
    "component": "7aac1b15c2090e4812bd4de6f0adb78a58ee34e4c7598ecc77ba60ebe012bff7",
    "shared": "9ab2a9edb60f4e3e4befebf11f55225659559a2a504075567218eab9bf903d06",
}
BARS = {"state_closure": 1e-4, "logit_replay": 1e-4,
        "direct_projection": .65, "direct_cosine": .90, "direction": .90,
        "control": .15, "response_abs_projection": .05,
        "decoder_interaction_abs_projection": .10}
PRICE = {"checkpoint_loads": 1, "model_forwards_exact": 4,
         "sequence_evaluations_exact": 256, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_hash_hooks_state_closure_logit_replay_finiteness_and_exact_price",
    "pred_b_direct_identity_carry_is_dominant_selective_and_split_stable",
    "pred_c_downstream_response_is_a_real_sign_stable_correction",
    "pred_d_final_decoder_direct_response_interaction_is_small",
    "pred_e_analytic_carry_is_exact_finite_and_nonzero",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, (int, float)) or isinstance(value, bool) or math.isfinite(value)


def carry_coefficient(model):
    coefficient = 1.0
    factors = []
    for layer in range(12, 18):
        value = float(model.transformer.h[layer].lambdas[0].float())
        factors.append(value)
        coefficient *= value
    return coefficient, factors


def main():
    paths = {"authority": AUTHORITY, "prior": PRIOR, "parent": PARENT,
             "reader": READER, "component": COMPONENT, "shared": SHARED}
    observed = {name: sha(path) for name, path in paths.items()}
    parent = json.loads(PARENT.read_text())
    dependency_ok = bool(
        observed == EXPECTED and parent.get("terminal") == "direct_residual_readout_candidate"
        and parent.get("predictions", {}).get(
            "pred_a_authority_residual_replay_hooks_finiteness_and_exact_price") is True
        and parent.get("predictions", {}).get(
            "pred_b_no_single_complete_downstream_module_is_a_dominant_reader") is True
        and parent.get("predictions", {}).get(
            "pred_e_residual_branch_remains_reporter_split_stable") is True
        and parent.get("selected_modules") == []
    )
    dryrun = {
        "candidate_id": "cross_task.temporal_iswas.v23_direct_residual_readout_factorial_v1",
        "dryrun": True, "gpu_accessed": False, "model_loaded": False,
        "queue_touched": False, "dependency_ok": dependency_ok,
        "rows": 64, "paths": ["base", "donor", "removed", "rescued"],
        "carry_layers": list(range(12, 18)), "decoder_arms": ["direct", "response", "joint"],
        "v25_accessed": False, "bars": BARS, "price": PRICE,
    }
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dryrun, sort_keys=True))
        return
    if not dependency_ok:
        raise RuntimeError("direct residual readout dependency changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.perf_counter()
    rows = reader.fresh.build_rows()
    confirmation = json.loads(reader.CONFIRMATION.read_text())
    target_ids = set(confirmation["population"]["target_row_ids"])
    backend = reader.producer.Bilin18TorchBackend.load("cuda")
    torch, F = backend.torch, backend.F
    base_batch = reader.das._batch(backend, rows, side="base")
    donor_batch = reader.das._batch(backend, rows, side="donor")
    forwards = 0
    with torch.no_grad():
        base_logits, base_lengths, _, base_cache, _, _ = reader.shared.capture_native(
            backend, base_batch, False)
        forwards += 1
        donor_pack, donor_state, _donor_output, donor_calls = reader.component.execute_capture(
            backend.model, lambda: reader.shared.capture_native(backend, donor_batch, False))
        donor_logits, donor_lengths, donor_final, _, _, _ = donor_pack
        forwards += 1
        removed_pack, removed_state, _removed_output, removed_calls = reader.component.execute_capture(
            backend.model, lambda: reader.shared.run_patch(
                backend, donor_batch, base_cache, reader.ROUTES, False))
        removed_logits, removed_lengths, removed_final, _ = removed_pack
        forwards += 1
        delta_y = (reader.boundary.pre_mlp_raw_state(backend.model, donor_state, layer=11)
                   - reader.boundary.pre_mlp_raw_state(backend.model, removed_state, layer=11))
        rescued_pack, _saved, rescued_calls = reader.run_path(
            backend, donor_batch, base_cache, correction=delta_y)
        rescued_logits, rescued_lengths, rescued_final, _ = rescued_pack
        forwards += 1
        coefficient, lambda_factors = carry_coefficient(backend.model)
        direct_state = coefficient * delta_y
        response_state = rescued_final.float() - removed_final.float() - direct_state.float()
        reconstructed_final = removed_final.float() + direct_state.float() + response_state

        def decode(final):
            return 30.0 * torch.tanh(
                backend.model.lm_head(F.rms_norm(final, (backend.model.config.n_embd,))) / 30.0)

        direct_logits = decode(removed_final.float() + direct_state.float())
        response_logits = decode(removed_final.float() + response_state)
        joint_logits = decode(reconstructed_final)

    base_margin = reader.shared.margin(torch, base_logits, base_lengths, rows, backend.device)
    donor_margin = reader.shared.margin(torch, donor_logits, donor_lengths, rows, backend.device)
    removed_margin = reader.shared.margin(torch, removed_logits, removed_lengths, rows, backend.device)
    rescued_margin = reader.shared.margin(torch, rescued_logits, rescued_lengths, rows, backend.device)
    direct_margin = reader.shared.margin(torch, direct_logits, removed_lengths, rows, backend.device)
    response_margin = reader.shared.margin(torch, response_logits, removed_lengths, rows, backend.device)
    joint_margin = reader.shared.margin(torch, joint_logits, removed_lengths, rows, backend.device)
    full_effect = rescued_margin - removed_margin
    effects = {"direct": direct_margin - removed_margin,
               "response": response_margin - removed_margin,
               "joint": joint_margin - removed_margin}
    effects["decoder_interaction"] = effects["joint"] - effects["direct"] - effects["response"]
    target = torch.as_tensor([row["row_id"] in target_ids for row in rows], device=backend.device)
    controls = {panel: torch.as_tensor([row["family"] == panel for row in rows],
                                      device=backend.device) for panel in ("P", "C")}
    halves = {name: target & torch.as_tensor(
        [int(row["group_number"]) % 4 in residues for row in rows], device=backend.device)
        for name, residues in (("first", (0, 1)), ("second", (2, 3)))}
    target_rms = full_effect[target]

    def report(effect):
        return {
            "target": reader.metrics(torch, effect, full_effect, target),
            "controls": {panel: reader.shared.rms_ratio(effect[mask], target_rms)
                         for panel, mask in controls.items()},
            "halves": {name: reader.metrics(torch, effect, full_effect, mask)
                       for name, mask in halves.items()},
        }

    reports = {name: report(effect) for name, effect in effects.items()}
    state_closure = float((reconstructed_final - rescued_final.float()).abs().max())
    logit_replay = float((joint_logits - rescued_logits.float()).abs().max())
    hook_values = list(donor_calls.values()) + list(removed_calls.values())
    hook_values += list(rescued_calls["residual_correction"].values())
    A = bool(observed == EXPECTED and dependency_ok and forwards == 4
             and forwards * len(rows) == 256 and set(hook_values) == {1}
             and state_closure <= BARS["state_closure"]
             and logit_replay <= BARS["logit_replay"]
             and finite([reports, coefficient, lambda_factors]))
    direct = reports["direct"]
    B = bool(direct["target"]["signed_projection"] >= BARS["direct_projection"]
             and direct["target"]["cosine"] >= BARS["direct_cosine"]
             and direct["target"]["direction_fraction"] >= BARS["direction"]
             and all(direct["controls"][panel] <= BARS["control"] for panel in ("P", "C"))
             and all(direct["halves"][half]["signed_projection"] > 0
                     for half in ("first", "second")))
    response = reports["response"]
    response_projection = response["target"]["signed_projection"]
    half_response = [response["halves"][half]["signed_projection"]
                     for half in ("first", "second")]
    C = bool(abs(response_projection) >= BARS["response_abs_projection"]
             and half_response[0] * half_response[1] > 0)
    D = abs(reports["decoder_interaction"]["target"]["signed_projection"]) <= \
        BARS["decoder_interaction_abs_projection"]
    recomputed = math.prod(lambda_factors)
    E = bool(math.isfinite(coefficient) and coefficient != 0
             and math.isclose(coefficient, recomputed, rel_tol=0, abs_tol=1e-15))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = "invalid_instrument" if not A else (
        "direct_residual_readout_program_screen" if B and E
        else "distributed_downstream_response")
    result = {
        "schema": "temporal_iswas_v23_direct_residual_readout_factorial_result_v1",
        "candidate_id": dryrun["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": observed, "parent_terminal": parent["terminal"],
        "carry": {"layers": list(range(12, 18)), "lambda0_factors": lambda_factors,
                  "coefficient": coefficient},
        "state_closure_max_abs": state_closure, "rescued_logit_replay_max_abs": logit_replay,
        "reports": reports, "predictions": predictions, "terminal": terminal,
        "bars": BARS, "price": PRICE, "model_forwards": forwards,
        "v25_accessed": False, "fit_parameters": 0,
        "selected_module_or_threshold_after_outcome": None,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "carry", "state_closure_max_abs", "rescued_logit_replay_max_abs",
        "reports", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
