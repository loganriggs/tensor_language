#!/usr/bin/env python3
"""Compile the fresh-confirmed five-MLP responses into literal bilinear weights."""

# BQGATE: EXPERIMENT pred_a_authority_architecture_gauge_replay_finiteness_and_price pred_b_all_five_local_weight_expansions_close pred_c_weight_propagated_modes_equal_causal_site_effects pred_d_all_five_weight_tensors_have_causal_mode2_support pred_e_complete_zero_fit_literal_factor_program
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v13 as temporal
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as iswas
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import attention_source_destination_eval as attention_eval
import run_temporal_iswas_canonical_downstream_response_removal_atlas_v1 as atlas
import run_temporal_iswas_downstream_ten_site_response_lattice_v1 as lattice
import run_temporal_iswas_upstream_full_response_mode_atlas_v1 as response
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population
import subspace_weight_atlas as weight_atlas

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_literal_weight_tensor_program_v1.json"
CONFIRMATION = ROOT / "circuits/followups/temporal_v13_iswas_v12_four_five_mlp_program_v1_result.json"
CONFIRMATION_AUDIT = ROOT / "circuits/followups/temporal_v13_iswas_v12_four_five_mlp_program_v1_authority_audit.json"
CONFIRMATION_RUNNER = ROOT / "ops/run_temporal_v13_iswas_v12_four_five_mlp_program_v1.py"
WEIGHT_LIBRARY = ROOT / "ops/subspace_weight_atlas.py"
TEMPORAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v13.py"
TEMPORAL_CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
ISWAS_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
ISWAS_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
WEIGHT_RESULT = ROOT / "circuits/followups/temporal_iswas_two_mode_weight_pullback_v3_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_literal_weight_tensor_program_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.five_mlp_literal_weight_tensor_program_v1"
SITES = ("MLP12", "MLP13", "MLP15", "MLP16", "MLP17")
TERMS = ("left_change", "right_change", "bilinear_interaction")
MAX_FORWARDS, MAX_EVALUATIONS = 34, 1836
EXPECTED = {
    "prior": "fbde996c00cf20c8f19c5709b493e5dbf14e08f8f94eb6aecba1c94a0e2f1d99",
    "confirmation": "9fc9c6aad9257ba1d75745bc1938ec2451120c1363a66ec933cae222d43c515d",
    "confirmation_audit": "c648b261dd2f541a8d99e8a0da75b547ddbe356db5d78f687ba162ea4c0e02a5",
    "confirmation_runner": "dac30cf1b67502fae42c2db9546bf2e8f4d25db2c9627a19b3b64d7ac9fee18a",
    "weight_library": "2e7d3a546813a6029eca6fae455ad5abd03b429fcee92432ca6fe06e835e83f5",
    "temporal_builder": "3f738bf2fb2d4a5425dba85eaf948d7ed888e4b891666ff77a51c8638acd2509",
    "temporal_capability": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
    "iswas_builder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2",
    "iswas_capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
    "weight_result": "c8ab608fa116342f9cbc8af4955e6087faa0f1eee9dd74dacb5c0ec168c5bf4d",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def capture_selected_io(backend, call):
    values, handles = {}, []
    for site in SITES:
        layer = int(site.removeprefix("MLP"))
        module = backend.model.transformer.h[layer].mlp

        def save_input(_module, arguments, site=site):
            values.setdefault(site, {})["input"] = arguments[0].detach().clone()

        def save_output(_module, _arguments, output, site=site):
            values.setdefault(site, {})["output"] = output.detach().clone()

        handles.append(module.register_forward_pre_hook(save_input))
        handles.append(module.register_forward_hook(save_output))
    try:
        output, downstream = atlas.capture_downstream(backend, call)
    finally:
        for handle in handles:
            handle.remove()
    if set(values) != set(SITES) or any(set(values[site]) != {"input", "output"} for site in SITES):
        raise RuntimeError("incomplete selected-MLP input/output capture")
    return output, downstream, values


def main():
    paths = {
        "prior": PRIOR, "confirmation": CONFIRMATION,
        "confirmation_audit": CONFIRMATION_AUDIT, "confirmation_runner": CONFIRMATION_RUNNER,
        "weight_library": WEIGHT_LIBRARY, "temporal_builder": TEMPORAL_BUILDER,
        "temporal_capability": TEMPORAL_CAPABILITY, "iswas_builder": ISWAS_BUILDER,
        "iswas_capability": ISWAS_CAPABILITY, "weight_result": WEIGHT_RESULT,
    }
    if {key: sha(value) for key, value in paths.items()} != EXPECTED:
        raise RuntimeError("literal weight-program authority changed")
    prior = json.loads(PRIOR.read_text())
    confirmation = json.loads(CONFIRMATION.read_text())
    audit = json.loads(CONFIRMATION_AUDIT.read_text())
    tcap = json.loads(TEMPORAL_CAPABILITY.read_text())
    icap = json.loads(ISWAS_CAPABILITY.read_text())
    weights = json.loads(WEIGHT_RESULT.read_text())
    temporal_rows = sum((population.capable_rows(temporal, tcap, panel, 12)
                         for panel in ("A1", "A2")), [])
    iswas_rows = sum((population.capable_rows(iswas, icap, panel)
                      for panel in ("A1", "A2")), [])
    rows = temporal_rows + iswas_rows
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "sites": list(SITES), "terms": list(TERMS), "factor_arms": len(SITES) * len(TERMS),
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    authority_ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and confirmation.get("terminal") == "temporal_fresh_five_mlp_response_program"
        and audit.get("verdict") == "stale_inherited_three_site_guard_not_an_authority_or_scientific_failure"
        and tcap.get("terminal") == "manifest"
        and icap.get("terminal") == "screen"
        and len(temporal_rows) == 24 and len(iswas_rows) == 30
        and len({row["row_id"] for row in rows}) == 54
    )
    if not authority_ok or len(rows) != 54 or set(SITES) - set(atlas.SITES):
        raise RuntimeError("literal weight-program population or decision changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, _base_up = response.capture(backend, base_batch)
    donor_output, donor_up = response.capture(backend, donor_batch)
    base_down_output, base_down, base_io = capture_selected_io(
        backend, lambda: backend.native(base_batch, capture=True))
    live_pair, live_down, live_io = capture_selected_io(
        backend, lambda: response.run_patch(backend, base_batch, donor_up, atlas.WRITER_SITES))
    live_output = live_pair[0]
    self_output = lattice.run_mixed(
        backend, base_batch, base_down, lambda: backend.native(base_batch, capture=True))
    replay_pair = lattice.run_mixed(
        backend, base_batch, live_down,
        lambda: response.run_patch(backend, base_batch, donor_up, atlas.WRITER_SITES))
    replay_output = replay_pair[0]
    base_state, donor_state, live_state, self_state, replay_state = (
        response.states(torch, backend, output, rows)
        for output in (base_output, donor_output, live_output, self_output, replay_output)
    )
    identity_error = max(
        float((response.states(torch, backend, base_down_output, rows) - base_state).abs().max()),
        float((self_state - base_state).abs().max()),
        float((replay_state - live_state).abs().max()),
    )
    reconstruction = 0.0
    for layer in atlas.LAYERS:
        _replay, captured = attention_eval.capture_layer_attention(backend, base_batch, layer)
        reconstruction = max(
            reconstruction,
            float((captured["head_output"].reshape_as(base_down[f"L{layer}H0"])
                   - base_down[f"L{layer}H0"]).abs().max()),
        )
    family, _singular, _energy = atlas.frontier.parent.family_builder.build_family(
        backend, json.loads(atlas.frontier.parent.SUBSPACE.read_text()))
    q = family[8]
    full_gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                          for layer in atlas.LAYERS)
    raw_modes, orientation_error, _wrong = atlas.frontier.parent.overlap.residual_modes(
        backend, q, full_gain)
    state_basis = torch.linalg.qr(raw_modes, mode="reduced").Q
    reader_coordinates = torch.as_tensor(
        weights["mode_artifacts"]["reader_coordinates"], device=backend.device).float()
    physical_reader = state_basis @ reader_coordinates
    reader_hash_ok = (
        tensor_sha(physical_reader)
        == weights["mode_artifacts"]["physical_reader_covectors_sha256"]
    )
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    index = torch.arange(len(rows), device=backend.device)

    def margin(state):
        logits = das.head_logits(backend, state)
        return logits[index, answer] - logits[index, foil]

    base_margin, live_margin = margin(base_state), margin(live_state)
    full_behavior = live_margin - base_margin
    full_modes = (live_state - base_state) @ physical_reader
    task_indices = {
        "temporal": torch.arange(0, len(temporal_rows), device=backend.device),
        "iswas": torch.arange(len(temporal_rows), len(rows), device=backend.device),
    }
    factor_outputs, local_metrics, tensor_manifests = {}, {}, {}
    architectures_ok = True
    for site in SITES:
        layer = int(site.removeprefix("MLP"))
        mlp = backend.model.transformer.h[layer].mlp
        architectures_ok = architectures_ok and not bool(mlp.config.gated)
        x0, x1 = base_io[site]["input"], live_io[site]["input"]
        with torch.no_grad():
            left0, left1 = mlp.Left(x0).float(), mlp.Left(x1).float()
            right0, right1 = mlp.Right(x0).float(), mlp.Right(x1).float()
            delta_left, delta_right = left1 - left0, right1 - right0
            hidden = {
                "left_change": delta_left * right0,
                "right_change": left0 * delta_right,
                "bilinear_interaction": delta_left * delta_right,
            }
            down = mlp.Down.weight.detach().float()
            factors = {name: torch.nn.functional.linear(value, down, None)
                       for name, value in hidden.items()}
        factor_outputs[site] = factors
        observed = live_io[site]["output"].float() - base_io[site]["output"].float()
        reconstructed = sum(factors.values(), torch.zeros_like(observed))
        error = reconstructed - observed
        relative = float(error.square().sum() / observed.square().sum().clamp_min(1e-30))
        later_gain = math.prod(
            float(backend.model.transformer.h[later].lambdas[0].detach().float())
            for later in range(layer + 1, 18))
        read_map = (later_gain * physical_reader).T
        tensor_item = weight_atlas.mlp_writer_to_read_tensor(mlp, read_map)
        tensor_manifests[site] = {
            "layer": layer, "later_residual_gain": later_gain,
            "architecture": "ungated_bilinear",
            "shapes": {
                "read_down": list(tensor_item["output"].shape),
                "left": list(tensor_item["left"].shape),
                "right": list(tensor_item["right"].shape),
                "tensor": list(tensor_item["tensor"].shape),
            },
            "sha256": {
                "read_down": tensor_sha(tensor_item["output"]),
                "left": tensor_sha(tensor_item["left"]),
                "right": tensor_sha(tensor_item["right"]),
                "tensor": tensor_sha(tensor_item["tensor"]),
            },
            "tensor_frobenius": tensor_item["score"],
            "tensor_normalized_score": tensor_item["normalized_score"],
        }
        local_metrics[site] = {
            "max_abs_error": float(error.abs().max()),
            "relative_squared_error": relative,
            "observed_output_norm": float(observed.norm()),
            "term_output_norms": {name: float(value.norm()) for name, value in factors.items()},
        }
        del tensor_item
    selected_values = {site: (live_down[site] if site in SITES else base_down[site])
                       for site in atlas.SITES}
    direct_values = {site: base_down[site] for site in atlas.SITES}
    price_counter = {"model_forwards": 12, "example_evaluations": 12 * len(rows)}

    def execute(values):
        pair = lattice.run_mixed(
            backend, base_batch, values,
            lambda: response.run_patch(backend, base_batch, donor_up, atlas.WRITER_SITES))
        price_counter["model_forwards"] += 1
        price_counter["example_evaluations"] += len(rows)
        return response.states(torch, backend, pair[0], rows)

    direct_state = execute(direct_values)
    selected_state = execute(selected_values)
    minus_states = {}
    for site in SITES:
        values = dict(selected_values)
        values[site] = base_down[site]
        minus_states[site] = execute(values)
    term_states = {}
    for site in SITES:
        for term in TERMS:
            values = dict(selected_values)
            values[site] = base_down[site].float() + factor_outputs[site][term]
            term_states[(site, term)] = execute(values)
    propagation, term_metrics = {}, {}
    for site in SITES:
        layer = int(site.removeprefix("MLP"))
        later_gain = tensor_manifests[site]["later_residual_gain"]
        local_delta = response.query_rows(
            live_io[site]["output"].float() - base_io[site]["output"].float(), base_batch)
        predicted_modes = later_gain * (local_delta @ physical_reader)
        actual_modes = (selected_state - minus_states[site]) @ physical_reader
        mode_error = predicted_modes - actual_modes
        propagation[site] = {
            "mode_max_abs_error": float(mode_error.abs().max()),
            "mode_relative_squared_error": float(
                mode_error.square().sum() / actual_modes.square().sum().clamp_min(1e-30)),
            "tasks": {},
        }
        site_behavior = margin(selected_state) - margin(minus_states[site])
        for task, ids in task_indices.items():
            propagation[site]["tasks"][task] = {
                "behavior": response.vector_stats(torch, site_behavior[ids], full_behavior[ids]),
                "mode1": response.vector_stats(torch, actual_modes[ids, 0], full_modes[ids, 0]),
                "mode2": response.vector_stats(torch, actual_modes[ids, 1], full_modes[ids, 1]),
            }
        term_metrics[site] = {}
        for term in TERMS:
            state = term_states[(site, term)]
            term_behavior = margin(state) - margin(minus_states[site])
            term_modes = (state - minus_states[site]) @ physical_reader
            term_metrics[site][term] = {
                task: {
                    "behavior": response.vector_stats(torch, term_behavior[ids], site_behavior[ids]),
                    "mode1": response.vector_stats(torch, term_modes[ids, 0], actual_modes[ids, 0]),
                    "mode2": response.vector_stats(torch, term_modes[ids, 1], actual_modes[ids, 1]),
                }
                for task, ids in task_indices.items()
            }
    finite = all(
        math.isfinite(value)
        for metrics in local_metrics.values()
        for key, item in metrics.items()
        for value in (item.values() if isinstance(item, dict) else (item,))
    ) and all(math.isfinite(manifest[key]) for manifest in tensor_manifests.values()
              for key in ("later_residual_gain", "tensor_frobenius", "tensor_normalized_score"))
    pred_a = bool(
        authority_ok and architectures_ok and reader_hash_ok and orientation_error <= 1e-6
        and identity_error <= 1e-4 and reconstruction <= 5e-4 and finite
        and price_counter["model_forwards"] == MAX_FORWARDS
        and price_counter["example_evaluations"] == MAX_EVALUATIONS
    )
    pred_b = all(metrics["max_abs_error"] <= 0.001
                 and metrics["relative_squared_error"] <= 1e-8
                 for metrics in local_metrics.values())
    pred_c = all(metrics["mode_max_abs_error"] <= 0.001
                 and metrics["mode_relative_squared_error"] <= 1e-6
                 for metrics in propagation.values())
    pred_d = all(propagation[site]["tasks"][task]["mode2"]["signed_projection"] >= 0.02
                 for site in SITES for task in task_indices)
    pred_e = (len(tensor_manifests) == 5 and len(term_states) == 15
              and all(manifest["tensor_frobenius"] > 0 for manifest in tensor_manifests.values()))
    forwards, evaluations = price_counter["model_forwards"], price_counter["example_evaluations"]
    predictions = {
        "pred_a_authority_architecture_gauge_replay_finiteness_and_price": pred_a,
        "pred_b_all_five_local_weight_expansions_close": bool(pred_b),
        "pred_c_weight_propagated_modes_equal_causal_site_effects": bool(pred_c),
        "pred_d_all_five_weight_tensors_have_causal_mode2_support": bool(pred_d),
        "pred_e_complete_zero_fit_literal_factor_program": bool(pred_e),
    }
    terminal = (
        "invalid" if not pred_a or not pred_b
        else "literal_bilinear_weight_program" if all(predictions.values())
        else "cached_to_weight_compilation_null"
    )
    result = {
        "schema": "temporal_five_mlp_literal_weight_tensor_program_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "evidence_scope": {"temporal": "fresh_v13", "iswas": "v12_replay_control"},
        "dependency_boundary": {
            "compiled": "literal checkpoint Left/Right/Down weights, paired normalized MLP inputs, later residual gains, and physical mode readers",
            "uncompiled": "upstream generation of the activation-conditioned paired MLP inputs",
        },
        "instrument": {
            "authority_ok": authority_ok, "ungated_bilinear_architecture": architectures_ok,
            "physical_reader_hash_ok": reader_hash_ok, "orientation_max_abs": orientation_error,
            "identity_self_and_live_replay_max_abs": identity_error,
            "attention_reconstruction_max_abs": reconstruction,
        },
        "tensor_manifests": tensor_manifests, "local_weight_closure": local_metrics,
        "propagated_mode_checks": propagation, "factor_intervention_metrics": term_metrics,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "factor_arms": len(term_states), "fit_updates": 0,
                  "model_updates": 0, "transformer_backwards": 0},
        "serial_seconds": time.perf_counter() - started,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "candidate_id", "instrument", "local_weight_closure", "propagated_mode_checks",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
