#!/usr/bin/env python3
"""Fresh zero-fit confirmation of the compact three-MLP response program."""

# BQGATE: EXPERIMENT pred_a_authority_capability_replay_self_clamp_finiteness_price pred_b_selected_three_mlp_program_transfers pred_c_full_ten_site_pool_is_faithful pred_d_all_three_mlp_mode2_terms_are_necessary pred_e_selected_improves_direct_in_all_mode_cells
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_temporal_auxiliary_fresh_cues_v12 as temporal
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as iswas
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import attention_source_destination_eval as attention_eval
import run_temporal_iswas_canonical_downstream_response_removal_atlas_v1 as atlas
import run_temporal_iswas_downstream_ten_site_response_lattice_v1 as lattice
import run_temporal_iswas_upstream_full_response_mode_atlas_v1 as response

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_three_mlp_response_program_fresh_v12_v1.json"
SELECTION_RESULT = ROOT / "circuits/followups/temporal_iswas_downstream_ten_site_response_lattice_v1_result.json"
SELECTION_RUNNER = ROOT / "ops/run_temporal_iswas_downstream_ten_site_response_lattice_v1.py"
ATLAS_RUNNER = ROOT / "ops/run_temporal_iswas_canonical_downstream_response_removal_atlas_v1.py"
RESPONSE_RUNNER = ROOT / "ops/run_temporal_iswas_upstream_full_response_mode_atlas_v1.py"
WEIGHT_RESULT = ROOT / "circuits/followups/temporal_iswas_two_mode_weight_pullback_v3_result.json"
FRONTIER_RESULT = ROOT / "circuits/followups/temporal_iswas_v11_writer_frontier_holdout_v1_result.json"
FRONTIER_RUNNER = ROOT / "ops/run_temporal_iswas_v11_writer_frontier_holdout_v1.py"
TEMPORAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v12.py"
TEMPORAL_CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ISWAS_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
ISWAS_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_three_mlp_response_program_fresh_v12_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_iswas_three_mlp_response_program_fresh_v12_v1"
SELECTED = ("MLP13", "MLP15", "MLP16")
POOL = ("L13H6", "L15H1", "L15H5", "L17H2", "MLP12", "MLP13", "MLP14", "MLP15", "MLP16", "MLP17")
ARMS = {
    "direct": (),
    "selected": SELECTED,
    "full_ten_site_pool": POOL,
    "minus_MLP13": ("MLP15", "MLP16"),
    "minus_MLP15": ("MLP13", "MLP16"),
    "minus_MLP16": ("MLP13", "MLP15"),
}
MAX_FORWARDS, MAX_EVALUATIONS = 18, 972
EXPECTED = {
    "prior": "9927b0b93b3a7ef027f762ea3ee23cb8d8d6cf8eeb483813baf1502a9c73361b",
    "selection_result": "b39eaf86f1e6f8479acff263ae97dcb2b6ec62ee7975be599e6cefc86ac17f59",
    "selection_runner": "0a36e82329333f5aae5c9a6df43808d30c4ec90aec19a7f539d22fa0937a6ed9",
    "atlas_runner": "2e0309b9d39bbd5a61c2d9da428164667c038699f4c623100d0ef8214c5dc5d6",
    "response_runner": "8f1c2a1680163daaded60589540761f6b4903cb8160857185f0c313b146ce017",
    "weight_result": "c8ab608fa116342f9cbc8af4955e6087faa0f1eee9dd74dacb5c0ec168c5bf4d",
    "frontier_result": "9262c379be1a485b826a0f414c822e75832a1162a69881a4b40d2322e26ab07b",
    "frontier_runner": "08c9b1e85b9599c4cac2195b22ab2f04514c5bf4c861fff3192e3d1bf8e1431d",
    "temporal_builder": "4cf4624361ff2bd3c87cd987a7c4d16a1eeefb1c7f78287b78319862ab8d8de9",
    "temporal_capability": "4758b02cd026c85289dc3eaf352cc496d238057c6f8b52dfc6fe49ae17893324",
    "iswas_builder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2",
    "iswas_capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def capable_rows(builder, capability, panel, count=None):
    allowed = set(capability["jointly_capable_row_ids"][panel])
    rows = [row for row in builder.build_rows()
            if row.get("transform_id", row.get("family")) == panel and row["row_id"] in allowed]
    return rows if count is None else rows[:count]


def main():
    paths = {
        "prior": PRIOR,
        "selection_result": SELECTION_RESULT,
        "selection_runner": SELECTION_RUNNER,
        "atlas_runner": ATLAS_RUNNER,
        "response_runner": RESPONSE_RUNNER,
        "weight_result": WEIGHT_RESULT,
        "frontier_result": FRONTIER_RESULT,
        "frontier_runner": FRONTIER_RUNNER,
        "temporal_builder": TEMPORAL_BUILDER,
        "temporal_capability": TEMPORAL_CAPABILITY,
        "iswas_builder": ISWAS_BUILDER,
        "iswas_capability": ISWAS_CAPABILITY,
    }
    if {key: sha(value) for key, value in paths.items()} != EXPECTED:
        raise RuntimeError("fresh response-program authority changed")
    prior = json.loads(PRIOR.read_text())
    selection = json.loads(SELECTION_RESULT.read_text())
    weights = json.loads(WEIGHT_RESULT.read_text())
    frontier = json.loads(FRONTIER_RESULT.read_text())
    tcap = json.loads(TEMPORAL_CAPABILITY.read_text())
    icap = json.loads(ISWAS_CAPABILITY.read_text())
    temporal_rows = sum((capable_rows(temporal, tcap, panel, 12) for panel in ("A1", "A2")), [])
    iswas_rows = sum((capable_rows(iswas, icap, panel) for panel in ("A1", "A2")), [])
    rows = temporal_rows + iswas_rows
    dryrun = {
        "candidate_id": CANDIDATE_ID,
        "dryrun": True,
        "gpu_accessed": False,
        "model_loaded": False,
        "queue_touched": False,
        "rows": len(rows),
        "task_rows": {"temporal": len(temporal_rows), "iswas": len(iswas_rows)},
        "arms": {key: list(value) for key, value in ARMS.items()},
        "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS,
        "fit_updates": 0,
        "model_updates": 0,
        "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    authority_ok = (
        prior.get("candidate_id") == CANDIDATE_ID
        and selection.get("terminal") == "compact_response_program"
        and selection.get("selected_mask") == 416
        and tuple(selection["selected_metrics"]["sites"]) == SELECTED
        and frontier.get("terminal") == "conditional_writer_screen"
        and tcap.get("terminal") == "manifest"
        and icap.get("terminal") == "screen"
        and len(temporal_rows) == 24
        and len(iswas_rows) == 30
        and len({row["row_id"] for row in rows}) == 54
    )
    if len(rows) != 54 or set(POOL) - set(atlas.SITES):
        raise RuntimeError("frozen fresh population or pool changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    base_batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    base_output, _base_up = response.capture(backend, base_batch)
    donor_output, donor_up = response.capture(backend, donor_batch)
    base_down_output, base_down = atlas.capture_downstream(
        backend, lambda: backend.native(base_batch, capture=True))
    live_pair, live_down = atlas.capture_downstream(
        backend, lambda: response.run_patch(backend, base_batch, donor_up, atlas.WRITER_SITES))
    live_output = live_pair[0]
    self_output = lattice.run_mixed(
        backend, base_batch, base_down, lambda: backend.native(base_batch, capture=True))
    live_replay_pair = lattice.run_mixed(
        backend, base_batch, live_down,
        lambda: response.run_patch(backend, base_batch, donor_up, atlas.WRITER_SITES))
    live_replay = live_replay_pair[0]
    base_state, donor_state, live_state, self_state, replay_state = (
        response.states(torch, backend, output, rows)
        for output in (base_output, donor_output, live_output, self_output, live_replay)
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
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in atlas.LAYERS)
    raw_modes, orientation_error, _wrong = atlas.frontier.parent.overlap.residual_modes(
        backend, q, gain)
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

    base_margin = margin(base_state)
    live_margin = margin(live_state)
    full_behavior = live_margin - base_margin
    full_modes = (live_state - base_state) @ physical_reader
    task_indices = {
        "temporal": torch.arange(0, len(temporal_rows), device=backend.device),
        "iswas": torch.arange(len(temporal_rows), len(rows), device=backend.device),
    }

    def measure(state):
        behavior = margin(state) - base_margin
        modes = (state - base_state) @ physical_reader
        tasks = {}
        residuals = []
        for task, ids in task_indices.items():
            behavior_stats = response.vector_stats(torch, behavior[ids], full_behavior[ids])
            behavior_stats["squared_residual"] = float(
                (behavior[ids] - full_behavior[ids]).square().sum()
                / full_behavior[ids].square().sum())
            tasks[task] = {"behavior": behavior_stats}
            residuals.append(behavior_stats["squared_residual"])
            for mode in range(2):
                mode_stats = response.vector_stats(torch, modes[ids, mode], full_modes[ids, mode])
                mode_stats["squared_residual"] = float(
                    (modes[ids, mode] - full_modes[ids, mode]).square().sum()
                    / full_modes[ids, mode].square().sum())
                tasks[task][f"mode{mode + 1}"] = mode_stats
                residuals.append(mode_stats["squared_residual"])
        return {
            "tasks": tasks,
            "behavior_direction_fraction": float(((behavior / full_behavior) > 0).float().mean()),
            "worst_six_cell_residual": max(residuals),
        }

    arm_metrics = {}
    forwards, evaluations = 12, 648
    for name, live_sites in ARMS.items():
        values = {site: (live_down[site] if site in live_sites else base_down[site])
                  for site in atlas.SITES}
        pair = lattice.run_mixed(
            backend, base_batch, values,
            lambda: response.run_patch(backend, base_batch, donor_up, atlas.WRITER_SITES))
        state = response.states(torch, backend, pair[0], rows)
        arm_metrics[name] = {"sites": list(live_sites), **measure(state)}
        forwards += 1
        evaluations += len(rows)
    full_live_metrics = measure(replay_state)
    selected = arm_metrics["selected"]
    direct = arm_metrics["direct"]
    full_pool = arm_metrics["full_ten_site_pool"]
    pred_b = (
        selected["behavior_direction_fraction"] >= 0.90
        and selected["worst_six_cell_residual"] <= 0.20
        and all(selected["tasks"][task]["behavior"]["signed_projection"] >= 0.80
                and selected["tasks"][task]["mode1"]["signed_projection"] >= 0.60
                and selected["tasks"][task]["mode2"]["signed_projection"] >= 0.60
                for task in task_indices)
    )
    pred_c = (full_pool["behavior_direction_fraction"] >= 0.95
              and full_pool["worst_six_cell_residual"] <= 0.02)
    mode2_decrements = {
        site: {
            task: selected["tasks"][task]["mode2"]["signed_projection"]
            - arm_metrics[f"minus_{site}"]["tasks"][task]["mode2"]["signed_projection"]
            for task in task_indices
        }
        for site in SELECTED
    }
    pred_d = all(value >= 0.03 for by_task in mode2_decrements.values()
                 for value in by_task.values())
    mode_cells = [(task, mode) for task in task_indices for mode in ("mode1", "mode2")]
    pred_e = (
        all(selected["tasks"][task][mode]["squared_residual"]
            <= direct["tasks"][task][mode]["squared_residual"]
            for task, mode in mode_cells)
        and all(selected["tasks"][task]["mode2"]["squared_residual"]
                <= 0.75 * direct["tasks"][task]["mode2"]["squared_residual"]
                for task in task_indices)
    )
    finite = all(
        math.isfinite(value)
        for metrics in [*arm_metrics.values(), full_live_metrics]
        for task in metrics["tasks"].values()
        for family_stats in task.values()
        for value in family_stats.values()
    ) and all(math.isfinite(value) for by_task in mode2_decrements.values()
              for value in by_task.values())
    pred_a = (
        authority_ok
        and reader_hash_ok
        and orientation_error <= 1e-6
        and identity_error <= 1e-4
        and reconstruction <= 5e-4
        and full_live_metrics["worst_six_cell_residual"] <= 1e-10
        and finite
        and forwards == MAX_FORWARDS
        and evaluations == MAX_EVALUATIONS
    )
    predictions = {
        "pred_a_authority_capability_replay_self_clamp_finiteness_price": bool(pred_a),
        "pred_b_selected_three_mlp_program_transfers": bool(pred_b),
        "pred_c_full_ten_site_pool_is_faithful": bool(pred_c),
        "pred_d_all_three_mlp_mode2_terms_are_necessary": bool(pred_d),
        "pred_e_selected_improves_direct_in_all_mode_cells": bool(pred_e),
    }
    terminal = (
        "invalid" if not pred_a
        else "fresh_component_resolved_response_program" if all(predictions.values())
        else "response_program_transfer_null"
    )
    result = {
        "schema": "temporal_iswas_three_mlp_response_program_fresh_v12_result_v1",
        "candidate_id": CANDIDATE_ID,
        "execution_policy": "managed_queue_only",
        "started_utc": started_utc,
        "finished_utc": utc_now(),
        "authority_sha256": EXPECTED,
        "dryrun": dryrun,
        "population": {
            "temporal_row_ids": [row["row_id"] for row in temporal_rows],
            "iswas_row_ids": [row["row_id"] for row in iswas_rows],
        },
        "instrument": {
            "authority_and_capability_ok": authority_ok,
            "physical_reader_hash_ok": reader_hash_ok,
            "orientation_max_abs": orientation_error,
            "identity_self_and_live_replay_max_abs": identity_error,
            "attention_reconstruction_max_abs": reconstruction,
        },
        "arm_metrics": arm_metrics,
        "full_live_replay_metrics": full_live_metrics,
        "mode2_signed_projection_decrements": mode2_decrements,
        "predictions": predictions,
        "terminal": terminal,
        "price": {
            "model_forwards": forwards,
            "example_evaluations": evaluations,
            "fit_updates": 0,
            "model_updates": 0,
            "transformer_backwards": 0,
        },
        "serial_seconds": time.perf_counter() - started,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "candidate_id", "instrument", "mode2_signed_projection_decrements",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
