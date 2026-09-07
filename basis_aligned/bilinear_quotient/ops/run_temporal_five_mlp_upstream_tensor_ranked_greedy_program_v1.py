#!/usr/bin/env python3
"""Greedy causal composition of the tensor-ranked upstream writer pool."""

# BQGATE: EXPERIMENT pred_a_authority_alignment_self_patch_finiteness_and_price pred_b_top20_pool_is_target_sufficient pred_c_eight_site_greedy_program_is_selective pred_d_selected_sites_are_jointly_necessary pred_e_shared_writer_structure_survives_composition
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
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1.json"
PROMOTED = ROOT / "circuits/followups/temporal_five_mlp_upstream_input_tensor_incidence_atlas_v2_tolerance_audit_result.json"
ATLAS = ROOT / "circuits/followups/temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1_result.json"
ATLAS_RUNNER = ROOT / "ops/run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1.py"
CONFIRMATION = ROOT / "circuits/followups/temporal_v13_iswas_v12_four_five_mlp_program_v1_result.json"
TEMPORAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v13.py"
TEMPORAL_CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
ISWAS_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
ISWAS_CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
WEIGHT_RESULT = ROOT / "circuits/followups/temporal_iswas_two_mode_weight_pullback_v3_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.five_mlp_upstream_tensor_ranked_greedy_program_v1"
KNOWN = ("L9H1", "L9H4", "MLP7", "MLP9")
MAX_STEPS = 8
MAX_TARGET_FORWARDS = 150
MAX_CONTROL_FORWARDS = 4
MAX_EVALUATIONS = 8420
EXPECTED = {
    "prior": "3a3cdd8c89328f6059d08d7593ac49a913c1ea119f0f36d273580f376f75e817",
    "promoted": "9b76cb14b1793c83f8dc71ad45dc9da72cb16ce36d8462bd6789d7070e6ed52b",
    "atlas": "0cc9909dcab7a17b93820300da56a07f4cd9a2610f71a1de1c7008710d064467",
    "atlas_runner": "6e28d38ec1446eafb3518c1bfe603a5e3469ceadb6f80266e2c695a274692366",
    "confirmation": "9fc9c6aad9257ba1d75745bc1938ec2451120c1363a66ec933cae222d43c515d",
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


def physical_reader(backend, weights):
    torch = backend.torch
    family, _singular, _energy = atlasrun.literal.atlas.frontier.parent.family_builder.build_family(
        backend, json.loads(atlasrun.literal.atlas.frontier.parent.SUBSPACE.read_text()))
    q = family[8]
    full_gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                          for layer in atlasrun.literal.atlas.LAYERS)
    raw_modes, orientation_error, _wrong = (
        atlasrun.literal.atlas.frontier.parent.overlap.residual_modes(backend, q, full_gain))
    state_basis = torch.linalg.qr(raw_modes, mode="reduced").Q
    coordinates = torch.as_tensor(
        weights["mode_artifacts"]["reader_coordinates"], device=backend.device).float()
    reader = state_basis @ coordinates
    hash_ok = tensor_sha(reader) == weights["mode_artifacts"][
        "physical_reader_covectors_sha256"]
    return reader, orientation_error, hash_ok


def rows_and_controls(tcap, icap):
    temporal_rows = sum((population.capable_rows(temporal, tcap, panel, 12)
                         for panel in ("A1", "A2")), [])
    iswas_rows = sum((population.capable_rows(iswas, icap, panel)
                      for panel in ("A1", "A2")), [])
    temporal_all, iswas_all = temporal.build_rows(), iswas.build_rows()
    temporal_control = [row for row in temporal_all if row["transform_id"] == "C"][:8]
    iswas_control = [row for row in iswas_all if row["transform_id"] == "C"][:8]
    return temporal_rows, iswas_rows, temporal_control, iswas_control


def make_target_evaluator(backend, rows, base_batch, donor_cache, base_state,
                          full_behavior, full_modes, task_indices, counter):
    torch = backend.torch
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    index = torch.arange(len(rows), device=backend.device)
    def margin(state):
        logits = das.head_logits(backend, state)
        return logits[index, answer] - logits[index, foil]
    base_margin = margin(base_state)

    def evaluate(sites):
        output, _io = atlasrun.run_patch(backend, base_batch, donor_cache, sites)
        state = atlasrun.states(torch, backend, output, rows)
        behavior = margin(state) - base_margin
        modes = (state - base_state) @ counter["reader"]
        cells = {}
        for task, ids in task_indices.items():
            behavior_denominator = full_behavior[ids].square().sum().clamp_min(1e-30)
            cells[f"{task}_behavior"] = float(
                (behavior[ids] - full_behavior[ids]).square().sum() / behavior_denominator)
            for mode in range(2):
                denominator = full_modes[ids, mode].square().sum().clamp_min(1e-30)
                cells[f"{task}_mode{mode + 1}"] = float(
                    (modes[ids, mode] - full_modes[ids, mode]).square().sum() / denominator)
        counter["target_forwards"] += 1
        counter["evaluations"] += len(rows)
        return {"sites": list(sites), "worst_residual": max(cells.values()),
                "mean_residual": sum(cells.values()) / len(cells), "cells": cells}
    return evaluate


def main():
    paths = {
        "prior": PRIOR, "promoted": PROMOTED, "atlas": ATLAS,
        "atlas_runner": ATLAS_RUNNER, "confirmation": CONFIRMATION,
        "temporal_builder": TEMPORAL_BUILDER, "temporal_capability": TEMPORAL_CAPABILITY,
        "iswas_builder": ISWAS_BUILDER, "iswas_capability": ISWAS_CAPABILITY,
        "weight_result": WEIGHT_RESULT,
    }
    if {key: sha(path) for key, path in paths.items()} != EXPECTED:
        raise RuntimeError("tensor-ranked greedy authority changed")
    prior, promoted, atlas, confirmation, tcap, icap, weights = [
        json.loads(path.read_text()) for path in (
            PRIOR, PROMOTED, ATLAS, CONFIRMATION, TEMPORAL_CAPABILITY,
            ISWAS_CAPABILITY, WEIGHT_RESULT)]
    pool = tuple(atlas["tensor_ranking"][:20])
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "pool": list(pool),
        "pool_size": len(pool), "max_greedy_steps": MAX_STEPS,
        "target_rows": 54, "control_rows": 16,
        "target_model_forwards_max": MAX_TARGET_FORWARDS,
        "control_model_forwards_max": MAX_CONTROL_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS,
        "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    temporal_rows, iswas_rows, temporal_control, iswas_control = rows_and_controls(tcap, icap)
    rows = temporal_rows + iswas_rows
    controls = temporal_control + iswas_control
    authority_ok = bool(
        prior.get("candidate_id") == CANDIDATE_ID
        and promoted.get("terminal") == "shared_upstream_writer_screen"
        and atlas.get("terminal") == "invalid"
        and confirmation.get("terminal") == "temporal_fresh_five_mlp_response_program"
        and len(pool) == 20 and len(set(pool)) == 20
        and len(temporal_rows) == 24 and len(iswas_rows) == 30
        and len(temporal_control) == len(iswas_control) == 8
        and all(row["base_answer_id"] == row["donor_answer_id"] for row in controls))
    if not authority_ok:
        raise RuntimeError("tensor-ranked greedy population or decision changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    reader, orientation_error, reader_hash_ok = physical_reader(backend, weights)
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_output, base_cache = atlasrun.capture_native(backend, base_batch)
    donor_output, donor_cache = atlasrun.capture_native(backend, donor_batch)
    base_state = atlasrun.states(torch, backend, base_output, rows)
    donor_state = atlasrun.states(torch, backend, donor_output, rows)
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    index = torch.arange(len(rows), device=backend.device)
    def margin(state):
        logits = das.head_logits(backend, state)
        return logits[index, answer] - logits[index, foil]
    full_behavior = margin(donor_state) - margin(base_state)
    full_modes = (donor_state - base_state) @ reader
    task_indices = {
        "temporal": torch.arange(0, len(temporal_rows), device=backend.device),
        "iswas": torch.arange(len(temporal_rows), len(rows), device=backend.device),
    }
    self_output, _self_io = atlasrun.run_patch(backend, base_batch, base_cache, pool)
    self_state = atlasrun.states(torch, backend, self_output, rows)
    self_error = float((self_state - base_state).abs().max())
    counter = {"target_forwards": 3, "control_forwards": 0,
               "evaluations": 3 * len(rows), "reader": reader}
    evaluate = make_target_evaluator(
        backend, rows, base_batch, donor_cache, base_state,
        full_behavior, full_modes, task_indices, counter)
    memo = {}
    def cached(sites):
        key = tuple(sorted(sites, key=pool.index))
        if key not in memo:
            memo[key] = evaluate(key)
        return memo[key]
    full_pool = cached(pool)
    selected = []
    greedy_trace = []
    current = {"worst_residual": 1.0, "mean_residual": 1.0}
    for step in range(1, MAX_STEPS + 1):
        candidates = [cached(tuple(selected) + (site,)) for site in pool if site not in selected]
        best = min(candidates, key=lambda report: (
            report["worst_residual"], report["mean_residual"],
            pool.index(next(site for site in report["sites"] if site not in selected))))
        if (best["worst_residual"], best["mean_residual"]) >= (
                current["worst_residual"], current["mean_residual"]):
            break
        added = next(site for site in best["sites"] if site not in selected)
        selected.append(added)
        current = best
        greedy_trace.append({"step": step, "added": added,
                             "report": best, "candidate_count": len(candidates)})
    singleton_reports = {site: cached((site,)) for site in pool}
    best_singleton_site = min(pool, key=lambda site: (
        singleton_reports[site]["worst_residual"],
        singleton_reports[site]["mean_residual"], pool.index(site)))
    selected_report = cached(tuple(selected)) if selected else current
    leave_one_out = {
        site: cached(tuple(item for item in selected if item != site)) for site in selected
    }
    nonnecessary = [
        site for site, report in leave_one_out.items()
        if report["mean_residual"] < selected_report["mean_residual"] + 0.01
    ]
    # Keep the outcome-frozen greedy set.  Removing every first-pass LOO failure
    # simultaneously would be an adaptive second selector and, at eight sites,
    # could exceed the preregistered 150-forward ceiling when re-audited.
    pruned = list(selected)
    pruned_report = selected_report
    pruned_leave_one_out = leave_one_out

    control_batch_base, control_batch_donor = (
        das._batch(backend, controls, side="base"), das._batch(backend, controls, side="donor"))
    control_base_output, control_base_cache = atlasrun.capture_native(backend, control_batch_base)
    _control_donor_output, control_donor_cache = atlasrun.capture_native(backend, control_batch_donor)
    control_selected_output, _control_io = atlasrun.run_patch(
        backend, control_batch_base, control_donor_cache, pruned)
    control_pool_output, _control_pool_io = atlasrun.run_patch(
        backend, control_batch_base, control_donor_cache, pool)
    counter["control_forwards"] += 4
    counter["evaluations"] += 4 * len(controls)
    control_base_state = atlasrun.states(torch, backend, control_base_output, controls)
    control_selected_state = atlasrun.states(torch, backend, control_selected_output, controls)
    control_pool_state = atlasrun.states(torch, backend, control_pool_output, controls)
    control_answer = torch.as_tensor(
        [row["donor_answer_id"] for row in controls], device=backend.device)
    control_foil = torch.as_tensor(
        [row["donor_foil_id"] for row in controls], device=backend.device)
    control_index = torch.arange(len(controls), device=backend.device)
    def control_margin(state):
        logits = das.head_logits(backend, state)
        return logits[control_index, control_answer] - logits[control_index, control_foil]
    base_control_margin = control_margin(control_base_state)
    target_scales = {
        "temporal": float(full_behavior[task_indices["temporal"]].square().mean().sqrt()),
        "iswas": float(full_behavior[task_indices["iswas"]].square().mean().sqrt()),
    }
    control_indices = {
        "temporal": torch.arange(0, len(temporal_control), device=backend.device),
        "iswas": torch.arange(len(temporal_control), len(controls), device=backend.device),
    }
    control_reports = {}
    for name, state in (("selected", control_selected_state), ("full_pool", control_pool_state)):
        delta = control_margin(state) - base_control_margin
        control_reports[name] = {
            task: {
                "rms_margin_effect": float(delta[ids].square().mean().sqrt()),
                "fraction_of_target_scale": float(delta[ids].square().mean().sqrt())
                / max(target_scales[task], 1e-30),
            }
            for task, ids in control_indices.items()
        }

    finite = all(math.isfinite(value) for report in memo.values()
                 for value in report["cells"].values())
    pred_a = bool(
        authority_ok and reader_hash_ok and orientation_error <= 1e-6
        and self_error <= 1e-4 and finite
        and counter["target_forwards"] <= MAX_TARGET_FORWARDS
        and counter["control_forwards"] <= MAX_CONTROL_FORWARDS
        and counter["evaluations"] <= MAX_EVALUATIONS)
    pred_b = full_pool["worst_residual"] <= 0.10
    pred_c = bool(
        len(pruned) <= 8 and pruned_report["worst_residual"] <= 0.15
        and all(control_reports["selected"][task]["fraction_of_target_scale"] <= 0.10
                for task in control_indices))
    pred_d = bool(pruned and all(
        report["mean_residual"] >= pruned_report["mean_residual"] + 0.01
        for report in pruned_leave_one_out.values()))
    best_singleton = singleton_reports[best_singleton_site]
    pred_e = bool(
        "MLP1" in pruned and any(site in pruned for site in KNOWN)
        and best_singleton["worst_residual"] - pruned_report["worst_residual"] >= 0.20)
    predictions = {
        "pred_a_authority_alignment_self_patch_finiteness_and_price": pred_a,
        "pred_b_top20_pool_is_target_sufficient": bool(pred_b),
        "pred_c_eight_site_greedy_program_is_selective": pred_c,
        "pred_d_selected_sites_are_jointly_necessary": pred_d,
        "pred_e_shared_writer_structure_survives_composition": pred_e,
    }
    terminal = (
        "invalid" if not pred_a
        else "selective_upstream_tensor_program" if all(predictions.values())
        else "generic_transport_null" if pred_b and not pred_c
        else "distributed_program_null" if pred_b and pruned_report["worst_residual"] > 0.15
        else "upstream_composition_null" if not pred_b
        else "partial")
    result = {
        "schema": "temporal_five_mlp_upstream_tensor_ranked_greedy_program_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"authority_ok": authority_ok, "physical_reader_hash_ok": reader_hash_ok,
                       "orientation_max_abs": orientation_error,
                       "self_patch_max_abs": self_error},
        "pool": list(pool), "full_pool_report": full_pool,
        "greedy_trace": greedy_trace, "selected": selected,
        "selected_report": selected_report, "nonnecessary_by_first_loo": nonnecessary,
        "pruned": pruned, "pruned_report": pruned_report,
        "leave_one_out": leave_one_out, "pruned_leave_one_out": pruned_leave_one_out,
        "best_singleton": {"site": best_singleton_site, "report": best_singleton},
        "control_reports": control_reports, "target_behavior_rms_scales": target_scales,
        "predictions": predictions, "terminal": terminal,
        "price": {"target_forwards": counter["target_forwards"],
                  "control_forwards": counter["control_forwards"],
                  "evaluations": counter["evaluations"],
                  "candidate_unions_evaluated": len(memo),
                  "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "candidate_id", "instrument", "full_pool_report", "greedy_trace",
        "selected", "pruned", "pruned_report", "best_singleton",
        "control_reports", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
