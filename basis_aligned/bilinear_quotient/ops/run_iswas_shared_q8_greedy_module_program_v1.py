#!/usr/bin/env python3
"""Greedy complete-module program for the natural is/was shared-Q8 write."""

# BQGATE: EXPERIMENT pred_a_exact_authority_alignment_identity_search_coverage_and_price pred_b_selected_union_predicts_shared_q8_on_confirmation pred_c_selected_union_predicts_shared_behavior_on_confirmation pred_d_greedy_union_beats_singleton_and_unpruned_pool pred_e_program_is_small_and_attention_split_is_licensed_only_if_selected
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_candidate_aspectual_different_readout_is_was_v2 as v2
import circuit_candidate_aspectual_different_readout_is_was_v3 as v3
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_shared_q8_full_module_writer_atlas_v1 as module_atlas
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_shared_q8_greedy_module_program_v1.json"
ATLAS_RESULT = ROOT / "circuits/followups/iswas_shared_q8_full_module_writer_atlas_v1_result.json"
SHARED_CAUSAL = ROOT / "circuits/followups/temporal_q8_iswas_cdas_shared_specific_causal_v2_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
ISWAS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
ATLAS_RUNNER = ROOT / "ops/run_iswas_shared_q8_full_module_writer_atlas_v1.py"
OVERLAP_RUNNER = ROOT / "ops/run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1.py"
OUT = ROOT / "circuits/followups/iswas_shared_q8_greedy_module_program_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_shared_q8_greedy_module_program_v1"
EXPECTED = {
    "prior": "1d8c7a15fdf65ad506a99ab1ab9c26318614c3c81bf4846183ac12ed165f890d",
    "atlas_result": "bf055d107a97ff50f769700049a7f0ec8a886d644d1f3d635053b2a7de76a6dc",
    "shared_causal": "bd302cb0d104db5afe43906885dff52f851a03e638c6ff30de9d87224ce235bc",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "iswas": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "atlas_runner": "bcdd8ddd97431ff6d8d133b2ed408e2f1d6561bc28c4164a51d76496bcda151d",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1",
}
POOL = ("mlp:1", "attn:9", "mlp:0", "mlp:2", "mlp:3", "mlp:4", "mlp:5", "mlp:8")
MAX_FORWARDS, MAX_EVALUATIONS, SELECTION_EVALUATIONS = 34, 1104, 26


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def site_order(site):
    kind, layer = site.split(":")
    return int(layer), 0 if kind == "attn" else 1


def split_rows():
    discovery, confirmation, slices = [], [], {}
    for builder_name, builder in (("v2", v2), ("v3", v3)):
        rows = builder.build_rows()
        for family in ("A1", "A2"):
            selected = [row for row in rows if row["family"] == family]
            discovery.extend(selected[:4])
            start = len(confirmation); confirmation.extend(selected[4:])
            slices[f"{builder_name}_{family}"] = slice(start, len(confirmation))
    return discovery, confirmation, slices


def capture(backend, rows):
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_output, base_cache = module_atlas.capture_modules(backend, base_batch)
    donor_output, donor_cache = module_atlas.capture_modules(backend, donor_batch)
    torch = backend.torch
    base18 = torch.stack([torch.as_tensor(base_output.captured[(row["row_id"], "resid:18")])
                          for row in rows]).to(backend.device).float()
    donor18 = torch.stack([torch.as_tensor(donor_output.captured[(row["row_id"], "resid:18")])
                           for row in rows]).to(backend.device).float()
    return base_batch, base_output, donor_output, base_cache, donor_cache, base18, donor18


def target_write(backend, base18, donor18, axis, shared_axis):
    return ((donor18-base18)@axis)@shared_axis.T


def margins(backend, states, rows):
    torch = backend.torch
    logits = das.head_logits(backend, states)
    index = torch.arange(len(rows), device=backend.device)
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    return logits[index, answer]-logits[index, foil]


def score(backend, rows, base18, patched18, target, s):
    coordinates, target_coordinates = (patched18-base18)@s, target@s
    coordinate_loss = float(((coordinates-target_coordinates)**2).mean()
                            / target_coordinates.square().mean())
    base_margin = margins(backend, base18, rows)
    target_effect = margins(backend, base18+target, rows)-base_margin
    actual_effect = margins(backend, patched18, rows)-base_margin
    behavior_loss = float(((actual_effect-target_effect)**2).mean()/target_effect.square().mean())
    return .5*(coordinate_loss+behavior_loss), coordinate_loss, behavior_loss


def cosine(x, y):
    denominator = float(x.norm()*y.norm())
    return float((x*y).sum())/denominator if denominator else float("nan")


def main():
    paths = {"prior": PRIOR, "atlas_result": ATLAS_RESULT, "shared_causal": SHARED_CAUSAL,
        "subspace": SUBSPACE, "iswas": ISWAS, "atlas_runner": ATLAS_RUNNER,
        "overlap_runner": OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("greedy module program authority changed")
    prior, atlas_result, shared_causal, subspace, iswas = [json.loads(path.read_text())
        for path in (PRIOR, ATLAS_RESULT, SHARED_CAUSAL, SUBSPACE, ISWAS)]
    discovery, confirmation, family_slices = split_rows()
    if (prior.get("candidate_id") != CANDIDATE_ID or atlas_result.get("terminal") != "null"
            or shared_causal.get("terminal") != "screen" or iswas.get("terminal") != "screen"
            or tuple(atlas_result["rankings"]["by_causal_score"][:8]) != POOL
            or len(discovery) != 16 or len(confirmation) != 48
            or set(row["row_id"] for row in discovery)&set(row["row_id"] for row in confirmation)):
        raise RuntimeError("authority, pool, or row split changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "pool": list(POOL),
        "discovery_rows": 16, "confirmation_rows": 48, "selection_candidates": SELECTION_EVALUATIONS,
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    q = family[8]
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in range(12, 18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, q, gain)
    s = torch.linalg.qr(modes, mode="reduced").Q
    axis_values = np.asarray(iswas["basis"]["values_column_major"], dtype=np.float32)
    if hashlib.sha256(axis_values.tobytes()).hexdigest() != iswas["basis"]["sha256"]:
        raise RuntimeError("is/was axis changed")
    axis = torch.as_tensor(axis_values, device=backend.device).reshape(1152, 1)
    axis = axis/torch.linalg.vector_norm(axis); shared_axis = s@(s.T@axis)
    batch, _base_output, _donor_output, base_cache, donor_cache, base18, donor18 = capture(backend, discovery)
    target = target_write(backend, base18, donor18, axis, shared_axis)
    forwards, evaluations = 2, 2*len(discovery)
    self_output = module_atlas.run_patch(backend, batch, base_cache, POOL)
    self18 = torch.stack([torch.as_tensor(self_output.captured[(row["row_id"], "resid:18")])
                          for row in discovery]).to(backend.device).float()
    self_error = float((self18-base18).abs().max())
    forwards += 1; evaluations += len(discovery)
    selected, trace, prefix_outputs = [], [], []
    remaining = list(POOL)
    for step in range(4):
        candidates = []
        for site in remaining:
            sites = tuple(sorted(selected+[site], key=site_order))
            output = module_atlas.run_patch(backend, batch, donor_cache, sites)
            patched18 = torch.stack([torch.as_tensor(output.captured[(row["row_id"], "resid:18")])
                                     for row in discovery]).to(backend.device).float()
            objective, coordinate_loss, behavior_loss = score(
                backend, discovery, base18, patched18, target, s)
            candidates.append({"site": site, "sites": list(sites), "objective": objective,
                "coordinate_loss": coordinate_loss, "behavior_loss": behavior_loss,
                "patched18": patched18})
            forwards += 1; evaluations += len(discovery)
        winner = min(candidates, key=lambda row: (row["objective"], POOL.index(row["site"])))
        selected.append(winner["site"]); remaining.remove(winner["site"])
        prefix_outputs.append(winner["patched18"])
        trace.append({key: value for key, value in winner.items() if key != "patched18"})
    best_prefix_index = min(range(4), key=lambda index: (trace[index]["objective"], index))
    selected_union = tuple(sorted(selected[:best_prefix_index+1], key=site_order))
    best_singleton = (trace[0]["site"],)
    batch_c, _base_c, _donor_c, _base_cache_c, donor_cache_c, base18_c, donor18_c = capture(
        backend, confirmation)
    target_c = target_write(backend, base18_c, donor18_c, axis, shared_axis)
    forwards += 2; evaluations += 2*len(confirmation)
    arm_sites = {"best_singleton": best_singleton, "selected_union": selected_union,
                 "all_pool": tuple(sorted(POOL, key=site_order))}
    arm_states = {"base": base18_c, "target_shared_write": base18_c+target_c}
    for arm, sites in arm_sites.items():
        output = module_atlas.run_patch(backend, batch_c, donor_cache_c, sites)
        arm_states[arm] = torch.stack([torch.as_tensor(output.captured[(row["row_id"], "resid:18")])
                                       for row in confirmation]).to(backend.device).float()
        forwards += 1; evaluations += len(confirmation)
    target_coordinates = target_c@s
    target_effect = margins(backend, arm_states["target_shared_write"], confirmation)-margins(
        backend, base18_c, confirmation)
    reports = {}
    for arm in ("best_singleton", "selected_union", "all_pool"):
        coordinates = (arm_states[arm]-base18_c)@s
        effect = margins(backend, arm_states[arm], confirmation)-margins(backend, base18_c, confirmation)
        coordinate_rmse = float(torch.sqrt(((coordinates-target_coordinates)**2).mean()))
        coordinate_rms = float(torch.sqrt(target_coordinates.square().mean()))
        behavior_rmse = float(torch.sqrt(((effect-target_effect)**2).mean()))
        behavior_rms = float(torch.sqrt(target_effect.square().mean()))
        reports[arm] = {"coordinate_cosine": cosine(coordinates.reshape(-1), target_coordinates.reshape(-1)),
            "coordinate_relative_rmse": coordinate_rmse/coordinate_rms,
            "behavior_cosine": cosine(effect, target_effect),
            "behavior_relative_rmse": behavior_rmse/behavior_rms,
            "mean_effect_fraction": float(effect.mean()/target_effect.mean()),
            "family_mean_effect": {name: float(effect[span].mean()) for name, span in family_slices.items()}}
    search_count = sum(8-step for step in range(4))
    pred_a = bool(orientation_error <= 1e-6 and self_error <= 1e-4
        and search_count == SELECTION_EVALUATIONS and forwards <= MAX_FORWARDS
        and evaluations <= MAX_EVALUATIONS and all(math.isfinite(item["objective"]) for item in trace))
    selected_report = reports["selected_union"]
    pred_b = bool(selected_report["coordinate_cosine"] >= .80
        and selected_report["coordinate_relative_rmse"] <= .75)
    pred_c = bool(selected_report["behavior_cosine"] >= .90
        and selected_report["behavior_relative_rmse"] <= .50
        and all(value > 0 for value in selected_report["family_mean_effect"].values()))
    pred_d = bool(selected_report["coordinate_relative_rmse"]
        <= .80*reports["best_singleton"]["coordinate_relative_rmse"]
        and selected_report["coordinate_relative_rmse"] <= reports["all_pool"]["coordinate_relative_rmse"])
    selected_attention = [site for site in selected_union if site.startswith("attn")]
    pred_e = len(selected_union) <= 4
    predictions = {"pred_a_exact_authority_alignment_identity_search_coverage_and_price": pred_a,
        "pred_b_selected_union_predicts_shared_q8_on_confirmation": pred_b,
        "pred_c_selected_union_predicts_shared_behavior_on_confirmation": pred_c,
        "pred_d_greedy_union_beats_singleton_and_unpruned_pool": pred_d,
        "pred_e_program_is_small_and_attention_split_is_licensed_only_if_selected": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "iswas_shared_q8_greedy_module_program_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "joint_pool_base_self_patch_resid18_max_abs": self_error,
            "discovery_confirmation_overlap": 0, "selection_evaluations": search_count},
        "greedy_trace": trace, "selected_union": list(selected_union),
        "selected_attention_modules": selected_attention, "confirmation_reports": reports,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "selection_candidates": search_count, "fit_updates": 0, "transformer_backwards": 0,
            "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "greedy_trace",
        "selected_union", "selected_attention_modules", "confirmation_reports", "predictions",
        "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
