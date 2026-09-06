#!/usr/bin/env python3
"""Complete native-module writer atlas for the cross-task temporal Q8 state."""

# BQGATE: EXPERIMENT pred_a_exact_authority_alignment_identity_coverage_and_price pred_b_at_least_one_complete_module_writes_shared_q8 pred_c_weight_incidence_predicts_causal_writer_location pred_d_module_localization_is_not_uniform
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
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_shared_q8_full_module_writer_atlas_v1.json"
HANKEL = ROOT / "circuits/followups/temporal_iswas_q8_finite_causal_hankel_v1_result.json"
SHARED_CAUSAL = ROOT / "circuits/followups/temporal_q8_iswas_cdas_shared_specific_causal_v2_result.json"
ISWAS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
V2_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v2.py"
V3_BUILDER = ROOT / "ops/circuit_candidate_aspectual_different_readout_is_was_v3.py"
OVERLAP_RUNNER = ROOT / "ops/run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1.py"
OUT = ROOT / "circuits/followups/iswas_shared_q8_full_module_writer_atlas_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_shared_q8_full_module_writer_atlas_v1"
EXPECTED = {
    "prior": "6d92e5b179b3d3af47659c2db5ed0f95c611ac209b03e0167d853fc5d80c5ab9",
    "hankel": "f8fa10c21c30cd3420648641b4a284ba3cb41152872db8cf77d25213c597bb62",
    "shared_causal": "bd302cb0d104db5afe43906885dff52f851a03e638c6ff30de9d87224ce235bc",
    "iswas": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "v2_builder": "4ed62d06e2ffe5c471efc70eeaa35c7524a66923f19c64c803faef7200d4a62f",
    "v3_builder": "ac240f13478168948814f59c0425270dd345ba03053332b72b34857e9a022638",
    "overlap_runner": "15f2e1119c4df7c29aaf51c615ec92d87555d65efce029f16a89478212f29af1",
}
SITES = tuple(f"{kind}:{layer}" for layer in range(18) for kind in ("attn", "mlp"))
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 40, 640, 576


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def select_rows():
    groups = []
    for builder in (v2, v3):
        rows = builder.build_rows()
        for family in ("A1", "A2"):
            groups.extend([row for row in rows if row["family"] == family][:4])
    return groups


def module(backend, site):
    kind, layer_text = site.split(":")
    block = backend.model.transformer.h[int(layer_text)]
    return block.attn.c_proj if kind == "attn" else block.mlp


def capture_modules(backend, batch):
    cache, handles = {}, []
    for site in SITES:
        if site.startswith("attn"):
            def capture(_module, arguments, site=site):
                cache[site] = arguments[0].detach().clone()
            handles.append(module(backend, site).register_forward_pre_hook(capture))
        else:
            def capture(_module, _arguments, output, site=site):
                cache[site] = output.detach().clone()
            handles.append(module(backend, site).register_forward_hook(capture))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(cache) != set(SITES):
        raise RuntimeError("incomplete module capture")
    return output, cache


def patch_hook(batch, values, kind):
    if kind == "attn":
        def patch(_module, arguments):
            changed = arguments[0].clone()
            for index, query in enumerate(batch.semantic_positions):
                changed[index, :int(query)+1] = values[index, :int(query)+1].to(changed)
            return (changed,) + tuple(arguments[1:])
    else:
        def patch(_module, _arguments, output):
            changed = output.clone()
            for index, query in enumerate(batch.semantic_positions):
                changed[index, :int(query)+1] = values[index, :int(query)+1].to(changed)
            return changed
    return patch


def run_patch(backend, batch, caches, sites):
    handles = []
    for site in sites:
        kind = site.split(":")[0]
        hook = patch_hook(batch, caches[site], kind)
        handles.append(module(backend, site).register_forward_pre_hook(hook) if kind == "attn"
                       else module(backend, site).register_forward_hook(hook))
    try:
        return backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()


def rank_values(values):
    order = np.argsort(np.argsort(np.asarray(values, dtype=np.float64)))
    return order.astype(np.float64)


def main():
    paths = {"prior": PRIOR, "hankel": HANKEL, "shared_causal": SHARED_CAUSAL,
        "iswas": ISWAS, "subspace": SUBSPACE, "v2_builder": V2_BUILDER,
        "v3_builder": V3_BUILDER, "overlap_runner": OVERLAP_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("full-module writer atlas authority changed")
    prior, hankel, shared_causal, iswas, subspace = [json.loads(path.read_text())
        for path in (PRIOR, HANKEL, SHARED_CAUSAL, ISWAS, SUBSPACE)]
    rows = select_rows()
    aligned = all(len(row["base_ids"]) == len(row["donor_ids"]) for row in rows)
    if (prior.get("candidate_id") != CANDIDATE_ID or hankel.get("terminal") != "screen"
            or shared_causal.get("terminal") != "screen" or iswas.get("terminal") != "screen"
            or len(rows) != 16 or not aligned):
        raise RuntimeError("authority terminals or alignment changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "sites": list(SITES),
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "records": RECORDS, "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}
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
    axis = axis/torch.linalg.vector_norm(axis)
    shared_axis = s@(s.T@axis)
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_output, base_cache = capture_modules(backend, base_batch)
    donor_output, donor_cache = capture_modules(backend, donor_batch)
    forwards, evaluations = 2, 2*len(rows)
    shape_ok = all(base_cache[site].shape == donor_cache[site].shape for site in SITES)
    base18 = torch.stack([torch.as_tensor(base_output.captured[(row["row_id"], "resid:18")])
                          for row in rows]).to(backend.device).float()
    donor18 = torch.stack([torch.as_tensor(donor_output.captured[(row["row_id"], "resid:18")])
                           for row in rows]).to(backend.device).float()
    target = ((donor18-base18)@axis)@shared_axis.T
    target_coordinates = target@s
    self_output = run_patch(backend, base_batch, base_cache, SITES)
    forwards += 1; evaluations += len(rows)
    self18 = torch.stack([torch.as_tensor(self_output.captured[(row["row_id"], "resid:18")])
                          for row in rows]).to(backend.device).float()
    self_state_error = float((self18-base18).abs().max())
    answer_ids = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil_ids = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    index = torch.arange(len(rows), device=backend.device)
    def margins(states):
        logits = das.head_logits(backend, states)
        return logits[index, answer_ids]-logits[index, foil_ids]
    base_margin, donor_margin = margins(base18), margins(donor18)
    denominator = donor_margin-base_margin
    weight_incidence, site_metrics, records = {}, {}, []
    for site in SITES:
        kind, layer_text = site.split(":")
        layer = int(layer_text)
        weight = (backend.model.transformer.h[layer].attn.c_proj.weight.detach().float()
                  if kind == "attn" else backend.model.transformer.h[layer].mlp.Down.weight.detach().float())
        weight_incidence[site] = float(torch.linalg.matrix_norm(s.T@weight)
                                       / torch.linalg.matrix_norm(weight))
        output = run_patch(backend, base_batch, donor_cache, (site,))
        forwards += 1; evaluations += len(rows)
        patched18 = torch.stack([torch.as_tensor(output.captured[(row["row_id"], "resid:18")])
                                 for row in rows]).to(backend.device).float()
        coordinates = (patched18-base18)@s
        cosines, ratios = [], []
        for row_index in range(len(rows)):
            denom = float(torch.linalg.vector_norm(target_coordinates[row_index]))
            value_norm = float(torch.linalg.vector_norm(coordinates[row_index]))
            cosine_denom = denom*value_norm
            cosines.append(float(coordinates[row_index]@target_coordinates[row_index])/cosine_denom
                           if cosine_denom else 0.0)
            ratios.append(value_norm/denom if denom else 0.0)
        recovery = (margins(patched18)-base_margin)/denominator
        site_metrics[site] = {"mean_coordinate_cosine": float(np.mean(cosines)),
            "mean_target_norm_ratio": float(np.mean(ratios)),
            "mean_recovery": float(recovery.mean()),
            "mean_absolute_recovery": float(recovery.abs().mean()),
            "direction_fraction": float((recovery > 0).float().mean()),
            "weight_incidence": weight_incidence[site]}
        records.extend({"row_id": row["row_id"], "site": site,
            "coordinate_cosine": cosines[row_index], "target_norm_ratio": ratios[row_index],
            "recovery": float(recovery[row_index])} for row_index, row in enumerate(rows))
    incidence_values = [weight_incidence[site] for site in SITES]
    norm_values = [site_metrics[site]["mean_target_norm_ratio"] for site in SITES]
    incidence_ranks, norm_ranks = rank_values(incidence_values), rank_values(norm_values)
    spearman = float(np.corrcoef(incidence_ranks, norm_ranks)[0, 1])
    causal_scores = {site: site_metrics[site]["mean_target_norm_ratio"]
        * max(site_metrics[site]["mean_coordinate_cosine"], 0.0) for site in SITES}
    top_causal = max(SITES, key=lambda site: causal_scores[site])
    incidence_order = sorted(SITES, key=lambda site: weight_incidence[site], reverse=True)
    top_incidence_quartile = incidence_order[:9]
    top_norm = max(norm_values); median_norm = float(np.median(norm_values))
    passing_sites = [site for site in SITES if site_metrics[site]["mean_coordinate_cosine"] >= .50
        and site_metrics[site]["mean_target_norm_ratio"] >= .25
        and site_metrics[site]["mean_absolute_recovery"] >= .10]
    pred_a = bool(shape_ok and orientation_error <= 1e-6 and self_state_error <= 1e-4
        and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS and len(records) == RECORDS
        and all(math.isfinite(value) for metrics in site_metrics.values() for value in metrics.values()))
    pred_b = bool(passing_sites)
    pred_c = bool(top_causal in top_incidence_quartile and spearman >= .25)
    pred_d = top_norm >= 2*median_norm
    predictions = {"pred_a_exact_authority_alignment_identity_coverage_and_price": pred_a,
        "pred_b_at_least_one_complete_module_writes_shared_q8": pred_b,
        "pred_c_weight_incidence_predicts_causal_writer_location": pred_c,
        "pred_d_module_localization_is_not_uniform": pred_d}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    rankings = {"by_causal_score": sorted(SITES, key=lambda site: causal_scores[site], reverse=True),
        "by_shared_norm": sorted(SITES, key=lambda site: norm_values[SITES.index(site)], reverse=True),
        "by_weight_incidence": incidence_order}
    result = {"schema": "iswas_shared_q8_full_module_writer_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"aligned_module_shapes": shape_ok,
            "f_linear_orientation_max_abs": orientation_error,
            "joint_base_self_patch_resid18_max_abs": self_state_error},
        "site_metrics": site_metrics, "rankings": rankings,
        "summary": {"passing_sites": passing_sites, "top_causal_site": top_causal,
            "top_weight_incidence_quartile": top_incidence_quartile,
            "weight_incidence_causal_norm_spearman": spearman,
            "top_to_median_shared_norm_ratio": top_norm/median_norm},
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "records": len(records), "fit_updates": 0, "transformer_backwards": 0,
            "model_updates": 0}, "records": records}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "rankings",
        "summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
