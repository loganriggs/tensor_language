#!/usr/bin/env python3
"""All-layer complete-module causal-response atlas on is/was-v15."""
# BQGATE: EXPERIMENT pred_a_authority_population_closure_finiteness_and_exact_price pred_b_old_five_behavior_replays pred_c_new_module_outside_old_five_is_material pred_d_proper_prefix_restores_behavior_and_state pred_e_final_state_enriches_behavioral_localization
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v15_all_layer_complete_module_response_atlas_v1.json"
SOURCE_ATLAS = ROOT / "circuits/followups/temporal_iswas_v15_complete_source_mlp_write_atlas_v1_result.json"
SOURCE_RUNNER = ROOT / "ops/run_temporal_iswas_v15_complete_source_mlp_write_atlas_v1.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v15_capability_v2_audit_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v15.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v15_all_layer_complete_module_response_atlas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.iswas_v15_all_layer_complete_module_response_atlas_v1"
EXPECTED = {
    "prior": "702fc38ec90f7e7001aed992ecc4f334a59194d69f4dc836c6556a98ae1eaa22",
    "source_atlas": "69c3a1d0a2946c45ac3be05bb6ffca8a397177eb5803679eae5828bd8aee7c38",
    "source_runner": "0657bd48be18acb17b55069fcec8914743301f2b24134009a94004711f807b0f",
    "capability": "fbe395731de4fa7850999864f67bd45ec709e5da3e3e237e9dcae352c1211c2f",
    "builder": "e5774cd5e93c564ad3bdb3169c482c2e1ddb295b85cc7c9885a7b077493a8343",
}
INPUT = "input_residual"
MODULE_SITES = tuple(f"{kind}:{layer}" for layer in range(18) for kind in ("attn", "mlp"))
ALL_SITES = (INPUT,) + MODULE_SITES
OLD_FIVE = tuple(f"mlp:{layer}" for layer in (0, 1, 2, 3, 6))
EXACT_FORWARDS = 60


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def site_module(backend, site: str):
    kind, layer_text = site.split(":")
    block = backend.model.transformer.h[int(layer_text)]
    return block.attn.c_proj if kind == "attn" else block.mlp


def capture(backend, batch):
    cache, handles = {}, []

    def capture_input(_module, arguments):
        cache[INPUT] = arguments[0].detach().clone()

    handles.append(backend.model.transformer.h[0].register_forward_pre_hook(capture_input))
    for site in MODULE_SITES:
        kind = site.split(":")[0]
        if kind == "attn":
            def save_attention(_module, arguments, site=site):
                cache[site] = arguments[0].detach().clone()
            handles.append(site_module(backend, site).register_forward_pre_hook(save_attention))
        else:
            def save_mlp(_module, _arguments, output, site=site):
                cache[site] = output.detach().clone()
            handles.append(site_module(backend, site).register_forward_hook(save_mlp))
    try:
        output = backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(cache) != set(ALL_SITES):
        raise RuntimeError(f"incomplete native response capture: {sorted(set(ALL_SITES) - set(cache))}")
    return output, cache


def patch_hook(batch, values, site: str):
    if site == INPUT or site.startswith("attn:"):
        def hook(_module, arguments):
            changed = arguments[0].clone()
            for index, query in enumerate(batch.semantic_positions):
                changed[index, : int(query) + 1] = values[index, : int(query) + 1].to(changed)
            return (changed,) + tuple(arguments[1:])
    else:
        def hook(_module, _arguments, output):
            changed = output.clone()
            for index, query in enumerate(batch.semantic_positions):
                changed[index, : int(query) + 1] = values[index, : int(query) + 1].to(changed)
            return changed
    return hook


def run_patch(backend, batch, cache, sites):
    handles = []
    for site in sites:
        hook = patch_hook(batch, cache[site], site)
        if site == INPUT:
            handles.append(backend.model.transformer.h[0].register_forward_pre_hook(hook))
        elif site.startswith("attn:"):
            handles.append(site_module(backend, site).register_forward_pre_hook(hook))
        else:
            handles.append(site_module(backend, site).register_forward_hook(hook))
    try:
        return backend.native(batch, capture=True)
    finally:
        for handle in handles:
            handle.remove()


def states(torch, backend, output, rows):
    return torch.stack([
        torch.as_tensor(output.captured[(row["row_id"], "resid:18")]) for row in rows
    ]).to(backend.device).float()


def vector_metrics(torch, value, target):
    x, y = value.reshape(-1).double(), target.reshape(-1).double()
    yy, xx = float(y @ y), float(x @ x)
    xy = float(x @ y)
    return {
        "signed_projection": xy / yy if yy else 0.0,
        "relative_squared_error": float(((x - y) @ (x - y))) / yy if yy else 0.0,
        "cosine": xy / math.sqrt(xx * yy) if xx and yy else 0.0,
        "norm_ratio": math.sqrt(xx / yy) if yy else 0.0,
    }


def ranks(values):
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def spearman(left, right):
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def main():
    paths = {
        "prior": PRIOR, "source_atlas": SOURCE_ATLAS, "source_runner": SOURCE_RUNNER,
        "capability": CAPABILITY, "builder": BUILDER,
    }
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != EXPECTED:
        raise RuntimeError(f"all-layer module atlas authority changed: {observed}")
    prior, source, capability = (json.loads(path.read_text()) for path in (PRIOR, SOURCE_ATLAS, CAPABILITY))
    rows = [row for row in fresh.build_rows() if row["transform_id"] in {"A1", "A2"}]
    panel_counts = {panel: sum(row["transform_id"] == panel for row in rows) for panel in ("A1", "A2")}
    if (prior.get("candidate_id") != CANDIDATE_ID
            or source.get("terminal") != "source_graph_construction_failure"
            or capability.get("terminal") != "manifest"
            or panel_counts != {"A1": 16, "A2": 16}
            or len({row["row_id"] for row in rows}) != 32):
        raise RuntimeError("population or licensing terminal changed")
    dryrun = {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows),
        "module_sites": list(MODULE_SITES), "input_diagnostic": INPUT,
        "singleton_arms": len(ALL_SITES), "prefix_arms": 18,
        "model_forwards_exact": EXACT_FORWARDS, "fit_updates": 0,
        "model_updates": 0, "transformer_backwards": 0,
    }
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_output, base_cache = capture(backend, base_batch)
    donor_output, donor_cache = capture(backend, donor_batch)
    forwards = 2
    aligned = all(base_cache[site].shape == donor_cache[site].shape for site in ALL_SITES)
    if not aligned:
        raise RuntimeError("base/donor native response shapes changed")
    base_state, donor_state = states(torch, backend, base_output, rows), states(torch, backend, donor_output, rows)
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    index = torch.arange(len(rows), device=backend.device)

    def margins(state):
        logits = das.head_logits(backend, state)
        return logits[index, answer] - logits[index, foil]

    base_margin, donor_margin = margins(base_state), margins(donor_state)
    target_margin, target_state = donor_margin - base_margin, donor_state - base_state

    def report(output):
        state = states(torch, backend, output, rows)
        delta_margin, delta_state = margins(state) - base_margin, state - base_state
        behavior = vector_metrics(torch, delta_margin, target_margin)
        behavior["direction_fraction"] = float(((delta_margin * target_margin) > 0).float().mean())
        return {"behavior": behavior, "final_residual": vector_metrics(torch, delta_state, target_state)}

    self_output = run_patch(backend, base_batch, base_cache, ALL_SITES)
    donor_all_output = run_patch(backend, base_batch, donor_cache, ALL_SITES)
    old_five_output = run_patch(backend, base_batch, donor_cache, OLD_FIVE)
    forwards += 3
    self_state = states(torch, backend, self_output, rows)
    donor_all_state = states(torch, backend, donor_all_output, rows)
    self_error = max(float((self_state - base_state).abs().max()), float((margins(self_state) - base_margin).abs().max()))
    donor_error = max(float((donor_all_state - donor_state).abs().max()), float((margins(donor_all_state) - donor_margin).abs().max()))
    old_five = report(old_five_output)

    singletons = {}
    for site in ALL_SITES:
        singletons[site] = report(run_patch(backend, base_batch, donor_cache, (site,)))
        forwards += 1

    prefixes = {}
    for layer in range(18):
        sites = (INPUT,) + tuple(
            site for site in MODULE_SITES if int(site.split(":")[1]) <= layer
        )
        prefixes[str(layer)] = report(run_patch(backend, base_batch, donor_cache, sites))
        forwards += 1

    behavior_values = [abs(singletons[site]["behavior"]["signed_projection"]) for site in MODULE_SITES]
    residual_values = [abs(singletons[site]["final_residual"]["signed_projection"]) for site in MODULE_SITES]
    response_behavior_spearman = spearman(behavior_values, residual_values)
    top_module = max(MODULE_SITES, key=lambda site: behavior_values[MODULE_SITES.index(site)])
    top_new = max(
        (site for site in MODULE_SITES if site not in OLD_FIVE),
        key=lambda site: abs(singletons[site]["behavior"]["signed_projection"]),
    )
    proper_prefixes = [
        layer for layer in range(17)
        if prefixes[str(layer)]["behavior"]["signed_projection"] >= .75
        and prefixes[str(layer)]["behavior"]["direction_fraction"] >= .90
        and prefixes[str(layer)]["final_residual"]["relative_squared_error"] <= .20
    ]
    earliest_prefix = min(proper_prefixes) if proper_prefixes else None
    old_replay_error = abs(
        old_five["behavior"]["signed_projection"]
        - source["reports"]["whole_all_gain100"]["behavior_signed_projection"]["iswas"]
    )
    pred_a = bool(
        aligned and self_error <= 1e-4 and donor_error <= 1e-4 and forwards == EXACT_FORWARDS
        and finite([old_five, singletons, prefixes, response_behavior_spearman])
    )
    pred_b = old_replay_error <= 1e-5
    pred_c = bool(
        abs(singletons[top_new]["behavior"]["signed_projection"]) >= .10
        and singletons[top_new]["behavior"]["direction_fraction"] >= .75
    )
    pred_d = earliest_prefix is not None
    pred_e = response_behavior_spearman >= .40
    predictions = {
        "pred_a_authority_population_closure_finiteness_and_exact_price": pred_a,
        "pred_b_old_five_behavior_replays": pred_b,
        "pred_c_new_module_outside_old_five_is_material": pred_c,
        "pred_d_proper_prefix_restores_behavior_and_state": pred_d,
        "pred_e_final_state_enriches_behavioral_localization": pred_e,
    }
    top_value = abs(singletons[top_module]["behavior"]["signed_projection"])
    if not pred_a:
        terminal = "invalid"
    elif not pred_b:
        terminal = "old_five_replay_mismatch"
    elif pred_c and top_value >= .50:
        terminal = "localized_new_module"
    elif not pred_d:
        terminal = "proper_prefix_interaction_failure"
    elif not pred_c:
        terminal = "new_module_singleton_null"
    elif not pred_e:
        terminal = "response_block_mismatch"
    else:
        terminal = "distributed_changed_graph"
    result = {
        "schema": "temporal_iswas_v15_all_layer_complete_module_response_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "population": {"rows": len(rows), "panel_counts": panel_counts,
            "row_ids": [row["row_id"] for row in rows]},
        "instrument": {"aligned_shapes": aligned, "base_self_max_abs_error": self_error,
            "all_donor_max_abs_error": donor_error, "old_five_behavior_replay_max_abs_error": old_replay_error},
        "old_five_report": old_five, "singleton_reports": singletons,
        "prefix_reports": prefixes,
        "summary": {"top_module": top_module, "top_module_behavior_abs": top_value,
            "top_new_module": top_new,
            "top_new_module_behavior_abs": abs(singletons[top_new]["behavior"]["signed_projection"]),
            "earliest_eligible_proper_prefix_layer": earliest_prefix,
            "final_residual_behavior_spearman": response_behavior_spearman,
            "module_ranking_by_behavior_abs": sorted(MODULE_SITES,
                key=lambda site: abs(singletons[site]["behavior"]["signed_projection"]), reverse=True)},
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards_observed": forwards, "model_forwards_exact": EXACT_FORWARDS,
            "example_evaluations": forwards * len(rows), "fit_updates": 0,
            "model_updates": 0, "transformer_backwards": 0},
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "candidate_id", "instrument", "summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
