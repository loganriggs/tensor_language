#!/usr/bin/env python3
"""Greedy complete-MLP program for the distributed auxiliary c_v readers."""

# BQGATE: EXPERIMENT pred_a_authority_capture_monotonic_inventory_finiteness_and_price pred_b_greedy_program_materially_reduces_joint_reader_residual pred_c_best_prefix_is_jointly_sufficient_screen pred_d_discovery_to_evaluation_is_stable pred_e_distributed_program_uses_multiple_mlps
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_iswas_mlp8_auxiliary_cv_upstream_module_atlas_v2 as atlas
import run_iswas_mlp8_complement_attn11_attn15_conditional_head_atlas_v1 as auxiliary
import run_iswas_shared_q8_mlp8_postcue_weight_modes_v1 as weight
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1 as overlap


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/iswas_mlp8_auxiliary_cv_greedy_mlp_program_v1.json"
PARENT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_cv_upstream_module_atlas_v2_result.json"
PARENT_RUNNER = ROOT / "ops/run_iswas_mlp8_auxiliary_cv_upstream_module_atlas_v2.py"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
OUT = ROOT / "circuits/followups/iswas_mlp8_auxiliary_cv_greedy_mlp_program_v1_result.json"
CANDIDATE_ID = "cross_task.iswas_mlp8_auxiliary_cv_greedy_mlp_program_v1"
POOL = ("mlp:09", "mlp:10", "mlp:11", "mlp:12", "mlp:13", "mlp:14")
EXPECTED = {
    "prior": "5cb24f88420f30254d6543e0f6027adedf514c579257d77f9a0e40f2882e0163",
    "parent": "da3bd0473cf259537f5d70f1dff0ffe513bd5a7b420f6b6317259b522e4a3e67",
    "parent_runner": "ce3df585dea1cc20bacada055bf9926c0752839748075daec2f85759474ec63b",
    "capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
    "builder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2",
}
MAX_FORWARDS, MAX_EVALUATIONS = 25, 750


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def split_metric(writes, targets, mask):
    values = {}
    for layer in atlas.LAYERS:
        residual = writes[layer][mask]
        target = targets[layer][mask]
        values[str(layer)] = {
            "residual_norm_fraction": float(residual.norm() / target.norm()),
            "residual_cosine_to_parent": atlas.cosine(residual.reshape(-1), target.reshape(-1)),
        }
    values["objective"] = sum(row["residual_norm_fraction"] for row in values.values()) / len(atlas.LAYERS)
    return values


def main():
    paths = {"prior": PRIOR, "parent": PARENT, "parent_runner": PARENT_RUNNER,
             "capability": CAPABILITY, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("greedy auxiliary-MLP authority changed")
    prior, parent, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, PARENT, CAPABILITY, weight.SUBSPACE)]
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["family"] in ("A1", "A2") and row["row_id"] in allowed]
    positions = [weight.postcue_positions(row) for row in rows]
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen"
            or tuple(prior["authority"]["pool"]) != POOL or len(rows) != 30):
        raise RuntimeError("parent decision, pool, or population changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "pool": list(POOL),
        "greedy_trials_exact": 21, "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS, "fit_updates": 0,
        "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in range(12, 18))
    modes, orientation_error, _wrong = overlap.residual_modes(backend, family[8], gain)
    s = torch.linalg.qr(modes, mode="reduced").Q
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_hidden_output, base_hidden_capture = weight.capture_mlp8(backend, base_batch)
    _donor_output, donor_hidden_capture = weight.capture_mlp8(backend, donor_batch)
    down = backend.model.transformer.h[8].mlp.Down.weight.detach().float()
    _u, _singular, vh = torch.linalg.svd(s.T @ down, full_matrices=False)
    full_hidden_delta = donor_hidden_capture["hidden"].float() - base_hidden_capture["hidden"].float()
    complement = full_hidden_delta - weight.project(full_hidden_delta, vh, vh.shape[0])
    base_output, base_readers, base_raw9, base_modules = atlas.capture_readers_and_modules(
        backend, base_batch, capture_raw9=True, call=lambda: backend.native(base_batch, capture=True))
    live_output, live_readers, _raw, _modules = atlas.capture_readers_and_modules(
        backend, base_batch, call=lambda: auxiliary.run_heads(backend, base_batch,
            base_hidden_capture["hidden"], complement, positions, {9: base_raw9}, {9: atlas.CORE}))
    targets = atlas.reader_writes(backend, rows, base_readers, live_readers)
    discovery = torch.as_tensor([int(row["group_number"]) % 2 == 0 for row in rows], device=backend.device)
    evaluation = ~discovery
    selected, remaining, trials, prefixes, prefix_writes = [], list(POOL), [], [], {}
    previous_objective = 1.0
    reconstruction = max(value["reconstruction_max_abs"] for value in
                         (*base_readers.values(), *live_readers.values()))
    while remaining:
        candidates = []
        for site in remaining:
            sites = tuple(selected + [site])
            output, captures, _raw, _modules = atlas.run_variant(backend, base_batch,
                base_hidden_capture["hidden"], complement, positions, base_raw9,
                base_modules, sites, core_clamped=True)
            writes = atlas.reader_writes(backend, rows, base_readers, captures)
            metric = split_metric(writes, targets, discovery)
            candidates.append((metric["objective"], site, output, captures, writes, metric))
            trials.append({"prefix": list(selected), "added": site, "discovery": metric})
            reconstruction = max(reconstruction,
                *(value["reconstruction_max_abs"] for value in captures.values()))
        objective, chosen, output, captures, writes, metric = min(candidates, key=lambda row: (row[0], row[1]))
        selected.append(chosen); remaining.remove(chosen)
        prefixes.append({"sites": list(selected), "discovery": metric,
            "evaluation": split_metric(writes, targets, evaluation)})
        prefix_writes[tuple(selected)] = writes
        previous_objective = objective
    best_index = min(range(len(prefixes)), key=lambda i: (
        prefixes[i]["discovery"]["objective"], len(prefixes[i]["sites"]), prefixes[i]["sites"]))
    best = prefixes[best_index]
    monotone = all(prefixes[i]["discovery"]["objective"] <=
                   (1.0 if i == 0 else prefixes[i - 1]["discovery"]["objective"]) + 1e-7
                   for i in range(len(prefixes)))
    identity_error = float((atlas.converter.state(base_hidden_output, rows, torch, backend.device)
        - atlas.converter.state(base_output, rows, torch, backend.device)).abs().max())
    finite = all(math.isfinite(value) for record in prefixes
        for split in (record["discovery"], record["evaluation"])
        for row in split.values() if isinstance(row, dict) for value in row.values())
    pred_a = bool(orientation_error <= 1e-6 and reconstruction <= .001 and identity_error <= .001
        and len(trials) == 21 and monotone and finite and 25 <= MAX_FORWARDS
        and 25 * len(rows) <= MAX_EVALUATIONS)
    pred_b = best["discovery"]["objective"] <= .90 and monotone
    residuals = [best["evaluation"][str(layer)]["residual_norm_fraction"] for layer in atlas.LAYERS]
    pred_c = max(residuals) <= .90 and sum(residuals) / len(residuals) <= .75
    pred_d = best["evaluation"]["objective"] <= best["discovery"]["objective"] + .15
    pred_e = len(best["sites"]) >= 2
    predictions = {"pred_a_authority_capture_monotonic_inventory_finiteness_and_price": pred_a,
        "pred_b_greedy_program_materially_reduces_joint_reader_residual": pred_b,
        "pred_c_best_prefix_is_jointly_sufficient_screen": pred_c,
        "pred_d_discovery_to_evaluation_is_stable": pred_d,
        "pred_e_distributed_program_uses_multiple_mlps": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "iswas_mlp8_auxiliary_cv_greedy_mlp_program_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"f_linear_orientation_max_abs": orientation_error,
            "native_attention_reconstruction_max_abs": reconstruction,
            "native_base_identity_max_abs": identity_error, "rows": len(rows),
            "discovery_rows": int(discovery.sum()), "evaluation_rows": int(evaluation.sum())},
        "greedy_trials": trials, "prefixes": prefixes, "best_prefix": best,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": 25, "example_evaluations": 25 * len(rows),
            "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "prefixes",
        "best_prefix", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
