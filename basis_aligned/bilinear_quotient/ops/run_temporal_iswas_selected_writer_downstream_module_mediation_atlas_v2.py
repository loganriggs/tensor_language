#!/usr/bin/env python3
"""Screen complete downstream attention/MLP writes for the is-was writer bypass."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_writer_replay_module_capture_finiteness_and_exact_price pred_b_l11_attention_module_replays_the_known_small_aligned_branch pred_c_original_fit_nominates_a_non_l11_bypass_module pred_d_fit_selected_modules_validate_on_original_holdout pred_e_module_mediators_are_temporally_selective
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_l11h3_source_writer_formula_mediation_v2 as mediation


ROOT = Path(__file__).resolve().parents[1]
PRIOR_V1 = ROOT / "circuits/prior_art/temporal_iswas_selected_writer_downstream_module_mediation_atlas_v1.json"
PRIOR_V2 = ROOT / "circuits/prior_art/temporal_iswas_selected_writer_downstream_module_mediation_atlas_v2_price_amendment.json"
ROUTE_RESULT = ROOT / "circuits/followups/temporal_iswas_l11h3_writer_route_closure_ladder_v1_result.json"
GREEDY_RESULT = ROOT / "circuits/followups/temporal_iswas_l11h3_source_writer_weight_ordered_greedy_v1_result.json"
MEDIATION_RUNNER = ROOT / "ops/run_temporal_iswas_l11h3_source_writer_formula_mediation_v2.py"
OUT = ROOT / "circuits/followups/temporal_iswas_selected_writer_downstream_module_mediation_atlas_v2_result.json"
EXPECTED = {
    "prior_v1": "7a3b77c0a723bcbd4d3df838abfc55c74df5685ca89d69ec1c0343f668b7e908",
    "prior_v2": "7823ebe63d87680fbef12860bd8fe4a32839fdddcaf7c40623246ee186ad93e2",
    "route_result": "a5401d739775526fa1802c09d7cf633d6ae1253625a32671aabbb822a0754e6c",
    "greedy_result": "6a2ae6598ae95269f7d9e24e13196155342f690d331c78efed5115a8cf84bda2",
    "mediation_runner": "88a37dbd000505713c7504bb4c28a262df438349dcbefcbdc5b3ee29ed9aa1f8",
}
MODULES = tuple(name for layer in range(10, 18) for name in (f"A{layer}", f"M{layer}"))
PRICE = {"checkpoint_loads": 1, "model_forwards": 19,
         "sequence_evaluations": 2432, "scored_token_positions": 4864,
         "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_pairing_writer_replay_module_capture_finiteness_and_exact_price",
    "pred_b_l11_attention_module_replays_the_known_small_aligned_branch",
    "pred_c_original_fit_nominates_a_non_l11_bypass_module",
    "pred_d_fit_selected_modules_validate_on_original_holdout",
    "pred_e_module_mediators_are_temporally_selective",
)
BARS = {"replay": 1e-5, "l11_recovery": .1, "l11_cosine": .9,
        "fit_recovery": .2, "fit_cosine": .9, "fit_direction": .9,
        "holdout_recovery": .1, "holdout_cosine": .8, "holdout_direction": .75,
        "maximum_selected": 4, "collateral": .01}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_targets(model):
    targets = {}
    for layer in range(10, 18):
        targets[f"A{layer}"] = model.transformer.h[layer].attn.c_proj
        targets[f"M{layer}"] = model.transformer.h[layer].mlp
    return targets


def replace_positions(output, replacement, position_rows):
    changed = output.clone()
    for index, positions in enumerate(position_rows):
        changed[index, list(positions)] = replacement[index, list(positions)].to(
            device=output.device, dtype=output.dtype)
    return changed


def capture_modules(backend, tokens, *, head_captures=None, selected_labels=(),
                    pairs=None, position_rows=None):
    saved, calls, handles = {}, {name: 0 for name in MODULES}, []

    def hook(name):
        def capture(_module, _inputs, output):
            calls[name] += 1
            if not hasattr(output, "detach"):
                raise RuntimeError(f"{name} output is not a tensor")
            saved[name] = output.detach().clone()
        return capture

    for name, module in module_targets(backend.model).items():
        handles.append(module.register_forward_hook(hook(name)))
    try:
        logits, writer_captures, l11 = mediation.forward_with_l11_capture(
            backend, tokens, head_captures=head_captures,
            selected_labels=selected_labels, pairs=pairs, position_rows=position_rows)
    finally:
        for handle in handles:
            handle.remove()
    if set(calls.values()) != {1} or set(saved) != set(MODULES):
        raise RuntimeError("downstream module capture coverage changed")
    return logits, writer_captures, l11, saved, calls


def patch_module(backend, tokens, name, replacement, position_rows):
    calls = 0

    def hook(_module, _inputs, output):
        nonlocal calls
        calls += 1
        return replace_positions(output, replacement, position_rows)

    target = module_targets(backend.model)[name]
    handle = target.register_forward_hook(hook)
    try:
        logits, _captures = mediation.parent._forward(backend, tokens)
    finally:
        handle.remove()
    if calls != 1:
        raise RuntimeError(f"{name} patch call count changed: {calls}")
    return logits, calls


def finite_reports(reports):
    return all(np.isfinite(value) for row in reports for value in row.values()
               if isinstance(value, (int, float)) and not isinstance(value, bool))


def main():
    paths = {"prior_v1": PRIOR_V1, "prior_v2": PRIOR_V2,
             "route_result": ROUTE_RESULT, "greedy_result": GREEDY_RESULT,
             "mediation_runner": MEDIATION_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    prior_v1, prior_v2 = json.loads(PRIOR_V1.read_text()), json.loads(PRIOR_V2.read_text())
    dry = {"candidate_id": prior_v2["candidate_id"], "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "authority_ok": observed == EXPECTED, "modules": list(MODULES), "price": PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    route, greedy = json.loads(ROUTE_RESULT.read_text()), json.loads(GREEDY_RESULT.read_text())
    authority_ok = bool(observed == EXPECTED
        and route["terminal"] == "l11h3_bypass_dominant"
        and tuple(route["predictions"].values()) == (True, True, False, False, True)
        and prior_v1["locked_writer"]["heads"] == greedy["selected_prefixes"]["iswas"]["heads"]
        and prior_v2["amended_price"] == PRICE)
    if not authority_ok:
        raise RuntimeError("downstream module atlas authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    loc, parent, source = mediation.parent.loc, mediation.parent, mediation.source
    backend = loc.producer.Bilin18TorchBackend.load("cuda")
    rows = loc.original.build_rows()
    endpoints, lookup = loc.parent.endpoint_bank(rows)
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    torch = backend.torch
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints],
                          dtype=torch.long, device=backend.device)
    role, other = "iswas", "temporal"
    pairs = loc.parent.pair_indices(endpoints, lookup, role)
    queries = [endpoint[f"{role}_position"] - 1 for _row, _cell, endpoint in endpoints]
    regions = [loc.source_regions(endpoint["ids"], endpoints[pairs[index]][2]["ids"],
                                  queries[index])
               for index, (_row, _cell, endpoint) in enumerate(endpoints)]
    source_rows = [item[parent.LOCKED[role]] for item in regions]
    prefix_rows = [list(range(query + 1)) for query in queries]
    selected_heads = tuple(greedy["selected_prefixes"][role]["heads"])
    with torch.no_grad():
        native_logits, head_captures, native_l11, native_modules, native_calls = capture_modules(
            backend, tokens)
        writer_logits, _unused, _writer_l11, writer_modules, writer_calls = capture_modules(
            backend, tokens, head_captures=head_captures, selected_labels=selected_heads,
            pairs=pairs, position_rows=source_rows)
        reference_logits, _ = source._forward(backend, tokens, donor_v=native_l11["v"],
                                              pairs=pairs, value_positions=source_rows)
        module_logits, patch_calls = {}, {}
        for name in MODULES:
            module_logits[name], patch_calls[name] = patch_module(
                backend, tokens, name, writer_modules[name], prefix_rows)
    native = {name: loc.margins(native_logits, endpoints, name) for name in parent.LOCKED}
    writer_effect = loc.margins(writer_logits, endpoints, role) - native[role]
    writer_other = loc.margins(writer_logits, endpoints, other) - native[other]
    reference_effect = loc.margins(reference_logits, endpoints, role) - native[role]
    gold = native[role][pairs] - native[role]
    other_gold = native[other][loc.parent.pair_indices(endpoints, lookup, other)] - native[other]
    greedy_reports = greedy["panels"]["original"]["reports"]
    replay_errors, reports = [], []
    for phase in ("FIT", "HOLDOUT"):
        chosen = np.asarray([row["phase"] == phase for row, _cell, _endpoint in endpoints])
        replay = mediation.accounting.effect_metrics(
            writer_effect[chosen], reference_effect[chosen], writer_other[chosen])
        replay_gold = mediation.accounting.effect_metrics(
            writer_effect[chosen], gold[chosen], writer_other[chosen])
        old = next(row for row in greedy_reports if row["role"] == role
            and row["phase"] == phase and row["template_id"] == "ALL"
            and row["arm"] == greedy["selected_prefixes"][role]["arm"])
        for field in ("signed_recovery", "cosine", "relative_residual",
                      "direction_agreement", "non_target_to_target_gold_norm"):
            replay_errors.append(abs(replay[field] - old[field]))
        replay_errors.append(abs(replay_gold["non_target_to_target_gold_norm"]
                                 - old["command_gold_non_target_norm_ratio"]))
        for name in MODULES:
            effect = loc.margins(module_logits[name], endpoints, role) - native[role]
            effect_other = loc.margins(module_logits[name], endpoints, other) - native[other]
            metric = mediation.accounting.effect_metrics(
                effect[chosen], writer_effect[chosen], effect_other[chosen])
            temporal_gold_norm = float(np.linalg.norm(other_gold[chosen]))
            if temporal_gold_norm <= 0:
                raise RuntimeError("temporal command-gold effect is zero")
            reports.append({"module": name, "phase": phase, **metric,
                "temporal_command_gold_collateral":
                    float(np.linalg.norm(effect_other[chosen]) / temporal_gold_norm),
                "module_delta_norm": float((writer_modules[name].float()
                                            - native_modules[name].float()).norm())})
    fit = {row["module"]: row for row in reports if row["phase"] == "FIT"}
    holdout = {row["module"]: row for row in reports if row["phase"] == "HOLDOUT"}
    eligible = [name for name in MODULES if fit[name]["signed_recovery"] >= BARS["fit_recovery"]
        and fit[name]["cosine"] >= BARS["fit_cosine"]
        and fit[name]["direction_agreement"] >= BARS["fit_direction"]]
    selected = sorted(eligible, key=lambda name: (-fit[name]["signed_recovery"], name))[
        :BARS["maximum_selected"]]
    l11 = [fit["A11"], holdout["A11"]]
    A = bool(authority_ok and max(replay_errors) <= BARS["replay"]
        and set(native_calls.values()) == {1} and set(writer_calls.values()) == {1}
        and set(patch_calls.values()) == {1} and finite_reports(reports)
        and all(tuple(native_modules[name].shape) == tuple(writer_modules[name].shape)
                for name in MODULES) and PRICE == prior_v2["amended_price"])
    B = bool(all(row["signed_recovery"] >= BARS["l11_recovery"]
        and row["cosine"] >= BARS["l11_cosine"] for row in l11))
    C = any(name != "A11" for name in eligible)
    D = bool(selected and all(holdout[name]["signed_recovery"] >= BARS["holdout_recovery"]
        and holdout[name]["cosine"] >= BARS["holdout_cosine"]
        and holdout[name]["direction_agreement"] >= BARS["holdout_direction"]
        for name in selected))
    E = bool(selected and all(fit[name]["temporal_command_gold_collateral"] <= BARS["collateral"]
        and holdout[name]["temporal_command_gold_collateral"] <= BARS["collateral"]
        for name in selected))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = ("invalid" if not (A and B) else
                "downstream_module_bypass_candidates" if all((C, D, E)) else
                "distributed_or_unstable_bypass")
    result = {"schema": "temporal_iswas_selected_writer_downstream_module_mediation_atlas_result_v2",
        "candidate_id": prior_v2["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": {**observed}, "selected_writer_heads": list(selected_heads),
        "reports": reports, "eligible_fit_modules": eligible, "selected_modules": selected,
        "writer_replay_max_abs_error": max(replay_errors),
        "capture_shapes": {name: list(value.shape) for name, value in native_modules.items()},
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({"selected_modules": selected, "eligible_fit_modules": eligible,
        "writer_replay_max_abs_error": max(replay_errors),
        "fit": fit, "holdout": holdout, "predictions": predictions,
        "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
