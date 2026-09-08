#!/usr/bin/env python3
"""Causally test exact-weight upstream heads at locked L11 value sources."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_reference_replay_head_capture_coverage_finiteness_and_exact_price pred_b_at_least_one_weight_shared_head_is_a_cross_task_causal_writer pred_c_each_task_top5_union_recovers_the_l11_value_source_effect pred_d_upstream_head_effects_are_distributive pred_e_upstream_writing_is_task_selective_and_causal
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import dual_command_head_module_factorial_contract as accounting
import run_temporal_iswas_l11h3_value_source_region_localization_v1 as loc


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_l11h3_source_tensor_upstream_head_factorial_v1.json"
ATLAS = ROOT / "circuits/followups/temporal_iswas_l11h3_source_tensor_upstream_weight_atlas_v1_result.json"
EXTRACTOR = ROOT / "circuits/followups/temporal_iswas_l11h3_native_routing_source_term_extraction_v1_result.json"
LOCALIZATION = ROOT / "circuits/followups/temporal_iswas_l11h3_value_source_region_localization_v1_result.json"
ORIGINAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_v2.py"
OOD_BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_ood_v1.py"
ACCOUNTING = ROOT / "ops/dual_command_head_module_factorial_contract.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_l11h3_source_tensor_upstream_head_factorial_v1_result.json"
EXPECTED = {
    "prior": "1c028d2b5ad321b17bef9600d21b86de166f660cc40a16277e774df8867c96ae",
    "atlas": "f1b1a057c0ad48f3f514970430551b51d42cfa75a549e6491ea2d724664740d6",
    "extractor": "d55e448e5c950f9a6e316cc43e8d1ca7c95fd625b31021b2572a8e20ba4f245a",
    "localization": "74d7e51e227c9799bb6e079edff1fb8b5c49e9e6725960958e0269e19d74882e",
    "original_builder": "72da11860ce5bf1c03ea126bc10fcb6c46dd2648169edb00a1e9e1dbeee8a0d0",
    "ood_builder": "6062f2b8dfaa54da71477b43cb9fc63650db389c5883c9e64125f0a200afeb5a",
    "accounting": "985b824c7d3d8b622dfb92e3522f107e15e5e095b1f40a59a46122ad9af68496",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
LOCKED = {"temporal": "bridge", "iswas": "postcue"}
CANDIDATES = ((3, 4), (5, 1), (6, 7), (7, 7), (9, 1), (9, 4))
LABELS = tuple(f"L{layer:02d}H{head:02d}" for layer, head in CANDIDATES)
SHARED = ("L06H07", "L07H07", "L09H01", "L09H04")
TOP5 = {
    "temporal": ("L07H07", "L09H04", "L09H01", "L06H07", "L05H01"),
    "iswas": ("L07H07", "L09H04", "L06H07", "L09H01", "L03H04"),
}
ARMS = LABELS + ("top5_union",)
PRICE = {"checkpoint_loads": 1, "model_forwards": 34, "sequence_evaluations": 4352,
         "scored_token_positions": 8704, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_pairing_reference_replay_head_capture_coverage_finiteness_and_exact_price",
    "pred_b_at_least_one_weight_shared_head_is_a_cross_task_causal_writer",
    "pred_c_each_task_top5_union_recovers_the_l11_value_source_effect",
    "pred_d_upstream_head_effects_are_distributive",
    "pred_e_upstream_writing_is_task_selective_and_causal",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def patch_heads(output, captures, selected_labels, pairs, position_rows):
    if output.ndim != 3:
        raise ValueError("upstream head output shape changed")
    width = output.shape[-1] // 9
    changed = output.clone()
    for label in selected_labels:
        layer = int(label[1:3])
        head = int(label[4:6])
        donor = captures[label]
        if donor.shape[:2] != output.shape[:2] or donor.shape[-1] != width:
            raise ValueError("upstream donor capture shape changed")
        sl = slice(head * width, (head + 1) * width)
        for index, (pair, positions) in enumerate(zip(pairs, position_rows)):
            for position in positions:
                changed[index, position, sl] = donor[pair, position]
    return changed


def _forward(backend, tokens, *, capture=False, captures=None, selected_labels=(),
             pairs=None, position_rows=None, donor_v=None):
    torch, F, model = backend.torch, backend.F, backend.model
    saved, handles, calls = {}, [], {layer: 0 for layer in {item[0] for item in CANDIDATES}}
    v_calls = 0
    by_layer = {}
    for label in selected_labels:
        by_layer.setdefault(int(label[1:3]), []).append(label)

    def make_pre(layer):
        def hook(_module, arguments):
            calls[layer] += 1
            output = arguments[0]
            width = output.shape[-1] // 9
            if capture:
                for candidate_layer, head in CANDIDATES:
                    if candidate_layer == layer:
                        label = f"L{layer:02d}H{head:02d}"
                        saved[label] = output[..., head * width:(head + 1) * width].detach().clone()
            if layer in by_layer:
                output = patch_heads(output, captures, by_layer[layer], pairs, position_rows)
                return (output,) + tuple(arguments[1:])
            return None
        return hook

    def value_hook(_module, _inputs, output):
        nonlocal v_calls
        v_calls += 1
        if capture:
            saved["L11H3:v"] = output.detach().clone()
        if donor_v is not None:
            return loc.patch_value_tensor(output, donor_v, pairs, position_rows)
        return None

    for layer in calls:
        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(make_pre(layer)))
    handles.append(model.transformer.h[11].attn.c_v.register_forward_hook(value_hook))
    try:
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
        x0, first = x, None
        for block in model.transformer.h:
            x, first = block(x, first, x0)
        logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)).float()
    finally:
        for handle in handles:
            handle.remove()
    if set(calls.values()) != {1} or v_calls != 1:
        raise RuntimeError("upstream head hook coverage changed")
    if capture and set(saved) != {*LABELS, "L11H3:v"}:
        raise RuntimeError("upstream capture inventory changed")
    return logits, saved


def vector_composition(parts, union):
    summed = sum(parts)
    denominator = max(float(np.linalg.norm(union)), 1e-30)
    summed_norm = max(float(np.linalg.norm(summed)), 1e-30)
    return {"relative_l2_error": float(np.linalg.norm(union - summed) / denominator),
            "cosine": float(np.dot(union, summed) / (denominator * summed_norm))}


def run_population(backend, authority, population, localization):
    rows = authority.build_rows()
    endpoints, lookup = loc.parent.endpoint_bank(rows)
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    torch = backend.torch
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints],
                          dtype=torch.long, device=backend.device)
    pairs = {role: loc.parent.pair_indices(endpoints, lookup, role) for role in LOCKED}
    queries = {role: [endpoint[f"{role}_position"] - 1
                      for _row, _cell, endpoint in endpoints] for role in LOCKED}
    regions = {role: [loc.source_regions(endpoint["ids"],
                endpoints[pairs[role][index]][2]["ids"], queries[role][index])
                for index, (_row, _cell, endpoint) in enumerate(endpoints)] for role in LOCKED}
    with torch.no_grad():
        native_logits, captures = _forward(backend, tokens, capture=True)
    native = {role: loc.margins(native_logits, endpoints, role) for role in LOCKED}
    effects, collateral, references, reference_collateral = {}, {}, {}, {}
    for role, source in LOCKED.items():
        source_rows = [item[source] for item in regions[role]]
        other = "iswas" if role == "temporal" else "temporal"
        with torch.no_grad():
            logits, _ = _forward(backend, tokens, donor_v=captures["L11H3:v"],
                                 pairs=pairs[role], position_rows=source_rows)
        references[role] = loc.margins(logits, endpoints, role) - native[role]
        reference_collateral[role] = loc.margins(logits, endpoints, other) - native[other]
        for arm in ARMS:
            selected = TOP5[role] if arm == "top5_union" else (arm,)
            with torch.no_grad():
                logits, _ = _forward(backend, tokens, captures=captures,
                    selected_labels=selected, pairs=pairs[role], position_rows=source_rows)
            effects[(role, arm)] = loc.margins(logits, endpoints, role) - native[role]
            collateral[(role, arm)] = loc.margins(logits, endpoints, other) - native[other]
    reports, compositions, replay_errors = [], [], []
    for phase in ("FIT", "HOLDOUT"):
        for template in ("ALL",) + authority.TEMPLATES:
            selected = np.asarray([row["phase"] == phase and
                (template == "ALL" or row["template_id"] == template)
                for row, _cell, _endpoint in endpoints])
            for role, source in LOCKED.items():
                gold = native[role][pairs[role]] - native[role]
                reference_gold = accounting.effect_metrics(references[role][selected], gold[selected],
                    reference_collateral[role][selected])
                old = next(item for item in localization["panels"][population]["reports"]
                    if item["role"] == role and item["phase"] == phase
                    and item["template_id"] == template and item["arm"] == source)
                replay_errors.extend((abs(reference_gold["signed_recovery"]
                                          - old["command_gold_signed_recovery"]),
                                      abs(reference_gold["non_target_to_target_gold_norm"]
                                          - old["command_gold_non_target_norm_ratio"])))
                reports.append({"population": population, "role": role, "phase": phase,
                    "template_id": template, "arm": "L11H3:v_reference", **reference_gold})
                for arm in ARMS:
                    report = accounting.effect_metrics(effects[(role, arm)][selected],
                                                        references[role][selected],
                                                        collateral[(role, arm)][selected])
                    gold_report = accounting.effect_metrics(effects[(role, arm)][selected],
                        gold[selected], collateral[(role, arm)][selected])
                    reports.append({"population": population, "role": role, "phase": phase,
                        "template_id": template, "arm": arm, **report,
                        "command_gold_non_target_norm_ratio":
                            gold_report["non_target_to_target_gold_norm"]})
                compositions.append({"population": population, "role": role, "phase": phase,
                    "template_id": template, **vector_composition([
                        effects[(role, arm)][selected] for arm in TOP5[role]],
                        effects[(role, "top5_union")][selected])})
    causal_zero = max(abs(value) for arm in ARMS for value in collateral[("iswas", arm)])
    return {"reports": reports, "compositions": compositions,
        "reference_replay_max_abs_error": max(replay_errors), "causal_zero": float(causal_zero),
        "capture_shapes": {name: list(value.shape) for name, value in captures.items()},
        "coverage_ok": all(set(item[LOCKED[role]]) <= set(item["full_prefix"])
                           for role in LOCKED for item in regions[role])}


def main():
    paths = {"prior": PRIOR, "atlas": ATLAS, "extractor": EXTRACTOR,
             "localization": LOCALIZATION, "original_builder": ORIGINAL_BUILDER,
             "ood_builder": OOD_BUILDER, "accounting": ACCOUNTING, "producer": PRODUCER}
    observed = {name: sha(path) for name, path in paths.items()}
    atlas, extractor, localization = (json.loads(path.read_text())
        for path in (ATLAS, EXTRACTOR, LOCALIZATION))
    authority_ok = bool(observed == EXPECTED
        and atlas.get("terminal") == "shared_upstream_weight_candidates"
        and extractor.get("terminal") == "native_routing_source_term_extracted"
        and atlas.get("frozen_top5") == {role: [f"{label}:attn_out" for label in labels]
                                        for role, labels in TOP5.items()})
    dry = {"candidate_id": json.loads(PRIOR.read_text())["candidate_id"], "dryrun": True,
           "authority_ok": authority_ok, "gpu_accessed": False, "model_loaded": False,
           "queue_touched": False, "candidate_heads": LABELS, "arms": ARMS, "price": PRICE}
    if not authority_ok:
        raise RuntimeError(f"upstream head authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = loc.producer.Bilin18TorchBackend.load("cuda")
    panels = {"original": run_population(backend, loc.original, "original", localization),
              "ood": run_population(backend, loc.ood, "ood", localization)}
    reports = [row for panel in panels.values() for row in panel["reports"]]
    compositions = [row for panel in panels.values() for row in panel["compositions"]]
    def pooled(population, role, phase, arm):
        return next(row for row in reports if row["population"] == population
            and row["role"] == role and row["phase"] == phase
            and row["template_id"] == "ALL" and row["arm"] == arm)
    shared_passing = []
    for arm in SHARED:
        original_ok = all(pooled("original", role, phase, arm)["signed_recovery"] >= .10
            and pooled("original", role, phase, arm)["cosine"] >= .60
            and pooled("original", role, phase, arm)["direction_agreement"] >= .65
            for role in LOCKED for phase in ("FIT", "HOLDOUT"))
        ood_ok = all(pooled("ood", role, phase, arm)["signed_recovery"] >= 0
            and pooled("ood", role, phase, arm)["direction_agreement"] >= .55
            for role in LOCKED for phase in ("FIT", "HOLDOUT"))
        if original_ok and ood_ok:
            shared_passing.append(arm)
    A = bool(authority_ok and max(panel["reference_replay_max_abs_error"]
        for panel in panels.values()) <= 1e-5 and all(panel["coverage_ok"] for panel in panels.values())
        and all(set(panel["capture_shapes"]) == {*LABELS, "L11H3:v"}
                for panel in panels.values()) and finite(panels)
        and PRICE == json.loads(PRIOR.read_text())["frozen_design"]["price"])
    B = bool(shared_passing)
    C = bool(all(pooled(population, role, phase, "top5_union")["signed_recovery"]
        >= (.50 if population == "original" else .35)
        and pooled(population, role, phase, "top5_union")["cosine"] >= .80
        and pooled(population, role, phase, "top5_union")["direction_agreement"] >= .75
        for population in panels for role in LOCKED for phase in ("FIT", "HOLDOUT")))
    D = bool(all(row["relative_l2_error"] <= .25 and row["cosine"] >= .95
                 for row in compositions))
    E = bool(all(panel["causal_zero"] <= 1e-5 for panel in panels.values()) and all(
        pooled(population, role, phase, "L11H3:v_reference")[
            "non_target_to_target_gold_norm"] <= .01
        and pooled(population, role, phase, "top5_union")[
            "command_gold_non_target_norm_ratio"] <= .01
        for population in panels for role in LOCKED for phase in ("FIT", "HOLDOUT")))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    if not A:
        terminal = "invalid"
    elif B and E:
        terminal = "shared_upstream_head_writer"
    elif C and E:
        terminal = "distributed_upstream_head_union"
    else:
        terminal = "upstream_weight_candidate_causal_null"
    diagnostics = {"L05H01_original_fit_recovery": {role:
        pooled("original", role, "FIT", "L05H01")["signed_recovery"] for role in LOCKED},
        "L03H04_original_fit_recovery": {role:
        pooled("original", role, "FIT", "L03H04")["signed_recovery"] for role in LOCKED}}
    result = {"schema": "temporal_iswas_l11h3_source_tensor_upstream_head_factorial_result_v1",
        "candidate_id": json.loads(PRIOR.read_text())["candidate_id"],
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "shared_singletons_passing": shared_passing, "diagnostics": diagnostics,
        "panels": panels, "predictions": predictions, "terminal": terminal, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({"reference_replay_max_abs_error": max(panel[
        "reference_replay_max_abs_error"] for panel in panels.values()),
        "shared_singletons_passing": shared_passing,
        "top5_unions": {population: {role: {phase: pooled(population, role, phase,
            "top5_union") for phase in ("FIT", "HOLDOUT")} for role in LOCKED}
            for population in panels}, "composition_max_relative_l2": max(row[
                "relative_l2_error"] for row in compositions), "predictions": predictions,
        "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
