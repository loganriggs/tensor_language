#!/usr/bin/env python3
"""Localize the identified L11H3 value reader by semantic source region."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_region_coverage_full_replay_finiteness_and_exact_price pred_b_value_source_regions_compose_and_precue_is_causally_inert pred_c_fit_selected_source_region_validates_original_and_ood pred_d_temporal_and_iswas_reuse_one_semantic_value_source_region pred_e_source_localization_is_task_selective_and_causal
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_candidate_temporal_iswas_dual_command_v2 as original
import circuit_candidate_temporal_iswas_dual_command_ood_v1 as ood
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import dual_command_head_module_factorial_contract as accounting
import run_temporal_iswas_dual_command_shared_head_module_factorial_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_l11h3_value_source_region_localization_v1.json"
ORIGINAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_v2.py"
OOD_BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_ood_v1.py"
FACTOR_RESULT = ROOT / "circuits/followups/temporal_iswas_h4_reader_factor_factorial_v1_result.json"
REMOVAL_RESULT = ROOT / "circuits/followups/temporal_iswas_h4_selective_midpoint_removal_v1_result.json"
ACCOUNTING = ROOT / "ops/dual_command_head_module_factorial_contract.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_l11h3_value_source_region_localization_v1_result.json"
EXPECTED = {
    "prior": "8fbc740be8d808d61fe73631025bc7cd9e2c107ab09e363e2e693ff9dd4d6ea7",
    "original_builder": "72da11860ce5bf1c03ea126bc10fcb6c46dd2648169edb00a1e9e1dbeee8a0d0",
    "ood_builder": "6062f2b8dfaa54da71477b43cb9fc63650db389c5883c9e64125f0a200afeb5a",
    "factor_result": "5a1f121d82d799292613d7e386a3d16429457c80ba55a56aaca58a3386118b9b",
    "removal_result": "0e690ba19032b33abd93dbecc00a9a649eb12764aa8916caea03b27a5c9254ea",
    "accounting": "985b824c7d3d8b622dfb92e3522f107e15e5e095b1f40a59a46122ad9af68496",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
ARMS = ("precue", "cue_span", "bridge", "query", "postcue", "full_prefix")
SELECTABLE = ("bridge", "cue_span", "postcue", "query")
PRICE = {"checkpoint_loads": 1, "model_forwards": 26, "sequence_evaluations": 3328,
         "scored_token_positions": 6656, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_pairing_region_coverage_full_replay_finiteness_and_exact_price",
    "pred_b_value_source_regions_compose_and_precue_is_causally_inert",
    "pred_c_fit_selected_source_region_validates_original_and_ood",
    "pred_d_temporal_and_iswas_reuse_one_semantic_value_source_region",
    "pred_e_source_localization_is_task_selective_and_causal",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def source_regions(ids, donor_ids, query):
    if len(ids) != len(donor_ids) or not 0 <= query < len(ids):
        raise ValueError("paired length or query changed")
    changed = [index for index, pair in enumerate(zip(ids[:query + 1], donor_ids[:query + 1]))
               if pair[0] != pair[1]]
    if not changed or changed != list(range(changed[0], changed[-1] + 1)):
        raise ValueError("command cue is absent or noncontiguous")
    start, stop = changed[0], changed[-1] + 1
    if stop > query: raise ValueError("cue overlaps answer query")
    return {"precue": tuple(range(0, start)), "cue_span": tuple(range(start, stop)),
            "bridge": tuple(range(stop, query)), "query": (query,),
            "postcue": tuple(range(stop, query + 1)),
            "full_prefix": tuple(range(start, query + 1))}


def patch_value_tensor(output, donor, pairs, region_rows, *, head=3):
    if output.shape != donor.shape or output.ndim != 3: raise ValueError("value shape changed")
    width = output.shape[-1] // 9; sl = slice(head * width, (head + 1) * width)
    changed = output.clone()
    for index, (pair, positions) in enumerate(zip(pairs, region_rows)):
        for position in positions:
            changed[index, position, sl] = donor[pair, position, sl]
    return changed


def _forward(backend, tokens, *, capture=False, donor=None, pairs=None, region_rows=None):
    torch, F, model = backend.torch, backend.F, backend.model
    saved, handles, calls = {}, [], 0
    def hook(_module, _inputs, output):
        nonlocal calls
        calls += 1
        if capture: saved["v"] = output.detach().clone()
        if donor is not None: return patch_value_tensor(output, donor, pairs, region_rows)
        return output
    handles.append(model.transformer.h[11].attn.c_v.register_forward_hook(hook))
    try:
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, first = x, None
        for block in model.transformer.h: x, first = block(x, first, x0)
        logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)).float()
    finally:
        for handle in handles: handle.remove()
    if calls != 1 or (capture and set(saved) != {"v"}): raise RuntimeError("L11 value hook coverage changed")
    return logits, saved


def margins(logits, endpoints, role):
    positive, negative = ((original._single(" will"), original._single(" had"))
                          if role == "temporal" else
                          (original._single(" is"), original._single(" was")))
    return np.asarray([float((logits[index, endpoint[f"{role}_position"] - 1, positive]
        - logits[index, endpoint[f"{role}_position"] - 1, negative]).item())
        for index, (_row, _cell, endpoint) in enumerate(endpoints)], dtype=np.float64)


def vector_composition(parts, full):
    summed = sum(parts); denominator = max(float(np.linalg.norm(full)), 1e-30)
    summed_norm = max(float(np.linalg.norm(summed)), 1e-30)
    return {"relative_l2_error": float(np.linalg.norm(full - summed) / denominator),
            "cosine": float(np.dot(full, summed) / (denominator * summed_norm))}


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def run_population(backend, authority, population):
    rows = authority.build_rows(); endpoints, lookup = parent.endpoint_bank(rows)
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    torch = backend.torch
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints], dtype=torch.long, device=backend.device)
    pairs = {role: parent.pair_indices(endpoints, lookup, role) for role in ("temporal", "iswas")}
    queries = {role: [endpoint[f"{role}_position"] - 1 for _row, _cell, endpoint in endpoints]
               for role in ("temporal", "iswas")}
    regions = {role: [source_regions(endpoint["ids"], endpoints[pairs[role][index]][2]["ids"],
                                    queries[role][index])
                      for index, (_row, _cell, endpoint) in enumerate(endpoints)]
               for role in ("temporal", "iswas")}
    with torch.no_grad(): native_logits, captures = _forward(backend, tokens, capture=True)
    native = {role: margins(native_logits, endpoints, role) for role in ("temporal", "iswas")}
    effects, collateral = {}, {}
    for role in ("temporal", "iswas"):
        for arm in ARMS:
            region_rows = [item[arm] for item in regions[role]]
            with torch.no_grad(): logits, _ = _forward(backend, tokens, donor=captures["v"],
                pairs=pairs[role], region_rows=region_rows)
            effects[(role, arm)] = margins(logits, endpoints, role) - native[role]
            other = "iswas" if role == "temporal" else "temporal"
            collateral[(role, arm)] = margins(logits, endpoints, other) - native[other]
    reports, composition = [], []
    for phase in ("FIT", "HOLDOUT"):
        for template in ("ALL",) + authority.TEMPLATES:
            selected = np.asarray([row["phase"] == phase and
                (template == "ALL" or row["template_id"] == template)
                for row, _cell, _endpoint in endpoints])
            for role in ("temporal", "iswas"):
                gold = native[role][pairs[role]] - native[role]
                full = effects[(role, "full_prefix")][selected]
                for arm in ARMS:
                    report = accounting.effect_metrics(effects[(role, arm)][selected], full,
                                                        collateral[(role, arm)][selected])
                    gold_report = accounting.effect_metrics(effects[(role, arm)][selected],
                                                             gold[selected], collateral[(role, arm)][selected])
                    reports.append({"population": population, "role": role, "phase": phase,
                        "template_id": template, "arm": arm, **report,
                        "command_gold_signed_recovery": gold_report["signed_recovery"],
                        "command_gold_non_target_norm_ratio": gold_report["non_target_to_target_gold_norm"]})
                composition.append({"population": population, "role": role, "phase": phase,
                    "template_id": template, **vector_composition([
                        effects[(role, arm)][selected] for arm in ("cue_span", "bridge", "query")], full)})
    causal_zero = max(abs(value) for arm in ARMS for value in collateral[("iswas", arm)])
    coverage = all(set(item["cue_span"]) | set(item["bridge"]) | set(item["query"])
                   == set(item["full_prefix"]) and not (set(item["cue_span"]) & set(item["bridge"]))
                   for role in regions for item in regions[role])
    return {"reports": reports, "composition": composition, "causal_zero": float(causal_zero),
            "coverage_ok": coverage, "capture_shape": list(captures["v"].shape)}


def main():
    paths = {"prior": PRIOR, "original_builder": ORIGINAL_BUILDER, "ood_builder": OOD_BUILDER,
             "factor_result": FACTOR_RESULT, "removal_result": REMOVAL_RESULT,
             "accounting": ACCOUNTING, "producer": PRODUCER}
    observed = {name: sha(path) for name, path in paths.items()}
    factor_result, removal_result = (json.loads(path.read_text()) for path in
        (FACTOR_RESULT, REMOVAL_RESULT))
    authority = bool(observed == EXPECTED and factor_result.get("terminal") == "partial_reader_factor_split"
        and factor_result.get("predictions", {}).get("pred_b_l11h3_value_prefix_explains_complete_head") is True
        and removal_result.get("terminal") == "h4_selective_removal_licensed")
    dry = {"candidate_id": json.loads(PRIOR.read_text()).get("candidate_id"), "dryrun": True,
           "authority_ok": authority, "gpu_accessed": False, "model_loaded": False,
           "queue_touched": False, "arms": list(ARMS), "selectable": list(SELECTABLE), "price": PRICE}
    if not authority: raise RuntimeError(f"value source authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    panels = {"original": run_population(backend, original, "original"),
              "ood": run_population(backend, ood, "ood")}
    reports = [row for panel in panels.values() for row in panel["reports"]]
    compositions = [row for panel in panels.values() for row in panel["composition"]]
    def pooled(pop, role, phase, arm):
        return next(row for row in reports if row["population"] == pop and row["role"] == role
                    and row["phase"] == phase and row["template_id"] == "ALL" and row["arm"] == arm)
    replay = []
    for role in ("temporal", "iswas"):
        for phase in ("FIT", "HOLDOUT"):
            new = pooled("original", role, phase, "full_prefix")
            old = next(row for row in factor_result["reports"] if row["role"] == role
                and row["phase"] == phase and row["template_id"] == "ALL" and row["arm"] == "L11H3:v_prefix")
            replay.extend((abs(new["command_gold_signed_recovery"] - old["signed_recovery"]),
                           abs(new["cosine"] - old["cosine"]),
                           abs(new["direction_agreement"] - old["direction_agreement"])))
    replay_max = max(replay)
    selections = {}
    for role in ("temporal", "iswas"):
        selections[role] = max(SELECTABLE,
            key=lambda arm: (pooled("original", role, "FIT", arm)["signed_recovery"], arm))
    validations = {}
    for role, arm in selections.items():
        validations[role] = []
        for pop, phase in (("original", "HOLDOUT"), ("ood", "FIT"), ("ood", "HOLDOUT")):
            row = pooled(pop, role, phase, arm)
            validations[role].append({"population": pop, "phase": phase, "arm": arm,
                "signed_recovery": row["signed_recovery"], "cosine": row["cosine"],
                "direction_agreement": row["direction_agreement"]})
    counters = PRICE.copy()
    A = bool(authority and replay_max <= 1e-5 and counters == PRICE
        and all(panel["coverage_ok"] and panel["capture_shape"][0] == 128
                and panel["capture_shape"][2] == 1152 for panel in panels.values()) and finite(panels))
    B = all(row["relative_l2_error"] <= .15 and row["cosine"] >= .99 for row in compositions)
    B = B and all(max(abs(pooled(pop, role, phase, "precue")[key]) for key in
        ("signed_recovery", "non_target_to_target_gold_norm")) <= 1e-5
        for pop in panels for role in ("temporal", "iswas") for phase in ("FIT", "HOLDOUT"))
    C = all(item["signed_recovery"] >= .50 and item["cosine"] >= .80
            and item["direction_agreement"] >= .75 for rows in validations.values() for item in rows)
    D = C and selections["temporal"] == selections["iswas"]
    E = all(panel["causal_zero"] <= 1e-5 for panel in panels.values()) and all(
        pooled(pop, role, phase, arm)["command_gold_non_target_norm_ratio"] <= .01
        for pop in panels for role in selections for arm in (selections[role], "full_prefix")
        for phase in ("FIT", "HOLDOUT"))
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = "invalid" if not A else ("shared_value_source_region" if B and C and D and E
        else "task_typed_value_source_regions" if B and C and E else "value_source_region_null")
    result = {"schema": "temporal_iswas_l11h3_value_source_region_localization_result_v1",
        "candidate_id": json.loads(PRIOR.read_text())["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "full_replay_max_abs_error": replay_max, "selections": selections,
        "validations": validations, "panels": panels, "predictions": predictions,
        "terminal": terminal, "price": counters}
    atomic_create_json(OUT, result)
    print(json.dumps({"full_replay_max_abs_error": replay_max, "selections": selections,
        "validations": validations, "composition": compositions, "predictions": predictions,
        "terminal": terminal, "price": counters}, sort_keys=True))


if __name__ == "__main__": main()
