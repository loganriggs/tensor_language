#!/usr/bin/env python3
"""Full-head and parent-module swaps for exact shared weight nominations."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_capture_closure_finiteness_and_exact_price pred_b_each_command_has_a_material_complete_module_route pred_c_a_nominated_head_accounts_for_its_parent_route pred_d_three_head_union_is_distributive_and_improves_recovery pred_e_task_typed_selectivity_survives_shared_weight_use
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_candidate_temporal_iswas_dual_command_v2 as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import dual_command_head_module_factorial_contract as accounting

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_dual_command_shared_head_module_factorial_v1.json"
ATLAS = ROOT / "circuits/followups/temporal_iswas_common_gauge_weight_interface_atlas_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_iswas_dual_command_prefix_preserved_native_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_v2.py"
ACCOUNTING = ROOT / "ops/dual_command_head_module_factorial_contract.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_dual_command_shared_head_module_factorial_v1_result.json"
EXPECTED = {
    "prior": "7332c7358da591dbb7fac0c0251314f4f5e9895a56b91a0ec4c3d6f22c9abf8d",
    "atlas": "d951420fef51f9d550237927072bfc721bd73112f87d5f716b3cb7c2ca3483ae",
    "capability": "PENDING_CAPABILITY_RESULT_SHA256",
    "builder": "72da11860ce5bf1c03ea126bc10fcb6c46dd2648169edb00a1e9e1dbeee8a0d0",
    "accounting": "985b824c7d3d8b622dfb92e3522f107e15e5e095b1f40a59a46122ad9af68496",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
SITES = ((9, 1), (11, 3), (15, 5))
PRICE = {"checkpoint_loads": 1, "model_forwards": 15, "sequence_evaluations": 1920,
         "scored_token_positions": 3840, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
EXPECTED_CONFIG = {"vocab_size": 50304, "n_layer": 18, "n_head": 9,
                   "n_embd": 1152, "squared_mlp": False, "bilinear": True,
                   "expansion_factor": 4, "gated": False,
                   "squared_attn": True, "bilinear_attn": True}
PREDICTION_KEYS = (
    "pred_a_authority_pairing_capture_closure_finiteness_and_exact_price",
    "pred_b_each_command_has_a_material_complete_module_route",
    "pred_c_a_nominated_head_accounts_for_its_parent_route",
    "pred_d_three_head_union_is_distributive_and_improves_recovery",
    "pred_e_task_typed_selectivity_survives_shared_weight_use",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_structure(value):
    if isinstance(value, dict): return all(finite_structure(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite_structure(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def endpoint_bank(rows):
    endpoints = [(row, cell, row["endpoints"][cell]) for row in rows for cell in candidate.CELLS]
    lookup = {(row["row_id"], cell): index for index, (row, cell, _endpoint) in enumerate(endpoints)}
    return endpoints, lookup


def pair_indices(endpoints, lookup, role):
    return [lookup[(row["row_id"], accounting.paired_cell(cell, role))]
            for row, cell, _endpoint in endpoints]


def _forward(backend, tokens, *, capture_layers=(), captures=None, pairs=None,
             positions=None, patch_spec=None):
    torch, F, model = backend.torch, backend.F, backend.model
    saved, handles = {}, []
    for layer in capture_layers:
        def save(_module, inputs, layer=layer): saved[layer] = inputs[0].detach().clone()
        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(save))
    if patch_spec:
        if captures is None or pairs is None or positions is None: raise ValueError("patch inputs absent")
        width = model.config.n_embd // model.config.n_head
        for layer, heads in patch_spec.items():
            def patch(_module, inputs, layer=layer, heads=heads):
                value = inputs[0]; changed = value.clone(); donor = captures[layer]
                for index, position in enumerate(positions):
                    donor_index = pairs[index]; donor_position = positions[donor_index]
                    if heads is None:
                        changed[index, position] = donor[donor_index, donor_position]
                    else:
                        for head in heads:
                            sl = slice(head * width, (head + 1) * width)
                            changed[index, position, sl] = donor[donor_index, donor_position, sl]
                return (changed,) + tuple(inputs[1:])
            handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    try:
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
        x0, first = x, None
        for block in model.transformer.h: x, first = block(x, first, x0)
        logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)).float()
    finally:
        for handle in handles: handle.remove()
    return logits, saved


def margins(logits, endpoints, role):
    positive, negative = ((candidate._single(" will"), candidate._single(" had"))
                          if role == "temporal" else
                          (candidate._single(" is"), candidate._single(" was")))
    values = []
    for index, (_row, _cell, endpoint) in enumerate(endpoints):
        position = endpoint[f"{role}_position"] - 1
        values.append(float((logits[index, position, positive] - logits[index, position, negative]).item()))
    return np.asarray(values, dtype=np.float64)


def report_lookup(reports, role, phase, arm):
    return next(row for row in reports if row["role"] == role and row["phase"] == phase
                and row["template_id"] == "ALL" and row["arm"] == arm)


def main():
    files = {"prior": PRIOR, "atlas": ATLAS, "capability": CAPABILITY, "builder": BUILDER,
             "accounting": ACCOUNTING, "producer": PRODUCER}
    if not CAPABILITY.exists(): raise RuntimeError("dual capability receipt has not landed")
    observed = {name: sha(path) for name, path in files.items()}
    prior, atlas, capability = (json.loads(path.read_text()) for path in (PRIOR, ATLAS, CAPABILITY))
    authority = bool(observed == EXPECTED and atlas.get("terminal") == "shared_weight_interface_candidates"
                     and capability.get("terminal") == "dual_command_license_issued"
                     and len(capability.get("licensed_row_ids", [])) == 32)
    rows = candidate.build_rows(); endpoints, lookup = endpoint_bank(rows)
    dryrun = {"candidate_id": prior.get("candidate_id"), "dryrun": True,
              "authority_ok": authority, "gpu_accessed": False, "model_loaded": False,
              "queue_touched": False, "rows": 32, "sequences": 128,
              "sites": [list(site) for site in SITES], "price": PRICE}
    if not authority: raise RuntimeError(f"shared head factorial authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints], dtype=torch.long,
                          device=backend.device)
    with torch.no_grad():
        native_logits, captures = _forward(backend, tokens, capture_layers=tuple(x[0] for x in SITES))
    native = {role: margins(native_logits, endpoints, role) for role in ("temporal", "iswas")}
    actual_forwards = 1; records = []; effects = {}
    for role in ("temporal", "iswas"):
        pairs = pair_indices(endpoints, lookup, role)
        positions = [endpoint[f"{role}_position"] - 1 for _row, _cell, endpoint in endpoints]
        arms = [(f"head:L{layer:02d}H{head:02d}", {layer: (head,)}) for layer, head in SITES]
        arms += [(f"module:L{layer:02d}", {layer: None}) for layer, _head in SITES]
        arms += [("head_union:L09H01+L11H03+L15H05",
                  {layer: (head,) for layer, head in SITES})]
        for arm, patch_spec in arms:
            with torch.no_grad(): patched_logits, _saved = _forward(
                backend, tokens, captures=captures, pairs=pairs, positions=positions,
                patch_spec=patch_spec)
            actual_forwards += 1
            patched = {name: margins(patched_logits, endpoints, name) for name in ("temporal", "iswas")}
            target_effect = patched[role] - native[role]
            gold = native[role][pairs] - native[role]
            other = "iswas" if role == "temporal" else "temporal"
            non_target = patched[other] - native[other]
            effects[(role, arm)] = target_effect
            for index, (row, cell, _endpoint) in enumerate(endpoints):
                records.append({"row_id": row["row_id"], "phase": row["phase"],
                                "template_id": row["template_id"], "cell": cell,
                                "paired_cell": accounting.paired_cell(cell, role), "role": role,
                                "arm": arm, "target_effect": float(target_effect[index]),
                                "target_gold": float(gold[index]),
                                "non_target_effect": float(non_target[index])})
    reports = accounting.aggregate(records)
    additivity = []
    for role in ("temporal", "iswas"):
        union = effects[(role, "head_union:L09H01+L11H03+L15H05")]
        singles = [effects[(role, f"head:L{layer:02d}H{head:02d}")] for layer, head in SITES]
        for phase in ("FIT", "HOLDOUT"):
            for template in ("ALL",) + candidate.TEMPLATES:
                mask = np.asarray([row["phase"] == phase and (template == "ALL" or row["template_id"] == template)
                                   for row, _cell, _endpoint in endpoints])
                additivity.append({"role": role, "phase": phase, "template_id": template,
                                   "relative_union_minus_singleton_sum": accounting.union_additivity(
                                       union[mask], [value[mask] for value in singles])})

    config = {key: getattr(backend.model.config, key) for key in EXPECTED_CONFIG}
    counters = {"checkpoint_loads": 1, "model_forwards": actual_forwards,
                "sequence_evaluations": actual_forwards * len(endpoints),
                "scored_token_positions": actual_forwards * len(endpoints) * 2,
                "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
    pairing_ok = all(pair_indices(endpoints, lookup, role)[pair_indices(endpoints, lookup, role)[i]] == i
                     for role in ("temporal", "iswas") for i in range(len(endpoints)))
    position_pairing_ok = all(
        endpoints[i][2][f"{role}_position"] == endpoints[pair][2][f"{role}_position"]
        for role in ("temporal", "iswas")
        for i, pair in enumerate(pair_indices(endpoints, lookup, role)))
    capture_ok = set(captures) == {9, 11, 15} and all(
        tuple(value.shape[:2]) == (128, maximum) and value.shape[2] == 1152 for value in captures.values())
    finite = finite_structure({"records": records, "reports": reports, "additivity": additivity})
    A = bool(authority and config == EXPECTED_CONFIG and pairing_ok and position_pairing_ok
             and capture_ok and finite and len(records) == 14 * 128 and counters == PRICE)
    material_modules, admitted_heads = {}, {}
    for role in ("temporal", "iswas"):
        material_modules[role] = []
        admitted_heads[role] = []
        for layer, head in SITES:
            module_arm, head_arm = f"module:L{layer:02d}", f"head:L{layer:02d}H{head:02d}"
            module_ok = all((lambda report: report["signed_recovery"] >= .50
                             and report["direction_agreement"] >= .75 and report["cosine"] >= .50)(
                                 report_lookup(reports, role, phase, module_arm))
                            for phase in ("FIT", "HOLDOUT"))
            if module_ok: material_modules[role].append(module_arm)
            head_ok = module_ok and all((lambda head_report, module_report:
                head_report["signed_recovery"] >= .25
                and head_report["signed_recovery"] >= .50 * module_report["signed_recovery"]
                and head_report["direction_agreement"] >= .75 and head_report["cosine"] >= .50)(
                    report_lookup(reports, role, phase, head_arm),
                    report_lookup(reports, role, phase, module_arm)) for phase in ("FIT", "HOLDOUT"))
            if head_ok: admitted_heads[role].append(head_arm)
    B = all(material_modules[role] for role in ("temporal", "iswas"))
    C = all(admitted_heads[role] for role in ("temporal", "iswas"))
    union_arm = "head_union:L09H01+L11H03+L15H05"
    D = all(next(row for row in additivity if row["role"] == role and row["phase"] == phase
                 and row["template_id"] == "ALL")["relative_union_minus_singleton_sum"] <= .25
            and report_lookup(reports, role, phase, union_arm)["signed_recovery"] >=
                max(report_lookup(reports, role, phase, f"head:L{layer:02d}H{head:02d}")["signed_recovery"]
                    for layer, head in SITES) - .05
            for role in ("temporal", "iswas") for phase in ("FIT", "HOLDOUT"))
    admitted_reports = [report_lookup(reports, role, phase, arm) for role in ("temporal", "iswas")
                        for arm in admitted_heads[role] for phase in ("FIT", "HOLDOUT")]
    causal_zero = max(abs(record["non_target_effect"]) for record in records
                      if record["role"] == "iswas")
    E = bool(C and all(report["non_target_to_target_gold_norm"] <= .25 for report in admitted_reports)
             and causal_zero <= 1e-5)
    predictions = dict(zip(PREDICTION_KEYS, (A, B, C, D, E)))
    terminal = ("invalid" if not A else "shared_selective_distributive_head_routes" if B and C and D and E
                else "shared_selective_interacting_head_routes" if B and C and E
                else "parent_module_routes_only" if B else "shared_weight_causal_route_null")
    result = {"schema": "temporal_iswas_dual_command_shared_head_module_factorial_result_v1",
              "candidate_id": prior["candidate_id"], "started_utc": started_utc,
              "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
              "model_config": config, "material_modules": material_modules,
              "admitted_heads": admitted_heads, "iswas_to_temporal_max_abs_effect": causal_zero,
              "reports": reports, "union_additivity": additivity, "records": records,
              "predictions": predictions, "terminal": terminal, "price": counters}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("material_modules", "admitted_heads",
        "iswas_to_temporal_max_abs_effect", "union_additivity", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__": main()
