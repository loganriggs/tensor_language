#!/usr/bin/env python3
"""Selective binary-command contrast removal at the frozen H4 interface."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_midpoint_invariance_finiteness_and_exact_price pred_b_h4_midpoint_selectively_removes_original_command_contrasts pred_c_h4_midpoint_selectively_removes_ood_command_contrasts pred_d_simultaneous_midpoint_removal_is_additive_and_preserves_effects pred_e_h4_selective_removal_is_licensed
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
import joint_command_composition_contract as joint
import run_temporal_iswas_dual_command_shared_head_module_factorial_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_h4_selective_midpoint_removal_v1.json"
ORIGINAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_v2.py"
OOD_BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_ood_v1.py"
ORIGINAL_H4 = ROOT / "circuits/followups/temporal_iswas_four_head_union_joint_composition_v1_result.json"
OOD_H4 = ROOT / "circuits/followups/temporal_iswas_h4_ood_joint_composition_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_dual_command_shared_head_module_factorial_v1.py"
JOINT = ROOT / "ops/joint_command_composition_contract.py"
ACCOUNTING = ROOT / "ops/dual_command_head_module_factorial_contract.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_h4_selective_midpoint_removal_v1_result.json"
EXPECTED = {
    "prior": "72b41d4c71f8b600a4a142d6234cf20fbc455d7eec50b87fc6d916e5e70bdf92",
    "original_builder": "72da11860ce5bf1c03ea126bc10fcb6c46dd2648169edb00a1e9e1dbeee8a0d0",
    "ood_builder": "6062f2b8dfaa54da71477b43cb9fc63650db389c5883c9e64125f0a200afeb5a",
    "original_h4": "a79a172f23a28befcb4ba0b78f1a5ca4420c5f49323d2c51d912b46f93bb611d",
    "ood_h4": "a040a70cbe92728155f4e5affc232db194089e7b5ea501ab9c96f9ad164f45bd",
    "parent_runner": "61cc52e73d51441512b073b75eef86f41ff5941c8f97770b7a062446c47ef849",
    "joint": "940e34ca48f1ef998ba0a45f51d0c77f5ea6860a0364a71781b026cf264b72c2",
    "accounting": "985b824c7d3d8b622dfb92e3522f107e15e5e095b1f40a59a46122ad9af68496",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
UNION = {9: (1, 4), 11: (3,), 15: (5,)}
PRICE = {"checkpoint_loads": 1, "model_forwards": 8, "sequence_evaluations": 1024,
         "scored_token_positions": 2048, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_pairing_midpoint_invariance_finiteness_and_exact_price",
    "pred_b_h4_midpoint_selectively_removes_original_command_contrasts",
    "pred_c_h4_midpoint_selectively_removes_ood_command_contrasts",
    "pred_d_simultaneous_midpoint_removal_is_additive_and_preserves_effects",
    "pred_e_h4_selective_removal_is_licensed",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def midpoint_tensor(current, donor):
    if current.shape != donor.shape: raise ValueError("midpoint tensors differ")
    return (current + donor) * 0.5


def _forward_midpoint(backend, tokens, captures, swaps):
    torch, F, model = backend.torch, backend.F, backend.model
    width, handles = model.config.n_embd // model.config.n_head, []
    for layer, heads in UNION.items():
        def patch(_module, inputs, layer=layer, heads=heads):
            changed, native = inputs[0].clone(), captures[layer]
            for pairs, positions in swaps:
                for index, position in enumerate(positions):
                    donor_index = pairs[index]; donor_position = positions[donor_index]
                    for head in heads:
                        sl = slice(head * width, (head + 1) * width)
                        changed[index, position, sl] = midpoint_tensor(
                            native[index, position, sl], native[donor_index, donor_position, sl])
            return (changed,) + tuple(inputs[1:])
        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    try:
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, first = x, None
        for block in model.transformer.h: x, first = block(x, first, x0)
        return (30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)).float()
    finally:
        for handle in handles: handle.remove()


def margins(logits, endpoints, role):
    positive, negative = ((original._single(" will"), original._single(" had"))
                          if role == "temporal" else
                          (original._single(" is"), original._single(" was")))
    return np.asarray([float((logits[index, endpoint[f"{role}_position"] - 1, positive]
        - logits[index, endpoint[f"{role}_position"] - 1, negative]).item())
        for index, (_row, _cell, endpoint) in enumerate(endpoints)], dtype=np.float64)


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
    positions = {role: [endpoint[f"{role}_position"] - 1 for _row, _cell, endpoint in endpoints]
                 for role in ("temporal", "iswas")}
    with torch.no_grad():
        native_logits, captures = parent._forward(backend, tokens, capture_layers=(9, 11, 15))
        temporal_logits = _forward_midpoint(backend, tokens, captures,
            ((pairs["temporal"], positions["temporal"]),))
        iswas_logits = _forward_midpoint(backend, tokens, captures,
            ((pairs["iswas"], positions["iswas"]),))
        both_logits = _forward_midpoint(backend, tokens, captures,
            ((pairs["temporal"], positions["temporal"]), (pairs["iswas"], positions["iswas"])))
    logits = {"00": native_logits, "10": temporal_logits, "01": iswas_logits, "11": both_logits}
    arm_margins = {role: {arm: margins(value, endpoints, role) for arm, value in logits.items()}
                   for role in ("temporal", "iswas")}
    midpoint_error = 0.0
    for role in pairs:
        for layer, heads in UNION.items():
            for index, pair in enumerate(pairs[role]):
                for head in heads:
                    width = captures[layer].shape[-1] // 9; sl = slice(head * width, (head + 1) * width)
                    left = midpoint_tensor(captures[layer][index, positions[role][index], sl],
                        captures[layer][pair, positions[role][pair], sl])
                    right = midpoint_tensor(captures[layer][pair, positions[role][pair], sl],
                        captures[layer][index, positions[role][index], sl])
                    midpoint_error = max(midpoint_error, float((left - right).abs().max()))
    single_reports, composition_reports = [], []
    templates = authority.TEMPLATES
    for phase in ("FIT", "HOLDOUT"):
        for template in ("ALL",) + templates:
            selected = np.asarray([row["phase"] == phase and
                (template == "ALL" or row["template_id"] == template)
                for row, _cell, _endpoint in endpoints])
            for role, arm, bit in (("temporal", "10", 0), ("iswas", "01", 1)):
                other = "iswas" if role == "temporal" else "temporal"
                half_gold = 0.5 * (arm_margins[role]["00"][pairs[role]] - arm_margins[role]["00"])
                effect = arm_margins[role][arm] - arm_margins[role]["00"]
                report = accounting.effect_metrics(effect[selected], half_gold[selected],
                    (arm_margins[other][arm] - arm_margins[other]["00"])[selected])
                simultaneous = accounting.effect_metrics(
                    (arm_margins[role]["11"] - arm_margins[role]["00"])[selected],
                    half_gold[selected],
                    (arm_margins[other]["11"] - arm_margins[other]["00"])[selected])
                signs = np.asarray([1.0 if cell[bit] == "0" else -1.0
                    for _row, cell, _endpoint in endpoints])
                native_correct = (signs * arm_margins[role]["00"] > 0) & selected
                reductions = signs * (arm_margins[role]["00"] - arm_margins[role][arm])
                fraction = float(np.mean(reductions[native_correct] > 0)) if native_correct.any() else 0.0
                single_reports.append({"population": population, "role": role, "phase": phase,
                    "template_id": template, "arm": arm, **report,
                    "native_correct_count": int(native_correct.sum()),
                    "native_correct_margin_reduction_fraction": fraction,
                    "simultaneous_signed_recovery": simultaneous["signed_recovery"],
                    "simultaneous_cosine": simultaneous["cosine"]})
            for role in ("temporal", "iswas"):
                decomposition = joint.decompose(torch, {arm: arm_margins[role][arm][selected]
                                                        for arm in joint.CELLS})
                composition_reports.append({"population": population, "output_role": role,
                    "phase": phase, "template_id": template,
                    "closure_max_abs_error": decomposition["closure_max_abs_error"],
                    "interaction_rms_over_both_rms": decomposition["interaction_rms_over_both_rms"],
                    "additive_cosine_to_both_effect": decomposition["additive_cosine_to_both_effect"]})
    causal_zero = float(np.max(np.abs(arm_margins["temporal"]["01"] - arm_margins["temporal"]["00"])))
    pairing_ok = all(pairs[role][pairs[role][i]] == i for role in pairs for i in range(128))
    positions_ok = all(positions[role][i] == positions[role][pair] for role in pairs
                       for i, pair in enumerate(pairs[role]))
    capture_ok = set(captures) == {9, 11, 15} and all(value.shape == (128, maximum, 1152)
                                                           for value in captures.values())
    return {"single_reports": single_reports, "composition_reports": composition_reports,
            "iswas_to_temporal_max_abs_effect": causal_zero,
            "midpoint_pair_invariance_max_abs_error": midpoint_error,
            "pairing_ok": pairing_ok, "positions_ok": positions_ok, "capture_ok": capture_ok}


def main():
    paths = {"prior": PRIOR, "original_builder": ORIGINAL_BUILDER, "ood_builder": OOD_BUILDER,
             "original_h4": ORIGINAL_H4, "ood_h4": OOD_H4, "parent_runner": PARENT_RUNNER,
             "joint": JOINT, "accounting": ACCOUNTING, "producer": PRODUCER}
    observed = {name: sha(path) for name, path in paths.items()}
    original_h4, ood_h4 = (json.loads(path.read_text()) for path in (ORIGINAL_H4, OOD_H4))
    authority = bool(observed == EXPECTED
        and original_h4.get("terminal") == "four_head_joint_program_licensed"
        and ood_h4.get("terminal") == "h4_ood_joint_program_licensed")
    dry = {"candidate_id": json.loads(PRIOR.read_text()).get("candidate_id"), "dryrun": True,
           "authority_ok": authority, "gpu_accessed": False, "model_loaded": False,
           "queue_touched": False, "populations": ["original", "ood"], "union": UNION,
           "price": PRICE}
    if not authority: raise RuntimeError(f"H4 removal authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    panels = {"original": run_population(backend, original, "original"),
              "ood": run_population(backend, ood, "ood")}
    singles = [row for panel in panels.values() for row in panel["single_reports"]]
    compositions = [row for panel in panels.values() for row in panel["composition_reports"]]
    counters = PRICE.copy()
    A = bool(authority and counters == PRICE and all(panel["pairing_ok"] and panel["positions_ok"]
        and panel["capture_ok"] and panel["midpoint_pair_invariance_max_abs_error"] <= 1e-6
        and panel["iswas_to_temporal_max_abs_effect"] <= 1e-5 for panel in panels.values())
        and max(row["closure_max_abs_error"] for row in compositions) <= 1e-6
        and finite(panels))
    def population_pass(name):
        pooled = [row for row in singles if row["population"] == name and row["template_id"] == "ALL"]
        return all(row["signed_recovery"] >= (.80 if row["role"] == "temporal" else .65)
            and row["cosine"] >= .98 and row["direction_agreement"] == 1.0
            and row["non_target_to_target_gold_norm"] <= .01
            and row["native_correct_margin_reduction_fraction"] >= .75 for row in pooled)
    B, C = population_pass("original"), population_pass("ood")
    D = all(row["interaction_rms_over_both_rms"] <= .10
            and row["additive_cosine_to_both_effect"] >= .99 for row in compositions
            if row["template_id"] != "ALL") and all(
            row["simultaneous_signed_recovery"] >= row["signed_recovery"] - .05 for row in singles
            if row["template_id"] == "ALL")
    E = A and B and C and D
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = "invalid" if not A else "h4_selective_removal_licensed" if E else "h4_removal_null"
    result = {"schema": "temporal_iswas_h4_selective_midpoint_removal_result_v1",
        "candidate_id": json.loads(PRIOR.read_text())["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "union": UNION, "panels": panels, "predictions": predictions,
        "terminal": terminal, "price": counters}
    atomic_create_json(OUT, result)
    print(json.dumps({"instrument": {name: {key: panel[key] for key in
        ("midpoint_pair_invariance_max_abs_error", "iswas_to_temporal_max_abs_effect",
         "pairing_ok", "positions_ok", "capture_ok")} for name, panel in panels.items()},
        "single_reports": singles, "composition_reports": compositions,
        "predictions": predictions, "terminal": terminal, "price": counters}, sort_keys=True))


if __name__ == "__main__": main()
