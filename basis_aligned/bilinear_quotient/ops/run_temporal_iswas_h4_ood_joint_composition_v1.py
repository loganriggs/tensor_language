#!/usr/bin/env python3
"""Fresh-template/lexicon OOD joint confirmation for the frozen H4 program."""

# BQGATE: EXPERIMENT pred_a_authority_ood_license_pairing_finiteness_and_exact_price pred_b_h4_single_command_quality_and_selectivity_transfer_ood pred_c_h4_simultaneous_execution_remains_additive_ood pred_d_h4_simultaneous_execution_preserves_recoveries_ood pred_e_h4_ood_joint_program_is_licensed
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import circuit_candidate_temporal_iswas_dual_command_ood_v1 as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import dual_command_head_module_factorial_contract as accounting
import joint_command_composition_contract as joint
import run_temporal_iswas_dual_command_shared_head_module_factorial_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_h4_ood_joint_composition_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_ood_v1.py"
CAPABILITY = ROOT / "circuits/followups/temporal_iswas_dual_command_ood_native_capability_v1_result.json"
H4_RESULT = ROOT / "circuits/followups/temporal_iswas_four_head_union_joint_composition_v1_result.json"
PARENT_RUNNER = ROOT / "ops/run_temporal_iswas_dual_command_shared_head_module_factorial_v1.py"
JOINT = ROOT / "ops/joint_command_composition_contract.py"
ACCOUNTING = ROOT / "ops/dual_command_head_module_factorial_contract.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_h4_ood_joint_composition_v1_result.json"
EXPECTED = {
    "prior": "46f8d9d5ce584d8fd00aabb72b1641337b6eed4554a94c93712636da4c367092",
    "builder": "6062f2b8dfaa54da71477b43cb9fc63650db389c5883c9e64125f0a200afeb5a",
    "capability": "bf0c095dd78bd2e7f495104f34626669ba774f8ad9660771abf3202bcfb2802a",
    "h4_result": "a79a172f23a28befcb4ba0b78f1a5ca4420c5f49323d2c51d912b46f93bb611d",
    "parent_runner": "61cc52e73d51441512b073b75eef86f41ff5941c8f97770b7a062446c47ef849",
    "joint": "940e34ca48f1ef998ba0a45f51d0c77f5ea6860a0364a71781b026cf264b72c2",
    "accounting": "985b824c7d3d8b622dfb92e3522f107e15e5e095b1f40a59a46122ad9af68496",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
UNION = {9: (1, 4), 11: (3,), 15: (5,)}
PRICE = {"checkpoint_loads": 1, "model_forwards": 4, "sequence_evaluations": 512,
         "scored_token_positions": 1024, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_ood_license_pairing_finiteness_and_exact_price",
    "pred_b_h4_single_command_quality_and_selectivity_transfer_ood",
    "pred_c_h4_simultaneous_execution_remains_additive_ood",
    "pred_d_h4_simultaneous_execution_preserves_recoveries_ood",
    "pred_e_h4_ood_joint_program_is_licensed",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def joint_forward(backend, tokens, captures, swaps):
    torch, F, model = backend.torch, backend.F, backend.model
    width, handles = model.config.n_embd // model.config.n_head, []
    for layer, heads in UNION.items():
        def patch(_module, inputs, layer=layer, heads=heads):
            changed, donor = inputs[0].clone(), captures[layer]
            for pairs, positions in swaps:
                for index, position in enumerate(positions):
                    donor_index = pairs[index]; donor_position = positions[donor_index]
                    for head in heads:
                        sl = slice(head * width, (head + 1) * width)
                        changed[index, position, sl] = donor[donor_index, donor_position, sl]
            return (changed,) + tuple(inputs[1:])
        handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch))
    try:
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, first = x, None
        for block in model.transformer.h: x, first = block(x, first, x0)
        return (30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)).float()
    finally:
        for handle in handles: handle.remove()


def margins(logits, endpoints, role):
    positive, negative = ((_single(" will"), _single(" had")) if role == "temporal"
                          else (_single(" is"), _single(" was")))
    return np.asarray([float((logits[index, endpoint[f"{role}_position"] - 1, positive]
        - logits[index, endpoint[f"{role}_position"] - 1, negative]).item())
        for index, (_row, _cell, endpoint) in enumerate(endpoints)], dtype=np.float64)


def _single(text): return candidate._single(text)


def masks(endpoints, phase, template):
    return np.asarray([row["phase"] == phase and
        (template == "ALL" or row["template_id"] == template)
        for row, _cell, _endpoint in endpoints])


def finite(value): return parent.finite_structure(value)


def main():
    paths = {"prior": PRIOR, "builder": BUILDER, "capability": CAPABILITY,
             "h4_result": H4_RESULT, "parent_runner": PARENT_RUNNER,
             "joint": JOINT, "accounting": ACCOUNTING, "producer": PRODUCER}
    observed = {name: sha(path) for name, path in paths.items()}
    prior, capability, h4_result = (json.loads(path.read_text()) for path in
        (PRIOR, CAPABILITY, H4_RESULT))
    authority = bool(observed == EXPECTED
        and capability.get("terminal") == "dual_command_ood_license_issued"
        and len(capability.get("licensed_row_ids", [])) == 32
        and h4_result.get("terminal") == "four_head_joint_program_licensed"
        and h4_result.get("predictions", {}).get("pred_e_four_head_joint_program_is_licensed") is True)
    rows = candidate.build_rows(); endpoints, lookup = parent.endpoint_bank(rows)
    dry = {"candidate_id": prior.get("candidate_id"), "dryrun": True,
           "authority_ok": authority, "gpu_accessed": False, "model_loaded": False,
           "queue_touched": False, "union": UNION, "rows": 32, "anchors": 128,
           "arms": ["00", "10", "01", "11"], "price": PRICE}
    if not authority: raise RuntimeError(f"H4 OOD authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints], dtype=torch.long, device=backend.device)
    pairs = {role: parent.pair_indices(endpoints, lookup, role) for role in ("temporal", "iswas")}
    positions = {role: [endpoint[f"{role}_position"] - 1 for _row, _cell, endpoint in endpoints]
                 for role in ("temporal", "iswas")}
    with torch.no_grad():
        native_logits, captures = parent._forward(backend, tokens, capture_layers=(9, 11, 15))
        temporal_logits, _ = parent._forward(backend, tokens, captures=captures,
            pairs=pairs["temporal"], positions=positions["temporal"], patch_spec=UNION)
        iswas_logits, _ = parent._forward(backend, tokens, captures=captures,
            pairs=pairs["iswas"], positions=positions["iswas"], patch_spec=UNION)
        both_logits = joint_forward(backend, tokens, captures,
            ((pairs["temporal"], positions["temporal"]), (pairs["iswas"], positions["iswas"])))
    logits = {"00": native_logits, "10": temporal_logits, "01": iswas_logits, "11": both_logits}
    arm_margins = {role: {arm: margins(value, endpoints, role) for arm, value in logits.items()}
                   for role in ("temporal", "iswas")}
    single_reports, composition_reports = [], []
    for phase in ("FIT", "HOLDOUT"):
        for template in ("ALL",) + candidate.TEMPLATES:
            selected = masks(endpoints, phase, template)
            for role, arm in (("temporal", "10"), ("iswas", "01")):
                other = "iswas" if role == "temporal" else "temporal"
                gold = arm_margins[role]["00"][pairs[role]] - arm_margins[role]["00"]
                report = accounting.effect_metrics(
                    (arm_margins[role][arm] - arm_margins[role]["00"])[selected], gold[selected],
                    (arm_margins[other][arm] - arm_margins[other]["00"])[selected])
                simultaneous = accounting.effect_metrics(
                    (arm_margins[role]["11"] - arm_margins[role]["00"])[selected], gold[selected],
                    (arm_margins[other]["11"] - arm_margins[other]["00"])[selected])
                single_reports.append({"role": role, "phase": phase, "template_id": template,
                    "arm": arm, **report,
                    "simultaneous_signed_recovery": simultaneous["signed_recovery"],
                    "simultaneous_cosine": simultaneous["cosine"]})
            for role in ("temporal", "iswas"):
                decomposition = joint.decompose(torch, {arm: arm_margins[role][arm][selected]
                                                        for arm in joint.CELLS})
                composition_reports.append({"output_role": role, "phase": phase,
                    "template_id": template,
                    "closure_max_abs_error": decomposition["closure_max_abs_error"],
                    "interaction_rms_over_both_rms": decomposition["interaction_rms_over_both_rms"],
                    "additive_cosine_to_both_effect": decomposition["additive_cosine_to_both_effect"]})
    causal_zero = float(np.max(np.abs(arm_margins["temporal"]["01"] - arm_margins["temporal"]["00"])))
    counters = {"checkpoint_loads": 1, "model_forwards": 4, "sequence_evaluations": 512,
                "scored_token_positions": 1024, "transformer_backwards": 0,
                "model_updates": 0, "fit_parameters": 0}
    pairing_ok = all(pairs[role][pairs[role][i]] == i for role in pairs for i in range(128))
    position_ok = all(positions[role][i] == positions[role][pair] for role in pairs
                      for i, pair in enumerate(pairs[role]))
    capture_ok = set(captures) == {9, 11, 15} and all(value.shape == (128, maximum, 1152)
                                                           for value in captures.values())
    A = bool(authority and pairing_ok and position_ok and capture_ok and counters == PRICE
             and finite({"single": single_reports, "composition": composition_reports,
                         "causal_zero": causal_zero})
             and max(row["closure_max_abs_error"] for row in composition_reports) <= 1e-6)
    pooled = [row for row in single_reports if row["template_id"] == "ALL"]
    B = all(row["signed_recovery"] >= (.80 if row["role"] == "temporal" else .65)
            and row["cosine"] >= .98 and row["direction_agreement"] == 1.0
            and row["non_target_to_target_gold_norm"] <= .01 for row in pooled) and causal_zero <= 1e-5
    C = all(row["interaction_rms_over_both_rms"] <= .10
            and row["additive_cosine_to_both_effect"] >= .99 for row in composition_reports
            if row["template_id"] != "ALL")
    D = all(row["simultaneous_signed_recovery"] >= row["signed_recovery"] - .05 for row in pooled)
    E = A and B and C and D
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    terminal = "invalid" if not A else "h4_ood_joint_program_licensed" if E else "h4_ood_joint_null"
    result = {"schema": "temporal_iswas_h4_ood_joint_composition_result_v1",
        "candidate_id": prior["candidate_id"], "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "population_logical_sha256": candidate.EXPECTED_AUTHORITY_SHA256, "union": UNION,
        "iswas_to_temporal_max_abs_effect": causal_zero, "single_reports": single_reports,
        "composition_reports": composition_reports, "predictions": predictions,
        "terminal": terminal, "price": counters}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("iswas_to_temporal_max_abs_effect",
        "single_reports", "composition_reports", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
