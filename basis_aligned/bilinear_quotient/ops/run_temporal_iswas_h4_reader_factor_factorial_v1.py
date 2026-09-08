#!/usr/bin/env python3
"""Split causal H4 head swaps into exact L11 value and L15 query readers."""

# BQGATE: EXPERIMENT pred_a_authority_pairing_replay_factor_coverage_finiteness_and_exact_price pred_b_l11h3_value_prefix_explains_complete_head pred_c_l15h5_q_q2_split_explains_complete_head pred_d_weight_nominated_reader_union_explains_two_head_parent pred_e_reader_factor_transfer_is_task_selective_and_causal
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import dual_command_head_module_factorial_contract as accounting
import run_temporal_iswas_dual_command_shared_head_module_factorial_v1 as parent

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_h4_reader_factor_factorial_v1.json"
ATLAS = ROOT / "circuits/followups/temporal_iswas_common_gauge_weight_interface_atlas_v1_result.json"
HEAD_RESULT = ROOT / "circuits/followups/temporal_iswas_dual_command_shared_head_module_factorial_v1_result.json"
AUGMENTATION = ROOT / "circuits/followups/temporal_iswas_greedy_shared_writer_augmentation_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_h4_reader_factor_factorial_v1_result.json"
FILES = {
    "prior": PRIOR,
    "atlas": ATLAS,
    "head_result": HEAD_RESULT,
    "augmentation": AUGMENTATION,
    "parent_runner": ROOT / "ops/run_temporal_iswas_dual_command_shared_head_module_factorial_v1.py",
    "augmentation_runner": ROOT / "ops/run_temporal_iswas_greedy_shared_writer_augmentation_v1.py",
    "builder": ROOT / "ops/circuit_candidate_temporal_iswas_dual_command_v2.py",
    "accounting": ROOT / "ops/dual_command_head_module_factorial_contract.py",
    "producer": ROOT / "ops/circuit_fast_screen_producer.py",
}
EXPECTED = {
    "prior": "7fdbd875363a01996ec61c43ac6479dedf2c1483319c6b41a4b9d4921d00952d",
    "atlas": "d951420fef51f9d550237927072bfc721bd73112f87d5f716b3cb7c2ca3483ae",
    "head_result": "870c34d6e4d6b786eafe9bcdd960e790cab5ffb5b865b530b258997d405dbae2",
    "augmentation": "c81853c47e6ed2ecf40840a2951f67fe3f249d0f3f8b5ca5e6a4282e5ade4e4e",
    "parent_runner": "61cc52e73d51441512b073b75eef86f41ff5941c8f97770b7a062446c47ef849",
    "augmentation_runner": "961c2e968da2953133a29df5145241f53c1b570cd74debaebbbe54692a752a6a",
    "builder": "72da11860ce5bf1c03ea126bc10fcb6c46dd2648169edb00a1e9e1dbeee8a0d0",
    "accounting": "985b824c7d3d8b622dfb92e3522f107e15e5e095b1f40a59a46122ad9af68496",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
PRICE = {"checkpoint_loads": 1, "model_forwards": 19, "sequence_evaluations": 2432,
         "scored_token_positions": 4864, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
FACTORS = ((11, 3, "v"), (15, 5, "q"), (15, 5, "q2"))
ARMS = (
    "L11H3:head_output", "L11H3:v_prefix", "L15H5:head_output",
    "L15H5:q_query", "L15H5:q2_query", "L15H5:q_plus_q2_query",
    "L11H3_plus_L15H5:head_output", "L11H3:v_plus_L15H5:q_plus_q2",
    "H4_head_output_replay",
)
PREDICTION_KEYS = (
    "pred_a_authority_pairing_replay_factor_coverage_finiteness_and_exact_price",
    "pred_b_l11h3_value_prefix_explains_complete_head",
    "pred_c_l15h5_q_q2_split_explains_complete_head",
    "pred_d_weight_nominated_reader_union_explains_two_head_parent",
    "pred_e_reader_factor_transfer_is_task_selective_and_causal",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def factor_site(layer, head, factor): return f"L{layer}H{head}:{factor}"


def patch_factor_tensor(output, donor, pairs, positions, *, head, scope):
    """Return an absolute native-donor factor patch for one head."""
    if output.shape != donor.shape or output.ndim != 3:
        raise ValueError("factor tensors must share [batch, token, residual] shape")
    if scope not in ("query", "prefix"):
        raise ValueError("factor scope must be query or prefix")
    width = output.shape[-1] // 9
    if width * 9 != output.shape[-1] or not 0 <= head < 9:
        raise ValueError("invalid head geometry")
    changed = output.clone(); sl = slice(head * width, (head + 1) * width)
    for index, (pair, position) in enumerate(zip(pairs, positions)):
        pair, position = int(pair), int(position)
        if not 0 <= pair < output.shape[0] or not 0 <= position < output.shape[1]:
            raise ValueError("factor pair or position out of range")
        if scope == "query":
            changed[index, position, sl] = donor[pair, position, sl]
        else:
            changed[index, :position + 1, sl] = donor[pair, :position + 1, sl]
    return changed


def mobius_report(q, q2, both):
    interaction = both - q - q2
    denominator = max(float(np.linalg.norm(both)), 1e-30)
    additive = q + q2
    additive_norm = max(float(np.linalg.norm(additive)), 1e-30)
    return {
        "interaction_rms_over_both_rms": float(np.linalg.norm(interaction) / denominator),
        "additive_cosine_to_both_effect": float(np.dot(additive, both) /
            (additive_norm * denominator)),
        "interaction": interaction.tolist(),
    }


def _forward(backend, tokens, *, captures=None, pairs=None, positions=None,
             head_spec=None, factor_spec=(), capture=False):
    torch, F, model = backend.torch, backend.F, backend.model
    saved = {"heads": {}, "factors": {}}; handles = []; calls = {}
    if capture:
        for layer in (9, 11, 15):
            def save_head(_module, inputs, layer=layer):
                saved["heads"][layer] = inputs[0].detach().clone()
            handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(save_head))
        for layer, head, factor in FACTORS:
            name = factor_site(layer, head, factor); calls[name] = 0
            def save_factor(_module, _inputs, output, name=name):
                calls[name] += 1; saved["factors"][name] = output.detach().clone()
            handles.append(getattr(model.transformer.h[layer].attn, "c_" + factor).register_forward_hook(save_factor))
    if head_spec:
        if captures is None or pairs is None or positions is None: raise ValueError("head patch inputs absent")
        width = model.config.n_embd // model.config.n_head
        for layer, heads in head_spec.items():
            def patch_head(_module, inputs, layer=layer, heads=heads):
                value = inputs[0]; changed = value.clone(); donor = captures["heads"][layer]
                for index, position in enumerate(positions):
                    pair = pairs[index]
                    for head in heads:
                        sl = slice(head * width, (head + 1) * width)
                        changed[index, position, sl] = donor[pair, position, sl]
                return (changed,) + tuple(inputs[1:])
            handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(patch_head))
    for layer, head, factor, scope in factor_spec:
        if captures is None or pairs is None or positions is None: raise ValueError("factor patch inputs absent")
        name = factor_site(layer, head, factor); donor = captures["factors"][name]
        def patch_factor(_module, _inputs, output, donor=donor, pairs=pairs,
                         positions=positions, head=head, scope=scope):
            return patch_factor_tensor(output, donor, pairs, positions, head=head, scope=scope)
        handles.append(getattr(model.transformer.h[layer].attn, "c_" + factor).register_forward_hook(patch_factor))
    try:
        x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, first = x, None
        for block in model.transformer.h: x, first = block(x, first, x0)
        logits = (30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)).float()
    finally:
        for handle in handles: handle.remove()
    if capture and (set(saved["heads"]) != {9, 11, 15}
                    or set(saved["factors"]) != {factor_site(*site) for site in FACTORS}
                    or any(count != 1 for count in calls.values())):
        raise RuntimeError("factor/head capture coverage changed")
    return logits, saved


def arm_specs():
    return {
        "L11H3:head_output": ({11: (3,)}, ()),
        "L11H3:v_prefix": ({}, ((11, 3, "v", "prefix"),)),
        "L15H5:head_output": ({15: (5,)}, ()),
        "L15H5:q_query": ({}, ((15, 5, "q", "query"),)),
        "L15H5:q2_query": ({}, ((15, 5, "q2", "query"),)),
        "L15H5:q_plus_q2_query": ({}, ((15, 5, "q", "query"), (15, 5, "q2", "query"))),
        "L11H3_plus_L15H5:head_output": ({11: (3,), 15: (5,)}, ()),
        "L11H3:v_plus_L15H5:q_plus_q2": ({}, ((11, 3, "v", "prefix"),
            (15, 5, "q", "query"), (15, 5, "q2", "query"))),
        "H4_head_output_replay": ({9: (1, 4), 11: (3,), 15: (5,)}, ()),
    }


def main():
    observed = {name: sha(path) for name, path in FILES.items()}
    prior, atlas, head_result, augmentation = (json.loads(path.read_text()) for path in
        (PRIOR, ATLAS, HEAD_RESULT, AUGMENTATION))
    authority = bool(observed == EXPECTED and atlas.get("terminal") == "shared_weight_interface_candidates"
                     and head_result.get("terminal") == "shared_weight_causal_route_null"
                     and head_result.get("predictions", {}).get(
                         "pred_d_three_head_union_is_distributive_and_improves_recovery") is True
                     and augmentation.get("terminal") == "augmented_union_licensed"
                     and augmentation.get("selected_arm") == "H3_plus_L9H4")
    rows = parent.candidate.build_rows(); endpoints, lookup = parent.endpoint_bank(rows)
    dry = {"candidate_id": prior.get("candidate_id"), "dryrun": True, "authority_ok": authority,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "arms": list(ARMS), "factor_sites": [factor_site(*site) for site in FACTORS],
           "price": PRICE}
    if not authority: raise RuntimeError(f"reader-factor authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter(); started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    maximum = max(len(endpoint["ids"]) for _row, _cell, endpoint in endpoints)
    tokens = torch.tensor([endpoint["ids"] + [50256] * (maximum - len(endpoint["ids"]))
                           for _row, _cell, endpoint in endpoints], dtype=torch.long, device=backend.device)
    with torch.no_grad(): native_logits, captures = _forward(backend, tokens, capture=True)
    native = {role: parent.margins(native_logits, endpoints, role) for role in ("temporal", "iswas")}
    effects = {}; non_targets = {}; records = []; actual_forwards = 1
    specs = arm_specs()
    for role in ("temporal", "iswas"):
        pairs = parent.pair_indices(endpoints, lookup, role)
        positions = [endpoint[f"{role}_position"] - 1 for _row, _cell, endpoint in endpoints]
        if any(len(endpoints[i][2]["ids"]) != len(endpoints[pair][2]["ids"])
               or positions[i] != positions[pair] for i, pair in enumerate(pairs)):
            raise RuntimeError("paired sequence length/query position changed")
        for arm in ARMS:
            head_spec, factor_spec = specs[arm]
            with torch.no_grad(): logits, _ = _forward(backend, tokens, captures=captures,
                pairs=pairs, positions=positions, head_spec=head_spec, factor_spec=factor_spec)
            actual_forwards += 1
            margins = {name: parent.margins(logits, endpoints, name) for name in ("temporal", "iswas")}
            effects[(role, arm)] = margins[role] - native[role]
            other = "iswas" if role == "temporal" else "temporal"
            non_targets[(role, arm)] = margins[other] - native[other]
            gold = native[role][pairs] - native[role]
            for index, (row, cell, _endpoint) in enumerate(endpoints):
                records.append({"row_id": row["row_id"], "phase": row["phase"],
                    "template_id": row["template_id"], "cell": cell,
                    "paired_cell": accounting.paired_cell(cell, role), "role": role, "arm": arm,
                    "target_effect": float(effects[(role, arm)][index]),
                    "target_gold": float(gold[index]),
                    "non_target_effect": float(non_targets[(role, arm)][index])})
    reports = accounting.aggregate(records)

    parent_map = {
        "L11H3:v_prefix": "L11H3:head_output",
        "L15H5:q_query": "L15H5:head_output",
        "L15H5:q2_query": "L15H5:head_output",
        "L15H5:q_plus_q2_query": "L15H5:head_output",
        "L11H3:v_plus_L15H5:q_plus_q2": "L11H3_plus_L15H5:head_output",
    }
    parent_reports = []
    for role in ("temporal", "iswas"):
        for phase in ("FIT", "HOLDOUT"):
            for template in ("ALL",) + parent.candidate.TEMPLATES:
                selected = np.asarray([row["phase"] == phase and
                    (template == "ALL" or row["template_id"] == template)
                    for row, _cell, _endpoint in endpoints])
                for arm, parent_arm in parent_map.items():
                    metrics = accounting.effect_metrics(effects[(role, arm)][selected],
                        effects[(role, parent_arm)][selected], non_targets[(role, arm)][selected])
                    parent_reports.append({"role": role, "phase": phase, "template_id": template,
                        "arm": arm, "parent_arm": parent_arm, **metrics})
    q_factorials = []
    for role in ("temporal", "iswas"):
        for phase in ("FIT", "HOLDOUT"):
            for template in ("ALL",) + parent.candidate.TEMPLATES:
                selected = np.asarray([row["phase"] == phase and
                    (template == "ALL" or row["template_id"] == template)
                    for row, _cell, _endpoint in endpoints])
                q_factorials.append({"role": role, "phase": phase, "template_id": template,
                    **mobius_report(effects[(role, "L15H5:q_query")][selected],
                        effects[(role, "L15H5:q2_query")][selected],
                        effects[(role, "L15H5:q_plus_q2_query")][selected])})

    def pooled(rows_, role, phase, arm):
        return next(row for row in rows_ if row["role"] == role and row["phase"] == phase
                    and row["template_id"] == "ALL" and row["arm"] == arm)

    replay = []
    for role in ("temporal", "iswas"):
        for phase in ("FIT", "HOLDOUT"):
            for new_arm, source, old_arm in (
                ("L11H3:head_output", head_result, "head:L11H03"),
                ("L15H5:head_output", head_result, "head:L15H05"),
                ("H4_head_output_replay", augmentation, "H3_plus_L9H4")):
                new = pooled(reports, role, phase, new_arm)
                old = next(row for row in source["reports"] if row["role"] == role
                    and row["phase"] == phase and row["template_id"] == "ALL" and row["arm"] == old_arm)
                replay.extend(abs(new[key] - old[key]) for key in
                    ("signed_recovery", "cosine", "direction_agreement", "relative_residual",
                     "non_target_to_target_gold_norm"))
    replay_max = max(replay)
    causal_zero = max(abs(value) for (role, _arm), values in non_targets.items()
                      if role == "iswas" for value in values)
    counters = {"checkpoint_loads": 1, "model_forwards": actual_forwards,
                "sequence_evaluations": actual_forwards * len(endpoints),
                "scored_token_positions": actual_forwards * len(endpoints) * 2,
                "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0}
    capture_shapes = {name: list(value.shape) for name, value in captures["factors"].items()}
    A = bool(authority and counters == PRICE and replay_max <= 1e-5
             and set(capture_shapes) == {factor_site(*site) for site in FACTORS}
             and all(shape == [128, maximum, 1152] for shape in capture_shapes.values())
             and finite({"reports": reports, "parent_reports": parent_reports,
                         "q_factorials": q_factorials, "causal_zero": causal_zero}))
    parent_ok = lambda role, phase, arm: (lambda row: row["signed_recovery"] >= .50
        and row["cosine"] >= .80 and row["direction_agreement"] >= .75)(
            pooled(parent_reports, role, phase, arm))
    B = all(parent_ok(role, phase, "L11H3:v_prefix")
            for role in ("temporal", "iswas") for phase in ("FIT", "HOLDOUT"))
    C = True
    for role in ("temporal", "iswas"):
        for phase in ("FIT", "HOLDOUT"):
            q = pooled(parent_reports, role, phase, "L15H5:q_query")
            q2 = pooled(parent_reports, role, phase, "L15H5:q2_query")
            both = pooled(parent_reports, role, phase, "L15H5:q_plus_q2_query")
            interaction = next(row for row in q_factorials if row["role"] == role
                and row["phase"] == phase and row["template_id"] == "ALL")
            C = C and q["signed_recovery"] >= .10 and q2["signed_recovery"] >= .10
            C = C and both["signed_recovery"] >= .50 and both["cosine"] >= .80
            C = C and both["direction_agreement"] >= .75
            C = C and both["signed_recovery"] >= max(q["signed_recovery"], q2["signed_recovery"]) + .05
            C = C and interaction["interaction_rms_over_both_rms"] <= .35
    D = all((lambda row: row["signed_recovery"] >= .60 and row["cosine"] >= .90
             and row["direction_agreement"] >= .875)(pooled(parent_reports, role, phase,
                "L11H3:v_plus_L15H5:q_plus_q2"))
            for role in ("temporal", "iswas") for phase in ("FIT", "HOLDOUT"))
    E = causal_zero <= 1e-5 and all(pooled(reports, role, phase, arm)[
        "non_target_to_target_gold_norm"] <= .01 for role in ("temporal", "iswas")
        for phase in ("FIT", "HOLDOUT") for arm in ARMS if "head_output" not in arm)
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D, E))))
    if not A: terminal = "invalid"
    elif B and C and D and E: terminal = "h4_reader_factors_licensed"
    elif B or C: terminal = "partial_reader_factor_split"
    else: terminal = "weight_reader_factor_null_keep_h4_heads"
    result = {"schema": "temporal_iswas_h4_reader_factor_factorial_result_v1",
              "candidate_id": prior["candidate_id"], "started_utc": started_utc,
              "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
              "capture_shapes": capture_shapes, "replay_max_abs_error": replay_max,
              "iswas_to_temporal_max_abs_effect": causal_zero, "reports": reports,
              "parent_relative_reports": parent_reports, "q_q2_factorials": q_factorials,
              "predictions": predictions, "terminal": terminal, "price": counters}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("replay_max_abs_error",
        "iswas_to_temporal_max_abs_effect", "parent_relative_reports", "q_q2_factorials",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
