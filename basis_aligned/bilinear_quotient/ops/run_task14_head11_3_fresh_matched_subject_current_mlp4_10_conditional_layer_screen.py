#!/usr/bin/env python3
"""Conditional per-layer Task14 screen inside upstream MLP4--10."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_at_least_one_standalone_layer pred_c_at_least_one_conditional_layer pred_d_same_layer_is_stable pred_e_context_dependence pred_f_number_specific pred_g_lexical_collateral

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_fresh_matched_natural_native_capability as capability
import run_task14_head11_3_fresh_matched_subject_current_mlp_depth_factorial as depth
import run_task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v2 as value_v2
import run_task14_head11_3_subject_attractor_score_payload_factorial as factors
import attention_source_factor_primitive as source_factor


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen_v1.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_matched_subject_current_mlp4_10_conditional_layer_screen_v1_capability_license.json"
PARENT_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_mlp_depth_factorial_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen_v1_result.json"
PRIOR_ART_SHA256 = "de8d1dc9ba4fe13b200540ee1df43f43d20a80c89361c1c61e0eb5903905b312"
LICENSE_SHA256 = "672c7b309d5ed623d141d2bd6e1673ad7055133f715f1cb192acad34ca2a769c"
PARENT_RESULT_SHA256 = "f18ce285152a1b3388ca3083de344dd9dd3675e4932b27bf44ea1f9e8e3f3073"
CANDIDATE_ID = "subject_verb.number_agreement.head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen_v1"
LAYER, HEAD, SELF_POSITION = depth.LAYER, depth.HEAD, depth.SELF_POSITION
LAYERS = tuple(range(4, 11))
CONDITIONS = (
    "recipient_M4_10",
    *(f"opposite_M{layer}" for layer in LAYERS),
    *(f"opposite_except_M{layer}" for layer in LAYERS),
    "opposite_full_M4_10",
    *(f"lexical_M{layer}" for layer in LAYERS),
    "lexical_full_M4_10",
)
BARS = {
    "maximum_native_replay_absolute_logit_error": 7e-5,
    "maximum_state_sum_absolute_error": 5e-5,
    "maximum_normalized_state_absolute_error": 5e-5,
    "maximum_source_term_sum_absolute_error": 5e-5,
    "maximum_full_donor_current_head_absolute_error": 5e-5,
    "maximum_same_batch_native_noop_endpoint_error": 7e-5,
    "maximum_installed_head_absolute_error": 5e-5,
    "maximum_recipient_high_group_absolute_error": 5e-5,
    "maximum_donor_high_group_absolute_error": 5e-5,
    "minimum_full_donor_mean_target_margin_improvement": .05,
    "minimum_full_donor_mean_target_CE_improvement": 0.0,
    "minimum_helpful_row_fraction": .75,
    "minimum_layer_recovery_fraction": .25,
    "minimum_context_difference_fraction": .25,
    "maximum_number_specific_lexical_ratio": .25,
    "minimum_lexical_collateral_ratio": .50,
}


class ConditionalLayerScreenError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    return depth.build_rows()


def validate_preflight():
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "prior-art receipt"),
        (PARENT_RESULT, PARENT_RESULT_SHA256, "parent result"),
    ):
        if _sha256(path) != expected:
            raise ConditionalLayerScreenError(f"{label} changed")
    result = json.loads(PARENT_RESULT.read_text())
    parent_keys = ["pred" + suffix for suffix in (
        "_a_instrument_live", "_b_G0_carries_task", "_c_G1_carries_task",
        "_d_G2_carries_task", "_e_distributed_across_depth_groups",
        "_f_interaction_is_needed", "_g_number_specific", "_h_lexical_collateral")]
    parent_values = (True, False, False, False, False, False, True, False)
    if result.get("terminal") != "valid_causal_screen" \
            or result.get("score", {}).get("predictions") != dict(zip(parent_keys, parent_values)):
        raise ConditionalLayerScreenError("parent result no longer licenses conditional layer screen")
    return licensing.validate_causal_preflight(
        capability.build_gate(), capability.CAPABILITY_RESULT, LICENSE,
        expected_license_sha256=LICENSE_SHA256, causal_candidate_id=CANDIDATE_ID)


def compile_plan():
    license_value = validate_preflight()
    return {
        "schema": "task14_head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen_plan_v1",
        "candidate_id": CANDIDATE_ID, "split": "LICENSED_HOLDOUT",
        "row_count": len(build_rows()), "conditions": list(CONDITIONS),
        "layers": list(LAYERS), "prior_art_sha256": PRIOR_ART_SHA256,
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "license_sha256": LICENSE_SHA256,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "preflight_validated": True, "bars": dict(BARS),
        "partition": "exact propagated MLP4--10 slots; singleton and full-minus-leave-one effects",
        "remainder_rule": "parent MR remains with fixed recipient MLP0--3; high rho_H follows MLP4 and is added last",
        "fixed_context": "recipient E+A+R, MLP0--3+MR, p_8, cached value, and non-subject complement",
        "layer_boundary_limit": "native MLP layers are localization handles, not final semantic units",
        "price": {"model_forwards": 4, "example_evaluations": 864,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["answer_directed_target_margin_improvement",
                     "target_full_vocab_CE_improvement"],
        "closed_claims": ["within_MLP_semantic_basis", "necessity", "syntax_generality",
                          "FIT", "downstream_reader", "rank", "reconstruction"],
    }


def _decomposed_forward(model, tokens, finals, torch, F, facade, *,
                        replacement_heads=None, native_reinstall_mask=None):
    x0 = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
    embedding = x0.clone()
    attention_sum = torch.zeros_like(x0)
    mlp_sum = torch.zeros_like(x0)
    slots = [torch.zeros_like(x0) for _ in range(LAYER)]
    reference = x0.clone()
    captured = {}
    projection = None
    expected_attention_site = expected_mlp_site = 0

    def attention(event):
        nonlocal embedding, attention_sum, mlp_sum, reference, projection
        nonlocal expected_attention_site
        if event.site != expected_attention_site:
            raise ConditionalLayerScreenError("attention sites are not sequential")
        expected_attention_site += 1
        residual_scale, skip_scale = event.block.lambdas[0], event.block.lambdas[1]
        reference = residual_scale * reference + skip_scale * x0
        embedding = residual_scale * embedding + skip_scale * x0
        attention_sum = residual_scale * attention_sum
        mlp_sum = residual_scale * mlp_sum
        for index in range(min(event.site, LAYER)):
            slots[index] = residual_scale * slots[index]
        rebuilt = embedding + attention_sum + mlp_sum
        if event.site == LAYER:
            write, base = source_factor.replay_attention_with_source_factors(
                event.state, event.first_value, event.block.attn, finals, HEAD, torch, F)
            current, cached, effective, projection = value_v2._raw_value_branches(
                event.state, event.first_value, event.block.attn, torch, F)
            g0 = sum((slots[i] for i in range(4)), start=torch.zeros_like(x0))
            g1 = sum((slots[i] for i in range(4, 8)), start=torch.zeros_like(x0))
            g2 = sum((slots[i] for i in range(8, 11)), start=torch.zeros_like(x0))
            parent_mr = mlp_sum - ((g0 + g1) + g2)
            high = g1 + g2
            slot_high = sum((slots[i] for i in LAYERS), start=torch.zeros_like(x0))
            captured.update({name: value.detach().clone() for name, value in base.items()})
            captured.update({
                "E": embedding.detach().clone(), "A": attention_sum.detach().clone(),
                "M": mlp_sum.detach().clone(), "M0_3": g0.detach().clone(),
                "MR": parent_mr.detach().clone(), "H": high.detach().clone(),
                "HR": (high - slot_high).detach().clone(),
                "G1": g1.detach().clone(), "G2": g2.detach().clone(),
                "R": (reference - rebuilt).detach().clone(),
                "raw_state": reference.detach().clone(),
                "normalized_state": event.state.detach().clone(),
                "current_pre": current.detach().clone(), "cached_pre": cached.detach().clone(),
                "effective_pre": effective.detach().clone(),
                **{f"M{i}": slots[i].detach().clone() for i in LAYERS},
            })
            if replacement_heads is not None:
                rows = torch.arange(tokens.size(0), device=tokens.device)
                installed = factors._same_batch_native_heads(
                    replacement_heads, base["head"], native_reinstall_mask, torch)
                write[rows, finals] += (installed - base["head"]).to(write.dtype)
            next_first_value = event.first_value
        else:
            write, next_first_value = event.block.attn(event.state, event.first_value)
        attention_sum = attention_sum + write
        reference = reference + write
        return write, next_first_value

    def mlp(event):
        nonlocal mlp_sum, reference, expected_mlp_site
        if event.site != expected_mlp_site:
            raise ConditionalLayerScreenError("MLP sites are not sequential")
        expected_mlp_site += 1
        write = event.block.mlp(event.state)
        mlp_sum = mlp_sum + write
        reference = reference + write
        if event.site < LAYER:
            slots[event.site] = slots[event.site] + write
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=False).float()
    required = {"p", "u", "head", "E", "A", "M", "M0_3", "MR", "H", "HR",
                "G1", "G2", "R", "raw_state", "normalized_state", "current_pre",
                "cached_pre", "effective_pre", *(f"M{i}" for i in LAYERS)}
    if set(captured) != required or projection is None \
            or expected_attention_site != 18 or expected_mlp_site != 18:
        raise ConditionalLayerScreenError("decomposed forward audit failed")
    corrected = (captured["E"] + captured["A"] + captured["M"]) + captured["R"]
    return logits, captured, projection.detach().clone(), {
        "state_sum_max_absolute_error": float((corrected - captured["raw_state"]).abs().max()),
        "normalized_state_max_absolute_error": float((
            F.rms_norm(corrected, (corrected.size(-1),)) - captured["normalized_state"]
        ).abs().max()),
    }


def _high_from_choices(roles, choices):
    high = sum((roles[role][f"M{layer}"] for role, layer in zip(choices, LAYERS)),
               start=roles["r"]["M4"].new_zeros(roles["r"]["M4"].shape))
    return high + roles[choices[0]]["HR"]


def _current_from_high(recipient, high, attention, torch, F):
    grouped_M = (recipient["M0_3"] + high) + recipient["MR"]
    state = (recipient["E"] + recipient["A"] + grouped_M) + recipient["R"]
    normalized = F.rms_norm(state, (state.size(-1),))
    batch, length, width = normalized.shape
    raw = F.linear(normalized, attention.c_v.weight.to(normalized.dtype)).view(
        batch, length, 9, width // 9)
    return (1 - attention.lamb) * raw[:, :, HEAD]


def _condition_choices():
    choices = {"recipient_M4_10": "r" * len(LAYERS),
               "opposite_full_M4_10": "o" * len(LAYERS),
               "lexical_full_M4_10": "l" * len(LAYERS)}
    for index, layer in enumerate(LAYERS):
        choices[f"opposite_M{layer}"] = "r" * index + "o" + "r" * (len(LAYERS)-index-1)
        choices[f"opposite_except_M{layer}"] = "o" * index + "r" + "o" * (len(LAYERS)-index-1)
        choices[f"lexical_M{layer}"] = "r" * index + "l" + "r" * (len(LAYERS)-index-1)
    return choices


def _compile(recipient_tokens, recipient, opposite, lexical, attention, projection,
             rows, torch, F):
    recipient_terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    mask = torch.arange(recipient_terms.shape[1], device=recipient_terms.device) != SELF_POSITION
    complement = recipient_terms[:, mask].sum(1)
    native_p = recipient["p"][:, SELF_POSITION].unsqueeze(-1)
    roles = {"r": recipient, "o": opposite, "l": lexical}
    heads = {}
    for condition, choices in _condition_choices().items():
        current = _current_from_high(
            recipient, _high_from_choices(roles, choices), attention, torch, F)
        value = value_v2._project_once(current, recipient["cached_pre"], projection, F)
        heads[condition] = complement + native_p * value[:, SELF_POSITION]
    indices, replacements, specs, reinstall = [], [], [], []
    for row_index, row in enumerate(rows):
        for condition in CONDITIONS:
            indices.append(row_index); replacements.append(heads[condition][row_index])
            specs.append((row_index, condition, f"{row['direction_id']}__{row['template_id']}"))
            reinstall.append(condition == "recipient_M4_10")
    index = torch.tensor(indices, dtype=torch.long, device=recipient_tokens.device)
    return {
        "tokens": recipient_tokens[index], "finals": torch.full_like(index, SELF_POSITION),
        "replacement_heads": torch.stack(replacements),
        "native_reinstall_mask": torch.tensor(reinstall, dtype=torch.bool,
                                               device=recipient_tokens.device),
        "specs": specs, "heads": heads,
    }


def _metrics(logits, row, condition, torch):
    role = "recipient" if condition == "recipient_M4_10" or condition.startswith("lexical_") \
        else "opposite_same_lemma"
    target = int(row["endpoints"][role]["answer_id"])
    foil = int(row["endpoints"][role]["foil_id"])
    lp = torch.log_softmax(logits, dim=-1)
    return {"target_margin": float(logits[target] - logits[foil]),
            "target_CE": float(-lp[target])}


def evaluate(model, torch, F, facade):
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = depth.parent.v1._role_batch(rows, torch, device)
    native = factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, closure = _decomposed_forward(
        model, tokens, finals, torch, F, facade)
    recipient = {key: value[:n] for key, value in captured.items()}
    opposite = {key: value[n:2*n] for key, value in captured.items()}
    lexical = {key: value[2*n:] for key, value in captured.items()}
    attention = model.transformer.h[LAYER].attn
    patch = _compile(tokens[:n], recipient, opposite, lexical, attention, projection,
                     rows, torch, F)
    native_patch = factors._native_logits(model, patch["tokens"], torch, F)
    patched, _, _, patch_closure = _decomposed_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])
    direct_current = depth._current_from_groups(
        recipient["E"], recipient["A"], recipient["R"], recipient["M0_3"],
        opposite["G1"], opposite["G2"], recipient["MR"], attention, torch, F)
    direct_value = value_v2._project_once(
        direct_current, recipient["cached_pre"], projection, F)
    recipient_terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    mask = torch.arange(recipient_terms.shape[1], device=device) != SELF_POSITION
    direct_head = recipient_terms[:, mask].sum(1) + \
        recipient["p"][:, SELF_POSITION].unsqueeze(-1) * direct_value[:, SELF_POSITION]
    roles = {"r": recipient, "o": opposite, "l": lexical}
    exactness = {
        "native_replay_max_absolute_logit_error": float((native - replay).abs().max()),
        "state_sum_max_absolute_error": max(
            closure["state_sum_max_absolute_error"], patch_closure["state_sum_max_absolute_error"]),
        "normalized_state_max_absolute_error": max(
            closure["normalized_state_max_absolute_error"],
            patch_closure["normalized_state_max_absolute_error"]),
        "source_term_sum_max_absolute_error": max(float((
            torch.einsum("bk,bkd->bd", role["p"], role["u"]) - role["head"]
        ).abs().max()) for role in (recipient, opposite, lexical)),
        "full_donor_current_head_max_absolute_error": float((
            patch["heads"]["opposite_full_M4_10"] - direct_head).abs().max()),
        "same_batch_native_noop_endpoint_max_absolute_error": 0.0,
        "installed_head_max_absolute_error": 0.0,
        "recipient_high_group_max_absolute_error": float((
            _high_from_choices(roles, "r" * len(LAYERS)) - recipient["H"]).abs().max()),
        "donor_high_group_max_absolute_error": float((
            _high_from_choices(roles, "o" * len(LAYERS)) - opposite["H"]).abs().max()),
    }
    evidence = []
    for out_index, (row_index, condition, cell_id) in enumerate(patch["specs"]):
        base = _metrics(native_patch[out_index, SELF_POSITION], rows[row_index], condition, torch)
        value = _metrics(patched[out_index, SELF_POSITION], rows[row_index], condition, torch)
        evidence.append({
            "row_id": rows[row_index]["row_id"], "cell_id": cell_id,
            "condition": condition,
            "target_margin_improvement": value["target_margin"] - base["target_margin"],
            "target_CE_improvement": base["target_CE"] - value["target_CE"],
        })
        exactness["installed_head_max_absolute_error"] = max(
            exactness["installed_head_max_absolute_error"], float((
                patch["replacement_heads"][out_index] - patch["heads"][condition][row_index]
            ).abs().max()))
        if condition == "recipient_M4_10":
            exactness["same_batch_native_noop_endpoint_max_absolute_error"] = max(
                exactness["same_batch_native_noop_endpoint_max_absolute_error"],
                float((patched[out_index] - native_patch[out_index]).abs().max()))
    return evidence, exactness


def _positive_fraction(values):
    return sum(value > 0 for value in values) / len(values)


def score(evidence, exactness, bars=BARS):
    expected = {(row["row_id"], f"{row['direction_id']}__{row['template_id']}", condition)
                for row in build_rows() for condition in CONDITIONS}
    observed = [(x.get("row_id"), x.get("cell_id"), x.get("condition")) for x in evidence]
    if len(observed) != len(expected) or set(observed) != expected \
            or len(set(observed)) != len(expected):
        raise ConditionalLayerScreenError("evidence does not cover exact licensed screen")
    if any(type(x.get(key)) not in (int, float) or not math.isfinite(float(x[key]))
           for x in evidence for key in ("target_margin_improvement", "target_CE_improvement")):
        raise ConditionalLayerScreenError("non-finite or missing task metric")
    grouped = defaultdict(dict)
    for item in evidence:
        grouped[item["cell_id"]].setdefault(item["condition"], []).append(item)
    cells = {}
    for cell_id, conditions in sorted(grouped.items()):
        summaries = {}
        for condition in CONDITIONS:
            margins = [float(x["target_margin_improvement"]) for x in conditions[condition]]
            ces = [float(x["target_CE_improvement"]) for x in conditions[condition]]
            summaries[condition] = {
                "mean_margin": statistics.fmean(margins), "mean_CE": statistics.fmean(ces),
                "margin_values": margins, "CE_values": ces,
            }
        full = summaries["opposite_full_M4_10"]
        den = {"margin": max(abs(full["mean_margin"]), 1e-12),
               "CE": max(abs(full["mean_CE"]), 1e-12)}
        layers = {}
        for layer in LAYERS:
            standalone = summaries[f"opposite_M{layer}"]
            leave = summaries[f"opposite_except_M{layer}"]
            conditional = {
                "margin_values": [a-b for a, b in zip(full["margin_values"], leave["margin_values"])],
                "CE_values": [a-b for a, b in zip(full["CE_values"], leave["CE_values"])],
            }
            conditional["mean_margin"] = statistics.fmean(conditional["margin_values"])
            conditional["mean_CE"] = statistics.fmean(conditional["CE_values"])
            standalone_recovery = {"margin": standalone["mean_margin"] / den["margin"],
                                   "CE": standalone["mean_CE"] / den["CE"]}
            conditional_recovery = {"margin": conditional["mean_margin"] / den["margin"],
                                    "CE": conditional["mean_CE"] / den["CE"]}
            layers[str(layer)] = {
                "standalone": standalone,
                "conditional_full_minus_leave_one": conditional,
                "standalone_recovery": standalone_recovery,
                "conditional_recovery": conditional_recovery,
                "context_difference": {
                    metric: conditional_recovery[metric] - standalone_recovery[metric]
                    for metric in ("margin", "CE")
                },
            }
        lexical_conditions = [*(f"lexical_M{layer}" for layer in LAYERS),
                              "lexical_full_M4_10"]
        summaries["derived"] = {
            "layers": layers,
            "maximum_lexical_ratio": {
                "margin": max(abs(summaries[x]["mean_margin"]) / den["margin"]
                              for x in lexical_conditions),
                "CE": max(abs(summaries[x]["mean_CE"]) / den["CE"]
                          for x in lexical_conditions),
            },
        }
        cells[cell_id] = summaries

    exact_live = all(exactness[name] <= bars[bar] for name, bar in (
        ("native_replay_max_absolute_logit_error", "maximum_native_replay_absolute_logit_error"),
        ("state_sum_max_absolute_error", "maximum_state_sum_absolute_error"),
        ("normalized_state_max_absolute_error", "maximum_normalized_state_absolute_error"),
        ("source_term_sum_max_absolute_error", "maximum_source_term_sum_absolute_error"),
        ("full_donor_current_head_max_absolute_error", "maximum_full_donor_current_head_absolute_error"),
        ("same_batch_native_noop_endpoint_max_absolute_error", "maximum_same_batch_native_noop_endpoint_error"),
        ("installed_head_max_absolute_error", "maximum_installed_head_absolute_error"),
        ("recipient_high_group_max_absolute_error", "maximum_recipient_high_group_absolute_error"),
        ("donor_high_group_max_absolute_error", "maximum_donor_high_group_absolute_error"),
    ))

    def full_helpful(summary):
        return summary["mean_margin"] >= bars["minimum_full_donor_mean_target_margin_improvement"] \
            and summary["mean_CE"] >= bars["minimum_full_donor_mean_target_CE_improvement"] \
            and _positive_fraction(summary["margin_values"]) >= bars["minimum_helpful_row_fraction"] \
            and _positive_fraction(summary["CE_values"]) >= bars["minimum_helpful_row_fraction"]

    instrument = exact_live and all(full_helpful(cell["opposite_full_M4_10"])
                                    for cell in cells.values())

    def layer_pass(layer, mode):
        return instrument and all(
            min(cell["derived"]["layers"][str(layer)][f"{mode}_recovery"].values())
                >= bars["minimum_layer_recovery_fraction"]
            and _positive_fraction(cell["derived"]["layers"][str(layer)][
                "standalone" if mode == "standalone" else "conditional_full_minus_leave_one"
            ]["margin_values"]) >= bars["minimum_helpful_row_fraction"]
            and _positive_fraction(cell["derived"]["layers"][str(layer)][
                "standalone" if mode == "standalone" else "conditional_full_minus_leave_one"
            ]["CE_values"]) >= bars["minimum_helpful_row_fraction"]
            for cell in cells.values())

    standalone = {layer: layer_pass(layer, "standalone") for layer in LAYERS}
    conditional = {layer: layer_pass(layer, "conditional") for layer in LAYERS}
    context_dependent = instrument and any(all(
        max(abs(value) for value in cell["derived"]["layers"][str(layer)][
            "context_difference"].values()) >= bars["minimum_context_difference_fraction"]
        for cell in cells.values()) for layer in LAYERS)
    number_specific = instrument and all(
        max(cell["derived"]["maximum_lexical_ratio"].values())
            <= bars["maximum_number_specific_lexical_ratio"] for cell in cells.values())
    collateral = instrument and any(
        max(cell["derived"]["maximum_lexical_ratio"].values())
            >= bars["minimum_lexical_collateral_ratio"] for cell in cells.values())
    return {**exactness, "cells": cells, "predictions": {
        "pred_a_instrument_live": bool(instrument),
        "pred_b_at_least_one_standalone_layer": bool(any(standalone.values())),
        "pred_c_at_least_one_conditional_layer": bool(any(conditional.values())),
        "pred_d_same_layer_is_stable": bool(any(
            standalone[layer] and conditional[layer] for layer in LAYERS)),
        "pred_e_context_dependence": bool(context_dependent),
        "pred_f_number_specific": bool(number_specific),
        "pred_g_lexical_collateral": bool(collateral),
    }, "layer_predictions": {
        str(layer): {"standalone": standalone[layer], "conditional": conditional[layer]}
        for layer in LAYERS
    }}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True))
        return
    if OUT.exists():
        raise ConditionalLayerScreenError(f"refusing to overwrite {OUT}")
    torch, F, facade = factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    terminal = "valid_causal_screen" if scored["predictions"]["pred_a_instrument_live"] else "invalid"
    result = {
        "schema": "task14_head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen_result_v1",
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256, "score": scored,
        "evidence": evidence, "evaluated_splits": ["LICENSED_HOLDOUT"],
        "forbidden_splits_opened": [], "model_forwards": 4,
        "causal_interventions": len(evidence),
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
