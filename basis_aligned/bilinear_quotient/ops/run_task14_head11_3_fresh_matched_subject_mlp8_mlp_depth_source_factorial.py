#!/usr/bin/env python3
"""Exact E/A/U/V source factorial for MLP8's Task14 response."""

# BQGATE: EXPERIMENT pred_a_instrument_and_parent_closure pred_b_V_late_dominant pred_c_U_early_dominant pred_d_distributed_depth pred_e_cross_depth_composition pred_f_direction_switch pred_g_number_specific

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import attention_source_factor_primitive as source_factor
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_fresh_matched_natural_native_capability as capability
import run_task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v2 as value_v2
import run_task14_head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen as downstream
import run_task14_head11_3_fresh_matched_subject_mlp8_input_writer_response_factorial as parent
import run_task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial as polarized_v1
import run_task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v2 as polarized_v2
import run_task14_head11_3_subject_attractor_score_payload_factorial as factors


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_fresh_matched_subject_mlp8_mlp_depth_source_factorial_v1.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_matched_subject_mlp8_mlp_depth_source_factorial_v1_capability_license.json"
PARENT_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_input_writer_response_factorial_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_mlp_depth_source_factorial_v1_result.json"
PRIOR_ART_SHA256 = "03304b189bd8d1c98dc007543dd9f64253a91d23a3808546f8c13949ca8e1bbb"
LICENSE_SHA256 = "b69c9548dc80342d9e374ea4d627630e5558012c6b98c392e757ac11a152e09e"
PARENT_RESULT_SHA256 = "da639bb23aef25b78da20170b52e31e4e2d5f64a95fd5586b2bf07a12cb1a7ed"
CANDIDATE_ID = "subject_verb.number_agreement.head11_3_fresh_matched_subject_mlp8_mlp_depth_source_factorial_v1"
LAYER, HEAD = parent.LAYER, parent.HEAD
SUBJECT_POSITION, MLP_LAYER = parent.SUBJECT_POSITION, parent.MLP_LAYER
FAMILIES = ("E", "A", "U", "V")
SUBSETS = tuple("".join(parts) for size in range(1, 5)
                for parts in combinations(FAMILIES, size))
COMPONENTS, SOURCES = parent.COMPONENTS, parent.SOURCES
CONDITIONS = ("recipient",) + tuple(
    f"{source}_{subset}_{component}"
    for source in SOURCES for subset in SUBSETS for component in COMPONENTS)
PARENT_TO_CHILD = {"M": "UV", "EM": "EUV", "AM": "AUV", "EAM": "EAUV"}
PARENT_CORNERS = {"E": "E", "A": "A", "EA": "EA", **PARENT_TO_CHILD}
PARENT_AGGREGATES = tuple(PARENT_TO_CHILD)
BARS = {
    **parent.BARS,
    "maximum_M_grouping_closure_absolute_error": 5e-5,
    "maximum_parent_raw_state_absolute_error": 5e-5,
    "maximum_parent_normalized_input_absolute_error": 5e-5,
    "maximum_parent_MLP8_output_absolute_error": 5e-5,
    "maximum_parent_propagated_slot_absolute_error": 5e-5,
    "maximum_parent_installed_head_absolute_error": 5e-5,
    "maximum_parent_downstream_outcome_absolute_error": 5e-5,
    "minimum_dominant_recovery_fraction": .70,
    "maximum_minor_recovery_fraction": .25,
    "minimum_distributed_recovery_fraction": .25,
    "minimum_cross_depth_interaction_fraction": .25,
    "maximum_number_specific_lexical_ratio": .25,
}


class MLP8MLPDepthSourceError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    return parent.build_rows()


def validate_preflight():
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "prior-art receipt"),
        (PARENT_RESULT, PARENT_RESULT_SHA256, "parent E/A/M result"),
    ):
        if _sha256(path) != expected:
            raise MLP8MLPDepthSourceError(f"{label} changed")
    parent_result = json.loads(PARENT_RESULT.read_text())
    predictions = parent_result.get("score", {}).get("predictions", {})
    if parent_result.get("terminal") != "valid_causal_screen" \
            or predictions.get("pred_a_instrument_live") is not True \
            or predictions.get("pred_f_source_interaction_needed") is not True \
            or predictions.get("pred_i_number_specific") is not True:
        raise MLP8MLPDepthSourceError("parent result no longer licenses depth split")
    return licensing.validate_causal_preflight(
        capability.build_gate(), capability.CAPABILITY_RESULT, LICENSE,
        expected_license_sha256=LICENSE_SHA256, causal_candidate_id=CANDIDATE_ID)


def compile_plan():
    license_value = validate_preflight()
    return {
        "schema": "task14_head11_3_fresh_matched_subject_mlp8_mlp_depth_source_factorial_plan_v1",
        "candidate_id": CANDIDATE_ID, "split": "LICENSED_HOLDOUT",
        "row_count": len(build_rows()), "conditions": list(CONDITIONS),
        "condition_count": len(CONDITIONS), "subject_position": SUBJECT_POSITION,
        "mlp_layer": MLP_LAYER, "prior_art_sha256": PRIOR_ART_SHA256,
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "license_sha256": LICENSE_SHA256,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "preflight_validated": True, "bars": dict(BARS),
        "input_partition": {
            "E": "propagated embedding/skip through block 8",
            "A": "propagated attention writes A0--A8",
            "U": "propagated MLP writes M0--M3; the float32 M regrouping remainder follows U and is added last",
            "V": "propagated MLP writes M4--M7",
            "epsilon": "the parent raw-state remainder follows E",
        },
        "response_partition": "exact cross, quadratic, and full MLP8 responses at all 15 nonempty E/A/U/V donor corners",
        "parent_closure": "U+V corners reproduce parent M, EM, AM, and EAM at every registered numerical and task endpoint",
        "causal_statistics": "complete 2^4 Moebius decomposition of each task-level set function",
        "downstream_background": "standalone fixed L11H3 interface; every other MLP4--10 slot remains recipient",
        "price": {"model_forwards": 4, "example_evaluations": 3008,
                  "causal_interventions": 1440, "backwards": 0,
                  "parameter_updates": 0, "capability_GPU_price": 0},
        "outcomes": ["answer_directed_target_margin_improvement",
                     "target_full_vocab_CE_improvement"],
        "closed_claims": ["unique_semantic_basis", "individual_MLP_identity",
                          "rank", "quantization", "activation_reconstruction",
                          "new_independent_data", "necessity_outside_fixed_L11H3_interface"],
    }


def _decomposed_forward(model, tokens, finals, torch, F, facade, *,
                        replacement_heads=None, native_reinstall_mask=None):
    """Replay while tracking exact MLP0--3 and MLP4--7 writes into MLP8."""
    x0 = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,))
    embedding = x0.clone()
    attention_sum = torch.zeros_like(x0)
    mlp_sum = torch.zeros_like(x0)
    slots = [torch.zeros_like(x0) for _ in range(LAYER)]
    reference = x0.clone()
    captured, mlp8 = {}, {}
    projection = None
    expected_attention_site = expected_mlp_site = 0

    def attention(event):
        nonlocal embedding, attention_sum, mlp_sum, reference, projection
        nonlocal expected_attention_site
        if event.site != expected_attention_site:
            raise MLP8MLPDepthSourceError("attention sites are not sequential")
        expected_attention_site += 1
        residual_scale, skip_scale = event.block.lambdas[0], event.block.lambdas[1]
        reference = residual_scale * reference + skip_scale * x0
        embedding = residual_scale * embedding + skip_scale * x0
        attention_sum = residual_scale * attention_sum
        mlp_sum = residual_scale * mlp_sum
        for index in range(min(event.site, LAYER)):
            slots[index] = residual_scale * slots[index]
        if event.site == LAYER:
            write, base = source_factor.replay_attention_with_source_factors(
                event.state, event.first_value, event.block.attn, finals, HEAD, torch, F)
            current, cached, effective, projection = value_v2._raw_value_branches(
                event.state, event.first_value, event.block.attn, torch, F)
            # Match the validated parent replay's float32 grouping exactly.  The
            # downstream slot installer consumes HR to retain the numerical
            # difference between the grouped and sequential MLP4--10 sums.
            middle = sum((slots[i] for i in range(4, 8)),
                         start=torch.zeros_like(x0))
            late = sum((slots[i] for i in range(8, 11)),
                       start=torch.zeros_like(x0))
            high = middle + late
            slot_high = sum((slots[i] for i in downstream.LAYERS),
                            start=torch.zeros_like(x0))
            captured.update({name: value.detach().clone() for name, value in base.items()})
            captured.update({
                "E": embedding.detach().clone(), "A": attention_sum.detach().clone(),
                "M": mlp_sum.detach().clone(),
                "H": high.detach().clone(),
                "HR": (high - slot_high).detach().clone(),
                "R": (reference - (embedding + attention_sum + mlp_sum)).detach().clone(),
                "raw_state": reference.detach().clone(),
                "normalized_state": event.state.detach().clone(),
                "current_pre": current.detach().clone(), "cached_pre": cached.detach().clone(),
                "effective_pre": effective.detach().clone(),
                **{f"M{i}": slots[i].detach().clone() for i in downstream.LAYERS},
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

    def mlp_dispatch(event):
        nonlocal mlp_sum, reference, expected_mlp_site
        if event.site != expected_mlp_site:
            raise MLP8MLPDepthSourceError("MLP sites are not sequential")
        expected_mlp_site += 1
        if event.site == MLP_LAYER:
            u_base = sum((slots[i] for i in range(4)), start=torch.zeros_like(x0))
            v_value = sum((slots[i] for i in range(4, 8)), start=torch.zeros_like(x0))
            group_remainder = mlp_sum - (u_base + v_value)
            regrouped = ((embedding + attention_sum) + (u_base + v_value)) \
                + group_remainder
            mlp8.update({
                "E": embedding.detach().clone(), "A": attention_sum.detach().clone(),
                "U": u_base.detach().clone(), "V": v_value.detach().clone(),
                "M_group_remainder": group_remainder.detach().clone(),
                "M": mlp_sum.detach().clone(),
                "epsilon": (reference - (embedding + attention_sum + mlp_sum)).detach().clone(),
                "raw_state": reference.detach().clone(), "input": event.state.detach().clone(),
                "regrouped": regrouped.detach().clone(),
            })
        write = event.block.mlp(event.state)
        if event.site == MLP_LAYER:
            mlp8["output"] = write.detach().clone()
        mlp_sum = mlp_sum + write
        reference = reference + write
        if event.site < LAYER:
            slots[event.site] = slots[event.site] + write
        return write

    logits = facade.forward_with_dispatch(
        model, tokens, attention, mlp_dispatch, require_production=False).float()
    required = {"p", "u", "head", "E", "A", "M", "H", "HR", "R", "raw_state",
                "normalized_state", "current_pre", "cached_pre", "effective_pre",
                *(f"M{i}" for i in downstream.LAYERS)}
    input_required = {"E", "A", "U", "V", "M_group_remainder", "M",
                      "epsilon", "raw_state", "input", "regrouped", "output"}
    if set(captured) != required or set(mlp8) != input_required or projection is None \
            or expected_attention_site != 18 or expected_mlp_site != 18:
        raise MLP8MLPDepthSourceError("decomposed forward audit failed")
    corrected = (captured["E"] + captured["A"] + captured["M"]) + captured["R"]
    input_corrected = mlp8["regrouped"] + mlp8["epsilon"]
    closure = {
        "state_sum_max_absolute_error": float((corrected - captured["raw_state"]).abs().max()),
        "normalized_state_max_absolute_error": float((
            F.rms_norm(corrected, (corrected.size(-1),))
            - captured["normalized_state"]).abs().max()),
        "input_state_closure_max_absolute_error": float((
            input_corrected - mlp8["raw_state"]).abs().max()),
        "input_normalized_closure_max_absolute_error": float((
            F.rms_norm(input_corrected, (input_corrected.size(-1),))
            - mlp8["input"]).abs().max()),
        "M_grouping_closure_max_absolute_error": float((
            ((mlp8["U"] + mlp8["V"]) + mlp8["M_group_remainder"])
            - mlp8["M"]).abs().max()),
    }
    return logits, captured, projection.detach().clone(), closure, mlp8


def _role_slice(values, start, stop):
    return {key: value[start:stop] for key, value in values.items()}


def _hybrid_input(recipient, source, subset, F):
    if subset not in SUBSETS:
        raise MLP8MLPDepthSourceError("unknown E/A/U/V subset")
    chosen = {family: source[family] if family in subset else recipient[family]
              for family in FAMILIES}
    epsilon = source["epsilon"] if "E" in subset else recipient["epsilon"]
    remainder = source["M_group_remainder"] if "U" in subset \
        else recipient["M_group_remainder"]
    raw = ((chosen["E"] + chosen["A"]) + (chosen["U"] + chosen["V"])) \
        + remainder + epsilon
    normalized = F.rms_norm(raw, (raw.size(-1),))
    endpoint_error = 0.0
    if subset == "EAUV":
        endpoint_error = float((normalized - source["input"]).abs().max())
        normalized = source["input"]
    return normalized, raw, endpoint_error


def _compile(recipient_tokens, recipient, opposite, attention, projection,
             slots, rows, torch, F):
    heads = {"recipient": parent._head_from_slot(
        recipient, opposite, recipient["M8"], attention, projection, torch, F)}
    for source in SOURCES:
        for subset in SUBSETS:
            for component in COMPONENTS:
                condition = f"{source}_{subset}_{component}"
                heads[condition] = parent._head_from_slot(
                    recipient, opposite, slots[source][subset][component],
                    attention, projection, torch, F)
    indices, replacements, specs, reinstall = [], [], [], []
    for row_index, row in enumerate(rows):
        for condition in CONDITIONS:
            indices.append(row_index)
            replacements.append(heads[condition][row_index])
            specs.append((row_index, condition,
                          f"{row['direction_id']}__{row['template_id']}"))
            reinstall.append(condition == "recipient")
    index = torch.tensor(indices, dtype=torch.long, device=recipient_tokens.device)
    return {"tokens": recipient_tokens[index],
            "finals": torch.full_like(index, SUBJECT_POSITION),
            "replacement_heads": torch.stack(replacements),
            "native_reinstall_mask": torch.tensor(
                reinstall, dtype=torch.bool, device=recipient_tokens.device),
            "specs": specs, "heads": heads}


def evaluate(model, torch, F, facade):
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = downstream.depth.parent.v1._role_batch(rows, torch, device)
    native = factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, closure, mlp8_all = _decomposed_forward(
        model, tokens, finals, torch, F, facade)
    recipient = _role_slice(captured, 0, n)
    opposite = _role_slice(captured, n, 2*n)
    lexical = _role_slice(captured, 2*n, 3*n)
    input_roles = {"recipient": _role_slice(mlp8_all, 0, n),
                   "opposite": _role_slice(mlp8_all, n, 2*n),
                   "lexical": _role_slice(mlp8_all, 2*n, 3*n)}
    mlp = model.transformer.h[MLP_LAYER].mlp
    attention = model.transformer.h[LAYER].attn
    slots, outputs, raw_inputs, normalized_inputs = (
        defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict))
    exactness = {
        "native_replay_max_absolute_logit_error": float((native - replay).abs().max()),
        "input_state_closure_max_absolute_error": closure["input_state_closure_max_absolute_error"],
        "input_normalized_closure_max_absolute_error": closure["input_normalized_closure_max_absolute_error"],
        "M_grouping_closure_max_absolute_error": closure["M_grouping_closure_max_absolute_error"],
        "hybrid_endpoint_max_absolute_error": 0.0,
        "source_term_sum_max_absolute_error": max(float((
            torch.einsum("bk,bkd->bd", role["p"], role["u"]) - role["head"]
        ).abs().max()) for role in (recipient, opposite, lexical)),
        "product_closure_max_absolute_error": 0.0,
        "output_closure_max_absolute_error": 0.0,
        "propagated_endpoint_max_absolute_error": 0.0,
        "gauge_invariance_max_absolute_error": 0.0,
        "parent_head_endpoint_max_absolute_error": 0.0,
        "same_batch_native_noop_endpoint_max_absolute_error": 0.0,
        "installed_head_max_absolute_error": 0.0,
        "parent_raw_state_max_absolute_error": 0.0,
        "parent_normalized_input_max_absolute_error": 0.0,
        "parent_MLP8_output_max_absolute_error": 0.0,
        "parent_propagated_slot_max_absolute_error": 0.0,
        "parent_installed_head_max_absolute_error": 0.0,
        "parent_downstream_outcome_max_absolute_error": 0.0,
    }
    numerical = defaultdict(dict)
    for source_name in SOURCES:
        source_input = input_roles[source_name]
        for subset in SUBSETS:
            hybrid, raw, endpoint_error = _hybrid_input(
                input_roles["recipient"], source_input, subset, F)
            products, algebra = polarized_v2._polarized_products(
                mlp, input_roles["recipient"]["input"], hybrid, torch, F)
            subset_slots, subset_outputs, output_diag = polarized_v2._propagated_slots(
                model, mlp, products, recipient["M8"], F,
                native_recipient_output=input_roles["recipient"]["output"],
                native_source_output=source_input["output"] if subset == "EAUV" else None)
            slots[source_name][subset] = subset_slots
            outputs[source_name][subset] = subset_outputs
            raw_inputs[source_name][subset] = raw
            normalized_inputs[source_name][subset] = hybrid
            numerical[source_name][subset] = {**algebra, **output_diag}
            exactness["hybrid_endpoint_max_absolute_error"] = max(
                exactness["hybrid_endpoint_max_absolute_error"], endpoint_error)
            for key, diagnostic in (
                ("product_closure_max_absolute_error", algebra),
                ("output_closure_max_absolute_error", output_diag),
                ("gauge_invariance_max_absolute_error", algebra)):
                exactness[key] = max(exactness[key], diagnostic[key])
            if subset == "EAUV":
                target = opposite if source_name == "opposite" else lexical
                exactness["propagated_endpoint_max_absolute_error"] = max(
                    exactness["propagated_endpoint_max_absolute_error"], float((
                        subset_slots["full"][:, SUBJECT_POSITION]
                        - target["M8"][:, SUBJECT_POSITION]).abs().max()))

        for parent_subset, child_subset in PARENT_TO_CHILD.items():
            parent_hybrid, parent_raw, _ = parent._hybrid_input(
                input_roles["recipient"], source_input, parent_subset, F)
            parent_products, _ = polarized_v2._polarized_products(
                mlp, input_roles["recipient"]["input"], parent_hybrid, torch, F)
            parent_slots, parent_outputs, _ = polarized_v2._propagated_slots(
                model, mlp, parent_products, recipient["M8"], F,
                native_recipient_output=input_roles["recipient"]["output"],
                native_source_output=source_input["output"] if parent_subset == "EAM" else None)
            exactness["parent_raw_state_max_absolute_error"] = max(
                exactness["parent_raw_state_max_absolute_error"],
                float((raw_inputs[source_name][child_subset] - parent_raw).abs().max()))
            exactness["parent_normalized_input_max_absolute_error"] = max(
                exactness["parent_normalized_input_max_absolute_error"],
                float((normalized_inputs[source_name][child_subset] - parent_hybrid).abs().max()))
            for component in COMPONENTS:
                exactness["parent_MLP8_output_max_absolute_error"] = max(
                    exactness["parent_MLP8_output_max_absolute_error"], float((
                        outputs[source_name][child_subset][component]
                        - parent_outputs[component]).abs().max()))
                exactness["parent_propagated_slot_max_absolute_error"] = max(
                    exactness["parent_propagated_slot_max_absolute_error"], float((
                        slots[source_name][child_subset][component]
                        - parent_slots[component]).abs().max()))
                child_head = parent._head_from_slot(
                    recipient, opposite, slots[source_name][child_subset][component],
                    attention, projection, torch, F)
                parent_head = parent._head_from_slot(
                    recipient, opposite, parent_slots[component],
                    attention, projection, torch, F)
                exactness["parent_installed_head_max_absolute_error"] = max(
                    exactness["parent_installed_head_max_absolute_error"],
                    float((child_head - parent_head).abs().max()))

    patch = _compile(tokens[:n], recipient, opposite, attention, projection,
                     slots, rows, torch, F)
    native_patch = factors._native_logits(model, patch["tokens"], torch, F)
    patched, _, _, patch_closure = downstream._decomposed_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])
    exactness["downstream_state_closure_max_absolute_error"] = patch_closure["state_sum_max_absolute_error"]
    exactness["downstream_normalized_closure_max_absolute_error"] = patch_closure["normalized_state_max_absolute_error"]
    exactness["parent_head_endpoint_max_absolute_error"] = max(
        float((patch["heads"][condition] - expected).abs().max())
        for condition, expected in {
            "recipient": polarized_v1._parent_head(
                recipient, opposite, "rrrrrrr", attention, projection, torch, F),
            "opposite_EAUV_full": polarized_v1._parent_head(
                recipient, opposite, "rrrrorr", attention, projection, torch, F),
            "lexical_EAUV_full": parent._head_from_slot(
                recipient, opposite, lexical["M8"], attention, projection, torch, F),
        }.items())
    evidence = []
    for out_index, (row_index, condition, cell_id) in enumerate(patch["specs"]):
        base = parent._both_metrics(native_patch[out_index, SUBJECT_POSITION],
                                    rows[row_index], torch)
        value = parent._both_metrics(patched[out_index, SUBJECT_POSITION],
                                     rows[row_index], torch)
        item = {"row_id": rows[row_index]["row_id"], "cell_id": cell_id,
                "condition": condition}
        for key in base:
            item[f"{key}_improvement"] = (
                base[key] - value[key] if key.endswith("CE") else value[key] - base[key])
        evidence.append(item)
        exactness["installed_head_max_absolute_error"] = max(
            exactness["installed_head_max_absolute_error"], float((
                patch["replacement_heads"][out_index]
                - patch["heads"][condition][row_index]).abs().max()))
        if condition == "recipient":
            exactness["same_batch_native_noop_endpoint_max_absolute_error"] = max(
                exactness["same_batch_native_noop_endpoint_max_absolute_error"],
                float((patched[out_index] - native_patch[out_index]).abs().max()))

    parent_by_key = {(item["row_id"], item["condition"]): item
                     for item in json.loads(PARENT_RESULT.read_text())["evidence"]}
    for item in evidence:
        if item["condition"] == "recipient":
            parent_condition = "recipient"
        else:
            source, subset, component = item["condition"].split("_")
            inverse = {value: key for key, value in PARENT_TO_CHILD.items()}
            if subset not in inverse:
                continue
            parent_condition = f"{source}_{inverse[subset]}_{component}"
        prior = parent_by_key[(item["row_id"], parent_condition)]
        for key in ("opposite_target_margin_improvement", "opposite_target_CE_improvement",
                    "lexical_target_margin_improvement", "lexical_target_CE_improvement"):
            exactness["parent_downstream_outcome_max_absolute_error"] = max(
                exactness["parent_downstream_outcome_max_absolute_error"],
                abs(float(item[key]) - float(prior[key])))
    return evidence, exactness, dict(numerical)


def _mobius(values):
    result = {}
    for subset in SUBSETS:
        members = tuple(subset)
        total = 0.0
        for size in range(len(members) + 1):
            for parts in combinations(members, size):
                key = "".join(family for family in FAMILIES if family in parts)
                total += (-1) ** (len(members) - size) * values[key]
        result[subset] = total
    return result


def _aggregate_terms(terms, parent_subset, depth_filter=None):
    required = set(parent_subset) - {"M"}
    excluded = ({"E", "A"} - required)
    selected = []
    for key, value in terms.items():
        families = set(key)
        if not required.issubset(families) or families & excluded:
            continue
        if not families & {"U", "V"}:
            continue
        if depth_filter == "U" and "U" not in families:
            continue
        if depth_filter == "V" and "V" not in families:
            continue
        if depth_filter == "U_only" and ("U" not in families or "V" in families):
            continue
        if depth_filter == "V_only" and ("V" not in families or "U" in families):
            continue
        if depth_filter == "UV" and not {"U", "V"}.issubset(families):
            continue
        selected.append(value)
    return sum(selected)


def score(evidence, exactness, bars=BARS):
    expected = {(row["row_id"], f"{row['direction_id']}__{row['template_id']}", condition)
                for row in build_rows() for condition in CONDITIONS}
    observed = [(item.get("row_id"), item.get("cell_id"), item.get("condition"))
                for item in evidence]
    metric_keys = tuple(f"{source}_target_{metric}_improvement"
                        for source in SOURCES for metric in ("margin", "CE"))
    if len(observed) != len(expected) or set(observed) != expected \
            or len(set(observed)) != len(expected):
        raise MLP8MLPDepthSourceError("evidence does not cover exact 91-condition screen")
    if any(type(item.get(key)) not in (int, float)
           or not math.isfinite(float(item[key]))
           for item in evidence for key in metric_keys):
        raise MLP8MLPDepthSourceError("task metric is missing or non-finite")
    grouped = defaultdict(lambda: defaultdict(list))
    for item in evidence:
        grouped[item["cell_id"]][item["condition"]].append(item)
    cells = {}
    for cell_id, conditions in sorted(grouped.items()):
        baseline = {item["row_id"]: item for item in conditions["recipient"]}
        cell = defaultdict(lambda: defaultdict(dict))
        for source in SOURCES:
            for component in COMPONENTS:
                for metric in ("margin", "CE"):
                    key = f"{source}_target_{metric}_improvement"
                    values = {"": 0.0}
                    row_effects = {}
                    for subset in SUBSETS:
                        effects = [float(item[key]) - float(baseline[item["row_id"]][key])
                                   for item in conditions[f"{source}_{subset}_{component}"]]
                        row_effects[subset] = effects
                        values[subset] = statistics.fmean(effects)
                    cell[source][component][metric] = {
                        "effects": values, "row_effects": row_effects,
                        "mobius": _mobius(values)}
        cells[cell_id] = {source: dict(value) for source, value in cell.items()}

    exact_pairs = (
        ("native_replay_max_absolute_logit_error", "maximum_native_replay_absolute_logit_error"),
        ("input_state_closure_max_absolute_error", "maximum_input_state_closure_absolute_error"),
        ("input_normalized_closure_max_absolute_error", "maximum_input_normalized_closure_absolute_error"),
        ("M_grouping_closure_max_absolute_error", "maximum_M_grouping_closure_absolute_error"),
        ("hybrid_endpoint_max_absolute_error", "maximum_hybrid_endpoint_absolute_error"),
        ("source_term_sum_max_absolute_error", "maximum_source_term_sum_absolute_error"),
        ("product_closure_max_absolute_error", "maximum_product_closure_absolute_error"),
        ("output_closure_max_absolute_error", "maximum_output_closure_absolute_error"),
        ("propagated_endpoint_max_absolute_error", "maximum_propagated_endpoint_absolute_error"),
        ("gauge_invariance_max_absolute_error", "maximum_gauge_invariance_absolute_error"),
        ("parent_head_endpoint_max_absolute_error", "maximum_parent_head_endpoint_absolute_error"),
        ("same_batch_native_noop_endpoint_max_absolute_error", "maximum_same_batch_native_noop_endpoint_error"),
        ("installed_head_max_absolute_error", "maximum_installed_head_absolute_error"),
        ("parent_raw_state_max_absolute_error", "maximum_parent_raw_state_absolute_error"),
        ("parent_normalized_input_max_absolute_error", "maximum_parent_normalized_input_absolute_error"),
        ("parent_MLP8_output_max_absolute_error", "maximum_parent_MLP8_output_absolute_error"),
        ("parent_propagated_slot_max_absolute_error", "maximum_parent_propagated_slot_absolute_error"),
        ("parent_installed_head_max_absolute_error", "maximum_parent_installed_head_absolute_error"),
        ("parent_downstream_outcome_max_absolute_error", "maximum_parent_downstream_outcome_absolute_error"),
    )
    exact_live = all(exactness[name] <= bars[bar] for name, bar in exact_pairs)

    parent_evidence = []
    inverse = {value: key for key, value in PARENT_CORNERS.items()}
    for item in evidence:
        if item["condition"] == "recipient":
            parent_evidence.append(dict(item))
            continue
        source, subset, component = item["condition"].split("_")
        if subset in inverse:
            copied = dict(item)
            copied["condition"] = f"{source}_{inverse[subset]}_{component}"
            parent_evidence.append(copied)
    parent_score = parent.score(parent_evidence, exactness, bars)
    instrument = exact_live and parent_score["predictions"]["pred_a_instrument_live"]

    direction_stats = defaultdict(lambda: defaultdict(list))
    for cell_id, cell in cells.items():
        direction = cell_id.split("__", 1)[0]
        for component in COMPONENTS:
            for metric in ("margin", "CE"):
                terms = cell["opposite"][component][metric]["mobius"]
                for aggregate in PARENT_AGGREGATES:
                    denominator = _aggregate_terms(terms, aggregate)
                    for depth in ("U", "V", "U_only", "V_only", "UV"):
                        numerator = _aggregate_terms(terms, aggregate, depth)
                        ratio = numerator / denominator if abs(denominator) > 1e-12 \
                            else float("nan")
                        direction_stats[direction][(component, metric, aggregate, depth)].append(ratio)
    direction_stats = {direction: {key: statistics.fmean(values)
                                    for key, values in entries.items()}
                       for direction, entries in direction_stats.items()}

    def dominates(depth, selected_directions=None):
        selected_directions = tuple(direction_stats) if selected_directions is None \
            else tuple(selected_directions)
        other_only = "U_only" if depth == "V" else "V_only"
        return instrument and all(
            direction_stats[direction][(component, metric, aggregate, depth)]
                >= bars["minimum_dominant_recovery_fraction"]
            and abs(direction_stats[direction][(
                component, metric, aggregate, other_only)])
                <= bars["maximum_minor_recovery_fraction"]
            for direction in selected_directions for component in COMPONENTS
            for metric in ("margin", "CE") for aggregate in ("M", "EM", "AM"))

    v_dominant, u_dominant = dominates("V"), dominates("U")
    distributed = instrument and not v_dominant and not u_dominant and all(any(
        all(abs(entries[(component, metric, aggregate, depth)])
            >= bars["minimum_distributed_recovery_fraction"]
            for entries in direction_stats.values() for metric in ("margin", "CE"))
        for component in COMPONENTS for aggregate in ("M", "EM", "AM"))
        for depth in ("U", "V"))
    cross_depth = instrument and any(all(
        abs(entries[(component, metric, aggregate, "UV")])
            >= bars["minimum_cross_depth_interaction_fraction"]
        for direction, entries in direction_stats.items()
        for metric in ("margin", "CE"))
        for component in COMPONENTS for aggregate in ("M", "EM", "AM"))
    winners = {}
    for direction in direction_stats:
        choices = [depth for depth in ("U", "V") if dominates(depth, (direction,))]
        winners[direction] = choices[0] if len(choices) == 1 else None
    direction_switch = instrument and len(set(winners.values())) > 1 \
        and all(value is not None for value in winners.values())

    lexical_ratios = []
    for cell in cells.values():
        for subset in SUBSETS:
            for component in COMPONENTS:
                for metric in ("margin", "CE"):
                    lexical = cell["lexical"][component][metric]["effects"][subset]
                    opposite_scale = cell["opposite"][component][metric]["effects"]["EAUV"]
                    lexical_ratios.append(abs(lexical) / max(abs(opposite_scale), 1e-12))
    number_specific = instrument and max(lexical_ratios) \
        <= bars["maximum_number_specific_lexical_ratio"]
    return {**exactness, "cells": cells, "parent_score": parent_score,
            "direction_aggregate_recovery": direction_stats,
            "direction_winners": winners,
            "maximum_lexical_ratio": max(lexical_ratios), "predictions": {
        "pred_a_instrument_and_parent_closure": bool(instrument),
        "pred_b_V_late_dominant": bool(v_dominant),
        "pred_c_U_early_dominant": bool(u_dominant),
        "pred_d_distributed_depth": bool(distributed),
        "pred_e_cross_depth_composition": bool(cross_depth),
        "pred_f_direction_switch": bool(direction_switch),
        "pred_g_number_specific": bool(number_specific),
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
        raise MLP8MLPDepthSourceError(f"refusing to overwrite {OUT}")
    torch, F, facade = factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness, numerical = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    terminal = "valid_causal_screen" if scored["predictions"][
        "pred_a_instrument_and_parent_closure"] else "invalid"
    result = {
        "schema": "task14_head11_3_fresh_matched_subject_mlp8_mlp_depth_source_factorial_result_v1",
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256, "score": scored,
        "numerical_diagnostics": numerical, "evidence": evidence,
        "evaluated_splits": ["LICENSED_HOLDOUT"], "forbidden_splits_opened": [],
        "model_forwards": 4, "causal_interventions": 1440,
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
