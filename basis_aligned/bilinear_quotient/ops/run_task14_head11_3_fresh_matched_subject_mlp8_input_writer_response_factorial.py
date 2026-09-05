#!/usr/bin/env python3
"""Exact E/A/M input-source factorial for MLP8's Task14 response."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_M_source_dominant pred_c_E_source_dominant pred_d_A_source_dominant pred_e_distributed_additive pred_f_source_interaction_needed pred_g_direction_stable pred_h_direction_switch pred_i_number_specific pred_j_lexical_collateral

from __future__ import annotations

from collections import defaultdict
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
import run_task14_head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen as parent
import run_task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial as polarized_v1
import run_task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v2 as polarized_v2
import run_task14_head11_3_subject_attractor_score_payload_factorial as factors


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_fresh_matched_subject_mlp8_input_writer_response_factorial_v1.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_matched_subject_mlp8_input_writer_response_factorial_v1_capability_license.json"
POLARIZED_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v2_result.json"
UPSTREAM_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_upstream_writer_factorial_v2_result.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_input_writer_response_factorial_v1_result.json"
PRIOR_ART_SHA256 = "6ceb69fa0860890534e89151c0b4a20290a271f400c13844c229008e56849b8b"
LICENSE_SHA256 = "693580f63d4e40ee9f36a0b32a733d7768aa1a28c3d13ad689080241f70adba2"
POLARIZED_RESULT_SHA256 = "55d5413306f4471b0c9b8345732d317d0c1c4b82395153a119af3d56514f5ad6"
UPSTREAM_RESULT_SHA256 = "5c021cad2f73663f2176a813fc1f4ceffef555b48d7d00c050d0f60d0a2434fa"
CANDIDATE_ID = "subject_verb.number_agreement.head11_3_fresh_matched_subject_mlp8_input_writer_response_factorial_v1"
LAYER, HEAD = parent.LAYER, parent.HEAD
SUBJECT_POSITION, MLP_LAYER = parent.SELF_POSITION, 8
FAMILIES = ("E", "A", "M")
SUBSETS = ("E", "A", "M", "EA", "EM", "AM", "EAM")
COMPONENTS = ("cross", "quadratic", "full")
SOURCES = ("opposite", "lexical")
CONDITIONS = ("recipient",) + tuple(
    f"{source}_{subset}_{component}"
    for source in SOURCES for subset in SUBSETS for component in COMPONENTS
)
BARS = {
    "maximum_native_replay_absolute_logit_error": 7e-5,
    "maximum_input_state_closure_absolute_error": 5e-5,
    "maximum_input_normalized_closure_absolute_error": 5e-5,
    "maximum_hybrid_endpoint_absolute_error": 5e-5,
    "maximum_source_term_sum_absolute_error": 5e-5,
    "maximum_product_closure_absolute_error": 5e-5,
    "maximum_output_closure_absolute_error": 5e-5,
    "maximum_propagated_endpoint_absolute_error": 5e-5,
    "maximum_gauge_invariance_absolute_error": 5e-5,
    "maximum_parent_head_endpoint_absolute_error": 5e-5,
    "maximum_same_batch_native_noop_endpoint_error": 7e-5,
    "maximum_installed_head_absolute_error": 5e-5,
    "minimum_EAM_full_margin_effect": .03,
    "minimum_EAM_full_CE_effect": 0.0,
    "minimum_helpful_row_fraction": .75,
    "minimum_dominant_recovery_fraction": .70,
    "maximum_minor_recovery_fraction": .25,
    "minimum_distributed_recovery_fraction": .25,
    "maximum_additive_residual_fraction": .25,
    "minimum_interaction_residual_fraction": .25,
    "maximum_number_specific_lexical_ratio": .25,
    "minimum_lexical_collateral_ratio": .50,
}


class MLP8InputWriterResponseError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    return parent.build_rows()


def validate_preflight():
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "prior-art receipt"),
        (POLARIZED_RESULT, POLARIZED_RESULT_SHA256, "polarized parent result"),
        (UPSTREAM_RESULT, UPSTREAM_RESULT_SHA256, "upstream-writer parent result"),
    ):
        if _sha256(path) != expected:
            raise MLP8InputWriterResponseError(f"{label} changed")
    polarized = json.loads(POLARIZED_RESULT.read_text())
    upstream = json.loads(UPSTREAM_RESULT.read_text())
    if polarized.get("terminal") != "valid_causal_screen" \
            or polarized.get("score", {}).get("predictions", {}).get(
                "pred_f_background_stable") is not True \
            or polarized.get("score", {}).get("predictions", {}).get(
                "pred_g_number_specific") is not True:
        raise MLP8InputWriterResponseError("polarized result no longer licenses source split")
    if upstream.get("terminal") != "valid_causal_screen" \
            or upstream.get("score", {}).get("predictions", {}).get(
                "pred_d_MLP_carries_task") is not True:
        raise MLP8InputWriterResponseError("upstream result no longer supplies the motivation")
    return licensing.validate_causal_preflight(
        capability.build_gate(), capability.CAPABILITY_RESULT, LICENSE,
        expected_license_sha256=LICENSE_SHA256, causal_candidate_id=CANDIDATE_ID)


def compile_plan():
    license_value = validate_preflight()
    return {
        "schema": "task14_head11_3_fresh_matched_subject_mlp8_input_writer_response_factorial_plan_v1",
        "candidate_id": CANDIDATE_ID, "split": "LICENSED_HOLDOUT",
        "row_count": len(build_rows()), "conditions": list(CONDITIONS),
        "condition_count": len(CONDITIONS), "subject_position": SUBJECT_POSITION,
        "mlp_layer": MLP_LAYER, "prior_art_sha256": PRIOR_ART_SHA256,
        "polarized_result_sha256": POLARIZED_RESULT_SHA256,
        "upstream_result_sha256": UPSTREAM_RESULT_SHA256,
        "license_sha256": LICENSE_SHA256,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "preflight_validated": True, "bars": dict(BARS),
        "input_partition": {
            "E": "propagated and reinjected normalized embedding/skip through block 8",
            "A": "propagated attention writes A0--A8",
            "M": "propagated MLP writes M0--M7",
            "epsilon": "float regrouping remainder follows E and is numerical only",
        },
        "response_partition": "exact invariant cross, quadratic, and full MLP8 responses for every nonempty E/A/M donor subset",
        "downstream_background": "standalone: every other MLP4--10 slot is recipient",
        "causal_statistics": "full 2^3 Moebius decomposition of each task-level source set function; normalization-mediated interactions are causal set-function interactions, not tensor identities",
        "gauge_limit": "E/A/M are operational native writer families; only the cross/quadratic MLP8 response is invariant to product-wise Left/Right swap and reciprocal scaling",
        "price": {"model_forwards": 4, "example_evaluations": 1472,
                  "causal_interventions": 672, "backwards": 0,
                  "parameter_updates": 0, "capability_GPU_price": 0},
        "outcomes": ["answer_directed_target_margin_improvement",
                     "target_full_vocab_CE_improvement"],
        "closed_claims": ["unique_semantic_basis", "individual_writer_identity",
                          "rank", "reconstruction", "new_independent_data",
                          "necessity_outside_fixed_L11H3_interface"],
    }


def _decomposed_forward(model, tokens, finals, torch, F, facade, *,
                        replacement_heads=None, native_reinstall_mask=None):
    """Replay while exactly tracking writer families into MLP8 and L11H3."""
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
            raise MLP8InputWriterResponseError("attention sites are not sequential")
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
            g0 = sum((slots[i] for i in range(4)), start=torch.zeros_like(x0))
            g1 = sum((slots[i] for i in range(4, 8)), start=torch.zeros_like(x0))
            g2 = sum((slots[i] for i in range(8, 11)), start=torch.zeros_like(x0))
            high = g1 + g2
            slot_high = sum((slots[i] for i in parent.LAYERS),
                            start=torch.zeros_like(x0))
            captured.update({name: value.detach().clone() for name, value in base.items()})
            captured.update({
                "E": embedding.detach().clone(), "A": attention_sum.detach().clone(),
                "M": mlp_sum.detach().clone(), "M0_3": g0.detach().clone(),
                "MR": (mlp_sum - ((g0 + g1) + g2)).detach().clone(),
                "H": high.detach().clone(),
                "HR": (high - slot_high).detach().clone(),
                "R": (reference - (embedding + attention_sum + mlp_sum)).detach().clone(),
                "raw_state": reference.detach().clone(),
                "normalized_state": event.state.detach().clone(),
                "current_pre": current.detach().clone(), "cached_pre": cached.detach().clone(),
                "effective_pre": effective.detach().clone(),
                **{f"M{i}": slots[i].detach().clone() for i in parent.LAYERS},
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
            raise MLP8InputWriterResponseError("MLP sites are not sequential")
        expected_mlp_site += 1
        if event.site == MLP_LAYER:
            regrouped = embedding + attention_sum + mlp_sum
            mlp8.update({
                "E": embedding.detach().clone(), "A": attention_sum.detach().clone(),
                "M": mlp_sum.detach().clone(),
                "epsilon": (reference - regrouped).detach().clone(),
                "raw_state": reference.detach().clone(),
                "input": event.state.detach().clone(),
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
    required = {"p", "u", "head", "E", "A", "M", "M0_3", "MR", "H", "HR",
                "R", "raw_state", "normalized_state", "current_pre", "cached_pre",
                "effective_pre", *(f"M{i}" for i in parent.LAYERS)}
    if set(captured) != required or set(mlp8) != {
        "E", "A", "M", "epsilon", "raw_state", "input", "output"
    } or projection is None or expected_attention_site != 18 or expected_mlp_site != 18:
        raise MLP8InputWriterResponseError("decomposed forward audit failed")
    corrected = (captured["E"] + captured["A"] + captured["M"]) + captured["R"]
    input_corrected = (mlp8["E"] + mlp8["A"] + mlp8["M"]) + mlp8["epsilon"]
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
    }
    return logits, captured, projection.detach().clone(), closure, mlp8


def _role_slice(values, start, stop):
    return {key: value[start:stop] for key, value in values.items()}


def _hybrid_input(recipient, source, subset, F):
    if subset not in SUBSETS:
        raise MLP8InputWriterResponseError("unknown E/A/M subset")
    chosen = {}
    for family in FAMILIES:
        chosen[family] = source[family] if family in subset else recipient[family]
    epsilon = source["epsilon"] if "E" in subset else recipient["epsilon"]
    raw = (chosen["E"] + chosen["A"] + chosen["M"]) + epsilon
    normalized = F.rms_norm(raw, (raw.size(-1),))
    if subset == "EAM":
        endpoint_error = float((normalized - source["input"]).abs().max())
        normalized = source["input"]
    else:
        endpoint_error = 0.0
    return normalized, raw, endpoint_error


def _head_from_slot(recipient, opposite, slot, attention, projection, torch, F):
    return polarized_v1._head_from_slot(
        recipient, opposite, slot, "standalone", attention, projection, torch, F)


def _compile(recipient_tokens, recipient, opposite, lexical, attention, projection,
             slots, rows, torch, F):
    heads = {"recipient": _head_from_slot(
        recipient, opposite, recipient["M8"], attention, projection, torch, F)}
    for source in SOURCES:
        for subset in SUBSETS:
            for component in COMPONENTS:
                condition = f"{source}_{subset}_{component}"
                heads[condition] = _head_from_slot(
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
    return {
        "tokens": recipient_tokens[index],
        "finals": torch.full_like(index, SUBJECT_POSITION),
        "replacement_heads": torch.stack(replacements),
        "native_reinstall_mask": torch.tensor(reinstall, dtype=torch.bool,
                                               device=recipient_tokens.device),
        "specs": specs, "heads": heads,
    }


def _both_metrics(logits, row, torch):
    result = {}
    for source, role in (("opposite", "opposite_same_lemma"),
                         ("lexical", "recipient")):
        endpoint = row["endpoints"][role]
        target, foil = int(endpoint["answer_id"]), int(endpoint["foil_id"])
        lp = torch.log_softmax(logits, dim=-1)
        result[f"{source}_target_margin"] = float(logits[target] - logits[foil])
        result[f"{source}_target_CE"] = float(-lp[target])
    return result


def evaluate(model, torch, F, facade):
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = parent.depth.parent.v1._role_batch(rows, torch, device)
    native = factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, closure, mlp8_all = _decomposed_forward(
        model, tokens, finals, torch, F, facade)
    recipient = _role_slice(captured, 0, n)
    opposite = _role_slice(captured, n, 2*n)
    lexical = _role_slice(captured, 2*n, 3*n)
    input_roles = {
        "recipient": _role_slice(mlp8_all, 0, n),
        "opposite": _role_slice(mlp8_all, n, 2*n),
        "lexical": _role_slice(mlp8_all, 2*n, 3*n),
    }
    mlp = model.transformer.h[MLP_LAYER].mlp
    attention = model.transformer.h[LAYER].attn
    slots, numerical = defaultdict(dict), defaultdict(dict)
    exactness = {
        "native_replay_max_absolute_logit_error": float((native - replay).abs().max()),
        "input_state_closure_max_absolute_error": closure["input_state_closure_max_absolute_error"],
        "input_normalized_closure_max_absolute_error": closure["input_normalized_closure_max_absolute_error"],
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
    }
    for source_name in SOURCES:
        source_input = input_roles[source_name]
        for subset in SUBSETS:
            hybrid, _raw, endpoint_error = _hybrid_input(
                input_roles["recipient"], source_input, subset, F)
            products, algebra = polarized_v2._polarized_products(
                mlp, input_roles["recipient"]["input"], hybrid, torch, F)
            subset_slots, _outputs, output_diag = polarized_v2._propagated_slots(
                model, mlp, products, recipient["M8"], F,
                native_recipient_output=input_roles["recipient"]["output"],
                native_source_output=source_input["output"] if subset == "EAM" else None)
            slots[source_name][subset] = subset_slots
            numerical[source_name][subset] = {**algebra, **output_diag}
            exactness["hybrid_endpoint_max_absolute_error"] = max(
                exactness["hybrid_endpoint_max_absolute_error"], endpoint_error)
            exactness["product_closure_max_absolute_error"] = max(
                exactness["product_closure_max_absolute_error"],
                algebra["product_closure_max_absolute_error"])
            exactness["output_closure_max_absolute_error"] = max(
                exactness["output_closure_max_absolute_error"],
                output_diag["output_closure_max_absolute_error"])
            exactness["gauge_invariance_max_absolute_error"] = max(
                exactness["gauge_invariance_max_absolute_error"],
                algebra["gauge_invariance_max_absolute_error"])
            if subset == "EAM":
                exactness["propagated_endpoint_max_absolute_error"] = max(
                    exactness["propagated_endpoint_max_absolute_error"], float((
                        subset_slots["full"][:, SUBJECT_POSITION]
                        - (opposite if source_name == "opposite" else lexical)["M8"][:, SUBJECT_POSITION]
                    ).abs().max()))
    patch = _compile(tokens[:n], recipient, opposite, lexical, attention, projection,
                     slots, rows, torch, F)
    native_patch = factors._native_logits(model, patch["tokens"], torch, F)
    patched, _, _, patch_closure = parent._decomposed_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])
    exactness["downstream_state_closure_max_absolute_error"] = \
        patch_closure["state_sum_max_absolute_error"]
    exactness["downstream_normalized_closure_max_absolute_error"] = \
        patch_closure["normalized_state_max_absolute_error"]
    expected = {
        "recipient": polarized_v1._parent_head(
            recipient, opposite, "rrrrrrr", attention, projection, torch, F),
        "opposite_EAM_full": polarized_v1._parent_head(
            recipient, opposite, "rrrrorr", attention, projection, torch, F),
        "lexical_EAM_full": _head_from_slot(
            recipient, opposite, lexical["M8"], attention, projection, torch, F),
    }
    exactness["parent_head_endpoint_max_absolute_error"] = max(
        float((patch["heads"][condition] - value).abs().max())
        for condition, value in expected.items())
    evidence = []
    for out_index, (row_index, condition, cell_id) in enumerate(patch["specs"]):
        base = _both_metrics(native_patch[out_index, SUBJECT_POSITION], rows[row_index], torch)
        value = _both_metrics(patched[out_index, SUBJECT_POSITION], rows[row_index], torch)
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
    return evidence, exactness, dict(numerical)


def _positive_fraction(values):
    return sum(value > 0 for value in values) / len(values)


def _mean(values):
    return statistics.fmean(values)


def _mobius(values):
    empty = values[""]
    return {
        "E": values["E"] - empty,
        "A": values["A"] - empty,
        "M": values["M"] - empty,
        "EA": values["EA"] - values["E"] - values["A"] + empty,
        "EM": values["EM"] - values["E"] - values["M"] + empty,
        "AM": values["AM"] - values["A"] - values["M"] + empty,
        "EAM": values["EAM"] - values["EA"] - values["EM"] - values["AM"]
               + values["E"] + values["A"] + values["M"] - empty,
    }


def score(evidence, exactness, bars=BARS):
    expected = {(row["row_id"], f"{row['direction_id']}__{row['template_id']}", condition)
                for row in build_rows() for condition in CONDITIONS}
    observed = [(item.get("row_id"), item.get("cell_id"), item.get("condition"))
                for item in evidence]
    metric_keys = tuple(
        f"{source}_target_{metric}_improvement"
        for source in SOURCES for metric in ("margin", "CE"))
    if len(observed) != len(expected) or set(observed) != expected \
            or len(set(observed)) != len(expected):
        raise MLP8InputWriterResponseError("evidence does not cover exact 43-condition screen")
    if any(type(item.get(key)) not in (int, float)
           or not math.isfinite(float(item[key]))
           for item in evidence for key in metric_keys):
        raise MLP8InputWriterResponseError("task metric is missing or non-finite")
    grouped = defaultdict(lambda: defaultdict(list))
    for item in evidence:
        grouped[item["cell_id"]][item["condition"]].append(item)
    for conditions in grouped.values():
        for items in conditions.values():
            items.sort(key=lambda item: item["row_id"])
    cells = {}
    for cell_id, conditions in sorted(grouped.items()):
        derived = {}
        baseline = {item["row_id"]: item for item in conditions["recipient"]}
        for source in SOURCES:
            derived[source] = {}
            for component in COMPONENTS:
                stats = {}
                for subset in SUBSETS:
                    items = conditions[f"{source}_{subset}_{component}"]
                    by_metric = {}
                    for metric in ("margin", "CE"):
                        key = f"{source}_target_{metric}_improvement"
                        values = [float(item[key]) - float(baseline[item["row_id"]][key])
                                  for item in items]
                        by_metric[metric] = values
                    stats[subset] = by_metric
                derived[source][component] = {}
                for metric in ("margin", "CE"):
                    row_effects = {subset: stats[subset][metric] for subset in SUBSETS}
                    values = {"": 0.0, **{subset: _mean(row_effects[subset])
                                          for subset in SUBSETS}}
                    denominator = values["EAM"]
                    recovery = {subset: values[subset] / denominator
                                if abs(denominator) > 1e-12 else float("nan")
                                for subset in SUBSETS}
                    mobius = _mobius(values)
                    interaction_residual = values["EAM"] - values["E"] \
                        - values["A"] - values["M"]
                    derived[source][component][metric] = {
                        "effects": values, "row_effects": row_effects,
                        "signed_recovery": recovery,
                        "mobius": mobius,
                        "interaction_residual": interaction_residual,
                        "interaction_residual_fraction": abs(interaction_residual) /
                            max(abs(denominator), 1e-12),
                    }
        cells[cell_id] = derived

    exact_live = all(exactness[name] <= bars[bar] for name, bar in (
        ("native_replay_max_absolute_logit_error", "maximum_native_replay_absolute_logit_error"),
        ("input_state_closure_max_absolute_error", "maximum_input_state_closure_absolute_error"),
        ("input_normalized_closure_max_absolute_error", "maximum_input_normalized_closure_absolute_error"),
        ("hybrid_endpoint_max_absolute_error", "maximum_hybrid_endpoint_absolute_error"),
        ("source_term_sum_max_absolute_error", "maximum_source_term_sum_absolute_error"),
        ("product_closure_max_absolute_error", "maximum_product_closure_absolute_error"),
        ("output_closure_max_absolute_error", "maximum_output_closure_absolute_error"),
        ("propagated_endpoint_max_absolute_error", "maximum_propagated_endpoint_absolute_error"),
        ("gauge_invariance_max_absolute_error", "maximum_gauge_invariance_absolute_error"),
        ("parent_head_endpoint_max_absolute_error", "maximum_parent_head_endpoint_absolute_error"),
        ("same_batch_native_noop_endpoint_max_absolute_error", "maximum_same_batch_native_noop_endpoint_error"),
        ("installed_head_max_absolute_error", "maximum_installed_head_absolute_error"),
    ))
    def row_values(cell_id, source, subset, component, metric):
        return cells[cell_id][source][component][metric]["row_effects"][subset]

    def full_live(cell_id):
        return all(
            cells[cell_id]["opposite"]["full"][metric]["effects"]["EAM"]
                >= bars[f"minimum_EAM_full_{metric}_effect"]
            and _positive_fraction(row_values(
                cell_id, "opposite", "EAM", "full", metric))
                >= bars["minimum_helpful_row_fraction"]
            for metric in ("margin", "CE"))

    def polarization(cell_id):
        expected_cross = 1 if cell_id.startswith("plural_to_singular") else -1
        return all(
            expected_cross * cells[cell_id]["opposite"]["cross"][metric]["effects"]["EAM"] > 0
            and -expected_cross * cells[cell_id]["opposite"]["quadratic"][metric]["effects"]["EAM"] > 0
            for metric in ("margin", "CE"))

    instrument = exact_live and all(full_live(cell_id) and polarization(cell_id)
                                    for cell_id in cells)

    def family_dominates(family, selected_cells=None, component=None):
        selected_cells = list(cells) if selected_cells is None else selected_cells
        components = COMPONENTS if component is None else (component,)
        others = [name for name in FAMILIES if name != family]
        return instrument and all(
            cells[cell_id]["opposite"][piece][metric]["signed_recovery"][family]
                >= bars["minimum_dominant_recovery_fraction"]
            and max(abs(cells[cell_id]["opposite"][piece][metric]
                        ["signed_recovery"][other]) for other in others)
                <= bars["maximum_minor_recovery_fraction"]
            and sum(value * cells[cell_id]["opposite"][piece][metric]
                    ["effects"]["EAM"] > 0 for value in row_values(
                        cell_id, "opposite", family, piece, metric)) /
                    len(row_values(cell_id, "opposite", family, piece, metric))
                >= bars["minimum_helpful_row_fraction"]
            for cell_id in selected_cells for piece in components
            for metric in ("margin", "CE"))

    dominant = {family: family_dominates(family) for family in FAMILIES}
    distributed = instrument and not any(dominant.values()) and all(
        sum(abs(cell["opposite"][piece][metric]["signed_recovery"][family])
            >= bars["minimum_distributed_recovery_fraction"] for family in FAMILIES) >= 2
        and cell["opposite"][piece][metric]["interaction_residual_fraction"]
            <= bars["maximum_additive_residual_fraction"]
        for cell in cells.values() for piece in COMPONENTS for metric in ("margin", "CE"))
    interaction_needed = instrument and any(all(
        cells[cell_id]["opposite"][piece][metric]["interaction_residual_fraction"]
            >= bars["minimum_interaction_residual_fraction"]
        for cell_id in cells for metric in ("margin", "CE")) for piece in COMPONENTS)

    winners = defaultdict(dict)
    for direction in ("plural_to_singular", "singular_to_plural"):
        selected = [cell_id for cell_id in cells if cell_id.startswith(direction)]
        for piece in COMPONENTS:
            matches = [family for family in FAMILIES
                       if family_dominates(family, selected, piece)]
            winners[direction][piece] = matches[0] if len(matches) == 1 else None
    paired = [(winners["plural_to_singular"][piece],
               winners["singular_to_plural"][piece]) for piece in COMPONENTS]
    direction_stable = instrument and any(a is not None and b is not None for a, b in paired) \
        and all((a is None and b is None) or (a is not None and a == b)
                for a, b in paired)
    direction_switch = instrument and any(
        a is not None and b is not None and a != b for a, b in paired)

    lexical_ratios = []
    for cell_id, cell in cells.items():
        for subset in SUBSETS:
            for piece in COMPONENTS:
                for metric in ("margin", "CE"):
                    lexical = cell["lexical"][piece][metric]["effects"][subset]
                    opposite_scale = cell["opposite"][piece][metric]["effects"]["EAM"]
                    lexical_ratios.append(abs(lexical) / max(abs(opposite_scale), 1e-12))
    number_specific = instrument and max(lexical_ratios) \
        <= bars["maximum_number_specific_lexical_ratio"]
    collateral = instrument and max(lexical_ratios) \
        >= bars["minimum_lexical_collateral_ratio"]
    return {**exactness, "cells": cells, "direction_component_winners": dict(winners),
            "maximum_lexical_ratio": max(lexical_ratios), "predictions": {
        "pred_a_instrument_live": bool(instrument),
        "pred_b_M_source_dominant": bool(dominant["M"]),
        "pred_c_E_source_dominant": bool(dominant["E"]),
        "pred_d_A_source_dominant": bool(dominant["A"]),
        "pred_e_distributed_additive": bool(distributed),
        "pred_f_source_interaction_needed": bool(interaction_needed),
        "pred_g_direction_stable": bool(direction_stable),
        "pred_h_direction_switch": bool(direction_switch),
        "pred_i_number_specific": bool(number_specific),
        "pred_j_lexical_collateral": bool(collateral),
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
        raise MLP8InputWriterResponseError(f"refusing to overwrite {OUT}")
    torch, F, facade = factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness, numerical = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    terminal = "valid_causal_screen" if scored["predictions"][
        "pred_a_instrument_live"] else "invalid"
    result = {
        "schema": "task14_head11_3_fresh_matched_subject_mlp8_input_writer_response_factorial_result_v1",
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256, "score": scored,
        "numerical_diagnostics": numerical, "evidence": evidence,
        "evaluated_splits": ["LICENSED_HOLDOUT"], "forbidden_splits_opened": [],
        "model_forwards": 4, "causal_interventions": 672,
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
