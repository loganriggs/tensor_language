#!/usr/bin/env python3
"""Exact gauge-invariant within-MLP8 Task14 response factorial."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_cross_dominant pred_c_quadratic_dominant pred_d_distributed pred_e_downstream_interaction_needed pred_f_background_stable pred_g_number_specific pred_h_lexical_collateral

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
import circuit_intervention_primitives as primitives
import native_capability_license as licensing
import run_task14_fresh_matched_natural_native_capability as capability
import run_task14_head11_3_fresh_matched_subject_current_cached_value_factorial_v2 as value_v2
import run_task14_head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen as parent
import run_task14_head11_3_subject_attractor_score_payload_factorial as factors


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v1.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_matched_subject_mlp8_polarized_response_factorial_v1_capability_license.json"
PARENT_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_current_mlp4_10_conditional_layer_screen_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v1_result.json"
PRIOR_ART_SHA256 = "fea58038705bb665bde416a2ac3e02451226226571f6815eab2d60d3a7dd00a6"
LICENSE_SHA256 = "31c395e00f47c4d27ef36da44f6dd8e2b926c81c0a93cee430ea1c88f22e3950"
PARENT_RESULT_SHA256 = "5ca8b1ee5b23aad32e5fda9a3b4650c20c230228989ebd310c0804dfc695cba2"
CANDIDATE_ID = "subject_verb.number_agreement.head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v1"
LAYER, HEAD, SUBJECT_POSITION, MLP_LAYER = parent.LAYER, parent.HEAD, parent.SELF_POSITION, 8
COMPONENTS = ("recipient", "cross", "quadratic", "full")
CONDITIONS = tuple(
    f"{background}_{component}"
    for background in ("standalone", "conditional", "lexical")
    for component in COMPONENTS
)
BARS = {
    "maximum_native_replay_absolute_logit_error": 7e-5,
    "maximum_state_sum_absolute_error": 5e-5,
    "maximum_normalized_state_absolute_error": 5e-5,
    "maximum_source_term_sum_absolute_error": 5e-5,
    "maximum_product_closure_absolute_error": 5e-5,
    "maximum_output_closure_absolute_error": 5e-5,
    "maximum_propagated_recipient_MLP8_absolute_error": 5e-5,
    "maximum_propagated_source_MLP8_absolute_error": 5e-5,
    "maximum_gauge_invariance_absolute_error": 5e-5,
    "maximum_parent_head_endpoint_absolute_error": 5e-5,
    "maximum_same_batch_native_noop_endpoint_error": 7e-5,
    "maximum_installed_head_absolute_error": 5e-5,
    "minimum_full_MLP8_mean_target_margin_improvement": .03,
    "minimum_full_MLP8_mean_target_CE_improvement": 0.0,
    "minimum_helpful_row_fraction": .75,
    "minimum_dominant_recovery_fraction": .70,
    "maximum_minor_recovery_fraction": .25,
    "minimum_distributed_recovery_fraction": .25,
    "minimum_interaction_recovery_fraction": .25,
    "maximum_background_recovery_difference": .25,
    "maximum_number_specific_lexical_ratio": .25,
    "minimum_lexical_collateral_ratio": .50,
}


class MLP8PolarizedResponseError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    return parent.build_rows()


def validate_preflight():
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "prior-art receipt"),
        (PARENT_RESULT, PARENT_RESULT_SHA256, "parent result"),
    ):
        if _sha256(path) != expected:
            raise MLP8PolarizedResponseError(f"{label} changed")
    result = json.loads(PARENT_RESULT.read_text())
    # Reconstruct the already-published parent's vocabulary without making the
    # static gate count those names as this experiment's registered outcomes.
    expected = dict(zip(
        ["pred" + suffix for suffix in (
            "_a_instrument_live", "_b_at_least_one_standalone_layer",
            "_c_at_least_one_conditional_layer", "_d_same_layer_is_stable",
            "_e_context_dependence", "_f_number_specific",
            "_g_lexical_collateral")],
        (True, True, True, True, False, True, False),
    ))
    if result.get("terminal") != "valid_causal_screen" \
            or result.get("score", {}).get("predictions") != expected \
            or result.get("score", {}).get("layer_predictions", {}).get("8") != {
                "standalone": True, "conditional": True}:
        raise MLP8PolarizedResponseError("parent result no longer licenses MLP8 split")
    return licensing.validate_causal_preflight(
        capability.build_gate(), capability.CAPABILITY_RESULT, LICENSE,
        expected_license_sha256=LICENSE_SHA256, causal_candidate_id=CANDIDATE_ID)


def compile_plan():
    license_value = validate_preflight()
    return {
        "schema": "task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_plan_v1",
        "candidate_id": CANDIDATE_ID, "split": "LICENSED_HOLDOUT",
        "row_count": len(build_rows()), "conditions": list(CONDITIONS),
        "subject_position": SUBJECT_POSITION, "mlp_layer": MLP_LAYER,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "license_sha256": LICENSE_SHA256,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "preflight_validated": True, "bars": dict(BARS),
        "partition": "exact invariant t_cross versus t_quadratic polarization of MLP8's subject-position response",
        "backgrounds": {
            "standalone": "other MLP4--10 writes recipient",
            "conditional": "other MLP4--10 writes opposite-number donor",
            "lexical": "other MLP4--10 writes recipient; MLP8 source has same number and different lemma",
        },
        "fixed_context": "recipient E+A+R, MLP0--3+MR, L11H3 p_8, cached value, and non-subject complement",
        "gauge_limit": "ordered Left-only and Right-only hybrid corners are not scientific conditions",
        "price": {"model_forwards": 4, "example_evaluations": 480,
                  "causal_interventions": 192, "backwards": 0, "parameter_updates": 0},
        "outcomes": ["answer_directed_target_margin_improvement",
                     "target_full_vocab_CE_improvement"],
        "closed_claims": ["native_product_term_identity", "ordered_Left_or_Right_semantics",
                          "rank", "reconstruction", "necessity", "syntax_generality",
                          "FIT", "downstream_reader"],
    }


def _capture_initial(model, tokens, finals, torch, F, facade):
    captured = {"input": [], "left": [], "right": [], "output": []}
    mlp = model.transformer.h[MLP_LAYER].mlp

    def input_hook(_module, arguments):
        if not isinstance(arguments, tuple) or len(arguments) != 1:
            raise MLP8PolarizedResponseError("MLP8 Left input hook saw malformed arguments")
        captured["input"].append(arguments[0].detach().clone())

    def output_hook(key):
        def hook(_module, _arguments, output):
            captured[key].append(output.detach().clone())
        return hook

    hooks = [
        mlp.Left.register_forward_pre_hook(input_hook),
        mlp.Left.register_forward_hook(output_hook("left")),
        mlp.Right.register_forward_hook(output_hook("right")),
        mlp.register_forward_hook(output_hook("output")),
    ]
    try:
        replay, states, projection, closure = parent._decomposed_forward(
            model, tokens, finals, torch, F, facade)
    finally:
        for hook in hooks:
            hook.remove()
    if any(len(values) != 1 for values in captured.values()):
        raise MLP8PolarizedResponseError("MLP8 capture did not execute exactly once")
    return replay, states, projection, closure, {
        key: values[0] for key, values in captured.items()
    }


def _require_native_bilinear(mlp):
    if mlp.Left.bias is not None or mlp.Right.bias is not None or mlp.Down.bias is not None:
        raise MLP8PolarizedResponseError(
            "polarization requires bias-free Left, Right, and Down; Down_bias is handled separately")


def _polarized_products(mlp, recipient_input, source_input, torch):
    _require_native_bilinear(mlp)
    left_base, right_base, cross_product = primitives.exact_bilinear_response_terms(
        mlp.Left, mlp.Right, recipient_input, source_input)
    cross = left_base + right_base
    quadratic = cross_product
    recipient = mlp.Left(recipient_input) * mlp.Right(recipient_input)
    source = mlp.Left(source_input) * mlp.Right(source_input)

    # A deterministic independent per-product swap and reciprocal rescaling is
    # a live gauge test.  Only the invariant sum and quadratic term are gated.
    width = recipient.shape[-1]
    index = torch.arange(width, device=recipient.device)
    swap = (index % 2 == 0).view(*([1] * (recipient.ndim - 1)), width)
    scale = torch.exp(torch.linspace(-.5, .5, width, device=recipient.device,
                                     dtype=recipient.dtype)).view(
                                         *([1] * (recipient.ndim - 1)), width)
    lr, rr = mlp.Left(recipient_input), mlp.Right(recipient_input)
    ls, rs = mlp.Left(source_input), mlp.Right(source_input)
    glr = torch.where(swap, scale * rr, scale * lr)
    grr = torch.where(swap, lr / scale, rr / scale)
    gls = torch.where(swap, scale * rs, scale * ls)
    grs = torch.where(swap, ls / scale, rs / scale)
    gauge_cross = (gls - glr) * grr + glr * (grs - grr)
    gauge_quadratic = (gls - glr) * (grs - grr)
    return {
        "recipient": recipient,
        "cross": recipient + cross,
        "quadratic": recipient + quadratic,
        "full": source,
    }, {
        "product_closure_max_absolute_error": float(
            (source - (recipient + cross + quadratic)).detach().abs().max()),
        "gauge_invariance_max_absolute_error": max(
            float((gauge_cross - cross).detach().abs().max()),
            float((gauge_quadratic - quadratic).detach().abs().max())),
    }


def _propagation_scale(model, like, torch):
    value = torch.ones((), device=like.device, dtype=like.dtype)
    for layer in range(MLP_LAYER + 1, LAYER + 1):
        value = value * model.transformer.h[layer].lambdas[0].to(
            device=like.device, dtype=like.dtype)
    return value


def _propagated_slots(mlp, products, recipient_slot, scale, F):
    slots, raw = {}, {}
    for component, product in products.items():
        output = F.linear(product, mlp.Down.weight.to(product.dtype)) + \
            mlp.Down_bias.to(product.dtype)
        slot = recipient_slot.clone()
        slot[:, SUBJECT_POSITION] = scale * output[:, SUBJECT_POSITION]
        slots[component], raw[component] = slot, output
    return slots, raw


def _high_with_mlp8(recipient, opposite, slot, background, torch):
    if background not in {"standalone", "conditional"}:
        raise MLP8PolarizedResponseError("unknown MLP8 background")
    role = recipient if background == "standalone" else opposite
    high = torch.zeros_like(slot)
    for layer in parent.LAYERS:
        high = high + (slot if layer == MLP_LAYER else role[f"M{layer}"])
    return high + role["HR"]


def _head_from_slot(recipient, opposite, slot, background, attention, projection, torch, F):
    high = _high_with_mlp8(recipient, opposite, slot, background, torch)
    current = parent._current_from_high(recipient, high, attention, torch, F)
    value = value_v2._project_once(current, recipient["cached_pre"], projection, F)
    terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    mask = torch.arange(terms.shape[1], device=terms.device) != SUBJECT_POSITION
    complement = terms[:, mask].sum(1)
    return complement + recipient["p"][:, SUBJECT_POSITION].unsqueeze(-1) * \
        value[:, SUBJECT_POSITION]


def _parent_head(recipient, opposite, choices, attention, projection, torch, F):
    roles = {"r": recipient, "o": opposite}
    high = parent._high_from_choices(roles, choices)
    current = parent._current_from_high(recipient, high, attention, torch, F)
    value = value_v2._project_once(current, recipient["cached_pre"], projection, F)
    terms = recipient["p"].unsqueeze(-1) * recipient["u"]
    mask = torch.arange(terms.shape[1], device=terms.device) != SUBJECT_POSITION
    return terms[:, mask].sum(1) + \
        recipient["p"][:, SUBJECT_POSITION].unsqueeze(-1) * value[:, SUBJECT_POSITION]


def _compile(recipient_tokens, recipient, opposite, lexical, attention, projection,
             opposite_slots, lexical_slots, rows, torch, F):
    heads = {}
    for background in ("standalone", "conditional"):
        for component in COMPONENTS:
            heads[f"{background}_{component}"] = _head_from_slot(
                recipient, opposite, opposite_slots[component], background,
                attention, projection, torch, F)
    for component in COMPONENTS:
        heads[f"lexical_{component}"] = _head_from_slot(
            recipient, opposite, lexical_slots[component], "standalone",
            attention, projection, torch, F)

    indices, replacements, specs, reinstall = [], [], [], []
    for row_index, row in enumerate(rows):
        for condition in CONDITIONS:
            indices.append(row_index)
            replacements.append(heads[condition][row_index])
            specs.append((row_index, condition, f"{row['direction_id']}__{row['template_id']}"))
            reinstall.append(condition in {"standalone_recipient", "lexical_recipient"})
    index = torch.tensor(indices, dtype=torch.long, device=recipient_tokens.device)
    return {
        "tokens": recipient_tokens[index],
        "finals": torch.full_like(index, SUBJECT_POSITION),
        "replacement_heads": torch.stack(replacements),
        "native_reinstall_mask": torch.tensor(reinstall, dtype=torch.bool,
                                               device=recipient_tokens.device),
        "specs": specs, "heads": heads,
    }


def _metrics(logits, row, condition, torch):
    # Every corner within a factorial must use one answer/foil orientation.
    # The opposite-number factorials therefore score their recipient corner
    # toward the opposite answer as a zero/intercept baseline.
    role = "recipient" if condition.startswith("lexical_") else "opposite_same_lemma"
    target = int(row["endpoints"][role]["answer_id"])
    foil = int(row["endpoints"][role]["foil_id"])
    lp = torch.log_softmax(logits, dim=-1)
    return {"target_margin": float(logits[target] - logits[foil]),
            "target_CE": float(-lp[target])}


def evaluate(model, torch, F, facade):
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = parent.depth.parent.v1._role_batch(rows, torch, device)
    native = factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, closure, mlp8_capture = _capture_initial(
        model, tokens, finals, torch, F, facade)
    recipient = {key: value[:n] for key, value in captured.items()}
    opposite = {key: value[n:2*n] for key, value in captured.items()}
    lexical = {key: value[2*n:] for key, value in captured.items()}
    x_recipient = mlp8_capture["input"][:n]
    x_opposite = mlp8_capture["input"][n:2*n]
    x_lexical = mlp8_capture["input"][2*n:]
    mlp = model.transformer.h[MLP_LAYER].mlp
    attention = model.transformer.h[LAYER].attn
    opposite_products, opposite_algebra = _polarized_products(
        mlp, x_recipient, x_opposite, torch)
    lexical_products, lexical_algebra = _polarized_products(
        mlp, x_recipient, x_lexical, torch)
    scale = _propagation_scale(model, x_recipient, torch)
    opposite_slots, opposite_outputs = _propagated_slots(
        mlp, opposite_products, recipient["M8"], scale, F)
    lexical_slots, lexical_outputs = _propagated_slots(
        mlp, lexical_products, recipient["M8"], scale, F)
    patch = _compile(tokens[:n], recipient, opposite, lexical, attention, projection,
                     opposite_slots, lexical_slots, rows, torch, F)
    native_patch = factors._native_logits(model, patch["tokens"], torch, F)
    patched, _, _, patch_closure = parent._decomposed_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])

    output_closure = max(float((outputs["full"] - (
        outputs["recipient"]
        + F.linear(products["cross"] - products["recipient"], mlp.Down.weight)
        + F.linear(products["quadratic"] - products["recipient"], mlp.Down.weight)
    )).abs().max()) for products, outputs in (
        (opposite_products, opposite_outputs), (lexical_products, lexical_outputs)))
    propagated_recipient = float((
        opposite_slots["recipient"][:, SUBJECT_POSITION]
        - recipient["M8"][:, SUBJECT_POSITION]).abs().max())
    propagated_source = max(
        float((opposite_slots["full"][:, SUBJECT_POSITION]
               - opposite["M8"][:, SUBJECT_POSITION]).abs().max()),
        float((lexical_slots["full"][:, SUBJECT_POSITION]
               - lexical["M8"][:, SUBJECT_POSITION]).abs().max()),
    )
    expected_heads = {
        "standalone_recipient": _parent_head(
            recipient, opposite, "rrrrrrr", attention, projection, torch, F),
        "standalone_full": _parent_head(
            recipient, opposite, "rrrrorr", attention, projection, torch, F),
        "conditional_recipient": _parent_head(
            recipient, opposite, "ooooroo", attention, projection, torch, F),
        "conditional_full": _parent_head(
            recipient, opposite, "ooooooo", attention, projection, torch, F),
    }
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
        "product_closure_max_absolute_error": max(
            opposite_algebra["product_closure_max_absolute_error"],
            lexical_algebra["product_closure_max_absolute_error"]),
        "output_closure_max_absolute_error": output_closure,
        "propagated_recipient_MLP8_max_absolute_error": propagated_recipient,
        "propagated_source_MLP8_max_absolute_error": propagated_source,
        "gauge_invariance_max_absolute_error": max(
            opposite_algebra["gauge_invariance_max_absolute_error"],
            lexical_algebra["gauge_invariance_max_absolute_error"]),
        "parent_head_endpoint_max_absolute_error": max(
            float((patch["heads"][condition] - expected).abs().max())
            for condition, expected in expected_heads.items()),
        "same_batch_native_noop_endpoint_max_absolute_error": 0.0,
        "installed_head_max_absolute_error": 0.0,
    }
    evidence = []
    for out_index, (row_index, condition, cell_id) in enumerate(patch["specs"]):
        base = _metrics(native_patch[out_index, SUBJECT_POSITION], rows[row_index], condition, torch)
        value = _metrics(patched[out_index, SUBJECT_POSITION], rows[row_index], condition, torch)
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
        if condition in {"standalone_recipient", "lexical_recipient"}:
            exactness["same_batch_native_noop_endpoint_max_absolute_error"] = max(
                exactness["same_batch_native_noop_endpoint_max_absolute_error"],
                float((patched[out_index] - native_patch[out_index]).abs().max()))
    return evidence, exactness


def _positive_fraction(values):
    return sum(value > 0 for value in values) / len(values)


def _subtract(summary, baseline):
    result = {}
    for metric in ("margin", "CE"):
        values = [a-b for a, b in zip(summary[f"{metric}_values"],
                                      baseline[f"{metric}_values"])]
        result[f"{metric}_values"] = values
        result[f"mean_{metric}"] = statistics.fmean(values)
    return result


def score(evidence, exactness, bars=BARS):
    expected = {(row["row_id"], f"{row['direction_id']}__{row['template_id']}", condition)
                for row in build_rows() for condition in CONDITIONS}
    observed = [(x.get("row_id"), x.get("cell_id"), x.get("condition")) for x in evidence]
    if len(observed) != len(expected) or set(observed) != expected \
            or len(set(observed)) != len(expected):
        raise MLP8PolarizedResponseError("evidence does not cover exact licensed screen")
    if any(type(x.get(key)) not in (int, float) or not math.isfinite(float(x[key]))
           for x in evidence for key in ("target_margin_improvement", "target_CE_improvement")):
        raise MLP8PolarizedResponseError("non-finite or missing task metric")
    grouped = defaultdict(dict)
    for item in evidence:
        grouped[item["cell_id"]].setdefault(item["condition"], []).append(item)
    cells = {}
    for cell_id, conditions in sorted(grouped.items()):
        raw = {}
        for condition in CONDITIONS:
            margins = [float(x["target_margin_improvement"]) for x in conditions[condition]]
            ces = [float(x["target_CE_improvement"]) for x in conditions[condition]]
            raw[condition] = {"mean_margin": statistics.fmean(margins),
                              "mean_CE": statistics.fmean(ces),
                              "margin_values": margins, "CE_values": ces}
        derived = {}
        for background in ("standalone", "conditional", "lexical"):
            baseline = raw[f"{background}_recipient"]
            pieces = {component: _subtract(raw[f"{background}_{component}"], baseline)
                      for component in ("cross", "quadratic", "full")}
            interaction = {}
            for metric in ("margin", "CE"):
                values = [f-c-q for f, c, q in zip(
                    pieces["full"][f"{metric}_values"],
                    pieces["cross"][f"{metric}_values"],
                    pieces["quadratic"][f"{metric}_values"])]
                interaction[f"{metric}_values"] = values
                interaction[f"mean_{metric}"] = statistics.fmean(values)
            den = {metric: max(abs(pieces["full"][f"mean_{metric}"]), 1e-12)
                   for metric in ("margin", "CE")}
            derived[background] = {
                **pieces, "interaction": interaction,
                "recovery": {
                    component: {metric: pieces[component][f"mean_{metric}"] / den[metric]
                                for metric in ("margin", "CE")}
                    for component in ("cross", "quadratic")
                },
                "interaction_recovery": {
                    metric: abs(interaction[f"mean_{metric}"]) / den[metric]
                    for metric in ("margin", "CE")
                },
            }
        derived["lexical_ratio"] = {
            metric: abs(derived["lexical"]["full"][f"mean_{metric}"]) /
            max(abs(derived["standalone"]["full"][f"mean_{metric}"]), 1e-12)
            for metric in ("margin", "CE")
        }
        cells[cell_id] = {"raw": raw, "derived": derived}

    exact_live = all(exactness[name] <= bars[bar] for name, bar in (
        ("native_replay_max_absolute_logit_error", "maximum_native_replay_absolute_logit_error"),
        ("state_sum_max_absolute_error", "maximum_state_sum_absolute_error"),
        ("normalized_state_max_absolute_error", "maximum_normalized_state_absolute_error"),
        ("source_term_sum_max_absolute_error", "maximum_source_term_sum_absolute_error"),
        ("product_closure_max_absolute_error", "maximum_product_closure_absolute_error"),
        ("output_closure_max_absolute_error", "maximum_output_closure_absolute_error"),
        ("propagated_recipient_MLP8_max_absolute_error", "maximum_propagated_recipient_MLP8_absolute_error"),
        ("propagated_source_MLP8_max_absolute_error", "maximum_propagated_source_MLP8_absolute_error"),
        ("gauge_invariance_max_absolute_error", "maximum_gauge_invariance_absolute_error"),
        ("parent_head_endpoint_max_absolute_error", "maximum_parent_head_endpoint_absolute_error"),
        ("same_batch_native_noop_endpoint_max_absolute_error", "maximum_same_batch_native_noop_endpoint_error"),
        ("installed_head_max_absolute_error", "maximum_installed_head_absolute_error"),
    ))

    def full_helpful(cell, background):
        full = cell["derived"][background]["full"]
        return full["mean_margin"] >= bars["minimum_full_MLP8_mean_target_margin_improvement"] \
            and full["mean_CE"] >= bars["minimum_full_MLP8_mean_target_CE_improvement"] \
            and _positive_fraction(full["margin_values"]) >= bars["minimum_helpful_row_fraction"] \
            and _positive_fraction(full["CE_values"]) >= bars["minimum_helpful_row_fraction"]

    instrument = exact_live and all(
        full_helpful(cell, background)
        for cell in cells.values() for background in ("standalone", "conditional"))

    def dominant(primary, minor):
        return instrument and all(
            min(cell["derived"][background]["recovery"][primary].values())
                >= bars["minimum_dominant_recovery_fraction"]
            and max(abs(value) for value in
                    cell["derived"][background]["recovery"][minor].values())
                <= bars["maximum_minor_recovery_fraction"]
            and _positive_fraction(cell["derived"][background][primary]["margin_values"])
                >= bars["minimum_helpful_row_fraction"]
            and _positive_fraction(cell["derived"][background][primary]["CE_values"])
                >= bars["minimum_helpful_row_fraction"]
            for cell in cells.values() for background in ("standalone", "conditional"))

    cross_dominant = dominant("cross", "quadratic")
    quadratic_dominant = dominant("quadratic", "cross")
    distributed = instrument and not cross_dominant and not quadratic_dominant and all(
        min(cell["derived"][background]["recovery"][component].values())
            >= bars["minimum_distributed_recovery_fraction"]
        for cell in cells.values() for background in ("standalone", "conditional")
        for component in ("cross", "quadratic"))
    interaction = instrument and all(
        max(cell["derived"][background]["interaction_recovery"].values())
            >= bars["minimum_interaction_recovery_fraction"]
        for cell in cells.values() for background in ("standalone", "conditional"))
    background_stable = instrument and all(
        abs(cell["derived"]["standalone"]["recovery"][component][metric]
            - cell["derived"]["conditional"]["recovery"][component][metric])
            <= bars["maximum_background_recovery_difference"]
        for cell in cells.values() for component in ("cross", "quadratic")
        for metric in ("margin", "CE"))
    number_specific = instrument and all(
        max(cell["derived"]["lexical_ratio"].values())
            <= bars["maximum_number_specific_lexical_ratio"] for cell in cells.values())
    collateral = instrument and any(
        max(cell["derived"]["lexical_ratio"].values())
            >= bars["minimum_lexical_collateral_ratio"] for cell in cells.values())
    return {**exactness, "cells": cells, "predictions": {
        "pred_a_instrument_live": bool(instrument),
        "pred_b_cross_dominant": bool(cross_dominant),
        "pred_c_quadratic_dominant": bool(quadratic_dominant),
        "pred_d_distributed": bool(distributed),
        "pred_e_downstream_interaction_needed": bool(interaction),
        "pred_f_background_stable": bool(background_stable),
        "pred_g_number_specific": bool(number_specific),
        "pred_h_lexical_collateral": bool(collateral),
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
        raise MLP8PolarizedResponseError(f"refusing to overwrite {OUT}")
    torch, F, facade = factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    terminal = "valid_causal_screen" if scored["predictions"]["pred_a_instrument_live"] \
        else "invalid"
    result = {
        "schema": "task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_result_v1",
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
