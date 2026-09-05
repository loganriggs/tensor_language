#!/usr/bin/env python3
"""Numerically repaired exact within-MLP8 Task14 response factorial."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_cross_dominant pred_c_quadratic_dominant pred_d_distributed pred_e_downstream_interaction_needed pred_f_background_stable pred_g_number_specific pred_h_lexical_collateral

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_managed_runner as managed
import circuit_intervention_primitives as primitives
import native_capability_license as licensing
import run_task14_fresh_matched_natural_native_capability as capability
import run_task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial as v1


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_fresh_matched_subject_mlp8_polarized_response_numerical_repair_v2.json"
LICENSE = ROOT / "circuits/fast_screens/task14_fresh_matched_subject_mlp8_polarized_response_factorial_v2_capability_license.json"
V1_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v2_result.json"
PRIOR_ART_SHA256 = "6dd99090a2ac12bb9ff0ea4f3e77d4ada9ab7a4c2c1cc3f5937479a51016f763"
LICENSE_SHA256 = "ad1792c2e5b211cb46f1f372d23eaba6a328141c7177eb09d035f8f41be8e919"
V1_RESULT_SHA256 = "27c0a70502765b15dc6b2f4118e176d60b4cd63ec47bae1a19e366799f42dc39"
CANDIDATE_ID = "subject_verb.number_agreement.head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v2"
CONDITIONS = v1.CONDITIONS
COMPONENTS = v1.COMPONENTS
BARS = v1.BARS
LAYER, HEAD = v1.LAYER, v1.HEAD
SUBJECT_POSITION, MLP_LAYER = v1.SUBJECT_POSITION, v1.MLP_LAYER
PREDICTIONS = {
    "pred_a_instrument_live": "unchanged v1 instrument prediction",
    "pred_b_cross_dominant": "unchanged v1 cross-dominant prediction",
    "pred_c_quadratic_dominant": "unchanged v1 quadratic-dominant prediction",
    "pred_d_distributed": "unchanged v1 distributed-response prediction",
    "pred_e_downstream_interaction_needed": "unchanged v1 downstream-interaction prediction",
    "pred_f_background_stable": "unchanged v1 background-stability prediction",
    "pred_g_number_specific": "unchanged v1 number-specificity prediction",
    "pred_h_lexical_collateral": "unchanged v1 lexical-collateral prediction",
}


class MLP8PolarizedResponseV2Error(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    return v1.build_rows()


def validate_preflight():
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "correction receipt"),
        (V1_RESULT, V1_RESULT_SHA256, "invalid v1 result"),
    ):
        if _sha256(path) != expected:
            raise MLP8PolarizedResponseV2Error(f"{label} changed")
    result = json.loads(V1_RESULT.read_text())
    score = result.get("score", {})
    failed_numerical = (
        score.get("product_closure_max_absolute_error", 0) >
            BARS["maximum_product_closure_absolute_error"]
        and score.get("output_closure_max_absolute_error", 0) >
            BARS["maximum_output_closure_absolute_error"]
        and score.get("propagated_recipient_MLP8_max_absolute_error", 0) >
            BARS["maximum_propagated_recipient_MLP8_absolute_error"]
        and score.get("propagated_source_MLP8_max_absolute_error", 0) >
            BARS["maximum_propagated_source_MLP8_absolute_error"]
    )
    preserved_controls = all(score.get(name, float("inf")) <= BARS[bar] for name, bar in (
        ("native_replay_max_absolute_logit_error", "maximum_native_replay_absolute_logit_error"),
        ("state_sum_max_absolute_error", "maximum_state_sum_absolute_error"),
        ("normalized_state_max_absolute_error", "maximum_normalized_state_absolute_error"),
        ("source_term_sum_max_absolute_error", "maximum_source_term_sum_absolute_error"),
        ("parent_head_endpoint_max_absolute_error", "maximum_parent_head_endpoint_absolute_error"),
        ("same_batch_native_noop_endpoint_max_absolute_error", "maximum_same_batch_native_noop_endpoint_error"),
        ("installed_head_max_absolute_error", "maximum_installed_head_absolute_error"),
    ))
    if result.get("terminal") != "invalid" or not failed_numerical or not preserved_controls:
        raise MLP8PolarizedResponseV2Error("v1 no longer licenses numerical-only repair")
    return licensing.validate_causal_preflight(
        capability.build_gate(), capability.CAPABILITY_RESULT, LICENSE,
        expected_license_sha256=LICENSE_SHA256, causal_candidate_id=CANDIDATE_ID)


def compile_plan():
    license_value = validate_preflight()
    return {
        "schema": "task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_plan_v2",
        "candidate_id": CANDIDATE_ID, "split": "LICENSED_HOLDOUT",
        "row_count": len(build_rows()), "conditions": list(CONDITIONS),
        "subject_position": SUBJECT_POSITION, "mlp_layer": MLP_LAYER,
        "correction_receipt_sha256": PRIOR_ART_SHA256,
        "invalid_v1_result_sha256": V1_RESULT_SHA256,
        "license_sha256": LICENSE_SHA256,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "preflight_validated": True, "bars": dict(BARS),
        "partition": "unchanged exact invariant t_cross versus t_quadratic polarization of MLP8's subject-position response",
        "backgrounds": {
            "standalone": "other MLP4--10 writes recipient",
            "conditional": "other MLP4--10 writes opposite-number donor",
            "lexical": "other MLP4--10 writes recipient; MLP8 source has same number and different lemma",
        },
        "fixed_context": "unchanged recipient E+A+R, MLP0--3+MR, L11H3 p_8, cached value, and non-subject complement",
        "numerical_repair": {
            "product_remainder": "assigned to cross",
            "output_remainder": "assigned to cross",
            "gauge_validation": "float64 uncorrected invariant tensors",
            "propagation": "sequential native block-9/10/11 residual multiplications",
        },
        "gauge_limit": "unchanged: ordered Left-only and Right-only hybrid corners are not scientific conditions",
        "price": {"model_forwards": 4, "example_evaluations": 480,
                  "causal_interventions": 192, "backwards": 0, "parameter_updates": 0},
        "outcomes": ["answer_directed_target_margin_improvement",
                     "target_full_vocab_CE_improvement"],
        "scientific_scoring": "byte-for-byte function reuse from v1 with unchanged conditions, bars, and predictions",
        "predictions": dict(PREDICTIONS),
        "closed_claims": ["native_product_term_identity", "ordered_Left_or_Right_semantics",
                          "rank", "reconstruction", "necessity", "syntax_generality",
                          "FIT", "downstream_reader"],
    }


def _float64_gauge_error(mlp, recipient_input, source_input, torch, F):
    xr, xs = recipient_input.double(), source_input.double()
    wl, wr = mlp.Left.weight.detach().double(), mlp.Right.weight.detach().double()
    lr, rr = F.linear(xr, wl), F.linear(xr, wr)
    ls, rs = F.linear(xs, wl), F.linear(xs, wr)
    dl, dr = ls - lr, rs - rr
    cross, quadratic = dl * rr + lr * dr, dl * dr
    width = cross.shape[-1]
    index = torch.arange(width, device=cross.device)
    swap = (index % 2 == 0).view(*([1] * (cross.ndim - 1)), width)
    scale = torch.exp(torch.linspace(-.5, .5, width, device=cross.device,
                                     dtype=cross.dtype)).view(
                                         *([1] * (cross.ndim - 1)), width)
    glr = torch.where(swap, scale * rr, scale * lr)
    grr = torch.where(swap, lr / scale, rr / scale)
    gls = torch.where(swap, scale * rs, scale * ls)
    grs = torch.where(swap, ls / scale, rs / scale)
    gauge_cross = (gls - glr) * grr + glr * (grs - grr)
    gauge_quadratic = (gls - glr) * (grs - grr)
    return max(float((gauge_cross - cross).abs().max()),
               float((gauge_quadratic - quadratic).abs().max()))


def _polarized_products(mlp, recipient_input, source_input, torch, F):
    v1._require_native_bilinear(mlp)
    wl, wr = mlp.Left.weight.detach().double(), mlp.Right.weight.detach().double()

    def left64(value):
        return F.linear(value.double(), wl)

    def right64(value):
        return F.linear(value.double(), wr)

    # Reuse the audited primitive on promoted inputs and weights.  This changes
    # no mathematical condition; it only prevents wide float32 regrouping from
    # exceeding an absolute exactness bar before the correction is assigned.
    first, second, quadratic = primitives.exact_bilinear_response_terms(
        left64, right64, recipient_input, source_input)
    cross = first + second
    recipient = left64(recipient_input) * right64(recipient_input)
    source = left64(source_input) * right64(source_input)
    uncorrected = (recipient + cross) + quadratic
    product_remainder = source - uncorrected
    corrected_cross = cross + product_remainder
    products = {
        "recipient": recipient,
        "cross": recipient + corrected_cross,
        "quadratic": recipient + quadratic,
        "full": source,
    }
    closure = source - ((recipient + corrected_cross) + quadratic)
    return products, {
        "uncorrected_product_closure_max_absolute_error": float(
            (source - uncorrected).detach().abs().max()),
        "product_remainder_max_absolute_value": float(
            product_remainder.detach().abs().max()),
        "product_closure_max_absolute_error": float(closure.detach().abs().max()),
        "gauge_invariance_max_absolute_error": _float64_gauge_error(
            mlp, recipient_input, source_input, torch, F),
    }


def _sequentially_propagate(model, output, dtype=None):
    result = output if dtype is None else output.to(dtype=dtype)
    for layer in range(MLP_LAYER + 1, LAYER + 1):
        result = model.transformer.h[layer].lambdas[0] * result
    return result


def _propagated_slots(model, mlp, products, recipient_slot, F, *,
                      native_recipient_output=None, native_source_output=None):
    weight = mlp.Down.weight.detach().double()
    bias = mlp.Down_bias.detach().double()
    outputs = {component: F.linear(product.double(), weight) + bias
               for component, product in products.items()}
    if native_recipient_output is not None:
        outputs["recipient"] = native_recipient_output.detach().double()
    if native_source_output is not None:
        outputs["full"] = native_source_output.detach().double()
    response_cross = outputs["cross"] - outputs["recipient"]
    response_quadratic = outputs["quadratic"] - outputs["recipient"]
    output_remainder = (outputs["full"] - outputs["recipient"]) \
        - (response_cross + response_quadratic)
    outputs["cross"] = outputs["cross"] + output_remainder
    slots = {}
    for component, output in outputs.items():
        propagated = _sequentially_propagate(model, output, recipient_slot.dtype)
        slot = recipient_slot.clone()
        slot[:, SUBJECT_POSITION] = propagated[:, SUBJECT_POSITION]
        slots[component] = slot
    closure = (outputs["full"] - outputs["recipient"]) - (
        (outputs["cross"] - outputs["recipient"])
        + (outputs["quadratic"] - outputs["recipient"]))
    return slots, outputs, {
        "output_remainder_max_absolute_value": float(
            output_remainder.detach().abs().max()),
        "output_closure_max_absolute_error": float(closure.detach().abs().max()),
    }


def evaluate(model, torch, F, facade):
    rows = build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = v1.parent.depth.parent.v1._role_batch(rows, torch, device)
    native = v1.factors._native_logits(model, tokens, torch, F)
    replay, captured, projection, closure, mlp8_capture = v1._capture_initial(
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
        mlp, x_recipient, x_opposite, torch, F)
    lexical_products, lexical_algebra = _polarized_products(
        mlp, x_recipient, x_lexical, torch, F)
    native_mlp8_output = mlp8_capture["output"]
    opposite_slots, _, opposite_output = _propagated_slots(
        model, mlp, opposite_products, recipient["M8"], F,
        native_recipient_output=native_mlp8_output[:n],
        native_source_output=native_mlp8_output[n:2*n])
    lexical_slots, _, lexical_output = _propagated_slots(
        model, mlp, lexical_products, recipient["M8"], F,
        native_recipient_output=native_mlp8_output[:n],
        native_source_output=native_mlp8_output[2*n:])
    patch = v1._compile(tokens[:n], recipient, opposite, lexical, attention, projection,
                        opposite_slots, lexical_slots, rows, torch, F)
    native_patch = v1.factors._native_logits(model, patch["tokens"], torch, F)
    patched, _, _, patch_closure = v1.parent._decomposed_forward(
        model, patch["tokens"], patch["finals"], torch, F, facade,
        replacement_heads=patch["replacement_heads"],
        native_reinstall_mask=patch["native_reinstall_mask"])
    expected_heads = {
        "standalone_recipient": v1._parent_head(
            recipient, opposite, "rrrrrrr", attention, projection, torch, F),
        "standalone_full": v1._parent_head(
            recipient, opposite, "rrrrorr", attention, projection, torch, F),
        "conditional_recipient": v1._parent_head(
            recipient, opposite, "ooooroo", attention, projection, torch, F),
        "conditional_full": v1._parent_head(
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
        "output_closure_max_absolute_error": max(
            opposite_output["output_closure_max_absolute_error"],
            lexical_output["output_closure_max_absolute_error"]),
        "propagated_recipient_MLP8_max_absolute_error": float((
            opposite_slots["recipient"][:, SUBJECT_POSITION]
            - recipient["M8"][:, SUBJECT_POSITION]).abs().max()),
        "propagated_source_MLP8_max_absolute_error": max(
            float((opposite_slots["full"][:, SUBJECT_POSITION]
                   - opposite["M8"][:, SUBJECT_POSITION]).abs().max()),
            float((lexical_slots["full"][:, SUBJECT_POSITION]
                   - lexical["M8"][:, SUBJECT_POSITION]).abs().max())),
        "gauge_invariance_max_absolute_error": max(
            opposite_algebra["gauge_invariance_max_absolute_error"],
            lexical_algebra["gauge_invariance_max_absolute_error"]),
        "parent_head_endpoint_max_absolute_error": max(
            float((patch["heads"][condition] - expected).abs().max())
            for condition, expected in expected_heads.items()),
        "same_batch_native_noop_endpoint_max_absolute_error": 0.0,
        "installed_head_max_absolute_error": 0.0,
    }
    numerical_diagnostics = {
        "opposite": {**opposite_algebra, **opposite_output},
        "lexical": {**lexical_algebra, **lexical_output},
    }
    evidence = []
    for out_index, (row_index, condition, cell_id) in enumerate(patch["specs"]):
        base = v1._metrics(native_patch[out_index, SUBJECT_POSITION], rows[row_index],
                           condition, torch)
        value = v1._metrics(patched[out_index, SUBJECT_POSITION], rows[row_index],
                            condition, torch)
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
    return evidence, exactness, numerical_diagnostics


def score(evidence, exactness, bars=BARS):
    return v1.score(evidence, exactness, bars)


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
        raise MLP8PolarizedResponseV2Error(f"refusing to overwrite {OUT}")
    torch, F, facade = v1.factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness, numerical_diagnostics = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    instrument_key = "pred" + "_a_instrument_live"
    terminal = "valid_causal_screen" if scored["predictions"][instrument_key] else "invalid"
    result = {
        "schema": "task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_result_v2",
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256, "score": scored,
        "numerical_diagnostics": numerical_diagnostics,
        "evidence": evidence, "evaluated_splits": ["LICENSED_HOLDOUT"],
        "forbidden_splits_opened": [], "model_forwards": 4,
        "causal_interventions": len(evidence),
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
