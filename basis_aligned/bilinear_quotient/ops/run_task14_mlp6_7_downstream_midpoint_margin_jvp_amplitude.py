#!/usr/bin/env python3
"""Generate grouped-MLP6--7 causal amplitudes from the native downstream margin JVP."""

# BQGATE: EXPERIMENT pred_a_gradient_instrument pred_b_midpoint_generates_amplitude pred_c_midpoint_generalizes_by_phase pred_d_both_backgrounds_are_predictable pred_e_midpoint_improves_endpoint pred_f_holdout_amplitude_beats_fit_mean
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_pristine_split_mlp6_7_absolute_composition as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as gate


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1.json"
PARENT_RESULT = ROOT / "circuits/fast_screens/task14_pristine_split_mlp6_7_absolute_composition_transfer_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1_result.json"
PRIOR_ART_SHA256 = "a9d0264529445482e2bf9cfe212c2f0231df33e0d2769f449c78e214440c059c"
PARENT_RESULT_SHA256 = "0053270e3c8179a2c9d1c71a1578cff5d992de1c39b199b2228cca1bcab8d920"
CANDIDATE_ID = "subject_verb.number_agreement.mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1"
BACKGROUNDS = ("", "EAUW")
POINTS = ("base", "midpoint")
BARS = {
    "maximum_numerical_absolute_error": 5e-5,
    "minimum_overall_cosine": .90,
    "maximum_overall_relative_l2_error": .30,
    "minimum_overall_sign_agreement": .90,
    "minimum_group_cosine": .85,
    "maximum_group_relative_l2_error": .35,
    "minimum_phase_sign_agreement": .875,
    "minimum_midpoint_error_reduction_over_base": .20,
    "minimum_holdout_sse_reduction_over_fit_mean": .30,
}


class DownstreamAmplitudeError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_price(row_count=40):
    amplitudes = row_count * len(BACKGROUNDS)
    role_rows = row_count * len(authority.ROLES)
    return {"physical_model_forwards": 1 + len(POINTS),
            "example_evaluations": role_rows + len(POINTS) * amplitudes,
            "backwards": len(POINTS), "causal_interventions": 0,
            "parameter_updates": 0,
            "predicted_amplitudes_per_linearization": amplitudes}


def validate_preflight():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise DownstreamAmplitudeError("prior-art receipt changed")
    if _sha256(PARENT_RESULT) != PARENT_RESULT_SHA256:
        raise DownstreamAmplitudeError("frozen causal target changed")
    parent = json.loads(PARENT_RESULT.read_text())
    if parent.get("terminal") != "valid_causal_screen" \
            or parent.get("score", {}).get("predictions", {}).get(
                "pred_b_causal_instrument") is not True:
        raise DownstreamAmplitudeError("parent causal instrument is not valid")
    if derive_price() != {"physical_model_forwards": 3,
            "example_evaluations": 280, "backwards": 2,
            "causal_interventions": 0, "parameter_updates": 0,
            "predicted_amplitudes_per_linearization": 80}:
        raise DownstreamAmplitudeError("derived price changed")


def compile_plan():
    validate_preflight()
    return {"schema": "task14_mlp6_7_downstream_midpoint_margin_jvp_amplitude_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "data_status": "RETROSPECTIVE_FROZEN_CAUSAL_TARGETS_NEW_DOWNSTREAM_GRADIENTS",
        "row_count": 40, "backgrounds": list(BACKGROUNDS),
        "linearization_points": list(POINTS),
        "prior_art_sha256": PRIOR_ART_SHA256,
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "predictor": "native answer-margin gradient at L11H3 dotted with exact grouped-MLP6--7 head displacement",
        "target_q_inputs_to_predictor": 0, "bars": dict(BARS),
        "price": derive_price(),
        "closed_claims": ["prospective_causal_validation", "semantic_uniqueness",
            "rank", "compression", "activation_reconstruction", "literal_compute_saving"]}


def _frozen_targets():
    result = json.loads(PARENT_RESULT.read_text())
    grouped = defaultdict(dict)
    for item in result["evidence"]:
        if item["source"] == "opposite" and item["background"] in BACKGROUNDS:
            grouped[(item["row_id"], item["background"])][item["method"]] = float(
                item["target_margin_improvement"])
    expected = {(row["row_id"], background)
                for row in authority.build_rows() for background in BACKGROUNDS}
    if set(grouped) != expected or any(set(methods) != {"base", "exact"}
                                       for methods in grouped.values()):
        raise DownstreamAmplitudeError("frozen target endpoints incomplete")
    return {key: methods["exact"] - methods["base"] for key, methods in grouped.items()}


def _vector_stats(actual, predicted):
    if len(actual) != len(predicted) or not actual:
        raise DownstreamAmplitudeError("invalid score vectors")
    dot = sum(a*p for a, p in zip(actual, predicted))
    an = math.sqrt(sum(a*a for a in actual)); pn = math.sqrt(sum(p*p for p in predicted))
    err = math.sqrt(sum((p-a)**2 for a, p in zip(actual, predicted)))
    sign = sum((a > 0) == (p > 0) for a, p in zip(actual, predicted)) / len(actual)
    return {"count": len(actual), "cosine": dot / max(an*pn, 1e-30),
        "relative_l2_error": err / max(an, 1e-30), "sign_agreement": sign,
        "actual_l2_norm": an, "predicted_l2_norm": pn}


def _signed_margins(logits, specs, rows, torch):
    values = []
    for index, (row_index, _) in enumerate(specs):
        endpoint = rows[row_index]["endpoints"]["opposite_same_lemma"]
        values.append(logits[index, tangent.parent.SUBJECT_POSITION, endpoint["answer_id"]]
                      - logits[index, tangent.parent.SUBJECT_POSITION, endpoint["foil_id"]])
    return torch.stack(values)


def evaluate(model, torch, F, facade):
    rows = authority.build_rows(); n = len(rows); parent = tangent.parent
    device = next(model.parameters()).device
    tokens, finals = parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    replay, captured, projection, role_closure, inputs = parent._decomposed_forward(
        model, tokens, finals, torch, F, facade)
    roles = {"recipient": tangent._role_slice(captured, 0, n),
             "opposite": tangent._role_slice(captured, n, 2*n)}
    input_roles = {"recipient": tangent._role_slice(inputs, 0, n),
                   "opposite": tangent._role_slice(inputs, n, 2*n)}
    function = tangent._head_function(model, roles["recipient"], roles["opposite"],
        model.transformer.h[parent.LAYER].attn, projection, torch, F)

    bases, exacts, specs = [], [], []
    with torch.no_grad():
        for row_index, _row in enumerate(rows):
            for background in BACKGROUNDS:
                base_raw = gate._raw_for(input_roles["recipient"], input_roles["opposite"],
                                         background, F)
                exact_raw = gate._raw_for(input_roles["recipient"], input_roles["opposite"],
                                          background + "YZ", F)
                bases.append(function(base_raw)[row_index])
                exacts.append(function(exact_raw)[row_index])
                specs.append((row_index, background))
    base_heads = torch.stack(bases).detach(); exact_heads = torch.stack(exacts).detach()
    delta = exact_heads - base_heads
    index = torch.tensor([row_index for row_index, _ in specs], dtype=torch.long, device=device)
    patch_tokens = tokens[:n][index]
    patch_finals = torch.full_like(index, parent.SUBJECT_POSITION)
    mask = torch.zeros(len(specs), dtype=torch.bool, device=device)

    predictions, downstream_closures, gradient_stats = {}, {}, {}
    for point in POINTS:
        center = base_heads if point == "base" else base_heads + .5 * delta
        replacement = center.detach().clone().requires_grad_(True)
        logits, _, _, closure = parent.downstream._decomposed_forward(
            model, patch_tokens, patch_finals, torch, F, facade,
            replacement_heads=replacement, native_reinstall_mask=mask)
        margins = _signed_margins(logits, specs, rows, torch)
        gradient = torch.autograd.grad(margins.sum(), replacement)[0]
        predicted = (gradient * delta).sum(dim=-1)
        predictions[point] = predicted.detach().cpu().tolist()
        downstream_closures[point] = closure
        gradient_stats[point] = {"finite": bool(torch.isfinite(gradient).all()),
            "l2_norm": float(torch.linalg.vector_norm(gradient)),
            "nonzero_row_count": int((torch.linalg.vector_norm(gradient, dim=-1) > 0).sum())}

    exactness = {
        "role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"],
        "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"],
        "downstream_state_closure_max_absolute_error": max(
            value["state_sum_max_absolute_error"] for value in downstream_closures.values()),
        "downstream_normalized_closure_max_absolute_error": max(
            value["normalized_state_max_absolute_error"] for value in downstream_closures.values()),
    }
    targets = _frozen_targets()
    evidence = []
    for output_index, (row_index, background) in enumerate(specs):
        row = rows[row_index]
        evidence.append({"row_id": row["row_id"], "phase": row["phase"],
            "direction": row["direction_id"], "template": row["template_id"],
            "background": background, "actual_q": targets[(row["row_id"], background)],
            "base_jvp_q": predictions["base"][output_index],
            "midpoint_jvp_q": predictions["midpoint"][output_index]})
    return evidence, exactness, gradient_stats


def score(evidence, exactness, gradient_stats, bars=BARS):
    actual = [item["actual_q"] for item in evidence]
    stats = {point: _vector_stats(actual, [item[f"{point}_jvp_q"] for item in evidence])
             for point in POINTS}
    phase = {}; background = {}
    for phase_name in ("FIT", "HOLDOUT"):
        selected = [item for item in evidence if item["phase"] == phase_name]
        phase[phase_name] = _vector_stats([x["actual_q"] for x in selected],
                                         [x["midpoint_jvp_q"] for x in selected])
    for background_name in BACKGROUNDS:
        selected = [item for item in evidence if item["background"] == background_name]
        background[background_name or "empty"] = _vector_stats(
            [x["actual_q"] for x in selected], [x["midpoint_jvp_q"] for x in selected])

    fit_means = {}
    for item in evidence:
        if item["phase"] == "FIT":
            fit_means.setdefault((item["direction"], item["background"]), []).append(item["actual_q"])
    fit_means = {key: sum(values)/len(values) for key, values in fit_means.items()}
    holdout = [item for item in evidence if item["phase"] == "HOLDOUT"]
    midpoint_sse = sum((x["midpoint_jvp_q"] - x["actual_q"])**2 for x in holdout)
    mean_sse = sum((fit_means[(x["direction"], x["background"])] - x["actual_q"])**2
                   for x in holdout)
    holdout_reduction = 1.0 - midpoint_sse / max(mean_sse, 1e-30)
    midpoint_improvement = 1.0 - stats["midpoint"]["relative_l2_error"] / max(
        stats["base"]["relative_l2_error"], 1e-30)
    instrument = all(value <= bars["maximum_numerical_absolute_error"]
                     for value in exactness.values()) and all(
        value["finite"] and value["l2_norm"] > 0 and value["nonzero_row_count"] == 80
        for value in gradient_stats.values()) and len(evidence) == 80
    overall = stats["midpoint"]
    overall_ok = overall["cosine"] >= bars["minimum_overall_cosine"] and \
        overall["relative_l2_error"] <= bars["maximum_overall_relative_l2_error"] and \
        overall["sign_agreement"] >= bars["minimum_overall_sign_agreement"]
    phase_ok = all(x["cosine"] >= bars["minimum_group_cosine"] and
        x["relative_l2_error"] <= bars["maximum_group_relative_l2_error"] and
        x["sign_agreement"] >= bars["minimum_phase_sign_agreement"] for x in phase.values())
    background_ok = all(x["cosine"] >= bars["minimum_group_cosine"] and
        x["relative_l2_error"] <= bars["maximum_group_relative_l2_error"]
        for x in background.values())
    return {**exactness, "gradient_stats": gradient_stats, "overall": stats,
        "by_phase": phase, "by_background": background,
        "midpoint_error_reduction_over_base": midpoint_improvement,
        "holdout_sse_reduction_over_fit_mean": holdout_reduction,
        "predictions": {"pred_a_gradient_instrument": bool(instrument),
            "pred_b_midpoint_generates_amplitude": bool(instrument and overall_ok),
            "pred_c_midpoint_generalizes_by_phase": bool(instrument and phase_ok),
            "pred_d_both_backgrounds_are_predictable": bool(instrument and background_ok),
            "pred_e_midpoint_improves_endpoint": bool(instrument and midpoint_improvement >= bars["minimum_midpoint_error_reduction_over_base"]),
            "pred_f_holdout_amplitude_beats_fit_mean": bool(instrument and holdout_reduction >= bars["minimum_holdout_sse_reduction_over_fit_mean"])} }


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise DownstreamAmplitudeError(f"refusing to overwrite {OUT}")
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    evidence, exactness, gradient_stats = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, gradient_stats)
    terminal = "valid_diagnostic" if scored["predictions"]["pred_a_gradient_instrument"] else "invalid"
    result = {"schema": "task14_mlp6_7_downstream_midpoint_margin_jvp_amplitude_result_v1",
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": scored, "evidence": evidence,
        "evaluated_splits": ["RETROSPECTIVE_FIT", "RETROSPECTIVE_HOLDOUT"],
        "forbidden_splits_opened": []}
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
        "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
