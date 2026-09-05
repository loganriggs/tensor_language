#!/usr/bin/env python3
"""Seal prospective downstream-JVP amplitude predictions before causal outcomes."""

# BQGATE: EXPERIMENT pred_a_capability_license pred_b_gradient_instrument pred_c_sixty_four_predictions_sealed
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import circuit_fast_screen_candidate_task14_prospective_jvp_amplitude as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as gate
import run_task14_prospective_jvp_amplitude_native_capability as capability


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_prospective_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1.json"
OUT = ROOT / "circuits/fast_screens/task14_prospective_mlp6_7_downstream_midpoint_margin_jvp_amplitude_v1_predictions.json"
PRIOR_ART_SHA256 = "b8e10492f622bb08cb0a2ae4370267e6782f9a4028ed907272d5f2ad04bad030"
CAPABILITY_RESULT_SHA256 = "9ee68c9297995cc5cf1f6a7c29759c7199b258ec35974fdf5c4000d3e5085749"
CAPABILITY_LICENSE_SHA256 = "27acd0cb5e7459630f89188abd2160622e07967ba0ee9194bf26708801fde33c"
BACKGROUNDS = ("", "EAUW")
POINTS = ("base", "midpoint")
MAXIMUM_NUMERICAL_ABSOLUTE_ERROR = 5e-5


class ProspectivePredictionError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_price(row_count=32):
    amplitudes = row_count * len(BACKGROUNDS)
    return {"physical_model_forwards": 1 + len(POINTS),
        "example_evaluations": row_count * len(authority.ROLES) + len(POINTS)*amplitudes,
        "backwards": len(POINTS), "causal_interventions": 0,
        "parameter_updates": 0, "sealed_predictions": amplitudes}


def validate_preflight():
    for path, expected, label in ((PRIOR_ART, PRIOR_ART_SHA256, "prior art"),
        (capability.RESULT, CAPABILITY_RESULT_SHA256, "capability result"),
        (capability.LICENSE, CAPABILITY_LICENSE_SHA256, "capability license")):
        if _sha256(path) != expected:
            raise ProspectivePredictionError(f"{label} changed")
    licensing.validate_causal_preflight(capability.build_gate(), capability.RESULT,
        capability.LICENSE, expected_license_sha256=CAPABILITY_LICENSE_SHA256,
        causal_candidate_id=authority.CAUSAL_CANDIDATE_ID)
    expected = {"physical_model_forwards": 3, "example_evaluations": 224,
        "backwards": 2, "causal_interventions": 0, "parameter_updates": 0,
        "sealed_predictions": 64}
    if derive_price() != expected:
        raise ProspectivePredictionError("derived price changed")


def compile_plan():
    validate_preflight()
    return {"schema": "task14_prospective_mlp6_7_jvp_amplitude_prediction_plan_v1",
        "candidate_id": authority.CAUSAL_CANDIDATE_ID,
        "split": "NEW_PROSPECTIVE_TEXT_PREDICTION_ONLY",
        "causal_outcomes_opened": False, "row_count": 32,
        "backgrounds": list(BACKGROUNDS), "points": list(POINTS),
        "prior_art_sha256": PRIOR_ART_SHA256,
        "capability_result_sha256": CAPABILITY_RESULT_SHA256,
        "capability_license_sha256": CAPABILITY_LICENSE_SHA256,
        "maximum_numerical_absolute_error": MAXIMUM_NUMERICAL_ABSOLUTE_ERROR,
        "predictions": {
            "pred_a_capability_license": "candidate-scoped native capability license validates",
            "pred_b_gradient_instrument": "source/downstream closures <=5e-5 and all gradients finite and nonzero",
            "pred_c_sixty_four_predictions_sealed": "exactly 64 unique base and midpoint JVP amplitudes are atomically written before causal evaluation"},
        "price": derive_price()}


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
    _, captured, projection, role_closure, inputs = parent._decomposed_forward(
        model, tokens, finals, torch, F, facade)
    roles = {"recipient": tangent._role_slice(captured, 0, n),
             "opposite": tangent._role_slice(captured, n, 2*n)}
    input_roles = {"recipient": tangent._role_slice(inputs, 0, n),
                   "opposite": tangent._role_slice(inputs, n, 2*n)}
    function = tangent._head_function(model, roles["recipient"], roles["opposite"],
        model.transformer.h[parent.LAYER].attn, projection, torch, F)
    bases, exacts, specs = [], [], []
    with torch.no_grad():
        base_raw_by_background = {background: gate._raw_for(
            input_roles["recipient"], input_roles["opposite"], background, F)
            for background in BACKGROUNDS}
        exact_raw_by_background = {background: gate._raw_for(
            input_roles["recipient"], input_roles["opposite"], background + "YZ", F)
            for background in BACKGROUNDS}
        base_head_by_background = {background: function(value).detach()
                                   for background, value in base_raw_by_background.items()}
        exact_head_by_background = {background: function(value).detach()
                                    for background, value in exact_raw_by_background.items()}
        for row_index, _row in enumerate(rows):
            for background in BACKGROUNDS:
                bases.append(base_head_by_background[background][row_index])
                exacts.append(exact_head_by_background[background][row_index])
                specs.append((row_index, background))
    base_heads = torch.stack(bases); exact_heads = torch.stack(exacts)
    delta = exact_heads - base_heads
    index = torch.tensor([row_index for row_index, _ in specs], dtype=torch.long, device=device)
    patch_tokens = tokens[:n][index]
    patch_finals = torch.full_like(index, parent.SUBJECT_POSITION)
    mask = torch.zeros(len(specs), dtype=torch.bool, device=device)
    predictions, closures, gradient_stats = {}, {}, {}
    for point in POINTS:
        center = base_heads if point == "base" else base_heads + .5*delta
        replacement = center.detach().clone().requires_grad_(True)
        logits, _, _, closure = parent.downstream._decomposed_forward(
            model, patch_tokens, patch_finals, torch, F, facade,
            replacement_heads=replacement, native_reinstall_mask=mask)
        margins = _signed_margins(logits, specs, rows, torch)
        gradient = torch.autograd.grad(margins.sum(), replacement)[0]
        predictions[point] = (gradient*delta).sum(dim=-1).detach().cpu().tolist()
        closures[point] = closure
        gradient_stats[point] = {"finite": bool(torch.isfinite(gradient).all()),
            "l2_norm": float(torch.linalg.vector_norm(gradient)),
            "nonzero_row_count": int((torch.linalg.vector_norm(gradient, dim=-1) > 0).sum())}
    exactness = {"role_state_closure_max_absolute_error": role_closure["input_state_closure_max_absolute_error"],
        "role_normalized_closure_max_absolute_error": role_closure["input_normalized_closure_max_absolute_error"],
        "downstream_state_closure_max_absolute_error": max(x["state_sum_max_absolute_error"] for x in closures.values()),
        "downstream_normalized_closure_max_absolute_error": max(x["normalized_state_max_absolute_error"] for x in closures.values())}
    evidence = []
    for i, (row_index, background) in enumerate(specs):
        row = rows[row_index]
        evidence.append({"row_id": row["row_id"], "direction": row["direction_id"],
            "template": row["template_id"], "background": background,
            "base_jvp_q": predictions["base"][i], "midpoint_jvp_q": predictions["midpoint"][i],
            "head_delta_l2_norm": float(torch.linalg.vector_norm(delta[i]))})
    instrument = all(value <= MAXIMUM_NUMERICAL_ABSOLUTE_ERROR for value in exactness.values()) \
        and all(x["finite"] and x["l2_norm"] > 0 and x["nonzero_row_count"] == 64
                for x in gradient_stats.values())
    keys = {(x["row_id"], x["background"]) for x in evidence}
    predictions_status = {"pred_a_capability_license": True,
        "pred_b_gradient_instrument": bool(instrument),
        "pred_c_sixty_four_predictions_sealed": bool(len(evidence) == 64 and len(keys) == 64)}
    return evidence, exactness, gradient_stats, predictions_status


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise ProspectivePredictionError(f"refusing to overwrite {OUT}")
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    evidence, exactness, gradient_stats, predictions = evaluate(model, torch, F, facade)
    terminal = "sealed_prediction" if all(predictions.values()) else "invalid"
    result = {"schema": "task14_prospective_mlp6_7_jvp_amplitude_predictions_v1",
        "candidate_id": authority.CAUSAL_CANDIDATE_ID, "terminal": terminal,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "exactness": exactness, "gradient_stats": gradient_stats,
        "predictions": predictions, "evidence": evidence,
        "causal_outcomes_opened": False}
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions,
        "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
