#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_parent_replays pred_b_response_weighting_improves pred_c_complete_program_and_null
"""Response-Jacobian-weighted donor-free subject-number prototype."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import circuit_fast_screen_candidate_task14_cardinality_prototype_transfer as authority
import circuit_fast_screen_managed_runner as managed
import run_subject_number_native_head_response_coordinate_discovery_v2 as discovery_runner
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parent.parent
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_V1_PREREGISTRATION.md"
LAW = POLY / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json"
AXIS = POLY / "SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_ARTIFACT.json"
DISCOVERY = ROOT / "circuits/fast_screens/subject_number_native_head_response_coordinate_discovery_v2_result.json"
MEAN_PROXY = ROOT / "circuits/fast_screens/subject_number_donor_free_head_response_proxy_v2_result.json"
BINDING = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_V1_BINDING.json"
OUT = ROOT / "circuits/fast_screens/subject_number_response_weighted_prototype_v1_result.json"
RANK = 16
NULLS = 8
RIDGE_FRACTION = 1e-2
SEED = 20260916
BASELINE_ERROR = .6093072967652493
ORACLE_ERROR = .37689362716534275
MEAN_RESPONSE_ERROR = .2880428105400366
PRICE = {"physical_model_forwards": 1, "role_sequences": 96,
         "response_weighted_fits": 4, "prototype_width": 1152,
         "candidate_rank": RANK, "permutation_controls": NULLS,
         "input_gradient_rows": 512, "coefficient_fits": 0,
         "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_parent_replays": None,
                       "pred_b_response_weighting_improves": None,
                       "pred_c_complete_program_and_null": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"preregistration": PREREG, "law": LAW, "native_axis": AXIS,
             "discovery": DISCOVERY, "mean_proxy": MEAN_PROXY,
             "authority": Path(authority.__file__)}
    if binding["files"] != {key: sha(path) for key, path in paths.items()} \
            or binding["price"] != PRICE or binding["rank"] != RANK \
            or binding["nulls"] != NULLS or binding["ridge_fraction"] != RIDGE_FRACTION \
            or binding["seed"] != SEED:
        raise ValueError("binding changed")
    law, axis, discovery, mean_proxy = (
        json.loads(path.read_text()) for path in (LAW, AXIS, DISCOVERY, MEAN_PROXY))
    if law["terminal"] != "bilinear_scalar_law_frozen_weights_only" \
            or axis["terminal"] != "native_weight_axis_frozen" \
            or discovery["terminal"] != "native_head_response_coordinate_selected" \
            or discovery["selected_form"] != "joint_interaction" \
            or mean_proxy["terminal"] != "donor_free_head_response_proxy_null":
        raise ValueError("parent status changed")
    return binding, law, axis, discovery, mean_proxy


def plan():
    binding, _, _, _, _ = load_bound()
    rows = authority.build_rows()
    return {"schema": "subject_number_response_weighted_prototype_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "role_sequences": 3 * len(rows),
            "background_subsets": list(factor_gate.BACKGROUND_SUBSETS),
            "rank": RANK, "permutation_controls": NULLS,
            "folds": sorted({row["template_id"] for row in rows}), "price": PRICE,
            "authority_sha256": authority.validate_rows(rows),
            "binding_sha256": sha(BINDING), "bound_files": sorted(binding["files"])}


def stats(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    yn, pn = np.linalg.norm(y), np.linalg.norm(prediction)
    return {"count": int(len(y)), "cosine": float(y @ prediction / max(yn * pn, 1e-30)),
            "relative_l2_error": float(np.linalg.norm(y - prediction) / max(yn, 1e-30)),
            "sign_agreement": float(np.mean((y > 0) == (prediction > 0)))}


def select(values, index):
    return {key: value[index] for key, value in values.items()}


def ridge_coefficients(design, target, ridge_fraction=RIDGE_FRACTION):
    gram = design.T @ design
    ridge = ridge_fraction * np.trace(gram) / design.shape[1]
    coefficients = np.linalg.solve(gram + ridge * np.eye(design.shape[1]), design.T @ target)
    return coefficients, float(ridge)


@np.errstate(all="raise")
def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    binding, law, axis_artifact, discovery, mean_parent = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    rows = authority.build_rows(); n = len(rows); device = next(model.parameters()).device
    tokens, finals = tangent.parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    with torch.no_grad():
        _, captured, projection, closure, inputs = tangent.parent._decomposed_forward(
            model, tokens, finals, torch, F, facade)
    recipient = tangent._role_slice(captured, 0, n)
    opposite = tangent._role_slice(captured, n, 2 * n)
    ir = tangent._role_slice(inputs, 0, n)
    io = tangent._role_slice(inputs, n, 2 * n)
    with torch.no_grad():
        base_all = factor_gate._raw_for(ir, io, "", F)
        yz_all = factor_gate._raw_for(ir, io, "YZ", F)
        delta_all = yz_all - base_all
    axis = torch.tensor(axis_artifact["axis"], device=device, dtype=base_all.dtype)
    templates_np = np.asarray([row["template_id"] for row in rows])
    directions_np = np.asarray([row["direction_id"] for row in rows])
    targets, exact_s_all, mean_s_all, candidate_s_all = [], [], [], []
    exact_alpha_all, mean_alpha_all, candidate_alpha_all, native_alpha_all = [], [], [], []
    null_alpha_all = [[] for _ in range(NULLS)]
    evidence, fit_audit = [], {}
    gradient_rows = 0
    rng = np.random.default_rng(SEED)
    for held_out in sorted(set(templates_np)):
        test_np = templates_np == held_out; train_np = ~test_np
        test_ids = np.flatnonzero(test_np)
        test = torch.tensor(test_ids, device=device, dtype=torch.long)
        fold_recipient, fold_opposite = select(recipient, test), select(opposite, test)
        fold_ir, fold_io = select(ir, test), select(io, test)
        test_function = tangent._head_function(model, fold_recipient, fold_opposite,
                                               model.transformer.h[tangent.parent.LAYER].attn,
                                               projection, torch, F)
        prototypes = {}
        null_prototypes = {}
        for direction in sorted(set(directions_np)):
            train_ids = np.flatnonzero(train_np & (directions_np == direction))
            train_index = torch.tensor(train_ids, device=device, dtype=torch.long)
            train_recipient, train_opposite = select(recipient, train_index), select(opposite, train_index)
            train_ir, train_io = select(ir, train_index), select(io, train_index)
            train_function = tangent._head_function(model, train_recipient, train_opposite,
                                                    model.transformer.h[tangent.parent.LAYER].attn,
                                                    projection, torch, F)
            mean_p = delta_all[train_index].mean(0)
            displacements = []
            for subset in factor_gate.BACKGROUND_SUBSETS:
                xb = factor_gate._raw_for(train_ir, train_io, subset, F)
                xy = factor_gate._raw_for(train_ir, train_io, subset + "YZ", F)
                displacements.append(xy - xb)
            centered = torch.cat(displacements) - mean_p
            _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
            modes = vh[:RANK].T.contiguous()
            designs, residuals = [], []
            for subset in factor_gate.BACKGROUND_SUBSETS:
                xb = factor_gate._raw_for(train_ir, train_io, subset, F)
                xy = factor_gate._raw_for(train_ir, train_io, subset + "YZ", F)
                with torch.no_grad():
                    h0 = train_function(xb)
                    h1 = train_function(xy)
                    hm = train_function(xb + mean_p)
                    exact_s = (h1 - h0) @ axis
                    mean_s = (hm - h0) @ axis
                point = (xb + mean_p).detach().requires_grad_(True)
                scalar = (train_function(point) @ axis).sum()
                gradient = torch.autograd.grad(scalar, point, create_graph=False)[0]
                designs.append(gradient @ modes)
                residuals.append(exact_s - mean_s)
                gradient_rows += len(xb)
            design = torch.cat(designs).detach().double().cpu().numpy()
            residual = torch.cat(residuals).detach().double().cpu().numpy()
            coefficients, ridge = ridge_coefficients(design, residual)
            correction = modes @ torch.tensor(coefficients, device=device, dtype=modes.dtype)
            candidate = mean_p + correction
            nulls = []
            for _ in range(NULLS):
                permuted = residual[rng.permutation(len(residual))]
                null_coefficients, _ = ridge_coefficients(design, permuted)
                nulls.append(mean_p + modes @ torch.tensor(null_coefficients, device=device, dtype=modes.dtype))
            prototypes[direction] = (mean_p, candidate)
            null_prototypes[direction] = nulls
            fit_audit[f"{held_out}.{direction}"] = {
                "training_rows": len(train_ids), "training_background_examples": len(design),
                "rank": RANK, "ridge": ridge,
                "top_rank_energy_fraction": float(singular[:RANK].square().sum() / singular.square().sum().clamp_min(1e-30)),
                "mean_norm": float(mean_p.norm()), "candidate_norm": float(candidate.norm()),
                "norm_ratio": float(candidate.norm() / mean_p.norm().clamp_min(1e-30)),
                "training_response_residual_l2": float(np.linalg.norm(residual)),
                "design_sha256": hashlib.sha256(design.astype(np.float32).tobytes()).hexdigest(),
                "candidate_sha256": hashlib.sha256(candidate.detach().float().cpu().numpy().tobytes()).hexdigest()}
        fold_rows = [rows[i] for i in test_ids]
        mean_batch = torch.stack([prototypes[row["direction_id"]][0] for row in fold_rows])
        candidate_batch = torch.stack([prototypes[row["direction_id"]][1] for row in fold_rows])
        null_batches = [torch.stack([null_prototypes[row["direction_id"]][k] for row in fold_rows])
                        for k in range(NULLS)]
        beta = np.asarray(discovery["reports"]["joint_interaction"]["folds"][held_out]["beta"])
        native_beta = np.asarray(discovery["reports"]["native_baseline"]["folds"][held_out]["beta"])
        for subset in factor_gate.BACKGROUND_SUBSETS:
            xb = factor_gate._raw_for(fold_ir, fold_io, subset, F)
            xy = factor_gate._raw_for(fold_ir, fold_io, subset + "YZ", F)
            with torch.no_grad():
                h0, h1 = test_function(xb), test_function(xy)
                hm, hc = test_function(xb + mean_batch), test_function(xb + candidate_batch)
                hn = [test_function(xb + batch) for batch in null_batches]
            z = (h0 @ axis).detach().double().cpu().numpy()
            exact_s = ((h1 - h0) @ axis).detach().double().cpu().numpy()
            mean_s = ((hm - h0) @ axis).detach().double().cpu().numpy()
            candidate_s = ((hc - h0) @ axis).detach().double().cpu().numpy()
            null_s = [((value - h0) @ axis).detach().double().cpu().numpy() for value in hn]
            def program(s):
                return np.c_[np.ones(len(z)), z, s, z * s] @ beta
            exact_alpha, mean_alpha, candidate_alpha = program(exact_s), program(mean_s), program(candidate_s)
            null_alpha = [program(value) for value in null_s]
            native_alpha = np.c_[np.ones(len(z)), z] @ native_beta
            for j, row in enumerate(fold_rows):
                target = law["predicted_coefficients"][f"{row['direction_id']}.cardinality_{len(subset)}"]
                targets.append(target); exact_s_all.append(exact_s[j]); mean_s_all.append(mean_s[j]); candidate_s_all.append(candidate_s[j])
                exact_alpha_all.append(exact_alpha[j]); mean_alpha_all.append(mean_alpha[j]); candidate_alpha_all.append(candidate_alpha[j]); native_alpha_all.append(native_alpha[j])
                for k in range(NULLS):
                    null_alpha_all[k].append(null_alpha[k][j])
                evidence.append({"held_out_template": held_out, "row_id": row["row_id"],
                                 "direction": row["direction_id"], "background": subset,
                                 "cardinality": len(subset), "z": float(z[j]),
                                 "exact_s": float(exact_s[j]), "mean_s": float(mean_s[j]),
                                 "candidate_s": float(candidate_s[j]), "target_alpha": float(target),
                                 "exact_alpha": float(exact_alpha[j]), "mean_alpha": float(mean_alpha[j]),
                                 "candidate_alpha": float(candidate_alpha[j]), "native_alpha": float(native_alpha[j])})
    exact_program = stats(targets, exact_alpha_all)
    mean_response = stats(exact_s_all, mean_s_all); candidate_response = stats(exact_s_all, candidate_s_all)
    mean_program = stats(targets, mean_alpha_all); candidate_program = stats(targets, candidate_alpha_all)
    native_program = stats(targets, native_alpha_all)
    null_programs = [stats(targets, values) for values in null_alpha_all]
    null_errors = [value["relative_l2_error"] for value in null_programs]
    registered_oracle = discovery["reports"]["joint_interaction"]["cross_construction"]
    exact_replay = max(abs(exact_program[k] - registered_oracle[k]) for k in ("cosine", "relative_l2_error", "sign_agreement"))
    mean_replays = [abs(mean_response[k] - mean_parent["response_proxy_metrics"][k]) for k in ("cosine", "relative_l2_error", "sign_agreement")]
    mean_replays += [abs(mean_program[k] - mean_parent["proxy_program_metrics"][k]) for k in ("cosine", "relative_l2_error", "sign_agreement")]
    mean_replay = max(mean_replays)
    response_improvement = mean_response["relative_l2_error"] - candidate_response["relative_l2_error"]
    program_improvement = BASELINE_ERROR - candidate_program["relative_l2_error"]
    program_degradation = candidate_program["relative_l2_error"] - ORACLE_ERROR
    null_median = float(np.median(null_errors)); null_advantage = null_median - candidate_program["relative_l2_error"]
    finite = np.isfinite(np.asarray([*exact_s_all, *mean_s_all, *candidate_s_all, *candidate_alpha_all])).all()
    norm_ok = max(value["norm_ratio"] for value in fit_audit.values()) <= 2
    instrument = (len(evidence) == 512 and len(tokens) == PRICE["role_sequences"] and
                  len(fit_audit) == PRICE["response_weighted_fits"] and gradient_rows == PRICE["input_gradient_rows"] and
                  exact_replay <= 1e-8 and mean_replay <= 1e-8 and finite and norm_ok and
                  max(closure["input_state_closure_max_absolute_error"], closure["input_normalized_closure_max_absolute_error"]) <= 5e-5)
    response_pass = response_improvement >= .03 and candidate_response["cosine"] >= .95
    program_pass = candidate_program["relative_l2_error"] <= .45 and program_improvement >= .10 and program_degradation <= .10
    null_pass = null_advantage >= .02
    predictions = {"pred_a_parent_replays": bool(instrument),
                   "pred_b_response_weighting_improves": bool(instrument and response_pass),
                   "pred_c_complete_program_and_null": bool(instrument and program_pass and null_pass)}
    terminal = "invalid" if not instrument else "response_weighted_prototype_held" if response_pass and program_pass and null_pass else "response_weighted_prototype_null"
    result = {"schema": "subject_number_response_weighted_prototype_v1_result", "terminal": terminal,
              "predictions": predictions, "exact_program_metrics": exact_program,
              "mean_response_metrics": mean_response, "candidate_response_metrics": candidate_response,
              "mean_program_metrics": mean_program, "candidate_program_metrics": candidate_program,
              "native_program_metrics": native_program, "permutation_program_metrics": null_programs,
              "response_absolute_improvement": response_improvement,
              "program_absolute_improvement_over_native": program_improvement,
              "program_absolute_degradation_from_oracle": program_degradation,
              "permutation_median_program_error": null_median, "permutation_median_advantage": null_advantage,
              "exact_replay_max_metric_error": exact_replay, "mean_proxy_replay_max_metric_error": mean_replay,
              "fit_audit": fit_audit, "joined_evidence": evidence,
              "instrument": {"examples": len(evidence), "role_sequences": len(tokens), "input_gradient_rows": gradient_rows,
                             "response_weighted_fits": len(fit_audit), "rank": RANK, "permutation_controls": NULLS,
                             "prototype_norm_gate": norm_ok,
                             "role_state_closure_max_absolute_error": closure["input_state_closure_max_absolute_error"],
                             "role_normalized_closure_max_absolute_error": closure["input_normalized_closure_max_absolute_error"]},
              "outcome_access": {"behavioral_effects": False, "answer_logits": False, "downstream_causal_outcomes": False,
                                 "fresh_authority": False, "held_out_row_specific_donor_used_by_prototype": False,
                                 "training_donor_response_used_for_discovery": True},
              "price": PRICE, "authority_sha256": authority.validate_rows(rows),
              "checkpoint_weights_sha256": checkpoint.weights_sha256, "binding_sha256": sha(BINDING),
              "runner_sha256": sha(RUNNER), "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Opened-authority response-Jacobian selection of one donor-free fixed grouped-MLP6/7 prototype per direction; fresh causal substitution remains required."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("terminal", "predictions", "mean_response_metrics", "candidate_response_metrics",
                                                    "mean_program_metrics", "candidate_program_metrics", "native_program_metrics",
                                                    "response_absolute_improvement", "program_absolute_improvement_over_native",
                                                    "program_absolute_degradation_from_oracle", "permutation_median_program_error",
                                                    "permutation_median_advantage", "exact_replay_max_metric_error",
                                                    "mean_proxy_replay_max_metric_error", "instrument")}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
