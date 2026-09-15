#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_parent_replays pred_b_rank2_response_improves pred_c_rank2_program_proxy
"""Cross-construction rank-2 recipient-state proxy for the L11H3 response interaction."""
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
PREREG = POLY / "SUBJECT_NUMBER_RANK2_RECIPIENT_STATE_RESPONSE_PROXY_V1_PREREGISTRATION.md"
LAW = POLY / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json"
AXIS = POLY / "SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_ARTIFACT.json"
DISCOVERY = ROOT / "circuits/fast_screens/subject_number_native_head_response_coordinate_discovery_v2_result.json"
MEAN_PROXY = ROOT / "circuits/fast_screens/subject_number_donor_free_head_response_proxy_v2_result.json"
BINDING = POLY / "SUBJECT_NUMBER_RANK2_RECIPIENT_STATE_RESPONSE_PROXY_V1_BINDING.json"
OUT = ROOT / "circuits/fast_screens/subject_number_rank2_recipient_state_response_proxy_v1_result.json"
PRICE = {"physical_model_forwards": 1, "role_sequences": 96,
         "rank2_predictive_map_fits": 4, "recipient_state_rank": 2,
         "displacement_rank": 2, "map_width": 1152,
         "offline_head_function_row_evaluations": 2048,
         "coefficient_fits": 0, "backwards": 0, "parameter_updates": 0}
BASELINE_ERROR = .6093072967652493
ORACLE_ERROR = .37689362716534275
PREDICTION_REGISTRY = {"pred_a_parent_replays": None,
                       "pred_b_rank2_response_improves": None,
                       "pred_c_rank2_program_proxy": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"preregistration": PREREG, "law": LAW, "native_axis": AXIS,
             "discovery": DISCOVERY, "mean_proxy": MEAN_PROXY,
             "authority": Path(authority.__file__)}
    if binding["files"] != {key: sha(path) for key, path in paths.items()} or binding["price"] != PRICE:
        raise ValueError("binding changed")
    law, axis, discovery = (json.loads(path.read_text()) for path in (LAW, AXIS, DISCOVERY))
    mean_proxy = json.loads(MEAN_PROXY.read_text())
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
    return {"schema": "subject_number_rank2_recipient_state_response_proxy_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "role_sequences": 3 * len(rows),
            "background_subsets": list(factor_gate.BACKGROUND_SUBSETS),
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
        base_all = factor_gate._raw_for(ir, io, "", F)
        yz_all = factor_gate._raw_for(ir, io, "YZ", F)
        delta_all = yz_all - base_all
        axis = torch.tensor(axis_artifact["axis"], device=device, dtype=base_all.dtype)
        templates_np = np.asarray([row["template_id"] for row in rows])
        directions_np = np.asarray([row["direction_id"] for row in rows])
        targets, exact_s_all, mean_s_all, rank2_s_all = [], [], [], []
        exact_alpha_all, mean_alpha_all, rank2_alpha_all, native_alpha_all = [], [], [], [] , []
        evidence, map_audit = [], {}
        evaluations = 0
        for held_out in sorted(set(templates_np)):
            test_np = templates_np == held_out; train_np = ~test_np
            test_ids = np.flatnonzero(test_np)
            test = torch.tensor(test_ids, device=device, dtype=torch.long)
            fold_recipient, fold_opposite = select(recipient, test), select(opposite, test)
            fold_ir, fold_io = select(ir, test), select(io, test)
            function = tangent._head_function(model, fold_recipient, fold_opposite,
                                              model.transformer.h[tangent.parent.LAYER].attn,
                                              projection, torch, F)
            maps = {}
            for direction in sorted(set(directions_np)):
                train_ids = np.flatnonzero(train_np & (directions_np == direction))
                train_index = torch.tensor(train_ids, device=device, dtype=torch.long)
                xs, ys = [], []
                for subset in factor_gate.BACKGROUND_SUBSETS:
                    xb = factor_gate._raw_for(select(ir, train_index), select(io, train_index), subset, F)
                    xy = factor_gate._raw_for(select(ir, train_index), select(io, train_index), subset + "YZ", F)
                    xs.append(xb); ys.append(xy - xb)
                xtrain, ytrain = torch.cat(xs), torch.cat(ys)
                xmean, ymean = xtrain.mean(0), ytrain.mean(0)
                xc, yc = xtrain - xmean, ytrain - ymean
                _, sx, vxh = torch.linalg.svd(xc, full_matrices=False)
                _, sy, vyh = torch.linalg.svd(yc, full_matrices=False)
                vx, vy = vxh[:2].T, vyh[:2].T
                r, d = xc @ vx, yc @ vy
                gram = r.T @ r
                ridge = 1e-3 * torch.trace(gram) / 2
                a = torch.linalg.solve(gram + ridge * torch.eye(2, device=device, dtype=gram.dtype), r.T @ d)
                mean_p = delta_all[train_index].mean(0)
                maps[direction] = (xmean, ymean, vx, vy, a, mean_p)
                packed = torch.cat([xmean, ymean, vx.flatten(), vy.flatten(), a.flatten()]).detach().float().cpu().numpy().tobytes()
                map_audit[f"{held_out}.{direction}"] = {
                    "training_rows": len(train_ids), "training_background_examples": len(xtrain),
                    "ridge": float(ridge), "x_top2_energy_fraction": float(sx[:2].square().sum() / sx.square().sum()),
                    "delta_top2_energy_fraction": float(sy[:2].square().sum() / sy.square().sum().clamp_min(1e-30)),
                    "packed_float32_sha256": hashlib.sha256(packed).hexdigest()}
            fold_rows = [rows[i] for i in test_ids]
            beta = np.asarray(discovery["reports"]["joint_interaction"]["folds"][held_out]["beta"])
            native_beta = np.asarray(discovery["reports"]["native_baseline"]["folds"][held_out]["beta"])
            for subset in factor_gate.BACKGROUND_SUBSETS:
                xb = factor_gate._raw_for(fold_ir, fold_io, subset, F)
                xb_yz = factor_gate._raw_for(fold_ir, fold_io, subset + "YZ", F)
                mean_batch, rank2_delta = [], []
                for j, row in enumerate(fold_rows):
                    xmean, ymean, vx, vy, a, mean_p = maps[row["direction_id"]]
                    score = (xb[j] - xmean) @ vx
                    rank2_delta.append(ymean + (score @ a) @ vy.T)
                    mean_batch.append(mean_p)
                mean_batch, rank2_delta = torch.stack(mean_batch), torch.stack(rank2_delta)
                h0, h1 = function(xb), function(xb_yz)
                hm, hr = function(xb + mean_batch), function(xb + rank2_delta)
                evaluations += 4 * len(fold_rows)
                z = (h0 @ axis).detach().double().cpu().numpy()
                exact_s = ((h1 - h0) @ axis).detach().double().cpu().numpy()
                mean_s = ((hm - h0) @ axis).detach().double().cpu().numpy()
                rank2_s = ((hr - h0) @ axis).detach().double().cpu().numpy()
                def program(s):
                    return np.c_[np.ones(len(z)), z, s, z * s] @ beta
                exact_alpha, mean_alpha, rank2_alpha = program(exact_s), program(mean_s), program(rank2_s)
                native_alpha = np.c_[np.ones(len(z)), z] @ native_beta
                for j, row in enumerate(fold_rows):
                    target = law["predicted_coefficients"][f"{row['direction_id']}.cardinality_{len(subset)}"]
                    targets.append(target); exact_s_all.append(exact_s[j]); mean_s_all.append(mean_s[j]); rank2_s_all.append(rank2_s[j])
                    exact_alpha_all.append(exact_alpha[j]); mean_alpha_all.append(mean_alpha[j]); rank2_alpha_all.append(rank2_alpha[j]); native_alpha_all.append(native_alpha[j])
                    evidence.append({"held_out_template": held_out, "row_id": row["row_id"], "direction": row["direction_id"],
                                     "background": subset, "cardinality": len(subset), "z": float(z[j]),
                                     "exact_s": float(exact_s[j]), "mean_s": float(mean_s[j]), "rank2_s": float(rank2_s[j]),
                                     "target_alpha": float(target), "exact_alpha": float(exact_alpha[j]),
                                     "mean_alpha": float(mean_alpha[j]), "rank2_alpha": float(rank2_alpha[j]),
                                     "native_alpha": float(native_alpha[j])})
    exact_program = stats(targets, exact_alpha_all)
    mean_response = stats(exact_s_all, mean_s_all); rank2_response = stats(exact_s_all, rank2_s_all)
    mean_program = stats(targets, mean_alpha_all); rank2_program = stats(targets, rank2_alpha_all)
    native_program = stats(targets, native_alpha_all)
    registered_oracle = discovery["reports"]["joint_interaction"]["cross_construction"]
    exact_replay = max(abs(exact_program[k] - registered_oracle[k]) for k in ("cosine", "relative_l2_error", "sign_agreement"))
    mean_replays = [abs(mean_response[k] - mean_parent["response_proxy_metrics"][k]) for k in ("cosine", "relative_l2_error", "sign_agreement")]
    mean_replays += [abs(mean_program[k] - mean_parent["proxy_program_metrics"][k]) for k in ("cosine", "relative_l2_error", "sign_agreement")]
    mean_replay = max(mean_replays)
    response_improvement = mean_response["relative_l2_error"] - rank2_response["relative_l2_error"]
    program_improvement = BASELINE_ERROR - rank2_program["relative_l2_error"]
    program_degradation = rank2_program["relative_l2_error"] - ORACLE_ERROR
    instrument = (len(evidence) == 512 and len(tokens) == PRICE["role_sequences"] and len(map_audit) == 4
                  and evaluations == PRICE["offline_head_function_row_evaluations"] and exact_replay <= 1e-8
                  and mean_replay <= 1e-8 and max(closure["input_state_closure_max_absolute_error"],
                  closure["input_normalized_closure_max_absolute_error"]) <= 5e-5)
    response_pass = response_improvement >= .05 and rank2_response["cosine"] >= .95
    program_pass = rank2_program["relative_l2_error"] <= .45 and program_improvement >= .10 and program_degradation <= .10
    predictions = {"pred_a_parent_replays": bool(instrument),
                   "pred_b_rank2_response_improves": bool(instrument and response_pass),
                   "pred_c_rank2_program_proxy": bool(instrument and program_pass)}
    terminal = "invalid" if not instrument else "rank2_recipient_state_response_proxy_held" if response_pass and program_pass else "rank2_recipient_state_response_proxy_null"
    result = {"schema": "subject_number_rank2_recipient_state_response_proxy_v1_result", "terminal": terminal,
              "predictions": predictions, "exact_program_metrics": exact_program,
              "mean_response_metrics": mean_response, "rank2_response_metrics": rank2_response,
              "mean_program_metrics": mean_program, "rank2_program_metrics": rank2_program,
              "native_program_metrics": native_program, "rank2_response_absolute_improvement": response_improvement,
              "rank2_program_absolute_improvement_over_native": program_improvement,
              "rank2_program_absolute_degradation_from_oracle": program_degradation,
              "exact_replay_max_metric_error": exact_replay, "mean_proxy_replay_max_metric_error": mean_replay,
              "map_audit": map_audit, "joined_evidence": evidence,
              "instrument": {"examples": len(evidence), "role_sequences": len(tokens),
                             "offline_head_function_row_evaluations": evaluations, "rank2_predictive_map_fits": len(map_audit),
                             "coefficient_fits": 0, "role_state_closure_max_absolute_error": closure["input_state_closure_max_absolute_error"],
                             "role_normalized_closure_max_absolute_error": closure["input_normalized_closure_max_absolute_error"]},
              "outcome_access": {"behavioral_effects": False, "answer_logits": False, "downstream_causal_outcomes": False,
                                 "fresh_authority": False, "held_out_row_specific_donor_used_by_proxy": False},
              "price": PRICE, "authority_sha256": authority.validate_rows(rows),
              "checkpoint_weights_sha256": checkpoint.weights_sha256, "binding_sha256": sha(BINDING),
              "runner_sha256": sha(RUNNER), "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Outcome-blind cross-construction rank-2 recipient-state prediction of grouped MLP6/7 displacement inside the frozen native-axis response interaction; fresh causal substitution remains required."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("terminal", "predictions", "mean_response_metrics", "rank2_response_metrics",
                                             "mean_program_metrics", "rank2_program_metrics", "native_program_metrics",
                                             "rank2_response_absolute_improvement", "rank2_program_absolute_improvement_over_native",
                                             "rank2_program_absolute_degradation_from_oracle", "exact_replay_max_metric_error",
                                             "mean_proxy_replay_max_metric_error", "instrument")}, indent=2))
    assert instrument

if __name__ == "__main__":
    main()
