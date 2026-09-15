#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_response_replay pred_b_donor_free_response_proxy pred_c_donor_free_program_proxy
"""Cross-construction donor-free proxy for the selected L11H3 response interaction."""
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
PREREG = POLY / "SUBJECT_NUMBER_DONOR_FREE_HEAD_RESPONSE_PROXY_V1_PREREGISTRATION.md"
CORRECTION = POLY / "SUBJECT_NUMBER_DONOR_FREE_HEAD_RESPONSE_PROXY_V2_CORRECTION.md"
LAW = POLY / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json"
AXIS = POLY / "SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_ARTIFACT.json"
DISCOVERY = ROOT / "circuits/fast_screens/subject_number_native_head_response_coordinate_discovery_v2_result.json"
INVALID_V1 = ROOT / "circuits/fast_screens/subject_number_donor_free_head_response_proxy_v1_result.json"
BINDING = POLY / "SUBJECT_NUMBER_DONOR_FREE_HEAD_RESPONSE_PROXY_V2_BINDING.json"
OUT = ROOT / "circuits/fast_screens/subject_number_donor_free_head_response_proxy_v2_result.json"
PRICE = {"physical_model_forwards": 1, "role_sequences": 96,
         "fold_direction_prototypes": 4, "prototype_width": 1152,
         "offline_head_function_row_evaluations": 1536,
         "scalar_least_squares_fits": 0, "backwards": 0, "parameter_updates": 0}
BASELINE_ERROR = .6093072967652493
ORACLE_ERROR = .37689362716534275
PREDICTION_REGISTRY = {"pred_a_exact_response_replay": None,
                       "pred_b_donor_free_response_proxy": None,
                       "pred_c_donor_free_program_proxy": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"preregistration": PREREG, "correction": CORRECTION,
             "invalid_v1": INVALID_V1, "law": LAW, "native_axis": AXIS,
             "discovery": DISCOVERY, "authority": Path(authority.__file__)}
    if binding["files"] != {key: sha(path) for key, path in paths.items()} or binding["price"] != PRICE:
        raise ValueError("binding changed")
    law, axis, discovery = (json.loads(path.read_text()) for path in (LAW, AXIS, DISCOVERY))
    if law["terminal"] != "bilinear_scalar_law_frozen_weights_only" \
            or axis["terminal"] != "native_weight_axis_frozen" \
            or discovery["terminal"] != "native_head_response_coordinate_selected" \
            or discovery["selected_form"] != "joint_interaction":
        raise ValueError("parent status changed")
    return binding, law, axis, discovery


def plan():
    binding, _, _, _ = load_bound()
    rows = authority.build_rows()
    return {"schema": "subject_number_donor_free_head_response_proxy_v2_plan",
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
    binding, law, axis_artifact, discovery = load_bound()
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
        input_recipient = tangent._role_slice(inputs, 0, n)
        input_opposite = tangent._role_slice(inputs, n, 2 * n)
        base_all = factor_gate._raw_for(input_recipient, input_opposite, "", F)
        yz_all = factor_gate._raw_for(input_recipient, input_opposite, "YZ", F)
        delta_all = yz_all - base_all
        axis = torch.tensor(axis_artifact["axis"], device=device, dtype=base_all.dtype)
        templates_np = np.asarray([row["template_id"] for row in rows])
        directions_np = np.asarray([row["direction_id"] for row in rows])
        exact_s_values, proxy_s_values, targets = [], [], []
        exact_alpha_values, proxy_alpha_values, native_alpha_values = [], [], []
        evidence, prototype_audit = [], {}
        evaluations = 0
        for held_out in sorted(set(templates_np)):
            test_np = templates_np == held_out; train_np = ~test_np
            test = torch.tensor(np.flatnonzero(test_np), device=device, dtype=torch.long)
            fold_recipient, fold_opposite = select(recipient, test), select(opposite, test)
            fold_ir, fold_io = select(input_recipient, test), select(input_opposite, test)
            function = tangent._head_function(model, fold_recipient, fold_opposite,
                                              model.transformer.h[tangent.parent.LAYER].attn,
                                              projection, torch, F)
            prototypes = {}
            for direction in sorted(set(directions_np)):
                train_direction = torch.tensor(np.flatnonzero(train_np & (directions_np == direction)),
                                               device=device, dtype=torch.long)
                p = delta_all[train_direction].mean(0)
                prototypes[direction] = p
                prototype_audit[f"{held_out}.{direction}"] = {
                    "training_rows": int(len(train_direction)), "l2_norm": float(p.norm()),
                    "float32_sha256": hashlib.sha256(p.detach().float().cpu().numpy().tobytes()).hexdigest()}
            fold_rows = [rows[i] for i in np.flatnonzero(test_np)]
            p_batch = torch.stack([prototypes[row["direction_id"]] for row in fold_rows])
            beta = np.asarray(discovery["reports"]["joint_interaction"]["folds"][held_out]["beta"])
            native_beta = np.asarray(discovery["reports"]["native_baseline"]["folds"][held_out]["beta"])
            for subset in factor_gate.BACKGROUND_SUBSETS:
                xb = factor_gate._raw_for(fold_ir, fold_io, subset, F)
                xb_yz = factor_gate._raw_for(fold_ir, fold_io, subset + "YZ", F)
                h0, h1, hp = function(xb), function(xb_yz), function(xb + p_batch)
                evaluations += 3 * len(fold_rows)
                z = (h0 @ axis).detach().double().cpu().numpy()
                exact_s = ((h1 - h0) @ axis).detach().double().cpu().numpy()
                proxy_s = ((hp - h0) @ axis).detach().double().cpu().numpy()
                exact_alpha = np.c_[np.ones(len(z)), z, exact_s, z * exact_s] @ beta
                proxy_alpha = np.c_[np.ones(len(z)), z, proxy_s, z * proxy_s] @ beta
                native_alpha = np.c_[np.ones(len(z)), z] @ native_beta
                for j, row in enumerate(fold_rows):
                    key = f"{row['direction_id']}.cardinality_{len(subset)}"
                    target = law["predicted_coefficients"][key]
                    exact_s_values.append(exact_s[j]); proxy_s_values.append(proxy_s[j]); targets.append(target)
                    exact_alpha_values.append(exact_alpha[j]); proxy_alpha_values.append(proxy_alpha[j]); native_alpha_values.append(native_alpha[j])
                    evidence.append({"held_out_template": held_out, "row_id": row["row_id"],
                                     "direction": row["direction_id"], "background": subset,
                                     "cardinality": len(subset), "z": float(z[j]),
                                     "exact_s": float(exact_s[j]), "proxy_s": float(proxy_s[j]),
                                     "target_alpha": float(target), "exact_alpha": float(exact_alpha[j]),
                                     "proxy_alpha": float(proxy_alpha[j]), "native_alpha": float(native_alpha[j])})
    response_metrics = stats(exact_s_values, proxy_s_values)
    exact_program_metrics = stats(targets, exact_alpha_values)
    proxy_program_metrics = stats(targets, proxy_alpha_values)
    native_program_metrics = stats(targets, native_alpha_values)
    registered_oracle = discovery["reports"]["joint_interaction"]["cross_construction"]
    replay_error = max(abs(exact_program_metrics[key] - registered_oracle[key])
                       for key in ("cosine", "relative_l2_error", "sign_agreement"))
    improvement = BASELINE_ERROR - proxy_program_metrics["relative_l2_error"]
    degradation = proxy_program_metrics["relative_l2_error"] - ORACLE_ERROR
    instrument = (len(evidence) == 512 and len(tokens) == PRICE["role_sequences"] and
                  len(prototype_audit) == PRICE["fold_direction_prototypes"] and
                  evaluations == PRICE["offline_head_function_row_evaluations"] and
                  replay_error <= 1e-8 and
                  max(closure["input_state_closure_max_absolute_error"],
                      closure["input_normalized_closure_max_absolute_error"]) <= 5e-5)
    response_pass = response_metrics["cosine"] >= .85 and response_metrics["relative_l2_error"] <= .50
    program_pass = (proxy_program_metrics["relative_l2_error"] <= .45 and improvement >= .10
                    and degradation <= .10)
    predictions = {"pred_a_exact_response_replay": bool(instrument),
                   "pred_b_donor_free_response_proxy": bool(instrument and response_pass),
                   "pred_c_donor_free_program_proxy": bool(instrument and program_pass)}
    terminal = "invalid" if not instrument else "donor_free_head_response_proxy_held" if response_pass and program_pass else "donor_free_head_response_proxy_null"
    result = {"schema": "subject_number_donor_free_head_response_proxy_v2_result",
              "terminal": terminal, "predictions": predictions,
              "response_proxy_metrics": response_metrics,
              "exact_program_metrics": exact_program_metrics,
              "proxy_program_metrics": proxy_program_metrics,
              "native_program_metrics": native_program_metrics,
              "proxy_absolute_improvement_over_native": improvement,
              "proxy_absolute_degradation_from_oracle": degradation,
              "exact_replay_max_metric_error": replay_error,
              "prototype_audit": prototype_audit, "joined_evidence": evidence,
              "instrument": {"examples": len(evidence), "role_sequences": len(tokens),
                             "offline_head_function_row_evaluations": evaluations,
                             "fits": 0,
                             "role_state_closure_max_absolute_error": closure["input_state_closure_max_absolute_error"],
                             "role_normalized_closure_max_absolute_error": closure["input_normalized_closure_max_absolute_error"]},
              "outcome_access": {"behavioral_effects": False, "answer_logits": False,
                                 "downstream_causal_outcomes": False, "fresh_authority": False,
                                 "held_out_row_specific_donor_used_by_proxy": False},
              "price": PRICE, "authority_sha256": authority.validate_rows(rows),
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Outcome-blind cross-construction test of a fixed direction prototype inside the selected native-axis response interaction; fresh causal substitution remains required."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("terminal", "predictions", "response_proxy_metrics",
                                                   "exact_program_metrics", "proxy_program_metrics",
                                                   "native_program_metrics", "proxy_absolute_improvement_over_native",
                                                   "proxy_absolute_degradation_from_oracle", "exact_replay_max_metric_error",
                                                   "instrument")}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
