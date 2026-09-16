#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_inputs_and_method_frozen pred_b_two_finite_bounded_prototypes pred_c_direct_artifact_replay
"""Freeze all-opened-row response-weighted prototypes before fresh testing."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import circuit_fast_screen_candidate_task14_cardinality_prototype_transfer as authority
import circuit_fast_screen_managed_runner as managed
import run_subject_number_response_weighted_prototype_v1 as method
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parent.parent
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_FREEZE_V1_PREREGISTRATION.md"
LAW = POLY / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json"
AXIS = POLY / "SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_ARTIFACT.json"
DISCOVERY = ROOT / "circuits/fast_screens/subject_number_native_head_response_coordinate_discovery_v2_result.json"
SELECTED = ROOT / "circuits/fast_screens/subject_number_response_weighted_prototype_v2_result.json"
BINDING = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_FREEZE_V1_BINDING.json"
OUT = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_FROZEN_V1_ARTIFACT.json"
RANK = 16
RIDGE_FRACTION = 1.0
PRICE = {"physical_model_forwards": 1, "role_sequences": 96,
         "prototype_fits": 2, "prototype_width": 1152, "candidate_rank": RANK,
         "input_gradient_rows": 512, "coefficient_fits": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_inputs_and_method_frozen": None,
                       "pred_b_two_finite_bounded_prototypes": None,
                       "pred_c_direct_artifact_replay": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    yn, pn = np.linalg.norm(y), np.linalg.norm(prediction)
    return {"count": int(len(y)), "cosine": float(y @ prediction / max(yn * pn, 1e-30)),
            "relative_l2_error": float(np.linalg.norm(y - prediction) / max(yn, 1e-30)),
            "sign_agreement": float(np.mean((y > 0) == (prediction > 0)))}


def select(values, index):
    return {key: value[index] for key, value in values.items()}


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"preregistration": PREREG, "law": LAW, "native_axis": AXIS,
             "discovery": DISCOVERY, "selected_result": SELECTED,
             "authority": Path(authority.__file__), "selected_runner": Path(method.__file__).with_name("run_subject_number_response_weighted_prototype_v2.py")}
    if binding["files"] != {key: sha(path) for key, path in paths.items()} \
            or binding["price"] != PRICE or binding["rank"] != RANK \
            or binding["ridge_fraction"] != RIDGE_FRACTION:
        raise ValueError("binding changed")
    law, axis, discovery, selected = (json.loads(path.read_text()) for path in (LAW, AXIS, DISCOVERY, SELECTED))
    if law["terminal"] != "bilinear_scalar_law_frozen_weights_only" \
            or axis["terminal"] != "native_weight_axis_frozen" \
            or discovery["terminal"] != "native_head_response_coordinate_selected" \
            or selected["terminal"] != "response_weighted_prototype_held" \
            or not all(selected["predictions"].values()):
        raise ValueError("parent status changed")
    return binding, law, axis, discovery


def plan():
    binding, _, _, _ = load_bound(); rows = authority.build_rows()
    return {"schema": "subject_number_response_weighted_prototype_freeze_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "role_sequences": 3 * len(rows),
            "background_subsets": list(factor_gate.BACKGROUND_SUBSETS),
            "directions": sorted({row["direction_id"] for row in rows}),
            "rank": RANK, "ridge_fraction": RIDGE_FRACTION, "price": PRICE,
            "authority_sha256": authority.validate_rows(rows),
            "binding_sha256": sha(BINDING), "bound_files": sorted(binding["files"])}


@np.errstate(all="raise")
def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
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
        ir = tangent._role_slice(inputs, 0, n)
        io = tangent._role_slice(inputs, n, 2 * n)
        base_all = factor_gate._raw_for(ir, io, "", F)
        yz_all = factor_gate._raw_for(ir, io, "YZ", F)
        delta_all = yz_all - base_all
    axis = torch.tensor(axis_artifact["axis"], device=device, dtype=base_all.dtype)
    directions = np.asarray([row["direction_id"] for row in rows])
    beta = np.asarray(discovery["reports"]["joint_interaction"]["all_row_beta"])
    prototypes, audits = {}, {}
    exact_s_all, candidate_s_all, targets, candidate_alpha_all = [], [], [], []
    gradient_rows = 0
    for direction in sorted(set(directions)):
        ids = np.flatnonzero(directions == direction)
        index = torch.tensor(ids, device=device, dtype=torch.long)
        direction_rows = [rows[i] for i in ids]
        r, o, ri, oi = select(recipient, index), select(opposite, index), select(ir, index), select(io, index)
        function = tangent._head_function(model, r, o, model.transformer.h[tangent.parent.LAYER].attn,
                                          projection, torch, F)
        mean_p = delta_all[index].mean(0)
        displacements = []
        for subset in factor_gate.BACKGROUND_SUBSETS:
            xb = factor_gate._raw_for(ri, oi, subset, F)
            xy = factor_gate._raw_for(ri, oi, subset + "YZ", F)
            displacements.append(xy - xb)
        centered = torch.cat(displacements) - mean_p
        _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
        modes = vh[:RANK].T.contiguous()
        designs, residuals = [], []
        for subset in factor_gate.BACKGROUND_SUBSETS:
            xb = factor_gate._raw_for(ri, oi, subset, F)
            xy = factor_gate._raw_for(ri, oi, subset + "YZ", F)
            with torch.no_grad():
                h0, h1, hm = function(xb), function(xy), function(xb + mean_p)
                exact_s = (h1 - h0) @ axis
                mean_s = (hm - h0) @ axis
            point = (xb + mean_p).detach().requires_grad_(True)
            gradient = torch.autograd.grad((function(point) @ axis).sum(), point, create_graph=False)[0]
            designs.append(gradient @ modes); residuals.append(exact_s - mean_s)
            gradient_rows += len(xb)
        design = torch.cat(designs).detach().double().cpu().numpy()
        residual = torch.cat(residuals).detach().double().cpu().numpy()
        coefficients, ridge = method.ridge_coefficients(design, residual, ridge_fraction=RIDGE_FRACTION)
        candidate = mean_p + modes @ torch.tensor(coefficients, device=device, dtype=modes.dtype)
        norm_ratio = float(candidate.norm() / mean_p.norm().clamp_min(1e-30))
        prototypes[direction] = candidate.detach().float().cpu().tolist()
        audits[direction] = {"training_rows": len(ids), "training_background_examples": len(design),
                             "rank": RANK, "ridge": ridge, "mean_norm": float(mean_p.norm()),
                             "prototype_norm": float(candidate.norm()), "norm_ratio": norm_ratio,
                             "span_energy_fraction": float(singular[:RANK].square().sum() / singular.square().sum().clamp_min(1e-30)),
                             "float32_sha256": hashlib.sha256(candidate.detach().float().cpu().numpy().tobytes()).hexdigest()}
        with torch.no_grad():
            for subset in factor_gate.BACKGROUND_SUBSETS:
                xb = factor_gate._raw_for(ri, oi, subset, F)
                xy = factor_gate._raw_for(ri, oi, subset + "YZ", F)
                h0, h1, hp = function(xb), function(xy), function(xb + candidate)
                z = (h0 @ axis).detach().double().cpu().numpy()
                exact_s = ((h1 - h0) @ axis).detach().double().cpu().numpy()
                candidate_s = ((hp - h0) @ axis).detach().double().cpu().numpy()
                alpha = np.c_[np.ones(len(z)), z, candidate_s, z * candidate_s] @ beta
                for j, row in enumerate(direction_rows):
                    targets.append(law["predicted_coefficients"][f"{direction}.cardinality_{len(subset)}"])
                    exact_s_all.append(exact_s[j]); candidate_s_all.append(candidate_s[j]); candidate_alpha_all.append(alpha[j])
    finite = bool(np.isfinite(np.asarray([*exact_s_all, *candidate_s_all, *candidate_alpha_all])).all()
                  and all(np.isfinite(np.asarray(value)).all() for value in prototypes.values()))
    bounded = max(value["norm_ratio"] for value in audits.values()) <= 2
    instrument = (len(prototypes) == 2 and all(len(value) == 1152 for value in prototypes.values())
                  and gradient_rows == PRICE["input_gradient_rows"] and finite and bounded
                  and max(closure["input_state_closure_max_absolute_error"], closure["input_normalized_closure_max_absolute_error"]) <= 5e-5)
    predictions = {"pred_a_inputs_and_method_frozen": bool(instrument),
                   "pred_b_two_finite_bounded_prototypes": bool(instrument),
                   "pred_c_direct_artifact_replay": bool(instrument and len(candidate_s_all) == 512)}
    terminal = "response_weighted_prototypes_frozen_opened_only" if all(predictions.values()) else "invalid"
    artifact = {"schema": "subject_number_response_weighted_prototype_frozen_v1_artifact",
                "terminal": terminal, "predictions": predictions,
                "prototypes": prototypes, "prototype_audit": audits,
                "native_axis": axis_artifact["axis"], "interaction_beta": beta.tolist(),
                "training_response_metrics": stats(exact_s_all, candidate_s_all),
                "training_coefficient_metrics": stats(targets, candidate_alpha_all),
                "instrument": {"rows": len(rows), "examples": len(targets), "role_sequences": len(tokens),
                               "input_gradient_rows": gradient_rows, "prototype_count": len(prototypes),
                               "finite": finite, "bounded": bounded,
                               "role_state_closure_max_absolute_error": closure["input_state_closure_max_absolute_error"],
                               "role_normalized_closure_max_absolute_error": closure["input_normalized_closure_max_absolute_error"]},
                "outcome_access": {"answer_logits": False, "behavioral_effects": False,
                                   "downstream_causal_outcomes": False, "new_text": False,
                                   "opened_training_donor_response_used": True},
                "price": PRICE, "authority_sha256": authority.validate_rows(rows),
                "checkpoint_weights_sha256": checkpoint.weights_sha256,
                "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
                "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "scope": "Two all-opened-row response-weighted prototypes frozen before fourth-corpus authority authorship; no generalization claim."}
    managed.atomic_create_json(OUT, artifact)
    print(json.dumps({key: artifact[key] for key in ("terminal", "predictions", "prototype_audit",
                                                      "training_response_metrics", "training_coefficient_metrics",
                                                      "instrument", "outcome_access")}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
