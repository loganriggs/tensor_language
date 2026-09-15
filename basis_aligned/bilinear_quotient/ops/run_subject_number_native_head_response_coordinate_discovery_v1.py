#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_head_response_instrument pred_b_response_coordinate_predicts_program pred_c_registered_form_selection
"""Outcome-blind subject-number native L11H3 response-coordinate discovery."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import circuit_fast_screen_candidate_task14_cardinality_prototype_transfer as authority
import circuit_fast_screen_managed_runner as managed
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as factor_gate


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parent.parent
POLY = ROOT.parent / "polynomial_causal"
PREREG = POLY / "SUBJECT_NUMBER_NATIVE_HEAD_RESPONSE_COORDINATE_DISCOVERY_V1_PREREGISTRATION.md"
LAW = POLY / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json"
AXIS = POLY / "SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_ARTIFACT.json"
PARENT = ROOT / "circuits/fast_screens/subject_number_upstream_context_coordinate_discovery_v1_result.json"
BINDING = POLY / "SUBJECT_NUMBER_NATIVE_HEAD_RESPONSE_COORDINATE_DISCOVERY_V1_BINDING.json"
OUT = ROOT / "circuits/fast_screens/subject_number_native_head_response_coordinate_discovery_v1_result.json"
FORMS = ("native_baseline", "response_only", "joint_additive", "joint_interaction")
PRICE = {"physical_model_forwards": 1, "role_sequences": 96,
         "offline_head_function_evaluations": 1024,
         "scalar_least_squares_fits": 12, "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_exact_head_response_instrument": None,
                       "pred_b_response_coordinate_predicts_program": None,
                       "pred_c_registered_form_selection": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"preregistration": PREREG, "law": LAW, "native_axis": AXIS,
             "parent_null": PARENT, "authority": Path(authority.__file__)}
    if binding["files"] != {key: sha(path) for key, path in paths.items()} \
            or binding["forms"] != list(FORMS) or binding["price"] != PRICE:
        raise ValueError("binding changed")
    law, axis, parent = (json.loads(path.read_text()) for path in (LAW, AXIS, PARENT))
    if law["terminal"] != "bilinear_scalar_law_frozen_weights_only" \
            or axis["terminal"] != "native_weight_axis_frozen" \
            or parent["terminal"] != "upstream_context_coordinate_null":
        raise ValueError("parent status changed")
    return binding, law, axis


def plan():
    binding, _, _ = load_bound()
    rows = authority.build_rows()
    return {"schema": "subject_number_native_head_response_coordinate_discovery_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "role_sequences": 3 * len(rows),
            "background_subsets": list(factor_gate.BACKGROUND_SUBSETS),
            "forms": list(FORMS), "price": PRICE,
            "authority_sha256": authority.validate_rows(rows),
            "binding_sha256": sha(BINDING), "bound_files": sorted(binding["files"])}


def stats(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    yn, pn = np.linalg.norm(y), np.linalg.norm(prediction)
    return {"count": int(len(y)), "cosine": float(y @ prediction / max(yn * pn, 1e-30)),
            "relative_l2_error": float(np.linalg.norm(y - prediction) / max(yn, 1e-30)),
            "sign_agreement": float(np.mean((y > 0) == (prediction > 0)))}


def design(form, z, s):
    one = np.ones(len(z))
    if form == "native_baseline":
        return np.c_[one, z]
    if form == "response_only":
        return np.c_[one, s]
    if form == "joint_additive":
        return np.c_[one, z, s]
    if form == "joint_interaction":
        return np.c_[one, z, s, z * s]
    raise ValueError(form)


def fit_predict(x, y, train, test):
    beta = np.linalg.lstsq(x[train], y[train], rcond=None)[0]
    return beta, x[test] @ beta


@np.errstate(all="raise")
def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    binding, law, axis_artifact = load_bound()
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    rows = authority.build_rows()
    n = len(rows)
    device = next(model.parameters()).device
    tokens, finals = tangent.parent.downstream.depth.parent.v1._role_batch(rows, torch, device)
    with torch.no_grad():
        _, captured, projection, closure, inputs = tangent.parent._decomposed_forward(
            model, tokens, finals, torch, F, facade)
        roles = {"recipient": tangent._role_slice(captured, 0, n),
                 "opposite": tangent._role_slice(captured, n, 2 * n)}
        input_roles = {"recipient": tangent._role_slice(inputs, 0, n),
                       "opposite": tangent._role_slice(inputs, n, 2 * n)}
        function = tangent._head_function(model, roles["recipient"], roles["opposite"],
                                          model.transformer.h[tangent.parent.LAYER].attn,
                                          projection, torch, F)
        native_axis = torch.tensor(axis_artifact["axis"], device=device,
                                   dtype=input_roles["recipient"].dtype)
        z_values, s_values, targets, templates, directions, cardinalities = [], [], [], [], [], []
        coordinate_summaries = {}
        evaluations = 0
        for subset in factor_gate.BACKGROUND_SUBSETS:
            xb = factor_gate._raw_for(input_roles["recipient"], input_roles["opposite"], subset, F)
            xb_yz = factor_gate._raw_for(input_roles["recipient"], input_roles["opposite"], subset + "YZ", F)
            h0, h1 = function(xb), function(xb_yz)
            evaluations += 2 * n
            z = (h0 @ native_axis).detach().double().cpu().numpy()
            s = ((h1 - h0) @ native_axis).detach().double().cpu().numpy()
            coordinate_summaries[subset or "empty"] = {
                "z_minimum": float(z.min()), "z_maximum": float(z.max()), "z_mean": float(z.mean()),
                "s_minimum": float(s.min()), "s_maximum": float(s.max()), "s_mean": float(s.mean())}
            for i, row in enumerate(rows):
                cardinality = len(subset)
                key = f"{row['direction_id']}.cardinality_{cardinality}"
                z_values.append(z[i]); s_values.append(s[i]); targets.append(law["predicted_coefficients"][key])
                templates.append(row["template_id"]); directions.append(row["direction_id"])
                cardinalities.append(cardinality)
    z_values, s_values, targets = map(np.asarray, (z_values, s_values, targets))
    templates, directions, cardinalities = map(np.asarray, (templates, directions, cardinalities))
    reports, fits = {}, 0
    for form in FORMS:
        x = design(form, z_values, s_values)
        prediction = np.empty(len(targets)); folds = {}
        for held_out in sorted(set(templates)):
            test = templates == held_out; train = ~test
            beta, values = fit_predict(x, targets, train, test)
            prediction[test] = values; fits += 1
            folds[held_out] = {"beta": beta.tolist(), "metrics": stats(targets[test], values)}
        beta = np.linalg.lstsq(x, targets, rcond=None)[0]; fits += 1
        reports[form] = {"cross_construction": stats(targets, prediction), "folds": folds,
                         "all_row_beta": beta.tolist(),
                         "by_direction": {value: stats(targets[directions == value], prediction[directions == value])
                                          for value in sorted(set(directions))},
                         "by_cardinality": {str(value): stats(targets[cardinalities == value], prediction[cardinalities == value])
                                            for value in sorted(set(cardinalities))}}
    baseline_error = reports["native_baseline"]["cross_construction"]["relative_l2_error"]
    response_forms = FORMS[1:]
    improvements = {form: baseline_error - reports[form]["cross_construction"]["relative_l2_error"]
                    for form in response_forms}
    passing = [form for form in response_forms
               if reports[form]["cross_construction"]["relative_l2_error"] <= .40
               and improvements[form] >= .10]
    selected = passing[0] if passing else None
    instrument = (len(targets) == 512 and len(tokens) == PRICE["role_sequences"] and
                  evaluations == PRICE["offline_head_function_evaluations"] and
                  fits == PRICE["scalar_least_squares_fits"] and
                  max(closure["input_state_closure_max_absolute_error"],
                      closure["input_normalized_closure_max_absolute_error"]) <= 5e-5 and
                  bool(np.isfinite(z_values).all() and np.isfinite(s_values).all()) and
                  float(np.linalg.norm(s_values)) > 1e-12)
    predictions = {"pred_a_exact_head_response_instrument": bool(instrument),
                   "pred_b_response_coordinate_predicts_program": bool(instrument and passing),
                   "pred_c_registered_form_selection": bool(instrument and selected is not None)}
    terminal = ("invalid" if not instrument else
                "native_head_response_coordinate_selected" if selected else
                "native_head_response_coordinate_null")
    result = {"schema": "subject_number_native_head_response_coordinate_discovery_v1_result",
              "terminal": terminal, "predictions": predictions, "selected_form": selected,
              "native_baseline_relative_l2_error": baseline_error,
              "response_improvements": improvements, "reports": reports,
              "coordinate_summaries": coordinate_summaries,
              "coordinate_geometry": {"z_l2_norm": float(np.linalg.norm(z_values)),
                                      "s_l2_norm": float(np.linalg.norm(s_values)),
                                      "z_s_cosine": float(z_values @ s_values / max(np.linalg.norm(z_values) * np.linalg.norm(s_values), 1e-30))},
              "instrument": {"examples": len(targets), "role_sequences": len(tokens),
                             "offline_head_function_evaluations": evaluations, "fits": fits,
                             "role_state_closure_max_absolute_error": closure["input_state_closure_max_absolute_error"],
                             "role_normalized_closure_max_absolute_error": closure["input_normalized_closure_max_absolute_error"]},
              "outcome_access": {"behavioral_effects": False, "answer_logits": False,
                                 "downstream_causal_outcomes": False, "fresh_authority": False,
                                 "opposite_number_donor_state_used": True},
              "price": PRICE, "authority_sha256": authority.validate_rows(rows),
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Opened-authority discovery of a donor-dependent native L11H3 response coordinate; passing would require donor-free prospective validation."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions,
                      "selected_form": selected, "baseline_error": baseline_error,
                      "response_improvements": improvements,
                      "errors": {form: reports[form]["cross_construction"] for form in FORMS},
                      "coordinate_geometry": result["coordinate_geometry"],
                      "instrument": result["instrument"]}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
