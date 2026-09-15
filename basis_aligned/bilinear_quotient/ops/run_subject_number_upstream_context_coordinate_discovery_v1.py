#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_exact_activation_instrument pred_b_bilinear_context_interaction pred_c_registered_form_selection
"""Outcome-blind subject-number z by MLP8-input secant-coordinate discovery."""
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
PREREG = POLY / "SUBJECT_NUMBER_UPSTREAM_CONTEXT_COORDINATE_DISCOVERY_V1_PREREGISTRATION.md"
LAW = POLY / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json"
AXIS = POLY / "SUBJECT_NUMBER_NATIVE_WEIGHT_AXIS_V1_ARTIFACT.json"
PARENT = ROOT / "circuits/fast_screens/subject_number_native_scalar_feature_discovery_v1_result.json"
BINDING = POLY / "SUBJECT_NUMBER_UPSTREAM_CONTEXT_COORDINATE_DISCOVERY_V1_BINDING.json"
OUT = ROOT / "circuits/fast_screens/subject_number_upstream_context_coordinate_discovery_v1_result.json"
FORMS = ("affine", "bilinear", "quadratic_context")
PRICE = {"physical_model_forwards": 1, "role_sequences": 96,
         "scalar_least_squares_fits": 9, "backwards": 0, "parameter_updates": 0}
PREDICTION_REGISTRY = {"pred_a_exact_activation_instrument": None,
                       "pred_b_bilinear_context_interaction": None,
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
            or parent["terminal"] != "native_scalar_feature_discovery_null":
        raise ValueError("parent status changed")
    return binding, law, axis


def plan():
    binding, _, _ = load_bound()
    rows = authority.build_rows()
    return {"schema": "subject_number_upstream_context_coordinate_discovery_v1_plan",
            "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
            "rows": len(rows), "role_sequences": 3 * len(rows),
            "background_subsets": list(factor_gate.BACKGROUND_SUBSETS),
            "forms": list(FORMS), "price": PRICE, "authority_sha256": authority.validate_rows(rows),
            "binding_sha256": sha(BINDING), "bound_files": sorted(binding["files"])}


def stats(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    yn, pn = np.linalg.norm(y), np.linalg.norm(prediction)
    return {"count": int(len(y)), "cosine": float(y @ prediction / max(yn * pn, 1e-30)),
            "relative_l2_error": float(np.linalg.norm(y - prediction) / max(yn, 1e-30)),
            "sign_agreement": float(np.mean((y > 0) == (prediction > 0)))}


def design(form, z, g):
    one = np.ones(len(z))
    if form == "affine":
        return np.c_[one, z]
    if form == "bilinear":
        return np.c_[one, z, g, z * g]
    if form == "quadratic_context":
        return np.c_[one, z, g, g * g, z * g, z * g * g]
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
        x0 = factor_gate._raw_for(input_roles["recipient"], input_roles["opposite"], "", F)
        x4 = factor_gate._raw_for(input_roles["recipient"], input_roles["opposite"], "EAUW", F)
        displacement = x4 - x0
        denominator = displacement.square().sum(-1)
        if not bool((denominator > 1e-12).all()):
            raise ValueError("full background displacement is degenerate")
        native_axis = torch.tensor(axis_artifact["axis"], device=device, dtype=x0.dtype)
        z_rows = (function(x0) @ native_axis).detach().double().cpu().numpy()
        z_values, g_values, targets, templates, directions, cardinalities = [], [], [], [], [], []
        coordinate_endpoints = {}
        for subset in factor_gate.BACKGROUND_SUBSETS:
            xb = factor_gate._raw_for(input_roles["recipient"], input_roles["opposite"], subset, F)
            g = (4 * ((xb - x0) * displacement).sum(-1) / denominator).detach().double().cpu().numpy()
            coordinate_endpoints[subset or "empty"] = {"minimum": float(g.min()),
                                                        "maximum": float(g.max()),
                                                        "mean": float(g.mean())}
            for i, row in enumerate(rows):
                cardinality = len(subset)
                key = f"{row['direction_id']}.cardinality_{cardinality}"
                z_values.append(z_rows[i]); g_values.append(g[i])
                targets.append(law["predicted_coefficients"][key])
                templates.append(row["template_id"]); directions.append(row["direction_id"])
                cardinalities.append(cardinality)
    z_values, g_values, targets = map(np.asarray, (z_values, g_values, targets))
    templates, directions, cardinalities = map(np.asarray, (templates, directions, cardinalities))
    reports, fits = {}, 0
    for form in FORMS:
        x = design(form, z_values, g_values)
        prediction = np.empty(len(targets)); folds = {}
        for held_out in sorted(set(templates)):
            test = templates == held_out; train = ~test
            beta, values = fit_predict(x, targets, train, test)
            prediction[test] = values; fits += 1
            folds[held_out] = {"beta": beta.tolist(), "metrics": stats(targets[test], values)}
        beta = np.linalg.lstsq(x, targets, rcond=None)[0]; fits += 1
        reports[form] = {"cross_construction": stats(targets, prediction),
                         "folds": folds, "all_row_beta": beta.tolist(),
                         "by_direction": {value: stats(targets[directions == value], prediction[directions == value])
                                          for value in sorted(set(directions))},
                         "by_cardinality": {str(value): stats(targets[cardinalities == value], prediction[cardinalities == value])
                                            for value in sorted(set(cardinalities))}}
    affine_error = reports["affine"]["cross_construction"]["relative_l2_error"]
    bilinear_error = reports["bilinear"]["cross_construction"]["relative_l2_error"]
    quadratic_error = reports["quadratic_context"]["cross_construction"]["relative_l2_error"]
    improvement = affine_error - bilinear_error
    bilinear_pass = bilinear_error <= .40 and improvement >= .10
    selected = "bilinear" if bilinear_pass else "quadratic_context" if quadratic_error <= .40 else None
    instrument = (len(targets) == 512 and len(tokens) == PRICE["role_sequences"] and
                  fits == PRICE["scalar_least_squares_fits"] and
                  max(closure["input_state_closure_max_absolute_error"],
                      closure["input_normalized_closure_max_absolute_error"]) <= 5e-5 and
                  abs(coordinate_endpoints["empty"]["maximum"]) <= 1e-7 and
                  max(abs(coordinate_endpoints["EAUW"]["minimum"] - 4),
                      abs(coordinate_endpoints["EAUW"]["maximum"] - 4)) <= 1e-5)
    predictions = {"pred_a_exact_activation_instrument": bool(instrument),
                   "pred_b_bilinear_context_interaction": bool(instrument and bilinear_pass),
                   "pred_c_registered_form_selection": bool(instrument and selected is not None)}
    terminal = ("invalid" if not instrument else
                "upstream_context_coordinate_selected" if selected is not None else
                "upstream_context_coordinate_null")
    result = {"schema": "subject_number_upstream_context_coordinate_discovery_v1_result",
              "terminal": terminal, "predictions": predictions, "selected_form": selected,
              "affine_relative_l2_error": affine_error,
              "bilinear_relative_l2_error": bilinear_error,
              "quadratic_context_relative_l2_error": quadratic_error,
              "bilinear_absolute_improvement": improvement, "reports": reports,
              "coordinate_endpoints": coordinate_endpoints,
              "instrument": {"examples": len(targets), "role_sequences": len(tokens), "fits": fits,
                             "role_state_closure_max_absolute_error": closure["input_state_closure_max_absolute_error"],
                             "role_normalized_closure_max_absolute_error": closure["input_normalized_closure_max_absolute_error"]},
              "outcome_access": {"behavioral_effects": False, "answer_logits": False,
                                 "exact_donor_displacements": False, "fresh_authority": False},
              "price": PRICE, "authority_sha256": authority.validate_rows(rows),
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "binding_sha256": sha(BINDING), "runner_sha256": sha(RUNNER),
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Outcome-blind opened-authority discovery of a native L11H3 axis by MLP8-input secant coordinate. A passing form still requires frozen fresh-authority causal substitution."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions,
                      "selected_form": selected, "errors": {form: reports[form]["cross_construction"]
                                                              for form in FORMS},
                      "bilinear_absolute_improvement": improvement,
                      "instrument": result["instrument"]}, indent=2))
    assert instrument


if __name__ == "__main__":
    main()
