#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_parent_replays pred_b_response_weighting_improves pred_c_complete_program_and_null
"""Trust-region correction for the response-weighted subject-number prototype."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_subject_number_response_weighted_prototype_v1 as base


RUNNER = Path(__file__).resolve()
POLY = base.POLY
CORRECTION = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_V2_CORRECTION.md"
INVALID_RUNNER = Path(base.__file__).resolve()
INVALID_RESULT = base.OUT
BINDING = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_V2_BINDING.json"
OUT = base.ROOT / "circuits/fast_screens/subject_number_response_weighted_prototype_v2_result.json"
RIDGE_FRACTION = 1.0
ORIGINAL_PLAN = base.plan
PREDICTION_REGISTRY = {"pred_a_parent_replays": None,
                       "pred_b_response_weighting_improves": None,
                       "pred_c_complete_program_and_null": None}


def load_bound():
    binding = json.loads(BINDING.read_text())
    paths = {"preregistration": base.PREREG, "correction": CORRECTION,
             "invalid_runner": INVALID_RUNNER, "invalid_result": INVALID_RESULT,
             "law": base.LAW, "native_axis": base.AXIS,
             "discovery": base.DISCOVERY, "mean_proxy": base.MEAN_PROXY,
             "authority": Path(base.authority.__file__)}
    if binding["files"] != {key: base.sha(path) for key, path in paths.items()} \
            or binding["price"] != base.PRICE or binding["rank"] != base.RANK \
            or binding["nulls"] != base.NULLS or binding["ridge_fraction"] != RIDGE_FRACTION \
            or binding["seed"] != base.SEED:
        raise ValueError("binding changed")
    law, axis, discovery, mean_proxy, invalid = (
        json.loads(path.read_text()) for path in
        (base.LAW, base.AXIS, base.DISCOVERY, base.MEAN_PROXY, INVALID_RESULT))
    if law["terminal"] != "bilinear_scalar_law_frozen_weights_only" \
            or axis["terminal"] != "native_weight_axis_frozen" \
            or discovery["terminal"] != "native_head_response_coordinate_selected" \
            or discovery["selected_form"] != "joint_interaction" \
            or mean_proxy["terminal"] != "donor_free_head_response_proxy_null" \
            or invalid["terminal"] != "invalid" \
            or invalid["instrument"]["prototype_norm_gate"] is not False:
        raise ValueError("parent status changed")
    return binding, law, axis, discovery, mean_proxy


def plan():
    value = ORIGINAL_PLAN()
    value.update(schema="subject_number_response_weighted_prototype_v2_plan",
                 binding_sha256=base.sha(BINDING), ridge_fraction=RIDGE_FRACTION,
                 correction_sha256=base.sha(CORRECTION),
                 invalid_result_sha256=base.sha(INVALID_RESULT))
    return value


def ridge_coefficients(design, target, ridge_fraction=RIDGE_FRACTION):
    gram = design.T @ design
    ridge = ridge_fraction * np.trace(gram) / design.shape[1]
    coefficients = np.linalg.solve(gram + ridge * np.eye(design.shape[1]), design.T @ target)
    return coefficients, float(ridge)


def main():
    original_atomic = base.managed.atomic_create_json

    def corrected_atomic(path, value):
        value["schema"] = "subject_number_response_weighted_prototype_v2_result"
        value["correction_sha256"] = base.sha(CORRECTION)
        value["invalid_parent_sha256"] = base.sha(INVALID_RESULT)
        value["ridge_fraction"] = RIDGE_FRACTION
        value["scope"] = ("Opened-authority, trust-region-corrected response-Jacobian selection of one "
                          "donor-free fixed grouped-MLP6/7 prototype per direction; fresh causal substitution remains required.")
        return original_atomic(path, value)

    base.RUNNER = RUNNER
    base.BINDING = BINDING
    base.OUT = OUT
    base.RIDGE_FRACTION = RIDGE_FRACTION
    base.load_bound = load_bound
    base.plan = plan
    base.ridge_coefficients = ridge_coefficients
    base.managed.atomic_create_json = corrected_atomic
    base.main()


if __name__ == "__main__":
    main()
