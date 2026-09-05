#!/usr/bin/env python3
"""Prospective OOD-text-reuse confirmation of Task14's MLP8 response split."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_plural_to_singular_cross_positive_quadratic_negative pred_c_singular_to_plural_cross_negative_quadratic_positive pred_d_signed_direction_pattern pred_e_background_stable pred_f_number_specific pred_g_selective_removal_direction_pattern

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import circuit_fast_screen_candidate_task14_ood_fronted_mlp8_polarized_response as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_ood_fronted_mlp8_native_capability as capability
import run_task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial as v1
import run_task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v2 as v2


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_head11_3_ood_fronted_subject_mlp8_polarized_response_factorial_v1.json"
OUT = ROOT / "circuits/fast_screens/task14_head11_3_ood_fronted_subject_mlp8_polarized_response_factorial_v1_result.json"
PRIOR_ART_SHA256 = "394c1e3c0233cefa3ba2bb07f7a83e2f9aaa2bb991e5625595e8b0c5f17c360f"
DISCOVERY_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_polarized_response_factorial_v2_result.json"
DISCOVERY_RESULT_SHA256 = "55d5413306f4471b0c9b8345732d317d0c1c4b82395153a119af3d56514f5ad6"
V2_RUNNER = Path(v2.__file__)
V2_RUNNER_SHA256 = "749bccf880e179c92c565c9eb43543faae0fcb6b04fd3813d2448581b0138cba"
CANDIDATE_ID = authority.CAUSAL_CANDIDATE_ID
CONDITIONS = v1.CONDITIONS
SUBJECT_POSITION = authority.SUBJECT_POSITION
BARS = {
    **v1.BARS,
    "plural_to_singular_minimum_cross_recovery": 1.5,
    "plural_to_singular_maximum_quadratic_recovery": -.5,
    "singular_to_plural_maximum_cross_recovery": -.1,
    "singular_to_plural_minimum_quadratic_recovery": 1.1,
}
PREDICTIONS = {
    "pred_a_instrument_live": "full MLP8 task effect and all exactness gates pass",
    "pred_b_plural_to_singular_cross_positive_quadratic_negative": "cross recovery >=1.5 and quadratic recovery <=-0.5 for CE and margin in both backgrounds",
    "pred_c_singular_to_plural_cross_negative_quadratic_positive": "cross recovery <=-0.1 and quadratic recovery >=1.1 for CE and margin in both backgrounds",
    "pred_d_signed_direction_pattern": "both opposing direction predictions hold",
    "pred_e_background_stable": "standalone and conditional recovery differ by <=0.25",
    "pred_f_number_specific": "same-number lexical full-effect ratio <=0.25",
    "pred_g_selective_removal_direction_pattern": "same corners imply direction-reversed selective-removal signs; algebraically dependent, not independent evidence",
}


class OODMLP8PolarizedResponseError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows():
    return authority.build_rows()


def validate_preflight():
    for path, expected, label in (
        (PRIOR_ART, PRIOR_ART_SHA256, "prior-art/preregistration"),
        (DISCOVERY_RESULT, DISCOVERY_RESULT_SHA256, "discovery result"),
        (V2_RUNNER, V2_RUNNER_SHA256, "audited v2 runner"),
    ):
        if _sha256(path) != expected:
            raise OODMLP8PolarizedResponseError(f"{label} changed")
    if not capability.RESULT.exists() or not capability.LICENSE.exists():
        raise OODMLP8PolarizedResponseError(
            "scoped OOD capability result/license must be created before the causal run")
    license_sha = _sha256(capability.LICENSE)
    value = licensing.validate_causal_preflight(
        capability.build_gate(), capability.RESULT, capability.LICENSE,
        expected_license_sha256=license_sha, causal_candidate_id=CANDIDATE_ID)
    return value, license_sha


def compile_plan():
    license_value, license_sha = validate_preflight()
    return {
        "schema": "task14_head11_3_ood_fronted_subject_mlp8_polarized_response_factorial_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "split": authority.SPLIT,
        "data_status": "OOD text and whole-head outcomes previously opened; only this MLP8 intervention is prospective",
        "row_count": 16,
        "subject_position": SUBJECT_POSITION,
        "mlp_layer": 8,
        "conditions": list(CONDITIONS),
        "authority_logical_sha256": authority.EXPECTED_AUTHORITY_SHA256,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "discovery_result_sha256": DISCOVERY_RESULT_SHA256,
        "v2_runner_sha256": V2_RUNNER_SHA256,
        "license_sha256": license_sha,
        "capability_result_sha256": license_value["capability_result_sha256"],
        "bars": dict(BARS),
        "backgrounds": {
            "standalone": "other MLP4-10 writes recipient",
            "conditional": "other MLP4-7 and MLP9-10 writes opposite-number donor",
            "lexical": "other MLP4-10 writes recipient; MLP8 source is same-number different-lemma",
        },
        "partition": "exact gauge-invariant MLP8 base-by-change cross response versus change-by-change quadratic response",
        "numerical_policy": "reuse v2 float64 local algebra, native endpoints, fixed remainders, native-dtype installation, and sequential propagation",
        "outcomes": ["answer-directed target margin improvement",
                     "target full-vocabulary CE improvement"],
        "predictions": dict(PREDICTIONS),
        "selective_removal": {
            "remove_cross": "conditional_full minus conditional_quadratic-corner",
            "remove_quadratic": "conditional_full minus conditional_cross-corner",
            "independence": "none; these are deterministic contrasts of the same four corners",
        },
        "capability_price_already_paid": {
            "model_forwards": 1, "example_evaluations": 48,
            "causal_interventions": 0, "backwards": 0, "parameter_updates": 0},
        "causal_price": {
            "model_forwards": 4, "example_evaluations": 480,
            "causal_interventions": 192, "backwards": 0, "parameter_updates": 0},
        "closed_claims": ["pristine OOD confirmation", "ordered Left/Right semantics",
                          "rank", "reconstruction", "independent removal replication",
                          "unrelated-behavior selectivity", "complete circuit"],
    }


def evaluate(model, torch, F, facade):
    # v2 is the audited numerical implementation. Temporarily supply this
    # frozen authority; restore the module immediately so no other experiment
    # can inherit the override.
    original = v2.build_rows
    try:
        v2.build_rows = build_rows
        return v2.evaluate(model, torch, F, facade)
    finally:
        v2.build_rows = original


def _positive_fraction(values):
    return sum(value > 0 for value in values) / len(values)


def _subtract(summary, baseline):
    result = {}
    for metric in ("margin", "CE"):
        values = [a - b for a, b in zip(summary[f"{metric}_values"],
                                        baseline[f"{metric}_values"])]
        result[f"{metric}_values"] = values
        result[f"mean_{metric}"] = statistics.fmean(values)
    return result


def score(evidence, exactness, bars=BARS):
    expected = {(row["row_id"], f"{row['direction_id']}__{row['template_id']}", condition)
                for row in build_rows() for condition in CONDITIONS}
    observed = [(item.get("row_id"), item.get("cell_id"), item.get("condition"))
                for item in evidence]
    if len(observed) != len(expected) or set(observed) != expected \
            or len(set(observed)) != len(expected):
        raise OODMLP8PolarizedResponseError("evidence does not cover exact OOD screen")
    if any(type(item.get(key)) not in (int, float)
           or not math.isfinite(float(item[key])) for item in evidence
           for key in ("target_margin_improvement", "target_CE_improvement")):
        raise OODMLP8PolarizedResponseError("non-finite or missing task metric")
    grouped = defaultdict(dict)
    for item in evidence:
        grouped[item["cell_id"]].setdefault(item["condition"], []).append(item)
    cells = {}
    for cell_id, conditions in sorted(grouped.items()):
        raw = {}
        for condition in CONDITIONS:
            margins = [float(item["target_margin_improvement"])
                       for item in conditions[condition]]
            ces = [float(item["target_CE_improvement"])
                   for item in conditions[condition]]
            raw[condition] = {
                "mean_margin": statistics.fmean(margins),
                "mean_CE": statistics.fmean(ces),
                "margin_values": margins,
                "CE_values": ces,
            }
        derived = {}
        for background in ("standalone", "conditional", "lexical"):
            baseline = raw[f"{background}_recipient"]
            pieces = {component: _subtract(raw[f"{background}_{component}"], baseline)
                      for component in ("cross", "quadratic", "full")}
            denominator = {metric: max(abs(pieces["full"][f"mean_{metric}"]), 1e-12)
                           for metric in ("margin", "CE")}
            derived[background] = {
                **pieces,
                "recovery": {
                    component: {
                        metric: pieces[component][f"mean_{metric}"] / denominator[metric]
                        for metric in ("margin", "CE")}
                    for component in ("cross", "quadratic")
                },
            }
        derived["lexical_ratio"] = {
            metric: abs(derived["lexical"]["full"][f"mean_{metric}"]) /
            max(abs(derived["standalone"]["full"][f"mean_{metric}"]), 1e-12)
            for metric in ("margin", "CE")
        }
        conditional = derived["conditional"]
        removal = {}
        for removed, remaining in (("cross", "quadratic"),
                                   ("quadratic", "cross")):
            removal[f"remove_{removed}"] = {
                metric: (conditional["full"][f"mean_{metric}"]
                         - conditional[remaining][f"mean_{metric}"]) /
                        max(abs(conditional["full"][f"mean_{metric}"]), 1e-12)
                for metric in ("margin", "CE")
            }
        derived["selective_removal"] = removal
        cells[cell_id] = {"raw": raw, "derived": derived}

    exact_live = all(exactness[name] <= bars[bar] for name, bar in (
        ("native_replay_max_absolute_logit_error", "maximum_native_replay_absolute_logit_error"),
        ("state_sum_max_absolute_error", "maximum_state_sum_absolute_error"),
        ("normalized_state_max_absolute_error", "maximum_normalized_state_absolute_error"),
        ("source_term_sum_max_absolute_error", "maximum_source_term_sum_absolute_error"),
        ("product_closure_max_absolute_error", "maximum_product_closure_absolute_error"),
        ("output_closure_max_absolute_error", "maximum_output_closure_absolute_error"),
        ("propagated_recipient_MLP8_max_absolute_error", "maximum_propagated_recipient_MLP8_absolute_error"),
        ("propagated_source_MLP8_max_absolute_error", "maximum_propagated_source_MLP8_absolute_error"),
        ("gauge_invariance_max_absolute_error", "maximum_gauge_invariance_absolute_error"),
        ("parent_head_endpoint_max_absolute_error", "maximum_parent_head_endpoint_absolute_error"),
        ("same_batch_native_noop_endpoint_max_absolute_error", "maximum_same_batch_native_noop_endpoint_error"),
        ("installed_head_max_absolute_error", "maximum_installed_head_absolute_error"),
    ))

    def full_helpful(cell, background):
        full = cell["derived"][background]["full"]
        return full["mean_margin"] >= bars["minimum_full_MLP8_mean_target_margin_improvement"] \
            and full["mean_CE"] > bars["minimum_full_MLP8_mean_target_CE_improvement"] \
            and _positive_fraction(full["margin_values"]) >= bars["minimum_helpful_row_fraction"] \
            and _positive_fraction(full["CE_values"]) >= bars["minimum_helpful_row_fraction"]

    instrument = exact_live and all(full_helpful(cell, background)
        for cell in cells.values() for background in ("standalone", "conditional"))

    def signed(direction, cross_test, quadratic_test):
        selected = [cell for name, cell in cells.items() if name.startswith(direction + "__")]
        return instrument and bool(selected) and all(
            cross_test(cell["derived"][background]["recovery"]["cross"][metric])
            and quadratic_test(cell["derived"][background]["recovery"]["quadratic"][metric])
            for cell in selected for background in ("standalone", "conditional")
            for metric in ("margin", "CE"))

    plural_to_singular = signed(
        "plural_to_singular",
        lambda value: value >= bars["plural_to_singular_minimum_cross_recovery"],
        lambda value: value <= bars["plural_to_singular_maximum_quadratic_recovery"])
    singular_to_plural = signed(
        "singular_to_plural",
        lambda value: value <= bars["singular_to_plural_maximum_cross_recovery"],
        lambda value: value >= bars["singular_to_plural_minimum_quadratic_recovery"])
    background_stable = instrument and all(
        abs(cell["derived"]["standalone"]["recovery"][component][metric]
            - cell["derived"]["conditional"]["recovery"][component][metric])
        <= bars["maximum_background_recovery_difference"]
        for cell in cells.values() for component in ("cross", "quadratic")
        for metric in ("margin", "CE"))
    number_specific = instrument and all(
        max(cell["derived"]["lexical_ratio"].values())
        <= bars["maximum_number_specific_lexical_ratio"] for cell in cells.values())

    removal_pattern = instrument and all(
        all(value > 0 for value in cell["derived"]["selective_removal"]["remove_cross"].values())
        and all(value < 0 for value in cell["derived"]["selective_removal"]["remove_quadratic"].values())
        if name.startswith("plural_to_singular__") else
        all(value < 0 for value in cell["derived"]["selective_removal"]["remove_cross"].values())
        and all(value > 0 for value in cell["derived"]["selective_removal"]["remove_quadratic"].values())
        for name, cell in cells.items())
    return {**exactness, "cells": cells,
            "selective_removal_independence": "algebraically dependent on the same four corners",
            "predictions": dict(zip(PREDICTIONS, (
                bool(instrument), bool(plural_to_singular), bool(singular_to_plural),
                bool(plural_to_singular and singular_to_plural), bool(background_stable),
                bool(number_specific), bool(removal_pattern))))}


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
        raise OODMLP8PolarizedResponseError(f"refusing to overwrite {OUT}")
    torch, F, facade = v1.factors._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, exactness, diagnostics = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, plan["bars"])
    terminal = "valid_causal_screen" if scored["predictions"]["pred_a_instrument_live"] else "invalid"
    result = {
        "schema": "task14_head11_3_ood_fronted_subject_mlp8_polarized_response_factorial_result_v1",
        "candidate_id": CANDIDATE_ID,
        "terminal": terminal,
        "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": scored,
        "numerical_diagnostics": diagnostics,
        "evidence": evidence,
        "evaluated_splits": [authority.SPLIT],
        "forbidden_splits_opened": [],
        "model_forwards": 4,
        "causal_interventions": len(evidence),
    }
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
