#!/usr/bin/env python3
"""Pristine zero-target-anchor transfer of the absolute grouped-MLP6--7 gate law."""

# BQGATE: EXPERIMENT pred_a_native_capability_license pred_b_causal_instrument pred_c_fit_five_scalar_law pred_d_zero_anchor_holdout_prediction pred_e_factor_specificity_beats_uniform pred_f_signed_gate_and_lexical_control
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics

import circuit_fast_screen_candidate_task14_pristine_split_mlp6_7_absolute_composition as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_pristine_split_mlp6_7_native_capability as capability
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as parent
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_pristine_split_mlp6_7_absolute_composition_transfer_v1.json"
OUT = ROOT / "circuits/fast_screens/task14_pristine_split_mlp6_7_absolute_composition_transfer_v1_result.json"
PRIOR_ART_SHA256 = "4a06619140c165c99b7ec5930e68bd8d2d3c6f5205722f31997ef1ee9cf8a06e"
CAPABILITY_RESULT_SHA256 = "34ec2813a997beb12c8d943bfab194ad4d32aa916c7f121a02bb554c370bccff"
CAPABILITY_LICENSE_SHA256 = "c74a00af0bd259ec79d2e63056eecf66db90f5dc611b5497477bf7937370f053"
CANDIDATE_ID = authority.CAUSAL_CANDIDATE_ID
PATCH_CHUNK_ROWS = 256
BARS = {"maximum_numerical_absolute_error": 5e-5,
        "maximum_fit_normalized_mae": .15,
        "maximum_fit_normalized_max_error": .35,
        "maximum_holdout_absolute_mae": .04,
        "maximum_holdout_normalized_mae": .30,
        "maximum_holdout_normalized_max_error": .60,
        "minimum_holdout_sse_reduction_over_uniform": .10,
        "plural_to_singular_maximum_holdout_context_shift": -.05,
        "singular_to_plural_minimum_holdout_context_shift": .05,
        "maximum_absolute_holdout_lexical_margin_effect": .02}
SUBSETS = parent.BACKGROUND_SUBSETS
INTERMEDIATE = tuple(s for s in SUBSETS if s not in {"", "EAUW"})


class AbsoluteCompositionError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive_price(row_count=40):
    installations = row_count * len(tangent.SOURCES) * len(SUBSETS) * len(parent.METHODS)
    chunks = math.ceil(installations / PATCH_CHUNK_ROWS)
    role_rows = row_count * len(authority.ROLES)
    return {"physical_model_forwards": 2 + 2 * chunks,
            "example_evaluations": 2 * role_rows + 2 * installations,
            "causal_installations": installations,
            "maximum_patch_chunk_rows": PATCH_CHUNK_ROWS,
            "patch_chunks": chunks, "backwards": 0, "parameter_updates": 0}


def validate_preflight():
    for path, expected, label in ((PRIOR_ART, PRIOR_ART_SHA256, "prior-art"),
                                  (capability.RESULT, CAPABILITY_RESULT_SHA256, "capability result"),
                                  (capability.LICENSE, CAPABILITY_LICENSE_SHA256, "capability license")):
        if _sha256(path) != expected:
            raise AbsoluteCompositionError(f"{label} changed")
    licensing.validate_causal_preflight(capability.build_gate(), capability.RESULT,
        capability.LICENSE, expected_license_sha256=CAPABILITY_LICENSE_SHA256,
        causal_candidate_id=CANDIDATE_ID)
    expected = {"physical_model_forwards": 22, "example_evaluations": 5360,
        "causal_installations": 2560, "maximum_patch_chunk_rows": 256,
        "patch_chunks": 10, "backwards": 0, "parameter_updates": 0}
    if derive_price() != expected:
        raise AbsoluteCompositionError("derived causal price changed")


def compile_plan():
    validate_preflight()
    return {"schema": "task14_pristine_split_mlp6_7_absolute_composition_transfer_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "split": "NEW_FIT_AND_UNTOUCHED_HOLDOUT_TEXT_CAUSAL_LATTICE",
        "row_count": 40, "fit_rows": 32, "holdout_rows": 8,
        "sources": list(tangent.SOURCES), "background_subsets": list(SUBSETS),
        "methods": list(parent.METHODS), "condition_count": 64,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "authority_logical_sha256": authority.EXPECTED_AUTHORITY_SHA256,
        "capability_result_sha256": CAPABILITY_RESULT_SHA256,
        "capability_license_sha256": CAPABILITY_LICENSE_SHA256,
        "deployed_gate_scalars": 10, "bars": dict(BARS), "price": derive_price()}


def evaluate(model, torch, F, facade):
    original = parent.build_rows
    try:
        parent.build_rows = authority.build_rows
        return parent.evaluate(model, torch, F, facade)
    finally:
        parent.build_rows = original


def _mobius4(q):
    terms = {}
    for size in range(1, 5):
        for parts in itertools.combinations(parent.BACKGROUND_FACTORS, size):
            subset = "".join(parts)
            terms[subset] = sum((-1) ** (size-inner_size) * q[
                "".join(f for f in parent.BACKGROUND_FACTORS if f in inner)]
                for inner_size in range(size+1)
                for inner in itertools.combinations(parts, inner_size))
    return terms


def _five_scalar(q):
    terms = _mobius4(q)
    attribution = {factor: sum(value / len(subset) for subset, value in terms.items()
                               if factor in subset)
                   for factor in parent.BACKGROUND_FACTORS}
    return {"q0": q[""], "attribution": attribution,
            "closure_error": abs(sum(attribution.values()) - (q["EAUW"] - q[""]))}


def _predict(law):
    return {s: law["q0"] + sum(law["attribution"][f] for f in s) for s in SUBSETS}


def _errors(predicted, observed, subsets, scale):
    residual = {s: predicted[s] - observed[s] for s in subsets}
    return {"absolute_mae": statistics.fmean(abs(v) for v in residual.values()),
        "normalized_mae": statistics.fmean(abs(v) for v in residual.values()) / max(abs(scale), 1e-30),
        "normalized_maximum_error": max(abs(v) for v in residual.values()) / max(abs(scale), 1e-30),
        "residual": residual}


def score(evidence, exactness, bars=BARS):
    rows = authority.build_rows(); by_row = {row["row_id"]: row for row in rows}
    expected = {(row["row_id"], f"{row['direction_id']}__{row['template_id']}", source, background, method)
        for row in rows for source in tangent.SOURCES for background in SUBSETS for method in parent.METHODS}
    observed_keys = [(x["row_id"], x["cell_id"], x["source"], x["background"], x["method"])
                     for x in evidence]
    if len(observed_keys) != len(expected) or set(observed_keys) != expected or len(set(observed_keys)) != len(expected):
        raise AbsoluteCompositionError("causal lattice incomplete or duplicated")
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for item in evidence:
        row = by_row[item["row_id"]]
        grouped[row["phase"]][item["cell_id"]][item["source"]][
            (item["background"], item["method"])].append(item)
    cells = {}; closure = 0.0
    for phase, phase_cells in grouped.items():
        cells[phase] = {}
        for cell_id, sources in phase_cells.items():
            cells[phase][cell_id] = {}
            for source, conditions in sources.items():
                metrics = {}
                for metric in ("margin", "CE"):
                    key = f"target_{metric}_improvement"; q = {}
                    for background in SUBSETS:
                        base = statistics.fmean(float(x[key]) for x in conditions[(background, "base")])
                        exact = statistics.fmean(float(x[key]) for x in conditions[(background, "exact")])
                        q[background] = exact - base
                    law = _five_scalar(q); closure = max(closure, law["closure_error"])
                    metrics[metric] = {"q": q, "five_scalar_law": law}
                cells[phase][cell_id][source] = metrics
    exactness = {**exactness, "mobius_shapley_closure_max_absolute_error": closure}
    instrument = all(float(value) <= bars["maximum_numerical_absolute_error"]
                     for value in exactness.values())
    fit_laws, fit_scores = {}, {}
    for direction in ("plural_to_singular", "singular_to_plural"):
        selected = [cell["opposite"]["margin"]["q"] for cell_id, cell in cells["FIT"].items()
                    if cell_id.startswith(direction + "__")]
        mean_q = {s: statistics.fmean(q[s] for q in selected) for s in SUBSETS}
        law = _five_scalar(mean_q); fit_laws[direction] = law
        fit_scores[direction] = _errors(_predict(law), mean_q, INTERMEDIATE,
                                        mean_q["EAUW"] - mean_q[""])
    fit_ok = all(x["normalized_mae"] <= bars["maximum_fit_normalized_mae"] and
                 x["normalized_maximum_error"] <= bars["maximum_fit_normalized_max_error"]
                 for x in fit_scores.values())
    holdout_scores = {}; specific_sse = uniform_sse = 0.0
    holdout_shifts = {}; lexical_max = 0.0
    for cell_id, cell in cells["HOLDOUT"].items():
        direction = cell_id.split("__", 1)[0]
        q = cell["opposite"]["margin"]["q"]; law = fit_laws[direction]
        predicted = _predict(law); shift = q["EAUW"] - q[""]
        value = _errors(predicted, q, SUBSETS, shift)
        value.update({"observed_q": q, "predicted_q": predicted,
                      "fit_law": law, "holdout_context_shift": shift})
        holdout_scores[cell_id] = value; holdout_shifts[direction] = shift
        specific_sse += sum((predicted[s] - q[s]) ** 2 for s in SUBSETS)
        uniform = {s: law["q0"] + sum(law["attribution"].values()) * len(s) / 4.0
                   for s in SUBSETS}
        uniform_sse += sum((uniform[s] - q[s]) ** 2 for s in SUBSETS)
        lexical_max = max(lexical_max, *(abs(x) for x in cell["lexical"]["margin"]["q"].values()))
    reduction = 1.0 - specific_sse / max(uniform_sse, 1e-30)
    holdout_ok = all(x["absolute_mae"] <= bars["maximum_holdout_absolute_mae"] and
        x["normalized_mae"] <= bars["maximum_holdout_normalized_mae"] and
        x["normalized_maximum_error"] <= bars["maximum_holdout_normalized_max_error"]
        for x in holdout_scores.values())
    signed = holdout_shifts["plural_to_singular"] <= bars[
        "plural_to_singular_maximum_holdout_context_shift"] and holdout_shifts[
            "singular_to_plural"] >= bars["singular_to_plural_minimum_holdout_context_shift"]
    lexical = lexical_max <= bars["maximum_absolute_holdout_lexical_margin_effect"]
    return {**exactness, "cells": cells, "fit_laws": fit_laws,
        "fit_five_scalar_scores": fit_scores, "holdout_zero_anchor_scores": holdout_scores,
        "holdout_sse_reduction_over_uniform": reduction,
        "holdout_direction_context_shift": holdout_shifts,
        "maximum_absolute_holdout_lexical_margin_effect": lexical_max,
        "predictions": {"pred_a_native_capability_license": True,
            "pred_b_causal_instrument": bool(instrument),
            "pred_c_fit_five_scalar_law": bool(instrument and fit_ok),
            "pred_d_zero_anchor_holdout_prediction": bool(instrument and holdout_ok),
            "pred_e_factor_specificity_beats_uniform": bool(instrument and reduction >= bars["minimum_holdout_sse_reduction_over_uniform"]),
            "pred_f_signed_gate_and_lexical_control": bool(instrument and signed and lexical)}}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists(): raise AbsoluteCompositionError(f"refusing to overwrite {OUT}")
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness)
    terminal = "valid_causal_screen" if scored["predictions"]["pred_b_causal_instrument"] else "invalid"
    result = {"schema": "task14_pristine_split_mlp6_7_absolute_composition_transfer_result_v1",
        "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "score": scored, "evidence": evidence,
        "evaluated_splits": ["NEW_FIT", "UNTOUCHED_HOLDOUT"],
        "forbidden_splits_opened": []}
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__": main()
