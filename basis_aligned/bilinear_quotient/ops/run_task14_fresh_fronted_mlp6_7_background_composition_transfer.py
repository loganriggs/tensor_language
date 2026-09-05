#!/usr/bin/env python3
"""Transfer the grouped MLP6--7 background-composition law to new fronted templates."""

# BQGATE: EXPERIMENT pred_a_native_capability_license pred_b_causal_instrument pred_c_transferred_subset_prediction pred_d_distributed_signed_gate pred_e_template_stability pred_f_lexical_effect_absolutely_small
from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import circuit_fast_screen_candidate_task14_fresh_fronted_natural_qk_number_specificity as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_fresh_fronted_mlp6_7_native_capability as capability
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as parent
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_fresh_fronted_mlp6_7_background_composition_transfer_v1.json"
AMENDMENT = ROOT / "circuits/prior_art/task14_fresh_fronted_mlp6_7_background_composition_transfer_v1_price_amendment.json"
STABILITY_RESULT = ROOT / "circuits/fast_screens/task14_mlp6_7_background_gate_cross_syntax_stability_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_fresh_fronted_mlp6_7_background_composition_transfer_v1_result.json"
PRIOR_ART_SHA256 = "1dbcb5bfbb09e6bf485c3a81c5b2912b5e5af41e596b8331d638d5e71f992cf1"
AMENDMENT_SHA256 = "bf86cc90efe6590430183c6560bb16af910dff6486aecfab6e11b31abd384eb5"
STABILITY_RESULT_SHA256 = "055b457d1300b4fb1db2ce09824b102cda949c33d2da7e2a09d3ae484283fd66"
CAPABILITY_RESULT_SHA256 = "d35b2b94e0fa7f61d24bc9fe69b81b75c8a88d9383e16d990f641cd7c8c30d1f"
CAPABILITY_LICENSE_SHA256 = "f28395a0f6bba2ef1f72a0503021e5b4c6501a8cea290dad16b56751f3819801"
CANDIDATE_ID = capability.CAUSAL_CANDIDATE_ID
BARS = {"maximum_numerical_absolute_error": 5e-5,
        "maximum_prediction_normalized_mae": .20,
        "maximum_prediction_normalized_max_error": .45,
        "minimum_distributed_factor_absolute_share": .20,
        "minimum_distributed_factor_count": 2,
        "plural_to_singular_maximum_mean_context_shift": -.05,
        "singular_to_plural_minimum_mean_context_shift": .05,
        "minimum_template_profile_cosine": .90,
        "maximum_template_share_difference": .15,
        "maximum_absolute_lexical_margin_effect": .02}


class FreshFrontedCompositionError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_rows():
    rows = []
    for source in authority.build_rows():
        endpoints = {}
        for role, old in (("recipient", "base"), ("opposite_same_lemma", "opposite"),
                          ("same_number_different_lemma", "same")):
            answer = int(source[f"{old}_answer_id"])
            endpoints[role] = {"text": source[f"{old}_text"],
                "ids": list(source[f"{old}_ids"]), "answer_id": answer,
                "foil_id": 389 if answer == 318 else 318}
        rows.append({"row_id": source["row_id"],
                     "direction_id": source["cell_id"].split("__", 1)[0],
                     "template_id": source["recipient_template"],
                     "subject_position": 8, "endpoints": endpoints})
    if len(rows) != 32 or len({row["row_id"] for row in rows}) != 32:
        raise FreshFrontedCompositionError("adapted authority is incomplete")
    return rows


def validate_preflight():
    for path, expected, label in ((PRIOR_ART, PRIOR_ART_SHA256, "prior-art"),
                                  (AMENDMENT, AMENDMENT_SHA256, "price amendment"),
                                  (STABILITY_RESULT, STABILITY_RESULT_SHA256, "stability result"),
                                  (capability.RESULT, CAPABILITY_RESULT_SHA256, "capability result"),
                                  (capability.LICENSE, CAPABILITY_LICENSE_SHA256, "capability license")):
        if _sha256(path) != expected:
            raise FreshFrontedCompositionError(f"{label} changed")
    licensing.validate_causal_preflight(
        capability.build_gate(), capability.RESULT, capability.LICENSE,
        expected_license_sha256=CAPABILITY_LICENSE_SHA256,
        causal_candidate_id=CANDIDATE_ID)
    stability = json.loads(STABILITY_RESULT.read_text())
    if stability.get("terminal") != "valid_cpu_receipt" or stability.get(
            "score", {}).get("predictions", {}).get(
                "pred_d_matched_to_ood_profile_transfers") is not True:
        raise FreshFrontedCompositionError("frozen coefficient source is not valid")


def compile_plan():
    validate_preflight()
    return {"schema": "task14_fresh_fronted_mlp6_7_background_composition_transfer_plan_v1",
            "candidate_id": CANDIDATE_ID,
            "split": "FRESH_TEXT_REUSE_NEW_MLP6_7_BACKGROUND_INTERVENTION",
            "data_status": "text/native outcomes open; grouped MLP6-7 intervention new",
            "row_count": 32, "sources": list(tangent.SOURCES),
            "background_subsets": list(parent.BACKGROUND_SUBSETS),
            "methods": list(parent.METHODS), "condition_count": 64,
            "prior_art_sha256": PRIOR_ART_SHA256,
            "price_amendment_sha256": AMENDMENT_SHA256,
            "stability_result_sha256": STABILITY_RESULT_SHA256,
            "capability_result_sha256": CAPABILITY_RESULT_SHA256,
            "capability_license_sha256": CAPABILITY_LICENSE_SHA256,
            "bars": dict(BARS),
            "price": {"physical_model_forwards": 18,
                      "example_evaluations": 4288, "causal_installations": 2048,
                      "backwards": 0, "parameter_updates": 0,
                      "maximum_patch_chunk_rows": parent.PATCH_CHUNK_ROWS}}


def evaluate(model, torch, F, facade):
    original = parent.build_rows
    try:
        parent.build_rows = build_rows
        return parent.evaluate(model, torch, F, facade)
    finally:
        parent.build_rows = original


def _cosine(left, right):
    dot = sum(left[f] * right[f] for f in parent.BACKGROUND_FACTORS)
    ln = math.sqrt(sum(left[f] ** 2 for f in parent.BACKGROUND_FACTORS))
    rn = math.sqrt(sum(right[f] ** 2 for f in parent.BACKGROUND_FACTORS))
    return dot / max(ln * rn, 1e-30)


def score(evidence, exactness, bars=BARS):
    expected = {(row["row_id"], f"{row['direction_id']}__{row['template_id']}",
                 source, background, method)
                for row in build_rows() for source in tangent.SOURCES
                for background in parent.BACKGROUND_SUBSETS for method in parent.METHODS}
    observed = [(x["row_id"], x["cell_id"], x["source"], x["background"], x["method"])
                for x in evidence]
    if len(observed) != len(expected) or set(observed) != expected or len(set(observed)) != len(expected):
        raise FreshFrontedCompositionError("causal lattice is incomplete or duplicated")
    if any(not math.isfinite(float(x[key])) for x in evidence
           for key in ("target_margin", "target_CE", "target_margin_improvement", "target_CE_improvement")):
        raise FreshFrontedCompositionError("causal evidence is non-finite")
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in evidence:
        grouped[item["cell_id"]][item["source"]][(item["background"], item["method"])].append(item)
    cells = {}
    closure = 0.0
    for cell_id, sources in grouped.items():
        cells[cell_id] = {}
        for source, conditions in sources.items():
            metrics = {}
            for metric in ("margin", "CE"):
                key = f"target_{metric}_improvement"; values = {}; q = {}
                for background in parent.BACKGROUND_SUBSETS:
                    base = statistics.fmean(float(x[key]) for x in conditions[(background, "base")])
                    exact = statistics.fmean(float(x[key]) for x in conditions[(background, "exact")])
                    values[background] = base; values[background + "X"] = exact
                    q[background] = exact - base
                terms = parent._mobius(values)
                contextual = {subset: terms[subset + "X"]
                    for size in range(1, 5) for subset in (
                        "".join(parts) for parts in __import__("itertools").combinations(
                            parent.BACKGROUND_FACTORS, size))}
                attribution = {factor: sum(value / len(subset)
                    for subset, value in contextual.items() if factor in subset)
                    for factor in parent.BACKGROUND_FACTORS}
                shift = q["EAUW"] - q[""]
                closure = max(closure, abs(sum(attribution.values()) - shift))
                mass = sum(abs(v) for v in attribution.values())
                metrics[metric] = {"q": q, "shapley_attribution": attribution,
                    "absolute_attribution_share": {f: abs(attribution[f]) / max(mass, 1e-30)
                                                   for f in parent.BACKGROUND_FACTORS},
                    "context_shift": shift}
            cells[cell_id][source] = metrics
    exactness = {**exactness, "mobius_shapley_closure_max_absolute_error": closure}
    instrument = all(float(value) <= bars["maximum_numerical_absolute_error"]
                     for value in exactness.values())
    matched_profiles = json.loads(STABILITY_RESULT.read_text())["score"]["matched_direction_profiles"]
    predictions = {}
    for cell_id, cell in cells.items():
        q = cell["opposite"]["margin"]["q"]
        source = matched_profiles[cell_id.split("__", 1)[0]]["absolute_share"]
        delta = q["EAUW"] - q[""]
        residual = {s: q[""] + delta * sum(source[f] for f in s) - q[s]
                    for s in parent.BACKGROUND_SUBSETS}
        intermediate = [s for s in parent.BACKGROUND_SUBSETS if s not in {"", "EAUW"}]
        predictions[cell_id] = {"normalized_mae": statistics.fmean(
            abs(residual[s]) for s in intermediate) / max(abs(delta), 1e-30),
            "normalized_maximum_error": max(abs(residual[s]) for s in intermediate)
            / max(abs(delta), 1e-30), "endpoint_shift": delta}
    transfer = instrument and all(x["normalized_mae"] <= bars[
        "maximum_prediction_normalized_mae"] and x["normalized_maximum_error"] <= bars[
            "maximum_prediction_normalized_max_error"] for x in predictions.values())
    opposite = {k: v["opposite"]["margin"] for k, v in cells.items()}
    distributed = all(sum(share >= bars["minimum_distributed_factor_absolute_share"]
                              for share in x["absolute_attribution_share"].values())
                          >= bars["minimum_distributed_factor_count"] for x in opposite.values())
    shifts = defaultdict(list)
    for cell_id, value in opposite.items(): shifts[cell_id.split("__", 1)[0]].append(value["context_shift"])
    mean_shifts = {k: statistics.fmean(v) for k, v in shifts.items()}
    signed = mean_shifts["plural_to_singular"] <= bars[
        "plural_to_singular_maximum_mean_context_shift"] and mean_shifts[
            "singular_to_plural"] >= bars["singular_to_plural_minimum_mean_context_shift"]
    templates = {}
    for direction in shifts:
        entries = [value for key, value in opposite.items() if key.startswith(direction + "__")]
        templates[direction] = {"cosine": _cosine(entries[0]["absolute_attribution_share"],
                                                    entries[1]["absolute_attribution_share"]),
            "maximum_share_difference": max(abs(entries[0]["absolute_attribution_share"][f]
                - entries[1]["absolute_attribution_share"][f]) for f in parent.BACKGROUND_FACTORS)}
    stable = all(x["cosine"] >= bars["minimum_template_profile_cosine"] and
                 x["maximum_share_difference"] <= bars["maximum_template_share_difference"]
                 for x in templates.values())
    maximum_lexical = max(abs(q) for cell in cells.values()
                          for q in cell["lexical"]["margin"]["q"].values())
    return {**exactness, "cells": cells, "transferred_predictions": predictions,
            "direction_mean_context_shift": mean_shifts,
            "template_profile_comparisons": templates,
            "maximum_absolute_lexical_margin_effect": maximum_lexical,
            "predictions": {"pred_a_native_capability_license": True,
                "pred_b_causal_instrument": bool(instrument),
                "pred_c_transferred_subset_prediction": bool(transfer),
                "pred_d_distributed_signed_gate": bool(instrument and distributed and signed),
                "pred_e_template_stability": bool(instrument and stable),
                "pred_f_lexical_effect_absolutely_small": bool(instrument and maximum_lexical <= bars["maximum_absolute_lexical_margin_effect"])}}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists(): raise FreshFrontedCompositionError(f"refusing to overwrite {OUT}")
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    evidence, exactness = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness)
    terminal = "valid_causal_screen" if scored["predictions"]["pred_b_causal_instrument"] else "invalid"
    result = {"schema": "task14_fresh_fronted_mlp6_7_background_composition_transfer_result_v1",
              "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "score": scored, "evidence": evidence,
              "evaluated_splits": ["FRESH_TEXT_REUSE_NEW_MLP6_7_BACKGROUND_INTERVENTION"],
              "forbidden_splits_opened": []}
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__": main()
