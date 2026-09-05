#!/usr/bin/env python3
"""Zero-GPU cross-syntax stability analysis of the grouped MLP6--7 gate."""

# BQGATE: EXPERIMENT pred_a_receipts_and_algebra_close pred_b_ood_direction_profile_stable pred_c_matched_template_profile_stable pred_d_matched_to_ood_profile_transfers pred_e_signed_direction_reversal_transfers pred_f_shared_factor_order

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import circuit_fast_screen_managed_runner as managed
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as gate


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_background_gate_cross_syntax_stability_v1.json"
MATCHED_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_mlp4_7_source_factorial_v1_result.json"
OOD_RESULT = ROOT / "circuits/fast_screens/task14_ood_fronted_mlp6_7_eauw_background_gate_factorial_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_background_gate_cross_syntax_stability_v1_result.json"
PRIOR_ART_SHA256 = "11925955f06e57ca8a14f29f3a675caa37f0fac4473d8299bb7611561c7634e8"
MATCHED_RESULT_SHA256 = "11d64cb3f3dca1b4d0d3bf50a1288c5503335e23eeb8c10754bc2907d8ee637f"
OOD_RESULT_SHA256 = "b4aec2e5b94b782f5d817c86b52567474f2986e28921044f90bec7cc5ae5e742"
CANDIDATE_ID = "subject_verb.number_agreement.mlp6_7_background_gate_cross_syntax_stability_v1"
BARS = {
    "maximum_algebra_absolute_error": 5e-5,
    "maximum_ood_reproduction_absolute_error": 1e-12,
    "minimum_ood_direction_profile_cosine": .995,
    "maximum_ood_direction_share_difference": .03,
    "minimum_matched_template_profile_cosine": .90,
    "maximum_matched_template_share_difference": .15,
    "minimum_cross_syntax_profile_cosine": .90,
    "maximum_cross_syntax_share_difference": .15,
}


class GateStabilityError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_preflight():
    for path, expected, label in ((PRIOR_ART, PRIOR_ART_SHA256, "prior-art"),
                                  (MATCHED_RESULT, MATCHED_RESULT_SHA256, "matched result"),
                                  (OOD_RESULT, OOD_RESULT_SHA256, "OOD result")):
        if _sha256(path) != expected:
            raise GateStabilityError(f"{label} changed")
    matched = json.loads(MATCHED_RESULT.read_text())
    ood = json.loads(OOD_RESULT.read_text())
    if matched.get("terminal") != "valid_causal_screen" \
            or matched.get("score", {}).get("predictions", {}).get(
                "pred_a_instrument_and_parent_closure") is not True:
        raise GateStabilityError("matched causal lattice is not valid")
    if ood.get("terminal") != "valid_causal_screen" \
            or ood.get("score", {}).get("predictions", {}).get(
                "pred_a_instrument_and_endpoint_closure") is not True:
        raise GateStabilityError("OOD causal lattice is not valid")


def compile_plan():
    validate_preflight()
    return {"schema": "task14_mlp6_7_background_gate_cross_syntax_stability_plan_v1",
            "candidate_id": CANDIDATE_ID,
            "data_status": "RETROSPECTIVE_FROZEN_RECEIPT_REANALYSIS",
            "prior_art_sha256": PRIOR_ART_SHA256,
            "matched_result_sha256": MATCHED_RESULT_SHA256,
            "ood_result_sha256": OOD_RESULT_SHA256,
            "bars": dict(BARS),
            "price": {"model_forwards": 0, "example_evaluations": 0,
                      "causal_interventions": 0, "backwards": 0,
                      "parameter_updates": 0}}


def _profile(attribution):
    total = sum(abs(float(attribution[f])) for f in gate.BACKGROUND_FACTORS)
    return {f: abs(float(attribution[f]))/max(total, 1e-30)
            for f in gate.BACKGROUND_FACTORS}


def _cosine(left, right):
    dot = sum(float(left[f])*float(right[f]) for f in gate.BACKGROUND_FACTORS)
    ln = math.sqrt(sum(float(left[f])**2 for f in gate.BACKGROUND_FACTORS))
    rn = math.sqrt(sum(float(right[f])**2 for f in gate.BACKGROUND_FACTORS))
    return dot/max(ln*rn, 1e-30)


def _maximum_difference(left, right):
    return max(abs(float(left[f])-float(right[f])) for f in gate.BACKGROUND_FACTORS)


def _matched_cells(document):
    grouped = defaultdict(lambda: defaultdict(list))
    for item in document["evidence"]:
        if item["condition"] == "recipient":
            continue
        source, subset, component = item["condition"].split("_")
        if source != "opposite" or component != "full":
            continue
        grouped[item["cell_id"]][subset].append(
            float(item["opposite_target_margin_improvement"]))
    cells = {}
    for cell_id, subsets in grouped.items():
        values = {"": 0.0}
        values.update({subset: statistics.fmean(items)
                       for subset, items in subsets.items()})
        if set(values) != {""} | {"".join(parts) for size in range(1, 6)
                                  for parts in __import__("itertools").combinations(
                                      gate.FACTORS, size)}:
            raise GateStabilityError("matched lattice is incomplete")
        terms = gate._mobius(values)
        contextual = {subset: terms[subset+"X"]
                      for size in range(1, 5)
                      for subset in ("".join(parts) for parts in __import__(
                          "itertools").combinations(gate.BACKGROUND_FACTORS, size))}
        attribution = {factor: sum(value/len(subset)
            for subset, value in contextual.items() if factor in subset)
            for factor in gate.BACKGROUND_FACTORS}
        q_empty = values["X"]-values[""]
        q_full = values["EAUWX"]-values["EAUW"]
        cells[cell_id] = {"attribution": attribution,
                          "absolute_share": _profile(attribution),
                          "context_shift": q_full-q_empty,
                          "algebra_error": abs(sum(attribution.values())-(q_full-q_empty))}
    return cells


def analyze():
    validate_preflight()
    matched_doc = json.loads(MATCHED_RESULT.read_text())
    ood_doc = json.loads(OOD_RESULT.read_text())
    matched = _matched_cells(matched_doc)
    ood = {}
    reproduction_error = 0.0
    for cell_id, cell in ood_doc["score"]["cells"].items():
        stored = cell["opposite"]["margin"]
        attribution = {f: float(stored["shapley_attribution"][f])
                       for f in gate.BACKGROUND_FACTORS}
        profile = _profile(attribution)
        reproduction_error = max(reproduction_error, _maximum_difference(
            profile, stored["absolute_attribution_share"]))
        ood[cell_id] = {"attribution": attribution, "absolute_share": profile,
                        "context_shift": float(stored["context_shift"]),
                        "algebra_error": abs(sum(attribution.values())
                                             - float(stored["context_shift"]))}
    max_algebra_error = max(x["algebra_error"] for x in [*matched.values(), *ood.values()])
    by_direction = defaultdict(list)
    for cell_id, cell in matched.items():
        by_direction[cell_id.split("__", 1)[0]].append(cell)
    template_comparisons = {}
    matched_direction = {}
    for direction, entries in by_direction.items():
        template_comparisons[direction] = {
            "cosine": _cosine(entries[0]["absolute_share"], entries[1]["absolute_share"]),
            "maximum_share_difference": _maximum_difference(
                entries[0]["absolute_share"], entries[1]["absolute_share"])}
        mean_attribution = {f: statistics.fmean(x["attribution"][f] for x in entries)
                            for f in gate.BACKGROUND_FACTORS}
        matched_direction[direction] = {"attribution": mean_attribution,
                                        "absolute_share": _profile(mean_attribution)}
    ood_direction = {cell_id.split("__", 1)[0]: cell for cell_id, cell in ood.items()}
    directions = ("plural_to_singular", "singular_to_plural")
    ood_direction_comparison = {
        "cosine": _cosine(ood_direction[directions[0]]["absolute_share"],
                           ood_direction[directions[1]]["absolute_share"]),
        "maximum_share_difference": _maximum_difference(
            ood_direction[directions[0]]["absolute_share"],
            ood_direction[directions[1]]["absolute_share"])}
    cross_syntax = {direction: {
        "cosine": _cosine(matched_direction[direction]["absolute_share"],
                           ood_direction[direction]["absolute_share"]),
        "maximum_share_difference": _maximum_difference(
            matched_direction[direction]["absolute_share"],
            ood_direction[direction]["absolute_share"])} for direction in directions}
    instrument = max_algebra_error <= BARS["maximum_algebra_absolute_error"] \
        and reproduction_error <= BARS["maximum_ood_reproduction_absolute_error"]
    ood_stable = instrument \
        and ood_direction_comparison["cosine"] >= BARS[
            "minimum_ood_direction_profile_cosine"] \
        and ood_direction_comparison["maximum_share_difference"] <= BARS[
            "maximum_ood_direction_share_difference"]
    matched_stable = instrument and all(
        x["cosine"] >= BARS["minimum_matched_template_profile_cosine"]
        and x["maximum_share_difference"] <= BARS[
            "maximum_matched_template_share_difference"]
        for x in template_comparisons.values())
    transfer = instrument and all(
        x["cosine"] >= BARS["minimum_cross_syntax_profile_cosine"]
        and x["maximum_share_difference"] <= BARS[
            "maximum_cross_syntax_share_difference"] for x in cross_syntax.values())
    signs = instrument and all(
        matched_direction["plural_to_singular"]["attribution"][f] < 0
        and matched_direction["singular_to_plural"]["attribution"][f] > 0
        and ood_direction["plural_to_singular"]["attribution"][f] < 0
        and ood_direction["singular_to_plural"]["attribution"][f] > 0
        for f in gate.BACKGROUND_FACTORS)
    def ordered(profile):
        ranking = sorted(gate.BACKGROUND_FACTORS, key=lambda f: profile[f], reverse=True)
        return set(ranking[:2]) == {"E", "A"} and ranking[2:] == ["W", "U"]
    factor_order = instrument and all(ordered(x["absolute_share"])
        for x in [*matched.values(), *ood.values()])
    return {"maximum_algebra_absolute_error": max_algebra_error,
            "ood_reproduction_maximum_absolute_error": reproduction_error,
            "matched_cells": matched, "ood_cells": ood,
            "matched_template_comparisons": template_comparisons,
            "matched_direction_profiles": matched_direction,
            "ood_direction_comparison": ood_direction_comparison,
            "cross_syntax_comparisons": cross_syntax,
            "predictions": {
                "pred_a_receipts_and_algebra_close": bool(instrument),
                "pred_b_ood_direction_profile_stable": bool(ood_stable),
                "pred_c_matched_template_profile_stable": bool(matched_stable),
                "pred_d_matched_to_ood_profile_transfers": bool(transfer),
                "pred_e_signed_direction_reversal_transfers": bool(signs),
                "pred_f_shared_factor_order": bool(factor_order)}}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise GateStabilityError(f"refusing to overwrite {OUT}")
    score = analyze()
    result = {"schema": "task14_mlp6_7_background_gate_cross_syntax_stability_result_v1",
              "candidate_id": CANDIDATE_ID,
              "terminal": "valid_cpu_receipt" if score["predictions"][
                  "pred_a_receipts_and_algebra_close"] else "invalid",
              "plan": plan, "score": score,
              "evaluated_splits": ["RETROSPECTIVE_FROZEN_RECEIPT_REANALYSIS"],
              "forbidden_splits_opened": []}
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": result["terminal"],
                      "predictions": score["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
