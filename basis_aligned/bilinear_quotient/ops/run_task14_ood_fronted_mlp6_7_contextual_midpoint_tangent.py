#!/usr/bin/env python3
"""OOD fronted-syntax test of the grouped MLP6--7 midpoint tangent law."""

# BQGATE: EXPERIMENT pred_a_instrument_live pred_b_ood_midpoint_geometry pred_c_ood_midpoint_task_prediction pred_d_matched_strength_transfers pred_e_context_changes_readout pred_f_ood_lexical_specificity

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
import run_task14_head11_3_ood_fronted_subject_mlp8_polarized_response_factorial as ood
import run_task14_mlp6_7_contextual_midpoint_tangent_readout as tangent


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_ood_fronted_mlp6_7_contextual_midpoint_tangent_v1.json"
CAPABILITY_RESULT = ROOT / "circuits/fast_screens/task14_ood_fronted_mlp8_native_capability_v1_result.json"
CAPABILITY_LICENSE = ROOT / "circuits/fast_screens/task14_ood_fronted_mlp8_polarized_response_v1_capability_license.json"
OOD_MLP8_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_ood_fronted_subject_mlp8_polarized_response_factorial_v1_result.json"
MATCHED_TANGENT_RESULT = ROOT / "circuits/fast_screens/task14_mlp6_7_contextual_midpoint_tangent_readout_v1_result.json"
TANGENT_RUNNER = Path(tangent.__file__)
OUT = ROOT / "circuits/fast_screens/task14_ood_fronted_mlp6_7_contextual_midpoint_tangent_v1_result.json"
PRIOR_ART_SHA256 = "97ed7c392c37eb04625263aa4ed9e5e9d6a80a55c99b81c8e76b09b3a5d26ffe"
HASHES = {
    CAPABILITY_RESULT: "5db771b8910bca085d201893552b409e96f66d0f01921f0c1cb5e4ba905d8615",
    CAPABILITY_LICENSE: "43090b258a75e257b8bd186dd970eb40c14ff166003a8d1a7fb160e2de3303d6",
    OOD_MLP8_RESULT: "31e379a376f29a6b71cb33fc77078edcaee9a64783793984ed0ca6df1a9cfd0b",
    MATCHED_TANGENT_RESULT: "48c72ea08c2573d520e639bbd34805ce6b60f4ec10d420fbbebed8e6112a65aa",
    TANGENT_RUNNER: "43232401ebd7f0ee03d8d5cdb57b0a2452ec8a4abb253665448198913d906aac",
}
CANDIDATE_ID = "subject_verb.number_agreement.ood_fronted_mlp6_7_contextual_midpoint_tangent_v1"
BARS = {
    "maximum_numerical_absolute_error": 5e-5,
    "minimum_midpoint_cosine": .95,
    "maximum_midpoint_relative_error": .25,
    "minimum_live_absolute_margin_effect": .002,
    "minimum_live_cells": 2,
    "minimum_task_recovery": .70,
    "maximum_task_recovery": 1.30,
    "matched_strength_minimum_cosine": .989,
    "matched_strength_maximum_relative_error": .064,
    "minimum_background_error_gap": .15,
    "maximum_lexical_ratio": .25,
    "lexical_denominator_floor": .002,
}


class OODTangentError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_rows():
    return ood.build_rows()


def validate_preflight():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256:
        raise OODTangentError("prior-art receipt changed")
    for path, expected in HASHES.items():
        if _sha256(path) != expected:
            raise OODTangentError(f"bound input changed: {path.name}")
    capability = json.loads(CAPABILITY_RESULT.read_text())
    license_doc = json.loads(CAPABILITY_LICENSE.read_text())
    ood_result = json.loads(OOD_MLP8_RESULT.read_text())
    matched = json.loads(MATCHED_TANGENT_RESULT.read_text())
    if capability.get("terminal") != "pass" \
            or license_doc.get("decision") != "pass" \
            or license_doc.get("capability_result_sha256") != HASHES[CAPABILITY_RESULT]:
        raise OODTangentError("OOD native capability/license no longer passes")
    if ood_result.get("terminal") != "valid_causal_screen" \
            or ood_result.get("score", {}).get("predictions", {}).get(
                "pred_d_signed_direction_pattern") is not True:
        raise OODTangentError("OOD MLP8 response authority no longer passes")
    if matched.get("terminal") != "valid_causal_screen" \
            or matched.get("score", {}).get("predictions", {}).get(
                "pred_b_midpoint_quadratic_readout") is not True:
        raise OODTangentError("matched tangent authority no longer passes")


def compile_plan():
    validate_preflight()
    return {
        "schema": "task14_ood_fronted_mlp6_7_contextual_midpoint_tangent_plan_v1",
        "candidate_id": CANDIDATE_ID,
        "split": "OOD_TEXT_REUSE_NEW_MLP6_7_TANGENT_INTERVENTION",
        "data_status": "already-open OOD text and whole-MLP8 outcomes; grouped-source tangent is prospective",
        "row_count": 16, "subject_position": 8,
        "sources": list(tangent.SOURCES),
        "backgrounds": list(tangent.BACKGROUNDS),
        "methods": list(tangent.METHODS),
        "prior_art_sha256": PRIOR_ART_SHA256,
        "input_sha256": {path.name: digest for path, digest in HASHES.items()},
        "bars": dict(BARS),
        "price": {"physical_model_forwards": 4, "example_evaluations": 608,
                  "causal_interventions": 192,
                  "backwards": "two JVPs per source/background; no optimization",
                  "parameter_updates": 0},
        "closed_claims": ["pristine_OOD_confirmation", "semantic_uniqueness",
                          "rank", "compression", "activation_reconstruction",
                          "independent_text_coefficient_sharing",
                          "necessity_outside_fixed_L11H3_interface"],
    }


def evaluate(model, torch, F, facade):
    original = tangent.build_rows
    try:
        tangent.build_rows = build_rows
        return tangent.evaluate(model, torch, F, facade)
    finally:
        tangent.build_rows = original


def score(evidence, exactness, geometry, torch, bars=BARS):
    expected = {(row["row_id"], source, background, method)
                for row in build_rows() for source in tangent.SOURCES
                for background in tangent.BACKGROUNDS for method in tangent.METHODS}
    observed = {(x["row_id"], x["source"], x["background"], x["method"])
                for x in evidence}
    if len(evidence) != len(expected) or observed != expected:
        raise OODTangentError("incomplete or duplicate OOD evidence lattice")
    numeric = ("target_margin", "target_CE", "target_margin_improvement",
               "target_CE_improvement")
    if not all(math.isfinite(float(x[k])) for x in evidence for k in numeric):
        raise OODTangentError("non-finite OOD task evidence")
    if not all(bool(torch.isfinite(value).all())
               for by_background in geometry.values()
               for tensors in by_background.values() for value in tensors.values()):
        raise OODTangentError("non-finite OOD tangent geometry")
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in evidence:
        grouped[item["cell_id"]][(item["source"], item["background"])][item["method"]].append(item)
    cells = {}
    rows = build_rows()
    for cell_id, pairs in grouped.items():
        cells[cell_id] = {}
        indices = [i for i, row in enumerate(rows)
                   if f"{row['direction_id']}__{row['template_id']}" == cell_id]
        for (source, background), methods in pairs.items():
            entry = {}
            for method, items in methods.items():
                entry[method] = {metric: statistics.fmean(float(x[metric]) for x in items)
                                 for metric in ("target_margin", "target_CE")}
            exact_effect = entry["exact"]["target_margin"] - entry["base"]["target_margin"]
            for method in ("exact", "endpoint", "midpoint"):
                effect = entry[method]["target_margin"] - entry["base"]["target_margin"]
                entry[method]["margin_effect"] = effect
                entry[method]["margin_recovery"] = (
                    effect / max(abs(exact_effect), 1e-12)
                    * (1 if exact_effect >= 0 else -1))
            geo = geometry[source][background]
            for method in ("endpoint", "midpoint"):
                entry[method].update(tangent._vector_stats(
                    geo["exact_delta"][indices], geo[f"{method}_delta"][indices], torch))
            cells[cell_id][f"{source}:{background}"] = entry
    instrument = all(float(value) <= bars["maximum_numerical_absolute_error"]
                     for value in exactness.values())
    opposite = [entry for cell in cells.values() for key, entry in cell.items()
                if key.startswith("opposite:")]
    geometry_pass = instrument and all(
        e["midpoint"]["cosine"] >= bars["minimum_midpoint_cosine"]
        and e["midpoint"]["relative_error"] <= bars["maximum_midpoint_relative_error"]
        for e in opposite)
    live = [e for e in opposite if abs(e["exact"]["margin_effect"])
            >= bars["minimum_live_absolute_margin_effect"]]
    task_pass = instrument and len(live) >= bars["minimum_live_cells"] and all(
        bars["minimum_task_recovery"] <= e["midpoint"]["margin_recovery"]
        <= bars["maximum_task_recovery"] for e in live)
    matched_strength = instrument and all(
        e["midpoint"]["cosine"] >= bars["matched_strength_minimum_cosine"]
        and e["midpoint"]["relative_error"] <= bars["matched_strength_maximum_relative_error"]
        for e in opposite)
    gaps = [abs(cell["opposite:recipient"]["midpoint"]["relative_error"]
                - cell["opposite:donor_context"]["midpoint"]["relative_error"])
            for cell in cells.values()]
    context_change = instrument and max(gaps) >= bars["minimum_background_error_gap"]
    lexical_ratios = []
    for cell in cells.values():
        for background in tangent.BACKGROUNDS:
            denominator = max(abs(cell[f"opposite:{background}"]["exact"]["margin_effect"]),
                              bars["lexical_denominator_floor"])
            for method in ("exact", "midpoint"):
                lexical_ratios.append(abs(cell[f"lexical:{background}"][method]["margin_effect"])
                                      / denominator)
    lexical = instrument and max(lexical_ratios) <= bars["maximum_lexical_ratio"]
    return {**exactness, "cells": cells, "live_opposite_task_cells": len(live),
            "minimum_ood_midpoint_cosine": min(e["midpoint"]["cosine"] for e in opposite),
            "maximum_ood_midpoint_relative_error": max(e["midpoint"]["relative_error"] for e in opposite),
            "minimum_live_task_recovery": min((e["midpoint"]["margin_recovery"] for e in live), default=None),
            "maximum_live_task_recovery": max((e["midpoint"]["margin_recovery"] for e in live), default=None),
            "maximum_background_midpoint_error_gap": max(gaps),
            "maximum_lexical_ratio": max(lexical_ratios), "predictions": {
                "pred_a_instrument_live": bool(instrument),
                "pred_b_ood_midpoint_geometry": bool(geometry_pass),
                "pred_c_ood_midpoint_task_prediction": bool(task_pass),
                "pred_d_matched_strength_transfers": bool(matched_strength),
                "pred_e_context_changes_readout": bool(context_change),
                "pred_f_ood_lexical_specificity": bool(lexical)}}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise OODTangentError(f"refusing to overwrite {OUT}")
    torch, F, facade = tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                            verify_weights_sha256=True)
    evidence, exactness, geometry = evaluate(model, torch, F, facade)
    scored = score(evidence, exactness, geometry, torch)
    terminal = "valid_causal_screen" if scored["predictions"]["pred_a_instrument_live"] else "invalid"
    result = {"schema": "task14_ood_fronted_mlp6_7_contextual_midpoint_tangent_result_v1",
              "candidate_id": CANDIDATE_ID, "terminal": terminal, "plan": plan,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "score": scored, "evidence": evidence,
              "evaluated_splits": ["OOD_TEXT_REUSE_NEW_MLP6_7_TANGENT_INTERVENTION"],
              "forbidden_splits_opened": []}
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": scored["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
