#!/usr/bin/env python3
"""Predict intermediate E/A/U/W backgrounds from transferred MLP6--7 gate coefficients."""

# BQGATE: EXPERIMENT pred_a_receipts_and_endpoints_close pred_b_matched_template_transfer pred_c_matched_to_ood_transfer pred_d_ood_to_matched_transfer pred_e_cardinality_unbiased pred_f_nontrivial_over_uniform

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

import circuit_fast_screen_managed_runner as managed
import run_task14_ood_fronted_mlp6_7_eauw_background_gate_factorial as gate


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/task14_mlp6_7_background_subset_composition_transfer_v2.json"
MATCHED_RESULT = ROOT / "circuits/fast_screens/task14_head11_3_fresh_matched_subject_mlp8_mlp4_7_source_factorial_v1_result.json"
OOD_RESULT = ROOT / "circuits/fast_screens/task14_ood_fronted_mlp6_7_eauw_background_gate_factorial_v1_result.json"
STABILITY_RESULT = ROOT / "circuits/fast_screens/task14_mlp6_7_background_gate_cross_syntax_stability_v1_result.json"
OUT = ROOT / "circuits/fast_screens/task14_mlp6_7_background_subset_composition_transfer_v2_result.json"
PRIOR_ART_SHA256 = "98177a8e28479a08f335db79a59b5d4e17b4366f0145b9ce27ab0b351f74a39a"
MATCHED_RESULT_SHA256 = "11d64cb3f3dca1b4d0d3bf50a1288c5503335e23eeb8c10754bc2907d8ee637f"
OOD_RESULT_SHA256 = "b4aec2e5b94b782f5d817c86b52567474f2986e28921044f90bec7cc5ae5e742"
STABILITY_RESULT_SHA256 = "055b457d1300b4fb1db2ce09824b102cda949c33d2da7e2a09d3ae484283fd66"
CANDIDATE_ID = "subject_verb.number_agreement.mlp6_7_background_subset_composition_transfer_v2"
SUBSETS = tuple("".join(parts) for size in range(5)
                for parts in itertools.combinations(gate.BACKGROUND_FACTORS, size))
INTERMEDIATE = tuple(x for x in SUBSETS if x not in {"", "EAUW"})
BARS = {"maximum_cell_normalized_mae": .20,
        "maximum_cell_normalized_max_error": .45,
        "maximum_cardinality_normalized_bias": .10,
        "minimum_aggregate_sse_reduction_over_uniform": .10,
        "maximum_endpoint_absolute_error": 1e-12}


class CompositionTransferError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_preflight():
    bindings = ((PRIOR_ART, PRIOR_ART_SHA256, "prior-art"),
                (MATCHED_RESULT, MATCHED_RESULT_SHA256, "matched result"),
                (OOD_RESULT, OOD_RESULT_SHA256, "OOD result"),
                (STABILITY_RESULT, STABILITY_RESULT_SHA256, "stability result"))
    for path, expected, label in bindings:
        if _sha256(path) != expected:
            raise CompositionTransferError(f"{label} changed")
    matched = json.loads(MATCHED_RESULT.read_text())
    ood = json.loads(OOD_RESULT.read_text())
    stability = json.loads(STABILITY_RESULT.read_text())
    if matched.get("terminal") != "valid_causal_screen":
        raise CompositionTransferError("matched result is not valid")
    if ood.get("terminal") != "valid_causal_screen":
        raise CompositionTransferError("OOD result is not valid")
    if stability.get("terminal") != "valid_cpu_receipt":
        raise CompositionTransferError("stability result is not valid")


def compile_plan():
    validate_preflight()
    return {"schema": "task14_mlp6_7_background_subset_composition_transfer_plan_v2",
            "candidate_id": CANDIDATE_ID,
            "data_status": "RETROSPECTIVE_FROZEN_RECEIPT_REANALYSIS",
            "prior_art_sha256": PRIOR_ART_SHA256,
            "matched_result_sha256": MATCHED_RESULT_SHA256,
            "ood_result_sha256": OOD_RESULT_SHA256,
            "stability_result_sha256": STABILITY_RESULT_SHA256,
            "bars": dict(BARS), "predicted_subsets_per_target_cell": 14,
            "price": {"model_forwards": 0, "example_evaluations": 0,
                      "causal_interventions": 0, "backwards": 0,
                      "parameter_updates": 0}}


def _matched_q(document):
    grouped = defaultdict(lambda: defaultdict(list))
    for item in document["evidence"]:
        if item["condition"] == "recipient":
            continue
        source, subset, component = item["condition"].split("_")
        if source == "opposite" and component == "full":
            grouped[item["cell_id"]][subset].append(
                float(item["opposite_target_margin_improvement"]))
    cells = {}
    all_lattice = {""} | {"".join(parts) for size in range(1, 6)
                           for parts in itertools.combinations(gate.FACTORS, size)}
    for cell_id, subsets in grouped.items():
        values = {"": 0.0}
        values.update({subset: statistics.fmean(rows) for subset, rows in subsets.items()})
        if set(values) != all_lattice:
            raise CompositionTransferError(f"incomplete matched lattice: {cell_id}")
        cells[cell_id] = {subset: values[subset + "X"] - values[subset]
                          for subset in SUBSETS}
    if len(cells) != 4:
        raise CompositionTransferError("expected four matched cells")
    return cells


def _ood_q(document):
    cells = {cell_id: {subset: float(cell["opposite"]["margin"]["q"][subset])
                       for subset in SUBSETS}
             for cell_id, cell in document["score"]["cells"].items()}
    if len(cells) != 2:
        raise CompositionTransferError("expected two OOD cells")
    return cells


def _direction(cell_id):
    return cell_id.split("__", 1)[0]


def _evaluate(target_id, q, coefficients, source_id):
    if abs(sum(coefficients.values()) - 1.0) > 1e-12:
        raise CompositionTransferError("source coefficients do not sum to one")
    delta = q["EAUW"] - q[""]
    if abs(delta) < 1e-12:
        raise CompositionTransferError(f"dead endpoint shift: {target_id}")
    predicted = {subset: q[""] + delta * sum(coefficients[f] for f in subset)
                 for subset in SUBSETS}
    residual = {subset: predicted[subset] - q[subset] for subset in SUBSETS}
    errors = [abs(residual[x]) for x in INTERMEDIATE]
    scale = abs(delta)
    cardinality_bias = {str(size): statistics.fmean(
        residual[x] / scale for x in INTERMEDIATE if len(x) == size)
        for size in (1, 2, 3)}
    return {"target_cell": target_id, "source_profile": source_id,
            "endpoint_shift": delta,
            "endpoint_maximum_absolute_error": max(abs(residual[""]), abs(residual["EAUW"])),
            "normalized_mae": statistics.fmean(errors) / scale,
            "normalized_maximum_error": max(errors) / scale,
            "cardinality_normalized_signed_bias": cardinality_bias,
            "observed_q": q, "predicted_q": predicted, "residual": residual}


def _uniform_sse(q):
    delta = q["EAUW"] - q[""]
    return sum((q[""] + delta * len(subset) / 4.0 - q[subset]) ** 2
               for subset in INTERMEDIATE)


def _score_group(entries, q_by_target):
    transferred_sse = sum(sum(float(e["residual"][s]) ** 2 for s in INTERMEDIATE)
                          for e in entries)
    uniform_sse = sum(_uniform_sse(q_by_target[e["target_cell"]]) for e in entries)
    reduction = 1.0 - transferred_sse / max(uniform_sse, 1e-30)
    return {"cell_count": len(entries),
            "maximum_normalized_mae": max(e["normalized_mae"] for e in entries),
            "maximum_normalized_error": max(e["normalized_maximum_error"] for e in entries),
            "maximum_cardinality_normalized_bias": max(
                abs(value) for e in entries
                for value in e["cardinality_normalized_signed_bias"].values()),
            "aggregate_sse_reduction_over_uniform": reduction,
            "cells": entries}


def analyze():
    validate_preflight()
    matched_doc = json.loads(MATCHED_RESULT.read_text())
    ood_doc = json.loads(OOD_RESULT.read_text())
    stability = json.loads(STABILITY_RESULT.read_text())["score"]
    matched_q, ood_q = _matched_q(matched_doc), _ood_q(ood_doc)
    matched_profiles = {k: v["absolute_share"] for k, v in stability["matched_cells"].items()}
    ood_profiles = {k: v["absolute_share"] for k, v in stability["ood_cells"].items()}
    matched_entries = []
    for target_id, q in matched_q.items():
        source_id = next(k for k in matched_q if k != target_id and _direction(k) == _direction(target_id))
        matched_entries.append(_evaluate(target_id, q, matched_profiles[source_id], source_id))
    matched_to_ood = [_evaluate(target_id, q,
        stability["matched_direction_profiles"][_direction(target_id)]["absolute_share"],
        f"matched_mean::{_direction(target_id)}") for target_id, q in ood_q.items()]
    ood_by_direction = {_direction(k): (k, v) for k, v in ood_profiles.items()}
    ood_to_matched = [_evaluate(target_id, q,
        ood_by_direction[_direction(target_id)][1], ood_by_direction[_direction(target_id)][0])
        for target_id, q in matched_q.items()]
    groups = {"matched_template_transfer": _score_group(matched_entries, matched_q),
              "matched_to_ood_transfer": _score_group(matched_to_ood, ood_q),
              "ood_to_matched_transfer": _score_group(ood_to_matched, matched_q)}
    endpoint_error = max(e["endpoint_maximum_absolute_error"]
                         for group in groups.values() for e in group["cells"])
    transfer_ok = lambda group: group["maximum_normalized_mae"] <= BARS[
        "maximum_cell_normalized_mae"] and group["maximum_normalized_error"] <= BARS[
            "maximum_cell_normalized_max_error"]
    biases_ok = all(group["maximum_cardinality_normalized_bias"] <= BARS[
        "maximum_cardinality_normalized_bias"] for group in groups.values())
    nontrivial = all(groups[name]["aggregate_sse_reduction_over_uniform"] >= BARS[
        "minimum_aggregate_sse_reduction_over_uniform"]
        for name in ("matched_to_ood_transfer", "ood_to_matched_transfer"))
    instrument = endpoint_error <= BARS["maximum_endpoint_absolute_error"]
    return {"endpoint_maximum_absolute_error": endpoint_error, **groups,
            "predictions": {"pred_a_receipts_and_endpoints_close": bool(instrument),
                "pred_b_matched_template_transfer": bool(instrument and transfer_ok(groups["matched_template_transfer"])),
                "pred_c_matched_to_ood_transfer": bool(instrument and transfer_ok(groups["matched_to_ood_transfer"])),
                "pred_d_ood_to_matched_transfer": bool(instrument and transfer_ok(groups["ood_to_matched_transfer"])),
                "pred_e_cardinality_unbiased": bool(instrument and biases_ok),
                "pred_f_nontrivial_over_uniform": bool(instrument and nontrivial)}}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise CompositionTransferError(f"refusing to overwrite {OUT}")
    score = analyze()
    result = {"schema": "task14_mlp6_7_background_subset_composition_transfer_result_v2",
              "candidate_id": CANDIDATE_ID,
              "terminal": "valid_cpu_receipt" if score["predictions"][
                  "pred_a_receipts_and_endpoints_close"] else "invalid",
              "plan": plan, "score": score,
              "evaluated_splits": ["RETROSPECTIVE_FROZEN_RECEIPT_REANALYSIS"],
              "forbidden_splits_opened": []}
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": result["terminal"], "predictions": score["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
