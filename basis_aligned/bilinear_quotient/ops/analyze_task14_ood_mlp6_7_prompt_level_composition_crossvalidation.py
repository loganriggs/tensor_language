#!/usr/bin/env python3
"""Leave-one-prompt-out validation of the OOD grouped-MLP6--7 composition law."""

# BQGATE: EXPERIMENT pred_a_receipt_and_row_lattices_close pred_b_endpoint_liveness_is_common pred_c_loo_prompt_prediction pred_d_loo_beats_uniform pred_e_direction_sign_is_row_stable pred_f_no_cardinality_drift

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
PRIOR_ART = ROOT / "circuits/prior_art/task14_ood_mlp6_7_prompt_level_composition_crossvalidation_v1.json"
SOURCE_RESULT = ROOT / "circuits/fast_screens/task14_ood_fronted_mlp6_7_eauw_background_gate_factorial_v1_result.json"
PARENT_RESULT = ROOT / "circuits/fast_screens/task14_mlp6_7_background_subset_composition_transfer_v2_result.json"
OUT = ROOT / "circuits/fast_screens/task14_ood_mlp6_7_prompt_level_composition_crossvalidation_v1_result.json"
PRIOR_ART_SHA256 = "b9eba11e4230c3d3ca95372a7bb6078f14c6d0c68234292dfb71aef42612e906"
SOURCE_RESULT_SHA256 = "b4aec2e5b94b782f5d817c86b52567474f2986e28921044f90bec7cc5ae5e742"
PARENT_RESULT_SHA256 = "41724c7a40cf50dbcec28995aaeea7a6d2106af5a8472652acdde54fea3e6344"
CANDIDATE_ID = "subject_verb.number_agreement.ood_mlp6_7_prompt_level_composition_crossvalidation_v1"
SUBSETS = tuple("".join(parts) for size in range(5)
                for parts in itertools.combinations(gate.BACKGROUND_FACTORS, size))
INTERMEDIATE = tuple(s for s in SUBSETS if s not in {"", "EAUW"})
BARS = {"minimum_live_endpoint_shift": .02, "minimum_live_rows": 12,
        "maximum_median_normalized_mae": .20,
        "maximum_75pct_normalized_mae": .30,
        "maximum_absolute_mae": .05,
        "minimum_sse_reduction_over_uniform": .10,
        "minimum_direction_sign_fraction": .75,
        "maximum_cardinality_normalized_bias": .15,
        "maximum_endpoint_absolute_error": 1e-12}


class PromptCrossvalidationError(ValueError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_preflight():
    for path, expected, label in ((PRIOR_ART, PRIOR_ART_SHA256, "prior-art"),
                                  (SOURCE_RESULT, SOURCE_RESULT_SHA256, "source result"),
                                  (PARENT_RESULT, PARENT_RESULT_SHA256, "parent result")):
        if _sha256(path) != expected:
            raise PromptCrossvalidationError(f"{label} changed")
    source = json.loads(SOURCE_RESULT.read_text())
    parent = json.loads(PARENT_RESULT.read_text())
    if source.get("terminal") != "valid_causal_screen":
        raise PromptCrossvalidationError("source result is not valid")
    if parent.get("terminal") != "valid_cpu_receipt":
        raise PromptCrossvalidationError("parent result is not valid")


def compile_plan():
    validate_preflight()
    return {"schema": "task14_ood_mlp6_7_prompt_level_composition_crossvalidation_plan_v1",
            "candidate_id": CANDIDATE_ID,
            "data_status": "RETROSPECTIVE_FROZEN_RECEIPT_REANALYSIS",
            "prior_art_sha256": PRIOR_ART_SHA256,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "parent_result_sha256": PARENT_RESULT_SHA256,
            "bars": dict(BARS), "folds": 16,
            "predicted_intermediate_conditions": 224,
            "price": {"model_forwards": 0, "example_evaluations": 0,
                      "causal_interventions": 0, "backwards": 0,
                      "parameter_updates": 0}}


def _row_lattices(document):
    values = defaultdict(dict)
    observed = set()
    for item in document["evidence"]:
        if item["source"] != "opposite":
            continue
        key = (item["row_id"], item["cell_id"])
        condition = (item["background"], item["method"])
        if (key, condition) in observed:
            raise PromptCrossvalidationError("duplicate row condition")
        observed.add((key, condition))
        values[key][condition] = float(item["target_margin_improvement"])
    cells = defaultdict(dict)
    expected = {(s, m) for s in SUBSETS for m in ("base", "exact")}
    for (row_id, cell_id), lattice in values.items():
        if set(lattice) != expected:
            raise PromptCrossvalidationError(f"incomplete row lattice: {row_id}")
        cells[cell_id][row_id] = {s: lattice[(s, "exact")] - lattice[(s, "base")]
                                  for s in SUBSETS}
    if sorted(len(rows) for rows in cells.values()) != [8, 8]:
        raise PromptCrossvalidationError("expected two eight-row direction cells")
    return dict(cells)


def _profile(q_rows):
    mean_q = {s: statistics.fmean(q[s] for q in q_rows) for s in SUBSETS}
    mobius = {}
    for size in range(1, 5):
        for parts in itertools.combinations(gate.BACKGROUND_FACTORS, size):
            subset = "".join(parts)
            mobius[subset] = sum((-1) ** (size-inner_size) * mean_q[
                "".join(f for f in gate.BACKGROUND_FACTORS if f in inner)]
                for inner_size in range(size+1)
                for inner in itertools.combinations(parts, inner_size))
    attribution = {factor: sum(value / len(subset) for subset, value in mobius.items()
                               if factor in subset)
                   for factor in gate.BACKGROUND_FACTORS}
    total = sum(abs(v) for v in attribution.values())
    return {factor: abs(attribution[factor]) / max(total, 1e-30)
            for factor in gate.BACKGROUND_FACTORS}


def _nearest_rank_75(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(.75 * len(ordered)) - 1)]


def analyze():
    validate_preflight()
    cells = _row_lattices(json.loads(SOURCE_RESULT.read_text()))
    row_scores = []
    endpoint_error = 0.0
    for cell_id, rows in cells.items():
        aggregate_delta = statistics.fmean(q["EAUW"] - q[""] for q in rows.values())
        for row_id, q in rows.items():
            coefficients = _profile([other for other_id, other in rows.items()
                                     if other_id != row_id])
            delta = q["EAUW"] - q[""]
            predicted = {s: q[""] + delta * sum(coefficients[f] for f in s)
                         for s in SUBSETS}
            residual = {s: predicted[s] - q[s] for s in SUBSETS}
            endpoint_error = max(endpoint_error, abs(residual[""]), abs(residual["EAUW"]))
            abs_errors = [abs(residual[s]) for s in INTERMEDIATE]
            uniform_residual = {s: q[""] + delta * len(s) / 4.0 - q[s]
                                for s in INTERMEDIATE}
            live = abs(delta) >= BARS["minimum_live_endpoint_shift"]
            row_scores.append({"cell_id": cell_id, "row_id": row_id,
                "endpoint_shift": delta, "endpoint_live": live,
                "direction_sign_matches": delta * aggregate_delta > 0,
                "absolute_mae": statistics.fmean(abs_errors),
                "normalized_mae": statistics.fmean(abs_errors) / max(abs(delta), 1e-30),
                "transferred_sse": sum(residual[s] ** 2 for s in INTERMEDIATE),
                "uniform_sse": sum(value ** 2 for value in uniform_residual.values()),
                "cardinality_normalized_signed_bias": {str(size): statistics.fmean(
                    residual[s] / max(abs(delta), 1e-30)
                    for s in INTERMEDIATE if len(s) == size) for size in (1, 2, 3)},
                "source_coefficients": coefficients})
    live = [row for row in row_scores if row["endpoint_live"]]
    normalized = [row["normalized_mae"] for row in live]
    maximum_absolute_mae = max(row["absolute_mae"] for row in row_scores)
    reduction = 1.0 - sum(row["transferred_sse"] for row in live) / max(
        sum(row["uniform_sse"] for row in live), 1e-30)
    sign_fraction = statistics.fmean(row["direction_sign_matches"] for row in live) if live else 0.0
    direction_bias = {}
    for cell_id in cells:
        selected = [row for row in live if row["cell_id"] == cell_id]
        direction_bias[cell_id] = {str(size): statistics.fmean(
            row["cardinality_normalized_signed_bias"][str(size)] for row in selected)
            if selected else math.inf for size in (1, 2, 3)}
    max_bias = max(abs(value) for x in direction_bias.values() for value in x.values())
    instrument = endpoint_error <= BARS["maximum_endpoint_absolute_error"]
    common = len(live) >= BARS["minimum_live_rows"]
    prediction = bool(live) and statistics.median(normalized) <= BARS[
        "maximum_median_normalized_mae"] and _nearest_rank_75(normalized) <= BARS[
            "maximum_75pct_normalized_mae"] and maximum_absolute_mae <= BARS[
                "maximum_absolute_mae"]
    return {"endpoint_maximum_absolute_error": endpoint_error,
            "live_row_count": len(live), "total_row_count": len(row_scores),
            "median_live_normalized_mae": statistics.median(normalized) if live else math.inf,
            "live_75pct_normalized_mae": _nearest_rank_75(normalized) if live else math.inf,
            "maximum_all_row_absolute_mae": maximum_absolute_mae,
            "live_aggregate_sse_reduction_over_uniform": reduction,
            "live_direction_sign_match_fraction": sign_fraction,
            "direction_cardinality_normalized_bias": direction_bias,
            "maximum_direction_cardinality_normalized_bias": max_bias,
            "rows": row_scores,
            "predictions": {"pred_a_receipt_and_row_lattices_close": bool(instrument),
                "pred_b_endpoint_liveness_is_common": bool(instrument and common),
                "pred_c_loo_prompt_prediction": bool(instrument and prediction),
                "pred_d_loo_beats_uniform": bool(instrument and live and reduction >= BARS["minimum_sse_reduction_over_uniform"]),
                "pred_e_direction_sign_is_row_stable": bool(instrument and live and sign_fraction >= BARS["minimum_direction_sign_fraction"]),
                "pred_f_no_cardinality_drift": bool(instrument and live and max_bias <= BARS["maximum_cardinality_normalized_bias"])}}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if OUT.exists():
        raise PromptCrossvalidationError(f"refusing to overwrite {OUT}")
    score = analyze()
    result = {"schema": "task14_ood_mlp6_7_prompt_level_composition_crossvalidation_result_v1",
              "candidate_id": CANDIDATE_ID,
              "terminal": "valid_cpu_receipt" if score["predictions"][
                  "pred_a_receipt_and_row_lattices_close"] else "invalid",
              "plan": plan, "score": score,
              "evaluated_splits": ["RETROSPECTIVE_FROZEN_RECEIPT_REANALYSIS"],
              "forbidden_splits_opened": []}
    payload = managed.atomic_create_json(OUT, result)
    print(json.dumps({"terminal": result["terminal"], "predictions": score["predictions"],
                      "result_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
