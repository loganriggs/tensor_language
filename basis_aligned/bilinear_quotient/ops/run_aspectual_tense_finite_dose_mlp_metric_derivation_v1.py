#!/usr/bin/env python3
"""Develop a fixed finite-dose causal screening metric for MLP writers."""

# BQGATE: EXPERIMENT pred_a_capability_position_coverage_finiteness_and_exact_price pred_b_finite_dose_improves_both_rank_correlations pred_c_finite_dose_is_strong_enough_for_fresh_test pred_d_fixed_dose_zero_fit_development_scope
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import attention_source_group_eval as source_groups
import circuit_candidate_aspectual_tense_matched_fresh_lexicon_v3 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import positioned_component_program_eval as positioned
import run_aspectual_tense_l9h1h4_source_position_weight_validation_v1 as source_rank


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_finite_dose_mlp_metric_derivation_v1.json"
FRESH_RESULT = ROOT / "circuits/followups/aspectual_tense_activation_conditioned_mlp_fresh_validation_v1_result.json"
PATH_RESULT = ROOT / "circuits/followups/aspectual_tense_path_conditioned_mlp_metric_derivation_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_tense_matched_fresh_lexicon_v3.py"
POSITIONED = ROOT / "ops/positioned_component_program_eval.py"
OUT = ROOT / "circuits/followups/aspectual_tense_finite_dose_mlp_metric_derivation_v1_result.json"
CANDIDATE_ID = "aspectual_tense.finite_dose_mlp_metric_derivation_v1"
EXPECTED = {
    "prior": "fefc2f10c2caec169ca617ed7a80b8077aaea335a0811aae9b73d9ab6139e6d8",
    "fresh_result": "7f0189ed40879285bb9d17f167f025e0fce345ae0402959d851ce189373e4d78",
    "path_result": "802110cec74ff79bbc5b57cd62a73c4d33efd5568d1cb76fba4a68b5a773ecb2",
    "builder": "9ba3fb077e1019a77b51415a64f5f0cda1e2ff93d82a88d47968dcdf5dac66ee",
    "positioned": "4fc2ac355ca7c97a5a1270f88cb3302b1a39a85e62c0b0a8a1193fbc6f61bd0d",
}
MLP_COMPONENTS = tuple(positioned.Component("mlp", layer) for layer in range(9))
BASELINES = {"has": 0.95, "is": 0.65}
DOSE = 0.25
FORWARDS, EVALUATIONS, RECORDS = 26, 394, 270


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = float(rank)
    return result


def spearman(left, right):
    return statistics.correlation(ranks(left), ranks(right))


def margins(output):
    return [float(answer) - float(foil) for answer, foil in output.answer_foil]


def scaled_patch(backend, batch, layer, base, donor, banks):
    def hook(_module, _arguments, output):
        changed = output.clone()
        for row, bank in enumerate(banks):
            for position in bank:
                changed[row, position] = (changed[row, position].float() + DOSE * (
                    donor[row, position].float() - base[row, position].float())).to(output.dtype)
        return changed
    handle = backend.model.transformer.h[layer].mlp.register_forward_hook(hook)
    try:
        return backend.native(batch, capture=False)
    finally:
        handle.remove()


def validate_static():
    paths = {"prior": PRIOR, "fresh_result": FRESH_RESULT, "path_result": PATH_RESULT,
             "builder": BUILDER, "positioned": POSITIONED}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, development results, or implementation hash changed")
    prior, causal, path = [json.loads(item.read_text()) for item in (PRIOR, FRESH_RESULT, PATH_RESULT)]
    banks = fresh.build_rows_by_bank()
    fresh.validate_rows_by_bank(banks)
    if (prior.get("candidate_id") != CANDIDATE_ID or causal.get("terminal") != "null"
            or path.get("terminal") != "null"
            or {task: path["correlations"][task]["path_conditioned"] for task in ("has", "is")}
                != BASELINES):
        raise ExperimentError("development causal or first-order baseline changed")
    rows = {"has": [row for row in banks["has_had"] if row["transform_id"] == "A1"],
            "is": [row for row in banks["is_was"] if row["transform_id"] == "A1"]}
    return causal, rows


def main():
    causal, rows_by_task = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "dose": DOSE,
              "model_forwards": FORWARDS, "example_evaluations": EVALUATIONS,
              "low_dose_records": RECORDS, "fitted_scalars": 0,
              "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    forwards = evaluations = 0
    records, scores, correlations, rankings = [], {}, {}, {}
    all_capable, all_finite = True, True
    for task in ("has", "is"):
        rows = rows_by_task[task]
        native = {}
        for side in ("base", "donor"):
            batch = das._batch(backend, rows, side=side)
            native[side] = backend.native(batch, capture=False)
            forwards += 1
            evaluations += len(rows)
        selected = [row for index, row in enumerate(rows)
                    if margins(native["base"])[index] > 0 and margins(native["donor"])[index] > 0]
        all_capable = all_capable and len(selected) == causal["capability"][f"{task}_A1"]["jointly_capable"]
        base_batch, donor_batch = das._batch(backend, selected, side="base"), das._batch(backend, selected, side="donor")
        base_output, base_cache = positioned.capture_full_components(backend, base_batch, MLP_COMPONENTS)
        donor_output, donor_cache = positioned.capture_full_components(backend, donor_batch, MLP_COMPONENTS)
        forwards += 2
        evaluations += 2 * len(selected)
        banks = source_rank.carrier_banks(task, base_batch, donor_batch)
        task_scores = []
        for layer in range(9):
            output = scaled_patch(backend, base_batch, layer,
                base_cache[MLP_COMPONENTS[layer].site_id],
                donor_cache[MLP_COMPONENTS[layer].site_id], banks)
            forwards += 1
            evaluations += len(selected)
            arm_records = source_groups.recovery_records(
                selected, base_output, donor_output, output, arm=f"MLP{layer}:dose{DOSE}")
            tagged = [dict(record, task=task, layer=layer, dose=DOSE) for record in arm_records]
            records.extend(tagged)
            score = statistics.fmean(abs(float(record["recovery"])) for record in tagged) / DOSE
            task_scores.append({"label": f"MLP{layer}", "layer": layer,
                                "finite_dose_score": score})
            all_finite = all_finite and math.isfinite(score)
        scores[task] = task_scores
        causal_summaries = causal["causal_summaries"][task]
        labels = sorted(causal_summaries)
        correlations[task] = {"first_order_baseline": BASELINES[task],
            "finite_dose": spearman(
                [next(row["finite_dose_score"] for row in task_scores if row["label"] == label)
                 for label in labels],
                [causal_summaries[label]["mean_absolute_recovery"] for label in labels])}
        rankings[task] = [row["label"] for row in sorted(task_scores,
            key=lambda row: (-row["finite_dose_score"], row["label"]))]

    pred_a = bool(all_capable and all_finite and forwards == FORWARDS
                  and evaluations == EVALUATIONS and len(records) == RECORDS
                  and len({(record["task"], record["layer"], record["row_id"])
                           for record in records}) == RECORDS)
    pred_b = all(correlations[task]["finite_dose"] > BASELINES[task] for task in ("has", "is"))
    pred_c = all(correlations[task]["finite_dose"] > 0.80 and "MLP4" in rankings[task][:3]
                 for task in ("has", "is"))
    pred_d = DOSE == 0.25
    predictions = {
        "pred_a_capability_position_coverage_finiteness_and_exact_price": pred_a,
        "pred_b_finite_dose_improves_both_rank_correlations": pred_b,
        "pred_c_finite_dose_is_strong_enough_for_fresh_test": pred_c,
        "pred_d_fixed_dose_zero_fit_development_scope": pred_d,
    }
    terminal = "invalid" if not pred_a or not pred_d else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_finite_dose_mlp_metric_derivation_result_v1",
        "candidate_id": CANDIDATE_ID, "scope": "development_metric_derivation_only",
        "execution_policy": "managed_queue_only", "started_utc": started_utc,
        "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun, "scores": scores,
        "rankings": rankings, "correlations": correlations, "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "low_dose_records": len(records), "fitted_scalars": 0,
                  "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal,
        "reason": "finite_dose_metric_graduates_to_second_fresh_test" if terminal == "screen"
                  else "direct_causal_screening_preferred_over_further_metric_complexity" if terminal == "null"
                  else "capability_position_coverage_matching_finiteness_dose_or_price_invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "rankings", "correlations",
        "predictions", "price", "terminal", "reason")}, sort_keys=True))


if __name__ == "__main__":
    main()
