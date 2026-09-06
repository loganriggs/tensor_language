#!/usr/bin/env python3
"""Fresh A1-score/A2-causal validation of conditioned MLP weight ranks."""

# BQGATE: EXPERIMENT pred_a_fresh_authority_capability_exact_instrument_and_price pred_b_conditioned_scores_predict_fresh_causal_effects pred_c_top_score_sets_are_freshly_enriched pred_d_mlp4_positive_control_replicates pred_e_zero_fit_and_a1_a2_separation
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
import subspace_weight_atlas as atlas
import run_aspectual_tense_activation_conditioned_mlp_writer_atlas_v1 as conditioned
import run_aspectual_tense_l9h1h4_source_position_weight_validation_v1 as source_rank


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_activation_conditioned_mlp_fresh_validation_v1.json"
BUILDER = ROOT / "ops/circuit_candidate_aspectual_tense_matched_fresh_lexicon_v3.py"
SUBSPACES = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_weight_subspace_v2_result.json"
CONDITIONED_RESULT = ROOT / "circuits/followups/aspectual_tense_activation_conditioned_mlp_writer_atlas_v1_result.json"
LIBRARY = ROOT / "ops/subspace_weight_atlas.py"
POSITIONED = ROOT / "ops/positioned_component_program_eval.py"
OUT = ROOT / "circuits/followups/aspectual_tense_activation_conditioned_mlp_fresh_validation_v1_result.json"
CANDIDATE_ID = "aspectual_tense.activation_conditioned_mlp_fresh_validation_v1"
EXPECTED = {
    "prior": "bded82f9d467fd3c2f768d8966d3b31cd92ef2255b4f4666de50494f6c1cc286",
    "builder": "9ba3fb077e1019a77b51415a64f5f0cda1e2ff93d82a88d47968dcdf5dac66ee",
    "subspaces": "0ae262ee932d6ecb93d1df028ac080f3a4597861620da79b7ad45a0f3ae2d16e",
    "conditioned_result": "1793307db54ee9a50794a55607aeabd0b5cf37e7f6d0e5e50a99da98112cede4",
    "library": "2e7d3a546813a6029eca6fae455ad5abd03b429fcee92432ca6fe06e835e83f5",
    "positioned": "4fc2ac355ca7c97a5a1270f88cb3302b1a39a85e62c0b0a8a1193fbc6f61bd0d",
}
HEADS = (1, 4)
MLP_COMPONENTS = tuple(positioned.Component("mlp", layer) for layer in range(9))
MAX_FORWARDS, MAX_EVALUATIONS, MAX_RECORDS = 34, 544, 288


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


def validate_static():
    paths = {"prior": PRIOR, "builder": BUILDER, "subspaces": SUBSPACES,
             "conditioned_result": CONDITIONED_RESULT, "library": LIBRARY,
             "positioned": POSITIONED}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, fresh authority, data, or implementation hash changed")
    prior, subspaces, parent = [json.loads(path.read_text())
        for path in (PRIOR, SUBSPACES, CONDITIONED_RESULT)]
    rows_by_bank = fresh.build_rows_by_bank()
    digests = fresh.validate_rows_by_bank(rows_by_bank)
    rows = {"has": [row for row in rows_by_bank["has_had"] if row["transform_id"] in {"A1", "A2"}],
            "is": [row for row in rows_by_bank["is_was"] if row["transform_id"] in {"A1", "A2"}]}
    if (prior.get("candidate_id") != CANDIDATE_ID or parent.get("terminal") != "screen"
            or digests != {"has_had": "e4ae130ce0827e28fa5e1a37799c447794e180ea888d8d6b5f64aba0a17efbcd",
                           "is_was": "049bb31eb0777197ec2fb476fa66ac8c087a6dc55c119eea97c5ec13fef295df"}
            or any(len(task_rows) != 32 for task_rows in rows.values())):
        raise ExperimentError("fresh authority counts, digests, or parent metric changed")
    return subspaces, rows


def capability_ok(selected):
    directions = {direction: sum(row["direction_id"] == direction for row in selected)
                  for direction in ("present_to_past", "past_to_present")}
    return len(selected) >= 12 and all(count >= 6 for count in directions.values()), directions


def main():
    subspaces, rows_by_task = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False,
              "maximum_model_forwards": MAX_FORWARDS,
              "maximum_example_evaluations": MAX_EVALUATIONS,
              "maximum_causal_records": MAX_RECORDS, "task_layer_scores": 18,
              "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    forwards = evaluations = 0
    selected_rows, capability = {}, {}
    for task in ("has", "is"):
        for family in ("A1", "A2"):
            rows = [row for row in rows_by_task[task] if row["transform_id"] == family]
            outputs = {}
            for side in ("base", "donor"):
                batch = das._batch(backend, rows, side=side)
                outputs[side] = backend.native(batch, capture=False)
                forwards += 1
                evaluations += len(rows)
            selected = [row for index, row in enumerate(rows)
                        if outputs["base"].answer_foil[index][0] - outputs["base"].answer_foil[index][1] > 0
                        and outputs["donor"].answer_foil[index][0] - outputs["donor"].answer_foil[index][1] > 0]
            passed, directions = capability_ok(selected)
            selected_rows[(task, family)] = selected
            capability[f"{task}_{family}"] = {"total": len(rows), "jointly_capable": len(selected),
                "direction_counts": directions, "minimum_rows": 12,
                "minimum_per_direction": 6, "passed": passed}

    scores, records, all_finite = {}, [], True
    for task in ("has", "is"):
        fit_rows = selected_rows[(task, "A1")]
        base_batch, donor_batch = das._batch(backend, fit_rows, side="base"), das._batch(backend, fit_rows, side="donor")
        base_output, base_inputs = conditioned.capture_mlp_inputs(backend, base_batch)
        donor_output, donor_inputs = conditioned.capture_mlp_inputs(backend, donor_batch)
        forwards += 2
        evaluations += 2 * len(fit_rows)
        banks = source_rank.carrier_banks(task, base_batch, donor_batch)
        basis = conditioned.stored_basis(torch, subspaces["subspaces"][task]["basis"]).to(backend.device)
        read = atlas.head_bank_value_read_map(model.transformer.h[9].attn, HEADS, basis)
        task_scores = []
        for layer in range(9):
            response = atlas.activation_conditioned_mlp_write(
                model.transformer.h[layer].mlp, read,
                base_inputs[layer], donor_inputs[layer])["response"]
            value = statistics.fmean(float(torch.linalg.vector_norm(response[index, position]))
                for index, bank in enumerate(banks) for position in bank)
            task_scores.append({"label": f"MLP{layer}", "layer": layer,
                                "conditioned_score": value})
            all_finite = all_finite and math.isfinite(value)
        scores[task] = task_scores

        causal_rows = selected_rows[(task, "A2")]
        base_batch, donor_batch = das._batch(backend, causal_rows, side="base"), das._batch(backend, causal_rows, side="donor")
        base_output = backend.native(base_batch, capture=False)
        donor_output, donor_cache = positioned.capture_full_components(
            backend, donor_batch, MLP_COMPONENTS)
        forwards += 2
        evaluations += 2 * len(causal_rows)
        banks = source_rank.carrier_banks(task, base_batch, donor_batch)
        for layer in range(9):
            output = positioned.patch_positioned_components(
                backend, base_batch, donor_batch, (MLP_COMPONENTS[layer],),
                donor_cache, banks, banks)
            forwards += 1
            evaluations += len(causal_rows)
            arm_records = source_groups.recovery_records(
                causal_rows, base_output, donor_output, output, arm=f"MLP{layer}")
            records.extend(dict(record, task=task, score_family="A1", causal_family="A2",
                                component_label=f"MLP{layer}") for record in arm_records)

    summaries, correlations, enrichment, mlp4_control = {}, {}, {}, {}
    for task in ("has", "is"):
        summaries[task] = {}
        score_by_label = {row["label"]: row["conditioned_score"] for row in scores[task]}
        for label in score_by_label:
            subset = [record for record in records if record["task"] == task
                      and record["component_label"] == label]
            summaries[task][label] = dict(source_groups.summarize(subset),
                                          conditioned_score=score_by_label[label])
        labels = sorted(score_by_label)
        correlations[task] = spearman(
            [score_by_label[label] for label in labels],
            [summaries[task][label]["mean_absolute_recovery"] for label in labels])
        ordered_score = sorted(labels, key=lambda label: (-score_by_label[label], label))
        ordered_causal = sorted(labels, key=lambda label: (
            -summaries[task][label]["mean_absolute_recovery"], label))
        top, bottom = ordered_score[:3], ordered_score[-3:]
        top_mean = statistics.fmean(summaries[task][label]["mean_absolute_recovery"] for label in top)
        bottom_mean = statistics.fmean(summaries[task][label]["mean_absolute_recovery"] for label in bottom)
        enrichment[task] = {"top_score_labels": top, "bottom_score_labels": bottom,
                            "top_mean_absolute_recovery": top_mean,
                            "bottom_mean_absolute_recovery": bottom_mean,
                            "top_over_bottom": top_mean / bottom_mean if bottom_mean > 0 else None}
        mlp4_control[task] = {"score_rank": ordered_score.index("MLP4") + 1,
                              "causal_rank": ordered_causal.index("MLP4") + 1}

    capability_pass = all(cell["passed"] for cell in capability.values())
    pred_a = bool(capability_pass and all_finite and forwards == MAX_FORWARDS
                  and evaluations <= MAX_EVALUATIONS and len(records) <= MAX_RECORDS
                  and len({(record["task"], record["component_label"], record["row_id"])
                           for record in records}) == len(records))
    pred_b = all(correlations[task] > 0.40 for task in ("has", "is"))
    pred_c = all(enrichment[task]["top_over_bottom"] is not None
                 and enrichment[task]["top_over_bottom"] >= 2.0 for task in ("has", "is"))
    pred_d = all(mlp4_control[task]["score_rank"] <= 3
                 and mlp4_control[task]["causal_rank"] <= 3 for task in ("has", "is"))
    pred_e = all(record["score_family"] == "A1" and record["causal_family"] == "A2"
                 for record in records)
    predictions = {
        "pred_a_fresh_authority_capability_exact_instrument_and_price": pred_a,
        "pred_b_conditioned_scores_predict_fresh_causal_effects": pred_b,
        "pred_c_top_score_sets_are_freshly_enriched": pred_c,
        "pred_d_mlp4_positive_control_replicates": pred_d,
        "pred_e_zero_fit_and_a1_a2_separation": pred_e,
    }
    terminal = "invalid" if not pred_a or not pred_e else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_activation_conditioned_mlp_fresh_validation_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "capability": capability, "scores": scores,
        "causal_summaries": summaries, "correlations": correlations,
        "enrichment": enrichment, "mlp4_positive_control": mlp4_control,
        "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "causal_records": len(records), "fitted_scalars": 0,
                  "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal,
        "reason": "conditioned_mlp_weight_scores_predict_fresh_causal_writers" if terminal == "screen"
                  else "conditioned_mlp_scores_do_not_meet_fresh_causal_bars" if terminal == "null"
                  else "fresh_authority_capability_separation_coverage_matching_or_price_invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "capability", "correlations",
        "enrichment", "mlp4_positive_control", "predictions", "price", "terminal", "reason")},
        sort_keys=True))


if __name__ == "__main__":
    main()
