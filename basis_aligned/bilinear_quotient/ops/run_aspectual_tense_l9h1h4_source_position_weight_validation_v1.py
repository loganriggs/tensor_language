#!/usr/bin/env python3
"""Validate weight-ranked L9 writers at exact contextual carrier positions."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_instrument_and_price pred_b_weight_scores_predict_source_position_causal_effect pred_c_top_weight_sets_are_source_enriched pred_d_carrier_position_corrects_query_scope
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
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import positioned_component_program_eval as positioned
import run_aspectual_anchor_block4_contextual_source_writer_factorial_v1 as has_source
import run_aspectual_tense_l9h1h4_shared_value_weight_subspace_v2 as capable_split
import run_tense_auxiliary_is_was_mlp4_contextual_source_factorial_v1 as is_source


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_l9h1h4_source_position_weight_validation_v1.json"
ATLAS = ROOT / "circuits/followups/aspectual_tense_l9h1h4_task_typed_weight_atlas_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_capability_manifest_v1_result.json"
QUERY_RESULT = ROOT / "circuits/followups/aspectual_tense_l9h1h4_weight_ranked_component_validation_v1_result.json"
SPLITTER = ROOT / "ops/run_aspectual_tense_l9h1h4_shared_value_weight_subspace_v2.py"
POSITIONED = ROOT / "ops/positioned_component_program_eval.py"
HAS_SOURCE = ROOT / "ops/run_aspectual_anchor_block4_contextual_source_writer_factorial_v1.py"
IS_SOURCE = ROOT / "ops/run_tense_auxiliary_is_was_mlp4_contextual_source_factorial_v1.py"
OUT = ROOT / "circuits/followups/aspectual_tense_l9h1h4_source_position_weight_validation_v1_result.json"
CANDIDATE_ID = "aspectual_tense.l9h1h4_source_position_weight_validation_v1"
EXPECTED = {
    "prior": "7e2540062ad80db5977f941c6c25fecce2ee6a30346cbd5f507c64a4384feccf",
    "atlas": "bd652e89fe14982c87a362d4fbd6926c1803ba3267a12d360f2151b0073ad9a3",
    "capability": "9299fe3501995b72cec637a58838fdaf85f056034a1566de3a6d6bb04e38edd6",
    "query_result": "5596eb43af7564e42d8254db564575ebf31729c85f02cde19363a7c03bac0e6b",
    "splitter": "b9d725ddd4680a9d6adb429112e4f5aa26184ae5e848b6d283ded8277ebe2e9d",
    "positioned": "4fc2ac355ca7c97a5a1270f88cb3302b1a39a85e62c0b0a8a1193fbc6f61bd0d",
    "has_source": "3246f56399abe00a2d178a6a5a5f81bed64e0f22487c06b5c0ce7f76cf1b5691",
    "is_source": "ed6a63425d9d498d51e3e6194bef724d7cfe37558bd94f5da86792acac4f7e36"
}
TYPES = ("upstream_attention", "upstream_mlp")
FORWARDS, EVALUATIONS, RECORDS = 88, 1474, 1340


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


def selections(task_atlas):
    specifications = {
        "upstream_attention": (task_atlas["upstream_attention_heads"], "normalized_score", 6),
        "upstream_mlp": (task_atlas["upstream_mlps"], "normalized_score", 4),
    }
    selected = {}
    for kind, (rows, metric, count) in specifications.items():
        ordered = sorted(rows, key=lambda row: (row[metric], row["label"]))
        bottom, top = ordered[:count], list(reversed(ordered[-count:]))
        selected[kind] = [dict(row, weight_score=row[metric], group=group)
                          for group, subset in (("top", top), ("bottom", bottom))
                          for row in subset]
    return selected


def validate_static():
    paths = {"prior": PRIOR, "atlas": ATLAS, "capability": CAPABILITY,
             "query_result": QUERY_RESULT, "splitter": SPLITTER,
             "positioned": POSITIONED, "has_source": HAS_SOURCE, "is_source": IS_SOURCE}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    prior, atlas, query = [json.loads(path.read_text()) for path in (PRIOR, ATLAS, QUERY_RESULT)]
    splits = capable_split.validate_v2()
    chosen = {task: selections(atlas["atlases"][task]) for task in ("has", "is")}
    counts = {name: len(rows) for name, rows in splits.items()}
    query_top = {task: {kind: statistics.fmean(
        summary["mean_absolute_recovery"]
        for summary in query["component_summaries"][task][kind].values()
        if summary["weight_group"] == "top") for kind in TYPES} for task in ("has", "is")}
    if (prior.get("candidate_id") != CANDIDATE_ID or atlas.get("terminal") != "null"
            or query.get("terminal") != "null"
            or counts != {"has_fit": 16, "has_heldout": 15, "has_a2": 31,
                          "is_fit": 8, "is_heldout": 6, "is_a2": 15}
            or any(len(chosen[task]["upstream_attention"]) != 12
                   or len(chosen[task]["upstream_mlp"]) != 8 for task in ("has", "is"))):
        raise ExperimentError("frozen selection, split, or query baseline changed")
    return splits, chosen, query_top


def component_spec(kind, component):
    if kind == "upstream_attention":
        return positioned.Component("attention_heads", component["layer"], (component["head"],))
    return positioned.Component("mlp", component["layer"])


def capture_specs(chosen_components):
    by_layer = {}
    mlp_layers = set()
    for component in chosen_components["upstream_attention"]:
        by_layer.setdefault(component["layer"], set()).add(component["head"])
    for component in chosen_components["upstream_mlp"]:
        mlp_layers.add(component["layer"])
    return tuple(positioned.Component("attention_heads", layer, tuple(sorted(heads)))
                 for layer, heads in sorted(by_layer.items())) + tuple(
        positioned.Component("mlp", layer) for layer in sorted(mlp_layers))


def carrier_banks(task, base_batch, donor_batch):
    if task == "has":
        banks = has_source.source_positions(base_batch, donor_batch)
    else:
        banks = is_source.source_positions(base_batch, donor_batch)
    return tuple(tuple(bank) for bank in banks)


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def main():
    splits, chosen, query_top = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "model_forwards": FORWARDS,
              "example_evaluations": EVALUATIONS, "intervention_records": RECORDS,
              "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    panels = {"has_heldout": splits["has_heldout"], "has_a2": splits["has_a2"],
              "is_heldout": splits["is_heldout"], "is_a2": splits["is_a2"]}
    records = []
    forwards = evaluations = 0
    all_capable = True
    observed_bank_widths = {}
    for panel, rows in panels.items():
        task = "has" if panel.startswith("has") else "is"
        base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
        base_output = backend.native(base_batch, capture=False)
        all_specs = capture_specs(chosen[task])
        donor_output, donor_cache = positioned.capture_full_components(backend, donor_batch, all_specs)
        forwards += 2
        evaluations += 2 * len(rows)
        all_capable = all_capable and capable(base_output) and capable(donor_output)
        banks = carrier_banks(task, base_batch, donor_batch)
        observed_bank_widths[panel] = sorted(set(map(len, banks)))
        for kind in TYPES:
            for component in chosen[task][kind]:
                spec = component_spec(kind, component)
                output = positioned.patch_positioned_components(
                    backend, base_batch, donor_batch, (spec,), donor_cache, banks, banks
                )
                forwards += 1
                evaluations += len(rows)
                arm = f"{kind}:{component['label']}"
                arm_records = source_groups.recovery_records(rows, base_output, donor_output, output, arm=arm)
                records.extend(dict(record, task=task, panel=panel, component_type=kind,
                    component_label=component["label"], weight_score=component["weight_score"],
                    weight_group=component["group"], carrier_bank_width=len(banks[index]))
                    for index, record in enumerate(arm_records))

    component_summaries, correlations, enrichment, carrier_over_query = {}, {}, {}, {}
    for task in ("has", "is"):
        component_summaries[task], correlations[task], enrichment[task], carrier_over_query[task] = {}, {}, {}, {}
        for kind in TYPES:
            summaries = {}
            for component in chosen[task][kind]:
                subset = [record for record in records if record["task"] == task
                          and record["component_type"] == kind
                          and record["component_label"] == component["label"]]
                summaries[component["label"]] = dict(source_groups.summarize(subset),
                    weight_score=component["weight_score"], weight_group=component["group"])
            component_summaries[task][kind] = summaries
            correlations[task][kind] = spearman(
                [component["weight_score"] for component in chosen[task][kind]],
                [summaries[component["label"]]["mean_absolute_recovery"]
                 for component in chosen[task][kind]])
            means = {group: statistics.fmean(
                summaries[component["label"]]["mean_absolute_recovery"]
                for component in chosen[task][kind] if component["group"] == group)
                for group in ("top", "bottom")}
            enrichment[task][kind] = {**means,
                "top_over_bottom": means["top"] / means["bottom"] if means["bottom"] > 0 else None}
            carrier_over_query[task][kind] = means["top"] / query_top[task][kind]

    correlation_values = [correlations[task][kind] for task in ("has", "is") for kind in TYPES]
    enrichment_passes = sum(enrichment[task][kind]["top"] > enrichment[task][kind]["bottom"]
                            for task in ("has", "is") for kind in TYPES)
    scope_correction_passes = sum(carrier_over_query[task][kind] >= 2.0
                                  for task in ("has", "is") for kind in TYPES)
    pred_a = bool(all_capable and observed_bank_widths == {
        "has_heldout": [3], "has_a2": [3], "is_heldout": [2], "is_a2": [2]}
        and forwards == FORWARDS and evaluations == EVALUATIONS and len(records) == RECORDS
        and len({(record["panel"], record["arm"], record["row_id"]) for record in records}) == RECORDS)
    pred_b = statistics.median(correlation_values) > 0.25 and sum(value > 0 for value in correlation_values) >= 3
    pred_c = enrichment_passes >= 3
    pred_d = scope_correction_passes >= 3
    predictions = {
        "pred_a_authority_capability_exact_instrument_and_price": pred_a,
        "pred_b_weight_scores_predict_source_position_causal_effect": pred_b,
        "pred_c_top_weight_sets_are_source_enriched": pred_c,
        "pred_d_carrier_position_corrects_query_scope": pred_d,
    }
    terminal = "invalid" if not pred_a else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_l9h1h4_source_position_weight_validation_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "instrument": {"all_native_capable": all_capable,
                  "observed_carrier_bank_widths": observed_bank_widths},
              "frozen_selections": chosen, "query_position_top_baseline": query_top,
              "component_summaries": component_summaries, "correlations": correlations,
              "enrichment": enrichment, "enrichment_passes": enrichment_passes,
              "carrier_over_query_top_effect": carrier_over_query,
              "scope_correction_passes": scope_correction_passes, "predictions": predictions,
              "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                        "intervention_records": len(records), "fitted_scalars": 0,
                        "transformer_backwards": 0, "model_updates": 0},
              "records": records, "terminal": terminal,
              "reason": "weight_ranked_upstream_components_are_carrier_position_writers" if terminal == "screen"
                        else "weight_ranked_components_do_not_validate_as_carrier_position_writers" if terminal == "null"
                        else "authority_capability_instrument_coverage_or_price_invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "correlations",
          "enrichment", "enrichment_passes", "carrier_over_query_top_effect",
          "scope_correction_passes", "predictions", "price", "terminal", "reason")}, sort_keys=True))


if __name__ == "__main__":
    main()
