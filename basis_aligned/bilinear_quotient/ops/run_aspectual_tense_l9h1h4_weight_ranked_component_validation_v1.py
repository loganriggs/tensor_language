#!/usr/bin/env python3
"""Causally validate weight-ranked writers/readers of typed L9 value states."""

# BQGATE: EXPERIMENT pred_a_authority_capability_selection_and_exact_price pred_b_weight_scores_predict_causal_effect pred_c_top_weight_sets_are_enriched pred_d_shared_downstream_infrastructure_is_causal
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
import run_aspectual_tense_l9h1h4_shared_value_weight_subspace_v2 as capable_split


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_l9h1h4_weight_ranked_component_validation_v1.json"
ATLAS = ROOT / "circuits/followups/aspectual_tense_l9h1h4_task_typed_weight_atlas_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_capability_manifest_v1_result.json"
SPLITTER = ROOT / "ops/run_aspectual_tense_l9h1h4_shared_value_weight_subspace_v2.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/aspectual_tense_l9h1h4_weight_ranked_component_validation_v1_result.json"
CANDIDATE_ID = "aspectual_tense.l9h1h4_weight_ranked_component_validation_v1"
EXPECTED = {
    "prior": "e4bcfff6a4e7224f5dd04435f0843c75076db1b8ac100b6aeb1a6a500809436e",
    "atlas": "bd652e89fe14982c87a362d4fbd6926c1803ba3267a12d360f2151b0073ad9a3",
    "capability": "9299fe3501995b72cec637a58838fdaf85f056034a1566de3a6d6bb04e38edd6",
    "splitter": "b9d725ddd4680a9d6adb429112e4f5aa26184ae5e848b6d283ded8277ebe2e9d",
    "producer": "528e9f50152d6dc2b28d084cd0828de58de042b893703d0f322b4a2f22c4a0a7",
}
TYPES = ("upstream_attention", "upstream_mlp", "downstream_attention")
SHARED_DOWNSTREAM = ("L11H3", "L15H5", "L11H1", "L15H1", "L10H5", "L10H1")
FORWARDS, EVALUATIONS, RECORDS = 136, 2278, 2144


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
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
        "downstream_attention": (task_atlas["downstream_attention_heads"], "value_read", 6),
    }
    selected = {}
    for kind, (rows, metric, count) in specifications.items():
        ordered = sorted(rows, key=lambda row: (row[metric], row["label"]))
        bottom, top = ordered[:count], list(reversed(ordered[-count:]))
        selected[kind] = [dict(row, weight_score=row[metric], group=group)
                          for group, subset in (("top", top), ("bottom", bottom)) for row in subset]
    return selected


def validate_static():
    paths = {"prior": PRIOR, "atlas": ATLAS, "capability": CAPABILITY,
             "splitter": SPLITTER, "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    prior, atlas = [json.loads(path.read_text()) for path in (PRIOR, ATLAS)]
    splits = capable_split.validate_v2()
    chosen = {task: selections(atlas["atlases"][task]) for task in ("has", "is")}
    downstream = {task: [row["label"] for row in chosen[task]["downstream_attention"]
                         if row["group"] == "top"] for task in ("has", "is")}
    counts = {name: len(rows) for name, rows in splits.items()}
    if (prior.get("candidate_id") != CANDIDATE_ID or atlas.get("terminal") != "null"
            or downstream != {"has": list(SHARED_DOWNSTREAM), "is": list(SHARED_DOWNSTREAM)}
            or counts != {"has_fit": 16, "has_heldout": 15, "has_a2": 31,
                          "is_fit": 8, "is_heldout": 6, "is_a2": 15}):
        raise ExperimentError("frozen atlas selection or capable split changed")
    return splits, chosen


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def main():
    splits, chosen = validate_static()
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
    records = []
    forwards = evaluations = 0
    all_capable = True
    panels = {"has_heldout": splits["has_heldout"], "has_a2": splits["has_a2"],
              "is_heldout": splits["is_heldout"], "is_a2": splits["is_a2"]}
    for panel, rows in panels.items():
        task = "has" if panel.startswith("has") else "is"
        base_batch = das._batch(backend, rows, side="base")
        donor_batch = das._batch(backend, rows, side="donor")
        base_output = backend.native(base_batch, capture=False)
        donor_output = backend.native(donor_batch, capture=True)
        forwards += 2
        evaluations += 2 * len(rows)
        all_capable = all_capable and capable(base_output) and capable(donor_output)
        for kind in TYPES:
            for component in chosen[task][kind]:
                if kind.endswith("attention"):
                    output = backend.patched_heads(base_batch, layer=component["layer"],
                        heads=(component["head"],), donor_cache=donor_output.captured)
                else:
                    output = backend.patched(base_batch,
                        site=producer._site(f"mlp:{component['layer']:02d}"),
                        donor_cache=donor_output.captured)
                forwards += 1
                evaluations += len(rows)
                arm = f"{kind}:{component['label']}"
                arm_records = source_groups.recovery_records(
                    rows, base_output, donor_output, output, arm=arm)
                records.extend(dict(record, task=task, panel=panel, component_type=kind,
                    component_label=component["label"], weight_score=component["weight_score"],
                    weight_group=component["group"]) for record in arm_records)

    component_summaries = {}
    correlations = {}
    enrichment = {}
    for task in ("has", "is"):
        component_summaries[task] = {}
        correlations[task] = {}
        enrichment[task] = {}
        for kind in TYPES:
            selected = chosen[task][kind]
            summaries = {}
            for component in selected:
                subset = [record for record in records if record["task"] == task
                          and record["component_type"] == kind
                          and record["component_label"] == component["label"]]
                summary = source_groups.summarize(subset)
                summaries[component["label"]] = dict(summary,
                    weight_score=component["weight_score"], weight_group=component["group"])
            component_summaries[task][kind] = summaries
            correlations[task][kind] = spearman(
                [component["weight_score"] for component in selected],
                [summaries[component["label"]]["mean_absolute_recovery"] for component in selected])
            group_means = {group: statistics.fmean(
                summaries[component["label"]]["mean_absolute_recovery"]
                for component in selected if component["group"] == group)
                for group in ("top", "bottom")}
            enrichment[task][kind] = {**group_means,
                "top_over_bottom": group_means["top"] / group_means["bottom"]
                if group_means["bottom"] > 0 else None}

    correlation_values = [correlations[task][kind] for task in ("has", "is") for kind in TYPES]
    enrichment_passes = sum(enrichment[task][kind]["top"] > enrichment[task][kind]["bottom"]
                            for task in ("has", "is") for kind in TYPES)
    shared_causal = {label: {task: component_summaries[task]["downstream_attention"]
                             [label]["mean_absolute_recovery"] for task in ("has", "is")}
                     for label in SHARED_DOWNSTREAM}
    shared_causal_count = sum(all(values[task] >= 0.05 for task in ("has", "is"))
                              for values in shared_causal.values())
    pred_a = bool(all_capable and forwards == FORWARDS and evaluations == EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["panel"], record["arm"], record["row_id"])
                           for record in records}) == RECORDS)
    pred_b = statistics.median(correlation_values) > 0.25 and sum(
        value > 0 for value in correlation_values) >= 4
    pred_c = enrichment_passes >= 4
    pred_d = shared_causal_count >= 3
    predictions = {"pred_a_authority_capability_selection_and_exact_price": pred_a,
                   "pred_b_weight_scores_predict_causal_effect": pred_b,
                   "pred_c_top_weight_sets_are_enriched": pred_c,
                   "pred_d_shared_downstream_infrastructure_is_causal": pred_d}
    terminal = "invalid" if not pred_a else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_l9h1h4_weight_ranked_component_validation_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
              "dryrun": dryrun, "instrument": {"all_native_capable": all_capable},
              "frozen_selections": chosen, "component_summaries": component_summaries,
              "correlations": correlations, "enrichment": enrichment,
              "enrichment_passes": enrichment_passes, "shared_downstream_causal": shared_causal,
              "shared_downstream_causal_count": shared_causal_count,
              "predictions": predictions,
              "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                        "intervention_records": len(records), "fitted_scalars": 0,
                        "transformer_backwards": 0, "model_updates": 0},
              "records": records, "terminal": terminal,
              "reason": "weight_rankings_predict_shared_causal_infrastructure" if terminal == "screen"
                        else "structural_weight_incidence_not_causally_enriched" if terminal == "null"
                        else "authority_capability_selection_coverage_or_price_invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "correlations",
          "enrichment", "enrichment_passes", "shared_downstream_causal",
          "shared_downstream_causal_count", "predictions", "price", "terminal", "reason")},
          sort_keys=True))


if __name__ == "__main__":
    main()
