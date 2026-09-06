#!/usr/bin/env python3
"""Cross-cue confirmation of the exact temporal reader value operation."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_exact_factor_closure pred_b_full_factor_response_is_sufficient pred_c_value_operation_repeats pred_d_pattern_and_interaction_remain_secondary pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_source_destination_eval as attention_eval
import attention_source_group_eval as source_groups
import circuit_fast_screen_candidate_temporal_auxiliary as original_builder
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_auxiliary_will_had_fresh_joint_reader_pattern_value_factorial_v1 as fresh
import run_temporal_auxiliary_will_had_fresh_writer_to_reader_coefficients_v3 as parent


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_original_joint_reader_pattern_value_factorial_v1.json"
FRESH_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_joint_reader_pattern_value_factorial_v1_result.json"
CROSS_CUE = ROOT / "circuits/followups/temporal_auxiliary_will_had_cross_cue_reader_modes_v1_result.json"
WRITER = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8h1_subject_source_groups_v1_result.json"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
PARENT = ROOT / "ops/run_temporal_auxiliary_will_had_fresh_writer_to_reader_coefficients_v3.py"
FRESH_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_fresh_joint_reader_pattern_value_factorial_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_original_joint_reader_pattern_value_factorial_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.original_joint_reader_pattern_value_factorial_v1"
EXPECTED = {
    "prior": "e22d4265bc8409c95e03e2e46183127c0e015e2025e500152d29f8dd735efd9c",
    "fresh_result": "57bed3ec25f00dd417478d5e6d9a8428a244ed7e4ecfa66440e41e3699c55a6d",
    "cross_cue": "4c91680557c7c2cdcef67f69586fdc758c5605369a0be7b62477574caf1e4f42",
    "writer": "8865fe22a3c12e367709706ff0b941b3c2488d1d9608ce1921a4cfa73b22c6b9",
    "attention": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
    "builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
    "parent": "ce7822a6a0ae41f330b478663ddc8b1a48f0ce0314609cfcad0c8bc35fbf24ab",
    "fresh_runner": "7472f046a20deb2a39930e2902cf27384e21960507b66890d9b4afe8286b0270",
}
FACTORS = attention_eval.RESPONSE_FACTORS
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 28, 896, 512


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "fresh_result": FRESH_RESULT, "cross_cue": CROSS_CUE,
             "writer": WRITER, "attention": ATTENTION, "builder": BUILDER,
             "parent": PARENT, "fresh_runner": FRESH_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, builder, or helper hash changed")
    prior, fresh_result, cross_cue, writer = [json.loads(path.read_text())
        for path in (PRIOR, FRESH_RESULT, CROSS_CUE, WRITER)]
    rows = [row for row in original_builder.build_rows() if row["transform_id"] in {"A1", "A2"}]
    counts = {family: sum(row["transform_id"] == family for row in rows)
              for family in ("A1", "A2")}
    if (prior.get("candidate_id") != CANDIDATE_ID or fresh_result.get("terminal") != "screen"
            or cross_cue.get("terminal") != "shared_mode" or writer.get("terminal") != "screen"
            or counts != {"A1": 32, "A2": 32} or len(fresh.subsets()) != 8):
        raise ExperimentError("authority terminal, population, or factorial changed")
    return rows


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "rows": 64,
            "factors": list(FACTORS), "factor_subsets": 8,
            "model_forwards": MODEL_FORWARDS, "example_evaluations": EXAMPLE_EVALUATIONS,
            "records": RECORDS, "fitted_scalars": 0, "transformer_backwards": 0,
            "model_updates": 0}


def native_capable(item):
    return all(float(answer) - float(foil) > 0
               for output in (item["base_output"], item["donor_output"])
               for answer, foil in output.answer_foil)


def main():
    rows = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    items, identity_error, reconstruction_error, factor_error = [], 0.0, 0.0, 0.0
    forwards = evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        item = parent.prepare_item(backend, family_rows)
        item["family"] = family
        item["subject_positions"] = item["destinations"]
        item["query_positions"] = tuple(int(value) for value in item["base_batch"].semantic_positions)
        for layer, heads in ((9, (1, 4)), (11, (3,))):
            item[f"factors{layer}"] = attention_eval.attention_response_factor_deltas(
                item[f"base{layer}"], item[f"changed{layer}"], item["query_positions"],
                item["subject_positions"], selected_heads=heads)
            factor_error = max(factor_error, fresh.factor_closure_error(
                item[f"base{layer}"], item[f"changed{layer}"], item[f"factors{layer}"],
                item["query_positions"], item["subject_positions"], heads))
        identity_error = max(identity_error, item["identity_error"])
        reconstruction_error = max(reconstruction_error, item["reconstruction_error"])
        items.append(item)
        forwards += 6
        evaluations += 6 * len(family_rows)

    writer_summaries = {}
    for item in items:
        reference = source_groups.recovery_records(
            item["rows"], item["base_output"], item["donor_output"],
            item["writer_output"], arm="writer_reference")
        writer_summaries[item["family"]] = source_groups.summarize(reference)

    records, summaries, values = [], {}, {}
    for subset in fresh.subsets():
        arm = fresh.arm_name(subset)
        for item in items:
            output = attention_eval.intervene_ordered_head_output_deltas(
                backend, item["base_batch"],
                (fresh.response_spec(item, 9, (1, 4), subset),
                 fresh.response_spec(item, 11, (3,), subset)))
            forwards += 1
            evaluations += len(item["rows"])
            records.extend(source_groups.recovery_records(
                item["rows"], item["base_output"], item["donor_output"], output, arm=arm))
        summaries[arm] = source_groups.summarize_by_family(
            [record for record in records if record["arm"] == arm])
        for family in ("A1", "A2"):
            values[(subset, family)] = summaries[arm][family]["mean_recovery"]

    full = FACTORS
    full_values = {family: values[(full, family)] for family in ("A1", "A2")}
    fractions = {fresh.arm_name(subset): {family: values[(subset, family)] / full_values[family]
                 for family in ("A1", "A2")} for subset in fresh.subsets()}
    writer_fraction = {family: full_values[family] / writer_summaries[family]["mean_recovery"]
                       for family in ("A1", "A2")}
    shapley = {family: {} for family in ("A1", "A2")}
    for family in shapley:
        for factor in FACTORS:
            total = 0.0
            for subset in fresh.subsets():
                if factor in subset:
                    continue
                extended = tuple(item for item in FACTORS if item in set(subset) | {factor})
                total += (math.factorial(len(subset)) * math.factorial(2 - len(subset))
                          / math.factorial(3) * (values[(extended, family)] - values[(subset, family)]))
            shapley[family][factor] = total

    value = ("base_pattern_on_value_change",)
    pattern = ("pattern_on_base_value",)
    interaction = ("pattern_value_interaction",)
    pattern_value = ("pattern_on_base_value", "base_pattern_on_value_change")
    pred_a = bool(all(native_capable(item) for item in items) and identity_error <= 1e-4
                  and reconstruction_error <= 5e-4 and factor_error <= 1e-4)
    pred_b = all(0.75 <= writer_fraction[family] <= 1.20
                 and summaries[fresh.arm_name(full)][family]["direction_fraction"] >= 0.90
                 for family in ("A1", "A2"))
    pred_c = all(fractions[fresh.arm_name(value)][family] >= 0.70
                 and shapley[family][value[0]] == max(shapley[family].values())
                 for family in ("A1", "A2"))
    pred_d = all(abs(fractions[fresh.arm_name(pattern)][family]) <= 0.30
                 and abs(fractions[fresh.arm_name(interaction)][family]) <= 0.30
                 and fractions[fresh.arm_name(pattern_value)][family] >= 0.85
                 for family in ("A1", "A2"))
    pred_e = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {
        "pred_a_authority_capability_and_exact_factor_closure": pred_a,
        "pred_b_full_factor_response_is_sufficient": pred_b,
        "pred_c_value_operation_repeats": pred_c,
        "pred_d_pattern_and_interaction_remain_secondary": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else (
        "null" if pred_a and pred_b and pred_e else "invalid")
    result = {
        "schema": "temporal_auxiliary_original_joint_reader_pattern_value_factorial_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"all_native_capable": all(native_capable(item) for item in items),
            "capture_identity_max_abs": identity_error,
            "attention_reconstruction_max_abs": reconstruction_error,
            "factor_closure_max_abs": factor_error},
        "writer_reference_summaries": writer_summaries,
        "full_factor_fraction_of_writer": writer_fraction,
        "summaries": summaries, "fraction_of_full_factor_response": fractions,
        "factorial_shapley": shapley, "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fitted_scalars": 0,
                  "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal,
        "reason": "cross_cue_value_operation" if terminal == "screen" else (
            "valid_original_factorial_rejects_value_operation" if terminal == "null"
            else "authority_capability_exactness_recurrence_coverage_or_price_invalid"),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
          "writer_reference_summaries", "full_factor_fraction_of_writer",
          "fraction_of_full_factor_response", "factorial_shapley", "predictions",
          "price", "terminal", "reason")}, sort_keys=True))


if __name__ == "__main__":
    main()
