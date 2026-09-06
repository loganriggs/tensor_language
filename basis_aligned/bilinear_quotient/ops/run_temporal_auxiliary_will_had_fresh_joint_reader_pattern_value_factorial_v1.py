#!/usr/bin/env python3
"""Exact pattern/value/interaction factorial across the fresh temporal reader path."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_exact_factor_closure pred_b_full_factor_response_recurs pred_c_base_pattern_value_change_dominates pred_d_routing_and_interaction_are_secondary pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as attention_eval
import attention_source_group_eval as source_groups
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_joint_reader_pattern_value_factorial_v1.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_capability_manifest_v1_result.json"
PARENT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_two_reader_response_program_v1_result.json"
OLD_L9 = ROOT / "circuits/followups/temporal_auxiliary_will_had_local_l9_value_reuse_v1_result.json"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_joint_reader_pattern_value_factorial_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_joint_reader_pattern_value_factorial_v1"
EXPECTED = {
    "prior": "ef76c4658877b31c8d5298670e8117c345eb76c5525113cefd6905918272798d",
    "capability": "d59fdc0659f7db4632607a5fae860887bdf0f69f03af659b02ffcd6cc8c3be59",
    "parent": "3fe8a3e7edf8dc24f7977d0ce4a37f564cac35211b2a2043e76336ea33e023cb",
    "old_l9": "52026d5df5994c75501ea005078fd649cf5685a5ac03f9ae680f87623b2ba4e7",
    "attention": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
    "mediation": "05d07578129db0cb8bf5e46d08e6cd6e9c2bb9c3883675586c83c3efd24b8fda",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
}
FACTORS = attention_eval.RESPONSE_FACTORS
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 28, 826, 472


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def subsets():
    return tuple(subset for width in range(4) for subset in itertools.combinations(FACTORS, width))


def arm_name(subset):
    return "empty" if not subset else "+".join(subset)


def pair_error(first, second):
    return max(abs(float(left) - float(right))
               for pair_left, pair_right in zip(first.answer_foil, second.answer_foil)
               for left, right in zip(pair_left, pair_right))


def validate_static():
    paths = {"prior": PRIOR, "capability": CAPABILITY, "parent": PARENT,
             "old_l9": OLD_L9, "attention": ATTENTION, "mediation": MEDIATION,
             "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or evaluator hash changed")
    prior, capability, parent, old_l9 = [json.loads(path.read_text())
                                         for path in (PRIOR, CAPABILITY, PARENT, OLD_L9)]
    if (prior.get("candidate_id") != CANDIDATE_ID or capability.get("terminal") != "manifest"
            or parent.get("terminal") != "screen" or old_l9.get("terminal") != "null"
            or len(subsets()) != 8):
        raise ExperimentError("authority terminal or factorial changed")
    all_rows = candidate.build_rows()
    capable = {family: set(capability["jointly_capable_row_ids"][family])
               for family in ("A1", "A2")}
    rows = [row for row in all_rows if row["transform_id"] in capable
            and row["row_id"] in capable[row["transform_id"]]]
    counts = {family: sum(row["transform_id"] == family for row in rows)
              for family in ("A1", "A2")}
    if counts != {"A1": 29, "A2": 30}:
        raise ExperimentError("prospective capable population changed")
    parent_targets = {}
    for family in ("A1", "A2"):
        values = [float(record["recovery"]) for record in parent["records"]
                  if record["arm"] == "install_both_responses"
                  and record["row_id"] in capable[family]]
        if len(values) != counts[family]:
            raise ExperimentError("parent capable-row coverage changed")
        parent_targets[family] = sum(values) / len(values)
    return rows, parent_targets


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "rows": 59,
            "factors": list(FACTORS), "factor_subsets": 8,
            "model_forwards": MODEL_FORWARDS,
            "example_evaluations": EXAMPLE_EVALUATIONS, "records": RECORDS,
            "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}


def capture_with_writer(backend, item, layer):
    hook = mediation.fixed_source_delta_hook(
        backend, item["base_batch"], item["donor_batch"], item["writer_base"],
        item["writer_donor"], item["subject_positions"], ("cue",), selected_heads=(1,))
    handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(hook)
    try:
        return attention_eval.capture_layer_attention(backend, item["base_batch"], layer)
    finally:
        handle.remove()


def factor_closure_error(base_capture, changed_capture, factors, destinations, positions, heads):
    maximum = 0.0
    for index, (destination, sources) in enumerate(zip(destinations, positions)):
        for head in heads:
            expected = sum(
                (changed_capture["pattern"][index, head, destination, source].float()
                 * changed_capture["value"][index, source, head].float()
                 - base_capture["pattern"][index, head, destination, source].float()
                 * base_capture["value"][index, source, head].float())
                for source in sources)
            actual = sum((factors[name][index, destination, head] for name in FACTORS),
                         expected.new_zeros(expected.shape))
            maximum = max(maximum, float((actual - expected).abs().max()))
    return maximum


def response_spec(item, layer, heads, subset):
    base_capture = item[f"base{layer}"]
    delta = sum((item[f"factors{layer}"][factor] for factor in subset),
                base_capture["head_output"].new_zeros(
                    base_capture["head_output"].shape, dtype=__import__("torch").float32))
    changed_capture = {"head_output": base_capture["head_output"].float() + delta}
    return {"layer": layer, "base_capture": base_capture,
            "changed_capture": changed_capture, "selected_heads": heads,
            "positions_by_row": tuple((int(query),) for query in item["query_positions"])}


def main():
    rows, parent_targets = validate_static()
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
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, writer_base = attention_eval.capture_layer_attention(backend, base_batch, 8)
        donor_output, writer_donor = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        base9_output, base9 = attention_eval.capture_layer_attention(backend, base_batch, 9)
        base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        item = {"family": family, "rows": family_rows, "base_batch": base_batch,
                "donor_batch": donor_batch, "base_output": base_output,
                "donor_output": donor_output, "writer_base": writer_base,
                "writer_donor": writer_donor, "base9": base9, "base11": base11,
                "subject_positions": onset.positions_for_group(base_batch, donor_batch, "subject_onset"),
                "query_positions": tuple(int(value) for value in base_batch.semantic_positions)}
        changed9_output, item["changed9"] = capture_with_writer(backend, item, 9)
        changed11_output, item["changed11"] = capture_with_writer(backend, item, 11)
        forwards += 6
        evaluations += 6 * len(family_rows)
        identity_error = max(identity_error, pair_error(base_output, base9_output),
                             pair_error(base_output, base11_output),
                             pair_error(changed9_output, changed11_output))
        reconstruction_error = max(reconstruction_error, *(float(capture["reconstruction_max_abs"])
            for capture in (writer_base, writer_donor, base9, base11,
                            item["changed9"], item["changed11"])))
        for layer, heads in ((9, (1, 4)), (11, (3,))):
            item[f"factors{layer}"] = attention_eval.attention_response_factor_deltas(
                item[f"base{layer}"], item[f"changed{layer}"], item["query_positions"],
                item["subject_positions"], selected_heads=heads)
            factor_error = max(factor_error, factor_closure_error(
                item[f"base{layer}"], item[f"changed{layer}"], item[f"factors{layer}"],
                item["query_positions"], item["subject_positions"], heads))
        items.append(item)

    records, summaries, values = [], {}, {}
    for subset in subsets():
        arm = arm_name(subset)
        for item in items:
            specs = [response_spec(item, 9, (1, 4), subset),
                     response_spec(item, 11, (3,), subset)]
            output = attention_eval.intervene_ordered_head_output_deltas(
                backend, item["base_batch"], specs)
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
    fractions = {arm_name(subset): {family: values[(subset, family)] / full_values[family]
                 for family in ("A1", "A2")} for subset in subsets()}
    shapley = {family: {} for family in ("A1", "A2")}
    for family in shapley:
        for factor in FACTORS:
            total = 0.0
            for subset in subsets():
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
    pred_a = bool(identity_error <= 1e-4 and reconstruction_error <= 5e-4
                  and factor_error <= 1e-4)
    pred_b = all(0.85 <= full_values[family] / parent_targets[family] <= 1.15
                 and summaries[arm_name(full)][family]["direction_fraction"] >= 0.75
                 for family in ("A1", "A2"))
    pred_c = all(fractions[arm_name(value)][family] >= 0.70
                 and shapley[family][value[0]] == max(shapley[family].values())
                 for family in ("A1", "A2"))
    pred_d = all(abs(fractions[arm_name(pattern)][family]) <= 0.30
                 and abs(fractions[arm_name(interaction)][family]) <= 0.30
                 and fractions[arm_name(pattern_value)][family] >= 0.85
                 for family in ("A1", "A2"))
    pred_e = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {
        "pred_a_authority_capability_and_exact_factor_closure": pred_a,
        "pred_b_full_factor_response_recurs": pred_b,
        "pred_c_base_pattern_value_change_dominates": pred_c,
        "pred_d_routing_and_interaction_are_secondary": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "screen" if all(predictions.values()) else (
        "null" if pred_a and pred_b and pred_e else "invalid")
    result = {
        "schema": "temporal_auxiliary_fresh_joint_reader_pattern_value_factorial_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "parent_capable_row_targets": parent_targets,
        "instrument": {"capture_identity_max_abs": identity_error,
                       "attention_reconstruction_max_abs": reconstruction_error,
                       "factor_closure_max_abs": factor_error},
        "summaries": summaries, "fraction_of_full_factor_response": fractions,
        "factorial_shapley": shapley, "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fitted_scalars": 0,
                  "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal,
        "reason": "shared_value_update_operation" if terminal == "screen" else (
            "valid_factorial_rejects_shared_value_account" if terminal == "null"
            else "authority_capability_exactness_recurrence_coverage_or_price_invalid"),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
          "parent_capable_row_targets", "fraction_of_full_factor_response",
          "factorial_shapley", "predictions", "price", "terminal", "reason")}, sort_keys=True))


if __name__ == "__main__":
    main()
