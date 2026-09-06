#!/usr/bin/env python3
"""Greedy actual-joint typed response program for the original temporal writer."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_exact_capture pred_b_greedy_path_strictly_improves pred_c_heldout_a1_is_nearly_complete pred_d_a2_transfer_is_nearly_complete pred_e_price_and_split_are_exact
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as attention_eval
import attention_source_group_eval as source_groups
import cached_response_program_eval as response_program
import circuit_das_subspace as das
import circuit_fast_screen_candidate_temporal_auxiliary as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_original_writer_greedy_response_program_v1.json"
ATLAS = ROOT / "circuits/followups/temporal_auxiliary_will_had_original_writer_downstream_module_atlas_v1_result.json"
PAIR = ROOT / "circuits/followups/temporal_auxiliary_will_had_original_writer_attn15_h5h1_joint_v1_result.json"
INSTALLER = ROOT / "ops/cached_response_program_eval.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_original_writer_greedy_response_program_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.original_writer_greedy_response_program_v1"
EXPECTED = {
    "prior": "484567e03b4b30c21730dfaae3a45fb56778b923d10c1e2bf01f796d9405772b",
    "atlas": "f8517f43f41444b966b95ff0d8da9449f25bd95d32b726c1082066605cfd076a",
    "pair": "73433f9c265e035e22b227edfc99165a8c34be9508053c081ca2f31f971eefb8",
    "installer": "b62586239869fabadad678e9a98b944e5a349b5d7bcd3382c0a9b236c1e8bb3d",
    "producer": "528e9f50152d6dc2b28d084cd0828de58de042b893703d0f322b4a2f22c4a0a7",
    "mediation": "05d07578129db0cb8bf5e46d08e6cd6e9c2bb9c3883675586c83c3efd24b8fda",
    "attention": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
    "builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
}
COMPONENTS = {
    "attn09_h1h4": ("attn", 9, (1, 4)),
    "mlp09": ("mlp", 9, ()),
    "mlp10": ("mlp", 10, ()),
    "attn11_h3": ("attn", 11, (3,)),
    "mlp11": ("mlp", 11, ()),
    "mlp12": ("mlp", 12, ()),
    "mlp13": ("mlp", 13, ()),
    "mlp14": ("mlp", 14, ()),
    "attn15_h5h1": ("attn", 15, (5, 1)),
    "mlp15": ("mlp", 15, ()),
    "mlp16": ("mlp", 16, ()),
    "mlp17": ("mlp", 17, ()),
}
POOL = tuple(sorted(COMPONENTS))
WRITER_TARGET = {"A1": 0.23824887105543294, "A2": 0.14472067356008747}
MAX_STEPS = 6
MAX_FORWARDS, MAX_EVALUATIONS, MAX_RECORDS = 100, 1920, 1792


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def stratified_a1(rows):
    counts = {}
    fit, heldout = [], []
    for row in [row for row in rows if row["transform_id"] == "A1"]:
        direction = row["direction_id"]
        occurrence = counts.get(direction, 0)
        counts[direction] = occurrence + 1
        (fit if occurrence % 2 == 0 else heldout).append(row)
    return fit, heldout


def validate_static():
    paths = {"prior": PRIOR, "atlas": ATLAS, "pair": PAIR, "installer": INSTALLER,
             "producer": PRODUCER, "mediation": MEDIATION, "attention": ATTENTION,
             "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or implementation hash changed")
    prior, atlas, pair = [json.loads(path.read_text()) for path in (PRIOR, ATLAS, PAIR)]
    rows = [row for row in candidate.build_rows() if row["transform_id"] in {"A1", "A2"}]
    fit, heldout = stratified_a1(rows)
    a2 = [row for row in rows if row["transform_id"] == "A2"]
    balance = lambda subset: {direction: sum(row["direction_id"] == direction for row in subset)
                              for direction in ("future_to_anterior", "anterior_to_future")}
    if (prior.get("candidate_id") != CANDIDATE_ID or atlas.get("terminal") != "screen"
            or pair.get("terminal") != "screen" or tuple(prior["frozen_design"]["candidate_pool"]) != (
                "attn09_h1h4", "attn11_h3", "attn15_h5h1", "mlp09", "mlp10", "mlp11",
                "mlp12", "mlp13", "mlp14", "mlp15", "mlp16", "mlp17")
            or len(POOL) != 12 or len(fit) != 16 or len(heldout) != 16 or len(a2) != 32
            or balance(fit) != {"future_to_anterior": 8, "anterior_to_future": 8}
            or balance(heldout) != {"future_to_anterior": 8, "anterior_to_future": 8}):
        raise ExperimentError("authority, pool, or stratified split changed")
    return {"A1_fit": fit, "A1_heldout": heldout, "A2": a2}


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False,
            "split_counts": {"A1_fit": 16, "A1_heldout": 16, "A2": 32},
            "candidate_pool": list(POOL), "maximum_steps": MAX_STEPS,
            "maximum_model_forwards": MAX_FORWARDS,
            "maximum_example_evaluations": MAX_EVALUATIONS,
            "maximum_records": MAX_RECORDS, "fitted_scalars": 0,
            "transformer_backwards": 0, "model_updates": 0}


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def tagged(records, split):
    return [dict(record, split=split) for record in records]


def mse(records, target_by_row):
    return statistics.fmean((float(record["recovery"]) - target_by_row[record["row_id"]]) ** 2
                            for record in records)


def rmse(records, target_by_row):
    return math.sqrt(mse(records, target_by_row))


def ordered(labels):
    return tuple(sorted((COMPONENTS[label] for label in labels),
                        key=lambda component: (component[1], component[0] == "mlp")))


def main():
    splits = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    items, reconstruction_error = {}, 0.0
    forwards = evaluations = 0
    all_native_capable = True
    writer_records = []
    for split, rows in splits.items():
        base_batch = das._batch(backend, rows, side="base")
        donor_batch = das._batch(backend, rows, side="donor")
        base_output, writer_base = attention_eval.capture_layer_attention(
            backend, base_batch, 8, call=lambda: backend.native(base_batch, capture=True))
        donor_output, writer_donor = attention_eval.capture_layer_attention(
            backend, donor_batch, 8, call=lambda: backend.native(donor_batch, capture=True))
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, writer_base, writer_donor,
            destinations, ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(hook)
        try:
            writer_output = backend.native(base_batch, capture=True)
        finally:
            handle.remove()
        forwards += 3
        evaluations += 3 * len(rows)
        reconstruction_error = max(reconstruction_error,
                                   float(writer_base["reconstruction_max_abs"]),
                                   float(writer_donor["reconstruction_max_abs"]))
        all_native_capable = all_native_capable and capable(base_output) and capable(donor_output)
        item = {"rows": rows, "base_batch": base_batch, "base_output": base_output,
                "donor_output": donor_output, "writer_output": writer_output}
        items[split] = item
        writer_records.extend(tagged(source_groups.recovery_records(
            rows, base_output, donor_output, writer_output, arm="writer_reference"), split))

    targets = {record["row_id"]: float(record["recovery"]) for record in writer_records}
    fit_item = items["A1_fit"]
    selected, selection_steps, selection_records = [], [], []
    current_mse = statistics.fmean(value * value for row_id, value in targets.items()
                                   if any(row["row_id"] == row_id for row in splits["A1_fit"]))
    for step in range(1, MAX_STEPS + 1):
        trials = []
        for label in sorted(set(POOL) - set(selected)):
            labels = selected + [label]
            output = response_program.intervene_cached_response_program(
                backend, fit_item["base_batch"], fit_item["base_output"].captured,
                fit_item["writer_output"].captured, ordered(labels))
            forwards += 1
            evaluations += len(fit_item["rows"])
            arm = f"select_step{step:02d}:{label}"
            records = tagged(source_groups.recovery_records(
                fit_item["rows"], fit_item["base_output"], fit_item["donor_output"],
                output, arm=arm), "A1_fit")
            selection_records.extend(records)
            trials.append((mse(records, targets), label, records))
        best_mse, best_label, _best_records = min(trials, key=lambda trial: (trial[0], trial[1]))
        if not best_mse < current_mse:
            break
        improvement = current_mse - best_mse
        selected.append(best_label)
        selection_steps.append({"step": step, "added": best_label,
                                "mse_before": current_mse, "mse_after": best_mse,
                                "improvement": improvement})
        current_mse = best_mse

    evaluation_records = []
    evaluation_outputs = {}
    prefixes = [tuple(selected[:index]) for index in range(1, len(selected) + 1)]
    for split in ("A1_heldout", "A2"):
        item = items[split]
        arms = [(f"singleton:{label}", (label,)) for label in POOL]
        arms.extend((f"prefix:{index:02d}", prefix)
                    for index, prefix in enumerate(prefixes[1:], start=2))
        for arm, labels in arms:
            output = response_program.intervene_cached_response_program(
                backend, item["base_batch"], item["base_output"].captured,
                item["writer_output"].captured, ordered(labels))
            forwards += 1
            evaluations += len(item["rows"])
            records = tagged(source_groups.recovery_records(
                item["rows"], item["base_output"], item["donor_output"], output, arm=arm), split)
            evaluation_records.extend(records)
            evaluation_outputs[(split, arm)] = records

    writer_summary = {
        split: source_groups.summarize([record for record in writer_records if record["split"] == split])
        for split in splits
    }
    final_metrics, singleton_benchmarks = {}, {}
    for split in ("A1_heldout", "A2"):
        final_arm = "singleton:" + selected[0] if len(selected) == 1 else f"prefix:{len(selected):02d}"
        final_records = evaluation_outputs[(split, final_arm)] if selected else []
        singletons = {label: rmse(evaluation_outputs[(split, f"singleton:{label}")], targets)
                      for label in POOL}
        singleton_benchmarks[split] = {
            "rmse_by_component": singletons,
            "best_component": min(singletons, key=lambda label: (singletons[label], label)),
            "best_rmse": min(singletons.values()),
        }
        summary = source_groups.summarize(final_records) if final_records else None
        final_metrics[split] = None if summary is None else {
            **summary,
            "fraction_of_writer_mean": summary["mean_recovery"] / writer_summary[split]["mean_recovery"],
            "rowwise_rmse_to_writer": rmse(final_records, targets),
        }

    combined_a1_writer = source_groups.summarize(
        [record for record in writer_records if record["split"].startswith("A1_")])
    pred_a = bool(all_native_capable and reconstruction_error <= 1e-4
                  and abs(combined_a1_writer["mean_recovery"] - WRITER_TARGET["A1"]) <= 0.03
                  and abs(writer_summary["A2"]["mean_recovery"] - WRITER_TARGET["A2"]) <= 0.03)
    pred_b = bool(2 <= len(selected) <= MAX_STEPS and all(
        step["mse_after"] < step["mse_before"] for step in selection_steps))
    a1 = final_metrics["A1_heldout"]
    a2 = final_metrics["A2"]
    pred_c = bool(a1 and 0.85 <= a1["fraction_of_writer_mean"] <= 1.15
                  and a1["direction_fraction"] >= 0.80
                  and a1["rowwise_rmse_to_writer"] < singleton_benchmarks["A1_heldout"]["best_rmse"])
    pred_d = bool(a2 and 0.80 <= a2["fraction_of_writer_mean"] <= 1.20
                  and a2["direction_fraction"] >= 0.80
                  and a2["rowwise_rmse_to_writer"] < singleton_benchmarks["A2"]["best_rmse"])
    records = writer_records + selection_records + evaluation_records
    pred_e = bool(forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS
                  and len(records) <= MAX_RECORDS and len(splits["A1_fit"]) == 16
                  and len(splits["A1_heldout"]) == 16 and len(splits["A2"]) == 32
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {
        "pred_a_authority_capability_and_exact_capture": pred_a,
        "pred_b_greedy_path_strictly_improves": pred_b,
        "pred_c_heldout_a1_is_nearly_complete": pred_c,
        "pred_d_a2_transfer_is_nearly_complete": pred_d,
        "pred_e_price_and_split_are_exact": pred_e,
    }
    terminal = "invalid" if not pred_a or not pred_e else (
        "screen" if all(predictions.values()) else "null")
    result = {
        "schema": "temporal_auxiliary_original_writer_greedy_response_program_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"all_native_capable": all_native_capable,
            "attention_reconstruction_max_abs": reconstruction_error},
        "selected_path": selected, "selection_steps": selection_steps,
        "writer_summary": writer_summary, "final_metrics": final_metrics,
        "singleton_benchmarks": singleton_benchmarks, "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fitted_scalars": 0,
                  "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal,
        "reason": "sparse_greedy_response_program_generalizes" if terminal == "screen" else (
            "greedy_program_not_complete_out_of_sample" if terminal == "null"
            else "authority_capability_split_capture_coverage_or_price_invalid"),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "selected_path",
          "selection_steps", "writer_summary", "final_metrics", "singleton_benchmarks",
          "predictions", "price", "terminal", "reason")}, sort_keys=True))


if __name__ == "__main__":
    main()
