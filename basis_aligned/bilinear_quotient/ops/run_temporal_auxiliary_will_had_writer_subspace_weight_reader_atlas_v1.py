#!/usr/bin/env python3
"""Weight-derived downstream reader predictions for the temporal writer subspace."""

# BQGATE: EXPERIMENT pred_a_authority_capability_capture_and_split pred_b_writer_positive_control pred_c_known_readers_are_weight_enriched pred_d_value_beats_routing_for_causal_prediction pred_e_top_weight_readers_are_causal pred_f_price_coverage_and_zero_causal_leakage
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
import circuit_das_subspace as das
import circuit_fast_screen_candidate_temporal_auxiliary as candidate
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset
import subspace_weight_atlas as weight_atlas


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_writer_subspace_weight_reader_atlas_v1.json"
GREEDY = ROOT / "circuits/followups/temporal_auxiliary_will_had_original_writer_greedy_response_program_v1_result.json"
WEIGHTS = ROOT / "ops/subspace_weight_atlas.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
ONSET = ROOT / "ops/residual_source_onset_eval.py"
BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_writer_subspace_weight_reader_atlas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.writer_subspace_weight_reader_atlas_v1"
EXPECTED = {
    "prior": "bb9603b72f7c8c6a312c4cce7bb70866a0d84ec84d92ebcbab154a905fc70bb2",
    "greedy": "791dd6e84d598cedd38df3c2d438ff7141ee2acd0b76546d575b40b12739c61c",
    "weights": "41daa3f612ce7e6d94d1bfbe9d990e786e4c5e3d621e6fe524d6edb14dca2d13",
    "producer": "528e9f50152d6dc2b28d084cd0828de58de042b893703d0f322b4a2f22c4a0a7",
    "mediation": "05d07578129db0cb8bf5e46d08e6cd6e9c2bb9c3883675586c83c3efd24b8fda",
    "attention": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
    "onset": "c276450cc9ec7c2b0a05e2be0e88bac3df9af7003e370b99b66552083c4f4b45",
    "builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
}
LAYERS = tuple(range(9, 18))
HEADS = tuple((layer, head) for layer in LAYERS for head in range(9))
KNOWN = {(9, 1), (9, 4), (11, 3), (15, 1), (15, 5)}
FORWARDS, EVALUATIONS, RECORDS = 177, 4208, 3888


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def label(head):
    return f"L{head[0]}H{head[1]}"


def stratified_a1(rows):
    counts, fit, heldout = {}, [], []
    for row in [row for row in rows if row["transform_id"] == "A1"]:
        direction = row["direction_id"]
        occurrence = counts.get(direction, 0)
        counts[direction] = occurrence + 1
        (fit if occurrence % 2 == 0 else heldout).append(row)
    return fit, heldout


def validate_static():
    paths = {"prior": PRIOR, "greedy": GREEDY, "weights": WEIGHTS,
             "producer": PRODUCER, "mediation": MEDIATION, "attention": ATTENTION,
             "onset": ONSET, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or implementation hash changed")
    prior, greedy = [json.loads(path.read_text()) for path in (PRIOR, GREEDY)]
    rows = [row for row in candidate.build_rows() if row["transform_id"] in {"A1", "A2"}]
    fit, heldout = stratified_a1(rows)
    a2 = [row for row in rows if row["transform_id"] == "A2"]
    balance = lambda subset: sorted(sum(row["direction_id"] == direction for row in subset)
                                    for direction in {row["direction_id"] for row in subset})
    if (prior.get("candidate_id") != CANDIDATE_ID or greedy.get("terminal") != "screen"
            or len(HEADS) != 81 or len(fit) != 16 or len(heldout) != 16 or len(a2) != 32
            or balance(fit) != [8, 8] or balance(heldout) != [8, 8]):
        raise ExperimentError("authority, inventory, or stratified split changed")
    return {"A1_fit": fit, "A1_heldout": heldout, "A2": a2}


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False,
            "split_counts": {"A1_fit": 16, "A1_heldout": 16, "A2": 32},
            "heads": len(HEADS), "model_forwards": FORWARDS,
            "example_evaluations": EVALUATIONS, "records": RECORDS,
            "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def pair_error(first, second):
    return max(abs(float(a) - float(b)) for left, right in zip(first.answer_foil, second.answer_foil)
               for a, b in zip(left, right))


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = float(rank)
    return result


def spearman(left, right):
    return statistics.correlation(ranks(left), ranks(right))


def main():
    splits = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = onset.ResidualGroupBackend.load("cuda")
    items, reconstruction_error, writer_identity_error = {}, 0.0, 0.0
    forwards = evaluations = 0
    all_native_capable = True
    for split, rows in splits.items():
        base_batch = das._batch(backend, rows, side="base")
        donor_batch = das._batch(backend, rows, side="donor")
        base_output, writer_base = attention_eval.capture_layer_attention(
            backend, base_batch, 8, call=lambda: backend.native(base_batch, capture=True))
        donor_output, writer_donor = attention_eval.capture_layer_attention(
            backend, donor_batch, 8, call=lambda: backend.native(donor_batch, capture=True))
        base_state_output, base_states = backend.forward_states(base_batch, maximum_boundary=9)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, writer_base, writer_donor,
            destinations, ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(hook)
        try:
            writer_state_output, writer_states = backend.forward_states(base_batch, maximum_boundary=9)
            writer_output = backend.native(base_batch, capture=True)
        finally:
            handle.remove()
        forwards += 5
        evaluations += 5 * len(rows)
        reconstruction_error = max(reconstruction_error,
                                   float(writer_base["reconstruction_max_abs"]),
                                   float(writer_donor["reconstruction_max_abs"]))
        writer_identity_error = max(writer_identity_error, pair_error(base_output, base_state_output),
                                    pair_error(writer_output, writer_state_output))
        all_native_capable = all_native_capable and capable(base_output) and capable(donor_output)
        items[split] = {"rows": rows, "base_batch": base_batch, "base_output": base_output,
                        "donor_output": donor_output, "writer_output": writer_output,
                        "base_states": base_states, "writer_states": writer_states,
                        "destinations": destinations}

    fit = items["A1_fit"]
    delta_rows = []
    row_means = []
    for index, positions in enumerate(fit["destinations"]):
        row_deltas = [fit["writer_states"][9][index, position].float()
                      - fit["base_states"][9][index, position].float() for position in positions]
        row_means.append(backend.torch.stack(row_deltas).mean(0))
        delta_rows.append(row_deltas)
    row_means = backend.torch.stack(row_means)
    orientation = (row_means @ row_means[0]).sign()
    orientation[orientation == 0] = 1
    delta = backend.torch.stack([value * orientation[index]
                                 for index, values in enumerate(delta_rows) for value in values])
    _left, singular, vh = backend.torch.linalg.svd(delta, full_matrices=False)
    energy = singular.square()
    rank = int((energy.cumsum(0) < 0.95 * energy.sum()).sum()) + 1
    # SVD vectors are mathematically orthonormal, but float32 can miss the atlas library's
    # strict 1e-6 contract. Re-QR the same selected span; this changes only its gauge.
    basis = backend.torch.linalg.qr(vh[:rank].T.contiguous()).Q
    explained = float(energy[:rank].sum() / energy.sum())

    weight_rows = []
    for layer in LAYERS:
        factors = weight_atlas.attention_subspace_factors(
            backend.model.transformer.h[layer].attn, basis)
        for head in range(9):
            scores = factors[head]["scores"]
            routing = math.sqrt(sum(scores.get(name, 0.0) ** 2
                                    for name in ("q", "k", "q2", "k2")))
            weight_rows.append({"layer": layer, "head": head, "label": label((layer, head)),
                                "value_read": scores["v"], "routing_read": routing,
                                "output_write": scores["o"], "ov_recurrence": scores["ov"]})
    for metric in ("value_read", "routing_read", "output_write", "ov_recurrence"):
        ranked = sorted(weight_rows, key=lambda row: (row[metric], row["label"]))
        for index, row in enumerate(ranked):
            row[f"{metric}_percentile"] = index / (len(ranked) - 1)
    by_label = {row["label"]: row for row in weight_rows}
    known_value_percentiles = [by_label[label(head)]["value_read_percentile"] for head in KNOWN]

    writer_capacity = []
    output_weight = backend.model.transformer.h[8].attn.c_proj.weight.detach().float()
    for head in range(9):
        section = output_weight[:, head * 128:(head + 1) * 128]
        writer_capacity.append({"head": head, "score": float(backend.torch.linalg.matrix_norm(basis.T @ section))})
    writer_capacity.sort(key=lambda row: (-row["score"], row["head"]))
    writer_h1_rank = next(index + 1 for index, row in enumerate(writer_capacity) if row["head"] == 1)

    records = []
    for split in ("A1_heldout", "A2"):
        item = items[split]
        for layer, head in HEADS:
            output = backend.patched_heads(
                item["base_batch"], layer=layer, heads=(head,),
                donor_cache=item["writer_output"].captured)
            forwards += 1
            evaluations += len(item["rows"])
            head_records = source_groups.recovery_records(
                item["rows"], item["base_output"], item["donor_output"], output,
                arm=label((layer, head)))
            records.extend(dict(record, split=split) for record in head_records)

    writer_summaries = {split: source_groups.summarize(source_groups.recovery_records(
        item["rows"], item["base_output"], item["donor_output"], item["writer_output"],
        arm="writer_reference")) for split, item in items.items()}
    causal = {}
    correlations = {}
    for split in ("A1_heldout", "A2"):
        summaries = {label(head): source_groups.summarize(
            [record for record in records if record["split"] == split and record["arm"] == label(head)])
            for head in HEADS}
        causal[split] = {name: {**summary,
            "absolute_fraction_of_writer": summary["mean_absolute_recovery"]
            / writer_summaries[split]["mean_recovery"]} for name, summary in summaries.items()}
        actual = [causal[split][row["label"]]["absolute_fraction_of_writer"] for row in weight_rows]
        correlations[split] = {
            "value_spearman": spearman([row["value_read"] for row in weight_rows], actual),
            "routing_spearman": spearman([row["routing_read"] for row in weight_rows], actual),
        }

    top_value = [row["label"] for row in sorted(
        weight_rows, key=lambda row: (-row["value_read"], row["label"]))[:6]]
    top_causal_count = sum(all(causal[split][name]["absolute_fraction_of_writer"] >= 0.05
                               for split in ("A1_heldout", "A2")) for name in top_value)
    pred_a = bool(all_native_capable and reconstruction_error <= 1e-4
                  and writer_identity_error <= 1e-4 and explained >= 0.95)
    pred_b = writer_h1_rank <= 3
    pred_c = statistics.median(known_value_percentiles) >= 0.70
    value_mean = statistics.fmean(correlations[split]["value_spearman"]
                                  for split in ("A1_heldout", "A2"))
    routing_mean = statistics.fmean(correlations[split]["routing_spearman"]
                                    for split in ("A1_heldout", "A2"))
    pred_d = all(correlations[split]["value_spearman"] > 0
                 for split in ("A1_heldout", "A2")) and value_mean > routing_mean
    pred_e = top_causal_count >= 2
    pred_f = bool(forwards == FORWARDS and evaluations == EVALUATIONS and len(records) == RECORDS
                  and len({(record["split"], record["arm"], record["row_id"])
                           for record in records}) == RECORDS)
    predictions = {
        "pred_a_authority_capability_capture_and_split": pred_a,
        "pred_b_writer_positive_control": pred_b,
        "pred_c_known_readers_are_weight_enriched": pred_c,
        "pred_d_value_beats_routing_for_causal_prediction": pred_d,
        "pred_e_top_weight_readers_are_causal": pred_e,
        "pred_f_price_coverage_and_zero_causal_leakage": pred_f,
    }
    terminal = "invalid" if not pred_a or not pred_f else (
        "screen" if all(predictions.values()) else "null")
    result = {
        "schema": "temporal_auxiliary_writer_subspace_weight_reader_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"all_native_capable": all_native_capable,
            "attention_reconstruction_max_abs": reconstruction_error,
            "state_forward_identity_max_abs": writer_identity_error},
        "subspace": {"rank": rank, "explained_fit_energy": explained,
                     "singular_values": [float(value) for value in singular.detach().cpu()]},
        "weight_rows": weight_rows, "writer_capacity_ranking": writer_capacity,
        "writer_h1_rank": writer_h1_rank,
        "known_reader_value_percentiles": {label(head): by_label[label(head)]["value_read_percentile"]
                                            for head in sorted(KNOWN)},
        "top_six_value_readers": top_value, "top_six_causal_count": top_causal_count,
        "writer_summaries": writer_summaries, "correlations": correlations,
        "causal_head_summaries": causal, "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fitted_scalars": 0,
                  "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal,
        "reason": "weight_value_reads_predict_causal_readers" if terminal == "screen" else (
            "raw_weight_incidence_requires_contextualization" if terminal == "null"
            else "authority_capability_capture_split_coverage_or_price_invalid"),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "subspace",
          "writer_capacity_ranking", "known_reader_value_percentiles", "top_six_value_readers",
          "top_six_causal_count", "correlations", "predictions", "price", "terminal", "reason")},
          sort_keys=True))


if __name__ == "__main__":
    main()
