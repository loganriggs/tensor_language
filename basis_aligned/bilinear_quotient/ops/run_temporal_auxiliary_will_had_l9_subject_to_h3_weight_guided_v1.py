#!/usr/bin/env python3
"""Causally test the weight-predicted L9H1/H4 subject write into H3."""

# BQGATE: EXPERIMENT pred_a_authority_capability_capture_closure_and_price pred_b_writer_and_full_l9_subject_response_are_material pred_c_weight_selected_pair_dominates_h3_response pred_d_unselected_complement_is_secondary pred_e_both_weight_selected_heads_rank_near_top
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as attention_eval
import attention_source_group_eval as source_groups
import circuit_candidate_temporal_auxiliary_fresh_cues_v4 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_l9_subject_to_h3_weight_guided_v1.json"
WEIGHT_AUTH = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_multicue_weight_interface_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v4_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v4.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
SOURCE = ROOT / "ops/attention_source_group_eval.py"
ONSET = ROOT / "ops/residual_source_onset_eval.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_weight_guided_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.l9_subject_to_h3_weight_guided_v1"
EXPECTED = {
    "prior": "686059383b643fdde3888d296927b893e2c6c2d4b050e055e9354038563bbd60",
    "weight_auth": "5bf804ee1e61f918edceb0dc9e31ac68fa157de384a943391c4ca4eeb672246a",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "capability": "63b69e3bc57a0a8a9afcffa252737614f9a8a41b6732b9fff655d9da128ef8b2",
    "builder": "31e40a5e8a8b285ce7afdb6327276c0aa28b4759083586d0310b0857c8b86764",
    "mediation": "05d07578129db0cb8bf5e46d08e6cd6e9c2bb9c3883675586c83c3efd24b8fda",
    "attention": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
    "source": "6ecf5f40b92f94cb32bccf1a703e527a3d468281936e63d4c7e91e8af66b4348",
    "onset": "c276450cc9ec7c2b0a05e2be0e88bac3df9af7003e370b99b66552083c4f4b45",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
HEADS = tuple(range(9))
ARM_HEADS = {**{f"head:{head}": (head,) for head in HEADS},
             "weight_pair_h1_h4": (1, 4),
             "pair_complement": (0, 2, 3, 5, 6, 7, 8),
             "all_heads": HEADS}
FORWARDS, EVALUATIONS, RECORDS = 36, 1152, 768


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def pair_error(first, second):
    return max(abs(float(a) - float(b)) for left, right in zip(first.answer_foil, second.answer_foil)
               for a, b in zip(left, right))


def projected_norms(backend, capture, base_capture, batch, q):
    values = []
    for index, query in enumerate(batch.semantic_positions):
        delta = (capture["head_output"][index, query, 3].float()
                 - base_capture["head_output"][index, query, 3].float())
        values.append(float(backend.torch.linalg.vector_norm(delta @ q)))
    return values


def main():
    paths = {"prior": PRIOR, "weight_auth": WEIGHT_AUTH, "subspace": SUBSPACE,
             "capability": CAPABILITY, "builder": BUILDER, "mediation": MEDIATION,
             "attention": ATTENTION, "source": SOURCE, "onset": ONSET, "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("authority or implementation hash changed")
    prior, weights, subspace, capability = [json.loads(path.read_text()) for path in
        (PRIOR, WEIGHT_AUTH, SUBSPACE, CAPABILITY)]
    if (prior.get("candidate_id") != CANDIDATE_ID or weights.get("terminal") != "partial"
            or capability.get("terminal") != "manifest"
            or not all(capability.get("predictions", {}).values())):
        raise RuntimeError("authority terminal changed")
    rows = [row for row in candidate.build_rows() if row["transform_id"] in {"A1", "A2"}]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "heads": list(HEADS),
              "arms": list(ARM_HEADS), "model_forwards": FORWARDS,
              "example_evaluations": EVALUATIONS, "records": RECORDS,
              "fit_updates": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    q = backend.torch.linalg.qr(backend.torch.tensor(
        subspace["axis_artifacts"]["two_task_dim_union_rank2"],
        device=backend.device).float(), mode="reduced").Q
    items, reconstruction_error, identity_error = [], 0.0, 0.0
    forwards = evaluations = 0
    for panel in ("A1", "A2"):
        panel_rows = [row for row in rows if row["transform_id"] == panel]
        base_batch = das._batch(backend, panel_rows, side="base")
        donor_batch = das._batch(backend, panel_rows, side="donor")
        base_output, base8 = attention_eval.capture_layer_attention(backend, base_batch, 8)
        donor_output, donor8 = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        base9_output, base9 = attention_eval.capture_layer_attention(backend, base_batch, 9)
        base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, base8, donor8, destinations,
            ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            writer9_output, writer9 = attention_eval.capture_layer_attention(backend, base_batch, 9)
        finally:
            handle.remove()
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            writer11_output, writer11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        finally:
            handle.remove()
        forwards += 6; evaluations += 6 * len(panel_rows)
        identity_error = max(identity_error, pair_error(base_output, base9_output),
                             pair_error(base_output, base11_output),
                             pair_error(writer9_output, writer11_output))
        reconstruction_error = max(reconstruction_error, *(float(value["reconstruction_max_abs"])
            for value in (base8, donor8, base9, base11, writer9, writer11)))
        items.append({"panel": panel, "rows": panel_rows, "base_batch": base_batch,
            "base_output": base_output, "donor_output": donor_output,
            "base9": base9, "writer9": writer9, "base11": base11,
            "writer_output": writer9_output, "writer11": writer11,
            "destinations": destinations})

    records, projected = [], {panel: {} for panel in ("A1", "A2")}
    writer_projected = {}
    for item in items:
        writer_projected[item["panel"]] = projected_norms(
            backend, item["writer11"], item["base11"], item["base_batch"], q)
        for arm, selected in ARM_HEADS.items():
            output, h3 = attention_eval.capture_layer_attention(
                backend, item["base_batch"], 11,
                call=lambda selected=selected, item=item: attention_eval.intervene_head_output_delta(
                    backend, item["base_batch"], item["base9"], item["writer9"],
                    layer=9, selected_heads=selected, positions_by_row=item["destinations"]))
            forwards += 1; evaluations += len(item["rows"])
            reconstruction_error = max(reconstruction_error, float(h3["reconstruction_max_abs"]))
            projected[item["panel"]][arm] = projected_norms(
                backend, h3, item["base11"], item["base_batch"], q)
            records.extend(dict(record, panel=item["panel"]) for record in
                source_groups.recovery_records(item["rows"], item["base_output"],
                    item["donor_output"], output, arm=arm))

    summaries = {panel: {arm: source_groups.summarize([record for record in records
        if record["panel"] == panel and record["arm"] == arm]) for arm in ARM_HEADS}
        for panel in ("A1", "A2")}
    writer_summaries = {item["panel"]: source_groups.summarize(source_groups.recovery_records(
        item["rows"], item["base_output"], item["donor_output"], item["writer_output"],
        arm="writer")) for item in items}
    projected_means = {panel: {arm: sum(values) / len(values)
        for arm, values in projected[panel].items()} for panel in ("A1", "A2")}
    projected_fraction = {panel: {arm: value / projected_means[panel]["all_heads"]
        if projected_means[panel]["all_heads"] > 1e-12 else None
        for arm, value in projected_means[panel].items()} for panel in ("A1", "A2")}
    singleton_rankings = {panel: sorted(
        ({"head": head, "projected_h3_norm": projected_means[panel][f"head:{head}"]}
         for head in HEADS), key=lambda row: (-row["projected_h3_norm"], row["head"]))
        for panel in ("A1", "A2")}
    pred_a = bool(reconstruction_error <= 5e-4 and identity_error <= 1e-4
                  and forwards == FORWARDS and evaluations == EVALUATIONS
                  and len(records) == RECORDS and all(math.isfinite(r["recovery"]) for r in records))
    pred_b = all(writer_summaries[p]["direction_fraction"] == 1.0
                 and abs(summaries[p]["all_heads"]["mean_recovery"])
                 >= 0.03 * abs(writer_summaries[p]["mean_recovery"]) for p in ("A1", "A2"))
    pred_c = all(projected_fraction[p]["weight_pair_h1_h4"] >= 0.60 for p in ("A1", "A2"))
    pred_d = all(projected_fraction[p]["pair_complement"] <= 0.40 for p in ("A1", "A2"))
    pred_e = all({1, 4}.issubset({row["head"] for row in singleton_rankings[p][:4]})
                 for p in ("A1", "A2"))
    predictions = {"pred_a_authority_capability_capture_closure_and_price": pred_a,
        "pred_b_writer_and_full_l9_subject_response_are_material": pred_b,
        "pred_c_weight_selected_pair_dominates_h3_response": pred_c,
        "pred_d_unselected_complement_is_secondary": pred_d,
        "pred_e_both_weight_selected_heads_rank_near_top": pred_e}
    terminal = ("invalid" if not pred_a else "screen" if all(predictions.values())
                else "distributed" if pred_b else "null")
    result = {"schema": "temporal_auxiliary_l9_subject_to_h3_weight_guided_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"reconstruction_max_abs": reconstruction_error,
            "capture_identity_max_abs": identity_error},
        "writer_summaries": writer_summaries, "behavior_summaries": summaries,
        "projected_h3_norm_means": projected_means,
        "projected_fraction_of_all_heads": projected_fraction,
        "singleton_rankings": singleton_rankings,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fit_updates": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
          "writer_summaries", "behavior_summaries", "projected_fraction_of_all_heads",
          "singleton_rankings", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
