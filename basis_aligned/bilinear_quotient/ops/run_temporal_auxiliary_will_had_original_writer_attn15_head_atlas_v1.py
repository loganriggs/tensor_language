#!/usr/bin/env python3
"""All-head split of the original temporal writer's attention15 response."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_writer_recurrence pred_b_same_head_is_largest pred_c_single_head_concentrates_attn15 pred_d_all_heads_reported pred_e_exact_zero_fit_price
from __future__ import annotations

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
import circuit_das_subspace as das
import circuit_fast_screen_candidate_temporal_auxiliary as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_original_writer_attn15_head_atlas_v1.json"
ATLAS = ROOT / "circuits/followups/temporal_auxiliary_will_had_original_writer_downstream_module_atlas_v1_result.json"
ATLAS_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_original_writer_downstream_module_atlas_v1.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_original_writer_attn15_head_atlas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.original_writer_attn15_head_atlas_v1"
EXPECTED = {
    "prior": "a60a6e9aa1e53e4d69f2815b525bcda8b3551de1c8b7850e767513cd1adc6b35",
    "atlas": "f8517f43f41444b966b95ff0d8da9449f25bd95d32b726c1082066605cfd076a",
    "atlas_runner": "a16fa52f76e0b7d947ed174b18898bced2ca242b735300ef592f2e3fc4bfeb4e",
    "producer": "528e9f50152d6dc2b28d084cd0828de58de042b893703d0f322b4a2f22c4a0a7",
}
ARMS = ("complete_attn15",) + tuple(f"head:{head:02d}" for head in range(9))
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 26, 832, 640


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "atlas": ATLAS, "atlas_runner": ATLAS_RUNNER,
             "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, atlas, or implementation hash changed")
    prior, atlas = [json.loads(path.read_text()) for path in (PRIOR, ATLAS)]
    rows = [row for row in candidate.build_rows() if row["transform_id"] in {"A1", "A2"}]
    if (prior.get("candidate_id") != CANDIDATE_ID or atlas.get("terminal") != "screen"
            or "attn:15" not in atlas.get("new_material_sites", []) or len(ARMS) != 10
            or len(rows) != 64):
        raise ExperimentError("atlas terminal, selected site, or population changed")
    targets = {family: {
        "writer": atlas["writer_summary"][family]["mean_recovery"],
        "complete": atlas["summaries"]["attn:15"][family]["mean_recovery"]}
        for family in ("A1", "A2")}
    return rows, targets


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "rows": 64,
            "arms": list(ARMS), "model_forwards": MODEL_FORWARDS,
            "example_evaluations": EXAMPLE_EVALUATIONS, "records": RECORDS,
            "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}


def capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


def main():
    rows, targets = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    items, reconstruction_error = [], 0.0
    forwards = evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, writer_base = attention_eval.capture_layer_attention(
            backend, base_batch, 8, call=lambda: backend.native(base_batch, capture=True))
        donor_output, writer_donor = attention_eval.capture_layer_attention(
            backend, donor_batch, 8, call=lambda: backend.native(donor_batch, capture=True))
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, writer_base, writer_donor,
            destinations, ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            writer_output = backend.native(base_batch, capture=True)
        finally:
            handle.remove()
        reconstruction_error = max(reconstruction_error,
                                   float(writer_base["reconstruction_max_abs"]),
                                   float(writer_donor["reconstruction_max_abs"]))
        items.append({"family": family, "rows": family_rows, "base_batch": base_batch,
                      "base_output": base_output, "donor_output": donor_output,
                      "writer_output": writer_output})
        forwards += 3
        evaluations += 3 * len(family_rows)

    writer_records, records = [], []
    for item in items:
        writer_records.extend(source_groups.recovery_records(
            item["rows"], item["base_output"], item["donor_output"],
            item["writer_output"], arm="writer_reference"))
        outputs = {"complete_attn15": backend.patched(
            item["base_batch"], site=kernel.SiteRef(site_id="attn:15", evidence_kind="residual"),
            donor_cache=item["writer_output"].captured)}
        outputs.update({f"head:{head:02d}": backend.patched_heads(
            item["base_batch"], layer=15, heads=(head,),
            donor_cache=item["writer_output"].captured) for head in range(9)})
        forwards += len(outputs)
        evaluations += len(outputs) * len(item["rows"])
        for arm, output in outputs.items():
            records.extend(source_groups.recovery_records(
                item["rows"], item["base_output"], item["donor_output"], output, arm=arm))

    writer_summary = source_groups.summarize_by_family(writer_records)
    summaries = {arm: source_groups.summarize_by_family(
        [record for record in records if record["arm"] == arm]) for arm in ARMS}
    fractions = {arm: {family: summaries[arm][family]["mean_recovery"]
                       / summaries["complete_attn15"][family]["mean_recovery"]
                       for family in ("A1", "A2")} for arm in ARMS[1:]}
    rankings = {family: sorted(
        ({"head": arm, "mean_recovery": summaries[arm][family]["mean_recovery"],
          "fraction_of_complete": fractions[arm][family]} for arm in ARMS[1:]),
        key=lambda row: row["mean_recovery"], reverse=True)
        for family in ("A1", "A2")}
    top = {family: rankings[family][0]["head"] for family in ("A1", "A2")}
    all_capable = all(capable(item["base_output"]) and capable(item["donor_output"])
                      for item in items)
    pred_a = bool(all_capable and reconstruction_error <= 1e-4 and all(
        abs(writer_summary[family]["mean_recovery"] - targets[family]["writer"]) <= 1e-5
        and abs(summaries["complete_attn15"][family]["mean_recovery"]
                - targets[family]["complete"]) <= 1e-5 for family in ("A1", "A2")))
    pred_b = top["A1"] == top["A2"]
    selected = top["A1"] if pred_b else None
    pred_c = bool(selected is not None and all(
        fractions[selected][family] >= 0.60
        and summaries[selected][family]["direction_fraction"] >= 0.75
        for family in ("A1", "A2")))
    pred_d = bool(len(summaries) == 10 and all(
        summaries[arm][family]["count"] == 32 for arm in ARMS for family in ("A1", "A2")))
    pred_e = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {
        "pred_a_authority_capability_and_writer_recurrence": pred_a,
        "pred_b_same_head_is_largest": pred_b,
        "pred_c_single_head_concentrates_attn15": pred_c,
        "pred_d_all_heads_reported": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "invalid" if not pred_a or not pred_d or not pred_e else (
        "screen" if pred_b and pred_c else "null")
    result = {
        "schema": "temporal_auxiliary_original_writer_attn15_head_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"all_native_capable": all_capable,
            "attention_reconstruction_max_abs": reconstruction_error},
        "writer_summary": writer_summary, "summaries": summaries,
        "fraction_of_complete_attn15": fractions, "rankings": rankings,
        "selected_head": selected, "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fitted_scalars": 0,
                  "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal,
        "reason": "concentrated_attn15_head" if terminal == "screen" else (
            "distributed_attn15_head_response" if terminal == "null"
            else "authority_capability_recurrence_coverage_or_price_invalid"),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
          "rankings", "selected_head", "predictions", "price", "terminal", "reason")},
          sort_keys=True))


if __name__ == "__main__":
    main()
