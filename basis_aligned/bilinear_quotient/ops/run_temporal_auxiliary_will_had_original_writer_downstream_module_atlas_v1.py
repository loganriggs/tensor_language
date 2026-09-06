#!/usr/bin/env python3
"""Writer-specific complete-module response atlas on original temporal cues."""

# BQGATE: EXPERIMENT pred_a_authority_capability_and_exact_capture pred_b_known_attention_readers_are_material pred_c_missing_singleton_module_exists pred_d_complete_ranked_atlas_reported pred_e_exact_zero_fit_price
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
import circuit_fast_screen_candidate_temporal_auxiliary as candidate
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_das_subspace as das
import residual_source_onset_eval as onset


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_original_writer_downstream_module_atlas_v1.json"
OPERATION = ROOT / "circuits/followups/temporal_auxiliary_will_had_original_joint_reader_pattern_value_factorial_v1_result.json"
WRITER = ROOT / "circuits/followups/temporal_auxiliary_will_had_block8h1_subject_source_groups_v1_result.json"
ATTENTION = ROOT / "ops/attention_source_destination_eval.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
BUILDER = ROOT / "ops/circuit_fast_screen_candidate_temporal_auxiliary.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_original_writer_downstream_module_atlas_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.original_writer_downstream_module_atlas_v1"
EXPECTED = {
    "prior": "24f5ca3278096810695c28c3932bc09d9dcc6a6d1e3412f7e52720cee207eb3c",
    "operation": "747ab4ee2a7df1d2540e5ce5a653711263b386e7754a12434ebc71d6f76b012b",
    "writer": "8865fe22a3c12e367709706ff0b941b3c2488d1d9608ce1921a4cfa73b22c6b9",
    "attention": "608ae6bf74af96663ec022b907c53d371670a36e5d7ec4fd1667b3c6add58dfd",
    "mediation": "05d07578129db0cb8bf5e46d08e6cd6e9c2bb9c3883675586c83c3efd24b8fda",
    "producer": "528e9f50152d6dc2b28d084cd0828de58de042b893703d0f322b4a2f22c4a0a7",
    "builder": "4d23947375504edaa51e4ef057ccd65f042c505728d1909bc06edf526633bc58",
}
SITES = tuple(f"{kind}:{layer:02d}" for layer in range(9, 18) for kind in ("attn", "mlp"))
WRITER_TARGET = {"A1": 0.23824887105543294, "A2": 0.14472067356008747}
MODEL_FORWARDS, EXAMPLE_EVALUATIONS, RECORDS = 42, 1344, 1152


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def pair_error(first, second):
    return max(abs(float(left) - float(right))
               for pair_left, pair_right in zip(first.answer_foil, second.answer_foil)
               for left, right in zip(pair_left, pair_right))


def validate_static():
    paths = {"prior": PRIOR, "operation": OPERATION, "writer": WRITER,
             "attention": ATTENTION, "mediation": MEDIATION,
             "producer": PRODUCER, "builder": BUILDER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or implementation hash changed")
    prior, operation, writer = [json.loads(path.read_text())
                                for path in (PRIOR, OPERATION, WRITER)]
    rows = [row for row in candidate.build_rows() if row["transform_id"] in {"A1", "A2"}]
    if (prior.get("candidate_id") != CANDIDATE_ID or operation.get("terminal") != "screen"
            or writer.get("terminal") != "screen" or len(SITES) != 18
            or {family: sum(row["transform_id"] == family for row in rows)
                for family in ("A1", "A2")} != {"A1": 32, "A2": 32}):
        raise ExperimentError("authority terminal, population, or site inventory changed")
    return rows


def dryrun_receipt():
    return {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
            "model_loaded": False, "queue_touched": False, "rows": 64,
            "sites": list(SITES), "model_forwards": MODEL_FORWARDS,
            "example_evaluations": EXAMPLE_EVALUATIONS, "records": RECORDS,
            "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}


def native_capable(output):
    return all(float(answer) - float(foil) > 0 for answer, foil in output.answer_foil)


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
    items, reconstruction_error, writer_identity_error = [], 0.0, 0.0
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
        forwards += 3
        evaluations += 3 * len(family_rows)
        reconstruction_error = max(reconstruction_error,
                                   float(writer_base["reconstruction_max_abs"]),
                                   float(writer_donor["reconstruction_max_abs"]))
        item = {"family": family, "rows": family_rows, "base_batch": base_batch,
                "base_output": base_output, "donor_output": donor_output,
                "writer_output": writer_output}
        items.append(item)

    writer_records, records = [], []
    for item in items:
        writer_records.extend(source_groups.recovery_records(
            item["rows"], item["base_output"], item["donor_output"],
            item["writer_output"], arm="writer_reference"))
        for site_id in SITES:
            output = backend.patched(
                item["base_batch"], site=kernel.SiteRef(site_id=site_id, evidence_kind="residual"),
                donor_cache=item["writer_output"].captured)
            forwards += 1
            evaluations += len(item["rows"])
            records.extend(source_groups.recovery_records(
                item["rows"], item["base_output"], item["donor_output"], output, arm=site_id))

    writer_summary = source_groups.summarize_by_family(writer_records)
    summaries = {site: source_groups.summarize_by_family(
        [record for record in records if record["arm"] == site]) for site in SITES}
    fractions = {site: {family: summaries[site][family]["mean_recovery"]
                        / writer_summary[family]["mean_recovery"]
                        for family in ("A1", "A2")} for site in SITES}
    rankings = {family: sorted(
        ({"site": site, "mean_recovery": summaries[site][family]["mean_recovery"],
          "fraction_of_writer": fractions[site][family]} for site in SITES),
        key=lambda row: row["mean_recovery"], reverse=True)
        for family in ("A1", "A2")}
    new_material = [site for site in SITES if site not in {"attn:09", "attn:11"}
                    and all(abs(fractions[site][family]) >= 0.05
                            for family in ("A1", "A2"))]
    attention_rankings = {family: [row["site"] for row in rankings[family]
                                   if row["site"].startswith("attn:")]
                          for family in ("A1", "A2")}
    capability = all(native_capable(item["base_output"]) and native_capable(item["donor_output"])
                     for item in items)
    pred_a = bool(capability and reconstruction_error <= 1e-4
                  and all(abs(writer_summary[family]["mean_recovery"] - WRITER_TARGET[family]) <= 0.03
                          for family in ("A1", "A2")))
    pred_b = all(summaries[site][family]["mean_recovery"] > 0
                 for site in ("attn:09", "attn:11") for family in ("A1", "A2")) and all(
        set(attention_rankings[family][:2]) == {"attn:09", "attn:11"}
        for family in ("A1", "A2"))
    pred_c = bool(new_material)
    pred_d = bool(len(summaries) == 18 and all(
        summaries[site][family]["count"] == 32 for site in SITES for family in ("A1", "A2")))
    pred_e = bool(forwards == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS
                  and len(records) == RECORDS
                  and len({(record["arm"], record["row_id"]) for record in records}) == RECORDS
                  and all(math.isfinite(float(record["recovery"])) for record in records))
    predictions = {
        "pred_a_authority_capability_and_exact_capture": pred_a,
        "pred_b_known_attention_readers_are_material": pred_b,
        "pred_c_missing_singleton_module_exists": pred_c,
        "pred_d_complete_ranked_atlas_reported": pred_d,
        "pred_e_exact_zero_fit_price": pred_e,
    }
    terminal = "invalid" if not pred_a or not pred_d or not pred_e else (
        "screen" if pred_c else "null")
    result = {
        "schema": "temporal_auxiliary_original_writer_downstream_module_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"all_native_capable": capability,
            "attention_reconstruction_max_abs": reconstruction_error,
            "writer_capture_identity_max_abs": writer_identity_error},
        "writer_summary": writer_summary, "summaries": summaries,
        "fraction_of_writer": fractions, "rankings": rankings,
        "new_material_sites": new_material, "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "records": len(records), "fitted_scalars": 0,
                  "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal,
        "reason": "new_singleton_module_for_head_split" if terminal == "screen" else (
            "distributed_remainder_requires_greedy_joint_modules" if terminal == "null"
            else "authority_capability_exactness_coverage_or_price_invalid"),
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
          "writer_summary", "rankings", "new_material_sites", "predictions",
          "price", "terminal", "reason")}, sort_keys=True))


if __name__ == "__main__":
    main()
