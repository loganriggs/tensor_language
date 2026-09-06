#!/usr/bin/env python3
"""Resolve the fresh writer-induced block11H3 response by semantic source."""

# BQGATE: EXPERIMENT pred_a_authority_capability_exact_source_response_instrument pred_b_h3_complete_recurrence pred_c_final_query_sources_cover_h3 pred_d_subject_terms_dominate pred_e_exact_zero_fit_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_path_mediation_eval as mediation
import attention_response_source_eval as response_source
import attention_source_destination_eval as attention_eval
import attention_source_group_eval as source_score
import circuit_candidate_temporal_auxiliary_fresh_cues_v1 as candidate
import circuit_das_subspace as das
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset
import written_state_block_factorial_eval as crossing


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_fresh_writer_block11h3_source_response_v1.json"
H3_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block11_h3_response_v1_result.json"
OFFSET = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_subject_consumer_offset_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v1.py"
ATTENTION_EVAL = ROOT / "ops/attention_source_destination_eval.py"
RESPONSE_EVAL = ROOT / "ops/attention_response_source_eval.py"
MEDIATION = ROOT / "ops/attention_path_mediation_eval.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block11h3_source_response_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.fresh_writer_block11h3_source_response_v1"
EXPECTED = {
    "prior": "9f16c79f6cd9aa0431f5e718d4604008c37a161b8325ddc6a420f531df145a8f",
    "h3_result": "ee95aef443d63ce936f011ce2d551b8a0b220aa701507ecad30a72383475405a",
    "offset": "864f40e041cd4028c242fb96c816347875c59511d04e908372fd533b8c58c7ca",
    "builder": "5a753c56b278024431d209d0e8c4ed353d8f2086206847a148591c11181e56c9",
    "attention_eval": "806bd970b773c839cf4eb8d74c1fdbf4102fda32d2188d22daa8a1d5624c2bdf",
    "response_eval": "7bda6cf00c24efc3670704e8e01d9eabd954913db867abfa2fdf290052910848",
    "mediation": "9180ef34ec376729103e200ae2b2a2ce93d5f8ed0b293b0b1b459a55d71a079d",
}
ROWS_SHA256 = "c35d7f2159c47620b6f36f12c4f6c41c1630afae61a7f101ab9ff74e4e7e5e0a"
H3_TARGET = {"A1": 0.08934141104223325, "A2": 0.05778063176379767}
GROUPS = source_score.GROUP_ORDER
ARMS = ("h3_complete_all_destinations", "all_query_sources") + GROUPS + ("non_subject",)
MODEL_FORWARDS = 26
EXAMPLE_EVALUATIONS = 832
INTERVENTION_RECORDS = 512


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "h3_result": H3_RESULT, "offset": OFFSET, "builder": BUILDER, "attention_eval": ATTENTION_EVAL, "response_eval": RESPONSE_EVAL, "mediation": MEDIATION}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, authority, or evaluator hash changed")
    prior, h3, offset = map(lambda path: json.loads(path.read_text()), (PRIOR, H3_RESULT, OFFSET))
    rows_all = candidate.build_rows()
    rows = [row for row in rows_all if row["transform_id"] in {"A1", "A2"}]
    if (
        prior.get("candidate_id") != CANDIDATE_ID
        or candidate.validate_rows(rows_all) != ROWS_SHA256
        or h3.get("terminal") != "screen"
        or offset.get("latest_material_boundary_at_0p015_and_0p75") != 11
        or len(rows) != 64 or len(ARMS) != 8
    ):
        raise ExperimentError("population or parent H3 authority changed")
    return rows


def dryrun_receipt():
    return {
        "candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": 64,
        "arms": list(ARMS), "model_forwards": MODEL_FORWARDS,
        "example_evaluations": EXAMPLE_EVALUATIONS,
        "intervention_records": INTERVENTION_RECORDS, "fitted_scalars": 0,
        "grid_evaluations": 0, "root_evaluations": 0,
        "transformer_backwards": 0, "model_updates": 0,
    }


def main():
    rows = validate_static()
    dryrun = dryrun_receipt()
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = onset.ResidualGroupBackend.load("cuda")
    if backend.model.config.n_head != 9:
        raise ExperimentError("frozen head inventory changed")
    records, reconstruction_error = [], 0.0
    native_identity_error = 0.0
    forward_calls = evaluations = 0
    for family in ("A1", "A2"):
        family_rows = [row for row in rows if row["transform_id"] == family]
        base_batch = das._batch(backend, family_rows, side="base")
        donor_batch = das._batch(backend, family_rows, side="donor")
        base_output, writer_base = attention_eval.capture_layer_attention(backend, base_batch, 8)
        donor_output, writer_donor = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        _writer_output, writer_states = mediation.capture_source_written_states(
            backend, base_batch, donor_batch, writer_base, writer_donor, destinations,
            maximum_boundary=12,
        )
        base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        _direct_output, changed11 = attention_eval.capture_layer_attention(
            backend, base_batch, 11,
            call=lambda: backend.forward_states(
                base_batch, maximum_boundary=12, donor_batch=donor_batch,
                donor_states=writer_states, boundary=11, group_name="subject_onset",
            )[0],
        )
        forward_calls += 5
        evaluations += 5 * len(family_rows)
        reconstruction_error = max(
            reconstruction_error,
            *(float(capture["reconstruction_max_abs"]) for capture in (writer_base, writer_donor, base11, changed11)),
        )
        native_identity_error = max(native_identity_error, crossing.pair_error(base_output, base11_output))
        outputs = {
            "h3_complete_all_destinations": attention_eval.intervene_head_output_delta(
                backend, base_batch, base11, changed11, layer=11, selected_heads=(3,)
            )
        }
        selections = {"all_query_sources": GROUPS, "non_subject": tuple(name for name in GROUPS if name != "subject_onset")}
        selections.update({name: (name,) for name in GROUPS})
        for arm, names in selections.items():
            outputs[arm] = response_source.intervene_response_groups(
                backend, base_batch, donor_batch, base11, changed11, names,
                layer=11, selected_heads=(3,),
            )
        forward_calls += len(outputs)
        evaluations += len(outputs) * len(family_rows)
        for arm in ARMS:
            records.extend(source_score.recovery_records(
                family_rows, base_output, donor_output, outputs[arm], arm=arm
            ))
    summaries = {arm: source_score.summarize_by_family([r for r in records if r["arm"] == arm]) for arm in ARMS}
    all_source_retained = {family: summaries["all_query_sources"][family]["mean_recovery"] / summaries["h3_complete_all_destinations"][family]["mean_recovery"] for family in ("A1", "A2")}
    subject_retained = {family: summaries["subject_onset"][family]["mean_recovery"] / summaries["all_query_sources"][family]["mean_recovery"] for family in ("A1", "A2")}
    pred_a = reconstruction_error <= 1e-4 and native_identity_error <= 1e-4 and source_score.verify_contract()
    pred_b = all(abs(summaries["h3_complete_all_destinations"][f]["mean_recovery"] - H3_TARGET[f]) <= 1e-6 and summaries["h3_complete_all_destinations"][f]["direction_fraction"] == 1.0 for f in ("A1", "A2"))
    pred_c = all(all_source_retained[f] >= 0.9 and summaries["all_query_sources"][f]["direction_fraction"] >= 0.75 for f in ("A1", "A2"))
    pred_d = all(subject_retained[f] >= 0.9 and summaries["subject_onset"][f]["direction_fraction"] >= 0.75 and abs(summaries["non_subject"][f]["mean_recovery"]) <= 0.1 * abs(summaries["all_query_sources"][f]["mean_recovery"]) for f in ("A1", "A2"))
    pred_e = forward_calls == MODEL_FORWARDS and evaluations == EXAMPLE_EVALUATIONS and len(records) == INTERVENTION_RECORDS and len({(r["arm"], r["row_id"]) for r in records}) == INTERVENTION_RECORDS and all(math.isfinite(float(r["recovery"])) for r in records)
    predictions = {"pred_a_authority_capability_exact_source_response_instrument": bool(pred_a), "pred_b_h3_complete_recurrence": pred_b, "pred_c_final_query_sources_cover_h3": pred_c, "pred_d_subject_terms_dominate": pred_d, "pred_e_exact_zero_fit_price": pred_e}
    terminal = "screen" if all(predictions.values()) else ("null" if pred_a and pred_b and pred_c and pred_e else "invalid")
    reason = {"screen": "subject_terms_identify_fresh_writer_block11h3_response", "null": "valid_h3_source_partition_but_subject_dominance_failed", "invalid": "authority_exactness_recurrence_coverage_or_price_invalid"}[terminal]
    result = {
        "schema": "temporal_auxiliary_will_had_fresh_writer_block11h3_source_response_result_v1", "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "rows_sha256": ROWS_SHA256, "dryrun": dryrun, "predictions": predictions,
        "instrument": {"attention_reconstruction_max_abs": reconstruction_error, "native_identity_max_abs": native_identity_error, "model_head_count": 9},
        "summaries": summaries, "all_query_sources_retained_fraction": all_source_retained, "subject_onset_retained_fraction": subject_retained,
        "price": {"model_forwards": forward_calls, "example_evaluations": evaluations, "intervention_records": len(records), "fitted_scalars": 0, "grid_evaluations": 0, "root_evaluations": 0, "transformer_backwards": 0, "model_updates": 0},
        "records": records, "terminal": terminal, "reason": reason,
        "next_action": "test joint block9 H1/H4 plus block11H3 mediation completeness" if terminal == "screen" else "retain H3 and pursue the empirically dominant source response",
    }
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID, "terminal": terminal, "reason": reason, "predictions": predictions, "summaries": summaries, "all_source_retained": all_source_retained, "subject_retained": subject_retained, "result": str(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
