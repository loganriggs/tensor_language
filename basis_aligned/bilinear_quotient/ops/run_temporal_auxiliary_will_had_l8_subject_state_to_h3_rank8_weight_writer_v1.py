#!/usr/bin/env python3
"""Compile the direct L8H1-written subject state through H3 value weights into Q8."""

# BQGATE: EXPERIMENT pred_a_authority_exact_capture_weight_coverage_and_price pred_b_subject_value_weights_explain_h3_q8 pred_c_nonvalue_and_nonsubject_terms_are_secondary pred_d_composed_margin_prediction_is_accurate pred_e_direct_route_dominates_l9_branch
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as attention_eval
import circuit_candidate_temporal_auxiliary_fresh_cues_v10 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset
import run_temporal_auxiliary_will_had_h3_rank8_analytic_final_reader_v1 as reader_impl
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder
import run_temporal_auxiliary_will_had_l9_triple_to_h3_rank8_weight_writer_v1 as upstream

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_l8_subject_state_to_h3_rank8_weight_writer_v1.json"
SOURCE_AUTHORITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_writer_block11h3_source_response_v1_result.json"
L9_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_triple_to_h3_rank8_weight_writer_v1_result.json"
RANK8 = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_weight_rank8_v10_confirmation_v1_result.json"
READER = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_analytic_final_reader_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v10_capability_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v10.py"
FAMILY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1.py"
READER_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank8_analytic_final_reader_v1.py"
UPSTREAM_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_l9_triple_to_h3_rank8_weight_writer_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l8_subject_state_to_h3_rank8_weight_writer_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.l8_subject_state_to_h3_rank8_weight_writer_v1"
EXPECTED = {
    "prior": "2a20c2cf0b227b62ec5e751069da9f5ec864ed82ee8eb991c09f119700e84d8c",
    "source_authority": "1fd089aeb63e2ec7e1771d54170461dc18b3bd105e2b7fc08413d8456a515cf1",
    "l9_result": "3fd5f94fc60d592d1823413db2607127d888b34cf305815475c494d2f552bed2",
    "rank8": "0d19330da62f37f404570455ea4aeb198a7da787d244278935b6798dffc6e7db",
    "reader": "63fbb7f56cb7f63aeb009d46fb45721ed0dde6165706b5a9b241b7cd2815eb9e",
    "capability": "9923322703c72d50b2a1f06138ef35269db48e0a8a4ccb365f82df3519b113ad",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "builder": "e945e0b4679fa74d6cea23594ba553d9b2ffd3ac653c353d09d1873f2a3e4494",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "reader_runner": "50df4d2393606957516bcc8752cf4f575713ed8f6fc843c3864f585565a7947d",
    "upstream_runner": "9dd491e50cc2b46ad3fa4071ef1c53c0335a70dcb94709ec89889328c35ec4b2",
}
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 14, 500, 63


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    paths = {"prior": PRIOR, "source_authority": SOURCE_AUTHORITY, "l9_result": L9_RESULT,
        "rank8": RANK8, "reader": READER, "capability": CAPABILITY, "subspace": SUBSPACE,
        "builder": BUILDER, "family_runner": FAMILY_RUNNER, "reader_runner": READER_RUNNER,
        "upstream_runner": UPSTREAM_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("direct subject weight-writer authority changed")
    prior, source_authority, l9_result, rank8, reader, capability, subspace = [
        json.loads(path.read_text()) for path in
        (PRIOR, SOURCE_AUTHORITY, L9_RESULT, RANK8, READER, CAPABILITY, SUBSPACE)]
    if (prior.get("candidate_id") != CANDIDATE_ID or source_authority.get("terminal") != "screen"
            or l9_result.get("terminal") != "null" or rank8.get("terminal") != "paired_confirmation"
            or reader.get("terminal") != "screen" or capability.get("terminal") != "manifest"):
        raise RuntimeError("authority terminal changed")
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "rank": 8,
        "model_forwards_max": MAX_FORWARDS, "example_evaluations_max": MAX_EVALUATIONS,
        "records": RECORDS, "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    q = family[8]
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in range(12, 18))
    residual_modes = reader_impl.build_residual_modes(backend, q, gain)
    l9_norm = {panel: sum(math.sqrt(sum(value*value for value in record["exact_coordinates"]))
        for record in l9_result["records"] if record["panel"] == panel)
        / sum(record["panel"] == panel for record in l9_result["records"])
        for panel in ("A1", "A2")}
    records, forwards, evaluations = [], 0, 0
    reconstruction = identity = value_weight_error = subject_factor_closure = 0.0
    coordinate_metrics, residual_fraction, behavior_metrics, dominance = {}, {}, {}, {}
    for panel in ("A1", "A2"):
        panel_rows = [row for row in rows if row["transform_id"] == panel]
        base_batch = das._batch(backend, panel_rows, side="base")
        donor_batch = das._batch(backend, panel_rows, side="donor")
        base_output, base8 = attention_eval.capture_layer_attention(
            backend, base_batch, 8, call=lambda: backend.native(base_batch, capture=True))
        donor_output, donor8 = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        base11_output, base_h3, base_current = upstream.capture_h3_current(
            backend, base_batch, lambda: backend.native(base_batch, capture=False))
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, base8, donor8, destinations, ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            _writer_output, writer_h3, writer_current = upstream.capture_h3_current(
                backend, base_batch, lambda: backend.native(base_batch, capture=False))
        finally:
            handle.remove()
        exact_coordinates, value_coordinates, remainder_coordinates = [], [], []
        exact_norms, remainder_norms = [], []
        for index, (query, sources) in enumerate(zip(base_batch.semantic_positions, destinations)):
            query, sources = int(query), tuple(int(source) for source in sources)
            exact_head = (writer_h3["head_output"][index, query, 3].float()
                          - base_h3["head_output"][index, query, 3].float())
            p0 = base_h3["pattern"][index, 3, query, list(sources)].float()
            p1 = writer_h3["pattern"][index, 3, query, list(sources)].float()
            v0 = base_h3["value"][index, list(sources), 3].float()
            v1 = writer_h3["value"][index, list(sources), 3].float()
            value_head = (p0[:, None]*(v1-v0)).sum(0)
            pattern_head = ((p1-p0)[:, None]*v0).sum(0)
            interaction_head = ((p1-p0)[:, None]*(v1-v0)).sum(0)
            exact_c, value_c = exact_head@q, value_head@q
            remainder_c = exact_c-value_c
            exact_coordinates.append(exact_c); value_coordinates.append(value_c)
            remainder_coordinates.append(remainder_c)
            exact_norms.append(float(backend.torch.linalg.vector_norm(exact_c)))
            remainder_norms.append(float(backend.torch.linalg.vector_norm(remainder_c)))
            subject_factor_closure = max(subject_factor_closure,
                float((exact_head-value_head-pattern_head-interaction_head).abs().max()))
            normalized_delta = (backend.F.rms_norm(writer_current["current"][index, list(sources)],
                (backend.model.config.n_embd,)).float() - backend.F.rms_norm(
                base_current["current"][index, list(sources)], (backend.model.config.n_embd,)).float())
            weighted = backend.F.linear(normalized_delta,
                backend.model.transformer.h[11].attn.c_v.weight).view(len(sources), 9, 128)[:, 3]
            weighted = weighted*(1.0-backend.model.transformer.h[11].attn.lamb.float())
            value_weight_error = max(value_weight_error, float((weighted-(v1-v0)).abs().max()))
        exact_tensor = backend.torch.stack(exact_coordinates)
        exact_cache = upstream.endpoint_cache(backend, base_batch, base_output, exact_tensor, q, gain)
        exact_output = backend.patched(base_batch,
            site=kernel.SiteRef(site_id="resid:18", evidence_kind="residual"), donor_cache=exact_cache)
        forwards += 5; evaluations += 5*len(panel_rows)
        reconstruction = max(reconstruction, *(float(value["reconstruction_max_abs"])
            for value in (base8, donor8, base_h3, writer_h3)))
        identity = max(identity, upstream.pair_error(base_output, base11_output))
        coordinate_metrics[panel] = upstream.vector_metrics(
            [[float(v) for v in row] for row in exact_coordinates],
            [[float(v) for v in row] for row in value_coordinates])
        residual_fraction[panel] = sum(remainder_norms)/sum(exact_norms)
        dominance[panel] = {"complete_h3_q8_mean_norm": sum(exact_norms)/len(exact_norms),
            "l9_branch_mean_norm": l9_norm[panel],
            "complete_to_l9_ratio": (sum(exact_norms)/len(exact_norms))/l9_norm[panel]}
        exact_effects, predicted_effects = [], []
        for index, row in enumerate(panel_rows):
            state = base_output.captured[(row["row_id"], "resid:18")]
            analytic_reader, _epsilon = reader_impl.analytic_reader(
                backend, state, base_batch.answer_ids[index], base_batch.foil_ids[index], residual_modes)
            predicted_effect = float(value_coordinates[index]@analytic_reader)
            base_margin = float(base_output.answer_foil[index][0]-base_output.answer_foil[index][1])
            exact_margin = float(exact_output.answer_foil[index][0]-exact_output.answer_foil[index][1])
            exact_effect = exact_margin-base_margin
            exact_effects.append(exact_effect); predicted_effects.append(predicted_effect)
            records.append({"row_id": row["row_id"], "panel": panel,
                "exact_coordinates": [float(v) for v in exact_coordinates[index]],
                "subject_value_weight_coordinates": [float(v) for v in value_coordinates[index]],
                "coordinate_remainder": [float(v) for v in remainder_coordinates[index]],
                "exact_margin_effect": exact_effect, "predicted_margin_effect": predicted_effect})
        rmse = math.sqrt(sum((a-b)**2 for a, b in zip(exact_effects, predicted_effects))/len(exact_effects))
        exact_rms = math.sqrt(sum(a*a for a in exact_effects)/len(exact_effects))
        behavior_metrics[panel] = {"cosine": upstream.cosine(exact_effects, predicted_effects),
            "rmse": rmse, "exact_rms": exact_rms, "relative_rmse": rmse/exact_rms}
    pred_a = bool(reconstruction <= 5e-4 and identity <= 1e-4 and value_weight_error <= 5e-4
        and len(records) == RECORDS and forwards <= MAX_FORWARDS and evaluations <= MAX_EVALUATIONS
        and all(math.isfinite(record["predicted_margin_effect"]) for record in records))
    pred_b = all(coordinate_metrics[panel]["cosine"] >= .95
        and coordinate_metrics[panel]["relative_rmse"] <= .20 for panel in ("A1", "A2"))
    pred_c = all(residual_fraction[panel] <= .20 for panel in ("A1", "A2"))
    pred_d = all(behavior_metrics[panel]["cosine"] >= .95
        and behavior_metrics[panel]["relative_rmse"] <= .25 for panel in ("A1", "A2"))
    pred_e = all(dominance[panel]["complete_to_l9_ratio"] >= 4.0 for panel in ("A1", "A2"))
    predictions = {"pred_a_authority_exact_capture_weight_coverage_and_price": pred_a,
        "pred_b_subject_value_weights_explain_h3_q8": pred_b,
        "pred_c_nonvalue_and_nonsubject_terms_are_secondary": pred_c,
        "pred_d_composed_margin_prediction_is_accurate": pred_d,
        "pred_e_direct_route_dominates_l9_branch": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "temporal_auxiliary_l8_subject_state_to_h3_rank8_weight_writer_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"attention_reconstruction_max_abs": reconstruction,
            "capture_identity_max_abs": identity, "value_weight_reconstruction_max_abs": value_weight_error,
            "subject_factor_head_closure_max_abs": subject_factor_closure},
        "subject_value_coordinate_metrics": coordinate_metrics,
        "non_subject_value_coordinate_norm_fraction": residual_fraction,
        "composed_margin_prediction_metrics": behavior_metrics,
        "direct_vs_l9_branch": dominance, "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "records": len(records), "fit_updates": 0, "transformer_backwards": 0,
            "model_updates": 0}, "records": records}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
        "subject_value_coordinate_metrics", "non_subject_value_coordinate_norm_fraction",
        "composed_margin_prediction_metrics", "direct_vs_l9_branch", "predictions",
        "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
