#!/usr/bin/env python3
"""Compile causal L9H1/H4/H7 responses through H3 value weights into Q8."""

# BQGATE: EXPERIMENT pred_a_authority_exact_capture_clamp_coverage_and_price pred_b_l9_triple_removes_material_rank8_coordinates pred_c_value_weight_writer_explains_rank8_coordinate_removal pred_d_pattern_terms_are_secondary pred_e_composed_margin_prediction_is_accurate
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

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_l9_triple_to_h3_rank8_weight_writer_v1.json"
L9_AUTHORITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_subject_to_h3_triple_v8_necessity_v1_result.json"
RANK8 = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_weight_rank8_v10_confirmation_v1_result.json"
READER = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_analytic_final_reader_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v10_capability_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v10.py"
FAMILY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1.py"
READER_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank8_analytic_final_reader_v1.py"
L9_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_l9_subject_to_h3_greedy_necessity_v1.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_l9_triple_to_h3_rank8_weight_writer_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.l9_triple_to_h3_rank8_weight_writer_v1"
EXPECTED = {
    "prior": "bfdc21c173ed161a56b127e95cd82bc39d190a45d4459037b02ea049e6ff65ef",
    "l9_authority": "dd64838525519a3addd679d1dbbcd6c7e18465b16a1faf06bea23940ef6d51f9",
    "rank8": "0d19330da62f37f404570455ea4aeb198a7da787d244278935b6798dffc6e7db",
    "reader": "63fbb7f56cb7f63aeb009d46fb45721ed0dde6165706b5a9b241b7cd2815eb9e",
    "capability": "9923322703c72d50b2a1f06138ef35269db48e0a8a4ccb365f82df3519b113ad",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "builder": "e945e0b4679fa74d6cea23594ba553d9b2ffd3ac653c353d09d1873f2a3e4494",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "reader_runner": "50df4d2393606957516bcc8752cf4f575713ed8f6fc843c3864f585565a7947d",
    "l9_runner": "1d4be0b54e81e9f272622739aee4d623011cedafdef290e53442b4f9462893ca",
}
TRIPLE = (1, 4, 7)
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 20, 700, 63


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def pair_error(first, second):
    return max(abs(float(a)-float(b)) for left, right in zip(first.answer_foil, second.answer_foil)
               for a, b in zip(left, right))


def cosine(first, second):
    numerator = sum(a*b for a, b in zip(first, second))
    denominator = math.sqrt(sum(a*a for a in first) * sum(b*b for b in second))
    return numerator/denominator if denominator else float("nan")


def vector_metrics(exact_rows, predicted_rows):
    exact = [value for row in exact_rows for value in row]
    predicted = [value for row in predicted_rows for value in row]
    rmse = math.sqrt(sum((a-b)**2 for a, b in zip(exact, predicted)) / len(exact))
    exact_rms = math.sqrt(sum(a*a for a in exact) / len(exact))
    return {"cosine": cosine(exact, predicted), "rmse": rmse,
            "exact_rms": exact_rms, "relative_rmse": rmse/exact_rms}


def capture_h3_current(backend, batch, call):
    captured = {}
    def hook(_module, arguments):
        captured["current"] = arguments[0].detach().clone()
        captured["initial_value"] = (arguments[1].detach().clone()
            if len(arguments) > 1 and arguments[1] is not None else None)
    handle = backend.model.transformer.h[11].attn.register_forward_pre_hook(hook)
    try:
        output, h3 = attention_eval.capture_layer_attention(backend, batch, 11, call=call)
    finally:
        handle.remove()
    return output, h3, captured


def run_l9_clamped(backend, batch, writer_hook, base9, destinations, *, enable_writer):
    handles, reconstructed = [], {}
    if enable_writer:
        handles.append(backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook))
    attention = backend.model.transformer.h[9].attn
    head_count = int(backend.model.config.n_head)
    head_width = int(backend.model.config.n_embd // head_count)
    def capture(_module, arguments):
        current = arguments[0]
        initial = arguments[1] if len(arguments) > 1 else None
        _pattern, _value, head_output = attention_eval._attention_terms(
            backend, attention, current, initial)
        reconstructed["head_output"] = head_output
    def clamp(_module, arguments):
        flattened = arguments[0]
        native = flattened.view(len(batch.row_ids), flattened.shape[1], head_count, head_width)
        reconstructed["error"] = float((reconstructed["head_output"].float()-native.float()).abs().max())
        changed = native.clone()
        for index, positions in enumerate(destinations):
            for position in positions:
                for head in TRIPLE:
                    changed[index, position, head] = base9["head_output"][index, position, head].to(changed)
        return (changed.reshape_as(flattened),) + tuple(arguments[1:])
    handles.extend((attention.register_forward_pre_hook(capture),
                    attention.c_proj.register_forward_pre_hook(clamp)))
    try:
        output, h3, current = capture_h3_current(
            backend, batch, lambda: backend.native(batch, capture=False))
    finally:
        for handle in handles:
            handle.remove()
    return output, h3, current, float(reconstructed.get("error", math.inf))


def endpoint_cache(backend, batch, base_output, coordinates, q, gain):
    weight = backend.model.transformer.h[11].attn.c_proj.weight
    head_count = int(backend.model.config.n_head)
    head_width = int(backend.model.config.n_embd // head_count)
    cache = {}
    for index, row_id in enumerate(batch.row_ids):
        head_delta = q @ coordinates[index]
        flattened = backend.torch.zeros(head_count*head_width, device=weight.device, dtype=weight.dtype)
        flattened[3*head_width:4*head_width] = head_delta.to(flattened)
        write = backend.F.linear(flattened, weight).float() * gain
        cache[(row_id, "resid:18")] = base_output.captured[(row_id, "resid:18")] + write.to(
            base_output.captured[(row_id, "resid:18")])
    return cache


def main():
    paths = {"prior": PRIOR, "l9_authority": L9_AUTHORITY, "rank8": RANK8,
        "reader": READER, "capability": CAPABILITY, "subspace": SUBSPACE,
        "builder": BUILDER, "family_runner": FAMILY_RUNNER,
        "reader_runner": READER_RUNNER, "l9_runner": L9_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("upstream Q8 writer authority changed")
    prior, l9_authority, rank8, reader, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, L9_AUTHORITY, RANK8, READER, CAPABILITY, SUBSPACE)]
    if (prior.get("candidate_id") != CANDIDATE_ID
            or l9_authority.get("terminal") != "paired_confirmation"
            or rank8.get("terminal") != "paired_confirmation" or reader.get("terminal") != "screen"
            or capability.get("terminal") != "manifest"):
        raise RuntimeError("authority terminal changed")
    allowed = {row_id for ids in capability["jointly_capable_row_ids"].values() for row_id in ids}
    rows = [row for row in candidate.build_rows()
            if row["transform_id"] in {"A1", "A2"} and row["row_id"] in allowed]
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "rows": len(rows), "rank": 8,
        "l9_heads": list(TRIPLE), "model_forwards_max": MAX_FORWARDS,
        "example_evaluations_max": MAX_EVALUATIONS, "records": RECORDS,
        "fit_updates": 0, "transformer_backwards": 0, "model_updates": 0}
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
    records, forwards, evaluations = [], 0, 0
    reconstruction = identity = self_clamp = value_weight_error = factor_closure = replay_error = 0.0
    panel_metrics, causal_fraction, residual_fraction, behavior_metrics = {}, {}, {}, {}
    for panel in ("A1", "A2"):
        panel_rows = [row for row in rows if row["transform_id"] == panel]
        base_batch = das._batch(backend, panel_rows, side="base")
        donor_batch = das._batch(backend, panel_rows, side="donor")
        base_output, base8 = attention_eval.capture_layer_attention(
            backend, base_batch, 8, call=lambda: backend.native(base_batch, capture=True))
        donor_output, donor8 = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        base9_output, base9 = attention_eval.capture_layer_attention(backend, base_batch, 9)
        base11_output, base_h3, _base_current = capture_h3_current(
            backend, base_batch, lambda: backend.native(base_batch, capture=False))
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, base8, donor8, destinations, ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            writer_output, writer_h3, writer_current = capture_h3_current(
                backend, base_batch, lambda: backend.native(base_batch, capture=False))
        finally:
            handle.remove()
        clamped_output, clamped_h3, clamped_current, clamp_error = run_l9_clamped(
            backend, base_batch, writer_hook, base9, destinations, enable_writer=True)
        self_output, _self_h3, _self_current, self_error = run_l9_clamped(
            backend, base_batch, writer_hook, base9, destinations, enable_writer=False)
        reconstruction = max(reconstruction, clamp_error, self_error, *(float(value["reconstruction_max_abs"])
            for value in (base8, donor8, base9, base_h3, writer_h3, clamped_h3)))
        identity = max(identity, pair_error(base_output, base9_output), pair_error(base_output, base11_output))
        self_clamp = max(self_clamp, pair_error(base_output, self_output))
        exact_coordinates, value_coordinates, remainder_coordinates = [], [], []
        full_coordinates, exact_norms, full_norms, remainder_norms = [], [], [], []
        for index, (query, sources) in enumerate(zip(base_batch.semantic_positions, destinations)):
            query, sources = int(query), tuple(int(source) for source in sources)
            exact_head = (writer_h3["head_output"][index, query, 3].float()
                          - clamped_h3["head_output"][index, query, 3].float())
            full_head = (writer_h3["head_output"][index, query, 3].float()
                         - base_h3["head_output"][index, query, 3].float())
            p0 = clamped_h3["pattern"][index, 3, query, list(sources)].float()
            p1 = writer_h3["pattern"][index, 3, query, list(sources)].float()
            v0 = clamped_h3["value"][index, list(sources), 3].float()
            v1 = writer_h3["value"][index, list(sources), 3].float()
            value_head = (p0[:, None] * (v1-v0)).sum(0)
            pattern_head = ((p1-p0)[:, None] * v0).sum(0)
            interaction_head = ((p1-p0)[:, None] * (v1-v0)).sum(0)
            exact_c, full_c, value_c = exact_head @ q, full_head @ q, value_head @ q
            remainder_c = exact_c - value_c
            exact_coordinates.append(exact_c)
            full_coordinates.append(full_c)
            value_coordinates.append(value_c)
            remainder_coordinates.append(remainder_c)
            exact_norms.append(float(backend.torch.linalg.vector_norm(exact_c)))
            full_norms.append(float(backend.torch.linalg.vector_norm(full_c)))
            remainder_norms.append(float(backend.torch.linalg.vector_norm(remainder_c)))
            subject_factor = value_head + pattern_head + interaction_head
            factor_closure = max(factor_closure, float((exact_head-subject_factor).abs().max()))
            normalized_delta = (backend.F.rms_norm(writer_current["current"][index, list(sources)],
                (backend.model.config.n_embd,)).float() - backend.F.rms_norm(
                clamped_current["current"][index, list(sources)], (backend.model.config.n_embd,)).float())
            weighted = backend.F.linear(normalized_delta,
                backend.model.transformer.h[11].attn.c_v.weight).view(len(sources), 9, 128)[:, 3]
            weighted = weighted * (1.0-backend.model.transformer.h[11].attn.lamb.float())
            value_weight_error = max(value_weight_error, float((weighted-(v1-v0)).abs().max()))
        exact_tensor = backend.torch.stack(exact_coordinates)
        value_tensor = backend.torch.stack(value_coordinates)
        exact_cache = endpoint_cache(backend, base_batch, base_output, exact_tensor, q, gain)
        exact_output = backend.patched(base_batch,
            site=kernel.SiteRef(site_id="resid:18", evidence_kind="residual"), donor_cache=exact_cache)
        forwards += 8
        evaluations += 8*len(panel_rows)
        panel_metrics[panel] = vector_metrics(
            [[float(v) for v in row] for row in exact_coordinates],
            [[float(v) for v in row] for row in value_coordinates])
        causal_fraction[panel] = sum(exact_norms)/sum(full_norms)
        residual_fraction[panel] = sum(remainder_norms)/sum(exact_norms)
        exact_effects, predicted_effects = [], []
        for index, row in enumerate(panel_rows):
            x = base_output.captured[(row["row_id"], "resid:18")]
            analytic_reader, _epsilon = reader_impl.analytic_reader(
                backend, x, base_batch.answer_ids[index], base_batch.foil_ids[index], residual_modes)
            predicted_effect = float(value_coordinates[index] @ analytic_reader)
            base_margin = float(base_output.answer_foil[index][0]-base_output.answer_foil[index][1])
            exact_margin = float(exact_output.answer_foil[index][0]-exact_output.answer_foil[index][1])
            exact_effect = exact_margin-base_margin
            exact_effects.append(exact_effect); predicted_effects.append(predicted_effect)
            records.append({"row_id": row["row_id"], "panel": panel,
                "exact_coordinates": [float(v) for v in exact_coordinates[index]],
                "value_weight_coordinates": [float(v) for v in value_coordinates[index]],
                "coordinate_remainder": [float(v) for v in remainder_coordinates[index]],
                "exact_margin_effect": exact_effect, "predicted_margin_effect": predicted_effect})
        rmse = math.sqrt(sum((a-b)**2 for a, b in zip(exact_effects, predicted_effects))/len(exact_effects))
        exact_rms = math.sqrt(sum(a*a for a in exact_effects)/len(exact_effects))
        behavior_metrics[panel] = {"cosine": cosine(exact_effects, predicted_effects),
            "rmse": rmse, "exact_rms": exact_rms, "relative_rmse": rmse/exact_rms}
        replay_error = max(replay_error, abs((sum(exact_effects)/len(exact_effects))))
    pred_a = bool(reconstruction <= 5e-4 and identity <= 1e-4 and self_clamp <= 1e-4
        and value_weight_error <= 5e-4 and len(records) == RECORDS and forwards <= MAX_FORWARDS
        and evaluations <= MAX_EVALUATIONS and all(math.isfinite(r["predicted_margin_effect"]) for r in records))
    pred_b = all(causal_fraction[panel] >= .80 for panel in ("A1", "A2"))
    pred_c = all(panel_metrics[panel]["cosine"] >= .95
        and panel_metrics[panel]["relative_rmse"] <= .25 for panel in ("A1", "A2"))
    pred_d = all(residual_fraction[panel] <= .25 for panel in ("A1", "A2"))
    pred_e = all(behavior_metrics[panel]["cosine"] >= .90
        and behavior_metrics[panel]["relative_rmse"] <= .35 for panel in ("A1", "A2"))
    predictions = {"pred_a_authority_exact_capture_clamp_coverage_and_price": pred_a,
        "pred_b_l9_triple_removes_material_rank8_coordinates": pred_b,
        "pred_c_value_weight_writer_explains_rank8_coordinate_removal": pred_c,
        "pred_d_pattern_terms_are_secondary": pred_d,
        "pred_e_composed_margin_prediction_is_accurate": pred_e}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "temporal_auxiliary_l9_triple_to_h3_rank8_weight_writer_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "instrument": {"attention_reconstruction_max_abs": reconstruction,
            "capture_identity_max_abs": identity, "base_self_clamp_max_abs": self_clamp,
            "value_weight_reconstruction_max_abs": value_weight_error,
            "subject_factor_head_closure_max_abs": factor_closure},
        "l9_fraction_of_live_h3_q8_norm": causal_fraction,
        "value_weight_coordinate_metrics": panel_metrics,
        "nonvalue_coordinate_norm_fraction": residual_fraction,
        "composed_margin_prediction_metrics": behavior_metrics,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "records": len(records), "fit_updates": 0, "transformer_backwards": 0,
            "model_updates": 0}, "records": records}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
        "l9_fraction_of_live_h3_q8_norm", "value_weight_coordinate_metrics",
        "nonvalue_coordinate_norm_fraction", "composed_margin_prediction_metrics",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
