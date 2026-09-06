#!/usr/bin/env python3
"""Factor H3 rank-eight causal effects into upstream and analytic weight-reader coordinates."""

# BQGATE: EXPERIMENT pred_a_authority_exact_formula_coverage_and_price pred_b_linearized_effect_cosine_at_least_point95 pred_c_linearized_relative_rmse_at_most_point25 pred_d_no_single_fixed_reader_coordinate_dominates
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import attention_path_mediation_eval as mediation
import attention_source_destination_eval as attention_eval
import attention_source_group_eval as scoring
import circuit_candidate_temporal_auxiliary_fresh_cues_v10 as candidate
import circuit_das_subspace as das
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import residual_source_onset_eval as onset
import run_temporal_auxiliary_will_had_h3_rank8_weight_direct_residual_route_v2 as direct
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_rank8_analytic_final_reader_v1.json"
DIRECT_RESULT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_weight_direct_residual_route_v2_result.json"
CAPABILITY = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v10_capability_v1_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v10.py"
FAMILY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1.py"
DIRECT_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_rank8_weight_direct_residual_route_v2.py"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_analytic_final_reader_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_rank8_analytic_final_reader_v1"
EXPECTED = {
    "prior": "63c8a8dfc1dc5bbbc5608e2e49c3ea2508381180e437799b6f7a33de6b74ead0",
    "direct_result": "571a3d0d22fe159adbc0e37825873b4dda25dda99c46b7a47a9cc6a260de471f",
    "capability": "9923322703c72d50b2a1f06138ef35269db48e0a8a4ccb365f82df3519b113ad",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "builder": "e945e0b4679fa74d6cea23594ba553d9b2ffd3ac653c353d09d1873f2a3e4494",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
    "direct_runner": "31d690664f7a069ab577340aa5768421ed761436cb765a668042ec2e39e9ba23",
}
MAX_FORWARDS, MAX_EVALUATIONS, RECORDS = 12, 400, 63


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def cosine(x, y):
    denominator = math.sqrt(sum(value*value for value in x) * sum(value*value for value in y))
    return sum(a*b for a, b in zip(x, y)) / denominator if denominator else float("nan")


def build_residual_modes(backend, q, gain):
    weight = backend.model.transformer.h[11].attn.c_proj.weight
    head_count = int(backend.model.config.n_head)
    head_width = int(backend.model.config.n_embd // head_count)
    modes = []
    for mode in range(q.shape[1]):
        flattened = backend.torch.zeros(head_count * head_width, device=weight.device, dtype=weight.dtype)
        flattened[3*head_width:4*head_width] = q[:, mode].to(flattened)
        modes.append(backend.F.linear(flattened, weight).float() * gain)
    return backend.torch.stack(modes)


def analytic_reader(backend, x, answer_id, foil_id, modes):
    x = x.float()
    epsilon = float(backend.torch.finfo(x.dtype).eps)
    scale = (x.square().mean() + epsilon).sqrt()
    normalized = x / scale
    weights = backend.model.lm_head.weight.float()
    answer_weight, foil_weight = weights[int(answer_id)], weights[int(foil_id)]
    answer_pre = answer_weight @ normalized
    foil_pre = foil_weight @ normalized
    answer_gate = 1.0 - backend.torch.tanh(answer_pre / 30.0).square()
    foil_gate = 1.0 - backend.torch.tanh(foil_pre / 30.0).square()
    gradient_normalized = answer_gate * answer_weight - foil_gate * foil_weight
    gradient_x = (gradient_normalized / scale
                  - x * (x @ gradient_normalized) / (x.numel() * scale.pow(3)))
    reader = modes @ gradient_x
    return reader, epsilon


def main():
    paths = {"prior": PRIOR, "direct_result": DIRECT_RESULT, "capability": CAPABILITY,
             "subspace": SUBSPACE, "builder": BUILDER, "family_runner": FAMILY_RUNNER,
             "direct_runner": DIRECT_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("analytic reader authority changed")
    prior, direct_result, capability, subspace = [json.loads(path.read_text())
        for path in (PRIOR, DIRECT_RESULT, CAPABILITY, SUBSPACE)]
    if (prior.get("candidate_id") != CANDIDATE_ID or direct_result.get("terminal") != "screen"
            or capability.get("terminal") != "manifest"):
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
    modes = build_residual_modes(backend, q, gain)
    records, forwards, evaluations = [], 0, 0
    reconstruction = identity_error = replay_error = 0.0
    epsilons, metrics, dominance = set(), {}, {}
    for panel in ("A1", "A2"):
        panel_rows = [row for row in rows if row["transform_id"] == panel]
        base_batch = das._batch(backend, panel_rows, side="base")
        donor_batch = das._batch(backend, panel_rows, side="donor")
        base_output, base8 = attention_eval.capture_layer_attention(
            backend, base_batch, 8, call=lambda: backend.native(base_batch, capture=True))
        donor_output, donor8 = attention_eval.capture_layer_attention(backend, donor_batch, 8)
        base11_output, base11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        destinations = onset.positions_for_group(base_batch, donor_batch, "subject_onset")
        writer_hook = mediation.fixed_source_delta_hook(
            backend, base_batch, donor_batch, base8, donor8, destinations, ("cue",), selected_heads=(1,))
        handle = backend.model.transformer.h[8].attn.c_proj.register_forward_pre_hook(writer_hook)
        try:
            _writer_output, writer11 = attention_eval.capture_layer_attention(backend, base_batch, 11)
        finally:
            handle.remove()
        cache18, _writes = direct.direct_cache(backend, base_batch, base_output, base11, writer11, q, gain)
        direct_output = backend.patched(base_batch,
            site=kernel.SiteRef(site_id="resid:18", evidence_kind="residual"), donor_cache=cache18)
        forwards += 5
        evaluations += 5 * len(panel_rows)
        identity_error = max(identity_error, max(abs(float(a)-float(b)) for left, right in zip(
            base_output.answer_foil, base11_output.answer_foil) for a, b in zip(left, right)))
        reconstruction = max(reconstruction, *(float(value["reconstruction_max_abs"])
            for value in (base8, donor8, base11, writer11)))
        recovery_summary = scoring.summarize(scoring.recovery_records(
            panel_rows, base_output, donor_output, direct_output, arm="direct"))
        replay_error = max(replay_error, abs(recovery_summary["mean_recovery"]
            - direct_result["summaries"][panel]["weight_direct_resid18"]["mean_recovery"]))
        for index, row in enumerate(panel_rows):
            query = int(base_batch.semantic_positions[index])
            delta = (writer11["head_output"][index, query, 3].float()
                     - base11["head_output"][index, query, 3].float())
            coefficients = delta @ q
            x = base_output.captured[(row["row_id"], "resid:18")]
            reader, epsilon = analytic_reader(
                backend, x, base_batch.answer_ids[index], base_batch.foil_ids[index], modes)
            epsilons.add(epsilon)
            contributions = coefficients * reader
            predicted = float(contributions.sum())
            base_margin = float(base_output.answer_foil[index][0] - base_output.answer_foil[index][1])
            direct_margin = float(direct_output.answer_foil[index][0] - direct_output.answer_foil[index][1])
            records.append({"row_id": row["row_id"], "panel": panel,
                "exact_margin_effect": direct_margin-base_margin,
                "predicted_margin_effect": predicted,
                "upstream_coordinates": [float(value) for value in coefficients],
                "reader_coordinates": [float(value) for value in reader],
                "coordinate_contributions": [float(value) for value in contributions]})
        selected = [record for record in records if record["panel"] == panel]
        exact = [record["exact_margin_effect"] for record in selected]
        predicted = [record["predicted_margin_effect"] for record in selected]
        rmse = math.sqrt(sum((a-b)**2 for a, b in zip(exact, predicted)) / len(exact))
        exact_rms = math.sqrt(sum(value*value for value in exact) / len(exact))
        metrics[panel] = {"cosine": cosine(exact, predicted), "rmse": rmse,
                          "exact_rms": exact_rms, "relative_rmse": rmse/exact_rms}
        mean_abs = [sum(abs(record["coordinate_contributions"][mode]) for record in selected)
                    / len(selected) for mode in range(8)]
        dominance[panel] = {"mean_absolute_contribution": mean_abs,
            "largest_fraction_of_sum": max(mean_abs)/sum(mean_abs),
            "largest_coordinate": max(range(8), key=lambda mode: mean_abs[mode])}
    pred_a = bool(reconstruction <= 5e-4 and identity_error <= 1e-4 and replay_error <= 1e-6
        and modes.shape == (8, 1152) and len(epsilons) == 1 and forwards <= MAX_FORWARDS
        and evaluations <= MAX_EVALUATIONS and len(records) == RECORDS
        and all(math.isfinite(record["predicted_margin_effect"]) for record in records))
    pred_b = all(metrics[panel]["cosine"] >= .95 for panel in ("A1", "A2"))
    pred_c = all(metrics[panel]["relative_rmse"] <= .25 for panel in ("A1", "A2"))
    pred_d = all(dominance[panel]["largest_fraction_of_sum"] <= .80 for panel in ("A1", "A2"))
    predictions = {"pred_a_authority_exact_formula_coverage_and_price": pred_a,
        "pred_b_linearized_effect_cosine_at_least_point95": pred_b,
        "pred_c_linearized_relative_rmse_at_most_point25": pred_c,
        "pred_d_no_single_fixed_reader_coordinate_dominates": pred_d}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "temporal_auxiliary_h3_rank8_analytic_final_reader_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "frozen_skip_gain": gain, "rms_epsilon": list(epsilons),
        "instrument": {"attention_reconstruction_max_abs": reconstruction,
            "base_capture_identity_max_abs": identity_error,
            "direct_route_receipt_mean_replay_max_abs": replay_error,
            "residual_mode_shape": list(modes.shape)},
        "prediction_metrics": metrics, "coordinate_dominance": dominance,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
            "records": len(records), "fit_updates": 0, "transformer_backwards": 0,
            "model_updates": 0}, "records": records}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument",
        "prediction_metrics", "coordinate_dominance", "predictions", "terminal", "price")},
        sort_keys=True))


if __name__ == "__main__":
    main()
