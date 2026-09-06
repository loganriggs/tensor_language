#!/usr/bin/env python3
"""Task-conditioned exact MLP writer scores from A1-fit carrier inputs."""

# BQGATE: EXPERIMENT pred_a_exact_factor_replay_gauge_capability_and_coverage pred_b_conditioning_improves_both_mlp_correlations pred_c_conditioning_repairs_both_rankings pred_d_zero_causal_fit_and_exact_price
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import subspace_weight_atlas as atlas
import run_aspectual_tense_l9h1h4_source_position_weight_validation_v1 as source_rank


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_activation_conditioned_mlp_writer_atlas_v1.json"
SUBSPACES = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_weight_subspace_v2_result.json"
CAUSAL = ROOT / "circuits/followups/aspectual_tense_l9h1h4_source_position_weight_validation_v1_result.json"
STATIC = ROOT / "circuits/followups/aspectual_tense_complete_mlp_weight_tensor_diagnostic_v1_result.json"
LIBRARY = ROOT / "ops/subspace_weight_atlas.py"
OUT = ROOT / "circuits/followups/aspectual_tense_activation_conditioned_mlp_writer_atlas_v1_result.json"
CANDIDATE_ID = "aspectual_tense.activation_conditioned_mlp_writer_atlas_v1"
EXPECTED = {
    "prior": "21d482a26203252bf5ef64028493bd3638066b743fc875ffdf93b27c878a89a8",
    "subspaces": "0ae262ee932d6ecb93d1df028ac080f3a4597861620da79b7ad45a0f3ae2d16e",
    "causal": "7692b9c3095e66935934a4a31c7263ea3986f3fd070d4faf271e8ddf6e5ec261",
    "static": "f8dc87a411b0bab4a4eb9781194920b87ffe1649a9ce2824502f08b4b249158a",
    "library": "2e7d3a546813a6029eca6fae455ad5abd03b429fcee92432ca6fe06e835e83f5",
}
HEADS = (1, 4)
STATIC_CORRELATIONS = {"has": 0.30952380952380953, "is": -0.47619047619047616}


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def stored_basis(torch, record):
    shape = tuple(record["shape"])
    values = torch.tensor(record["values_column_major"], dtype=torch.float32)
    basis = values.reshape(shape[1], shape[0]).T.contiguous()
    return torch.linalg.qr(basis.double()).Q.float()


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = float(rank)
    return result


def spearman(left, right):
    return statistics.correlation(ranks(left), ranks(right))


def capture_mlp_inputs(backend, batch):
    captured, handles = {}, []
    for layer in range(9):
        def save(_module, arguments, layer=layer):
            captured[layer] = arguments[0].detach().clone()
        handles.append(backend.model.transformer.h[layer].mlp.register_forward_pre_hook(save))
    try:
        output = backend.native(batch, capture=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(range(9)):
        raise ExperimentError("MLP input capture is incomplete")
    return output, captured


def validate_static():
    paths = {"prior": PRIOR, "subspaces": SUBSPACES, "causal": CAUSAL,
             "static": STATIC, "library": LIBRARY}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, data, or conditioned tensor library hash changed")
    prior, subspaces, causal, static = [json.loads(path.read_text())
        for path in (PRIOR, SUBSPACES, CAUSAL, STATIC)]
    splits, _chosen, _query = source_rank.validate_static()
    if (prior.get("candidate_id") != CANDIDATE_ID or static.get("terminal") != "null"
            or {task: static["correlations"][task]["complete_tensor_normalized"]
                for task in ("has", "is")} != STATIC_CORRELATIONS
            or any(len(causal["component_summaries"][task]["upstream_mlp"]) != 8
                   for task in ("has", "is"))
            or len(splits["has_fit"]) != 16 or len(splits["is_fit"]) != 8):
        raise ExperimentError("frozen development authority changed")
    return subspaces, causal, splits


def main():
    subspaces, causal, splits = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "model_forwards": 4,
              "example_evaluations": 48, "task_layer_scores": 18,
              "selected_causal_matches": 16, "causal_interventions": 0,
              "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    scores, correlations, rankings = {}, {}, {}
    all_capable, all_finite, matched = True, True, 0
    forwards = evaluations = 0
    for task in ("has", "is"):
        rows = splits[f"{task}_fit"]
        base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
        base_output, base_inputs = capture_mlp_inputs(backend, base_batch)
        donor_output, donor_inputs = capture_mlp_inputs(backend, donor_batch)
        forwards += 2
        evaluations += 2 * len(rows)
        all_capable = all_capable and all(a - f > 0 for a, f in base_output.answer_foil + donor_output.answer_foil)
        banks = source_rank.carrier_banks(task, base_batch, donor_batch)
        basis = stored_basis(torch, subspaces["subspaces"][task]["basis"]).to(backend.device)
        read = atlas.head_bank_value_read_map(model.transformer.h[9].attn, HEADS, basis)
        task_scores = []
        for layer in range(9):
            response = atlas.activation_conditioned_mlp_write(
                model.transformer.h[layer].mlp, read, base_inputs[layer], donor_inputs[layer])["response"]
            norms = [float(torch.linalg.vector_norm(response[index, position]))
                     for index, bank in enumerate(banks) for position in bank]
            score = statistics.fmean(norms)
            task_scores.append({"label": f"MLP{layer}", "layer": layer,
                                "mean_carrier_read_response_norm": score})
            all_finite = all_finite and math.isfinite(score)
        scores[task] = task_scores
        causal_rows = causal["component_summaries"][task]["upstream_mlp"]
        selected = [row for row in task_scores if row["label"] in causal_rows]
        matched += len(selected)
        conditioned = spearman(
            [row["mean_carrier_read_response_norm"] for row in selected],
            [causal_rows[row["label"]]["mean_absolute_recovery"] for row in selected])
        correlations[task] = {"complete_static": STATIC_CORRELATIONS[task],
                              "activation_conditioned": conditioned}
        rankings[task] = [row["label"] for row in sorted(task_scores,
            key=lambda row: (-row["mean_carrier_read_response_norm"], row["label"]))]

    pred_a = bool(all_capable and all_finite and matched == 16
                  and all(len(scores[task]) == 9 for task in ("has", "is")))
    pred_b = all(correlations[task]["activation_conditioned"] > STATIC_CORRELATIONS[task]
                 for task in ("has", "is"))
    values = [correlations[task]["activation_conditioned"] for task in ("has", "is")]
    pred_c = all(value > 0 for value in values) and statistics.median(values) > 0.40
    pred_d = forwards == 4 and evaluations == 48
    predictions = {
        "pred_a_exact_factor_replay_gauge_capability_and_coverage": pred_a,
        "pred_b_conditioning_improves_both_mlp_correlations": pred_b,
        "pred_c_conditioning_repairs_both_rankings": pred_c,
        "pred_d_zero_causal_fit_and_exact_price": pred_d,
    }
    terminal = "invalid" if not pred_a or not pred_d else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_activation_conditioned_mlp_writer_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "scope": "development_metric_diagnostic_only",
        "execution_policy": "managed_queue_only", "started_utc": started_utc,
        "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun, "scores": scores,
        "rankings": rankings, "correlations": correlations, "predictions": predictions,
        "price": {"model_forwards": forwards, "example_evaluations": evaluations,
                  "causal_interventions": 0, "fitted_scalars": 0,
                  "transformer_backwards": 0, "model_updates": 0},
        "terminal": terminal,
        "reason": "conditioned_mlp_metric_graduates_to_fresh_causal_validation" if terminal == "screen"
                  else "conditioned_local_write_still_omits_path_or_suffix_sensitivity" if terminal == "null"
                  else "factor_replay_gauge_capability_coverage_matching_or_price_invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "rankings", "correlations",
        "predictions", "price", "terminal", "reason")}, sort_keys=True))


if __name__ == "__main__":
    main()
