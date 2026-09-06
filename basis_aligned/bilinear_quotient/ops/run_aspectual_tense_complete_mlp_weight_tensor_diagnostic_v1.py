#!/usr/bin/env python3
"""Development diagnostic for complete quadratic MLP writer contractions."""

# BQGATE: EXPERIMENT pred_a_exact_tensor_replay_gauge_finiteness_and_coverage pred_b_complete_tensor_improves_both_mlp_rank_correlations pred_c_complete_tensor_repairs_is_was_ordering pred_d_zero_forward_development_scope
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import subspace_weight_atlas as atlas


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_complete_mlp_weight_tensor_diagnostic_v1.json"
SUBSPACES = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_weight_subspace_v2_result.json"
CAUSAL = ROOT / "circuits/followups/aspectual_tense_l9h1h4_source_position_weight_validation_v1_result.json"
LIBRARY = ROOT / "ops/subspace_weight_atlas.py"
OUT = ROOT / "circuits/followups/aspectual_tense_complete_mlp_weight_tensor_diagnostic_v1_result.json"
CANDIDATE_ID = "aspectual_tense.complete_mlp_weight_tensor_diagnostic_v1"
EXPECTED = {
    "prior": "8ce13a43bbc2fcb422ac1c58696b14b02b4d999f961d89663a37e73763f812be",
    "subspaces": "0ae262ee932d6ecb93d1df028ac080f3a4597861620da79b7ad45a0f3ae2d16e",
    "causal": "7692b9c3095e66935934a4a31c7263ea3986f3fd070d4faf271e8ddf6e5ec261",
    "library": "cc0a755b72f2c9035bf8efce0df0862bbf1c960d950d39a7e889c79860d46f5e",
}
HEADS = (1, 4)
OLD_CORRELATIONS = {"has": 0.2619047619047619, "is": -0.5952380952380952}


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


def validate_static():
    paths = {"prior": PRIOR, "subspaces": SUBSPACES, "causal": CAUSAL, "library": LIBRARY}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("prior, data, or tensor library hash changed")
    prior, subspaces, causal = [json.loads(path.read_text()) for path in (PRIOR, SUBSPACES, CAUSAL)]
    if (prior.get("candidate_id") != CANDIDATE_ID or subspaces.get("terminal") != "null"
            or causal.get("terminal") != "screen"
            or {task: causal["correlations"][task]["upstream_mlp"] for task in ("has", "is")}
                != OLD_CORRELATIONS
            or any(len(causal["component_summaries"][task]["upstream_mlp"]) != 8
                   for task in ("has", "is"))):
        raise ExperimentError("frozen subspace or causal development data changed")
    return subspaces, causal


def main():
    subspaces, causal = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "task_layer_tensors": 18,
              "selected_causal_matches": 16, "model_forwards": 0,
              "example_evaluations": 0, "causal_records": 0, "fitted_scalars": 0,
              "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    scores, correlations, rankings = {}, {}, {}
    all_finite = True
    matched = 0
    for task in ("has", "is"):
        basis = stored_basis(torch, subspaces["subspaces"][task]["basis"]).to(backend.device)
        read = atlas.head_bank_value_read_map(model.transformer.h[9].attn, HEADS, basis)
        rows = []
        for layer in range(9):
            item = atlas.mlp_writer_to_read_tensor(model.transformer.h[layer].mlp, read)
            row = {"label": f"MLP{layer}", "layer": layer,
                   "score": item["score"], "normalized_score": item["normalized_score"]}
            rows.append(row)
            all_finite = all_finite and all(math.isfinite(row[key]) for key in ("score", "normalized_score"))
            del item
        scores[task] = rows
        causal_rows = causal["component_summaries"][task]["upstream_mlp"]
        selected = [row for row in rows if row["label"] in causal_rows]
        matched += len(selected)
        correlations[task] = {
            "down_only_normalized": OLD_CORRELATIONS[task],
            "complete_tensor_raw": spearman(
                [row["score"] for row in selected],
                [causal_rows[row["label"]]["mean_absolute_recovery"] for row in selected]),
            "complete_tensor_normalized": spearman(
                [row["normalized_score"] for row in selected],
                [causal_rows[row["label"]]["mean_absolute_recovery"] for row in selected]),
        }
        rankings[task] = [row["label"] for row in sorted(
            rows, key=lambda row: (-row["normalized_score"], row["label"]))]

    pred_a = bool(all_finite and matched == 16 and all(len(scores[task]) == 9 for task in ("has", "is")))
    pred_b = all(correlations[task]["complete_tensor_normalized"] > OLD_CORRELATIONS[task]
                 for task in ("has", "is"))
    pred_c = correlations["is"]["complete_tensor_normalized"] > 0 and "MLP4" in rankings["is"][:6]
    pred_d = True
    predictions = {
        "pred_a_exact_tensor_replay_gauge_finiteness_and_coverage": pred_a,
        "pred_b_complete_tensor_improves_both_mlp_rank_correlations": pred_b,
        "pred_c_complete_tensor_repairs_is_was_ordering": pred_c,
        "pred_d_zero_forward_development_scope": pred_d,
    }
    terminal = "invalid" if not pred_a or not pred_d else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_complete_mlp_weight_tensor_diagnostic_result_v1",
        "candidate_id": CANDIDATE_ID, "scope": "development_metric_diagnostic_only",
        "execution_policy": "managed_queue_only", "started_utc": started_utc,
        "finished_utc": utc_now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun, "scores": scores,
        "rankings": rankings, "correlations": correlations, "predictions": predictions,
        "price": {"model_forwards": 0, "example_evaluations": 0, "causal_records": 0,
                  "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0},
        "terminal": terminal,
        "reason": "complete_mlp_tensor_metric_graduates_to_fresh_validation" if terminal == "screen"
                  else "static_complete_mlp_tensor_still_needs_activation_conditioning" if terminal == "null"
                  else "tensor_replay_gauge_matching_finiteness_or_price_invalid"}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "rankings", "correlations",
        "predictions", "price", "terminal", "reason")}, sort_keys=True))


if __name__ == "__main__":
    main()
