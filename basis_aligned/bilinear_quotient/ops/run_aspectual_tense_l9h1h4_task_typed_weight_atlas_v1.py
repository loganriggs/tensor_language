#!/usr/bin/env python3
"""Translate task-typed L9H1/H4 value subspaces into exact weight atlases."""

# BQGATE: EXPERIMENT pred_a_authority_basis_gauge_and_exact_coverage pred_b_mlp4_writer_positive_control pred_c_task_typed_weight_neighborhoods pred_d_zero_fit_zero_causal_leakage
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import subspace_weight_atlas as atlas


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_l9h1h4_task_typed_weight_atlas_v1.json"
SUBSPACES = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_weight_subspace_v2_result.json"
ATLAS = ROOT / "ops/subspace_weight_atlas.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/aspectual_tense_l9h1h4_task_typed_weight_atlas_v1_result.json"
CANDIDATE_ID = "aspectual_tense.l9h1h4_task_typed_weight_atlas_v1"
EXPECTED = {
    "prior": "cb8e70c11960a0564d6a443e99633802a68d7efde741e4ce942edbc9055c5687",
    "subspaces": "0ae262ee932d6ecb93d1df028ac080f3a4597861620da79b7ad45a0f3ae2d16e",
    "atlas": "b5cda0119a5028f2e8d2795e8b470fa680416576ba70519e0bb83ba1c1cf1f7d",
    "producer": "528e9f50152d6dc2b28d084cd0828de58de042b893703d0f322b4a2f22c4a0a7",
}
HEADS = (1, 4)
BASE_NAMES = ("has", "is", "joint", "has_not_is", "is_not_has")


class ExperimentError(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def stored_basis(torch, record):
    shape = tuple(record["shape"])
    values = torch.tensor(record["values_column_major"], dtype=torch.float32)
    basis = values.reshape(shape[1], shape[0]).T.contiguous()
    return torch.linalg.qr(basis).Q


def exclusive_basis(torch, own, other):
    residual = own - other @ (other.T @ own)
    left, singular, _right = torch.linalg.svd(residual, full_matrices=False)
    keep = singular > singular.max().clamp_min(1e-30) * 1e-6
    if not bool(keep.any()):
        raise ExperimentError("task-exclusive span is empty")
    return left[:, keep]


def validate_static():
    paths = {"prior": PRIOR, "subspaces": SUBSPACES, "atlas": ATLAS, "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    prior = json.loads(PRIOR.read_text())
    result = json.loads(SUBSPACES.read_text())
    ranks = {name: result["subspaces"][name]["rank"] for name in ("has", "is", "joint")}
    if (prior.get("candidate_id") != CANDIDATE_ID or result.get("terminal") != "null"
            or ranks != {"has": 18, "is": 3, "joint": 15}
            or result["subspaces"]["shared"]["rank"] != 0):
        raise ExperimentError("subspace authority or ranks changed")
    return result


def top_labels(rows, metric, count=6):
    return [row["label"] for row in sorted(rows, key=lambda row: (-row[metric], row["label"]))[:count]]


def main():
    result = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "bases": list(BASE_NAMES),
              "model_forwards": 0, "example_evaluations": 0, "causal_records": 0,
              "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    bases = {name: stored_basis(torch, result["subspaces"][name]["basis"]).cuda()
             for name in ("has", "is", "joint")}
    bases["has_not_is"] = exclusive_basis(torch, bases["has"], bases["is"])
    bases["is_not_has"] = exclusive_basis(torch, bases["is"], bases["has"])

    atlas_results = {}
    all_finite = True
    exact_counts = True
    for name in BASE_NAMES:
        basis = atlas.orthonormal_basis(bases[name])
        read = atlas.head_bank_value_read_map(model.transformer.h[9].attn, HEADS, basis)
        read_norm = float(torch.linalg.matrix_norm(read))
        upstream_heads = []
        upstream_mlps = []
        for layer in range(9):
            block = model.transformer.h[layer]
            for head in range(int(block.attn.n_head)):
                item = atlas.attention_writer_to_read_map(block.attn, head, read)
                start, width = head * int(block.attn.head_dim), int(block.attn.head_dim)
                output = block.attn.c_proj.weight.detach().float()[:, start:start + width]
                denominator = read_norm * float(torch.linalg.matrix_norm(output))
                upstream_heads.append({"label": f"L{layer}H{head}", "layer": layer,
                    "head": head, "score": item["score"],
                    "normalized_score": item["score"] / denominator if denominator > 0 else 0.0})
            item = atlas.mlp_writer_to_read_map(block.mlp, read)
            output = (block.mlp.Down.weight.detach().float() if hasattr(block.mlp, "Down")
                      else block.mlp.c_proj.weight.detach().float())
            denominator = read_norm * float(torch.linalg.matrix_norm(output))
            upstream_mlps.append({"label": f"MLP{layer}", "layer": layer,
                "score": item["score"],
                "normalized_score": item["score"] / denominator if denominator > 0 else 0.0})

        residual_basis, output_singular = atlas.map_head_bank_subspace_to_residual(
            model.transformer.h[9].attn, HEADS, basis)
        # The SVD span is exact, but float32 U can miss the atlas library's strict
        # 1e-6 Gram tolerance. QR changes only the within-span gauge.
        residual_basis = torch.linalg.qr(residual_basis).Q
        downstream_heads = []
        downstream_mlps = []
        for layer in range(10, 18):
            block = model.transformer.h[layer]
            factors = atlas.attention_subspace_factors(block.attn, residual_basis)
            for head in range(int(block.attn.n_head)):
                scores = factors[head]["scores"]
                downstream_heads.append({"label": f"L{layer}H{head}", "layer": layer,
                    "head": head, "value_read": scores["v"],
                    "routing_read": math.sqrt(sum(scores.get(key, 0.0) ** 2
                                                  for key in ("q", "k", "q2", "k2"))),
                    "output_write": scores["o"], "ov_recurrence": scores["ov"]})
            mlp_factors = atlas.mlp_subspace_tensor(block.mlp, residual_basis)
            downstream_mlps.append({"label": f"MLP{layer}", "layer": layer,
                                    **mlp_factors["scores"]})

        top_upstream_heads = top_labels(upstream_heads, "normalized_score")
        top_upstream_mlps = top_labels(upstream_mlps, "normalized_score", count=9)
        top_downstream_value = top_labels(downstream_heads, "value_read")
        atlas_results[name] = {
            "head_local_rank": int(basis.shape[1]), "read_map_norm": read_norm,
            "residual_output_rank": int(residual_basis.shape[1]),
            "residual_output_singular_values": [float(value) for value in output_singular.detach().cpu()],
            "upstream_attention_heads": upstream_heads, "upstream_mlps": upstream_mlps,
            "downstream_attention_heads": downstream_heads, "downstream_mlps": downstream_mlps,
            "top_six_upstream_heads": top_upstream_heads,
            "upstream_mlp_ranking": top_upstream_mlps,
            "top_six_downstream_value_readers": top_downstream_value,
        }
        exact_counts = exact_counts and (len(upstream_heads), len(upstream_mlps),
            len(downstream_heads), len(downstream_mlps)) == (81, 9, 72, 8)
        numeric = [read_norm, *[row[key] for row in upstream_heads
                    for key in ("score", "normalized_score")],
                   *[row[key] for row in upstream_mlps for key in ("score", "normalized_score")],
                   *[row[key] for row in downstream_heads
                     for key in ("value_read", "routing_read", "output_write", "ov_recurrence")],
                   *[row[key] for row in downstream_mlps for key in ("left", "right", "down", "tensor")]]
        all_finite = all_finite and all(math.isfinite(value) for value in numeric)

    pred_a = bool(exact_counts and all_finite
                  and {name: atlas_results[name]["head_local_rank"] for name in ("has", "is", "joint")}
                  == {"has": 18, "is": 3, "joint": 15})
    pred_b = all(atlas_results[name]["upstream_mlp_ranking"].index("MLP4") < 5
                 for name in ("has", "is", "joint"))
    upstream_overlap = len(set(atlas_results["has"]["top_six_upstream_heads"])
                           & set(atlas_results["is"]["top_six_upstream_heads"]))
    downstream_overlap = len(set(atlas_results["has"]["top_six_downstream_value_readers"])
                             & set(atlas_results["is"]["top_six_downstream_value_readers"]))
    pred_c = upstream_overlap <= 4 or downstream_overlap <= 4
    pred_d = True
    predictions = {"pred_a_authority_basis_gauge_and_exact_coverage": pred_a,
                   "pred_b_mlp4_writer_positive_control": pred_b,
                   "pred_c_task_typed_weight_neighborhoods": pred_c,
                   "pred_d_zero_fit_zero_causal_leakage": pred_d}
    terminal = "invalid" if not pred_a or not pred_d else (
        "screen" if all(predictions.values()) else "null")
    output = {
        "schema": "aspectual_tense_l9h1h4_task_typed_weight_atlas_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "atlases": atlas_results,
        "task_top_six_overlap": {"upstream_heads": upstream_overlap,
                                 "downstream_value_readers": downstream_overlap},
        "predictions": predictions,
        "price": {"model_forwards": 0, "example_evaluations": 0, "causal_records": 0,
                  "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0},
        "terminal": terminal,
        "reason": "task_typed_weight_neighborhoods_with_mlp4_positive_control" if terminal == "screen"
                  else "weight_atlas_positive_control_or_task_typing_null" if terminal == "null"
                  else "authority_basis_gauge_coverage_or_finiteness_invalid",
    }
    atomic_create_json(OUT, output)
    print(json.dumps({"candidate_id": CANDIDATE_ID,
        "top_rankings": {name: {key: atlas_results[name][key] for key in
          ("top_six_upstream_heads", "upstream_mlp_ranking", "top_six_downstream_value_readers")}
          for name in BASE_NAMES}, "task_top_six_overlap": output["task_top_six_overlap"],
        "predictions": predictions, "price": output["price"], "terminal": terminal,
        "reason": output["reason"]}, sort_keys=True))


if __name__ == "__main__":
    main()
