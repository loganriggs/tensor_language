#!/usr/bin/env python3
"""Compile validated carrier programs into exact task-subspace weight tensors."""

# BQGATE: EXPERIMENT pred_a_authority_basis_rank_gauge_finiteness_and_exact_replay pred_b_shared_backbone_is_top_weight_connected pred_c_shared_backbone_reuses_weight_modes pred_d_pruned_attention_edges_are_weight_connected pred_e_zero_causal_fit_and_exact_price
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
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_pruned_program_weight_tensor_manifest_v1.json"
PROGRAMS = ROOT / "circuits/followups/aspectual_tense_carrier_program_backward_pruning_v1_result.json"
SUBSPACES = ROOT / "circuits/followups/aspectual_tense_l9h1h4_shared_value_weight_subspace_v2_result.json"
ATLAS = ROOT / "ops/subspace_weight_atlas.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/aspectual_tense_pruned_program_weight_tensor_manifest_v1_result.json"
CANDIDATE_ID = "aspectual_tense.pruned_program_weight_tensor_manifest_v1"
EXPECTED = {"prior": "a2af0e86e731d2530444b6e33c3b117b84cce45a1e96ba00f5755d0c917d7ef2",
            "programs": "bb6344f6446a5426a9b6342c30cbcd56ca821a01b5750e1ef3b940a6b52e15c0",
            "subspaces": "0ae262ee932d6ecb93d1df028ac080f3a4597861620da79b7ad45a0f3ae2d16e",
            "atlas": "2e7d3a546813a6029eca6fae455ad5abd03b429fcee92432ca6fe06e835e83f5",
            "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498"}
HEADS = (1, 4)
SHARED = ("MLP3", "MLP4", "MLP8")
PRICE = {"model_forwards": 0, "example_evaluations": 0, "causal_records": 0,
         "fitted_scalars": 0, "transformer_backwards": 0, "model_updates": 0}


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def stored_basis(torch, record):
    shape = tuple(record["shape"])
    values = torch.tensor(record["values_column_major"], dtype=torch.float32)
    basis = values.reshape(shape[1], shape[0]).T.contiguous()
    return torch.linalg.qr(basis.double()).Q.float()


def validate_static():
    paths = {"prior": PRIOR, "programs": PROGRAMS, "subspaces": SUBSPACES,
             "atlas": ATLAS, "producer": PRODUCER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("authority or implementation hash changed")
    programs, subspaces = [json.loads(path.read_text()) for path in (PROGRAMS, SUBSPACES)]
    if (programs.get("terminal") != "screen" or subspaces.get("terminal") != "null"
            or {task: subspaces["subspaces"][task]["rank"] for task in ("has", "is")}
            != {"has": 18, "is": 3}
            or not all(label in programs["pruned_paths"][task]
                       for task in ("has", "is") for label in SHARED)):
        raise ExperimentError("program or task-subspace authority changed")
    return programs, subspaces


def main():
    programs, subspaces = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "tasks": ["has", "is"],
              "upstream_mlp_count_per_task": 9, "upstream_head_count_per_task": 81,
              **PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    bases = {task: stored_basis(torch, subspaces["subspaces"][task]["basis"]).cuda()
             for task in ("has", "is")}
    manifests, flats, replay_errors, gauge_errors = {}, {}, [], []
    for task in ("has", "is"):
        basis = atlas.orthonormal_basis(bases[task])
        read = atlas.head_bank_value_read_map(model.transformer.h[9].attn, HEADS, basis)
        reverse = torch.eye(read.shape[0], device=read.device).flip(0)
        gauged_read = reverse @ read
        generator = torch.Generator(device="cpu").manual_seed(9100 + (task == "is"))
        x = torch.randn(3, 128, generator=generator).to(backend.device)
        mlps, heads = [], []
        flats[task] = {}
        for layer in range(9):
            block = model.transformer.h[layer]
            item = atlas.mlp_writer_to_read_tensor(block.mlp, read)
            tensor, output = item["tensor"], item["output"]
            predicted = torch.einsum("aij,bi,bj->ba", tensor, x, x)
            direct = ((x @ item["left"].T) * (x @ item["right"].T)) @ output.T
            replay_errors.append(float((predicted - direct).abs().max()))
            gauged = atlas.mlp_writer_to_read_tensor(block.mlp, gauged_read)
            gauge_errors.append(abs(item["score"] - gauged["score"]))
            flat = tensor.reshape(tensor.shape[0], -1)
            singular = torch.linalg.svdvals(flat)
            rank = int((singular > singular.max().clamp_min(1e-30) * 1e-6).sum())
            label = f"MLP{layer}"
            flats[task][label] = flat
            mlps.append({"label": label, "layer": layer, "shape": list(tensor.shape),
                         "sha256": tensor_sha(tensor), "score": item["score"],
                         "normalized_score": item["normalized_score"], "rank": rank,
                         "singular_values": [float(value) for value in singular.detach().cpu()]})
            for head in range(int(block.attn.n_head)):
                mapped = atlas.attention_writer_to_read_map(block.attn, head, read)
                gauged = atlas.attention_writer_to_read_map(block.attn, head, gauged_read)
                gauge_errors.append(abs(mapped["score"] - gauged["score"]))
                width = int(block.attn.head_dim)
                output_weight = block.attn.c_proj.weight.detach().float()[:, head * width:(head + 1) * width]
                denominator = float(torch.linalg.matrix_norm(read)) * float(torch.linalg.matrix_norm(output_weight))
                heads.append({"label": f"L{layer}H{head}", "layer": layer, "head": head,
                              "shape": list(mapped["contraction"].shape),
                              "sha256": tensor_sha(mapped["contraction"]), "score": mapped["score"],
                              "normalized_score": mapped["score"] / denominator if denominator else 0.0})
        mlps.sort(key=lambda row: (-row["normalized_score"], row["label"]))
        heads.sort(key=lambda row: (-row["normalized_score"], row["label"]))
        manifests[task] = {"basis_rank": int(basis.shape[1]), "basis_sha256": tensor_sha(basis),
                           "read_map_shape": list(read.shape), "read_map_sha256": tensor_sha(read),
                           "read_map_norm": float(torch.linalg.matrix_norm(read)),
                           "mlps": mlps, "attention_heads": heads,
                           "mlp_ranking": [row["label"] for row in mlps],
                           "head_ranking": [row["label"] for row in heads]}
    shared_modes = {}
    for label in SHARED:
        rowspaces = {}
        for task in ("has", "is"):
            flat = flats[task][label]
            _u, singular, vh = torch.linalg.svd(flat, full_matrices=False)
            keep = singular > singular.max().clamp_min(1e-30) * 1e-6
            rowspaces[task] = vh[keep].T
        cosines = torch.linalg.svdvals(rowspaces["has"].T @ rowspaces["is"])
        shared_modes[label] = [float(value) for value in cosines.detach().cpu()]
    numeric = [*replay_errors, *gauge_errors,
               *[row[key] for task in manifests.values() for group in ("mlps", "attention_heads")
                 for row in task[group] for key in ("score", "normalized_score")]]
    pred_a = bool({task: manifests[task]["basis_rank"] for task in manifests} == {"has": 18, "is": 3}
                  and max(replay_errors) <= 1e-4 and max(gauge_errors) <= 1e-4
                  and all(math.isfinite(value) for value in numeric)
                  and all(len(row["sha256"]) == 64 for task in manifests.values()
                          for group in ("mlps", "attention_heads") for row in task[group]))
    pred_b = all(label in manifests[task]["mlp_ranking"][:6]
                 for task in ("has", "is") for label in SHARED)
    pred_c = all(max(shared_modes[label]) >= 0.50 for label in SHARED)
    retained_heads = {task: [label for label in programs["pruned_paths"][task]
                             if label.startswith("L")] for task in ("has", "is")}
    head_lookup = {task: {row["label"]: row for row in manifests[task]["attention_heads"]}
                   for task in ("has", "is")}
    pred_d = all(math.isfinite(head_lookup[task][label]["normalized_score"])
                 and head_lookup[task][label]["normalized_score"] > 0
                 for task in retained_heads for label in retained_heads[task])
    pred_e = True
    predictions = {
        "pred_a_authority_basis_rank_gauge_finiteness_and_exact_replay": pred_a,
        "pred_b_shared_backbone_is_top_weight_connected": pred_b,
        "pred_c_shared_backbone_reuses_weight_modes": pred_c,
        "pred_d_pruned_attention_edges_are_weight_connected": pred_d,
        "pred_e_zero_causal_fit_and_exact_price": pred_e,
    }
    terminal = "invalid" if not pred_a or not pred_e else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_pruned_program_weight_tensor_manifest_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started,
              "authority_sha256": EXPECTED, "programs": programs["pruned_paths"],
              "manifests": manifests, "shared_tensor_rowspace_principal_cosines": shared_modes,
              "maximum_exact_replay_error": max(replay_errors),
              "maximum_gauge_score_error": max(gauge_errors),
              "retained_attention_heads": retained_heads,
              "predictions": predictions, "price": PRICE, "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({"candidate_id": CANDIDATE_ID,
        "mlp_rankings": {task: manifests[task]["mlp_ranking"] for task in manifests},
        "retained_attention_heads": retained_heads,
        "shared_tensor_rowspace_principal_cosines": shared_modes,
        "maximum_exact_replay_error": result["maximum_exact_replay_error"],
        "maximum_gauge_score_error": result["maximum_gauge_score_error"],
        "predictions": predictions, "price": PRICE, "terminal": terminal}, sort_keys=True))


if __name__ == "__main__":
    main()
