#!/usr/bin/env python3
"""Relative-error audit of the exact pruned-program weight tensor manifest."""

# BQGATE: EXPERIMENT pred_a_authority_basis_and_hash_reproduction pred_b_relative_replay_passes pred_c_relative_gauge_passes pred_d_substantive_manifest_predictions_reproduce pred_e_zero_causal_fit_and_exact_price
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
import run_aspectual_tense_pruned_program_weight_tensor_manifest_v1 as manifest_runner
import subspace_weight_atlas as atlas


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/aspectual_tense_weight_tensor_relative_instrument_audit_v1.json"
MANIFEST = ROOT / "circuits/followups/aspectual_tense_pruned_program_weight_tensor_manifest_v1_result.json"
RUNNER = ROOT / "ops/run_aspectual_tense_pruned_program_weight_tensor_manifest_v1.py"
OUT = ROOT / "circuits/followups/aspectual_tense_weight_tensor_relative_instrument_audit_v1_result.json"
CANDIDATE_ID = "aspectual_tense.weight_tensor_relative_instrument_audit_v1"
EXPECTED = {"prior": "eac93970a2bc9bdbe6bc70ca67d0228732c8ba61f158c5821f09c58b242e8ed9",
            "manifest": "abc0db2279b7d7e8e1091fa7f6a0ffe0908b99abb56a57bccd3a9bb2bbad1386",
            "runner": "c097cc5c32b5a105c2bfe16c70f84862f50de40a03f5eea67e647bfddd16d4e8"}
TOLERANCE = 1e-5
PRICE = manifest_runner.PRICE


class ExperimentError(RuntimeError):
    pass


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_static():
    paths = {"prior": PRIOR, "manifest": MANIFEST, "runner": RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise ExperimentError("audit authority changed")
    programs, subspaces = manifest_runner.validate_static()
    result = json.loads(MANIFEST.read_text())
    pattern = list(result["predictions"].values())
    if result.get("terminal") != "invalid" or pattern != [False, True, True, True, True]:
        raise ExperimentError("invalid manifest pattern changed")
    return programs, subspaces, result


def main():
    _programs, subspaces, old = validate_static()
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False,
              "relative_tolerance": TOLERANCE, "expected_hashes": 180, **PRICE}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    replay_relative, gauge_relative, hash_checks = [], [], []
    for task in ("has", "is"):
        basis = manifest_runner.stored_basis(
            torch, subspaces["subspaces"][task]["basis"]).cuda()
        read = atlas.head_bank_value_read_map(model.transformer.h[9].attn,
                                              manifest_runner.HEADS, basis)
        reverse = torch.eye(read.shape[0], device=read.device).flip(0)
        gauged_read = reverse @ read
        generator = torch.Generator(device="cpu").manual_seed(9100 + (task == "is"))
        old_mlps = {row["label"]: row for row in old["manifests"][task]["mlps"]}
        old_heads = {row["label"]: row for row in old["manifests"][task]["attention_heads"]}
        for layer in range(9):
            block = model.transformer.h[layer]
            item = atlas.mlp_writer_to_read_tensor(block.mlp, read)
            x = torch.randn(3, item["tensor"].shape[1], generator=generator).to(backend.device)
            predicted = torch.einsum("aij,bi,bj->ba", item["tensor"], x, x)
            direct = ((x @ item["left"].T) * (x @ item["right"].T)) @ item["output"].T
            replay_relative.append(float(torch.linalg.vector_norm(predicted - direct)
                                         / torch.linalg.vector_norm(direct).clamp_min(1e-30)))
            gauged = atlas.mlp_writer_to_read_tensor(block.mlp, gauged_read)
            gauge_relative.append(abs(item["score"] - gauged["score"])
                                  / max(abs(item["score"]), abs(gauged["score"]), 1e-30))
            label = f"MLP{layer}"
            hash_checks.append(manifest_runner.tensor_sha(item["tensor"])
                               == old_mlps[label]["sha256"])
            for head in range(int(block.attn.n_head)):
                mapped = atlas.attention_writer_to_read_map(block.attn, head, read)
                gauged = atlas.attention_writer_to_read_map(block.attn, head, gauged_read)
                gauge_relative.append(abs(mapped["score"] - gauged["score"])
                                      / max(abs(mapped["score"]), abs(gauged["score"]), 1e-30))
                label = f"L{layer}H{head}"
                hash_checks.append(manifest_runner.tensor_sha(mapped["contraction"])
                                   == old_heads[label]["sha256"])
    finite = all(math.isfinite(value) for value in replay_relative + gauge_relative)
    pred_a = bool(len(hash_checks) == 180 and all(hash_checks) and finite
                  and {task: old["manifests"][task]["basis_rank"] for task in ("has", "is")}
                  == {"has": 18, "is": 3})
    pred_b = max(replay_relative) <= TOLERANCE
    pred_c = max(gauge_relative) <= TOLERANCE
    pred_d = all(value for key, value in old["predictions"].items()
                 if not key.startswith("pred_a_"))
    pred_e = old["price"] == PRICE
    predictions = {
        "pred_a_authority_basis_and_hash_reproduction": pred_a,
        "pred_b_relative_replay_passes": pred_b,
        "pred_c_relative_gauge_passes": pred_c,
        "pred_d_substantive_manifest_predictions_reproduce": pred_d,
        "pred_e_zero_causal_fit_and_exact_price": pred_e,
    }
    terminal = "invalid" if not pred_a or not pred_e else ("screen" if all(predictions.values()) else "null")
    result = {"schema": "aspectual_tense_weight_tensor_relative_instrument_audit_result_v1",
              "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
              "started_utc": started_utc, "finished_utc": utc_now(),
              "serial_seconds": time.perf_counter() - started,
              "authority_sha256": EXPECTED, "relative_tolerance": TOLERANCE,
              "hashes_checked": len(hash_checks), "hashes_matching": sum(hash_checks),
              "maximum_relative_replay_error": max(replay_relative),
              "maximum_relative_gauge_score_error": max(gauge_relative),
              "predictions": predictions, "price": PRICE, "terminal": terminal}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "hashes_checked",
          "hashes_matching", "maximum_relative_replay_error",
          "maximum_relative_gauge_score_error", "predictions", "price", "terminal")},
          sort_keys=True))


if __name__ == "__main__":
    main()
