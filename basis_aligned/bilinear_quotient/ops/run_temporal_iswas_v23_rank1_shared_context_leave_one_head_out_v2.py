#!/usr/bin/env python3
"""Float64 instrument repair of the sealed v23 rank-one shared-context LOO test."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_shapes_replay_certificates_gauge_invariance_and_zero_forward_price pred_b_rank1_context_core_predicts_every_omitted_head_beyond_random pred_c_three_head_fits_identify_one_stable_physical_direction pred_d_removing_the_common_direction_reduces_cross_head_context_similarity
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

import causal_checkpoint_translation as translation
from circuit_fast_screen_managed_runner import atomic_create_json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v23_rank1_shared_context_leave_one_head_out_v2.json"
V1_RESULT = ROOT / "circuits/followups/temporal_iswas_v23_rank1_shared_context_leave_one_head_out_v1_result.json"
V1_AUDIT = ROOT / "circuits/followups/temporal_iswas_v23_rank1_shared_context_leave_one_head_out_v1_interpretation_audit.json"
V1_RUNNER = ROOT / "ops/run_temporal_iswas_v23_rank1_shared_context_leave_one_head_out_v1.py"
ARRAYS = ROOT / "circuits/followups/temporal_iswas_v23_occupied_reader_weight_writer_grouping_v1.npz"
GROUPING = ROOT / "circuits/followups/temporal_iswas_v23_occupied_reader_weight_writer_grouping_v1_result.json"
GAUGE = ROOT / "circuits/followups/temporal_iswas_v23_occupied_reader_weight_writer_gauge_audit_v1_result.json"
LIBRARY = ROOT / "ops/causal_checkpoint_translation.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v23_rank1_shared_context_leave_one_head_out_v2_result.json"
EXPECTED = {
    "prior": "fcc61605a5c688905fc243ac9e580a134d6c829e6118b9467afd7afccd044b92",
    "v1_result": "821c97b231d9e0337380e015906f5c74672caa91b0ebd9db492618b5da1a62dc",
    "v1_audit": "6487f738d46ca25a047c3d5f3f14c36d548f3a5f57ed150a1ef98213aee10834",
    "v1_runner": "290343b4ed60e4f5a083a3b450a095927a29593a35d2fb0c945380f6e3019b94",
    "arrays": "e8dde195577c4ff8778ef471eee744469e8f95a346611376f5f964dae6ba215c",
    "grouping": "f9b5c7206c6c67d5bb1096b1f65c230be4327efd5c0372a75d3577c5e6114717",
    "gauge": "757ede89805372cd0634851fc12e0bd8e82d010240c52e8499b2dcb451d43868",
    "library": "32636472a76af0a22a8be5fd8d1e9c44ff671097f34cf31245417365a28255e7",
}
RANK = 1
ROTATION_SEED = 2308
RANDOM_SEED = 2401
RANDOM_DRAWS = 256
BARS = {"instrument": 2e-5, "heldout_capture": .05,
        "projector_overlap": .85, "gram_reduction": .05}
PRICE = {"checkpoint_loads": 0, "model_forwards": 0,
         "sequence_evaluations": 0, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0}
PREDICTION_KEYS = (
    "pred_a_authority_shapes_replay_certificates_gauge_invariance_and_zero_forward_price",
    "pred_b_rank1_context_core_predicts_every_omitted_head_beyond_random",
    "pred_c_three_head_fits_identify_one_stable_physical_direction",
    "pred_d_removing_the_common_direction_reduces_cross_head_context_similarity",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def gram_similarity(left, right):
    numerator = (left.transpose(0, 1) @ right).square().sum()
    left_norm = (left.transpose(0, 1) @ left).square().sum().sqrt()
    right_norm = (right.transpose(0, 1) @ right).square().sum().sqrt()
    return float(numerator / (left_norm * right_norm).clamp_min(1e-30))


def pairwise(values, score):
    matrix = torch.eye(len(values), dtype=torch.float64)
    pairs = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            value = float(score(values[left], values[right]))
            matrix[left, right] = matrix[right, left] = value
            pairs.append(value)
    return matrix, pairs


def projector_overlap(left, right):
    return float((left.transpose(0, 1) @ right).square().sum() / left.shape[1])


def random_capture_q99(value, *, generator):
    directions = torch.randn(
        RANDOM_DRAWS, value.shape[0], generator=generator, dtype=value.dtype
    )
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-30)
    captured = torch.einsum("nd,dp->np", directions, value).square().sum(dim=1)
    captured = captured / value.square().sum().clamp_min(1e-30)
    return float(torch.quantile(captured, .99)), float(captured.mean())


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    return not isinstance(value, (int, float)) or isinstance(value, bool) or math.isfinite(float(value))


def main():
    paths = {
        "prior": PRIOR, "v1_result": V1_RESULT, "v1_audit": V1_AUDIT,
        "v1_runner": V1_RUNNER, "arrays": ARRAYS, "grouping": GROUPING,
        "gauge": GAUGE, "library": LIBRARY,
    }
    observed = {name: sha(path) for name, path in paths.items()}
    dry = {
        "candidate_id": "cross_task.temporal_iswas.v23_rank1_shared_context_leave_one_head_out_v2",
        "dryrun": True, "gpu_accessed": False, "model_loaded": False,
        "queue_touched": False, "authority_ok": observed == EXPECTED,
        "rank": RANK, "arithmetic": "float64", "random_draws": RANDOM_DRAWS,
        "price": PRICE,
    }
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dry, sort_keys=True))
        return
    if observed != EXPECTED:
        raise RuntimeError("rank-one shared-context v2 authority changed")
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")

    started_utc, started = now(), time.perf_counter()
    frozen = np.load(ARRAYS, allow_pickle=False)
    maps = torch.from_numpy(frozen["contracted"]).to(torch.float64)
    routes = frozen["routes"].tolist()
    if maps.shape != (4, 1152, 128) or len(routes) != 4:
        raise RuntimeError("restricted map inventory changed")
    gauge_parent = json.loads(GAUGE.read_text())
    report = translation.shared_context_leave_one_out(maps, rank=RANK)

    rotation_generator = torch.Generator().manual_seed(ROTATION_SEED)
    rotated_maps = []
    for value in maps:
        rotation, _ = torch.linalg.qr(torch.randn(
            128, 128, generator=rotation_generator, dtype=torch.float64
        ))
        rotated_maps.append(value @ rotation)
    rotated = translation.shared_context_leave_one_out(tuple(rotated_maps), rank=RANK)
    gauge_errors = []
    for native_fold, rotated_fold in zip(report["folds"], rotated["folds"]):
        native_basis, rotated_basis = native_fold["basis"], rotated_fold["basis"]
        gauge_errors.append(float(((native_basis @ native_basis.transpose(0, 1))
                                   - (rotated_basis @ rotated_basis.transpose(0, 1))).abs().max()))
        gauge_errors.append(abs(float(native_fold["heldout_captured_energy_fraction"])
                                - float(rotated_fold["heldout_captured_energy_fraction"])))

    pooled = report["pooled_report"]
    reconstruction_error = max(float((common + tail - value).abs().max())
                               for common, tail, value in zip(
                                   pooled["common_maps"], pooled["private_tails"], maps))
    certificate_error = max([float(pooled["certificate_absolute_error"])] +
                            [float(fold["training_certificate_absolute_error"])
                             for fold in report["folds"]])
    raw_gram_matrix, raw_gram_pairs = pairwise(maps, gram_similarity)
    tail_gram_matrix, tail_gram_pairs = pairwise(pooled["private_tails"], gram_similarity)
    raw_gram_median = float(np.median(raw_gram_pairs))
    tail_gram_median = float(np.median(tail_gram_pairs))
    stored_gram_median = float(gauge_parent["summary"]["median_pair_context_gram_similarity"])
    raw_gram_replay_error = abs(raw_gram_median - stored_gram_median)

    fold_bases = [fold["basis"] for fold in report["folds"]]
    fold_overlap_matrix, fold_overlap_pairs = pairwise(fold_bases, projector_overlap)
    random_generator = torch.Generator().manual_seed(RANDOM_SEED)
    fold_reports = {}
    for route, value, fold in zip(routes, maps, report["folds"]):
        random_q99, random_mean = random_capture_q99(value, generator=random_generator)
        capture = float(fold["heldout_captured_energy_fraction"])
        fold_reports[route] = {
            "heldout_captured_energy_fraction": capture,
            "random_capture_q99": random_q99,
            "random_capture_mean": random_mean,
            "exceeds_random_q99": capture > random_q99,
        }

    dtype_live = bool(
        maps.dtype == torch.float64
        and pooled["basis"].dtype == torch.float64
        and all(fold["basis"].dtype == torch.float64 for fold in report["folds"])
        and all(value.dtype == torch.float64 for value in pooled["common_maps"])
        and all(value.dtype == torch.float64 for value in pooled["private_tails"])
    )
    instrument_error = max(
        reconstruction_error, certificate_error, raw_gram_replay_error, max(gauge_errors)
    )
    A = bool(
        observed == EXPECTED and dtype_live and instrument_error <= BARS["instrument"]
        and finite([fold_reports, raw_gram_matrix, tail_gram_matrix, fold_overlap_matrix])
        and all(value == 0 for value in PRICE.values())
    )
    B = bool(all(value["heldout_captured_energy_fraction"] >= BARS["heldout_capture"]
                 and value["exceeds_random_q99"] for value in fold_reports.values()))
    C = bool(min(fold_overlap_pairs) >= BARS["projector_overlap"])
    gram_reduction = raw_gram_median - tail_gram_median
    D = bool(gram_reduction >= BARS["gram_reduction"])
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = (
        "invalid_instrument" if not A else
        "shared_rank1_trunk_plus_private_structure" if B and C and D else
        "shared_rank1_inside_broader_common_geometry" if B and C else
        "no_universal_rank1_shared_context_capability"
    )
    result = {
        "schema": "temporal_iswas_v23_rank1_shared_context_leave_one_head_out_result_v2",
        "candidate_id": dry["candidate_id"], "started_utc": started_utc,
        "finished_utc": now(), "serial_seconds": time.perf_counter() - started,
        "authority_sha256": observed, "routes": routes, "rank": RANK,
        "component_weights": [1.0] * len(routes),
        "instrument": {
            "arithmetic": "float64", "float64_live": dtype_live,
            "reconstruction_max_abs": reconstruction_error,
            "certificate_max_abs": certificate_error,
            "private_gauge_max_abs": max(gauge_errors),
            "raw_gram_replay_abs": raw_gram_replay_error,
        },
        "fold_reports": fold_reports,
        "leave_one_out_projector_overlap": fold_overlap_matrix.tolist(),
        "minimum_leave_one_out_projector_overlap": min(fold_overlap_pairs),
        "raw_context_gram_similarity": raw_gram_matrix.tolist(),
        "tail_context_gram_similarity": tail_gram_matrix.tolist(),
        "raw_context_gram_median": raw_gram_median,
        "tail_context_gram_median": tail_gram_median,
        "context_gram_median_reduction": gram_reduction,
        "pooled_rank1_captured_energy_fraction_by_head":
            dict(zip(routes, map(float, pooled["component_captured_energy_fraction"]))),
        "predictions": predictions, "terminal": terminal, "bars": BARS,
        "price": PRICE, "selected_rank_weight_or_threshold_after_outcome": None,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "instrument", "fold_reports", "minimum_leave_one_out_projector_overlap",
        "raw_context_gram_median", "tail_context_gram_median",
        "context_gram_median_reduction", "predictions", "terminal", "price"
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
