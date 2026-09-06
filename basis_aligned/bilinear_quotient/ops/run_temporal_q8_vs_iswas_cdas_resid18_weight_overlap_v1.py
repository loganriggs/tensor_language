#!/usr/bin/env python3
"""Gauge-invariant cross-task overlap between temporal Q8 writes and is/was cDAS."""

# BQGATE: EXPERIMENT pred_a_exact_weight_construction_shapes_and_invariance pred_b_cross_task_overlap_exceeds_dimension_control pred_c_shared_and_specific_components_are_nontrivial
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1 as family_builder

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1.json"
TEMPORAL_TRANSFER = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_rank8_v11_causal_suffix_transfer_v2_result.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
ISWAS = ROOT / "circuits/followups/tense_auxiliary_is_was_selective_das_resid18_rank1_v1_result.json"
ISWAS_PROGRAM = ROOT / "circuits/followups/tense_auxiliary_is_was_transparent_path_program_release_v1_result.json"
FAMILY_RUNNER = ROOT / "ops/run_temporal_auxiliary_will_had_h3_weight_read_nested_rank_v1.py"
OUT = ROOT / "circuits/followups/temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1_result.json"
CANDIDATE_ID = "cross_task.temporal_q8_vs_iswas_cdas_resid18_weight_overlap_v1"
EXPECTED = {
    "prior": "2838fbd6f1b350b883a679830cada507d190e571b78c36ca9892461e2d42016a",
    "temporal_transfer": "42309b3d0a5bff27b3ff86314e9b03d6d6a054b6ad8944c8a4d32d8d0f3de51c",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "iswas": "36f80c04e4a1b2c6a7e0594126cd71e8781aab987521f8865d7e6de842346c02",
    "iswas_program": "9804a0d0f047f194f6cce3490828c3a6e9525940f8c2b822467bc52176957e98",
    "family_runner": "7133350078536e96b9fa7c740d089bf57b806cca9162d9ad36a1055e1971b410",
}
DIMENSION, RANK, RANDOM_SAMPLES, SEED = 1152, 8, 4096, 20260906
PREDICTION_KEYS = (
    "pred_a_exact_weight_construction_shapes_and_invariance",
    "pred_b_cross_task_overlap_exceeds_dimension_control",
    "pred_c_shared_and_specific_components_are_nontrivial",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def residual_modes(backend, q, gain):
    torch = backend.torch
    model = backend.model
    head_count = int(model.config.n_head)
    width = int(model.config.n_embd // head_count)
    flattened = torch.zeros(RANK, head_count * width, device=backend.device, dtype=torch.float32)
    flattened[:, 3*width:4*width] = q.T
    weight = model.transformer.h[11].attn.c_proj.weight.detach().float()
    modes = backend.F.linear(flattened, weight).T * gain
    explicit = (flattened @ weight.T).T * gain
    wrong = (flattened @ weight).T * gain
    return modes, float((modes-explicit).abs().max()), float((modes-wrong).abs().max())


def main():
    paths = {"prior": PRIOR, "temporal_transfer": TEMPORAL_TRANSFER, "subspace": SUBSPACE,
             "iswas": ISWAS, "iswas_program": ISWAS_PROGRAM, "family_runner": FAMILY_RUNNER}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("cross-task weight-overlap authority changed")
    prior, transfer, subspace, iswas, iswas_program = [json.loads(path.read_text())
        for path in (PRIOR, TEMPORAL_TRANSFER, SUBSPACE, ISWAS, ISWAS_PROGRAM)]
    if (prior.get("candidate_id") != CANDIDATE_ID or transfer.get("terminal") != "screen"
            or iswas.get("terminal") != "screen" or iswas_program.get("terminal") is None):
        raise RuntimeError("authority terminal changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
        "model_loaded": False, "queue_touched": False, "dimension": DIMENSION, "rank": RANK,
        "random_samples": RANDOM_SAMPLES, "seed": SEED, "model_forwards": 0,
        "example_evaluations": 0, "fit_updates": 0, "transformer_backwards": 0,
        "prediction_keys": list(PREDICTION_KEYS)}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    family, _singular, _energy = family_builder.build_family(backend, subspace)
    q = family[RANK]
    gain = math.prod(float(backend.model.transformer.h[layer].lambdas[0].detach().float())
                     for layer in range(12, 18))
    modes, orientation_error, wrong_orientation_gap = residual_modes(backend, q, gain)
    s = torch.linalg.qr(modes, mode="reduced").Q
    orth_error = float((s.T@s-torch.eye(RANK, device=s.device)).abs().max())
    axis_values = np.asarray(iswas["basis"]["values_column_major"], dtype=np.float32)
    axis_hash = hashlib.sha256(axis_values.tobytes()).hexdigest()
    if axis_values.shape != (DIMENSION,) or axis_hash != iswas["basis"]["sha256"]:
        raise RuntimeError("is/was axis artifact mismatch")
    axis = torch.as_tensor(axis_values, device=backend.device).reshape(DIMENSION, 1)
    axis = axis / torch.linalg.vector_norm(axis)
    rho = float(torch.linalg.vector_norm(s.T@axis).square())
    shared_norm, specific_norm = math.sqrt(max(rho, 0.0)), math.sqrt(max(1.0-rho, 0.0))
    generator = torch.Generator(device=backend.device)
    generator.manual_seed(SEED)
    rotation = torch.linalg.qr(torch.randn(RANK, RANK, generator=generator,
                                            device=backend.device), mode="reduced").Q
    rotated_modes, rotated_orientation_error, _wrong = residual_modes(backend, q@rotation, gain)
    s_rot = torch.linalg.qr(rotated_modes, mode="reduced").Q
    projector_error = float((s@s.T-s_rot@s_rot.T).abs().max())
    rho_rot = float(torch.linalg.vector_norm(s_rot.T@axis).square())
    rho_rotation_error = abs(rho-rho_rot)
    rng = np.random.default_rng(SEED)
    random_rhos = rng.beta(RANK/2, (DIMENSION-RANK)/2, size=RANDOM_SAMPLES)
    random_quantiles = {label: float(np.quantile(random_rhos, value))
        for label, value in (("p50", .5), ("p95", .95), ("p99", .99), ("p999", .999))}
    empirical_percentile = float((random_rhos < rho).mean())
    chance = RANK/DIMENSION
    pred_a = bool(modes.shape == (DIMENSION, RANK) and s.shape == (DIMENSION, RANK)
        and orth_error <= 1e-5 and orientation_error <= 1e-6
        and rotated_orientation_error <= 1e-6 and wrong_orientation_gap >= 1e-3
        and projector_error <= 1e-5 and rho_rotation_error <= 1e-5)
    pred_b = bool(rho >= .10 and rho >= 10*chance and rho > random_quantiles["p999"])
    pred_c = bool(shared_norm >= .10 and specific_norm >= .10)
    predictions = {"pred_a_exact_weight_construction_shapes_and_invariance": pred_a,
        "pred_b_cross_task_overlap_exceeds_dimension_control": pred_b,
        "pred_c_shared_and_specific_components_are_nontrivial": pred_c}
    terminal = "invalid" if not pred_a else "screen" if all(predictions.values()) else "null"
    result = {"schema": "temporal_q8_vs_iswas_cdas_resid18_weight_overlap_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter()-started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "frozen_skip_gain": gain,
        "instrument": {"temporal_write_shape": list(modes.shape), "orthonormal_shape": list(s.shape),
            "orthonormality_max_abs": orth_error, "f_linear_orientation_max_abs": orientation_error,
            "wrong_orientation_gap_max_abs": wrong_orientation_gap,
            "gauge_projector_max_abs": projector_error, "gauge_rho_abs_error": rho_rotation_error,
            "iswas_axis_sha256": axis_hash},
        "overlap": {"squared_projection_rho": rho, "shared_component_norm": shared_norm,
            "specific_component_norm": specific_norm, "isotropic_expectation": chance,
            "multiple_of_isotropic_expectation": rho/chance,
            "random_samples": RANDOM_SAMPLES, "random_quantiles": random_quantiles,
            "empirical_random_percentile": empirical_percentile},
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": 0, "example_evaluations": 0, "fit_updates": 0,
                  "transformer_backwards": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "instrument", "overlap",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
