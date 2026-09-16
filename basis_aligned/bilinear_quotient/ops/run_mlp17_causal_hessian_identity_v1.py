#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_weight_autodiff_identity pred_b_background_invariance pred_c_cross_difference_identity pred_d_factor_agreement pred_e_planted_solver_control pred_f_tensor_symmetry
"""Validate the one-layer DCT/causal-Hessian bridge on native MLP17 weights."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]

import torch

import circuit_fast_screen_managed_runner as managed


RUNNER = Path(__file__).resolve()
PREREG = P / "MLP17_CAUSAL_HESSIAN_IDENTITY_V1_PREREGISTRATION.md"
BINDING = P / "MLP17_CAUSAL_HESSIAN_IDENTITY_V1_BINDING.json"
OUT = P / "MLP17_CAUSAL_HESSIAN_IDENTITY_V1_RESULT.json"
SEED = 202609160400
INPUT_DIM = 16
OUTPUT_DIM = 8
RANK = 4
PRICE = {
    "checkpoint_loads": 1,
    "language_model_forwards": 0,
    "native_subspace_hessians": 2,
    "finite_cross_differences": 8,
    "factor_solver_calls": 3,
    "factor_rank": 4,
    "fits_to_model_outputs": 0,
    "parameter_updates": 0,
}
PREDICATES = {
    "pred_a_weight_autodiff_identity": None,
    "pred_b_background_invariance": None,
    "pred_c_cross_difference_identity": None,
    "pred_d_factor_agreement": None,
    "pred_e_planted_solver_control": None,
    "pred_f_tensor_symmetry": None,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(actual, expected):
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


def load_bound():
    binding = json.loads(BINDING.read_text())
    files = {
        "preregistration": PREREG,
        "fastload": RUNNER.parent / "fastload.py",
        "model_source": ROOT / "jacclust/tt_model.py",
    }
    if binding["files"] != {name: sha(path) for name, path in files.items()} or binding["price"] != PRICE:
        raise ValueError("bound input or price changed")
    return binding


def plan():
    load_bound()
    return {
        "schema": "mlp17_causal_hessian_identity_v1_plan",
        "model_loaded": False,
        "gpu_accessed": False,
        "queue_touched": False,
        "layer": 17,
        "input_probe_dimension": INPUT_DIM,
        "output_probe_dimension": OUTPUT_DIM,
        "factor_rank": RANK,
        "seed": SEED,
        "price": PRICE,
        "binding_sha256": sha(BINDING),
    }


def projected_function(theta, background, input_basis, output_basis, left, right, down, bias):
    state = background + input_basis @ theta
    hidden = (left @ state) * (right @ state)
    return output_basis.T @ (down @ hidden + bias)


def autodiff_tensor(background, input_basis, output_basis, left, right, down, bias):
    origin = torch.zeros(INPUT_DIM, device=background.device, dtype=torch.float64)
    tensors = []
    for output_index in range(OUTPUT_DIM):
        fn = lambda theta: projected_function(theta, background, input_basis, output_basis, left, right, down, bias)[output_index]
        tensors.append(torch.autograd.functional.hessian(fn, origin, vectorize=True))
    return torch.stack(tensors)


def canonical(vector):
    index = int(vector.abs().argmax())
    return -vector if vector[index] < 0 else vector


def orthogonalized_als(tensor, rank, seed, restarts=12, sweeps=30):
    generator = torch.Generator(device=tensor.device).manual_seed(seed)
    residual = tensor.clone()
    input_factors = []
    output_factors = []
    amplitudes = []
    dimension = tensor.shape[1]
    eye = torch.eye(dimension, device=tensor.device, dtype=tensor.dtype)
    for _ in range(rank):
        if input_factors:
            basis = torch.stack(input_factors, dim=1)
            projector = eye - basis @ basis.T
        else:
            projector = eye
        best = None
        for _restart in range(restarts):
            vector = projector @ torch.randn(dimension, generator=generator, device=tensor.device, dtype=tensor.dtype)
            vector /= vector.norm().clamp_min(1e-30)
            for _sweep in range(sweeps):
                output = torch.einsum("oij,i,j->o", residual, vector, vector)
                output /= output.norm().clamp_min(1e-30)
                matrix = torch.einsum("o,oij->ij", output, residual)
                matrix = projector @ ((matrix + matrix.T) / 2) @ projector
                eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
                vector = eigenvectors[:, int(eigenvalues.abs().argmax())]
                vector = projector @ vector
                vector /= vector.norm().clamp_min(1e-30)
                vector = canonical(vector)
            response = torch.einsum("oij,i,j->o", residual, vector, vector)
            amplitude = response.norm()
            output = response / amplitude.clamp_min(1e-30)
            candidate = (float(amplitude), vector, output, amplitude)
            if best is None or candidate[0] > best[0]:
                best = candidate
        _, vector, output, amplitude = best
        input_factors.append(vector)
        output_factors.append(output)
        amplitudes.append(amplitude)
        residual = residual - amplitude * torch.einsum("o,i,j->oij", output, vector, vector)
    reconstructed = sum(
        amplitudes[index] * torch.einsum("o,i,j->oij", output_factors[index], input_factors[index], input_factors[index])
        for index in range(rank)
    )
    return {
        "input": torch.stack(input_factors),
        "output": torch.stack(output_factors),
        "amplitude": torch.stack(amplitudes),
        "reconstruction_relative_error": rel(reconstructed, tensor),
    }


def match_factors(reference, candidate):
    best = None
    for permutation in itertools.permutations(range(RANK)):
        input_cosines = [float(reference["input"][index] @ candidate["input"][permutation[index]]) for index in range(RANK)]
        output_cosines = [float(reference["output"][index] @ candidate["output"][permutation[index]]) for index in range(RANK)]
        score = sum(abs(value) for value in input_cosines + output_cosines)
        if best is None or score > best[0]:
            amplitude_errors = [
                abs(float(candidate["amplitude"][permutation[index]] / reference["amplitude"][index] - 1))
                for index in range(RANK)
            ]
            best = (score, permutation, input_cosines, output_cosines, amplitude_errors)
    return {
        "permutation": list(best[1]),
        "input_absolute_cosines": [abs(value) for value in best[2]],
        "output_absolute_cosines": [abs(value) for value in best[3]],
        "amplitude_relative_errors": best[4],
    }


@torch.no_grad()
def planted_control(device):
    generator = torch.Generator(device=device).manual_seed(SEED + 1)
    input_q, _ = torch.linalg.qr(torch.randn(INPUT_DIM, RANK, generator=generator, device=device, dtype=torch.float64))
    output_q, _ = torch.linalg.qr(torch.randn(OUTPUT_DIM, RANK, generator=generator, device=device, dtype=torch.float64))
    amplitudes = torch.tensor([4.0, 3.0, 2.0, 1.0], device=device, dtype=torch.float64)
    tensor = sum(amplitudes[index] * torch.einsum("o,i,j->oij", output_q[:, index], input_q[:, index], input_q[:, index]) for index in range(RANK))
    solved = orthogonalized_als(tensor, RANK, SEED + 2)
    planted = {"input": input_q.T, "output": output_q.T, "amplitude": amplitudes}
    report = match_factors(planted, solved)
    report["reconstruction_relative_error"] = solved["reconstruction_relative_error"]
    return report


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast

    model = load_model_fast().cuda().eval()
    device = next(model.parameters()).device
    mlp = model.transformer.h[17].mlp
    left = mlp.Left.weight.detach().double()
    right = mlp.Right.weight.detach().double()
    down = mlp.Down.weight.detach().double()
    bias = mlp.Down_bias.detach().double()
    generator = torch.Generator(device=device).manual_seed(SEED)
    input_basis, _ = torch.linalg.qr(torch.randn(1152, INPUT_DIM, generator=generator, device=device, dtype=torch.float64))
    output_basis, _ = torch.linalg.qr(torch.randn(1152, OUTPUT_DIM, generator=generator, device=device, dtype=torch.float64))
    background0 = torch.randn(1152, generator=generator, device=device, dtype=torch.float64)
    background1 = torch.randn(1152, generator=generator, device=device, dtype=torch.float64)

    left_probe = left @ input_basis
    right_probe = right @ input_basis
    output_down = output_basis.T @ down
    analytic = torch.einsum("oh,hi,hj->oij", output_down, left_probe, right_probe)
    analytic = analytic + analytic.transpose(1, 2)

    with torch.enable_grad():
        autodiff0 = autodiff_tensor(background0, input_basis, output_basis, left, right, down, bias)
        autodiff1 = autodiff_tensor(background1, input_basis, output_basis, left, right, down, bias)

    tensor_relative_error = rel(autodiff0, analytic)
    tensor_maximum_absolute_error = float((autodiff0 - analytic).abs().max())
    background_relative_error = rel(autodiff1, autodiff0)
    cross_errors = []
    for _ in range(PRICE["finite_cross_differences"]):
        a = torch.randn(INPUT_DIM, generator=generator, device=device, dtype=torch.float64)
        b = torch.randn(INPUT_DIM, generator=generator, device=device, dtype=torch.float64)
        zero = torch.zeros(INPUT_DIM, device=device, dtype=torch.float64)
        observed = (
            projected_function(a + b, background0, input_basis, output_basis, left, right, down, bias)
            - projected_function(a, background0, input_basis, output_basis, left, right, down, bias)
            - projected_function(b, background0, input_basis, output_basis, left, right, down, bias)
            + projected_function(zero, background0, input_basis, output_basis, left, right, down, bias)
        )
        expected = torch.einsum("oij,i,j->o", analytic, a, b)
        cross_errors.append(rel(observed, expected))

    analytic_factors = orthogonalized_als(analytic, RANK, SEED + 3)
    autodiff_factors = orthogonalized_als(autodiff0, RANK, SEED + 3)
    factor_report = match_factors(analytic_factors, autodiff_factors)
    factor_report["analytic_reconstruction_relative_error"] = analytic_factors["reconstruction_relative_error"]
    factor_report["autodiff_reconstruction_relative_error"] = autodiff_factors["reconstruction_relative_error"]
    planted_report = planted_control(device)
    analytic_symmetry = float((analytic - analytic.transpose(1, 2)).abs().max())
    autodiff_symmetry = float((autodiff0 - autodiff0.transpose(1, 2)).abs().max())

    predictions = {
        "pred_a_weight_autodiff_identity": bool(tensor_relative_error <= 1e-10 and tensor_maximum_absolute_error <= 1e-9),
        "pred_b_background_invariance": bool(background_relative_error <= 1e-10),
        "pred_c_cross_difference_identity": bool(max(cross_errors) <= 1e-10),
        "pred_d_factor_agreement": bool(min(factor_report["input_absolute_cosines"] + factor_report["output_absolute_cosines"]) >= .999999 and max(factor_report["amplitude_relative_errors"]) <= 1e-6 and abs(factor_report["analytic_reconstruction_relative_error"] - factor_report["autodiff_reconstruction_relative_error"]) <= 1e-8),
        "pred_e_planted_solver_control": bool(min(planted_report["input_absolute_cosines"] + planted_report["output_absolute_cosines"]) >= .999 and max(planted_report["amplitude_relative_errors"]) <= 1e-3),
        "pred_f_tensor_symmetry": bool(max(analytic_symmetry, autodiff_symmetry) <= 1e-10),
    }
    result = {
        "schema": "mlp17_causal_hessian_identity_v1_result",
        "terminal": "mlp17_causal_hessian_identity" if all(predictions.values()) else "valid_identity_null",
        "predictions": predictions,
        "weight_autodiff_relative_error": tensor_relative_error,
        "weight_autodiff_maximum_absolute_error": tensor_maximum_absolute_error,
        "background_invariance_relative_error": background_relative_error,
        "cross_difference_relative_errors": cross_errors,
        "analytic_symmetry_maximum_absolute_error": analytic_symmetry,
        "autodiff_symmetry_maximum_absolute_error": autodiff_symmetry,
        "factor_agreement": factor_report,
        "planted_solver_control": planted_report,
        "price": PRICE,
        "seed": SEED,
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "binding_sha256": sha(BINDING),
        "runner_sha256": sha(RUNNER),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Weight-only FP64 identity certificate for the fixed-normalizer raw bilinear MLP17 slice in frozen random subspaces; no behavioral circuit or deep-slice claim.",
    }
    managed.atomic_create_json(OUT, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
