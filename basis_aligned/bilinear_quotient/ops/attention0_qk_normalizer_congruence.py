"""RUNG 435 -- algebraic common-congruence falsifier for attention0 normalizers.

Frozen registration:
  polynomial_causal/ATTENTION0_QK_NORMALIZER_CONGRUENCE_PREREGISTRATION.md
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path("/workspace/tensor_language")
BQ = ROOT / "basis_aligned/bilinear_quotient"
POLY = ROOT / "basis_aligned/polynomial_causal"
OPS = BQ / "ops"
OUT = BQ / "attention0_qk_normalizer_congruence_results.json"
PREREG = POLY / "ATTENTION0_QK_NORMALIZER_CONGRUENCE_PREREGISTRATION.md"

D = 1_152
N_HEAD = 9
HD = 128
MAPS = ("c_q", "c_k", "c_q2", "c_k2")
MIXTURE_SEEDS = tuple(43_500 + index for index in range(8))
CONTROL_SEEDS = tuple(435_100 + index for index in range(8))


def _generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _aggregate(factors: list[torch.Tensor]) -> torch.Tensor:
    result = torch.zeros((D, D), dtype=torch.float64, device=factors[0].device)
    for factor in factors:
        result.addmm_(factor, factor.T, beta=1.0, alpha=1.0 / len(factors))
    return result


def _whiten(factors: list[torch.Tensor]) -> tuple[list[torch.Tensor], dict]:
    aggregate = _aggregate(factors)
    eigenvalues, eigenvectors = torch.linalg.eigh(aggregate)
    inverse_root = (eigenvectors * eigenvalues.rsqrt().unsqueeze(0)) @ eigenvectors.T
    whitened = [inverse_root @ factor for factor in factors]
    mean_whitened = _aggregate(whitened)
    identity = torch.eye(D, dtype=torch.float64, device=aggregate.device)
    residual = torch.linalg.norm(mean_whitened - identity) / torch.linalg.norm(identity)

    reconstruction_max = 0.0
    for factor, white in zip(factors, whitened):
        direct = inverse_root @ (factor @ factor.T) @ inverse_root
        reconstructed = white @ white.T
        error = torch.linalg.norm(direct - reconstructed) / torch.linalg.norm(direct)
        reconstruction_max = max(reconstruction_max, float(error))
    return whitened, {
        "aggregate_min_eigenvalue": float(eigenvalues[0]),
        "aggregate_max_eigenvalue": float(eigenvalues[-1]),
        "aggregate_min_to_max_ratio": float(eigenvalues[0] / eigenvalues[-1]),
        "whitening_relative_frobenius_residual": float(residual),
        "factor_reconstruction_max_relative_frobenius": reconstruction_max,
    }


def _commutators(factors: list[torch.Tensor]) -> dict:
    grams = [factor.T @ factor for factor in factors]
    norms_squared = [float((gram * gram).sum()) for gram in grams]
    values = []
    for left in range(len(factors)):
        for right in range(left + 1, len(factors)):
            cross = factors[left].T @ factors[right]
            trace_a2b2 = torch.trace(grams[left] @ cross @ grams[right] @ cross.T)
            cross_square = cross @ cross.T
            trace_abab = (cross_square * cross_square.T).sum()
            squared = torch.clamp(2.0 * (trace_a2b2 - trace_abab), min=0.0)
            denominator = math.sqrt(norms_squared[left] * norms_squared[right])
            values.append(float(torch.sqrt(squared)) / max(denominator, 1e-300))
    ordered = sorted(values)
    return {
        "count": len(values),
        "minimum": ordered[0],
        "p10": ordered[int(.10 * (len(ordered) - 1))],
        "median": ordered[len(ordered) // 2],
        "p90": ordered[int(.90 * (len(ordered) - 1))],
        "maximum": ordered[-1],
        "mean": sum(values) / len(values),
    }


def _diagonalization_residual(factors: list[torch.Tensor]) -> dict:
    grams = [factor.T @ factor for factor in factors]
    total_energy = sum(float((gram * gram).sum()) for gram in grams)
    candidates = []
    device = factors[0].device
    for seed in MIXTURE_SEEDS:
        coefficients = torch.rand(
            len(factors), dtype=torch.float64, device=device,
            generator=_generator(seed, device)) + .25
        mixture = torch.zeros((D, D), dtype=torch.float64, device=device)
        for coefficient, factor in zip(coefficients, factors):
            mixture.addmm_(factor, factor.T, beta=1.0, alpha=float(coefficient))
        _, basis = torch.linalg.eigh(mixture)
        diagonal_energy = 0.0
        for factor in factors:
            rotated_factor = basis.T @ factor
            diagonal = rotated_factor.square().sum(1)
            diagonal_energy += float(diagonal.square().sum())
        residual = max(0.0, 1.0 - diagonal_energy / total_energy)
        candidates.append({"seed": seed, "off_diagonal_energy_fraction": residual})
    best = min(candidates, key=lambda item: item["off_diagonal_energy_fraction"])
    return {"best": best, "all": candidates}


def _metrics(factors: list[torch.Tensor]) -> dict:
    whitened, instrument = _whiten(factors)
    return {
        "instrument": instrument,
        "commutator": _commutators(whitened),
        "joint_diagonalization": _diagonalization_residual(whitened),
    }


def _union_range(aggregate_eigenvalues: torch.Tensor) -> dict:
    descending = aggregate_eigenvalues.flip(0).clamp_min(0)
    largest = descending[0]
    mass = descending / descending.sum()
    cumulative = mass.cumsum(0)
    energy_ranks = {}
    for target in (.90, .95, .99, .999):
        energy_ranks[str(target)] = int(
            torch.searchsorted(cumulative, torch.tensor(target, device=cumulative.device)).item() + 1)
    numerical = {}
    for singular_relative in (1e-6, 1e-8, 1e-10):
        numerical[str(singular_relative)] = int(
            (descending / largest >= singular_relative ** 2).sum().item())
    return {"numerical_rank_by_relative_singular_threshold": numerical,
            "squared_singular_energy_rank": energy_ranks}


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert PREREG.exists() and len(MAPS) * N_HEAD == 36
        assert len(MIXTURE_SEEDS) == 8 and len(CONTROL_SEEDS) == 8
        print("ATTENTION0 NORMALIZER CONGRUENCE | dry run: 36 PSD slices, 8 matched controls")
        return

    started = time.time()
    sys.path[:0] = [str(POLY), str(OPS), str(BQ)]
    import bilin18_observed_model_facade as facade

    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32)
    model.eval()
    block0 = model.transformer.h[0]
    device = torch.device("cuda")

    raw_factors = []
    labels = []
    for map_name in MAPS:
        weight = getattr(block0.attn, map_name).weight.detach().to(torch.float64)
        blocks = weight.reshape(N_HEAD, HD, D)
        for head in range(N_HEAD):
            raw_factors.append(blocks[head].T.contiguous() / math.sqrt(HD))
            labels.append(f"{map_name}.h{head}")

    raw_aggregate = _aggregate(raw_factors)
    raw_eigenvalues = torch.linalg.eigvalsh(raw_aggregate)
    union = _union_range(raw_eigenvalues)
    real = _metrics(raw_factors)

    base_trace = [float((factor * factor).sum()) for factor in raw_factors]
    base_frobenius = [float(((factor.T @ factor) ** 2).sum().sqrt()) for factor in raw_factors]
    controls = []
    preservation_max = 0.0
    for seed in CONTROL_SEEDS:
        generator = _generator(seed, device)
        permuted = []
        for index, factor in enumerate(raw_factors):
            permutation = torch.randperm(D, device=device, generator=generator)
            changed = factor[permutation]
            permuted.append(changed)
            changed_trace = float((changed * changed).sum())
            changed_frobenius = float(((changed.T @ changed) ** 2).sum().sqrt())
            preservation_max = max(
                preservation_max,
                abs(changed_trace - base_trace[index]) / max(base_trace[index], 1e-300),
                abs(changed_frobenius - base_frobenius[index]) / max(base_frobenius[index], 1e-300))
        control = _metrics(permuted)
        control["seed"] = seed
        controls.append(control)

    control_commutators = [item["commutator"]["median"] for item in controls]
    control_diagonalization = [
        item["joint_diagonalization"]["best"]["off_diagonal_energy_fraction"]
        for item in controls]
    real_commutator = real["commutator"]["median"]
    real_diagonalization = real["joint_diagonalization"]["best"]["off_diagonal_energy_fraction"]
    minimum_control_commutator = min(control_commutators)
    minimum_control_diagonalization = min(control_diagonalization)
    median_control_commutator = float(torch.tensor(control_commutators).median())
    median_control_diagonalization = float(torch.tensor(control_diagonalization).median())

    pred_a = bool(
        real["instrument"]["aggregate_min_to_max_ratio"] >= 1e-8
        and real["instrument"]["whitening_relative_frobenius_residual"] <= 1e-8
        and real["instrument"]["factor_reconstruction_max_relative_frobenius"] <= 1e-10
        and preservation_max <= 1e-10
        and real["commutator"]["count"] == 630
        and len(controls) == 8
        and all(math.isfinite(value) for value in (
            control_commutators + control_diagonalization
            + [real_commutator, real_diagonalization])))
    pred_b = bool(
        union["numerical_rank_by_relative_singular_threshold"]["1e-08"] > 256
        and union["squared_singular_energy_rank"]["0.99"] > 256)
    control_gap_commutator = real_commutator / minimum_control_commutator
    control_gap_diagonalization = real_diagonalization / minimum_control_diagonalization
    pred_c = bool(
        real_commutator <= .5 * minimum_control_commutator
        and real_diagonalization <= .5 * minimum_control_diagonalization)
    pred_d = bool(
        not pred_c
        and (real_commutator <= .9 * minimum_control_commutator
             or real_diagonalization <= .9 * minimum_control_diagonalization))
    strong_null = bool(
        real_commutator >= median_control_commutator
        and real_diagonalization >= median_control_diagonalization)

    output = {
        "schema": "attention0_qk_normalizer_congruence_v1",
        "status": "complete" if pred_a else "instrument_invalid",
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint),
        "dimensions": {"residual": D, "head_width": HD, "slices": len(raw_factors)},
        "labels": labels,
        "union_range": union,
        "real": real,
        "controls": controls,
        "control_summary": {
            "commutator_medians": control_commutators,
            "joint_diagonalization_best_residuals": control_diagonalization,
            "minimum_commutator_median": minimum_control_commutator,
            "median_commutator_median": median_control_commutator,
            "minimum_joint_diagonalization_residual": minimum_control_diagonalization,
            "median_joint_diagonalization_residual": median_control_diagonalization,
            "max_slice_spectrum_preservation_relative_error": preservation_max,
        },
        "comparisons": {
            "real_to_min_control_commutator_ratio": control_gap_commutator,
            "real_to_min_control_diagonalization_ratio": control_gap_diagonalization,
        },
        "predictions": {
            'pred_a_instrument': pred_a,
            'pred_b_exact_r256_common_square_impossible': pred_b,
            'pred_c_useful_common_congruence': pred_c,
            'pred_d_weaker_alignment_above_null': pred_d,
            "strong_null": strong_null,
        },
        "scope": "algebraic identification only; no token behavior, semantics, or adoption",
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "predictions": output["predictions"],
        "union_range": union,
        "real_commutator_median": real_commutator,
        "control_commutator_medians": control_commutators,
        "real_best_diagonalization_residual": real_diagonalization,
        "control_best_diagonalization_residuals": control_diagonalization,
        "output": str(OUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
