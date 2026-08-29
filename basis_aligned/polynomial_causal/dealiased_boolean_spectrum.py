#!/usr/bin/env python3
"""Orthogonal Boolean spectrum of the sealed five-action de-alias factorial.

This is an explicitly post-outcome diagnostic.  It gives an exact Parseval
decomposition under the uniform distribution on the 32 physical intervention masks;
it does not promote a causal interface or authorize a new model outcome.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import broad_mlp_suffix_dealias_v1 as assay
import broad_mlp_suffix_dealias_v1_lifecycle as lifecycle
import score_broad_mlp_suffix_dealias_v1 as scorer
import score_early_mlp_context_cross_v1 as parent_scorer


VARIABLES = ("MLP0", "MLP1", "MLP2", "A3:8", "M3:8")
DIMENSION = len(VARIABLES)
SIZE = 1 << DIMENSION
ROLE_NAMES = assay.ROLE_NAMES
BOOTSTRAP_DRAWS = 1_000
BOOTSTRAP_SEEDS = {"skip7000": 2026082905, "skip11000": 2026082906}
CURVE_K = (1, 2, 4, 8, 12, 16, 24, 31)


def fwht(values: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform in binary-mask order."""

    output = np.asarray(values, dtype=np.float64).copy()
    if output.ndim != 1 or len(output) == 0 or len(output) & (len(output) - 1) or not (
        np.all(np.isfinite(output))
    ):
        raise ValueError("Walsh input must be a finite power-of-two vector")
    width = 1
    while width < len(output):
        for start in range(0, len(output), 2 * width):
            left = output[start:start + width].copy()
            right = output[start + width:start + 2 * width].copy()
            output[start:start + width] = left + right
            output[start + width:start + 2 * width] = left - right
        width *= 2
    return output


def coefficients(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return fwht(values) / len(values)


def reconstruct(coefficients_: np.ndarray) -> np.ndarray:
    return fwht(np.asarray(coefficients_, dtype=np.float64))


def subset_label(mask: int) -> str:
    if type(mask) is not int or not 0 <= mask < SIZE:
        raise ValueError("subset mask is outside the five-action cube")
    return "constant" if mask == 0 else "*".join(
        variable for bit, variable in enumerate(VARIABLES) if mask & (1 << bit)
    )


def build_set_function(cost: dict[str, np.ndarray]) -> np.ndarray:
    """Map E/A/M/AM vectors into binary order (early bits, A bit, M bit)."""

    if set(cost) != {"e", "a", "m", "am"} or any(
        np.asarray(value).shape != (8,) or not np.all(np.isfinite(value))
        for value in cost.values()
    ):
        raise ValueError("cost arms must be four finite eight-prefix vectors")
    output = np.empty(SIZE, dtype=np.float64)
    for early in range(8):
        for attention in (0, 1):
            for mlp in (0, 1):
                arm = ("e", "a", "m", "am")[attention + 2 * mlp]
                output[early | (attention << 3) | (mlp << 4)] = cost[arm][early]
    return output


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(left) - np.asarray(right)))))


def spectrum_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (SIZE,) or not np.all(np.isfinite(values)):
        raise ValueError("five-action response must contain exactly 32 finite cells")
    coeff = coefficients(values)
    replay = reconstruct(coeff)
    if not np.allclose(replay, values, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Walsh transform failed exact reconstruction")
    variance = float(np.sum(np.square(coeff[1:])))
    empirical_variance = float(np.mean(np.square(values - np.mean(values))))
    if not np.isclose(variance, empirical_variance, atol=1e-12, rtol=1e-12):
        raise RuntimeError("Parseval variance identity failed")
    degree_energy = {
        str(degree): float(sum(
            coeff[mask] ** 2 for mask in range(SIZE) if mask.bit_count() == degree
        )) for degree in range(1, DIMENSION + 1)
    }
    interaction_energy = sum(value for degree, value in degree_energy.items() if int(degree) >= 2)
    suffix_am = [mask for mask in range(SIZE) if mask & 8 and mask & 16]
    ordered = sorted(range(1, SIZE), key=lambda mask: (-abs(coeff[mask]), mask))
    curve = {}
    baseline = float(np.sqrt(variance))
    for k in CURVE_K:
        support = ordered[:k]
        kept = np.zeros(SIZE, dtype=np.float64)
        kept[0] = coeff[0]
        kept[support] = coeff[support]
        prediction = reconstruct(kept)
        rmse = _rmse(values, prediction)
        curve[str(k)] = {
            "rmse": rmse,
            "nre_to_constant": rmse / baseline if baseline > 0 else None,
            "r2_to_mean": 1.0 - rmse ** 2 / variance if variance > 0 else None,
            "max_abs_error": float(np.max(np.abs(values - prediction))),
        }
    return {
        "constant": float(coeff[0]),
        "variance": variance,
        "degree_energy": degree_energy,
        "degree_energy_fraction": {
            degree: value / variance if variance > 0 else None
            for degree, value in degree_energy.items()
        },
        "interaction_energy_fraction": interaction_energy / variance if variance > 0 else None,
        "attention_mlp_coupling_energy_fraction": (
            float(sum(coeff[mask] ** 2 for mask in suffix_am)) / variance
            if variance > 0 else None
        ),
        "coefficients": coeff.tolist(),
        "top_terms": [
            {
                "mask": mask, "term": subset_label(mask),
                "degree": mask.bit_count(), "coefficient": float(coeff[mask]),
                "variance_fraction": float(coeff[mask] ** 2 / variance) if variance > 0 else None,
            } for mask in ordered[:16]
        ],
        "best_k_term_curve": curve,
        "exact_reconstruction_max_abs": float(np.max(np.abs(values - replay))),
    }


def transfer_curve(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    source_coeff, target_coeff = coefficients(source), coefficients(target)
    order = sorted(range(1, SIZE), key=lambda mask: (-abs(source_coeff[mask]), mask))
    target_scale = float(np.sqrt(np.mean(np.square(target - np.mean(target)))))
    output = {}
    for k in CURVE_K:
        support = order[:k]
        direct = np.zeros(SIZE, dtype=np.float64)
        direct[0], direct[support] = source_coeff[0], source_coeff[support]
        refit = np.zeros(SIZE, dtype=np.float64)
        refit[0], refit[support] = target_coeff[0], target_coeff[support]
        direct_prediction, refit_prediction = reconstruct(direct), reconstruct(refit)
        direct_rmse, refit_rmse = _rmse(target, direct_prediction), _rmse(
            target, refit_prediction,
        )
        output[str(k)] = {
            "direct_source_values_nre": direct_rmse / target_scale if target_scale > 0 else None,
            "target_refit_on_source_support_nre": (
                refit_rmse / target_scale if target_scale > 0 else None
            ),
            "support": support,
        }
    return output


def bootstrap_support(data: assay.RoleArrays, *, field: str, k: int = 8) -> dict[str, Any]:
    if field not in {"ce", "top1"} or type(k) is not int or not 0 < k < SIZE:
        raise ValueError("bootstrap field or support size changed")
    rng = np.random.default_rng(BOOTSTRAP_SEEDS[data.role])
    counts = np.zeros(SIZE, dtype=np.int64)
    degree_fraction = np.empty((BOOTSTRAP_DRAWS, DIMENSION), dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, data.document_count, size=data.document_count)
        multiplicity = np.bincount(sampled, minlength=data.document_count)
        values = build_set_function(assay.aggregate(data, multiplicity))
        coeff = coefficients(values)
        support = sorted(range(1, SIZE), key=lambda mask: (-abs(coeff[mask]), mask))[:k]
        counts[support] += 1
        variance = float(np.sum(np.square(coeff[1:])))
        for degree in range(1, DIMENSION + 1):
            energy = float(sum(
                coeff[mask] ** 2 for mask in range(1, SIZE)
                if mask.bit_count() == degree
            ))
            degree_fraction[draw, degree - 1] = energy / variance if variance > 0 else np.nan
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEEDS[data.role],
        "top_k": k,
        "selection_frequency": [
            {
                "mask": mask, "term": subset_label(mask),
                "frequency": float(counts[mask] / BOOTSTRAP_DRAWS),
            } for mask in sorted(range(1, SIZE), key=lambda mask: (-counts[mask], mask))
        ],
        "degree_energy_fraction_interval": {
            str(degree): {
                "q025": float(np.quantile(degree_fraction[:, degree - 1], 0.025)),
                "q975": float(np.quantile(degree_fraction[:, degree - 1], 0.975)),
            } for degree in range(1, DIMENSION + 1)
        },
    }


def load_arrays() -> dict[str, dict[str, assay.RoleArrays]]:
    old, _old_receipt = parent_scorer.load_terminal_bundles(
        lifecycle.PARENT_PATHS, require_authoritative=True,
    )
    new, authorities, _new_receipt = scorer.load_new_terminal_bundles(
        lifecycle.output_paths(), require_authoritative=True,
    )
    parent_authority = lifecycle.parent_authority()
    parent_receipt_sha = lifecycle.file_sha256(lifecycle.PARENT_PATHS.receipt)
    output = {"ce": {}, "top1": {}}
    for role in ROLE_NAMES:
        arguments = {
            "role": role, "old_bundle": old[role], "new_bundle": new[role],
            "new_authority": authorities[role], "parent_authority": parent_authority,
            "parent_receipt_file_sha256": parent_receipt_sha,
        }
        output["ce"][role] = scorer.join_ce_role_arrays(**arguments)
        output["top1"][role] = scorer.join_top1_role_arrays(**arguments)
    return output


def analyze() -> dict[str, Any]:
    arrays = load_arrays()
    result: dict[str, Any] = {
        "status": "post_outcome_descriptive_no_promotion",
        "basis": "uniform_orthonormal_walsh_on_five_physical_action_bits",
        "variables": list(VARIABLES),
        "roles": {},
        "transfer": {},
    }
    values: dict[str, dict[str, np.ndarray]] = {"ce": {}, "top1": {}}
    for field in ("ce", "top1"):
        result["roles"][field] = {}
        for role in ROLE_NAMES:
            cost = assay.aggregate(arrays[field][role])
            values[field][role] = build_set_function(cost)
            result["roles"][field][role] = {
                "spectrum": spectrum_summary(values[field][role]),
                "bootstrap": bootstrap_support(arrays[field][role], field=field),
            }
        left, right = ROLE_NAMES
        result["transfer"][field] = {
            f"{left}_to_{right}": transfer_curve(values[field][left], values[field][right]),
            f"{right}_to_{left}": transfer_curve(values[field][right], values[field][left]),
            "coefficient_correlation": float(np.corrcoef(
                coefficients(values[field][left])[1:], coefficients(values[field][right])[1:]
            )[0, 1]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = analyze()
    text = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(text, end="")
    else:
        arguments.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
