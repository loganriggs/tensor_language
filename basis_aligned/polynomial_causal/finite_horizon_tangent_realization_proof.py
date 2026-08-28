#!/usr/bin/env python3
"""Create-only CPU proof receipt for finite-horizon cut minimality machinery."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

import finite_horizon_tangent_realization as realization


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "finite_horizon_tangent_realization_proof_results.json"


def publish_create_only(value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def orthogonal(width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=torch.float64))
    return q


def time_varying_fixture() -> tuple[dict, dict, dict]:
    """A three-state realization with one unreachable/unobservable dummy dimension."""
    transitions = {
        1: torch.tensor([[0.8, 0.1, 0.0], [-0.2, 0.9, 0.0], [0.0, 0.0, 1.3]],
                        dtype=torch.float64),
        2: torch.tensor([[0.7, -0.3, 0.0], [0.4, 0.6, 0.0], [0.0, 0.0, 0.5]],
                        dtype=torch.float64),
    }
    injections = {
        0: torch.tensor([[1.0, 0.2], [0.1, 0.9], [0.0, 0.0]], dtype=torch.float64),
        1: torch.tensor([[0.3, 0.8], [1.0, -0.2], [0.0, 0.0]], dtype=torch.float64),
        2: torch.tensor([[0.7, -0.1], [0.2, 0.6], [0.0, 0.0]], dtype=torch.float64),
    }
    observations = {
        1: torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.9, 0.0]], dtype=torch.float64),
        2: torch.tensor([[0.6, 0.4, 0.0], [-0.1, 1.0, 0.0]], dtype=torch.float64),
        3: torch.tensor([[0.9, -0.2, 0.0], [0.3, 0.8, 0.0]], dtype=torch.float64),
    }
    blocks = {}
    for output_site in observations:
        for input_site in injections:
            if input_site >= output_site:
                continue
            transport = torch.eye(3, dtype=torch.float64)
            for layer in range(input_site + 1, output_site):
                transport = transitions[layer] @ transport
            blocks[(output_site, input_site)] = (
                observations[output_site] @ transport @ injections[input_site]
            )
    return blocks, {site: 2 for site in injections}, {site: 2 for site in observations}


def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("finite-horizon proof receipt is create-only and already exists")
    blocks, input_dims, output_dims = time_varying_fixture()
    cuts = (1, 2, 3)
    analyses = realization.analyze_all_cuts(
        blocks, input_dims, output_dims, cuts, energy_fraction=0.95, gap_ratio=2.0,
    )
    gauges_in = {site: orthogonal(width, 100 + site) for site, width in input_dims.items()}
    gauges_out = {site: orthogonal(width, 200 + site) for site, width in output_dims.items()}
    transformed = realization.transform_blocks_orthogonal(blocks, gauges_in, gauges_out)
    replay = realization.analyze_all_cuts(
        transformed, input_dims, output_dims, cuts, energy_fraction=0.95, gap_ratio=2.0,
    )
    gauge_max = 0.0
    factor_max = 0.0
    for cut in cuts:
        original_singular = torch.tensor(analyses[str(cut)]["singular_values"])
        replay_singular = torch.tensor(replay[str(cut)]["singular_values"])
        gauge_max = max(gauge_max, float((original_singular - replay_singular).abs().max()))
        matrix, _ = realization.assemble_cut(blocks, input_dims, output_dims, cut)
        left, right, receipt = realization.truncated_factorization(matrix, rank=2)
        factor_max = max(factor_max, float((matrix - left @ right).abs().max()),
                         float(receipt["maximum_absolute_error"]))
    predictions = {
        "A_all_cut_ranks_equal_reachable_observable_dimension_2": all(
            analyses[str(cut)]["exact_cut_rank"] == 2 for cut in cuts
        ),
        "B_unreachable_unobservable_third_state_removed": max(
            analyses[str(cut)]["exact_cut_rank"] for cut in cuts
        ) < 3,
        "C_orthogonal_gauge_spectra_invariant": gauge_max <= 1e-12,
        "D_rank2_cut_factorizations_exact": factor_max <= 1e-12,
    }
    result = {
        "status": "pass" if all(predictions.values()) else "proof_failure",
        "scope": "pure CPU finite-horizon time-varying realization proof fixture",
        "cuts": analyses,
        "gauge_max_singular_error": gauge_max,
        "rank2_factorization_max_error": factor_max,
        "predictions": predictions,
    }
    publish_create_only(result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
