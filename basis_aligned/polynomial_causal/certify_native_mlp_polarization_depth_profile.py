#!/usr/bin/env python3
"""Weight-only polarization-slice spectrum across all 18 native bilinear MLPs."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for source_root in (ROOT, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import bilin18_observed_model_facade as facade
import certify_early_mlp_quadratic_slice_rank as early


RESULT = HERE / "native_mlp_polarization_depth_profile.json"
SITES = tuple(range(18))
RANKS = (512, 768)


def analyze_singular_values(singular: torch.Tensor, rounding: float) -> dict:
    if singular.shape != (early.DIMENSION,) or not torch.isfinite(singular).all() \
            or bool((singular < 0).any()) or rounding <= 0:
        raise ValueError("polarization singular spectrum changed")
    total = torch.linalg.vector_norm(singular)
    output = {
        "numerical_slice_rank_lower_bound": int((singular > rounding).sum()),
        "smallest_singular_value": float(singular[-1]),
        "conservative_float64_roundoff_bound": float(rounding),
    }
    for rank in RANKS:
        tail = torch.linalg.vector_norm(singular[rank:])
        output[f"sigma_{rank + 1}"] = float(singular[rank])
        output[f"sigma_{rank + 1}_over_sigma_1"] = float(singular[rank] / singular[0])
        output[f"best_rank{rank}_relative_frobenius_error_lower_bound"] = float(tail / total)
    return output


def depth_summary(sites: dict[str, dict]) -> dict:
    values = torch.tensor([
        sites[str(site)]["best_rank768_relative_frobenius_error_lower_bound"]
        for site in SITES
    ], dtype=torch.float64)
    shallow_median = float(values[:10].median())
    late_median = float(values[10:].median())
    adjacent_ratio = float(values[10] / values[9])
    group_ratio = late_median / shallow_median
    return {
        "rank768_relative_tail_by_site": values.tolist(),
        "mlp10_over_mlp9_adjacent_ratio": adjacent_ratio,
        "late_10_17_over_shallow_0_9_median_ratio": group_ratio,
        "adjacent_1p20_knee": adjacent_ratio >= 1.20,
        "group_1p20_shift": group_ratio >= 1.20,
        "ruling": "coefficient_slice_supports_depth_knee" if (
            adjacent_ratio >= 1.20 or group_ratio >= 1.20
        ) else "shipped_knee_not_explained_by_e0_coefficient_slice",
    }


def main() -> None:
    if RESULT.exists():
        raise RuntimeError("polarization depth-profile result already exists")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    started = time.time()
    model, checkpoint = facade.load_bilin18(device=torch.device("cpu"), dtype=torch.bfloat16)
    sites = {}
    for site in SITES:
        mlp = model.transformer.h[site].mlp
        matrix = early.symmetric_slice(
            mlp.Down.weight, mlp.Left.weight, mlp.Right.weight, coordinate=0,
        )
        singular = torch.linalg.svdvals(matrix)
        sites[str(site)] = analyze_singular_values(
            singular, early.conservative_roundoff_bound(matrix),
        )
    result = {
        "schema": "native_mlp_polarization_depth_profile_v1",
        "status": "deterministic_weight_analysis_complete",
        "sites": sites, "summary": depth_summary(sites),
        "slice": "second_input_coordinate_e0", "ranks": list(RANKS),
        "checkpoint": checkpoint.__dict__, "runtime_seconds": time.time() - started,
        "claim_boundary": (
            "descriptive exact-coefficient slice; not independent confirmation of the "
            "known shipped knee and not a reachable-state, CE, semantic, edit, or OOD claim"
        ),
    }
    temporary = RESULT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    temporary.replace(RESULT)
    print(json.dumps(result["summary"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
