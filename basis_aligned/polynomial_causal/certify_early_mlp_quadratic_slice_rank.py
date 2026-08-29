#!/usr/bin/env python3
"""Numerical slice-rank lower bounds for the exact early bilinear MLP tensors.

For q(x)=D[(Lx)⊙(Rx)], its symmetric polarization tensor has the slice
A_y = 1/2 D[diag(Ry)L + diag(Ly)R].  An r-product quadratic program has
rank(A_y) <= r because each product contributes one output-vector outer product.
Hence sigma_513(A_y)>0 certifies r>=513, while the Eckart--Young tail after
singular value 512 lower-bounds every r=512 slice error.
"""

from __future__ import annotations

import hashlib
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

RESULT = HERE / "early_mlp_quadratic_slice_rank_certificate.json"
SITES = (0, 1, 2)
DIMENSION = 1152
NATIVE_PRODUCTS = 4608
TESTED_PRODUCTS = 512
SLICE_RANK_LIMIT = TESTED_PRODUCTS


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def symmetric_slice(down: torch.Tensor, left: torch.Tensor,
                    right: torch.Tensor, coordinate: int = 0) -> torch.Tensor:
    """Return the exact-coefficient float64 coordinate slice A_e_coordinate."""
    if left.shape != right.shape or left.shape != (NATIVE_PRODUCTS, DIMENSION) \
            or down.shape != (DIMENSION, NATIVE_PRODUCTS) \
            or not 0 <= coordinate < DIMENSION:
        raise ValueError("native bilinear factor shape or coordinate changed")
    # BF16 checkpoint coefficients convert exactly to float64.  e_j avoids an
    # additional random-vector/normalization approximation in the certificate.
    l = left.double(); r = right.double(); d = down.double()
    first = (d * r[:, coordinate].unsqueeze(0)) @ l
    second = (d * l[:, coordinate].unsqueeze(0)) @ r
    return 0.5 * (first + second)


def conservative_roundoff_bound(matrix: torch.Tensor) -> float:
    """Very conservative SVD backward-error allowance for a float64 1152 matrix."""
    unit = torch.finfo(torch.float64).eps / 2
    # 4608-term dot products plus a deliberately loose 100*n SVD allowance.
    gamma = NATIVE_PRODUCTS * unit / (1 - NATIVE_PRODUCTS * unit)
    return float((gamma + 100 * DIMENSION * unit) * torch.linalg.matrix_norm(
        matrix, ord="fro"))


def analyze_slice(matrix: torch.Tensor) -> dict[str, float | int | bool]:
    singular = torch.linalg.svdvals(matrix)
    threshold = singular[SLICE_RANK_LIMIT]
    rounding = conservative_roundoff_bound(matrix)
    tail = torch.linalg.vector_norm(singular[SLICE_RANK_LIMIT:])
    total = torch.linalg.vector_norm(singular)
    numerical_rank_lower_bound = int((singular > rounding).sum())
    return {
        "dimension": DIMENSION,
        "tested_product_rank": TESTED_PRODUCTS,
        "max_slice_rank_of_tested_program": SLICE_RANK_LIMIT,
        "sigma_513": float(threshold),
        "sigma_513_over_sigma_1": float(threshold / singular[0]),
        "smallest_singular_value": float(singular[-1]),
        "conservative_float64_roundoff_bound": rounding,
        "sigma_513_margin_over_roundoff": float(threshold / rounding),
        "rank_exceeds_512": bool(threshold > rounding),
        "numerical_slice_rank_lower_bound": numerical_rank_lower_bound,
        "numerically_certified_minimum_product_count": numerical_rank_lower_bound,
        "best_rank512_relative_frobenius_error_lower_bound": float(tail / total),
        "best_rank512_absolute_frobenius_error_lower_bound": float(tail),
    }


def main() -> None:
    if RESULT.exists():
        raise RuntimeError("slice-rank result already exists")
    started = time.time()
    model, checkpoint = facade.load_bilin18(device=torch.device("cpu"),
                                            dtype=torch.bfloat16)
    sites = {}
    for site in SITES:
        mlp = model.transformer.h[site].mlp
        matrix = symmetric_slice(mlp.Down.weight, mlp.Left.weight,
                                 mlp.Right.weight, coordinate=0)
        sites[str(site)] = analyze_slice(matrix)
    result = {
        "schema": "early_mlp_quadratic_slice_rank_certificate_v1",
        "status": "deterministic_weight_analysis_complete",
        "theorem": (
            "rank(A_y)<=r for every symmetric polarization slice of an "
            "r-product quadratic program; Eckart-Young bounds rank-512 slice error"
        ),
        "slice": "second_input_coordinate_e0",
        "sites": sites,
        "checkpoint": checkpoint.__dict__,
        "runtime_seconds": time.time() - started,
        "claim_boundary": (
            "global exact quadratic-map and fixed-slice Frobenius certificate; "
            "not a natural-distribution CE, semantic, suffix, or OOD-frequency claim"
        ),
    }
    temporary = RESULT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    temporary.replace(RESULT)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
