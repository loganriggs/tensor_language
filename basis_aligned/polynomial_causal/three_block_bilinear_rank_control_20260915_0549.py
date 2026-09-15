"""CPU control for the exact rank-two form of the retained QK1 three-block sum."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARENT = (
    ROOT
    / "bilinear_quotient/circuits/fast_screens/"
    / "setting2_regional_head9_8_qk1_late_group_fold_v1_result.json"
)
SEED = 202609150549
HEAD_WIDTH = 128


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def main() -> None:
    parent = json.loads(PARENT.read_text())
    if parent["terminal"] != "valid_qk1_late_group_fold":
        raise ValueError("bound four-block parent is not valid")
    if parent["term_names"] != [
        "late_x_late",
        "late_x_remainder",
        "remainder_x_late",
        "remainder_x_remainder",
    ]:
        raise ValueError("bound grouped-source convention changed")

    # C encodes DD + DR + RD, with rows (q_D,q_R), columns (k_D,k_R).
    coefficient = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    expanded = np.kron(coefficient, np.eye(HEAD_WIDTH, dtype=np.float64))
    coefficient_rank = int(np.linalg.matrix_rank(coefficient))
    scalar_bilinear_rank = int(np.linalg.matrix_rank(expanded))
    singular_values = np.linalg.svd(coefficient, compute_uv=False)
    best_rank_one_relative_frobenius_error = float(
        singular_values[1] / np.linalg.norm(singular_values)
    )

    rng = np.random.default_rng(SEED)
    shape = (7, 19, HEAD_WIDTH)
    q_d, q_r, k_d, k_r = (rng.standard_normal(shape) for _ in range(4))
    direct = (
        np.einsum("bth,bth->bt", q_d, k_d)
        + np.einsum("bth,bth->bt", q_d, k_r)
        + np.einsum("bth,bth->bt", q_r, k_d)
    ) / HEAD_WIDTH
    rank_two = (
        np.einsum("bth,bth->bt", q_d + q_r, k_d + k_r)
        - np.einsum("bth,bth->bt", q_r, k_r)
    ) / HEAD_WIDTH
    parent_minus_rr = (
        np.einsum("bth,bth->bt", q_d + q_r, k_d + k_r)
        - np.einsum("bth,bth->bt", q_r, k_r)
    ) / HEAD_WIDTH

    result = {
        "schema": "three_block_bilinear_rank_control_20260915_0549_result",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parent_sha256": sha256(PARENT),
        "seed": SEED,
        "test_shape": list(shape),
        "coefficient_matrix": coefficient.tolist(),
        "coefficient_determinant": float(np.linalg.det(coefficient)),
        "coefficient_rank": coefficient_rank,
        "coefficient_singular_values": singular_values.tolist(),
        "best_rank_one_relative_frobenius_error": best_rank_one_relative_frobenius_error,
        "scalar_bilinear_rank": scalar_bilinear_rank,
        "head_width": HEAD_WIDTH,
        "direct_three_dot_scalar_multiplications": 3 * HEAD_WIDTH,
        "rank_two_scalar_multiplications": 2 * HEAD_WIDTH,
        "standalone_multiplication_reduction_fraction": 1.0 / 3.0,
        "incremental_dot_products_when_native_parent_is_live": 1,
        "rank_two_relative_error": relative_error(rank_two, direct),
        "parent_minus_rr_relative_error": relative_error(parent_minus_rr, direct),
        "predictions": {
            "coefficient_is_rank_two": coefficient_rank == 2,
            "scalar_lower_bound_is_2h": scalar_bilinear_rank == 2 * HEAD_WIDTH,
            "two_product_identity_is_fp64_exact": relative_error(rank_two, direct) <= 1e-14,
        },
        "scope": (
            "Exact bilinear QK1 score subtotal DD+DR+RD with supplied native "
            "denominators. QK2, values, projections, residual propagation, suffix, "
            "and causal fidelity are not simplified or certified."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
