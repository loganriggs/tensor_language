"""Measure low-rank separability of the committed site-by-amplitude CE response.

This is a descriptive CPU analysis of S1840.  It does not load the model and does
not promote the empirical-token-mean intervention to the deployed length-1 compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ALPHAS = (0.9, 0.75, 0.5, 0.25, 0.0)
CALIBRATION_ALPHA = 0.25


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(left, right)[0, 1])


def _rank_one_energy(matrix: np.ndarray) -> float:
    singular = np.linalg.svd(matrix, compute_uv=False)
    return float(singular[0] ** 2 / np.sum(singular ** 2))


def analyze(payload: dict) -> dict:
    if payload.get("run") != "substitution_direction_curve":
        raise ValueError("unexpected source result")
    curves = payload.get("curve")
    if not isinstance(curves, dict) or len(curves) != 34:
        raise ValueError("expected exactly 34 site curves")
    names = tuple(curves)
    expected = {f"{kind}{layer}" for layer in range(1, 18) for kind in ("attn", "mlp")}
    if set(names) != expected:
        raise ValueError("site registry changed")
    base = float(payload["live_covered_ce"])
    matrix = np.asarray(
        [[float(curves[name][str(alpha)]) - base for alpha in ALPHAS] for name in names],
        dtype=np.float64,
    )
    if not np.isfinite(matrix).all() or matrix.shape != (34, 5):
        raise ValueError("response matrix is malformed")

    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    energy = np.cumsum(singular ** 2) / np.sum(singular ** 2)
    rank_one = singular[0] * np.outer(left[:, 0], right[0])

    full = matrix[:, -1]
    shared_curve = np.sum(full[:, None] * matrix, axis=0) / np.sum(full ** 2)
    anchored = full[:, None] * shared_curve[None, :]
    calibration_index = ALPHAS.index(CALIBRATION_ALPHA)
    predicted_full = matrix[:, calibration_index] / shared_curve[calibration_index]

    groups = {
        "attention": np.asarray([name.startswith("attn") for name in names]),
        "mlp": np.asarray([name.startswith("mlp") for name in names]),
        "early_layers_1_to_6": np.asarray(
            [int(re.search(r"\d+", name).group()) <= 6 for name in names]
        ),
        "deep_layers_7_to_17": np.asarray(
            [int(re.search(r"\d+", name).group()) >= 7 for name in names]
        ),
    }
    return {
        "analysis": "substitution_response_separability_v1",
        "scope": (
            "descriptive_empirical_token_mean_response_only; no deployed_length1, "
            "heldout, causal, OOD, edit, or whole_program_credit"
        ),
        "site_order": list(names),
        "alphas": list(ALPHAS),
        "singular_values": singular.tolist(),
        "cumulative_frobenius_energy": energy.tolist(),
        "rank1_nrmse": float(np.linalg.norm(matrix - rank_one) / np.linalg.norm(matrix)),
        "full_anchored_shared_curve": shared_curve.tolist(),
        "full_anchored_nrmse": float(np.linalg.norm(matrix - anchored) / np.linalg.norm(matrix)),
        "calibration_alpha": CALIBRATION_ALPHA,
        "calibration_to_full_relative_l2": float(
            np.linalg.norm(predicted_full - full) / np.linalg.norm(full)
        ),
        "calibration_to_full_spearman": _correlation(
            _rank(predicted_full), _rank(full)
        ),
        "group_rank1_energy": {
            name: _rank_one_energy(matrix[mask]) for name, mask in groups.items()
        },
        "predictions_for_length1_replication": {
            "rank1_energy_at_least": 0.98,
            "alpha_0_25_to_full_spearman_at_least": 0.95,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    result = analyze(json.loads(source_bytes))
    result["source_path"] = str(args.source)
    result["source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
