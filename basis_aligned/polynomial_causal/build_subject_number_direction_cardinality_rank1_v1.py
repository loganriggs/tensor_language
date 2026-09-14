#!/usr/bin/env python3
"""Freeze a rank-one factorization of ten validated subject-number writes."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/task14_mlp6_7_direction_cardinality_prototype_artifact_v1.json"
OUT = HERE / "SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
DIRECTIONS = ("plural_to_singular", "singular_to_plural")
KEYS = tuple(f"{direction}.cardinality_{cardinality}" for direction in DIRECTIONS for cardinality in range(5))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text())
    assert source["terminal"] == "prototype_artifact" and all(source["predictions"].values())
    matrix = np.asarray([source["prototypes"][key]["coordinates"] for key in KEYS], dtype=np.float64)
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    axis = vh[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    coefficients = matrix @ axis
    reconstruction = coefficients[:, None] * axis[None]
    row_energy = np.sum(reconstruction ** 2, axis=1) / np.sum(matrix ** 2, axis=1)
    payload = {
        "schema": "subject_number_direction_cardinality_rank1_v1_artifact",
        "source_sha256": sha(SOURCE), "source_keys": list(KEYS),
        "axis": axis.astype(np.float32).tolist(),
        "coefficients": {key: float(value) for key, value in zip(KEYS, coefficients)},
        "singular_values": singular.tolist(),
        "cumulative_energy": (np.cumsum(singular ** 2) / np.sum(singular ** 2)).tolist(),
        "row_energy_fraction": {key: float(value) for key, value in zip(KEYS, row_energy)},
        "original_write_scalars": 10 * 1152,
        "compressed_write_scalars": 1152 + 10,
        "unchanged_direction_vectors_scalars": 2 * 1152,
        "original_total_interface_scalars": 12 * 1152,
        "compressed_total_interface_scalars": 1152 + 10 + 2 * 1152,
        "model_loaded": False, "causal_outcomes_read": False, "rank_sweep": False,
        "rank": 1, "terminal": "rank1_frozen_weights_only",
    }
    payload["write_storage_fraction"] = payload["compressed_write_scalars"] / payload["original_write_scalars"]
    payload["total_interface_storage_fraction"] = payload["compressed_total_interface_scalars"] / payload["original_total_interface_scalars"]
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in ("terminal", "rank", "cumulative_energy", "write_storage_fraction", "total_interface_storage_fraction")}, indent=2))


if __name__ == "__main__": main()
