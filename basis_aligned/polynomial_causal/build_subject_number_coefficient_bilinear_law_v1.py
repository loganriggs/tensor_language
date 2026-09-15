#!/usr/bin/env python3
"""Freeze a four-scalar direction-by-cardinality law without causal outcomes."""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "SUBJECT_NUMBER_DIRECTION_CARDINALITY_RANK1_V1_ARTIFACT.json"
OUT = HERE / "SUBJECT_NUMBER_COEFFICIENT_BILINEAR_LAW_V1_ARTIFACT.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text())
    assert source["terminal"] == "rank1_frozen_weights_only"
    assert not source["causal_outcomes_read"] and not source["rank_sweep"]
    rows = []
    for key, target in source["coefficients"].items():
        direction, cardinality = key.split(".cardinality_")
        d = 1.0 if direction == "plural_to_singular" else -1.0
        c = float(cardinality)
        rows.append((key, d, c, float(target)))
    design = np.asarray([[1.0, d, c, d * c] for _, d, c, _ in rows])
    target = np.asarray([value for *_, value in rows])
    beta = np.linalg.lstsq(design, target, rcond=None)[0]
    prediction = design @ beta
    error = prediction - target
    payload = {
        "schema": "subject_number_coefficient_bilinear_law_v1_artifact",
        "source_sha256": sha(SOURCE),
        "feature_order": ["constant", "direction", "cardinality", "direction_x_cardinality"],
        "direction_encoding": {"plural_to_singular": 1.0, "singular_to_plural": -1.0},
        "beta": beta.tolist(),
        "predicted_coefficients": {row[0]: float(value) for row, value in zip(rows, prediction)},
        "coefficient_cosine": float(prediction @ target / (np.linalg.norm(prediction) * np.linalg.norm(target))),
        "coefficient_relative_l2_error": float(np.linalg.norm(error) / np.linalg.norm(target)),
        "coefficient_max_absolute_error": float(np.max(np.abs(error))),
        "stored_coefficient_scalars_before": 10,
        "stored_coefficient_scalars_after": 4,
        "compressed_total_interface_scalars": 1152 + 4 + 2 * 1152,
        "original_total_interface_scalars": 12 * 1152,
        "causal_outcomes_read": False,
        "model_loaded": False,
        "parameter_updates": 0,
        "terminal": "bilinear_scalar_law_frozen_weights_only",
    }
    payload["total_interface_storage_fraction"] = payload["compressed_total_interface_scalars"] / payload["original_total_interface_scalars"]
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in ("terminal", "beta", "coefficient_cosine", "coefficient_relative_l2_error", "coefficient_max_absolute_error")}, indent=2))


if __name__ == "__main__":
    main()
