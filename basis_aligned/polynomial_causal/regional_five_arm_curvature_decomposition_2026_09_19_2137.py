#!/usr/bin/env python3
"""Decompose the frozen regional five-arm interaction into head and suffix terms."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "CITY_INTERCHANGE_COMPOSITION_V1_ARTIFACT.pt"
RESULT = ROOT / "CITY_INTERCHANGE_COMPOSITION_V1_RESULT.json"
OUT = ROOT / "REGIONAL_FIVE_ARM_CURVATURE_DECOMPOSITION_2026-09-19_2137.json"
EXPECTED = {
    ARTIFACT.name: "e501aa82b770ba281aee80781fb4b942ee32c5a614d4a5dcd3325241701b851d",
    RESULT.name: "b3981e78c035a9725e0717d06bf0b71b3d4149a2070db5bcc4dd050d22157330",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metrics(curvature: np.ndarray, head: np.ndarray, total: np.ndarray) -> dict:
    nc = float(np.linalg.norm(curvature))
    nh = float(np.linalg.norm(head))
    nt = float(np.linalg.norm(total))
    dot = float(np.dot(curvature, head))
    gamma = float(np.dot(head, total) / np.dot(head, head))
    return {
        "suffix_curvature_norm": nc,
        "head_product_response_norm": nh,
        "total_interaction_norm": nt,
        "suffix_curvature_over_total": nc / nt,
        "head_product_response_over_total": nh / nt,
        "curvature_head_cosine": dot / (nc * nh),
        "exact_decomposition_max_abs": float(np.max(np.abs(curvature + head - total))),
        "best_scalar_head_to_total": gamma,
        "best_scalar_head_relative_l2_error": float(np.linalg.norm(total - gamma * head) / nt),
    }


def main() -> None:
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    hashes = {p.name: sha256(p) for p in (ARTIFACT, RESULT)}
    if hashes != EXPECTED:
        raise SystemExit(f"source hash mismatch: {hashes}")

    values = torch.load(ARTIFACT, map_location="cpu", weights_only=False)["values"]
    if tuple(values.shape) != (5, 240, 5) or values.dtype != torch.float64:
        raise SystemExit(f"unexpected values tensor: {tuple(values.shape)} {values.dtype}")
    f0, fk, fv, fm, fj = values.numpy()

    # Arm order is preregistered as native, key, value, summed mains without
    # head cross, and true joint. Therefore this identity is exact by addition.
    curvature = fm - fk - fv + f0
    head = fj - fm
    total = fj - fk - fv + f0

    source_result = json.loads(RESULT.read_text())
    documents = source_result["documents"]
    start = 0
    cells = []
    for document in documents:
        count = len(document["arms"]["joint"]["signed_target_effects"])
        stop = start + count
        cell = metrics(
            curvature[start:stop, 0].reshape(-1),
            head[start:stop, 0].reshape(-1),
            total[start:stop, 0].reshape(-1),
        )
        cell.update({"context_id": document["context_id"], "rows": count})
        cells.append(cell)
        start = stop
    if start != 240 or len(cells) != 20:
        raise SystemExit(f"cell manifest mismatch: rows={start}, cells={len(cells)}")

    target = metrics(curvature[:, 0], head[:, 0], total[:, 0])
    controls = metrics(curvature[:, 1:].reshape(-1), head[:, 1:].reshape(-1), total[:, 1:].reshape(-1))
    all_readouts = metrics(curvature.reshape(-1), head.reshape(-1), total.reshape(-1))

    def distribution(key: str) -> dict:
        a = np.asarray([cell[key] for cell in cells], dtype=np.float64)
        return {
            "minimum": float(a.min()),
            "median": float(np.median(a)),
            "maximum": float(a.max()),
        }

    receipt = {
        "schema": "regional.five_arm_curvature_decomposition.v1",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - t0,
        "execution": "CPU-only float64 replay of a frozen five-arm artifact; no model load or forward",
        "evidence_type": "response algebra on stored opened edits",
        "evaluation_status": "opened replay",
        "arm_order": ["native", "key_only", "value_only", "summed_mains_without_head_cross", "true_joint"],
        "identity": {
            "suffix_curvature": "f_mains - f_key - f_value + f_native",
            "head_product_response": "f_joint - f_mains",
            "total_interaction": "f_joint - f_key - f_value + f_native",
            "closure": "total_interaction = suffix_curvature + head_product_response",
        },
        "shape": {"arms": 5, "rows": 240, "readouts": 5, "distinct_context_cells": 20},
        "target": target,
        "four_controls": controls,
        "all_five_readouts": all_readouts,
        "target_cell_distribution": {
            "suffix_curvature_over_total": distribution("suffix_curvature_over_total"),
            "head_product_response_over_total": distribution("head_product_response_over_total"),
            "curvature_head_cosine": distribution("curvature_head_cosine"),
            "cells_negative_curvature_head_cosine": sum(cell["curvature_head_cosine"] < 0 for cell in cells),
        },
        "cells": cells,
        "predictions": {
            "pred_a_exact_decomposition_closes_below_1e-12": target["exact_decomposition_max_abs"] <= 1e-12,
            "pred_b_suffix_curvature_is_below_0_10_of_total": target["suffix_curvature_over_total"] <= 0.10,
            "pred_c_rescaled_head_response_replays_total_below_0_10_relative_l2": target["best_scalar_head_relative_l2_error"] <= 0.10,
        },
        "source_sha256": hashes,
        "interpretation": (
            "The total key-by-value interaction is exactly the sum of the transmitted explicit head-product "
            "response and a mixed finite difference of the native suffix. The latter is material and often "
            "opposes the head term, so a first-order or head-only response census cannot stand in for the joint "
            "full-suffix intervention on these opened cells. This is not fresh, causal identification, random-split "
            "specificity, extraction, or native-fidelity evidence."
        ),
    }
    OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(OUT),
        "predictions": receipt["predictions"],
        "target": target,
        "target_cell_distribution": receipt["target_cell_distribution"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
