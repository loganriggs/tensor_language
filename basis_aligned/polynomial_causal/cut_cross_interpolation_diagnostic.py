#!/usr/bin/env python3
"""Retrospective max-volume cross diagnostic for the sealed layer-5 mask grid.

This is discovery-only.  It uses the now-complete 8 x 8 grid to ask whether a
prospective tensor-cross experiment is warranted; it does not reuse any cell as
held out evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import torch


HERE = Path(__file__).resolve().parent
DEFAULT_PAYLOAD = HERE / "compilation_mask_cut_rank_v1_measurement_wave_v1_payload.pt"
DEFAULT_OUTPUT = HERE / "cut_cross_interpolation_diagnostic_results.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def anchored_inner(cost: torch.Tensor) -> torch.Tensor:
    """Return the non-anchor 7 x 7 block of the anchored interaction."""

    if not torch.is_tensor(cost) or cost.numel() != 64 or not bool(
        torch.isfinite(cost).all()
    ):
        raise ValueError("cost must contain 64 finite values")
    grid = cost.detach().cpu().double().reshape(8, 8)
    interaction = grid - grid[:, :1] - grid[:1, :] + grid[:1, :1]
    return interaction[1:, 1:].contiguous()


def _effective_support(squared_residual: torch.Tensor) -> float:
    total = squared_residual.sum()
    fourth = squared_residual.square().sum()
    return 0.0 if float(total) == 0.0 else float(total.square() / fourth)


def best_rank_diagnostic(matrix: torch.Tensor, rank: int) -> dict[str, Any]:
    """Best-SVD error and residual concentration for one rank."""

    if matrix.ndim != 2 or matrix.dtype != torch.float64 or rank <= 0 or rank > min(
        matrix.shape
    ):
        raise ValueError("matrix/rank request is malformed")
    left, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    approximation = (left[:, :rank] * singular[:rank]) @ right[:rank]
    residual = matrix - approximation
    squared = residual.square().flatten()
    ordered = squared.sort(descending=True).values
    denominator = matrix.square().sum()
    return {
        "nre": float(torch.sqrt(squared.sum() / denominator)),
        "rmse": float(squared.mean().sqrt()),
        "max_abs_error": float(residual.abs().max()),
        "effective_residual_support_cells": _effective_support(squared),
        "top_4_residual_energy_fraction": float(ordered[:4].sum() / squared.sum()),
    }


def maximum_volume_cross(matrix: torch.Tensor, rank: int) -> dict[str, Any]:
    """Exhaustive maximum-volume skeleton approximation for a small matrix."""

    if matrix.ndim != 2 or matrix.dtype != torch.float64 or rank <= 0 or rank > min(
        matrix.shape
    ):
        raise ValueError("matrix/rank request is malformed")
    rows: Iterable[tuple[int, ...]] = itertools.combinations(range(matrix.shape[0]), rank)
    best: tuple[float, tuple[int, ...], tuple[int, ...], torch.Tensor] | None = None
    for row_indices in rows:
        for column_indices in itertools.combinations(range(matrix.shape[1]), rank):
            pivot = matrix[list(row_indices)][:, list(column_indices)]
            volume = abs(float(torch.linalg.det(pivot)))
            if best is None or volume > best[0]:
                best = (volume, row_indices, column_indices, pivot)
    if best is None or best[0] <= 0.0:
        raise RuntimeError("no nonsingular cross of the requested rank exists")
    volume, row_indices, column_indices, pivot = best
    columns = matrix[:, list(column_indices)]
    rows_selected = matrix[list(row_indices), :]
    approximation = columns @ torch.linalg.solve(pivot, rows_selected)
    residual = matrix - approximation
    squared = residual.square()
    return {
        "row_indices_one_indexed": [index + 1 for index in row_indices],
        "column_indices_one_indexed": [index + 1 for index in column_indices],
        "pivot_abs_determinant": volume,
        "pivot_condition_number": float(torch.linalg.cond(pivot)),
        "nre": float(residual.norm() / matrix.norm()),
        "rmse": float(squared.mean().sqrt()),
        "max_abs_error": float(residual.abs().max()),
    }


def analyze_payload(payload_path: Path) -> dict[str, Any]:
    payload_path = payload_path.resolve()
    raw = torch.load(payload_path, map_location="cpu", weights_only=True)
    expected = {
        "schema_version", "authority_sha256", "ordered_document_ids_sha256",
        "document_row_count", "document_token_count", "top1_correct", "ce_sum",
        "per_document_payload_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != expected or raw["schema_version"] != 1:
        raise RuntimeError("sealed measurement payload schema changed")
    token_count = raw["document_token_count"].sum().double()
    if float(token_count) <= 0.0 or raw["top1_correct"].shape[1:] != (64,) or raw[
        "ce_sum"
    ].shape[1:] != (64,):
        raise RuntimeError("sealed measurement payload dimensions changed")
    accuracy = raw["top1_correct"].sum(0).double() / token_count
    mean_ce = raw["ce_sum"].sum(0).double() / token_count
    targets = {
        "top1_cost_pp": anchored_inner(100.0 * (accuracy[:1] - accuracy)),
        "ce_cost_nats": anchored_inner(mean_ce - mean_ce[:1]),
    }
    output: dict[str, Any] = {
        "schema_version": 1,
        "role": "retrospective_discovery_only_no_heldout_claim",
        "payload_path": str(payload_path),
        "payload_sha256": _sha256(payload_path),
        "document_count": int(len(raw["document_token_count"])),
        "token_count": int(token_count),
        "target_results": {},
    }
    for name, matrix in targets.items():
        singular = torch.linalg.svdvals(matrix)
        ranks: dict[str, Any] = {}
        for rank in range(1, 5):
            ranks[str(rank)] = {
                "best_svd": best_rank_diagnostic(matrix, rank),
                "maximum_volume_cross": maximum_volume_cross(matrix, rank),
            }
        output["target_results"][name] = {
            "singular_values": [float(value) for value in singular],
            "ranks": ranks,
        }
    output["shared_maxvol_pivot"] = {
        str(rank): (
            output["target_results"]["top1_cost_pp"]["ranks"][str(rank)][
                "maximum_volume_cross"
            ]["row_indices_one_indexed"]
            == output["target_results"]["ce_cost_nats"]["ranks"][str(rank)][
                "maximum_volume_cross"
            ]["row_indices_one_indexed"]
            and output["target_results"]["top1_cost_pp"]["ranks"][str(rank)][
                "maximum_volume_cross"
            ]["column_indices_one_indexed"]
            == output["target_results"]["ce_cost_nats"]["ranks"][str(rank)][
                "maximum_volume_cross"
            ]["column_indices_one_indexed"]
        )
        for rank in range(1, 5)
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze_payload(args.payload)
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
