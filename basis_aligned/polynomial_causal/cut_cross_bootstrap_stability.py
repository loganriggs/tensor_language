#!/usr/bin/env python3
"""Document-bootstrap stability of maximum-volume mask-grid crosses.

The source grid is fully revealed, so this remains retrospective discovery.  The
purpose is to decide whether a prospective cross-selected experiment has a stable
enough pivot rule to freeze before spending new model calls.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import torch

import cut_cross_interpolation_diagnostic as cross


HERE = Path(__file__).resolve().parent
DEFAULT_PAYLOAD = HERE / "compilation_mask_cut_rank_v1_measurement_wave_v1_payload.pt"
DEFAULT_OUTPUT = HERE / "cut_cross_bootstrap_stability_results.json"
BOOTSTRAP_REPETITIONS = 2_000
BOOTSTRAP_SEED = 2_026_082_851
RANKS = (2, 3, 4)


def document_bootstrap_weights(document_count: int) -> torch.Tensor:
    if type(document_count) is not int or document_count <= 1:
        raise ValueError("document count must exceed one")
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    sampled = torch.randint(
        document_count,
        (BOOTSTRAP_REPETITIONS, document_count),
        generator=generator,
    )
    weights = torch.zeros(
        (BOOTSTRAP_REPETITIONS, document_count), dtype=torch.float64,
    )
    weights.scatter_add_(
        1, sampled, torch.ones_like(sampled, dtype=torch.float64),
    )
    return weights.contiguous()


def _candidate_indices(size: int, rank: int) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...]], ...
]:
    return tuple(
        (rows, columns)
        for rows in itertools.combinations(range(size), rank)
        for columns in itertools.combinations(range(size), rank)
    )


def _volume_matrix(
    matrices: torch.Tensor,
    candidates: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
) -> torch.Tensor:
    volumes = [
        torch.linalg.det(matrices[:, rows, :][:, :, columns]).abs()
        for rows, columns in candidates
    ]
    return torch.stack(volumes, dim=1).contiguous()


def pivot_stability(
    draws: torch.Tensor, *, point_matrix: torch.Tensor, rank: int,
) -> dict[str, Any]:
    """Return exact finite-bootstrap max-volume selection stability."""

    if draws.ndim != 3 or draws.shape[1] != draws.shape[2] or draws.dtype != (
        torch.float64
    ) or point_matrix.shape != draws.shape[1:] or point_matrix.dtype != torch.float64 or (
        rank <= 0 or rank > draws.shape[1]
    ) or not bool(torch.isfinite(draws).all()) or not bool(
        torch.isfinite(point_matrix).all()
    ):
        raise ValueError("pivot-stability inputs are malformed")
    candidates = _candidate_indices(draws.shape[1], rank)
    volumes = _volume_matrix(draws, candidates)
    winners = volumes.argmax(dim=1)
    counts = torch.bincount(winners, minlength=len(candidates))

    point_volumes = _volume_matrix(point_matrix.unsqueeze(0), candidates)[0]
    point_index = int(point_volumes.argmax())
    mode_index = int(counts.argmax())
    top_two = torch.topk(volumes, 2, dim=1).values
    margin = top_two[:, 0] / top_two[:, 1].clamp_min(1e-30)

    point_rows, point_columns = candidates[point_index]
    pivot = draws[:, point_rows, :][:, :, point_columns]
    columns = draws[:, :, point_columns]
    selected_rows = draws[:, point_rows, :]
    approximation = columns @ torch.linalg.solve(pivot, selected_rows)
    nre = (draws - approximation).flatten(1).norm(dim=1) / draws.flatten(1).norm(
        dim=1
    ).clamp_min(1e-30)
    condition = torch.linalg.cond(pivot)

    def indices(index: int) -> dict[str, list[int]]:
        rows, columns = candidates[index]
        return {
            "rows_one_indexed": [value + 1 for value in rows],
            "columns_one_indexed": [value + 1 for value in columns],
        }

    return {
        "rank": rank,
        "bootstrap_repetitions": int(len(draws)),
        "point_pivot": indices(point_index),
        "point_pivot_selection_frequency": float((winners == point_index).double().mean()),
        "modal_pivot": indices(mode_index),
        "modal_pivot_selection_frequency": float(counts[mode_index] / len(draws)),
        "unique_winning_pivots": int((counts > 0).sum()),
        "winner_margin_top_over_runner_up": {
            "q05": float(torch.quantile(margin, 0.05)),
            "median": float(torch.quantile(margin, 0.50)),
            "q95": float(torch.quantile(margin, 0.95)),
        },
        "frozen_point_pivot_condition_number": {
            "q05": float(torch.quantile(condition, 0.05)),
            "median": float(torch.quantile(condition, 0.50)),
            "q95": float(torch.quantile(condition, 0.95)),
        },
        "frozen_point_pivot_cross_nre": {
            "q05": float(torch.quantile(nre, 0.05)),
            "median": float(torch.quantile(nre, 0.50)),
            "q95": float(torch.quantile(nre, 0.95)),
        },
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
    document_count = len(raw["document_token_count"])
    weights = document_bootstrap_weights(document_count)
    denominators = weights @ raw["document_token_count"].double()
    accuracy = (weights @ raw["top1_correct"].double()) / denominators[:, None]
    mean_ce = (weights @ raw["ce_sum"].double()) / denominators[:, None]
    total_tokens = raw["document_token_count"].sum().double()
    point_accuracy = raw["top1_correct"].sum(0).double() / total_tokens
    point_ce = raw["ce_sum"].sum(0).double() / total_tokens

    def inner_batch(costs: torch.Tensor) -> torch.Tensor:
        grids = costs.reshape(-1, 8, 8)
        return (
            grids - grids[:, :, :1] - grids[:, :1, :] + grids[:, :1, :1]
        )[:, 1:, 1:].contiguous()

    targets = {
        "top1_cost_pp": (
            inner_batch(100.0 * (accuracy[:, :1] - accuracy)),
            cross.anchored_inner(100.0 * (point_accuracy[:1] - point_accuracy)),
        ),
        "ce_cost_nats": (
            inner_batch(mean_ce - mean_ce[:, :1]),
            cross.anchored_inner(point_ce - point_ce[:1]),
        ),
    }
    return {
        "schema_version": 1,
        "role": "retrospective_discovery_only_no_heldout_claim",
        "payload_path": str(payload_path),
        "payload_sha256": cross._sha256(payload_path),
        "document_count": document_count,
        "token_count": int(total_tokens),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "target_results": {
            name: {
                str(rank): pivot_stability(draws, point_matrix=point, rank=rank)
                for rank in RANKS
            }
            for name, (draws, point) in targets.items()
        },
    }


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
