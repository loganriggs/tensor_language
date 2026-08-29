"""Pure exact-price dynamic program for uneven compiler rank allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


WIDTH = 1152
COVERED = 5419
MAP_CAP = 512


def site_cost(rank: int) -> int:
    if type(rank) is not int or not 1 <= rank <= WIDTH:
        raise ValueError("rank is outside the table width")
    return rank * (COVERED + WIDTH) + 2 * WIDTH + 2 * WIDTH * min(rank, MAP_CAP)


@dataclass(frozen=True)
class Allocation:
    ranks: dict[str, int]
    cost: int
    utility: float
    budget: int


def allocate(
    squared_spectra: Mapping[str, torch.Tensor],
    ranks: Sequence[int],
    budget: int,
    *,
    normalized: bool,
) -> Allocation:
    if not squared_spectra or type(budget) is not int or budget <= 0:
        raise ValueError("allocation inputs are empty or invalid")
    grid = tuple(ranks)
    if not grid or any(type(rank) is not int for rank in grid) or tuple(sorted(set(grid))) != grid:
        raise ValueError("rank grid must be strictly increasing integers")
    if grid[0] < 1 or grid[-1] > WIDTH:
        raise ValueError("rank grid is outside the supported width")
    names = tuple(sorted(squared_spectra))
    utility: dict[str, dict[int, float]] = {}
    for name in names:
        spectrum = squared_spectra[name]
        if (
            not torch.is_tensor(spectrum) or spectrum.device.type != "cpu"
            or spectrum.dtype != torch.float64 or spectrum.ndim != 1
            or spectrum.numel() < grid[-1] or not bool(torch.isfinite(spectrum).all())
            or bool((spectrum < 0).any())
        ):
            raise ValueError(f"invalid spectrum for {name}")
        total = float(spectrum.sum())
        if total <= 0:
            raise ValueError(f"zero-energy spectrum for {name}")
        cumulative = spectrum.cumsum(0)
        utility[name] = {
            rank: float(cumulative[rank - 1]) / (total if normalized else 1.0)
            for rank in grid
        }

    # State is cost -> (utility, rank tuple). Pareto pruning keeps only utility records
    # as cost increases, which is exact because all future site choices are additive.
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for name in names:
        candidates: dict[int, tuple[float, tuple[int, ...]]] = {}
        for old_cost, (old_utility, old_ranks) in states.items():
            for rank in grid:
                new_cost = old_cost + site_cost(rank)
                if new_cost > budget:
                    continue
                value = old_utility + utility[name][rank]
                rank_tuple = old_ranks + (rank,)
                previous = candidates.get(new_cost)
                if previous is None or value > previous[0] + 1e-15 or (
                    abs(value - previous[0]) <= 1e-15 and rank_tuple < previous[1]
                ):
                    candidates[new_cost] = (value, rank_tuple)
        if not candidates:
            raise RuntimeError("rank floor exceeds the allocation budget")
        pruned: dict[int, tuple[float, tuple[int, ...]]] = {}
        best = float("-inf")
        for cost in sorted(candidates):
            value = candidates[cost]
            if value[0] > best + 1e-15:
                pruned[cost] = value
                best = value[0]
        states = pruned

    best_cost, (best_utility, best_ranks) = min(
        states.items(), key=lambda item: (-item[1][0], item[0], item[1][1])
    )
    return Allocation(
        ranks=dict(zip(names, best_ranks, strict=True)), cost=best_cost,
        utility=best_utility, budget=budget,
    )


def type_shifted_null(allocation: Mapping[str, int]) -> dict[str, int]:
    if set(allocation) != {
        *(f"mlp{layer}" for layer in range(18)),
        *(f"attn{layer}" for layer in range(18)),
    }:
        raise ValueError("allocation must contain all 36 canonical sites")
    shifted: dict[str, int] = {}
    for kind in ("mlp", "attn"):
        values = [allocation[f"{kind}{layer}"] for layer in range(18)]
        values = values[-1:] + values[:-1]
        shifted.update({f"{kind}{layer}": values[layer] for layer in range(18)})
    return shifted


def allocation_cost(ranks: Mapping[str, int]) -> int:
    return sum(site_cost(rank) for rank in ranks.values())

