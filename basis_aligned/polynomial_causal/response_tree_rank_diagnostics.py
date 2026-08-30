"""Exact cut-rank diagnostics for four-mode tree tensor programs.

The intended production object is a *dense fitted response program* with modes
``[phase, source, target, document_code]``.  This module does not complete missing
data and deliberately rejects masks: zero-filling an unobserved response would turn
an assumption into a false rank certificate.

For each of the three unrooted binary trees on four labelled leaves, the rank of a
matricization across every tree edge is the minimum possible bond dimension at that
edge.  These ranks therefore give gauge-invariant lower bounds before any particular
factor coordinates are chosen.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch


FOUR_MODE_TREES: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3))


def _dense_tensor(value: torch.Tensor) -> torch.Tensor:
    if (
        not torch.is_tensor(value)
        or value.ndim != 4
        or any(size < 2 for size in value.shape)
        or not value.is_floating_point()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("tree-rank analysis requires one finite dense four-mode tensor")
    return value.detach().to(device="cpu", dtype=torch.float64).contiguous()


def _canonical_cut(left_modes: Iterable[int]) -> tuple[int, ...]:
    left = tuple(sorted(left_modes))
    if not left or len(set(left)) != len(left) or any(mode not in range(4) for mode in left):
        raise ValueError("cut modes must be a nonempty proper subset of four modes")
    if len(left) == 4:
        raise ValueError("cut modes must be a nonempty proper subset of four modes")
    right = tuple(mode for mode in range(4) if mode not in left)
    return min(left, right)


def matricize(value: torch.Tensor, left_modes: Iterable[int]) -> torch.Tensor:
    """Return the physical cut matrix, with no coordinate-dependent factor gauges."""
    tensor = _dense_tensor(value)
    left = _canonical_cut(left_modes)
    right = tuple(mode for mode in range(4) if mode not in left)
    order = left + right
    rows = math.prod(tensor.shape[mode] for mode in left)
    columns = math.prod(tensor.shape[mode] for mode in right)
    return tensor.permute(order).reshape(rows, columns).contiguous()


def cut_spectrum(
    value: torch.Tensor,
    left_modes: Iterable[int],
    *,
    support_rtol: float = 1e-11,
    energy_fraction: float = 0.999,
) -> dict[str, Any]:
    """Report exact numerical rank and a separate approximation-energy rank."""
    if support_rtol <= 0 or not 0 < energy_fraction <= 1:
        raise ValueError("rank diagnostic controls are invalid")
    left = _canonical_cut(left_modes)
    matrix = matricize(value, left)
    singular = torch.linalg.svdvals(matrix)
    leading = float(singular[0])
    threshold = support_rtol * max(leading, 1.0)
    exact_rank = int((singular > threshold).sum())
    energy = singular.square()
    total = float(energy.sum())
    if total == 0:
        energy_rank = 0
    else:
        cumulative = torch.cumsum(energy, dim=0) / total
        energy_rank = int(torch.searchsorted(
            cumulative, torch.tensor(energy_fraction, dtype=cumulative.dtype)
        )) + 1
    tail = float(energy[energy_rank:].sum()) if energy_rank < len(energy) else 0.0
    return {
        "left_modes": list(left),
        "right_modes": [mode for mode in range(4) if mode not in left],
        "matrix_shape": list(matrix.shape),
        "exact_numerical_rank": exact_rank,
        "support_threshold": threshold,
        "energy_fraction": energy_fraction,
        "energy_rank": energy_rank,
        "energy_tail_squared_frobenius": tail,
        "singular_values": [float(item) for item in singular],
    }


def _tree_nodes(pair: tuple[int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if pair not in FOUR_MODE_TREES:
        raise ValueError("unknown four-mode tree")
    complement = tuple(mode for mode in range(4) if mode not in pair)
    return pair, complement


def analyze_tree(
    value: torch.Tensor,
    pair: tuple[int, int],
    *,
    support_rtol: float = 1e-11,
    energy_fraction: float = 0.999,
) -> dict[str, Any]:
    """Return exact edge ranks and literal minimal HT storage for one tree.

    Storage counts scalar entries in four leaf frames, two transfer cores, and the
    root coupling.  It does not subtract gauge dimensions; both prices are useful but
    answer different questions.
    """
    tensor = _dense_tensor(value)
    left_pair, right_pair = _tree_nodes(pair)
    singleton_reports = {
        str(mode): cut_spectrum(
            tensor, (mode,), support_rtol=support_rtol, energy_fraction=energy_fraction
        )
        for mode in range(4)
    }
    pair_report = cut_spectrum(
        tensor, left_pair, support_rtol=support_rtol, energy_fraction=energy_fraction
    )
    leaf_ranks = [singleton_reports[str(mode)]["exact_numerical_rank"] for mode in range(4)]
    bond_rank = pair_report["exact_numerical_rank"]
    leaf_storage = sum(size * rank for size, rank in zip(tensor.shape, leaf_ranks, strict=True))
    left_core = leaf_ranks[left_pair[0]] * leaf_ranks[left_pair[1]] * bond_rank
    right_core = leaf_ranks[right_pair[0]] * leaf_ranks[right_pair[1]] * bond_rank
    root_coupling = bond_rank * bond_rank
    literal_storage = leaf_storage + left_core + right_core + root_coupling
    return {
        "tree_pair": list(left_pair),
        "tree_complement": list(right_pair),
        "singleton_cuts": singleton_reports,
        "internal_cut": pair_report,
        "edge_ranks": {
            "leaves": leaf_ranks,
            "internal_bond": bond_rank,
        },
        "literal_minimal_ht_storage": literal_storage,
        "dense_storage": tensor.numel(),
        "storage_fraction": literal_storage / tensor.numel(),
        "storage_breakdown": {
            "leaf_frames": leaf_storage,
            "left_transfer_core": left_core,
            "right_transfer_core": right_core,
            "root_coupling": root_coupling,
        },
    }


def rank_trees(
    value: torch.Tensor,
    *,
    support_rtol: float = 1e-11,
    energy_fraction: float = 0.999,
) -> dict[str, Any]:
    """Analyze all three trees and rank them by certified literal storage."""
    tensor = _dense_tensor(value)
    reports = [
        analyze_tree(
            tensor, pair, support_rtol=support_rtol, energy_fraction=energy_fraction
        )
        for pair in FOUR_MODE_TREES
    ]
    ordered = sorted(
        reports,
        key=lambda row: (
            row["literal_minimal_ht_storage"],
            row["edge_ranks"]["internal_bond"],
            row["tree_pair"],
        ),
    )
    return {
        "schema": "four_mode_tree_rank_diagnostics_v1",
        "shape": list(tensor.shape),
        "complete_dense_tensor_required": True,
        "zero_fill_missing_values_forbidden": True,
        "ranked_trees": ordered,
        "winner": ordered[0]["tree_pair"],
        "winner_unique_by_storage": (
            len(ordered) == 1
            or ordered[0]["literal_minimal_ht_storage"]
            < ordered[1]["literal_minimal_ht_storage"]
        ),
    }


def planted_tree_tensor(
    shape: tuple[int, int, int, int] = (5, 6, 7, 8),
    *,
    seed: int = 2026083031,
) -> torch.Tensor:
    """Known-answer tensor whose economical tree is ``(0,1)|(2,3)``."""
    if len(shape) != 4 or any(type(size) is not int or size < 4 for size in shape):
        raise ValueError("planted shape must contain four dimensions of at least four")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    leaf_ranks = (2, 2, 3, 3)
    left_bond, right_bond = 2, 2
    leaves = [
        torch.randn((shape[mode], leaf_ranks[mode]), generator=generator, dtype=torch.float64)
        for mode in range(4)
    ]
    left_core = torch.randn(
        (leaf_ranks[0], leaf_ranks[1], left_bond), generator=generator, dtype=torch.float64
    )
    right_core = torch.randn(
        (leaf_ranks[2], leaf_ranks[3], right_bond), generator=generator, dtype=torch.float64
    )
    root = torch.randn((left_bond, right_bond), generator=generator, dtype=torch.float64)
    left = torch.einsum("ia,jb,abk->ijk", leaves[0], leaves[1], left_core)
    right = torch.einsum("kc,ld,cdq->klq", leaves[2], leaves[3], right_core)
    return torch.einsum("ijk,kq,lmq->ijlm", left, root, right).contiguous()
