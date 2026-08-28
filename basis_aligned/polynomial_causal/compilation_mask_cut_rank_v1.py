"""Frozen CPU design for the layer-5 compilation-mask cut-rank assay.

This module performs no model or artifact I/O.  It fixes the 8 x 8 mask registry,
prospective train/validation/heldout cells, and the exact anchored interaction whose
matrix rank is a necessary condition for a low tensor-train rank across this cut.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import torch


Site = tuple[str, int]
Cell = tuple[int, int]


PREFIX_MASKS: tuple[tuple[Site, ...], ...] = (
    (),
    (("attn", 1),),
    (("mlp", 1),),
    (("attn", 1), ("mlp", 1)),
    (("mlp", 5),),
    (("attn", 5), ("mlp", 5)),
    (("attn", 1), ("mlp", 1), ("mlp", 5)),
    (("attn", 2), ("mlp", 3), ("attn", 4), ("mlp", 4)),
)

SUFFIX_MASKS: tuple[tuple[Site, ...], ...] = (
    (),
    (("attn", 15),),
    (("mlp", 17),),
    (("attn", 13), ("mlp", 13)),
    tuple((kind, layer) for layer in range(13, 18) for kind in ("attn", "mlp")),
    tuple(
        (kind, layer)
        for layer in (9, 11, 13, 15, 17)
        for kind in ("attn", "mlp")
    ),
    tuple((kind, layer) for layer in range(9, 18) for kind in ("attn", "mlp")),
    (
        ("attn", 6), ("mlp", 7), ("attn", 8), ("mlp", 10),
        ("attn", 12), ("mlp", 14), ("attn", 16), ("mlp", 17),
    ),
)

ANCHOR_CELLS: tuple[Cell, ...] = tuple(
    (i, j) for i in range(8) for j in range(8) if i == 0 or j == 0
)

# Four cyclic diagonals give every nonempty prefix and suffix degree four and make
# the bipartite training graph connected.  The seed that selected the remaining
# validation/heldout allocation is frozen in the preregistration; the literal cells
# below, rather than a runtime RNG, are authoritative.
TRAIN_CELLS: tuple[Cell, ...] = tuple(
    (i, j)
    for i in range(1, 8)
    for j in range(1, 8)
    if (j - i) % 7 in {0, 1, 2, 3}
)
VALIDATION_CELLS: tuple[Cell, ...] = (
    (1, 6), (1, 7), (2, 1), (2, 7), (4, 1),
    (6, 4), (6, 5), (7, 4), (7, 5), (7, 6),
)
HELDOUT_CELLS: tuple[Cell, ...] = (
    (1, 5), (2, 6), (3, 1), (3, 2), (3, 7), (4, 2),
    (4, 3), (5, 2), (5, 3), (5, 4), (6, 3),
)

MLP5_PREFIX_ROWS = frozenset({4, 5, 6})
DENSE_DEEP_SUFFIX_COLUMNS = frozenset({4, 5, 6})


def _connected_training_graph(cells: Sequence[Cell]) -> bool:
    adjacency = {node: set() for node in range(14)}
    for i, j in cells:
        left, right = i - 1, 7 + j - 1
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node] - seen:
            seen.add(neighbor)
            queue.append(neighbor)
    return len(seen) == 14


def validate_registry() -> None:
    if len(PREFIX_MASKS) != 8 or len(SUFFIX_MASKS) != 8 or PREFIX_MASKS[0] or (
        SUFFIX_MASKS[0]
    ):
        raise RuntimeError("cut-rank mask registry changed")
    for mask in PREFIX_MASKS:
        if len(mask) != len(set(mask)) or any(
            kind not in {"attn", "mlp"} or not 1 <= layer <= 5
            for kind, layer in mask
        ):
            raise RuntimeError("cut-rank prefix mask crossed the physical cut")
    for mask in SUFFIX_MASKS:
        if len(mask) != len(set(mask)) or any(
            kind not in {"attn", "mlp"} or not 6 <= layer <= 17
            for kind, layer in mask
        ):
            raise RuntimeError("cut-rank suffix mask crossed the physical cut")
    inner = {(i, j) for i in range(1, 8) for j in range(1, 8)}
    train, validation, heldout = map(
        set, (TRAIN_CELLS, VALIDATION_CELLS, HELDOUT_CELLS),
    )
    if len(ANCHOR_CELLS) != 15 or len(train) != 28 or len(validation) != 10 or (
        len(heldout) != 11 or train | validation | heldout != inner
    ) or train & validation or train & heldout or validation & heldout:
        raise RuntimeError("cut-rank prospective split changed")
    if any(
        sum(i == row for i, _j in train) < 3 for row in range(1, 8)
    ) or any(
        sum(j == column for _i, j in train) < 3 for column in range(1, 8)
    ) or not _connected_training_graph(TRAIN_CELLS):
        raise RuntimeError("cut-rank training support is not identifiable")
    for cells in (VALIDATION_CELLS, HELDOUT_CELLS):
        if not any(i in MLP5_PREFIX_ROWS for i, _j in cells) or not any(
            i not in MLP5_PREFIX_ROWS for i, _j in cells
        ) or not any(j in DENSE_DEEP_SUFFIX_COLUMNS for _i, j in cells) or not any(
            j not in DENSE_DEEP_SUFFIX_COLUMNS for _i, j in cells
        ):
            raise RuntimeError("cut-rank challenge groups are not represented")


def anchored_interaction(cost: torch.Tensor) -> torch.Tensor:
    """Return Δ_ij = H_ij - H_i0 - H_0j + H_00 for an 8 x 8 cost grid."""

    if not torch.is_tensor(cost) or tuple(cost.shape) != (8, 8) or not bool(
        torch.isfinite(cost).all()
    ):
        raise ValueError("cut-rank cost grid must be finite and 8 x 8")
    value = cost.detach().cpu().double().contiguous()
    return (
        value - value[:, :1] - value[:1, :] + value[0, 0]
    ).contiguous()


def spectral_tail_nre(interaction: torch.Tensor, rank: int) -> float:
    """Exact Frobenius tail ratio of the best rank-r approximation."""

    if not torch.is_tensor(interaction) or tuple(interaction.shape) != (8, 8) or (
        type(rank) is not int or not 0 <= rank <= 8
    ):
        raise ValueError("cut-rank spectral-tail request is malformed")
    singular = torch.linalg.svdvals(
        interaction.detach().cpu().double().contiguous()
    )
    denominator = singular.square().sum()
    if float(denominator) <= 1e-12:
        return 0.0
    return float(torch.sqrt(singular[rank:].square().sum() / denominator))


def inhomogeneous_tt_parameter_count(rank: int) -> int:
    """Gauge-adjusted count 8R + 44R² for length 17, alphabet 4 TT rank R."""

    if type(rank) is not int or rank <= 0:
        raise ValueError("tensor-train rank must be positive")
    return 8 * rank + 44 * rank * rank


validate_registry()
