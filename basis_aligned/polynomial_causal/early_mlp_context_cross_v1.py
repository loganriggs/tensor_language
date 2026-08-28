"""Frozen CPU registry for the prospective early-MLP/context cross assay.

The grid varies MLP0/1/2 substitutions to the left of a physical cut after layer
2 and contextual suffix substitutions at layers 3--17.  Rank-three cross entries
are discovery, the rank-four expansion is validation, and the nine entries outside
the rank-four cross are held out exactly once.

This module performs no artifact, row, or model I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

import torch


Site = tuple[str, int]
Cell = tuple[int, int]

PREFIX_MASKS: tuple[tuple[Site, ...], ...] = (
    (),
    (("mlp", 0),),
    (("mlp", 0), ("mlp", 1)),
    (("mlp", 1),),
    (("mlp", 0), ("mlp", 1), ("mlp", 2)),
    (("mlp", 2),),
    (("mlp", 1), ("mlp", 2)),
    (("mlp", 0), ("mlp", 2)),
)

SUFFIX_MASKS: tuple[tuple[Site, ...], ...] = (
    (),
    (("attn", 3),),
    (("mlp", 3),),
    (("attn", 3), ("mlp", 3)),
    tuple(("attn", layer) for layer in range(3, 9)),
    tuple(
        (kind, layer)
        for layer in range(3, 9)
        for kind in ("attn", "mlp")
    ),
    tuple(("attn", layer) for layer in range(3, 18)),
    tuple(
        (kind, layer)
        for layer in range(3, 18)
        for kind in ("attn", "mlp")
    ),
)

# Frozen from the stable CE cross family before any outcome on this registry.
# Rank three is nested inside rank four, so its seven rank-four expansion cells
# form a literal prospective validation set.
PIVOT_ROWS: Mapping[int, tuple[int, ...]] = {
    3: (3, 6, 7),
    4: (3, 5, 6, 7),
}
PIVOT_COLUMNS: Mapping[int, tuple[int, ...]] = {
    3: (2, 4, 7),
    4: (2, 4, 6, 7),
}

BOOTSTRAP_SEEDS: Mapping[str, int] = {
    "skip7000": 2026082803,
    "skip11000": 2026082804,
}
BOOTSTRAP_DRAWS = 2_000
ALS_RESTARTS = 8
ALS_SEED = 2026082805
ALS_SWEEPS = 100
# The ALS baseline first divides interactions by their observed RMS, then applies
# this dimensionless penalty.  This is scale-equivariant without outcome tuning.
ALS_RELATIVE_RIDGE = 1e-6

ALL_CELLS: tuple[Cell, ...] = tuple((i, j) for i in range(8) for j in range(8))
ANCHOR_CELLS: tuple[Cell, ...] = tuple(
    cell for cell in ALL_CELLS if cell[0] == 0 or cell[1] == 0
)


def cross_cells(rank: int) -> tuple[Cell, ...]:
    if rank not in PIVOT_ROWS:
        raise ValueError("cross rank must be one of the frozen ranks")
    rows, columns = set(PIVOT_ROWS[rank]), set(PIVOT_COLUMNS[rank])
    return tuple(
        (i, j)
        for i in range(1, 8)
        for j in range(1, 8)
        if i in rows or j in columns
    )


RANK3_DISCOVERY_CELLS: tuple[Cell, ...] = (*ANCHOR_CELLS, *cross_cells(3))
RANK4_VALIDATION_CELLS: tuple[Cell, ...] = tuple(
    cell for cell in cross_cells(4) if cell not in set(cross_cells(3))
)
RANK4_FIT_CELLS: tuple[Cell, ...] = (
    *RANK3_DISCOVERY_CELLS, *RANK4_VALIDATION_CELLS,
)
HELDOUT_CELLS: tuple[Cell, ...] = tuple(
    cell for cell in ALL_CELLS if cell not in set(RANK4_FIT_CELLS)
)
SCORE_CELLS: Mapping[str, tuple[Cell, ...]] = {
    "rank3_validation": RANK4_VALIDATION_CELLS,
    "rank4_heldout": HELDOUT_CELLS,
}


def mask_for_cell(cell: Cell) -> tuple[Site, ...]:
    if cell not in ALL_CELLS:
        raise ValueError("cell is outside the frozen 8 by 8 registry")
    value = (*PREFIX_MASKS[cell[0]], *SUFFIX_MASKS[cell[1]])
    if len(value) != len(set(value)):
        raise RuntimeError("prefix and suffix masks overlap")
    return value


def validate_registry() -> None:
    if len(PREFIX_MASKS) != 8 or len(SUFFIX_MASKS) != 8 or PREFIX_MASKS[0] or (
        SUFFIX_MASKS[0]
    ):
        raise RuntimeError("cross registry does not have one empty anchor per side")
    if len(set(PREFIX_MASKS)) != 8 or len(set(SUFFIX_MASKS)) != 8:
        raise RuntimeError("cross registry contains duplicate masks")
    for mask in PREFIX_MASKS:
        if len(mask) != len(set(mask)) or any(
            kind != "mlp" or not 0 <= layer <= 2 for kind, layer in mask
        ):
            raise RuntimeError("prefix mask crosses its physical boundary")
    for mask in SUFFIX_MASKS:
        if len(mask) != len(set(mask)) or any(
            kind not in {"attn", "mlp"} or not 3 <= layer <= 17
            for kind, layer in mask
        ):
            raise RuntimeError("suffix mask crosses its physical boundary")
    discovery = set(RANK3_DISCOVERY_CELLS)
    validation = set(RANK4_VALIDATION_CELLS)
    heldout = set(HELDOUT_CELLS)
    if len(ANCHOR_CELLS) != 15 or len(cross_cells(3)) != 33 or len(
        RANK4_VALIDATION_CELLS
    ) != 7 or len(RANK4_FIT_CELLS) != 55 or len(HELDOUT_CELLS) != 9 or (
        discovery & validation or discovery & heldout or validation & heldout
    ) or discovery | validation | heldout != set(ALL_CELLS):
        raise RuntimeError("discovery/validation/heldout partition changed")
    if not set(PIVOT_ROWS[3]).issubset(PIVOT_ROWS[4]) or not set(
        PIVOT_COLUMNS[3]
    ).issubset(PIVOT_COLUMNS[4]) or not set(cross_cells(3)).issubset(
        cross_cells(4)
    ):
        raise RuntimeError("rank-three cross is not nested inside rank four")
    if set(BOOTSTRAP_SEEDS) != {"skip7000", "skip11000"} or len(
        set(BOOTSTRAP_SEEDS.values())
    ) != 2 or BOOTSTRAP_DRAWS != 2_000 or ALS_RESTARTS != 8 or (
        ALS_SWEEPS != 100
    ) or ALS_RELATIVE_RIDGE <= 0.0:
        raise RuntimeError("resampling or matched-baseline constants changed")
    if set(SCORE_CELLS) != {"rank3_validation", "rank4_heldout"} or (
        SCORE_CELLS["rank3_validation"] != RANK4_VALIDATION_CELLS
    ) or SCORE_CELLS["rank4_heldout"] != HELDOUT_CELLS:
        raise RuntimeError("score-stage capabilities changed")
    # Heldout must contain singleton, pair, and triple early-MLP prefixes and
    # local-attention, local-block, and shallow-dense suffixes.
    if {i for i, _ in HELDOUT_CELLS} != {1, 2, 4} or {
        j for _, j in HELDOUT_CELLS
    } != {1, 3, 5}:
        raise RuntimeError("heldout semantic challenge groups changed")
    for cell in ALL_CELLS:
        mask_for_cell(cell)


def _licensed_values(
    observed: Mapping[Cell, float], expected_cells: Sequence[Cell],
) -> dict[Cell, float]:
    """Copy exactly one stage's capability, rejecting extra or absent cells."""

    if not isinstance(observed, Mapping) or len(expected_cells) != len(
        set(expected_cells)
    ) or set(observed) != set(expected_cells):
        raise ValueError("observed cells do not equal the stage capability")
    values: dict[Cell, float] = {}
    for cell in expected_cells:
        try:
            value = float(observed[cell])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("observed cost is not a scalar") from error
        if not math.isfinite(value):
            raise ValueError("observed cost is not finite")
        values[cell] = value
    return values


def _interaction_value(values: Mapping[Cell, float], i: int, j: int) -> float:
    return values[(i, j)] - values[(i, 0)] - values[(0, j)] + values[(0, 0)]


@dataclass(frozen=True, slots=True)
class CrossPrediction:
    rank: int
    prediction: torch.Tensor
    pivot_condition_number: float
    pivot_min_relative_singular_value: float

    def __post_init__(self) -> None:
        if self.rank not in PIVOT_ROWS or not torch.is_tensor(
            self.prediction
        ) or tuple(self.prediction.shape) != (8, 8) or self.prediction.dtype != (
            torch.float64
        ) or not bool(torch.isfinite(self.prediction).all()) or not math.isfinite(
            self.pivot_condition_number
        ) or not math.isfinite(self.pivot_min_relative_singular_value):
            raise ValueError("cross prediction is malformed")


def cross_prediction(
    observed: Mapping[Cell, float], rank: int,
) -> CrossPrediction:
    """Interpolate from exactly the licensed anchors and rank-r cross entries."""

    rows = PIVOT_ROWS.get(rank)
    columns = PIVOT_COLUMNS.get(rank)
    if rows is None or columns is None:
        raise ValueError("cross rank must be one of the frozen ranks")
    expected = RANK3_DISCOVERY_CELLS if rank == 3 else RANK4_FIT_CELLS
    values = _licensed_values(observed, expected)
    pivot = torch.tensor(
        [[_interaction_value(values, i, j) for j in columns] for i in rows],
        dtype=torch.float64,
    )
    singular = torch.linalg.svdvals(pivot)
    if float(singular[-1]) <= 1e-12 * max(float(singular[0]), 1.0):
        raise RuntimeError("frozen cross pivot is numerically singular")
    left = torch.tensor(
        [[_interaction_value(values, i, j) for j in columns] for i in range(8)],
        dtype=torch.float64,
    )
    right = torch.tensor(
        [[_interaction_value(values, i, j) for j in range(8)] for i in rows],
        dtype=torch.float64,
    )
    predicted_interaction = left @ torch.linalg.solve(
        pivot, right,
    )
    additive = torch.tensor(
        [
            [values[(i, 0)] + values[(0, j)] - values[(0, 0)] for j in range(8)]
            for i in range(8)
        ],
        dtype=torch.float64,
    )
    prediction = additive + predicted_interaction
    return CrossPrediction(
        rank=rank,
        prediction=prediction.contiguous(),
        pivot_condition_number=float(singular[0] / singular[-1]),
        pivot_min_relative_singular_value=float(singular[-1] / singular[0]),
    )


def additive_prediction(observed: Mapping[Cell, float]) -> torch.Tensor:
    values = _licensed_values(observed, ANCHOR_CELLS)
    return torch.tensor(
        [
            [values[(i, 0)] + values[(0, j)] - values[(0, 0)] for j in range(8)]
            for i in range(8)
        ],
        dtype=torch.float64,
    ).contiguous()


def score_prediction(
    actual: Mapping[Cell, float], predicted: torch.Tensor, stage: str,
) -> dict[str, float]:
    cells = SCORE_CELLS.get(stage)
    if not torch.is_tensor(predicted) or tuple(predicted.shape) != (8, 8) or (
        cells is None
    ) or not bool(torch.isfinite(predicted).all()):
        raise ValueError("prediction score inputs are malformed")
    truth_values = _licensed_values(actual, cells)
    truth = torch.tensor(
        [truth_values[cell] for cell in cells], dtype=torch.float64,
    )
    estimate = torch.tensor(
        [float(predicted[cell]) for cell in cells], dtype=torch.float64,
    )
    error = estimate - truth
    centered = truth - truth.mean()
    denominator = centered.square().sum()
    return {
        "rmse": float(error.square().mean().sqrt()),
        "max_abs_error": float(error.abs().max()),
        "r2": float(1.0 - error.square().sum() / denominator)
        if float(denominator) > 0.0 else 0.0,
    }


validate_registry()
