"""Frozen CPU design for the layer-5 compilation-mask cut-rank assay.

This module performs no model or artifact I/O.  It fixes the 8 x 8 mask registry,
prospective train/validation/heldout cells, and the exact anchored interaction whose
matrix rank is a necessary condition for a low tensor-train rank across this cut.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch


Site = tuple[str, int]
Cell = tuple[int, int]


RIDGE_GRID: tuple[float, ...] = tuple(2.0 ** exponent for exponent in range(-12, 5, 2))
RANK_GRID: tuple[int, ...] = (1, 2)
ALS_RESTARTS = 8
ALS_MAX_ITERATIONS = 400
ALS_RELATIVE_TOLERANCE = 1e-12
ALS_SEED = 2026082850
ZERO_INTERACTION_TOLERANCE = 1e-12
S1834_ALPHA = 0.8790830549627247


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

DEVELOPMENT_CELLS: tuple[Cell, ...] = (
    *ANCHOR_CELLS, *TRAIN_CELLS, *VALIDATION_CELLS,
)
FIT_CELLS: tuple[Cell, ...] = (*ANCHOR_CELLS, *TRAIN_CELLS)
ALL_CELLS: tuple[Cell, ...] = tuple((i, j) for i in range(8) for j in range(8))

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


def _finite_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
        float(value)
    ):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


def _sha256_text(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _exact_cell_mapping(
    name: str, value: Mapping[Cell, Any], expected: Sequence[Cell],
) -> dict[Cell, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly the frozen cells")
    output: dict[Cell, Any] = {}
    for cell in expected:
        if not isinstance(cell, tuple) or len(cell) != 2 or any(
            type(index) is not int or not 0 <= index < 8 for index in cell
        ):
            raise ValueError(f"{name} contains a malformed cell")
        output[cell] = value[cell]
    return output


@dataclass(frozen=True, slots=True)
class ObservedCell:
    """Aggregate same-wave observations for one compilation mask.

    ``top1_accuracy`` is a fraction in ``[0,1]``. ``mean_ce`` is mean next-token
    cross entropy in nats on the identical support.  Costs are derived relative to
    the observed empty/empty B0 cell; callers cannot supply either cost directly.
    """

    top1_accuracy: float
    mean_ce: float

    def __post_init__(self) -> None:
        top1 = _finite_number("top-1 accuracy", self.top1_accuracy)
        ce = _finite_number("mean CE", self.mean_ce)
        if not 0.0 <= top1 <= 1.0 or ce < 0.0:
            raise ValueError("observed top-1/CE values are outside their physical range")
        object.__setattr__(self, "top1_accuracy", top1)
        object.__setattr__(self, "mean_ce", ce)


@dataclass(frozen=True, slots=True)
class CellCosts:
    """Primary top-1 percentage-point and secondary CE-nat costs."""

    top1_pp: Mapping[Cell, float]
    ce_nats: Mapping[Cell, float]

    def __post_init__(self) -> None:
        top1 = dict(self.top1_pp)
        ce = dict(self.ce_nats)
        if set(top1) != set(ce) or (0, 0) not in top1 or any(
            not isinstance(cell, tuple) or len(cell) != 2 or any(
                type(index) is not int or not 0 <= index < 8 for index in cell
            ) for cell in top1
        ):
            raise ValueError("cost targets have malformed or unequal cell support")
        for name, values in (("top-1 cost", top1), ("CE cost", ce)):
            for cell, value in values.items():
                values[cell] = _finite_number(name, value)
        if abs(top1[(0, 0)]) > 1e-12 or abs(ce[(0, 0)]) > 1e-12:
            raise ValueError("B0 cost must be exactly zero up to numerical tolerance")
        object.__setattr__(self, "top1_pp", MappingProxyType(top1))
        object.__setattr__(self, "ce_nats", MappingProxyType(ce))


def observed_costs(observations: Mapping[Cell, ObservedCell]) -> CellCosts:
    """Derive cost targets from typed observations on any support containing B0."""

    if not isinstance(observations, Mapping) or (0, 0) not in observations or any(
        type(value) is not ObservedCell for value in observations.values()
    ):
        raise ValueError("observed costs require typed cells including B0")
    base = observations[(0, 0)]
    return CellCosts(
        top1_pp={
            cell: 100.0 * (base.top1_accuracy - value.top1_accuracy)
            for cell, value in observations.items()
        },
        ce_nats={
            cell: value.mean_ce - base.mean_ce
            for cell, value in observations.items()
        },
    )


@dataclass(frozen=True, slots=True)
class FrozenSingletonCosts:
    """Prospectively source-bound singleton baseline in one target currency."""

    target: str
    costs: Mapping[Site, float]
    source_sha256: str
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        expected = {
            (kind, layer) for layer in range(1, 18) for kind in ("attn", "mlp")
        }
        values = dict(self.costs)
        if self.target not in {"top1_pp", "ce_nats"} or set(values) != expected or (
            not _sha256_text(self.source_sha256)
        ):
            raise ValueError("frozen singleton baseline identity/support changed")
        for site, value in values.items():
            values[site] = _finite_number(f"singleton baseline {site}", value)
        object.__setattr__(self, "costs", MappingProxyType(values))
        payload = [
            [kind, layer, values[(kind, layer)]]
            for layer in range(1, 18) for kind in ("attn", "mlp")
        ]
        object.__setattr__(self, "content_sha256", hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest())


def anchored_interaction_cells(
    costs: Mapping[Cell, float], cells: Sequence[Cell],
) -> dict[Cell, float]:
    """Construct selected anchored interactions without materializing heldout cells."""

    if not isinstance(costs, Mapping):
        raise TypeError("anchored cell interactions require a cost mapping")
    requested = tuple(cells)
    if len(requested) != len(set(requested)) or any(
        cell[0] == 0 or cell[1] == 0 for cell in requested
    ):
        raise ValueError("anchored interaction cells must be distinct inner cells")
    required = {(0, 0)} | {
        anchor for i, j in requested for anchor in ((i, 0), (0, j), (i, j))
    }
    if not required <= set(costs):
        raise ValueError("anchored interaction is missing a requested cell or anchor")
    output: dict[Cell, float] = {}
    for i, j in requested:
        values = tuple(
            _finite_number("anchored interaction cost", costs[cell])
            for cell in ((i, j), (i, 0), (0, j), (0, 0))
        )
        output[(i, j)] = values[0] - values[1] - values[2] + values[3]
    return output


def _additive_anchor_prediction(costs: Mapping[Cell, float], cell: Cell) -> float:
    i, j = cell
    return float(costs[(i, 0)] + costs[(0, j)] - costs[(0, 0)])


@dataclass(frozen=True, slots=True)
class RankCandidateSummary:
    rank: int
    ridge: float
    train_objective: float
    validation_rmse: float
    restart: int
    iterations: int


class _RankModel:
    """Private immutable-enough owner of one fitted interaction factorization."""

    __slots__ = ("_left", "_right", "rank", "ridge", "scale")

    def __init__(
        self, *, left: torch.Tensor, right: torch.Tensor, rank: int,
        ridge: float, scale: float,
    ) -> None:
        if tuple(left.shape) != (7, rank) or tuple(right.shape) != (7, rank):
            raise ValueError("rank model factors have the wrong shape")
        self._left = left.detach().cpu().double().contiguous().clone()
        self._right = right.detach().cpu().double().contiguous().clone()
        self.rank = rank
        self.ridge = ridge
        self.scale = scale

    def predict_interaction(self, cell: Cell) -> float:
        i, j = cell
        if not 1 <= i <= 7 or not 1 <= j <= 7:
            raise ValueError("rank model predicts only inner cells")
        return float(torch.dot(self._left[i - 1], self._right[j - 1]) * self.scale)


def _factor_objective(
    left: torch.Tensor, right: torch.Tensor, entries: Sequence[Cell],
    target: torch.Tensor, ridge: float,
) -> float:
    prediction = torch.stack([
        torch.dot(left[i - 1], right[j - 1]) for i, j in entries
    ])
    penalty = 0.5 * ridge * (left.square().mean() + right.square().mean())
    return float((prediction - target).square().mean() + penalty)


def _initial_factors(
    rank: int, restart: int, entries: Sequence[Cell], target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    filled = torch.zeros((7, 7), dtype=torch.float64)
    counts = torch.zeros((7, 7), dtype=torch.float64)
    for cell, value in zip(entries, target, strict=True):
        i, j = cell
        filled[i - 1, j - 1] += value
        counts[i - 1, j - 1] += 1.0
    filled = torch.where(counts > 0, filled / counts.clamp_min(1.0), filled)
    if restart == 0:
        u, singular, vh = torch.linalg.svd(filled, full_matrices=False)
        root = torch.sqrt(singular[:rank].clamp_min(1e-12))
        return (u[:, :rank] * root).contiguous(), (
            vh[:rank].T * root
        ).contiguous()
    generator = torch.Generator(device="cpu").manual_seed(
        ALS_SEED + 1000 * rank + restart
    )
    scale = max(float(target.square().mean().sqrt()), 1e-3) ** 0.5
    return (
        scale * torch.randn((7, rank), generator=generator, dtype=torch.float64),
        scale * torch.randn((7, rank), generator=generator, dtype=torch.float64),
    )


def _fit_one_rank_model(
    interactions: Mapping[Cell, float], *, rank: int, ridge: float, scale: float,
) -> tuple[_RankModel, float, int, int]:
    entries = TRAIN_CELLS
    target = torch.tensor(
        [interactions[cell] / scale for cell in entries], dtype=torch.float64,
    )
    row_columns = torch.tensor([
        [column - 1 for row, column in entries if row == i]
        for i in range(1, 8)
    ], dtype=torch.long)
    row_targets = torch.stack([
        torch.stack([target[index] for index, (row, _column) in enumerate(entries) if row == i])
        for i in range(1, 8)
    ])
    column_rows = torch.tensor([
        [row - 1 for row, column in entries if column == j]
        for j in range(1, 8)
    ], dtype=torch.long)
    column_targets = torch.stack([
        torch.stack([target[index] for index, (_row, column) in enumerate(entries) if column == j])
        for j in range(1, 8)
    ])
    best: tuple[float, int, int, torch.Tensor, torch.Tensor] | None = None
    identity = torch.eye(rank, dtype=torch.float64)
    for restart in range(ALS_RESTARTS):
        left, right = _initial_factors(rank, restart, entries, target)
        previous = math.inf
        iterations = 0
        for iteration in range(1, ALS_MAX_ITERATIONS + 1):
            penalty = len(entries) * ridge / (7 * rank)
            row_design = right[row_columns]
            left = torch.linalg.solve(
                row_design.transpose(1, 2) @ row_design + penalty * identity,
                (row_design.transpose(1, 2) @ row_targets.unsqueeze(-1)).squeeze(-1),
            )
            column_design = left[column_rows]
            right = torch.linalg.solve(
                column_design.transpose(1, 2) @ column_design + penalty * identity,
                (
                    column_design.transpose(1, 2) @ column_targets.unsqueeze(-1)
                ).squeeze(-1),
            )
            objective = _factor_objective(left, right, entries, target, ridge)
            iterations = iteration
            if not math.isfinite(objective):
                break
            if math.isfinite(previous) and abs(previous - objective) <= (
                ALS_RELATIVE_TOLERANCE * max(1.0, abs(previous))
            ):
                break
            previous = objective
        objective = _factor_objective(left, right, entries, target, ridge)
        candidate = (objective, restart, iterations, left.clone(), right.clone())
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    if best is None or not math.isfinite(best[0]):
        raise RuntimeError("all deterministic ALS restarts failed")
    objective, restart, iterations, left, right = best
    return (
        _RankModel(left=left, right=right, rank=rank, ridge=ridge, scale=scale),
        objective, restart, iterations,
    )


def _rmse(actual: Mapping[Cell, float], predicted: Mapping[Cell, float]) -> float:
    if set(actual) != set(predicted) or not actual:
        raise ValueError("RMSE inputs differ or are empty")
    return math.sqrt(sum(
        (float(predicted[cell]) - float(actual[cell])) ** 2 for cell in actual
    ) / len(actual))


def _select_rank_model(
    interactions: Mapping[Cell, float],
) -> tuple[_RankModel | None, tuple[RankCandidateSummary, ...], float, bool]:
    train_values = torch.tensor(
        [interactions[cell] for cell in TRAIN_CELLS], dtype=torch.float64,
    )
    scale = float(torch.sqrt(torch.mean(train_values.square())))
    if scale <= ZERO_INTERACTION_TOLERANCE:
        return None, (), 0.0, False
    summaries: list[RankCandidateSummary] = []
    fitted: dict[tuple[int, float], _RankModel] = {}
    validation_actual = {cell: interactions[cell] for cell in VALIDATION_CELLS}
    for rank in RANK_GRID:
        for ridge in RIDGE_GRID:
            model, objective, restart, iterations = _fit_one_rank_model(
                interactions, rank=rank, ridge=ridge, scale=scale,
            )
            prediction = {
                cell: model.predict_interaction(cell) for cell in VALIDATION_CELLS
            }
            validation_rmse = _rmse(validation_actual, prediction)
            summaries.append(RankCandidateSummary(
                rank=rank, ridge=ridge, train_objective=objective,
                validation_rmse=validation_rmse, restart=restart,
                iterations=iterations,
            ))
            fitted[(rank, ridge)] = model
    selected = min(summaries, key=lambda value: (
        value.validation_rmse, value.rank, -value.ridge,
    ))
    return fitted[(selected.rank, selected.ridge)], tuple(summaries), scale, True


def _mask_for_cell(cell: Cell) -> tuple[Site, ...]:
    i, j = cell
    value = (*PREFIX_MASKS[i], *SUFFIX_MASKS[j])
    if len(value) != len(set(value)):
        raise RuntimeError("prefix/suffix mask union duplicated a site")
    return value


def _count_depth_features(cell: Cell) -> torch.Tensor:
    mask = _mask_for_cell(cell)
    attention = tuple(layer for kind, layer in mask if kind == "attn")
    mlp = tuple(layer for kind, layer in mask if kind == "mlp")
    both = len(set(attention) & set(mlp))
    return torch.tensor([
        1.0, float(len(attention)), float(len(mlp)), float(both),
        sum(attention) / 17.0, sum(mlp) / 17.0,
    ], dtype=torch.float64)


@dataclass(frozen=True, slots=True)
class BaselineSummary:
    name: str
    validation_rmse: float
    ridge: float | None


class _BaselineModel:
    __slots__ = ("name", "_predict")

    def __init__(self, name: str, predict: Any) -> None:
        self.name = name
        self._predict = predict

    def predict(self, cell: Cell, costs: Mapping[Cell, float]) -> float:
        return float(self._predict(cell, costs))


def _fit_ridge_baseline(
    costs: Mapping[Cell, float], ridge: float,
) -> _BaselineModel:
    design = torch.stack([_count_depth_features(cell) for cell in FIT_CELLS])
    target = torch.tensor([costs[cell] for cell in FIT_CELLS], dtype=torch.float64)
    scale = design[:, 1:].square().mean(dim=0).sqrt().clamp_min(1e-12)
    standardized = design.clone()
    standardized[:, 1:] /= scale
    penalty = torch.eye(design.shape[1], dtype=torch.float64)
    penalty[0, 0] = 0.0
    coefficient = torch.linalg.solve(
        standardized.T @ standardized + len(FIT_CELLS) * ridge * penalty,
        standardized.T @ target,
    )

    def predict(cell: Cell, _costs: Mapping[Cell, float]) -> float:
        feature = _count_depth_features(cell)
        feature[1:] /= scale
        return float(feature @ coefficient)

    return _BaselineModel("count_depth_type_ridge", predict)


def _singleton_sum(
    cell: Cell, singleton_costs: Mapping[Site, float],
) -> float:
    return sum(float(singleton_costs[site]) for site in _mask_for_cell(cell))


def _solve_monotone_quadratic(
    x: torch.Tensor, y: torch.Tensor, *, ridge: float,
) -> torch.Tensor:
    design = torch.stack((torch.ones_like(x), x, x.square()), dim=1)
    penalty = torch.diag(torch.tensor((0.0, 1.0, 1.0), dtype=torch.float64))
    hessian = design.T @ design + len(x) * ridge * penalty
    linear = design.T @ y
    xmin, xmax = float(x.min()), float(x.max())
    constraints = (
        torch.tensor((0.0, 1.0, 2.0 * xmin), dtype=torch.float64),
        torch.tensor((0.0, 1.0, 2.0 * xmax), dtype=torch.float64),
    )

    def objective(beta: torch.Tensor) -> float:
        return float((design @ beta - y).square().sum() + (
            len(x) * ridge * (beta[1:].square().sum())
        ))

    candidates: list[torch.Tensor] = [torch.linalg.solve(hessian, linear)]
    for active in ((0,), (1,), (0, 1)):
        a = torch.stack([constraints[index] for index in active])
        zeros = torch.zeros((len(active), len(active)), dtype=torch.float64)
        system = torch.cat((
            torch.cat((hessian, a.T), dim=1),
            torch.cat((a, zeros), dim=1),
        ), dim=0)
        rhs = torch.cat((linear, torch.zeros(len(active), dtype=torch.float64)))
        candidates.append(torch.linalg.lstsq(system, rhs).solution[:3])
    feasible = [
        beta for beta in candidates
        if all(float(constraint @ beta) >= -1e-10 for constraint in constraints)
    ]
    if not feasible:
        raise RuntimeError("monotone quadratic solver found no feasible candidate")
    return min(feasible, key=objective).contiguous()


def _baseline_models(
    costs: Mapping[Cell, float], singleton_costs: FrozenSingletonCosts | None,
) -> tuple[dict[str, _BaselineModel], tuple[BaselineSummary, ...], str]:
    models: dict[str, _BaselineModel] = {
        "additive_anchors": _BaselineModel(
            "additive_anchors", lambda cell, values: _additive_anchor_prediction(values, cell),
        ),
    }
    for ridge in RIDGE_GRID:
        model = _fit_ridge_baseline(costs, ridge)
        model.name = f"count_depth_type_ridge/{ridge:.12g}"
        models[model.name] = model
    if singleton_costs is not None:
        if type(singleton_costs) is not FrozenSingletonCosts:
            raise TypeError("singleton baseline must be prospectively source-bound")
        frozen = dict(singleton_costs.costs)
        models["literal_singleton_sum"] = _BaselineModel(
            "literal_singleton_sum",
            lambda cell, _values: _singleton_sum(cell, frozen),
        )
        models["s1834_scaled_singleton_sum"] = _BaselineModel(
            "s1834_scaled_singleton_sum",
            lambda cell, _values: S1834_ALPHA * _singleton_sum(cell, frozen),
        )
        x = torch.tensor(
            [_singleton_sum(cell, frozen) for cell in FIT_CELLS], dtype=torch.float64,
        )
        y = torch.tensor([costs[cell] for cell in FIT_CELLS], dtype=torch.float64)
        for ridge in RIDGE_GRID:
            coefficient = _solve_monotone_quadratic(x, y, ridge=ridge)
            name = f"monotone_quadratic_singleton/{ridge:.12g}"
            models[name] = _BaselineModel(
                name,
                lambda cell, _values, beta=coefficient: float(
                    beta @ torch.tensor((
                        1.0, _singleton_sum(cell, frozen),
                        _singleton_sum(cell, frozen) ** 2,
                    ), dtype=torch.float64)
                ),
            )
    validation_actual = {cell: costs[cell] for cell in VALIDATION_CELLS}
    summaries = tuple(BaselineSummary(
        name=name,
        validation_rmse=_rmse(validation_actual, {
            cell: model.predict(cell, costs) for cell in VALIDATION_CELLS
        }),
        ridge=(
            float(name.rsplit("/", 1)[1]) if "/" in name else None
        ),
    ) for name, model in models.items())
    selected = min(summaries, key=lambda value: (value.validation_rmse, value.name))
    return models, summaries, selected.name


@dataclass(frozen=True, slots=True)
class TargetDevelopmentSummary:
    target: str
    train_scale: float
    nontrivial_train_interaction: bool
    selected_rank: int
    selected_ridge: float | None
    selected_validation_rmse: float
    candidates: tuple[RankCandidateSummary, ...]
    baselines: tuple[BaselineSummary, ...]
    selected_baseline: str
    singleton_baseline_source_sha256: str | None
    singleton_baseline_content_sha256: str | None


class _TargetDevelopment:
    __slots__ = (
        "baseline_model", "costs", "rank_model", "summary",
    )

    def __init__(
        self, *, costs: Mapping[Cell, float], summary: TargetDevelopmentSummary,
        rank_model: _RankModel | None, baseline_model: _BaselineModel,
    ) -> None:
        self.costs = MappingProxyType(dict(costs))
        self.summary = summary
        self.rank_model = rank_model
        self.baseline_model = baseline_model


class CutRankDevelopment:
    """Selection-complete object which contains no heldout observations or metrics."""

    __slots__ = ("_ce", "_finalized", "_observations", "_top1")

    def __init__(
        self, *, observations: Mapping[Cell, ObservedCell],
        top1: _TargetDevelopment, ce: _TargetDevelopment,
    ) -> None:
        self._observations = MappingProxyType(dict(observations))
        self._top1 = top1
        self._ce = ce
        self._finalized = False

    @property
    def top1_summary(self) -> TargetDevelopmentSummary:
        return self._top1.summary

    @property
    def ce_summary(self) -> TargetDevelopmentSummary:
        return self._ce.summary

    def _begin_finalization(self) -> None:
        if self._finalized:
            raise RuntimeError("cut-rank heldout finalization was already attempted")
        self._finalized = True


def _prepare_target(
    name: str, costs: Mapping[Cell, float],
    singleton_costs: FrozenSingletonCosts | None,
) -> _TargetDevelopment:
    if singleton_costs is not None and singleton_costs.target != name:
        raise ValueError("singleton baseline target currency differs from fitted target")
    interactions = anchored_interaction_cells(
        costs, (*TRAIN_CELLS, *VALIDATION_CELLS),
    )
    rank_model, candidates, scale, nontrivial = _select_rank_model(interactions)
    if rank_model is None:
        selected_rank = 0
        selected_ridge = None
        selected_validation_rmse = _rmse(
            {cell: interactions[cell] for cell in VALIDATION_CELLS},
            {cell: 0.0 for cell in VALIDATION_CELLS},
        )
    else:
        selected = min(candidates, key=lambda value: (
            value.validation_rmse, value.rank, -value.ridge,
        ))
        selected_rank = selected.rank
        selected_ridge = selected.ridge
        selected_validation_rmse = selected.validation_rmse
    baseline_models, baseline_summaries, selected_baseline = _baseline_models(
        costs, singleton_costs,
    )
    summary = TargetDevelopmentSummary(
        target=name, train_scale=scale,
        nontrivial_train_interaction=nontrivial,
        selected_rank=selected_rank, selected_ridge=selected_ridge,
        selected_validation_rmse=selected_validation_rmse,
        candidates=candidates, baselines=baseline_summaries,
        selected_baseline=selected_baseline,
        singleton_baseline_source_sha256=(
            None if singleton_costs is None else singleton_costs.source_sha256
        ),
        singleton_baseline_content_sha256=(
            None if singleton_costs is None else singleton_costs.content_sha256
        ),
    )
    return _TargetDevelopment(
        costs=costs, summary=summary, rank_model=rank_model,
        baseline_model=baseline_models[selected_baseline],
    )


def prepare_development(
    observations: Mapping[Cell, ObservedCell], *,
    singleton_top1_pp: FrozenSingletonCosts | None = None,
    singleton_ce_nats: FrozenSingletonCosts | None = None,
) -> CutRankDevelopment:
    """Fit/select using exactly anchors+train+validation; heldout keys fail closed."""

    checked = _exact_cell_mapping(
        "cut-rank development observations", observations, DEVELOPMENT_CELLS,
    )
    if any(type(value) is not ObservedCell for value in checked.values()):
        raise TypeError("cut-rank development observations must be typed")
    costs = observed_costs(checked)
    return CutRankDevelopment(
        observations=checked,
        top1=_prepare_target("top1_pp", costs.top1_pp, singleton_top1_pp),
        ce=_prepare_target("ce_nats", costs.ce_nats, singleton_ce_nats),
    )


@dataclass(frozen=True, slots=True)
class HeldoutTargetMetrics:
    target: str
    selected_rank: int
    selected_ridge: float | None
    selected_baseline: str
    total_rmse: float
    maximum_absolute_error: float
    interaction_nre: float | None
    heldout_r2: float | None
    baseline_rmse: float
    rmse_ratio: float | None
    mlp5_group_rmse: float
    mlp5_group_baseline_rmse: float
    non_mlp5_group_rmse: float
    non_mlp5_group_baseline_rmse: float
    dense_deep_group_rmse: float
    dense_deep_group_baseline_rmse: float
    sparse_deep_group_rmse: float
    sparse_deep_group_baseline_rmse: float
    full_grid_rank2_spectral_tail_nre: float
    rank_signal_present: bool


@dataclass(frozen=True, slots=True)
class FinalizedCutRankAnalysis:
    """Point estimates only; bootstrap gates deliberately remain unavailable."""

    top1: HeldoutTargetMetrics
    ce: HeldoutTargetMetrics
    heldout_cell_count: int = len(HELDOUT_CELLS)
    bootstrap_complete: bool = False
    promotive_decision: None = None


def _subset_rmse(
    actual: Mapping[Cell, float], predicted: Mapping[Cell, float],
    predicate: Any,
) -> float:
    cells = tuple(cell for cell in actual if predicate(cell))
    if not cells:
        raise RuntimeError("frozen heldout challenge group is empty")
    return _rmse(
        {cell: actual[cell] for cell in cells},
        {cell: predicted[cell] for cell in cells},
    )


def _finalize_target(
    development: _TargetDevelopment, full_costs: Mapping[Cell, float],
) -> HeldoutTargetMetrics:
    actual = {cell: full_costs[cell] for cell in HELDOUT_CELLS}
    interactions = anchored_interaction_cells(full_costs, HELDOUT_CELLS)
    if development.rank_model is None:
        predicted_interaction = {cell: 0.0 for cell in HELDOUT_CELLS}
    else:
        predicted_interaction = {
            cell: development.rank_model.predict_interaction(cell)
            for cell in HELDOUT_CELLS
        }
    predicted = {
        cell: _additive_anchor_prediction(full_costs, cell) + predicted_interaction[cell]
        for cell in HELDOUT_CELLS
    }
    baseline = {
        cell: development.baseline_model.predict(cell, full_costs)
        for cell in HELDOUT_CELLS
    }
    errors = {cell: predicted[cell] - actual[cell] for cell in HELDOUT_CELLS}
    total_rmse = _rmse(actual, predicted)
    baseline_rmse = _rmse(actual, baseline)
    interaction_denominator = sum(value * value for value in interactions.values())
    interaction_nre = None if interaction_denominator <= 1e-12 else math.sqrt(
        sum((predicted_interaction[cell] - interactions[cell]) ** 2 for cell in interactions)
        / interaction_denominator
    )
    mean = sum(actual.values()) / len(actual)
    total_variation = sum((value - mean) ** 2 for value in actual.values())
    heldout_r2 = None if total_variation <= 1e-12 else 1.0 - (
        sum(value * value for value in errors.values()) / total_variation
    )
    rmse_ratio = None if baseline_rmse <= 1e-12 else total_rmse / baseline_rmse
    full_grid = torch.tensor([
        [full_costs[(i, j)] for j in range(8)] for i in range(8)
    ], dtype=torch.float64)

    def group(predicate: Any) -> tuple[float, float]:
        return (
            _subset_rmse(actual, predicted, predicate),
            _subset_rmse(actual, baseline, predicate),
        )

    mlp5, mlp5_baseline = group(lambda cell: cell[0] in MLP5_PREFIX_ROWS)
    non_mlp5, non_mlp5_baseline = group(lambda cell: cell[0] not in MLP5_PREFIX_ROWS)
    dense, dense_baseline = group(lambda cell: cell[1] in DENSE_DEEP_SUFFIX_COLUMNS)
    sparse, sparse_baseline = group(lambda cell: cell[1] not in DENSE_DEEP_SUFFIX_COLUMNS)
    return HeldoutTargetMetrics(
        target=development.summary.target,
        selected_rank=development.summary.selected_rank,
        selected_ridge=development.summary.selected_ridge,
        selected_baseline=development.summary.selected_baseline,
        total_rmse=total_rmse,
        maximum_absolute_error=max(abs(value) for value in errors.values()),
        interaction_nre=interaction_nre, heldout_r2=heldout_r2,
        baseline_rmse=baseline_rmse, rmse_ratio=rmse_ratio,
        mlp5_group_rmse=mlp5, mlp5_group_baseline_rmse=mlp5_baseline,
        non_mlp5_group_rmse=non_mlp5,
        non_mlp5_group_baseline_rmse=non_mlp5_baseline,
        dense_deep_group_rmse=dense,
        dense_deep_group_baseline_rmse=dense_baseline,
        sparse_deep_group_rmse=sparse,
        sparse_deep_group_baseline_rmse=sparse_baseline,
        full_grid_rank2_spectral_tail_nre=spectral_tail_nre(
            anchored_interaction(full_grid), 2,
        ),
        rank_signal_present=development.summary.nontrivial_train_interaction and (
            interaction_denominator > 1e-12
        ),
    )


def finalize_heldout(
    development: CutRankDevelopment,
    heldout_observations: Mapping[Cell, ObservedCell],
) -> FinalizedCutRankAnalysis:
    """Sole public boundary that computes metrics on the frozen heldout cells."""

    if type(development) is not CutRankDevelopment:
        raise TypeError("heldout finalization requires a development selection")
    heldout = _exact_cell_mapping(
        "cut-rank heldout observations", heldout_observations, HELDOUT_CELLS,
    )
    if any(type(value) is not ObservedCell for value in heldout.values()):
        raise TypeError("cut-rank heldout observations must be typed")
    development._begin_finalization()
    combined = {**development._observations, **heldout}
    if set(combined) != set(ALL_CELLS):
        raise RuntimeError("cut-rank final grid did not close exactly")
    costs = observed_costs(combined)
    return FinalizedCutRankAnalysis(
        top1=_finalize_target(development._top1, costs.top1_pp),
        ce=_finalize_target(development._ce, costs.ce_nats),
    )


validate_registry()
