"""CPU-only staged statistics and scorer for early-MLP/context cross v1.

This module cannot load rows, a checkpoint, or the model.  Its inputs are sealed
per-document sufficient statistics split into discovery, validation, and heldout
capabilities.  Rank-three scoring has no Python reference to heldout statistics;
rank-four scoring receives them only through its explicit final-stage argument.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from typing import Any

import torch

import early_mlp_context_cross_v1 as cross


SCHEMA_VERSION = 1
STAGE_CELLS: Mapping[str, tuple[cross.Cell, ...]] = {
    "discovery": cross.RANK3_DISCOVERY_CELLS,
    "validation": cross.RANK4_VALIDATION_CELLS,
    "heldout": cross.HELDOUT_CELLS,
}
ROLE_NAMES = tuple(cross.BOOTSTRAP_SEEDS)
ZERO_RMS_TOLERANCE = 1e-15
PIVOT_RELATIVE_TOLERANCE = 1e-12


def _sha256_text(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("tensor hash requires a tensor")
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps({
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
    }, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


class StageStatistics:
    """Sealed document-level statistics for exactly one licensed evidence stage."""

    __slots__ = (
        "_authority_sha256", "_ce_sum", "_document_token_count",
        "_expected_sha256", "_ordered_document_ids_sha256", "_role", "_sealed",
        "_stage", "_top1_correct",
    )

    def __init__(
        self, *, role: str, stage: str, authority_sha256: str,
        ordered_document_ids_sha256: str, document_token_count: torch.Tensor,
        top1_correct: torch.Tensor, ce_sum: torch.Tensor,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        cells = STAGE_CELLS.get(stage)
        if role not in ROLE_NAMES or cells is None or not _sha256_text(
            authority_sha256
        ) or not _sha256_text(ordered_document_ids_sha256) or not torch.is_tensor(
            document_token_count
        ) or not torch.is_tensor(top1_correct) or not torch.is_tensor(ce_sum) or (
            document_token_count.ndim != 1
        ) or top1_correct.shape != (len(document_token_count), len(cells)) or (
            ce_sum.shape != top1_correct.shape
        ) or document_token_count.dtype != torch.long or top1_correct.dtype != (
            torch.long
        ) or ce_sum.dtype != torch.float64:
            raise ValueError("stage sufficient-statistic schema changed")
        values = (document_token_count, top1_correct, ce_sum)
        if len(document_token_count) <= 1 or any(
            value.device.type != "cpu" or not value.is_contiguous() or value.requires_grad
            for value in values
        ) or bool((document_token_count <= 0).any()) or bool(
            (top1_correct < 0).any()
        ) or bool((top1_correct > document_token_count[:, None]).any()) or bool(
            (ce_sum < 0).any()
        ) or not bool(torch.isfinite(ce_sum).all()):
            raise ValueError("stage sufficient statistics violate bounds")
        self._role = role
        self._stage = stage
        self._authority_sha256 = authority_sha256
        self._ordered_document_ids_sha256 = ordered_document_ids_sha256
        self._document_token_count = document_token_count.detach().clone().contiguous()
        self._top1_correct = top1_correct.detach().clone().contiguous()
        self._ce_sum = ce_sum.detach().clone().contiguous()
        self._expected_sha256 = self._compute_sha256()
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("stage sufficient statistics are sealed")
        object.__setattr__(self, name, value)

    def _compute_sha256(self) -> str:
        return _logical_sha256({
            "schema_version": SCHEMA_VERSION,
            "role": self._role,
            "stage": self._stage,
            "cells": [list(cell) for cell in STAGE_CELLS[self._stage]],
            "authority_sha256": self._authority_sha256,
            "ordered_document_ids_sha256": self._ordered_document_ids_sha256,
            "document_token_count_sha256": tensor_sha256(self._document_token_count),
            "top1_correct_sha256": tensor_sha256(self._top1_correct),
            "ce_sum_sha256": tensor_sha256(self._ce_sum),
        })

    def _require_pristine(self) -> None:
        if self._compute_sha256() != self._expected_sha256:
            raise RuntimeError("stage sufficient statistics mutated")

    @property
    def sha256(self) -> str:
        self._require_pristine()
        return self._expected_sha256

    @property
    def role(self) -> str:
        return self._role

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def authority_sha256(self) -> str:
        return self._authority_sha256

    @property
    def ordered_document_ids_sha256(self) -> str:
        return self._ordered_document_ids_sha256

    @property
    def document_count(self) -> int:
        return len(self._document_token_count)

    @property
    def document_token_count(self) -> torch.Tensor:
        self._require_pristine()
        return self._document_token_count.clone()

    @property
    def top1_correct(self) -> torch.Tensor:
        self._require_pristine()
        return self._top1_correct.clone()

    @property
    def ce_sum(self) -> torch.Tensor:
        self._require_pristine()
        return self._ce_sum.clone()


def bootstrap_multiplicities(role: str, document_count: int) -> torch.Tensor:
    if role not in ROLE_NAMES or type(document_count) is not int or document_count <= 1:
        raise ValueError("bootstrap role/document count changed")
    generator = torch.Generator(device="cpu").manual_seed(cross.BOOTSTRAP_SEEDS[role])
    sampled = torch.randint(
        document_count, (cross.BOOTSTRAP_DRAWS, document_count), generator=generator,
    )
    weights = torch.zeros(
        (cross.BOOTSTRAP_DRAWS, document_count), dtype=torch.float64,
    )
    weights.scatter_add_(1, sampled, torch.ones_like(sampled, dtype=torch.float64))
    if not torch.equal(weights.sum(1), torch.full(
        (cross.BOOTSTRAP_DRAWS,), document_count, dtype=torch.float64,
    )):
        raise RuntimeError("bootstrap multiplicity realization changed")
    return weights.contiguous()


def _require_compatible(stages: Sequence[StageStatistics]) -> None:
    if not stages or len({stage.stage for stage in stages}) != len(stages) or any(
        not isinstance(stage, StageStatistics) for stage in stages
    ):
        raise ValueError("staged scorer capabilities are malformed")
    first = stages[0]
    if any(
        stage.role != first.role or stage.authority_sha256 != first.authority_sha256
        or stage.ordered_document_ids_sha256 != first.ordered_document_ids_sha256
        or stage.document_count != first.document_count
        or not torch.equal(stage.document_token_count, first.document_token_count)
        for stage in stages[1:]
    ):
        raise RuntimeError("stage capabilities do not share one role/support authority")


def _cost_draws(
    stages: Sequence[StageStatistics], target: str, weights: torch.Tensor,
) -> torch.Tensor:
    _require_compatible(stages)
    first = stages[0]
    if target not in {"ce_nats", "top1_pp"} or weights.ndim != 2 or (
        weights.shape[1] != first.document_count
    ) or weights.dtype != torch.float64 or not bool(torch.isfinite(weights).all()) or (
        bool((weights < 0).any())
    ):
        raise ValueError("cost-draw request changed")
    denominator = weights @ first.document_token_count.double()
    if bool((denominator <= 0).any()):
        raise RuntimeError("a resample has no scored tokens")
    rates = torch.full((len(weights), 8, 8), math.nan, dtype=torch.float64)
    for stage in stages:
        numerator = weights @ (
            stage.ce_sum if target == "ce_nats" else stage.top1_correct.double()
        )
        stage_rate = numerator / denominator[:, None]
        for column, (i, j) in enumerate(STAGE_CELLS[stage.stage]):
            rates[:, i, j] = stage_rate[:, column]
    anchor = rates[:, :1, :1]
    return (
        rates - anchor if target == "ce_nats" else 100.0 * (anchor - rates)
    ).contiguous()


def point_and_bootstrap_costs(
    stages: Sequence[StageStatistics], target: str,
) -> torch.Tensor:
    _require_compatible(stages)
    point = torch.ones((1, stages[0].document_count), dtype=torch.float64)
    bootstrap = bootstrap_multiplicities(stages[0].role, stages[0].document_count)
    return _cost_draws(stages, target, torch.cat((point, bootstrap), dim=0))


def _interaction_entries(cost: torch.Tensor, cells: Sequence[cross.Cell]) -> torch.Tensor:
    return torch.stack([
        cost[:, i, j] - cost[:, i, 0] - cost[:, 0, j] + cost[:, 0, 0]
        for i, j in cells
    ], dim=1)


def batched_cross_prediction(
    cost: torch.Tensor, rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total prediction, pivot condition, and singular-draw mask."""

    fit_cells = cross.RANK3_DISCOVERY_CELLS if rank == 3 else (
        cross.RANK4_FIT_CELLS if rank == 4 else None
    )
    if fit_cells is None or cost.ndim != 3 or tuple(cost.shape[1:]) != (8, 8) or (
        cost.dtype != torch.float64
    ) or not bool(torch.isfinite(torch.stack([
        cost[:, i, j] for i, j in fit_cells
    ])).all()):
        raise ValueError("batched cross input/capability changed")
    rows = cross.PIVOT_ROWS[rank]
    columns = cross.PIVOT_COLUMNS[rank]

    def delta(i: int, j: int) -> torch.Tensor:
        return cost[:, i, j] - cost[:, i, 0] - cost[:, 0, j] + cost[:, 0, 0]

    pivot = torch.stack([
        torch.stack([delta(i, j) for j in columns], dim=1) for i in rows
    ], dim=1)
    singular_values = torch.linalg.svdvals(pivot)
    threshold = PIVOT_RELATIVE_TOLERANCE * torch.maximum(
        singular_values[:, 0], torch.ones_like(singular_values[:, 0]),
    )
    singular = singular_values[:, -1] <= threshold
    condition = singular_values[:, 0] / singular_values[:, -1].clamp_min(
        torch.finfo(torch.float64).tiny
    )
    prediction = torch.full_like(cost, math.nan)
    valid = ~singular
    if bool(valid.any()):
        left = torch.stack([
            torch.stack([delta(i, j) for j in columns], dim=1) for i in range(8)
        ], dim=1)[valid]
        right = torch.stack([
            torch.stack([delta(i, j) for j in range(8)], dim=1) for i in rows
        ], dim=1)[valid]
        predicted_delta = left @ torch.linalg.solve(pivot[valid], right)
        additive = torch.stack([
            torch.stack([
                cost[:, i, 0] + cost[:, 0, j] - cost[:, 0, 0]
                for j in range(8)
            ], dim=1) for i in range(8)
        ], dim=1)[valid]
        prediction[valid] = additive + predicted_delta
    return prediction.contiguous(), condition.contiguous(), singular.contiguous()


def batched_als_prediction(
    cost: torch.Tensor, rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matched rank-r ALS interaction prediction under the exact frozen rule."""

    entries = cross.cross_cells(rank) if rank in (3, 4) else ()
    fit_cells = cross.RANK3_DISCOVERY_CELLS if rank == 3 else (
        cross.RANK4_FIT_CELLS if rank == 4 else None
    )
    if fit_cells is None or cost.ndim != 3 or tuple(cost.shape[1:]) != (8, 8) or (
        cost.dtype != torch.float64
    ) or not bool(torch.isfinite(torch.stack([
        cost[:, i, j] for i, j in fit_cells
    ])).all()):
        raise ValueError("ALS input/capability changed")
    target_raw = _interaction_entries(cost, entries)
    scale = target_raw.square().mean(1).sqrt()
    failed = scale <= ZERO_RMS_TOLERANCE
    output = torch.full((len(cost), 8, 8), math.nan, dtype=torch.float64)
    valid = ~failed
    if not bool(valid.any()):
        return output, failed
    target = target_raw[valid] / scale[valid, None]
    draws = len(target)
    left = torch.empty(
        (draws, cross.ALS_RESTARTS, 7, rank), dtype=torch.float64,
    )
    right = torch.empty_like(left)
    for restart in range(cross.ALS_RESTARTS):
        generator = torch.Generator(device="cpu").manual_seed(
            cross.ALS_SEED + 1000 * rank + restart
        )
        left[:, restart] = torch.randn(
            (7, rank), generator=generator, dtype=torch.float64,
        ) / math.sqrt(rank)
        right[:, restart] = torch.randn(
            (7, rank), generator=generator, dtype=torch.float64,
        ) / math.sqrt(rank)
    penalty = len(entries) * cross.ALS_RELATIVE_RIDGE / (7 * rank)
    identity = torch.eye(rank, dtype=torch.float64)
    entry_index = {cell: index for index, cell in enumerate(entries)}
    for _sweep in range(cross.ALS_SWEEPS):
        next_rows = []
        for i in range(1, 8):
            columns = [j for j in range(1, 8) if (i, j) in entry_index]
            indices = [entry_index[(i, j)] for j in columns]
            design = right[:, :, [j - 1 for j in columns], :]
            response = target[:, indices]
            next_rows.append(torch.linalg.solve(
                design.transpose(-1, -2) @ design + penalty * identity,
                (
                    design.transpose(-1, -2)
                    @ response[:, None, :, None]
                ).squeeze(-1),
            ))
        left = torch.stack(next_rows, dim=2)
        next_columns = []
        for j in range(1, 8):
            rows = [i for i in range(1, 8) if (i, j) in entry_index]
            indices = [entry_index[(i, j)] for i in rows]
            design = left[:, :, [i - 1 for i in rows], :]
            response = target[:, indices]
            next_columns.append(torch.linalg.solve(
                design.transpose(-1, -2) @ design + penalty * identity,
                (
                    design.transpose(-1, -2)
                    @ response[:, None, :, None]
                ).squeeze(-1),
            ))
        right = torch.stack(next_columns, dim=2)

    fitted_entries = torch.stack([
        (left[:, :, i - 1] * right[:, :, j - 1]).sum(-1) for i, j in entries
    ], dim=-1)
    objective = (fitted_entries - target[:, None]).square().mean(-1) + (
        cross.ALS_RELATIVE_RIDGE * (
            left.square().mean(dim=(-2, -1)) + right.square().mean(dim=(-2, -1))
        )
    )
    if not bool(torch.isfinite(objective).all()):
        raise RuntimeError("ALS produced a non-finite objective")
    selected = _select_restart(objective)
    index = torch.arange(draws)
    interaction = left[index, selected] @ right[index, selected].transpose(1, 2)
    interaction *= scale[valid, None, None]
    additive = torch.stack([
        torch.stack([
            cost[:, i, 0] + cost[:, 0, j] - cost[:, 0, 0]
            for j in range(8)
        ], dim=1) for i in range(8)
    ], dim=1)[valid]
    total = additive.clone()
    total[:, 1:, 1:] += interaction
    output[valid] = total
    return output.contiguous(), failed.contiguous()


def _select_restart(objective: torch.Tensor) -> torch.Tensor:
    """Select the first minimum exactly; torch.argmin's tie rule is contractual."""

    if not torch.is_tensor(objective) or objective.ndim != 2 or objective.shape[1] != (
        cross.ALS_RESTARTS
    ) or objective.dtype != torch.float64 or not bool(torch.isfinite(objective).all()):
        raise ValueError("ALS restart objective changed")
    return objective.argmin(1)


def _metric_vectors(
    cost: torch.Tensor, prediction: torch.Tensor, als: torch.Tensor,
    cells: Sequence[cross.Cell],
) -> dict[str, torch.Tensor]:
    actual = torch.stack([cost[:, i, j] for i, j in cells], dim=1)
    estimated = torch.stack([prediction[:, i, j] for i, j in cells], dim=1)
    als_estimated = torch.stack([als[:, i, j] for i, j in cells], dim=1)
    additive = torch.stack([
        cost[:, i, 0] + cost[:, 0, j] - cost[:, 0, 0] for i, j in cells
    ], dim=1)
    actual_delta = actual - additive
    predicted_delta = estimated - additive
    error = estimated - actual
    additive_error = additive - actual
    als_error = als_estimated - actual
    centered = actual - actual.mean(1, keepdim=True)
    metrics = {
        "rmse": error.square().mean(1).sqrt(),
        "max_abs_error": error.abs().max(1).values,
        "r2": 1.0 - error.square().sum(1) / centered.square().sum(1),
        "interaction_nre": (
            (predicted_delta - actual_delta).square().sum(1)
            / actual_delta.square().sum(1)
        ).sqrt(),
        "additive_rmse": additive_error.square().mean(1).sqrt(),
        "rmse_over_additive": (
            error.square().mean(1).sqrt()
            / additive_error.square().mean(1).sqrt()
        ),
        "als_rmse": als_error.square().mean(1).sqrt(),
    }
    return {name: value.contiguous() for name, value in metrics.items()}


def _summary(values: torch.Tensor) -> dict[str, float]:
    if values.shape != (cross.BOOTSTRAP_DRAWS + 1,) or values.dtype != (
        torch.float64
    ) or not bool(torch.isfinite(values).all()):
        raise RuntimeError("metric lacks a finite point and every bootstrap draw")
    bootstrap = values[1:]
    return {
        "point": float(values[0]),
        "q025": float(torch.quantile(bootstrap, 0.025, interpolation="linear")),
        "q95": float(torch.quantile(bootstrap, 0.95, interpolation="linear")),
        "q975": float(torch.quantile(bootstrap, 0.975, interpolation="linear")),
    }


def _subgroup_ratios(
    cost: torch.Tensor, prediction: torch.Tensor, cells: Sequence[cross.Cell],
) -> dict[str, float]:
    output: dict[str, float] = {}
    groups = {
        **{f"prefix_{i}": tuple(cell for cell in cells if cell[0] == i) for i in sorted({
            cell[0] for cell in cells
        })},
        **{f"suffix_{j}": tuple(cell for cell in cells if cell[1] == j) for j in sorted({
            cell[1] for cell in cells
        })},
    }
    for name, group in groups.items():
        actual = torch.tensor([float(cost[0, i, j]) for i, j in group])
        estimated = torch.tensor([float(prediction[0, i, j]) for i, j in group])
        additive = torch.tensor([
            float(cost[0, i, 0] + cost[0, 0, j] - cost[0, 0, 0]) for i, j in group
        ])
        denominator = (additive - actual).square().mean().sqrt()
        output[name] = float((estimated - actual).square().mean().sqrt() / denominator)
    return output


def score_rank(
    discovery: StageStatistics, validation: StageStatistics,
    heldout: StageStatistics | None, *, rank: int,
) -> dict[str, Any]:
    """Score one rank with exact staged capabilities; rank three cannot receive heldout."""

    if rank == 3:
        if heldout is not None or discovery.stage != "discovery" or validation.stage != (
            "validation"
        ):
            raise ValueError("rank-three scorer received the wrong capabilities")
        stages = (discovery, validation)
        score_cells = cross.RANK4_VALIDATION_CELLS
    elif rank == 4:
        if not isinstance(heldout, StageStatistics) or discovery.stage != (
            "discovery"
        ) or validation.stage != "validation" or heldout.stage != "heldout":
            raise ValueError("rank-four scorer received the wrong capabilities")
        stages = (discovery, validation, heldout)
        score_cells = cross.HELDOUT_CELLS
    else:
        raise ValueError("score rank must be three or four")
    _require_compatible(stages)
    targets: dict[str, Any] = {}
    ce_gate_metrics: dict[str, Any] | None = None
    ce_condition: dict[str, float] | None = None
    ce_singular = True
    ce_subgroups: dict[str, float] = {}
    for target in ("ce_nats", "top1_pp"):
        fit_stages = (discovery,) if rank == 3 else (discovery, validation)
        fit_cost = point_and_bootstrap_costs(fit_stages, target)
        score_cost = point_and_bootstrap_costs(stages, target)
        predicted, condition, singular = batched_cross_prediction(fit_cost, rank)
        als, als_failed = batched_als_prediction(fit_cost, rank)
        failed = singular | als_failed
        metrics: dict[str, Any] = {}
        if bool(failed.any()):
            metrics = {"status": "failed_nonfinite_draw"}
        else:
            vectors = _metric_vectors(score_cost, predicted, als, score_cells)
            if any(not bool(torch.isfinite(value).all()) for value in vectors.values()):
                metrics = {"status": "failed_zero_or_nonfinite_metric_denominator"}
            else:
                metrics = {name: _summary(value) for name, value in vectors.items()}
                metrics["pivot_condition"] = _summary(condition)
                metrics["subgroup_rmse_over_additive_point"] = _subgroup_ratios(
                    score_cost, predicted, score_cells,
                )
        metrics["singular_or_zero_rms_draw_count"] = int(failed[1:].sum())
        metrics["point_failed"] = bool(failed[0])
        targets[target] = metrics
        if target == "ce_nats" and not bool(failed.any()) and (
            metrics.get("status") is None
        ):
            ce_gate_metrics = metrics
            ce_condition = metrics["pivot_condition"]
            ce_singular = False
            ce_subgroups = metrics["subgroup_rmse_over_additive_point"]

    gates: dict[str, bool]
    if ce_singular or ce_gate_metrics is None or ce_condition is None:
        gates = {"finite_every_draw": False}
    else:
        gates = {
            "finite_every_draw": True,
            "pivot_condition": (
                ce_condition["point"] <= 20.0 and ce_condition["q95"] <= 25.0
            ),
            "interaction_nre": (
                ce_gate_metrics["interaction_nre"]["point"] <= 0.50
                and ce_gate_metrics["interaction_nre"]["q975"] <= 0.65
            ),
            "additive_ratio": (
                ce_gate_metrics["rmse_over_additive"]["point"] <= 0.75
                and ce_gate_metrics["rmse_over_additive"]["q975"] <= 0.90
            ),
            "positive_r2": (
                ce_gate_metrics["r2"]["point"] > 0.0
                and ce_gate_metrics["r2"]["q025"] > 0.0
            ),
            "beats_als_point": (
                ce_gate_metrics["rmse"]["point"]
                <= ce_gate_metrics["als_rmse"]["point"]
            ),
        }
        if rank == 4:
            gates.update({
                "absolute_rmse": (
                    ce_gate_metrics["rmse"]["point"] <= 0.10
                    and ce_gate_metrics["rmse"]["q975"] <= 0.15
                ),
                "r2_at_least_half": ce_gate_metrics["r2"]["point"] >= 0.50,
                "every_subgroup_beats_additive": all(
                    value <= 1.0 for value in ce_subgroups.values()
                ),
            })
    return {
        "schema_version": SCHEMA_VERSION,
        "role": discovery.role,
        "rank": rank,
        "stage": "rank3_validation" if rank == 3 else "rank4_heldout",
        "targets": targets,
        "ce_gates": gates,
        "ce_useful_pass": bool(gates) and all(gates.values()),
        "stage_payload_sha256s": {
            stage.stage: stage.sha256 for stage in stages
        },
    }


def validate_contract() -> None:
    if set(STAGE_CELLS) != {"discovery", "validation", "heldout"} or tuple(
        cell for stage in STAGE_CELLS.values() for cell in stage
    ) != (*cross.RANK3_DISCOVERY_CELLS, *cross.RANK4_VALIDATION_CELLS,
          *cross.HELDOUT_CELLS) or len(set(
        cell for stage in STAGE_CELLS.values() for cell in stage
    )) != 64:
        raise RuntimeError("staged statistics partition changed")


validate_contract()
