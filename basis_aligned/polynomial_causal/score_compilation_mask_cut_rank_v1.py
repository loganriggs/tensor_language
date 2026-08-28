#!/usr/bin/env python3
"""Receipt-bound, one-shot CPU scorer for compilation-mask cut-rank v1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import secrets
from typing import Any, Mapping, Sequence

import torch

import compilation_mask_cut_rank_v1 as cut
import compilation_mask_cut_rank_v1_gpu_adapter as adapter
import compilation_mask_cut_rank_v1_measurements as measurement


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_MEASUREMENT_RECEIPT = (
    HERE / "compilation_mask_cut_rank_v1_measurement_wave_v1_receipt.json"
)
DEFAULT_NAMESPACE = "compilation_mask_cut_rank_v1_score_v1"
SCORING_AMENDMENT = (
    "basis_aligned/polynomial_causal/COMPILATION_MASK_CUT_RANK_V1_SCORING_AMENDMENT.md"
)
HISTORICAL_SINGLETON_SOURCE = (
    "basis_aligned/bilinear_quotient/ops/site_cost_table_results.json"
)
SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/COMPILATION_MASK_CUT_RANK_V1_PREREGISTRATION.md",
    SCORING_AMENDMENT,
    "basis_aligned/polynomial_causal/compilation_mask_cut_rank_v1.py",
    "basis_aligned/polynomial_causal/compilation_mask_cut_rank_v1_measurements.py",
    "basis_aligned/polynomial_causal/score_compilation_mask_cut_rank_v1.py",
    HISTORICAL_SINGLETON_SOURCE,
)
BOOTSTRAP_REPETITIONS = 2_000
BOOTSTRAP_SEED = 2_026_082_851
LOWER_ORDER_ONE_INDEXED = 100
UPPER_ORDER_ONE_INDEXED = 1_900
HISTORICAL_LIVE_TOP1 = 0.3932
HISTORICAL_FULL_TOP1 = 0.1355
HISTORICAL_STAKE_PP = 100.0 * (HISTORICAL_LIVE_TOP1 - HISTORICAL_FULL_TOP1)


def _logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class ScorePaths:
    authority: Path
    development: Path
    result: Path
    receipt: Path
    failure: Path
    lock: Path


def score_paths(directory: Path, namespace: str) -> ScorePaths:
    if not namespace or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in namespace
    ):
        raise ValueError("score namespace is not a safe lowercase identifier")
    root = directory.resolve()
    return ScorePaths(
        authority=root / f"{namespace}_authority.json",
        development=root / f"{namespace}_development.json",
        result=root / f"{namespace}_results.json",
        receipt=root / f"{namespace}_receipt.json",
        failure=root / f"{namespace}_failure.json",
        lock=root / f".{namespace}.lock",
    )


def _measurement_receipt(value: Mapping[str, Any]) -> measurement.MeasurementReceipt:
    copied = dict(value)
    for name in (
        "cell_receipt_sha256s", "top1_correct_row_sha256s",
        "ce_sum_row_sha256s", "statistics_sha256s",
    ):
        copied[name] = tuple(copied[name])
    return measurement.MeasurementReceipt(**copied)


def _measurement_authority(value: Mapping[str, Any]) -> measurement.MeasurementWaveAuthority:
    copied = dict(value)
    copied["program_realization_sha256s"] = tuple(
        copied["program_realization_sha256s"]
    )
    return measurement.MeasurementWaveAuthority(**copied)


def validate_and_load_bundle(
    receipt_path: Path,
) -> tuple[measurement.FinalizedMeasurementBundle, dict[str, Any]]:
    """Validate every published binding before returning sealed sufficient statistics."""

    receipt_path = receipt_path.resolve()
    outer = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_outer = {
        "schema_version", "status", "authorized_for_final_role",
        "authority_path", "authority_file_sha256", "payload_path",
        "payload_file_sha256", "measurement_authority_sha256",
        "measurement_receipt", "measurement_receipt_sha256",
        "source_closure_sha256", "program_bank_sha256", "row_wave_sha256",
    }
    if set(outer) != expected_outer or outer["schema_version"] != 1 or outer[
        "status"
    ] != "complete_discovery_measurement_payload" or outer[
        "authorized_for_final_role"
    ] is not False:
        raise RuntimeError("measurement publication receipt schema/scope changed")
    typed_receipt = _measurement_receipt(outer["measurement_receipt"])
    if typed_receipt.sha256 != outer["measurement_receipt_sha256"] or (
        typed_receipt.authority_sha256 != outer["measurement_authority_sha256"]
    ) or typed_receipt.source_closure_sha256 != outer[
        "source_closure_sha256"
    ] or typed_receipt.program_bank_sha256 != outer["program_bank_sha256"]:
        raise RuntimeError("measurement receipt logical bindings differ")

    authority_path = Path(outer["authority_path"]).resolve()
    payload_path = Path(outer["payload_path"]).resolve()
    if authority_path.parent != receipt_path.parent or payload_path.parent != (
        receipt_path.parent
    ) or adapter.file_sha256(authority_path) != outer[
        "authority_file_sha256"
    ] or adapter.file_sha256(payload_path) != outer["payload_file_sha256"]:
        raise RuntimeError("measurement publication file binding differs")
    authority_outer = json.loads(authority_path.read_text(encoding="utf-8"))
    authority = _measurement_authority(authority_outer["measurement_authority"])
    if authority.sha256 != outer["measurement_authority_sha256"] or authority_outer[
        "measurement_authority_sha256"
    ] != authority.sha256 or authority_outer["source_closure_sha256"] != (
        typed_receipt.source_closure_sha256
    ) or authority_outer["program_bank_sha256"] != typed_receipt.program_bank_sha256:
        raise RuntimeError("measurement authority replay differs from receipt")

    raw = torch.load(payload_path, map_location="cpu", weights_only=True)
    expected_payload = {
        "schema_version", "authority_sha256", "ordered_document_ids_sha256",
        "document_row_count", "document_token_count", "top1_correct", "ce_sum",
        "per_document_payload_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != expected_payload or raw[
        "schema_version"
    ] != 1 or raw["authority_sha256"] != authority.sha256:
        raise RuntimeError("measurement payload schema/authority changed")
    payload = measurement.PerDocumentSufficientStatistics(
        authority_sha256=raw["authority_sha256"],
        ordered_document_ids_sha256=raw["ordered_document_ids_sha256"],
        document_row_count=raw["document_row_count"],
        document_token_count=raw["document_token_count"],
        top1_correct=raw["top1_correct"], ce_sum=raw["ce_sum"],
    )
    if payload.sha256 != raw["per_document_payload_sha256"]:
        raise RuntimeError("measurement payload content hash differs")
    return measurement.FinalizedMeasurementBundle(
        payload=payload, receipt=typed_receipt,
    ), outer


def load_historical_top1_singletons(repo: Path) -> cut.FrozenSingletonCosts:
    path = (repo / HISTORICAL_SINGLETON_SOURCE).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_names = {
        f"{kind}{layer}" for layer in range(1, 18) for kind in ("attn", "mlp")
    }
    if set(raw.get("single_site_cost", {})) != expected_names or raw.get(
        "alpha"
    ) != cut.S1834_ALPHA or raw.get("role") != "skip7000":
        raise RuntimeError("historical singleton source schema/currency changed")
    costs = {
        (kind, layer): HISTORICAL_STAKE_PP * float(
            raw["single_site_cost"][f"{kind}{layer}"]
        )
        for layer in range(1, 18) for kind in ("attn", "mlp")
    }
    return cut.FrozenSingletonCosts(
        target="top1_pp", costs=costs,
        source_sha256=adapter.file_sha256(path),
    )


def document_bootstrap_weights(
    document_count: int, *, repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> torch.Tensor:
    if type(document_count) is not int or document_count <= 1 or repetitions != (
        BOOTSTRAP_REPETITIONS
    ) or seed != BOOTSTRAP_SEED:
        raise ValueError("cut-rank bootstrap realization changed")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sampled = torch.randint(
        document_count, (repetitions, document_count), generator=generator,
    )
    weights = torch.zeros((repetitions, document_count), dtype=torch.float64)
    weights.scatter_add_(1, sampled, torch.ones_like(sampled, dtype=torch.float64))
    if not torch.equal(weights.sum(1), torch.full(
        (repetitions,), document_count, dtype=torch.float64,
    )):
        raise RuntimeError("document bootstrap multiplicities changed")
    return weights.contiguous()


def _pooled_costs(
    top1_correct: torch.Tensor, ce_sum: torch.Tensor,
    token_count: torch.Tensor, weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = weights @ token_count.double()
    if bool((denominator <= 0).any()):
        raise RuntimeError("bootstrap scored-token denominator is empty")
    accuracy = (weights @ top1_correct.double()) / denominator[:, None]
    mean_ce = (weights @ ce_sum.double()) / denominator[:, None]
    return (
        (100.0 * (accuracy[:, :1] - accuracy)).contiguous(),
        (mean_ce - mean_ce[:, :1]).contiguous(),
    )


def _interaction(costs: torch.Tensor) -> torch.Tensor:
    grid = costs.reshape(-1, 8, 8)
    return (grid - grid[:, :, :1] - grid[:, :1, :] + grid[:, :1, :1]).contiguous()


def _indices(cells: Sequence[cut.Cell]) -> torch.Tensor:
    return torch.tensor([8 * i + j for i, j in cells], dtype=torch.long)


def _batched_fixed_rank_predictions(
    interaction: torch.Tensor, *, rank: int, ridge: float,
) -> torch.Tensor:
    """Exact batched replay of the registered fixed-rank eight-restart ALS."""

    if interaction.ndim != 3 or tuple(interaction.shape[1:]) != (8, 8) or rank not in (
        cut.RANK_GRID
    ) or ridge not in cut.RIDGE_GRID or interaction.dtype != torch.float64 or not bool(
        torch.isfinite(interaction).all()
    ):
        raise ValueError("batched fixed-rank inputs changed")
    draws = len(interaction)
    entries = cut.TRAIN_CELLS
    target_raw = interaction[:, tuple(i for i, _ in entries), tuple(j for _, j in entries)]
    scale = target_raw.square().mean(1).sqrt()
    if bool((scale <= cut.ZERO_INTERACTION_TOLERANCE).any()):
        raise RuntimeError("bootstrap draw lost the registered rank signal")
    target = target_raw / scale[:, None]
    restarts = cut.ALS_RESTARTS
    left = torch.empty((draws, restarts, 7, rank), dtype=torch.float64)
    right = torch.empty_like(left)
    filled = torch.zeros((draws, 7, 7), dtype=torch.float64)
    for entry, (i, j) in enumerate(entries):
        filled[:, i - 1, j - 1] = target[:, entry]
    u, singular, vh = torch.linalg.svd(filled, full_matrices=False)
    root = torch.sqrt(singular[:, :rank].clamp_min(1e-12))
    left[:, 0] = u[:, :, :rank] * root[:, None, :]
    right[:, 0] = vh[:, :rank, :].transpose(1, 2) * root[:, None, :]
    initialization_scale = torch.maximum(
        target.square().mean(1).sqrt(), torch.full((draws,), 1e-3)
    ).sqrt()
    for restart in range(1, restarts):
        generator = torch.Generator(device="cpu").manual_seed(
            cut.ALS_SEED + 1000 * rank + restart
        )
        base_left = torch.randn((7, rank), generator=generator, dtype=torch.float64)
        base_right = torch.randn((7, rank), generator=generator, dtype=torch.float64)
        left[:, restart] = initialization_scale[:, None, None] * base_left
        right[:, restart] = initialization_scale[:, None, None] * base_right

    row_columns = torch.tensor([
        [column - 1 for row, column in entries if row == i] for i in range(1, 8)
    ], dtype=torch.long)
    row_entry_indices = torch.tensor([
        [index for index, (row, _column) in enumerate(entries) if row == i]
        for i in range(1, 8)
    ], dtype=torch.long)
    column_rows = torch.tensor([
        [row - 1 for row, column in entries if column == j] for j in range(1, 8)
    ], dtype=torch.long)
    column_entry_indices = torch.tensor([
        [index for index, (_row, column) in enumerate(entries) if column == j]
        for j in range(1, 8)
    ], dtype=torch.long)
    identity = torch.eye(rank, dtype=torch.float64)
    penalty = len(entries) * ridge / (7 * rank)
    previous = torch.full((draws, restarts), math.inf, dtype=torch.float64)
    active = torch.ones((draws, restarts), dtype=torch.bool)

    def objective(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        prediction = torch.stack([
            (a[:, :, i - 1] * b[:, :, j - 1]).sum(-1) for i, j in entries
        ], dim=-1)
        regularizer = 0.5 * ridge * (
            a.square().mean(dim=(-2, -1)) + b.square().mean(dim=(-2, -1))
        )
        return (prediction - target[:, None, :]).square().mean(-1) + regularizer

    for _iteration in range(1, cut.ALS_MAX_ITERATIONS + 1):
        row_design = right[:, :, row_columns, :]
        row_target = target[:, row_entry_indices]
        next_left = torch.linalg.solve(
            row_design.transpose(-1, -2) @ row_design + penalty * identity,
            (
                row_design.transpose(-1, -2)
                @ row_target[:, None, :, :, None]
            ).squeeze(-1),
        )
        column_design = next_left[:, :, column_rows, :]
        column_target = target[:, column_entry_indices]
        next_right = torch.linalg.solve(
            column_design.transpose(-1, -2) @ column_design + penalty * identity,
            (
                column_design.transpose(-1, -2)
                @ column_target[:, None, :, :, None]
            ).squeeze(-1),
        )
        left = torch.where(active[:, :, None, None], next_left, left)
        right = torch.where(active[:, :, None, None], next_right, right)
        current = objective(left, right)
        converged = active & torch.isfinite(previous) & (
            (previous - current).abs()
            <= cut.ALS_RELATIVE_TOLERANCE * torch.maximum(
                torch.ones_like(previous), previous.abs(),
            )
        )
        active = active & ~converged
        previous = torch.where(active, current, previous)
        if not bool(active.any()):
            break
    final_objective = objective(left, right)
    if not bool(torch.isfinite(final_objective).all()):
        raise RuntimeError("batched ALS produced a non-finite objective")
    selected = final_objective.argmin(1)
    index = torch.arange(draws)
    selected_left = left[index, selected]
    selected_right = right[index, selected]
    prediction = selected_left @ selected_right.transpose(1, 2)
    return (prediction * scale[:, None, None]).contiguous()


def _baseline_predictions(
    name: str, costs: torch.Tensor,
    singleton: cut.FrozenSingletonCosts | None,
) -> torch.Tensor:
    grid = costs.reshape(-1, 8, 8)
    heldout = cut.HELDOUT_CELLS
    if name == "additive_anchors":
        return torch.stack([
            grid[:, i, 0] + grid[:, 0, j] - grid[:, 0, 0] for i, j in heldout
        ], dim=1)
    if name.startswith("count_depth_type_ridge/"):
        ridge = float(name.rsplit("/", 1)[1])
        design = torch.stack([cut._count_depth_features(cell) for cell in cut.FIT_CELLS])
        feature_scale = design[:, 1:].square().mean(0).sqrt().clamp_min(1e-12)
        standardized = design.clone()
        standardized[:, 1:] /= feature_scale
        penalty = torch.eye(design.shape[1], dtype=torch.float64)
        penalty[0, 0] = 0.0
        inverse = torch.linalg.inv(
            standardized.T @ standardized + len(cut.FIT_CELLS) * ridge * penalty
        )
        target = costs[:, _indices(cut.FIT_CELLS)]
        coefficient = (inverse @ standardized.T @ target.T).T
        heldout_design = torch.stack([
            cut._count_depth_features(cell) for cell in heldout
        ])
        heldout_design[:, 1:] /= feature_scale
        return (coefficient @ heldout_design.T).contiguous()
    if singleton is None:
        raise RuntimeError("selected singleton baseline lacks its frozen source")
    singleton_prediction = torch.tensor([
        cut._singleton_sum(cell, singleton.costs) for cell in heldout
    ], dtype=torch.float64)
    if name == "literal_singleton_sum":
        return singleton_prediction.expand(len(costs), -1).clone().contiguous()
    if name == "s1834_scaled_singleton_sum":
        return (
            cut.S1834_ALPHA * singleton_prediction
        ).expand(len(costs), -1).clone().contiguous()
    if name.startswith("monotone_quadratic_singleton/"):
        ridge = float(name.rsplit("/", 1)[1])
        fit_x = torch.tensor([
            cut._singleton_sum(cell, singleton.costs) for cell in cut.FIT_CELLS
        ], dtype=torch.float64)
        heldout_x = singleton_prediction
        design = torch.stack((
            torch.ones_like(heldout_x), heldout_x, heldout_x.square(),
        ), dim=1)
        output = torch.empty((len(costs), len(heldout)), dtype=torch.float64)
        fit_indices = _indices(cut.FIT_CELLS)
        for draw in range(len(costs)):
            beta = cut._solve_monotone_quadratic(
                fit_x, costs[draw, fit_indices], ridge=ridge,
            )
            output[draw] = design @ beta
        return output.contiguous()
    raise RuntimeError("selected baseline family is unknown")


def fixed_selection_bootstrap_metrics(
    costs: torch.Tensor, *, development: cut.CutRankDevelopment,
    target: str, singleton: cut.FrozenSingletonCosts | None,
) -> dict[str, torch.Tensor]:
    if costs.ndim != 2 or tuple(costs.shape[1:]) != (64,) or costs.dtype != (
        torch.float64
    ) or target not in {"top1_pp", "ce_nats"}:
        raise ValueError("fixed-selection bootstrap costs are malformed")
    summary = development.top1_summary if target == "top1_pp" else development.ce_summary
    interaction = _interaction(costs)
    if summary.selected_rank == 0:
        predicted_interaction = torch.zeros(
            (len(costs), 7, 7), dtype=torch.float64,
        )
    else:
        if summary.selected_ridge is None:
            raise RuntimeError("selected rank lacks a ridge")
        predicted_interaction = _batched_fixed_rank_predictions(
            interaction, rank=summary.selected_rank, ridge=summary.selected_ridge,
        )
    heldout = cut.HELDOUT_CELLS
    actual = costs[:, _indices(heldout)]
    actual_interaction = torch.stack([
        interaction[:, i, j] for i, j in heldout
    ], dim=1)
    predicted_delta = torch.stack([
        predicted_interaction[:, i - 1, j - 1] for i, j in heldout
    ], dim=1)
    grid = costs.reshape(-1, 8, 8)
    additive = torch.stack([
        grid[:, i, 0] + grid[:, 0, j] - grid[:, 0, 0] for i, j in heldout
    ], dim=1)
    predicted = additive + predicted_delta
    baseline = _baseline_predictions(summary.selected_baseline, costs, singleton)
    error = predicted - actual
    baseline_error = baseline - actual
    rmse = error.square().mean(1).sqrt()
    baseline_rmse = baseline_error.square().mean(1).sqrt()
    interaction_denominator = actual_interaction.square().sum(1)
    nre = torch.sqrt(
        (predicted_delta - actual_interaction).square().sum(1)
        / interaction_denominator
    )
    centered = actual - actual.mean(1, keepdim=True)
    r2 = 1.0 - error.square().sum(1) / centered.square().sum(1)
    ratio = rmse / baseline_rmse
    singular = torch.linalg.svdvals(interaction)
    spectral = torch.sqrt(
        singular[:, 2:].square().sum(1) / singular.square().sum(1)
    )
    output = {
        "interaction_nre": nre, "heldout_r2": r2,
        "rmse_ratio": ratio, "full_grid_rank2_spectral_tail_nre": spectral,
    }
    if any(not bool(torch.isfinite(value).all()) for value in output.values()):
        raise RuntimeError("a bootstrap metric has a missing/non-finite denominator")
    return output


def _one_sided_bounds(values: torch.Tensor) -> dict[str, Any]:
    if values.shape != (BOOTSTRAP_REPETITIONS,) or values.dtype != torch.float64 or (
        not bool(torch.isfinite(values).all())
    ):
        raise ValueError("bootstrap bound requires every frozen draw")
    ordered = values.sort().values
    return {
        "lower_95": float(ordered[LOWER_ORDER_ONE_INDEXED - 1]),
        "upper_95": float(ordered[UPPER_ORDER_ONE_INDEXED - 1]),
        "lower_order_statistic_one_indexed": LOWER_ORDER_ONE_INDEXED,
        "upper_order_statistic_one_indexed": UPPER_ORDER_ONE_INDEXED,
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
    }


def _point_observations(
    payload: measurement.PerDocumentSufficientStatistics,
    cells: Sequence[cut.Cell],
) -> dict[cut.Cell, cut.ObservedCell]:
    tokens = float(payload.document_token_count.sum())
    top1 = payload.top1_correct.sum(0).double() / tokens
    ce = payload.ce_sum.sum(0) / tokens
    return {
        cell: cut.ObservedCell(
            top1_accuracy=float(top1[8 * cell[0] + cell[1]]),
            mean_ce=float(ce[8 * cell[0] + cell[1]]),
        )
        for cell in cells
    }


def _development_payload(development: cut.CutRankDevelopment) -> dict[str, Any]:
    return {
        "status": "development_selection_frozen_before_heldout_finalization",
        "heldout_metrics_present": False,
        "top1": asdict(development.top1_summary),
        "ce": asdict(development.ce_summary),
    }


def _point_predicates(
    point: cut.FinalizedCutRankAnalysis, bounds: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    top1 = point.top1
    ce = point.ce
    return {
        "gate1_selected_rank_at_most_2_with_signal": (
            top1.selected_rank <= 2 and top1.rank_signal_present
        ),
        "gate2_top1_total_rmse_and_max_error": (
            top1.total_rmse <= 5.0 and top1.maximum_absolute_error <= 10.0
        ),
        "gate3_top1_interaction_nre": (
            top1.interaction_nre is not None and top1.interaction_nre <= 0.50
            and bounds["top1_interaction_nre"]["upper_95"] <= 0.65
        ),
        "gate4_top1_heldout_r2": (
            top1.heldout_r2 is not None and top1.heldout_r2 >= 0.75
            and bounds["top1_heldout_r2"]["lower_95"] > 0.0
        ),
        "gate5_top1_paired_baseline_ratio": (
            top1.rmse_ratio is not None and top1.rmse_ratio <= 0.80
            and bounds["top1_rmse_ratio"]["upper_95"] < 1.0
        ),
        "gate6_top1_no_group_free_rider": all((
            top1.mlp5_group_rmse <= top1.mlp5_group_baseline_rmse,
            top1.non_mlp5_group_rmse <= top1.non_mlp5_group_baseline_rmse,
            top1.dense_deep_group_rmse <= top1.dense_deep_group_baseline_rmse,
            top1.sparse_deep_group_rmse <= top1.sparse_deep_group_baseline_rmse,
        )),
        "gate7_top1_full_grid_spectral_tail": (
            bounds["top1_full_grid_rank2_spectral_tail_nre"]["upper_95"] <= 0.50
        ),
        "gate8_ce_positive_r2_and_beats_available_baseline": (
            ce.heldout_r2 is not None and ce.heldout_r2 > 0.0
            and ce.total_rmse < ce.baseline_rmse
        ),
    }


def score_bundle(
    bundle: measurement.FinalizedMeasurementBundle, *, repo: Path,
    development_publisher: Any,
) -> dict[str, Any]:
    """Select on development, publish it, then unlock heldout exactly once."""

    singleton = load_historical_top1_singletons(repo)
    development = cut.prepare_development(
        _point_observations(bundle.payload, cut.DEVELOPMENT_CELLS),
        singleton_top1_pp=singleton,
    )
    development_value = _development_payload(development)
    development_publisher(development_value)
    point = cut.finalize_heldout(
        development, _point_observations(bundle.payload, cut.HELDOUT_CELLS),
    )

    weights = document_bootstrap_weights(bundle.payload.document_count)
    top1_costs, ce_costs = _pooled_costs(
        bundle.payload.top1_correct, bundle.payload.ce_sum,
        bundle.payload.document_token_count, weights,
    )
    top1_bootstrap = fixed_selection_bootstrap_metrics(
        top1_costs, development=development, target="top1_pp", singleton=singleton,
    )
    ce_bootstrap = fixed_selection_bootstrap_metrics(
        ce_costs, development=development, target="ce_nats", singleton=None,
    )
    bounds = {
        **{f"top1_{name}": _one_sided_bounds(value)
           for name, value in top1_bootstrap.items()},
        **{f"ce_{name}": _one_sided_bounds(value)
           for name, value in ce_bootstrap.items()},
    }
    predicates = _point_predicates(point, bounds)
    return {
        "status": "complete_nonpromotive_registered_ce_baseline_family_incomplete",
        "scientific_scope": "single_layer5_cut_discovery_only",
        "point": {"top1": asdict(point.top1), "ce": asdict(point.ce)},
        "bootstrap_bounds": bounds,
        "registered_predicates": predicates,
        "numerical_eight_gate_conjunction": all(predicates.values()),
        "registered_ce_baseline_family_complete": False,
        "useful_pass": None,
        "promotive": False,
        "interpretation": (
            "The original useful-pass conjunction is unevaluable because no sealed "
            "source provides the registered CE singleton-cost baseline family."
        ),
        "historical_top1_singleton": {
            "source_path": HISTORICAL_SINGLETON_SOURCE,
            "source_sha256": singleton.source_sha256,
            "content_sha256": singleton.content_sha256,
            "historical_live_top1": HISTORICAL_LIVE_TOP1,
            "historical_full_top1": HISTORICAL_FULL_TOP1,
            "stake_pp": HISTORICAL_STAKE_PP,
        },
        "bootstrap_contract": {
            "repetitions": BOOTSTRAP_REPETITIONS, "seed": BOOTSTRAP_SEED,
            "lower_order_statistic_one_indexed": LOWER_ORDER_ONE_INDEXED,
            "upper_order_statistic_one_indexed": UPPER_ORDER_ONE_INDEXED,
            "selection": "original development once; fixed pipeline refit per draw",
        },
    }


def run_score_transaction(
    *, measurement_receipt_path: Path, repo: Path, paths: ScorePaths,
) -> dict[str, Any]:
    if any(getattr(paths, name).exists() for name in (
        "authority", "development", "result", "receipt", "failure", "lock",
    )):
        raise RuntimeError("cut-rank score namespace is not pristine")
    lock = adapter.RunLock(paths.lock)
    lock.acquire()
    authority_written = False
    phase = "pre_outcome_authority"
    try:
        source = adapter.committed_source_closure(repo, SOURCE_PATHS)
        measurement_receipt_path = measurement_receipt_path.resolve()
        measurement_outer = json.loads(measurement_receipt_path.read_text(encoding="utf-8"))
        authority = {
            "schema_version": 1,
            "status": "frozen_before_score_payload_load",
            "authorized_for_final_role": False,
            "source_closure": asdict(source),
            "source_closure_sha256": source.sha256,
            "measurement_receipt_path": str(measurement_receipt_path),
            "measurement_receipt_file_sha256": adapter.file_sha256(
                measurement_receipt_path
            ),
            "measurement_receipt_sha256": measurement_outer.get(
                "measurement_receipt_sha256"
            ),
            "measurement_payload_file_sha256": measurement_outer.get(
                "payload_file_sha256"
            ),
            "scoring_amendment_sha256": adapter.file_sha256(repo / SCORING_AMENDMENT),
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "lower_order_statistic_one_indexed": LOWER_ORDER_ONE_INDEXED,
            "upper_order_statistic_one_indexed": UPPER_ORDER_ONE_INDEXED,
            "ce_singleton_family_available": False,
            "useful_pass_can_be_true": False,
            "nonce": secrets.token_hex(32),
        }
        authority["authority_sha256"] = _logical_sha256(authority)
        adapter._publish_bytes_create_only(paths.authority, _json_bytes(authority), lock)
        authority_written = True
        phase = "validate_and_load_measurement_bundle"
        bundle, validated_outer = validate_and_load_bundle(measurement_receipt_path)
        if validated_outer["measurement_receipt_sha256"] != authority[
            "measurement_receipt_sha256"
        ] or validated_outer["payload_file_sha256"] != authority[
            "measurement_payload_file_sha256"
        ]:
            raise RuntimeError("validated measurement differs from scoring authority")

        development_holder: dict[str, Any] = {}

        def publish_development(value: dict[str, Any]) -> None:
            if development_holder:
                raise RuntimeError("development selection publication repeated")
            wrapped = {
                **value,
                "scoring_authority_sha256": authority["authority_sha256"],
                "measurement_receipt_sha256": authority["measurement_receipt_sha256"],
            }
            wrapped["development_sha256"] = _logical_sha256(wrapped)
            adapter._publish_bytes_create_only(
                paths.development, _json_bytes(wrapped), lock,
            )
            development_holder.update(wrapped)

        phase = "development_then_one_shot_heldout_and_bootstrap"
        result = score_bundle(
            bundle, repo=repo, development_publisher=publish_development,
        )
        if not development_holder or not paths.development.exists():
            raise RuntimeError("heldout finalization lacked a published development selection")
        result.update({
            "scoring_authority_sha256": authority["authority_sha256"],
            "development_sha256": development_holder["development_sha256"],
            "measurement_receipt_sha256": authority["measurement_receipt_sha256"],
            "measurement_payload_file_sha256": authority[
                "measurement_payload_file_sha256"
            ],
            "source_closure_sha256": source.sha256,
        })
        result["result_sha256"] = _logical_sha256(result)
        phase = "publish_result"
        adapter._publish_bytes_create_only(paths.result, _json_bytes(result), lock)
        if json.loads(paths.result.read_text(encoding="utf-8")) != result:
            raise RuntimeError("installed cut-rank score result differs")
        receipt = {
            "schema_version": 1,
            "status": "complete_nonpromotive_score_receipt",
            "authorized_for_final_role": False,
            "scoring_authority_sha256": authority["authority_sha256"],
            "development_sha256": development_holder["development_sha256"],
            "result_sha256": result["result_sha256"],
            "result_file_sha256": adapter.file_sha256(paths.result),
            "measurement_receipt_sha256": authority["measurement_receipt_sha256"],
            "source_closure_sha256": source.sha256,
            "useful_pass": None,
            "promotive": False,
        }
        receipt["score_receipt_sha256"] = _logical_sha256(receipt)
        phase = "publish_receipt_last"
        adapter._publish_bytes_create_only(paths.receipt, _json_bytes(receipt), lock)
        return result
    except Exception as error:
        if authority_written and not paths.failure.exists():
            failure = {
                "schema_version": 1,
                "status": "failed_closed_no_scientific_interpretation",
                "authorized_for_final_role": False,
                "phase": phase,
                "exception_type": type(error).__name__,
                "authority_file_sha256": (
                    adapter.file_sha256(paths.authority)
                    if paths.authority.exists() else None
                ),
            }
            try:
                adapter._publish_bytes_create_only(
                    paths.failure, _json_bytes(failure), lock,
                )
            except Exception:
                pass
        raise
    finally:
        if lock.inode is not None:
            lock.release()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurement-receipt", type=Path, default=DEFAULT_MEASUREMENT_RECEIPT,
    )
    parser.add_argument("--output-directory", type=Path, default=HERE)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    arguments = parser.parse_args(argv)
    result = run_score_transaction(
        measurement_receipt_path=arguments.measurement_receipt,
        repo=REPO,
        paths=score_paths(arguments.output_directory, arguments.namespace),
    )
    print(json.dumps({
        "status": result["status"], "useful_pass": result["useful_pass"],
        "result_sha256": result["result_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
