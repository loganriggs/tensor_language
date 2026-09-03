"""Aggregate saved rung-522 metrics into provisional VALIDATION A/B gates.

This module performs no fitting, model execution, data loading, or CUDA work.
Inputs are ordinary nested mappings of already-saved scalar metrics, plus the
CPU projector frames needed by the matched three-fold stability helper.

Input layout
------------
``real``, ``recovery_only``, and ``oracles`` have shape
``{omitted_fold: {seed: {"healthy": bool, "cells": {cell: metrics}}}}``.
Each fold has seeds 52200..52204 and exactly the four cells in
``VALIDATION_CELLS``. A real cell additionally carries
``full_attention8_concentration`` and the two bootstrap lower bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Mapping, Sequence

import attention8_selective_shared_projector_rung522_protocol as protocol


REAL_SEEDS = protocol.REGISTERED_REAL_SEEDS
VALIDATION_CELLS = ("D0:forward", "D0:reverse", "D1:forward", "D1:reverse")


@dataclass(frozen=True)
class CellAGate:
    cell: str
    concentration: float
    oracle_recovery: float
    failures: tuple[str, ...]
    passes: bool


@dataclass(frozen=True)
class SeedAGate:
    seed: int
    fit_healthy: bool
    oracle_healthy: bool
    cells: tuple[CellAGate, ...]
    passes: bool


@dataclass(frozen=True)
class FoldAGate:
    fold: str
    seeds: tuple[SeedAGate, ...]
    passing_seed_count: int
    passes_four_of_five: bool


@dataclass(frozen=True)
class RecoverySeedGate:
    seed: int
    minimum_concentration_improvement: float
    maximum_signed_cosine_loss: float
    all_cell_bootstrap_improvements_positive: bool
    failures: tuple[str, ...]
    passes: bool


@dataclass(frozen=True)
class RecoveryFoldGate:
    fold: str
    seeds: tuple[RecoverySeedGate, ...]
    passing_seed_count: int
    sign_flip: protocol.SignFlipSummary
    passes: bool


@dataclass(frozen=True)
class JointSeedGate:
    seed: int
    real_joint_statistic: float
    haar_maximum: float
    label_null_q95_higher: float
    strictly_beats_both: bool


@dataclass(frozen=True)
class JointFoldGate:
    fold: str
    seeds: tuple[JointSeedGate, ...]
    passing_seed_count: int
    passes_four_of_five: bool


@dataclass(frozen=True)
class FoldBGate:
    fold: str
    recovery_comparison: RecoveryFoldGate
    joint_comparison: JointFoldGate
    passes: bool


@dataclass(frozen=True)
class ProvisionalValidationDecision:
    folds: tuple[str, str, str]
    prediction_a_folds: tuple[FoldAGate, ...]
    matched_stability: protocol.StabilitySummary
    real_fit_health_passes: bool
    recovery_only_fit_health_passes: bool
    oracle_liveness_passes: bool
    prediction_a_passes: bool
    prediction_b_folds: tuple[FoldBGate, ...]
    prediction_b_clauses_pass_without_a: bool
    prediction_b_passes: bool


def _number(mapping: Mapping[str, object], key: str, context: str) -> float:
    if key not in mapping:
        raise ValueError(f"{context} is missing scalar {key!r}")
    value = mapping[key]
    if isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be a finite number, not bool")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}.{key} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{context}.{key} must be finite")
    return result


def _fit(
    family: Mapping[str, Mapping[int, Mapping[str, object]]],
    fold: str,
    seed: int,
    name: str,
) -> tuple[bool, Mapping[str, Mapping[str, object]]]:
    raw = family[fold][seed]
    if not {"healthy", "cells"}.issubset(raw):
        raise ValueError(f"{name}[{fold}][{seed}] must contain healthy and cells")
    healthy = raw["healthy"]
    if type(healthy) is not bool:
        raise ValueError(f"{name}[{fold}][{seed}].healthy must be a literal bool")
    cells = raw["cells"]
    if not isinstance(cells, Mapping) or set(cells) != set(VALIDATION_CELLS):
        raise ValueError(
            f"{name}[{fold}][{seed}] must contain exactly the four VALIDATION cells"
        )
    if any(not isinstance(value, Mapping) for value in cells.values()):
        raise ValueError(f"{name}[{fold}][{seed}] cell metrics must be mappings")
    return healthy, cells  # type: ignore[return-value]


def _validate_family(
    family: Mapping[str, Mapping[int, Mapping[str, object]]],
    folds: tuple[str, str, str],
    name: str,
) -> None:
    if set(family) != set(folds):
        raise ValueError(f"{name} must contain exactly the same three omitted folds")
    for fold in folds:
        try:
            seeds = tuple(sorted(operator.index(seed) for seed in family[fold]))
        except TypeError as error:
            raise ValueError(f"{name}[{fold}] seed keys must be integers") from error
        if seeds != REAL_SEEDS:
            raise ValueError(f"{name}[{fold}] must contain seeds 52200..52204")
        for seed in REAL_SEEDS:
            _fit(family, fold, seed, name)


def _concentration(cell: Mapping[str, object], context: str) -> float:
    member = _number(cell, "member_rms", context)
    control = _number(cell, "control_rms", context)
    return protocol.selectivity_from_rms(member, control).concentration


def _score_a_fold(
    fold: str,
    real: Mapping[str, Mapping[int, Mapping[str, object]]],
    oracles: Mapping[str, Mapping[int, Mapping[str, object]]],
) -> FoldAGate:
    seed_results = []
    for seed in REAL_SEEDS:
        real_healthy, real_cells = _fit(real, fold, seed, "real")
        oracle_healthy, oracle_cells = _fit(oracles, fold, seed, "oracles")
        cell_results = []
        for cell_name in VALIDATION_CELLS:
            real_cell = real_cells[cell_name]
            oracle_cell = oracle_cells[cell_name]
            context = f"real[{fold}][{seed}][{cell_name}]"
            oracle_context = f"oracles[{fold}][{seed}][{cell_name}]"
            cosine = _number(real_cell, "signed_cosine", context)
            residual = _number(real_cell, "relative_residual", context)
            recovery = _number(real_cell, "aligned_recovery", context)
            member_rms = _number(real_cell, "member_rms", context)
            concentration = _concentration(real_cell, context)
            full_concentration = _number(
                real_cell, "full_attention8_concentration", context
            )
            margin_lower = _number(
                real_cell, "fourfold_margin_lower95", context
            )
            oracle_member = _number(oracle_cell, "member_rms", oracle_context)
            oracle_recovery = _number(
                oracle_cell, "aligned_recovery", oracle_context
            )
            failures = []
            if not oracle_healthy:
                failures.append("oracle_unhealthy")
            if oracle_member < 0.02:
                failures.append("oracle_member_rms_below_0.02")
            if oracle_recovery < 0.05:
                failures.append("oracle_recovery_below_0.05")
            if cosine < 0.75:
                failures.append("signed_cosine_below_0.75")
            if residual > 0.55:
                failures.append("relative_residual_above_0.55")
            if recovery <= 0:
                failures.append("aligned_recovery_not_positive")
            if recovery < 0.5 * oracle_recovery:
                failures.append("recovery_below_half_same_seed_oracle")
            if member_rms < 0.02:
                failures.append("member_rms_below_0.02")
            if concentration < 4.0:
                failures.append("concentration_below_4")
            if concentration - full_concentration < 1.0:
                failures.append("concentration_improvement_below_1")
            if margin_lower <= 0:
                failures.append("fourfold_margin_bootstrap_lower_not_positive")
            if "exact_token_tier0_or1" not in real_cell:
                raise ValueError(f"{context} is missing exact-token control result")
            exact_token = real_cell["exact_token_tier0_or1"]
            if exact_token is not None:
                if not isinstance(exact_token, Mapping) or type(
                    exact_token.get("passes")
                ) is not bool:
                    raise ValueError(
                        f"{context}.exact_token_tier0_or1 must be null or contain literal passes"
                    )
                if not exact_token["passes"]:
                    failures.append("powered_exact_token_specificity_failed")
            cell_results.append(
                CellAGate(
                    cell=cell_name,
                    concentration=concentration,
                    oracle_recovery=oracle_recovery,
                    failures=tuple(failures),
                    passes=not failures,
                )
            )
        seed_passes = real_healthy and oracle_healthy and all(
            cell.passes for cell in cell_results
        )
        seed_results.append(
            SeedAGate(
                seed=seed,
                fit_healthy=real_healthy,
                oracle_healthy=oracle_healthy,
                cells=tuple(cell_results),
                passes=seed_passes,
            )
        )
    count = sum(seed.passes for seed in seed_results)
    return FoldAGate(fold, tuple(seed_results), count, count >= 4)


def _score_recovery_fold(
    fold: str,
    real: Mapping[str, Mapping[int, Mapping[str, object]]],
    recovery_only: Mapping[str, Mapping[int, Mapping[str, object]]],
) -> RecoveryFoldGate:
    seed_results = []
    improvements = []
    for seed in REAL_SEEDS:
        real_healthy, real_cells = _fit(real, fold, seed, "real")
        recovery_healthy, recovery_cells = _fit(
            recovery_only, fold, seed, "recovery_only"
        )
        real_concentrations = []
        recovery_concentrations = []
        cosine_losses = []
        bootstrap_lowers = []
        for cell_name in VALIDATION_CELLS:
            real_cell = real_cells[cell_name]
            recovery_cell = recovery_cells[cell_name]
            real_context = f"real[{fold}][{seed}][{cell_name}]"
            control_context = f"recovery_only[{fold}][{seed}][{cell_name}]"
            real_concentrations.append(_concentration(real_cell, real_context))
            recovery_concentrations.append(
                _concentration(recovery_cell, control_context)
            )
            cosine_losses.append(
                _number(recovery_cell, "signed_cosine", control_context)
                - _number(real_cell, "signed_cosine", real_context)
            )
            bootstrap_lowers.append(
                _number(
                    real_cell,
                    "bounded_selectivity_improvement_lower95",
                    real_context,
                )
            )
        improvement = min(real_concentrations) - min(recovery_concentrations)
        maximum_cosine_loss = max(cosine_losses)
        all_bootstrap_positive = all(value > 0 for value in bootstrap_lowers)
        failures = []
        if not real_healthy:
            failures.append("task_conditioned_fit_unhealthy")
        if not recovery_healthy:
            failures.append("recovery_only_fit_unhealthy")
        if improvement < 0.5:
            failures.append("minimum_concentration_improvement_below_0.5")
        if maximum_cosine_loss > 0.05:
            failures.append("signed_cosine_loss_above_0.05")
        if not all_bootstrap_positive:
            failures.append("cell_bootstrap_selectivity_improvement_not_positive")
        improvements.append(improvement)
        seed_results.append(
            RecoverySeedGate(
                seed=seed,
                minimum_concentration_improvement=improvement,
                maximum_signed_cosine_loss=maximum_cosine_loss,
                all_cell_bootstrap_improvements_positive=all_bootstrap_positive,
                failures=tuple(failures),
                passes=not failures,
            )
        )
    sign_flip = protocol.exact_five_pair_sign_flip_null(improvements)
    count = sum(seed.passes for seed in seed_results)
    return RecoveryFoldGate(
        fold=fold,
        seeds=tuple(seed_results),
        passing_seed_count=count,
        sign_flip=sign_flip,
        passes=count >= 4 and sign_flip.strictly_exceeds_q95,
    )


def _control_values(
    controls: Mapping[str, Sequence[float]],
    fold: str,
    name: str,
    expected: int,
) -> tuple[float, ...]:
    if fold not in controls:
        raise ValueError(f"{name} is missing fold {fold!r}")
    values = tuple(float(value) for value in controls[fold])
    if len(values) != expected or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name}[{fold}] must contain exactly {expected} finite values")
    return values


def _score_joint_fold(
    fold: str,
    real: Mapping[str, Mapping[int, Mapping[str, object]]],
    haar_joint: Mapping[str, Sequence[float]],
    label_null_joint: Mapping[str, Sequence[float]],
) -> JointFoldGate:
    haar = _control_values(haar_joint, fold, "haar_joint", 20)
    null = _control_values(label_null_joint, fold, "label_null_joint", 16)
    haar_max = max(haar)
    null_q95 = protocol.higher_quantile(null, 0.95)
    seeds = []
    for seed in REAL_SEEDS:
        healthy, cells = _fit(real, fold, seed, "real")
        selectivities = []
        recoveries = []
        for cell_name in VALIDATION_CELLS:
            cell = cells[cell_name]
            context = f"real[{fold}][{seed}][{cell_name}]"
            member = _number(cell, "member_rms", context)
            control = _number(cell, "control_rms", context)
            selectivities.append(
                protocol.selectivity_from_rms(member, control).bounded_selectivity
            )
            recoveries.append(_number(cell, "aligned_recovery", context))
        joint = protocol.bounded_joint_statistic(selectivities, recoveries).product
        beats = healthy and joint > haar_max and joint > null_q95
        seeds.append(JointSeedGate(seed, joint, haar_max, null_q95, beats))
    count = sum(seed.strictly_beats_both for seed in seeds)
    return JointFoldGate(fold, tuple(seeds), count, count >= 4)


def evaluate_provisional_validation_gates(
    *,
    real: Mapping[str, Mapping[int, Mapping[str, object]]],
    recovery_only: Mapping[str, Mapping[int, Mapping[str, object]]],
    oracles: Mapping[str, Mapping[int, Mapping[str, object]]],
    haar_joint: Mapping[str, Sequence[float]],
    label_null_joint: Mapping[str, Sequence[float]],
    real_frames: Mapping[int, Mapping[str, object]],
    label_null_frames: Mapping[int, Mapping[str, object]],
) -> ProvisionalValidationDecision:
    """Apply every corrected provisional VALIDATION A/B aggregation clause."""
    folds_raw = tuple(sorted(real))
    if len(folds_raw) != 3:
        raise ValueError("real metrics must contain exactly three omitted-target folds")
    folds = (folds_raw[0], folds_raw[1], folds_raw[2])
    _validate_family(real, folds, "real")
    _validate_family(recovery_only, folds, "recovery_only")
    _validate_family(oracles, folds, "oracles")
    if set(haar_joint) != set(folds) or set(label_null_joint) != set(folds):
        raise ValueError("joint control mappings must contain the same three folds")

    stability = protocol.summarize_matched_three_fold_stability(
        real_frames, label_null_frames  # type: ignore[arg-type]
    )
    if set(stability.folds) != set(folds):
        raise ValueError("stability frame folds do not match metric folds")
    a_folds = tuple(_score_a_fold(fold, real, oracles) for fold in folds)
    real_health = all(
        _fit(real, fold, seed, "real")[0] for fold in folds for seed in REAL_SEEDS
    )
    recovery_health = all(
        _fit(recovery_only, fold, seed, "recovery_only")[0]
        for fold in folds
        for seed in REAL_SEEDS
    )
    oracle_live = all(
        seed.oracle_healthy
        and all(
            "oracle_member_rms_below_0.02" not in cell.failures
            and "oracle_recovery_below_0.05" not in cell.failures
            for cell in seed.cells
        )
        for fold in a_folds
        for seed in fold.seeds
    )
    prediction_a = (
        real_health
        and oracle_live
        and all(fold.passes_four_of_five for fold in a_folds)
        and stability.passes_four_of_five
    )

    b_folds = []
    for fold in folds:
        recovery = _score_recovery_fold(fold, real, recovery_only)
        joint = _score_joint_fold(fold, real, haar_joint, label_null_joint)
        b_folds.append(FoldBGate(fold, recovery, joint, recovery.passes and joint.passes_four_of_five))
    b_clauses = real_health and recovery_health and all(fold.passes for fold in b_folds)
    return ProvisionalValidationDecision(
        folds=folds,
        prediction_a_folds=a_folds,
        matched_stability=stability,
        real_fit_health_passes=real_health,
        recovery_only_fit_health_passes=recovery_health,
        oracle_liveness_passes=oracle_live,
        prediction_a_passes=prediction_a,
        prediction_b_folds=tuple(b_folds),
        prediction_b_clauses_pass_without_a=b_clauses,
        prediction_b_passes=prediction_a and b_clauses,
    )


__all__ = [
    "CellAGate",
    "FoldAGate",
    "FoldBGate",
    "JointFoldGate",
    "JointSeedGate",
    "ProvisionalValidationDecision",
    "REAL_SEEDS",
    "RecoveryFoldGate",
    "RecoverySeedGate",
    "SeedAGate",
    "VALIDATION_CELLS",
    "evaluate_provisional_validation_gates",
]
