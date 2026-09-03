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

``reserved_oracles`` has shape ``{seed: {"healthy": bool, "cells": ...}}``
for the five independently fitted ``r.2.0.1`` oracles.
``label_null_fit_health`` has shape ``{null_seed: {fold: bool}}`` for all
16 x 3 label-null fits. ``all_three`` has shape
``{seed: {"healthy": bool, "targets": {fold: {"cells": ...}}}}``.  The
all-three frame IDs and eligibility are derived here; callers cannot provide
or override either one.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Mapping, Sequence

import attention8_selective_shared_projector_rung522_protocol as protocol


REAL_SEEDS = protocol.REGISTERED_REAL_SEEDS
NULL_SEEDS = protocol.REGISTERED_NULL_SEEDS
VALIDATION_CELLS = ("D0:forward", "D0:reverse", "D1:forward", "D1:reverse")
COMBINED_CELLS = tuple(f"validation:{cell}" for cell in VALIDATION_CELLS) + tuple(
    f"test:{cell}" for cell in VALIDATION_CELLS
)
RESERVED_TARGET = "r.2.0.1"


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
class LabelNullFitHealth:
    frame_id: str
    seed: int
    fold: str
    healthy: bool


@dataclass(frozen=True)
class OracleCellLiveness:
    cell: str
    member_rms: float
    aligned_recovery: float
    failures: tuple[str, ...]
    passes: bool


@dataclass(frozen=True)
class OracleFitLiveness:
    frame_id: str
    target: str
    seed: int
    healthy: bool
    cells: tuple[OracleCellLiveness, ...]
    passes: bool


@dataclass(frozen=True)
class AllThreeTargetGate:
    target: str
    cells: tuple[CellAGate, ...]
    passes: bool


@dataclass(frozen=True)
class AllThreeFrameGate:
    frame_id: str
    seed: int
    healthy: bool
    targets: tuple[AllThreeTargetGate, ...]
    failures: tuple[str, ...]
    passes: bool


@dataclass(frozen=True)
class ProvisionalValidationDecision:
    folds: tuple[str, str, str]
    prediction_a_folds: tuple[FoldAGate, ...]
    matched_stability: protocol.StabilitySummary
    real_fit_health_passes: bool
    recovery_only_fit_health_passes: bool
    label_null_fit_health: tuple[LabelNullFitHealth, ...]
    label_null_fit_health_passes: bool
    oracle_fit_liveness: tuple[OracleFitLiveness, ...]
    fitted_target_oracle_liveness_passes: bool
    oracle_liveness_passes: bool
    prediction_a_passes: bool
    prediction_b_folds: tuple[FoldBGate, ...]
    prediction_b_clauses_pass_without_a: bool
    prediction_b_passes: bool
    all_three_frames: tuple[AllThreeFrameGate, ...]
    eligible_all_three_frame_ids: tuple[str, ...]
    all_three_eligibility_nonempty: bool
    pretest_passes: bool


@dataclass(frozen=True)
class FinalValidationTestDecision:
    """The same frozen A/B rules recomputed over VALIDATION and TEST jointly."""

    cells: tuple[str, ...]
    folds: tuple[str, str, str]
    prediction_a_folds: tuple[FoldAGate, ...]
    matched_stability: protocol.StabilitySummary
    real_fit_health_passes: bool
    recovery_only_fit_health_passes: bool
    oracle_fit_liveness: tuple[OracleFitLiveness, ...]
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
    required_cells: tuple[str, ...] = VALIDATION_CELLS,
) -> tuple[bool, Mapping[str, Mapping[str, object]]]:
    raw = family[fold][seed]
    return _fit_record(raw, f"{name}[{fold}][{seed}]", required_cells)


def _fit_record(
    raw: Mapping[str, object],
    context: str,
    required_cells: tuple[str, ...] = VALIDATION_CELLS,
) -> tuple[bool, Mapping[str, Mapping[str, object]]]:
    if not {"healthy", "cells"}.issubset(raw):
        raise ValueError(f"{context} must contain healthy and cells")
    healthy = raw["healthy"]
    if type(healthy) is not bool:
        raise ValueError(f"{context}.healthy must be a literal bool")
    cells = raw["cells"]
    if not isinstance(cells, Mapping) or set(cells) != set(required_cells):
        raise ValueError(
            f"{context} must contain exactly the {len(required_cells)} required cells"
        )
    if any(not isinstance(value, Mapping) for value in cells.values()):
        raise ValueError(f"{context} cell metrics must be mappings")
    return healthy, cells  # type: ignore[return-value]


def _validate_family(
    family: Mapping[str, Mapping[int, Mapping[str, object]]],
    folds: tuple[str, str, str],
    name: str,
    required_cells: tuple[str, ...] = VALIDATION_CELLS,
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
            _fit(family, fold, seed, name, required_cells)


def _concentration(cell: Mapping[str, object], context: str) -> float:
    member = _number(cell, "member_rms", context)
    control = _number(cell, "control_rms", context)
    return protocol.selectivity_from_rms(member, control).concentration


def _score_a_cell(
    cell_name: str,
    real_cell: Mapping[str, object],
    oracle_cell: Mapping[str, object],
    *,
    oracle_healthy: bool,
    context: str,
    oracle_context: str,
) -> CellAGate:
    """Apply the literal per-cell A predicate, including the paired oracle."""
    cosine = _number(real_cell, "signed_cosine", context)
    residual = _number(real_cell, "relative_residual", context)
    recovery = _number(real_cell, "aligned_recovery", context)
    member_rms = _number(real_cell, "member_rms", context)
    concentration = _concentration(real_cell, context)
    full_concentration = _number(
        real_cell, "full_attention8_concentration", context
    )
    margin_lower = _number(real_cell, "fourfold_margin_lower95", context)
    oracle_member = _number(oracle_cell, "member_rms", oracle_context)
    oracle_recovery = _number(oracle_cell, "aligned_recovery", oracle_context)
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
    return CellAGate(
        cell=cell_name,
        concentration=concentration,
        oracle_recovery=oracle_recovery,
        failures=tuple(failures),
        passes=not failures,
    )


def _score_a_fold(
    fold: str,
    real: Mapping[str, Mapping[int, Mapping[str, object]]],
    oracles: Mapping[str, Mapping[int, Mapping[str, object]]],
    required_cells: tuple[str, ...] = VALIDATION_CELLS,
) -> FoldAGate:
    seed_results = []
    for seed in REAL_SEEDS:
        real_healthy, real_cells = _fit(real, fold, seed, "real", required_cells)
        oracle_healthy, oracle_cells = _fit(
            oracles, fold, seed, "oracles", required_cells
        )
        cell_results = []
        for cell_name in required_cells:
            real_cell = real_cells[cell_name]
            oracle_cell = oracle_cells[cell_name]
            context = f"real[{fold}][{seed}][{cell_name}]"
            oracle_context = f"oracles[{fold}][{seed}][{cell_name}]"
            cell_results.append(
                _score_a_cell(
                    cell_name,
                    real_cell,
                    oracle_cell,
                    oracle_healthy=oracle_healthy,
                    context=context,
                    oracle_context=oracle_context,
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
    required_cells: tuple[str, ...] = VALIDATION_CELLS,
) -> RecoveryFoldGate:
    seed_results = []
    improvements = []
    for seed in REAL_SEEDS:
        real_healthy, real_cells = _fit(real, fold, seed, "real", required_cells)
        recovery_healthy, recovery_cells = _fit(
            recovery_only, fold, seed, "recovery_only", required_cells
        )
        real_concentrations = []
        recovery_concentrations = []
        cosine_losses = []
        bootstrap_lowers = []
        for cell_name in required_cells:
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
    required_cells: tuple[str, ...] = VALIDATION_CELLS,
) -> JointFoldGate:
    haar = _control_values(haar_joint, fold, "haar_joint", 20)
    null = _control_values(label_null_joint, fold, "label_null_joint", 16)
    haar_max = max(haar)
    null_q95 = protocol.higher_quantile(null, 0.95)
    seeds = []
    for seed in REAL_SEEDS:
        healthy, cells = _fit(real, fold, seed, "real", required_cells)
        selectivities = []
        recoveries = []
        for cell_name in required_cells:
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


def _normalize_seed_mapping(
    values: Mapping[int, object], expected: tuple[int, ...], name: str
) -> dict[int, object]:
    normalized: dict[int, object] = {}
    for raw_seed, value in values.items():
        if isinstance(raw_seed, bool):
            raise ValueError(f"{name} seed keys must be integers, not bool")
        try:
            seed = operator.index(raw_seed)
        except TypeError as error:
            raise ValueError(f"{name} seed keys must be integers") from error
        if seed in normalized:
            raise ValueError(f"{name} contains duplicate normalized seed {seed}")
        normalized[seed] = value
    if tuple(sorted(normalized)) != expected:
        raise ValueError(f"{name} must contain exactly seeds {expected}")
    return normalized


def _score_oracle_fit(
    target: str,
    seed: int,
    raw: Mapping[str, object],
    context: str,
    required_cells: tuple[str, ...] = VALIDATION_CELLS,
) -> OracleFitLiveness:
    healthy, cells = _fit_record(raw, context, required_cells)
    cell_results = []
    for cell_name in required_cells:
        cell = cells[cell_name]
        cell_context = f"{context}[{cell_name}]"
        member_rms = _number(cell, "member_rms", cell_context)
        recovery = _number(cell, "aligned_recovery", cell_context)
        failures = []
        if member_rms < 0.02:
            failures.append("member_rms_below_0.02")
        if recovery < 0.05:
            failures.append("aligned_recovery_below_0.05")
        cell_results.append(
            OracleCellLiveness(
                cell=cell_name,
                member_rms=member_rms,
                aligned_recovery=recovery,
                failures=tuple(failures),
                passes=not failures,
            )
        )
    return OracleFitLiveness(
        frame_id=f"target_oracle:{target}:{seed}",
        target=target,
        seed=seed,
        healthy=healthy,
        cells=tuple(cell_results),
        passes=healthy and all(cell.passes for cell in cell_results),
    )


def _score_all_oracles(
    folds: tuple[str, str, str],
    oracles: Mapping[str, Mapping[int, Mapping[str, object]]],
    reserved_oracles: Mapping[int, Mapping[str, object]],
    required_cells: tuple[str, ...] = VALIDATION_CELLS,
) -> tuple[OracleFitLiveness, ...]:
    normalized_reserved = _normalize_seed_mapping(
        reserved_oracles, REAL_SEEDS, "reserved_oracles"
    )
    results = []
    for target in folds:
        for seed in REAL_SEEDS:
            results.append(
                _score_oracle_fit(
                    target,
                    seed,
                    oracles[target][seed],
                    f"oracles[{target}][{seed}]",
                    required_cells,
                )
            )
    for seed in REAL_SEEDS:
        raw = normalized_reserved[seed]
        if not isinstance(raw, Mapping):
            raise ValueError(f"reserved_oracles[{seed}] must be a mapping")
        results.append(
            _score_oracle_fit(
                RESERVED_TARGET,
                seed,
                raw,
                f"reserved_oracles[{seed}]",
                required_cells,
            )
        )
    if len(results) != 20 or len({result.frame_id for result in results}) != 20:
        raise AssertionError("internal oracle census is not 20 unique fits")
    return tuple(results)


def _score_label_null_health(
    values: Mapping[int, Mapping[str, bool]], folds: tuple[str, str, str]
) -> tuple[LabelNullFitHealth, ...]:
    normalized = _normalize_seed_mapping(
        values, NULL_SEEDS, "label_null_fit_health"
    )
    results = []
    for seed in NULL_SEEDS:
        by_fold = normalized[seed]
        if not isinstance(by_fold, Mapping) or set(by_fold) != set(folds):
            raise ValueError(
                f"label_null_fit_health[{seed}] must contain exactly the three folds"
            )
        for fold in folds:
            healthy = by_fold[fold]
            if type(healthy) is not bool:
                raise ValueError(
                    f"label_null_fit_health[{seed}][{fold}] must be a literal bool"
                )
            results.append(
                LabelNullFitHealth(
                    frame_id=f"label_null:{seed}:{fold}",
                    seed=seed,
                    fold=fold,
                    healthy=healthy,
                )
            )
    if len(results) != 48 or len({result.frame_id for result in results}) != 48:
        raise AssertionError("internal label-null census is not 48 unique fits")
    return tuple(results)


def _all_three_target_cells(
    raw: Mapping[str, object], context: str
) -> Mapping[str, Mapping[str, object]]:
    forbidden = {"eligible", "eligibility", "passes"}.intersection(raw)
    if forbidden:
        raise ValueError(
            f"{context} must not contain caller-provided eligibility fields {sorted(forbidden)}"
        )
    cells = raw.get("cells")
    if not isinstance(cells, Mapping) or set(cells) != set(VALIDATION_CELLS):
        raise ValueError(f"{context} must contain exactly the four VALIDATION cells")
    if any(not isinstance(value, Mapping) for value in cells.values()):
        raise ValueError(f"{context} cell metrics must be mappings")
    return cells  # type: ignore[return-value]


def _score_all_three_frames(
    values: Mapping[int, Mapping[str, object]],
    folds: tuple[str, str, str],
    oracles: Mapping[str, Mapping[int, Mapping[str, object]]],
) -> tuple[AllThreeFrameGate, ...]:
    normalized = _normalize_seed_mapping(values, REAL_SEEDS, "all_three")
    frames = []
    for seed in REAL_SEEDS:
        raw = normalized[seed]
        if not isinstance(raw, Mapping):
            raise ValueError(f"all_three[{seed}] must be a mapping")
        forbidden = {"eligible", "eligibility", "passes"}.intersection(raw)
        if forbidden:
            raise ValueError(
                "all_three eligibility is derived; caller-provided fields are forbidden: "
                f"{sorted(forbidden)}"
            )
        healthy = raw.get("healthy")
        if type(healthy) is not bool:
            raise ValueError(f"all_three[{seed}].healthy must be a literal bool")
        targets = raw.get("targets")
        if not isinstance(targets, Mapping):
            raise ValueError(f"all_three[{seed}].targets must be a mapping")
        if RESERVED_TARGET in targets:
            raise ValueError("the reserved r.2.0.1 target cannot enter all-three eligibility")
        if set(targets) != set(folds):
            raise ValueError(
                f"all_three[{seed}].targets must contain exactly the three fitted targets"
            )
        target_results = []
        for target in folds:
            target_raw = targets[target]
            if not isinstance(target_raw, Mapping):
                raise ValueError(f"all_three[{seed}].targets[{target}] must be a mapping")
            cells = _all_three_target_cells(
                target_raw, f"all_three[{seed}].targets[{target}]"
            )
            oracle_healthy, oracle_cells = _fit(
                oracles, target, seed, "oracles"
            )
            cell_results = tuple(
                _score_a_cell(
                    cell_name,
                    cells[cell_name],
                    oracle_cells[cell_name],
                    oracle_healthy=oracle_healthy,
                    context=f"all_three[{seed}][{target}][{cell_name}]",
                    oracle_context=f"oracles[{target}][{seed}][{cell_name}]",
                )
                for cell_name in VALIDATION_CELLS
            )
            target_results.append(
                AllThreeTargetGate(
                    target=target,
                    cells=cell_results,
                    passes=all(cell.passes for cell in cell_results),
                )
            )
        failures = []
        if not healthy:
            failures.append("optimizer_unhealthy")
        failures.extend(
            f"target_validation_gate_failed:{target.target}"
            for target in target_results
            if not target.passes
        )
        frames.append(
            AllThreeFrameGate(
                frame_id=f"all_three:{seed}",
                seed=seed,
                healthy=healthy,
                targets=tuple(target_results),
                failures=tuple(failures),
                passes=not failures,
            )
        )
    return tuple(frames)


@dataclass(frozen=True)
class _ABGateResult:
    folds: tuple[str, str, str]
    prediction_a_folds: tuple[FoldAGate, ...]
    matched_stability: protocol.StabilitySummary
    real_fit_health_passes: bool
    recovery_only_fit_health_passes: bool
    oracle_fit_liveness: tuple[OracleFitLiveness, ...]
    fitted_target_oracle_liveness_passes: bool
    oracle_liveness_passes: bool
    prediction_a_passes: bool
    prediction_b_folds: tuple[FoldBGate, ...]
    prediction_b_clauses_pass_without_a: bool
    prediction_b_passes: bool


def _evaluate_ab_gates(
    *,
    real: Mapping[str, Mapping[int, Mapping[str, object]]],
    recovery_only: Mapping[str, Mapping[int, Mapping[str, object]]],
    oracles: Mapping[str, Mapping[int, Mapping[str, object]]],
    reserved_oracles: Mapping[int, Mapping[str, object]],
    haar_joint: Mapping[str, Sequence[float]],
    label_null_joint: Mapping[str, Sequence[float]],
    real_frames: Mapping[int, Mapping[str, object]],
    label_null_frames: Mapping[int, Mapping[str, object]],
    required_cells: tuple[str, ...],
) -> _ABGateResult:
    if not required_cells or len(set(required_cells)) != len(required_cells):
        raise ValueError("required cells must be nonempty and unique")
    folds_raw = tuple(sorted(real))
    if len(folds_raw) != 3:
        raise ValueError("real metrics must contain exactly three omitted-target folds")
    folds = (folds_raw[0], folds_raw[1], folds_raw[2])
    if RESERVED_TARGET in folds:
        raise ValueError("the reserved r.2.0.1 target is not an omitted fitted-target fold")
    _validate_family(real, folds, "real", required_cells)
    _validate_family(recovery_only, folds, "recovery_only", required_cells)
    _validate_family(oracles, folds, "oracles", required_cells)
    if set(haar_joint) != set(folds) or set(label_null_joint) != set(folds):
        raise ValueError("joint control mappings must contain the same three folds")

    stability = protocol.summarize_matched_three_fold_stability(
        real_frames, label_null_frames  # type: ignore[arg-type]
    )
    if set(stability.folds) != set(folds):
        raise ValueError("stability frame folds do not match metric folds")
    a_folds = tuple(
        _score_a_fold(fold, real, oracles, required_cells) for fold in folds
    )
    real_health = all(
        _fit(real, fold, seed, "real", required_cells)[0]
        for fold in folds
        for seed in REAL_SEEDS
    )
    recovery_health = all(
        _fit(recovery_only, fold, seed, "recovery_only", required_cells)[0]
        for fold in folds
        for seed in REAL_SEEDS
    )
    oracle_evidence = _score_all_oracles(
        folds, oracles, reserved_oracles, required_cells
    )
    fitted_oracle_live = all(
        evidence.passes
        for evidence in oracle_evidence
        if evidence.target != RESERVED_TARGET
    )
    all_oracle_live = all(evidence.passes for evidence in oracle_evidence)
    prediction_a = (
        real_health
        and all_oracle_live
        and all(fold.passes_four_of_five for fold in a_folds)
        and stability.passes_four_of_five
    )

    b_folds = []
    for fold in folds:
        recovery = _score_recovery_fold(
            fold, real, recovery_only, required_cells
        )
        joint = _score_joint_fold(
            fold, real, haar_joint, label_null_joint, required_cells
        )
        b_folds.append(
            FoldBGate(
                fold,
                recovery,
                joint,
                recovery.passes and joint.passes_four_of_five,
            )
        )
    b_clauses = real_health and recovery_health and all(
        fold.passes for fold in b_folds
    )
    return _ABGateResult(
        folds=folds,
        prediction_a_folds=a_folds,
        matched_stability=stability,
        real_fit_health_passes=real_health,
        recovery_only_fit_health_passes=recovery_health,
        oracle_fit_liveness=oracle_evidence,
        fitted_target_oracle_liveness_passes=fitted_oracle_live,
        oracle_liveness_passes=all_oracle_live,
        prediction_a_passes=prediction_a,
        prediction_b_folds=tuple(b_folds),
        prediction_b_clauses_pass_without_a=b_clauses,
        prediction_b_passes=prediction_a and b_clauses,
    )


def evaluate_provisional_validation_gates(
    *,
    real: Mapping[str, Mapping[int, Mapping[str, object]]],
    recovery_only: Mapping[str, Mapping[int, Mapping[str, object]]],
    oracles: Mapping[str, Mapping[int, Mapping[str, object]]],
    haar_joint: Mapping[str, Sequence[float]],
    label_null_joint: Mapping[str, Sequence[float]],
    real_frames: Mapping[int, Mapping[str, object]],
    label_null_frames: Mapping[int, Mapping[str, object]],
    label_null_fit_health: Mapping[int, Mapping[str, bool]],
    reserved_oracles: Mapping[int, Mapping[str, object]],
    all_three: Mapping[int, Mapping[str, object]],
) -> ProvisionalValidationDecision:
    """Apply every corrected provisional VALIDATION A/B aggregation clause."""
    ab = _evaluate_ab_gates(
        real=real,
        recovery_only=recovery_only,
        oracles=oracles,
        reserved_oracles=reserved_oracles,
        haar_joint=haar_joint,
        label_null_joint=label_null_joint,
        real_frames=real_frames,
        label_null_frames=label_null_frames,
        required_cells=VALIDATION_CELLS,
    )
    null_health = _score_label_null_health(label_null_fit_health, ab.folds)
    all_three_frames = _score_all_three_frames(all_three, ab.folds, oracles)
    eligible = tuple(frame.frame_id for frame in all_three_frames if frame.passes)
    null_health_passes = all(fit.healthy for fit in null_health)
    eligibility_nonempty = bool(eligible)
    pretest_passes = (
        ab.prediction_a_passes
        and ab.prediction_b_passes
        and null_health_passes
        and ab.oracle_liveness_passes
        and eligibility_nonempty
    )
    return ProvisionalValidationDecision(
        folds=ab.folds,
        prediction_a_folds=ab.prediction_a_folds,
        matched_stability=ab.matched_stability,
        real_fit_health_passes=ab.real_fit_health_passes,
        recovery_only_fit_health_passes=ab.recovery_only_fit_health_passes,
        label_null_fit_health=null_health,
        label_null_fit_health_passes=null_health_passes,
        oracle_fit_liveness=ab.oracle_fit_liveness,
        fitted_target_oracle_liveness_passes=ab.fitted_target_oracle_liveness_passes,
        oracle_liveness_passes=ab.oracle_liveness_passes,
        prediction_a_passes=ab.prediction_a_passes,
        prediction_b_folds=ab.prediction_b_folds,
        prediction_b_clauses_pass_without_a=ab.prediction_b_clauses_pass_without_a,
        prediction_b_passes=ab.prediction_b_passes,
        all_three_frames=all_three_frames,
        eligible_all_three_frame_ids=eligible,
        all_three_eligibility_nonempty=eligibility_nonempty,
        pretest_passes=pretest_passes,
    )


def evaluate_final_validation_test_gates(
    *,
    real: Mapping[str, Mapping[int, Mapping[str, object]]],
    recovery_only: Mapping[str, Mapping[int, Mapping[str, object]]],
    oracles: Mapping[str, Mapping[int, Mapping[str, object]]],
    reserved_oracles: Mapping[int, Mapping[str, object]],
    haar_joint: Mapping[str, Sequence[float]],
    label_null_joint: Mapping[str, Sequence[float]],
    real_frames: Mapping[int, Mapping[str, object]],
    label_null_frames: Mapping[int, Mapping[str, object]],
) -> FinalValidationTestDecision:
    """Recompute the frozen A/B rules over the exact combined eight cells.

    Cell keys must be the four ``validation:...`` names followed by their four
    ``test:...`` counterparts in :data:`COMBINED_CELLS`.  Minima, maxima, and
    every-cell predicates are consequently taken across all eight cells.
    """
    ab = _evaluate_ab_gates(
        real=real,
        recovery_only=recovery_only,
        oracles=oracles,
        reserved_oracles=reserved_oracles,
        haar_joint=haar_joint,
        label_null_joint=label_null_joint,
        real_frames=real_frames,
        label_null_frames=label_null_frames,
        required_cells=COMBINED_CELLS,
    )
    return FinalValidationTestDecision(
        cells=COMBINED_CELLS,
        folds=ab.folds,
        prediction_a_folds=ab.prediction_a_folds,
        matched_stability=ab.matched_stability,
        real_fit_health_passes=ab.real_fit_health_passes,
        recovery_only_fit_health_passes=ab.recovery_only_fit_health_passes,
        oracle_fit_liveness=ab.oracle_fit_liveness,
        oracle_liveness_passes=ab.oracle_liveness_passes,
        prediction_a_passes=ab.prediction_a_passes,
        prediction_b_folds=ab.prediction_b_folds,
        prediction_b_clauses_pass_without_a=ab.prediction_b_clauses_pass_without_a,
        prediction_b_passes=ab.prediction_b_passes,
    )


__all__ = [
    "AllThreeFrameGate",
    "AllThreeTargetGate",
    "CellAGate",
    "COMBINED_CELLS",
    "FinalValidationTestDecision",
    "FoldAGate",
    "FoldBGate",
    "JointFoldGate",
    "JointSeedGate",
    "LabelNullFitHealth",
    "NULL_SEEDS",
    "OracleCellLiveness",
    "OracleFitLiveness",
    "ProvisionalValidationDecision",
    "REAL_SEEDS",
    "RESERVED_TARGET",
    "RecoveryFoldGate",
    "RecoverySeedGate",
    "SeedAGate",
    "VALIDATION_CELLS",
    "evaluate_final_validation_test_gates",
    "evaluate_provisional_validation_gates",
]
