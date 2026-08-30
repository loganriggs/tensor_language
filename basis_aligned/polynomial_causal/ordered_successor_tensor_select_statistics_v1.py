"""Pure-CPU scorer and launch-readiness ruling for successor SELECT v1.

This module deliberately owns no filesystem, row, model, checkpoint, authority, or
publication capability.  It can score an already-produced document sufficient-stat
ledger with the frozen 20,000-draw simultaneous bootstrap.  The separate readiness
ruling makes the currently frozen v1 assay prospectively NO-GO before any row or model
access rather than silently choosing missing protocol objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Mapping

import torch

import ordered_successor_tensor_discovery_v1 as discovery
import ordered_successor_tensor_select_registry_v2 as v2_registry
from successor_attention_backend import StoredSuccessorFactors


SOURCE_PATHS = (
    *discovery.SOURCE_CLOSURE,
    "basis_aligned/polynomial_causal/test_ordered_successor_masks_v1.py",
    "basis_aligned/polynomial_causal/test_ordered_successor_tensor_discovery_v1.py",
    "basis_aligned/polynomial_causal/test_ordered_successor_tensor_backend_adapter_v1.py",
    "basis_aligned/polynomial_causal/test_successor_attention_backend.py",
    "basis_aligned/polynomial_causal/test_bilin18_observed_model_facade.py",
    "basis_aligned/polynomial_causal/test_circuit_campaign_runtime.py",
    "basis_aligned/polynomial_causal/test_circuit_successor_tensor.py",
    "basis_aligned/polynomial_causal/test_tensor_preserving_attention.py",
    "basis_aligned/polynomial_causal/ordered_successor_tensor_select_statistics_v1.py",
    "basis_aligned/polynomial_causal/test_ordered_successor_tensor_select_statistics_v1.py",
    "basis_aligned/polynomial_causal/ordered_successor_tensor_select_registry_v2.py",
)
CELL_NAMES = (
    "all_positions",
    "positive_clean",
    "successor_copy_overlap",
    "copy_only",
    "wrong_source_clean",
    "no_source_clean",
    "excluded_local_or_ambiguous",
)
POWERED_CELLS = ("positive_clean", "wrong_source_clean", "no_source_clean")


@dataclass(frozen=True)
class SelectDocumentLedger:
    """No-logit document sufficient statistics in frozen arm/cell order."""

    document_ids: tuple[str, ...]
    pair_names: tuple[str, ...]
    count: torch.Tensor  # [document, cell], int64
    pair_count: torch.Tensor  # [document, ordered pair], int64
    ce_sum: torch.Tensor  # [document, arm, cell], float64
    native_kl_sum: torch.Tensor
    top1_change_sum: torch.Tensor
    successor_margin_sum: torch.Tensor
    arm_names: tuple[str, ...] = discovery.ARM_NAMES

    def __post_init__(self) -> None:
        documents = len(self.document_ids)
        if (
            type(self.document_ids) is not tuple or documents <= 0
            or len(set(self.document_ids)) != documents
            or any(not isinstance(value, str) or not value for value in self.document_ids)
            or type(self.pair_names) is not tuple or not self.pair_names
            or len(set(self.pair_names)) != len(self.pair_names)
            or any(not isinstance(value, str) or not value for value in self.pair_names)
        ):
            raise ValueError("successor document/pair identity is malformed")
        if (
            not torch.is_tensor(self.count) or self.count.device.type != "cpu"
            or self.count.dtype != torch.int64
            or tuple(self.count.shape) != (documents, len(CELL_NAMES))
            or not self.count.is_contiguous() or bool((self.count < 0).any())
            or not torch.is_tensor(self.pair_count) or self.pair_count.device.type != "cpu"
            or self.pair_count.dtype != torch.int64
            or tuple(self.pair_count.shape) != (documents, len(self.pair_names))
            or not self.pair_count.is_contiguous() or bool((self.pair_count < 0).any())
        ):
            raise ValueError("successor support ledger is malformed")
        if type(self.arm_names) is not tuple or self.arm_names not in (
            discovery.ARM_NAMES, v2_registry.ARM_NAMES,
        ):
            raise ValueError("successor arm registry is not an exact supported version")
        expected = (documents, len(self.arm_names), len(CELL_NAMES))
        values = (
            self.ce_sum, self.native_kl_sum, self.top1_change_sum,
            self.successor_margin_sum,
        )
        if any(
            not torch.is_tensor(value) or value.device.type != "cpu"
            or value.dtype != torch.float64 or tuple(value.shape) != expected
            or not value.is_contiguous() or not bool(torch.isfinite(value).all())
            for value in values
        ):
            raise ValueError("successor metric ledger is malformed")
        expanded = self.count[:, None, :]
        if bool((self.ce_sum < 0).any()) or bool((self.native_kl_sum < -1e-12).any()) or bool(
            (self.top1_change_sum.abs() > expanded).any()
        ):
            raise ValueError("successor KL/top1 sufficient statistics are impossible")
        zero = expanded == 0
        if any(bool((value.masked_select(zero.expand_as(value)) != 0).any()) for value in values):
            raise ValueError("zero-support metric sums must be exactly zero")
        native = self.arm_names.index("native")
        if not torch.equal(
            self.native_kl_sum[:, native], torch.zeros_like(self.native_kl_sum[:, native]),
        ) or not torch.equal(
            self.top1_change_sum[:, native],
            torch.zeros_like(self.top1_change_sum[:, native]),
        ):
            raise ValueError("native self-KL/top1-change sums must be exactly zero")
        if not torch.equal(self.pair_count.sum(dim=1), self.count[:, 1:].sum(dim=1)):
            raise ValueError("pair occupancy does not equal eligible-cell occupancy")


@dataclass(frozen=True)
class SimultaneousSelectScore:
    coordinate_names: tuple[str, ...]
    point: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    critical_value: float
    bootstrap_draws: int
    bootstrap_seed: int
    order_index: int
    support: Mapping[str, Mapping[str, int]]
    pair_support: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class ArmCellPoint:
    arm: str
    cell: str
    count: int
    ce: float
    native_kl: float
    top1_change: float
    successor_margin: float


@dataclass(frozen=True)
class IntegrityEvidence:
    native_full_replay_max_abs: float
    rank128_full_replay_max_abs: float
    native_full_replay_kl: float
    rank128_full_replay_kl: float
    call_ledgers_passed: bool
    row_support_program_hashes_passed: bool
    finite_currencies_passed: bool

    def __post_init__(self) -> None:
        numbers = (
            self.native_full_replay_max_abs, self.rank128_full_replay_max_abs,
            self.native_full_replay_kl, self.rank128_full_replay_kl,
        )
        flags = (
            self.call_ledgers_passed, self.row_support_program_hashes_passed,
            self.finite_currencies_passed,
        )
        if any(type(value) is not float or not math.isfinite(value) or value < 0 for value in numbers):
            raise ValueError("successor integrity numerics are malformed")
        if any(type(value) is not bool for value in flags):
            raise ValueError("successor integrity flags must be literal booleans")

    @property
    def passed(self) -> bool:
        return (
            self.native_full_replay_max_abs <= 1e-4
            and self.rank128_full_replay_max_abs <= 1e-4
            and self.native_full_replay_kl <= 1e-8
            and self.rank128_full_replay_kl <= 1e-8
            and self.call_ledgers_passed
            and self.row_support_program_hashes_passed
            and self.finite_currencies_passed
        )


@dataclass(frozen=True)
class PromotionDecision:
    passing_arms: tuple[str, ...]
    selected_arm: str | None
    selected_rank: int | None
    integrity_passed: bool
    support_passed: bool


@dataclass(frozen=True)
class ReadinessBlocker:
    code: str
    evidence: str
    cheapest_prospective_repair: str


@dataclass(frozen=True)
class SelectV1Readiness:
    status: str
    authority_allowed: bool
    row_freeze_allowed: bool
    model_forward_allowed: bool
    blockers: tuple[ReadinessBlocker, ...]
    frozen_arm_count: int
    promotive_arm_count: int


def _arm_index(name: str, arm_count: int) -> int:
    arm_names = (
        v2_registry.ARM_NAMES if arm_count == len(v2_registry.ARM_NAMES)
        else discovery.ARM_NAMES
    )
    try:
        return arm_names.index(name)
    except ValueError as error:
        raise ValueError(f"unknown frozen successor arm: {name}") from error


def _cell_index(name: str) -> int:
    try:
        return CELL_NAMES.index(name)
    except ValueError as error:
        raise ValueError(f"unknown frozen successor cell: {name}") from error


def _pooled(
    values: torch.Tensor, counts: torch.Tensor, arm: str, cell: str,
    weights: torch.Tensor,
) -> torch.Tensor:
    numerator = weights @ values[:, _arm_index(arm, values.shape[1]), _cell_index(cell)]
    denominator = weights @ counts[:, _cell_index(cell)].to(torch.float64)
    if bool((denominator <= 0).any()):
        raise ZeroDivisionError(f"zero token denominator for {cell}")
    return numerator / denominator


def _coordinate_family(
    ledger: SelectDocumentLedger, weights: torch.Tensor,
) -> tuple[tuple[str, ...], torch.Tensor]:
    ce = ledger.ce_sum
    margin = ledger.successor_margin_sum
    names: list[str] = []
    values: list[torch.Tensor] = []

    def add(name: str, value: torch.Tensor) -> None:
        if value.ndim != 1 or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"nonfinite successor coordinate: {name}")
        names.append(name)
        values.append(value)

    ce_native = {
        cell: _pooled(ce, ledger.count, "native", cell, weights) for cell in CELL_NAMES
    }
    ce_deleted = {
        cell: _pooled(ce, ledger.count, discovery.HEAD_DELETED, cell, weights)
        for cell in CELL_NAMES
    }
    margin_native = _pooled(
        margin, ledger.count, "native", "positive_clean", weights,
    )
    margin_deleted = _pooled(
        margin, ledger.count, discovery.HEAD_DELETED, "positive_clean", weights,
    )
    ce_stake = ce_deleted["positive_clean"] - ce_native["positive_clean"]
    margin_stake = margin_native - margin_deleted
    if bool((ce_stake <= 0).any()) or bool((margin_stake <= 0).any()):
        raise ZeroDivisionError("nonpositive deletion denominator in successor recovery")

    add("deletion_damage_positive_clean", ce_stake)
    add(
        "deletion_specificity_wrong_source_clean",
        ce_stake - (ce_deleted["wrong_source_clean"] - ce_native["wrong_source_clean"]),
    )
    add(
        "deletion_specificity_no_source_clean",
        ce_stake - (ce_deleted["no_source_clean"] - ce_native["no_source_clean"]),
    )
    for true_arm in discovery.PROMOTIVE_ARMS:
        null_arm = discovery.NULL_BY_TRUE[true_arm]
        true_ce = _pooled(ce, ledger.count, true_arm, "positive_clean", weights)
        null_ce = _pooled(ce, ledger.count, null_arm, "positive_clean", weights)
        true_margin = _pooled(
            margin, ledger.count, true_arm, "positive_clean", weights,
        )
        add(f"{true_arm}:ce_recovery", (ce_deleted["positive_clean"] - true_ce) / ce_stake)
        add(f"{true_arm}:margin_recovery", (true_margin - margin_deleted) / margin_stake)
        add(f"{true_arm}:true_minus_null_ce_benefit", null_ce - true_ce)
        add(f"{true_arm}:null_ce_recovery", (ce_deleted["positive_clean"] - null_ce) / ce_stake)
        add(
            f"{true_arm}:all_position_ce_drift",
            _pooled(ce, ledger.count, true_arm, "all_positions", weights)
            - ce_native["all_positions"],
        )
        for cell in ("wrong_source_clean", "no_source_clean"):
            add(
                f"{true_arm}:abs_{cell}_ce_drift",
                (
                    _pooled(ce, ledger.count, true_arm, cell, weights)
                    - ce_native[cell]
                ).abs(),
            )
    return tuple(names), torch.stack(values, dim=1)


def score_select_ledger(ledger: SelectDocumentLedger) -> SimultaneousSelectScore:
    """Apply the exact 20k shared-document bootstrap and order statistic."""

    documents = len(ledger.document_ids)
    point_weights = torch.ones(1, documents, dtype=torch.float64)
    names, point_matrix = _coordinate_family(ledger, point_weights)
    point = point_matrix[0]
    generator = torch.Generator(device="cpu").manual_seed(discovery.BOOTSTRAP_SEED)
    samples = torch.randint(
        documents,
        (discovery.BOOTSTRAP_DRAWS, documents),
        generator=generator,
        dtype=torch.int64,
    )
    weights = torch.zeros(
        discovery.BOOTSTRAP_DRAWS, documents, dtype=torch.float64,
    )
    weights.scatter_add_(1, samples, torch.ones_like(samples, dtype=torch.float64))
    draw_names, draws = _coordinate_family(ledger, weights)
    if draw_names != names:
        raise AssertionError("successor bootstrap coordinate order changed")
    maximum_error = (draws - point).abs().amax(dim=1)
    critical = float(torch.sort(maximum_error).values[discovery.MAX_ERROR_ORDER_INDEX])
    support = {
        cell: {
            "positions": int(ledger.count[:, index].sum()),
            "documents": int((ledger.count[:, index] > 0).sum()),
        }
        for index, cell in enumerate(CELL_NAMES)
    }
    pair_support = {
        pair: {
            "positions": int(ledger.pair_count[:, index].sum()),
            "documents": int((ledger.pair_count[:, index] > 0).sum()),
        }
        for index, pair in enumerate(ledger.pair_names)
    }
    return SimultaneousSelectScore(
        coordinate_names=names,
        point=point,
        lower=point - critical,
        upper=point + critical,
        critical_value=critical,
        bootstrap_draws=discovery.BOOTSTRAP_DRAWS,
        bootstrap_seed=discovery.BOOTSTRAP_SEED,
        order_index=discovery.MAX_ERROR_ORDER_INDEX,
        support=support,
        pair_support=pair_support,
    )


def point_metric_table(ledger: SelectDocumentLedger) -> tuple[ArmCellPoint, ...]:
    """Report every frozen arm/cell point metric without exposing row-level values."""

    answer = []
    for arm_index, arm in enumerate(ledger.arm_names):
        for cell_index, cell in enumerate(CELL_NAMES):
            count = int(ledger.count[:, cell_index].sum())
            if count <= 0:
                raise ZeroDivisionError(f"zero token denominator for {cell}")
            answer.append(ArmCellPoint(
                arm=arm,
                cell=cell,
                count=count,
                ce=float(ledger.ce_sum[:, arm_index, cell_index].sum() / count),
                native_kl=float(
                    ledger.native_kl_sum[:, arm_index, cell_index].sum() / count
                ),
                top1_change=float(
                    ledger.top1_change_sum[:, arm_index, cell_index].sum() / count
                ),
                successor_margin=float(
                    ledger.successor_margin_sum[:, arm_index, cell_index].sum() / count
                ),
            ))
    return tuple(answer)


def decide_promotions(
    score: SimultaneousSelectScore, integrity: IntegrityEvidence,
) -> PromotionDecision:
    if (
        type(score) is not SimultaneousSelectScore
        or score.bootstrap_draws != discovery.BOOTSTRAP_DRAWS
        or score.bootstrap_seed != discovery.BOOTSTRAP_SEED
        or score.order_index != discovery.MAX_ERROR_ORDER_INDEX
        or len(score.coordinate_names) != len(score.point)
        or score.point.dtype != torch.float64
        or score.lower.dtype != torch.float64 or score.upper.dtype != torch.float64
    ):
        raise ValueError("successor simultaneous score is malformed")
    lookup = {name: index for index, name in enumerate(score.coordinate_names)}

    def value(name: str, currency: str) -> float:
        if name not in lookup:
            raise ValueError(f"successor score coordinate is absent: {name}")
        tensor = {"point": score.point, "lower": score.lower, "upper": score.upper}[currency]
        return float(tensor[lookup[name]])

    support_passed = all(
        score.support.get(cell) == {
            "positions": score.support.get(cell, {}).get("positions"),
            "documents": score.support.get(cell, {}).get("documents"),
        }
        and type(score.support[cell].get("positions")) is int
        and type(score.support[cell].get("documents")) is int
        and score.support[cell]["positions"] >= 200
        and score.support[cell]["documents"] >= 30
        for cell in POWERED_CELLS
    )
    common = (
        integrity.passed
        and support_passed
        and value("deletion_damage_positive_clean", "lower") > 0
        and value("deletion_specificity_wrong_source_clean", "lower") > 0
        and value("deletion_specificity_no_source_clean", "lower") > 0
    )
    passing = []
    for true_arm in discovery.PROMOTIVE_ARMS:
        ce_point = value(f"{true_arm}:ce_recovery", "point")
        arm_passed = (
            common
            and ce_point >= 0.80
            and value(f"{true_arm}:ce_recovery", "lower") >= 0.60
            and value(f"{true_arm}:margin_recovery", "point") >= 0.80
            and value(f"{true_arm}:margin_recovery", "lower") >= 0.60
            and value(f"{true_arm}:true_minus_null_ce_benefit", "lower") > 0
            and value(f"{true_arm}:null_ce_recovery", "upper") < 0.5 * ce_point
            and value(f"{true_arm}:all_position_ce_drift", "upper") <= 0.01
            and value(f"{true_arm}:abs_wrong_source_clean_ce_drift", "upper") <= 0.01
            and value(f"{true_arm}:abs_no_source_clean_ce_drift", "upper") <= 0.01
        )
        if arm_passed:
            passing.append(true_arm)
    passing.sort(key=lambda arm: (discovery.arm_stored_parameters(arm), _candidate_rank(arm)))
    selected = None if not passing else passing[0]
    return PromotionDecision(
        passing_arms=tuple(passing),
        selected_arm=selected,
        selected_rank=None if selected is None else _candidate_rank(selected),
        integrity_passed=integrity.passed,
        support_passed=support_passed,
    )


def _candidate_rank(arm: str) -> int:
    matches = tuple(candidate.rank for candidate in discovery.CANDIDATES if candidate.arm == arm)
    if len(matches) != 1:
        raise ValueError("promotive arm has no unique rank")
    return matches[0]


def evaluate_v1_readiness() -> SelectV1Readiness:
    """Return the exact prospective blockers in the frozen committed v1 protocol."""

    factor_signature = inspect.signature(StoredSuccessorFactors.__init__)
    source_parameters = (factor_signature.parameters["current_right"], factor_signature.parameters["saved_right"])
    omission_aware = any(parameter.default is not inspect.Parameter.empty for parameter in source_parameters)
    blockers: list[ReadinessBlocker] = []
    if not omission_aware:
        blockers.append(ReadinessBlocker(
            code="nonmaterializable_registered_diagnostics",
            evidence=(
                "CURRENT_ONLY and V1_ONLY are frozen in the 17-arm registry, but "
                "StoredSuccessorFactors requires and stores both source factors."
            ),
            cheapest_prospective_repair=(
                "Before outcomes, either source-close an omission-aware backend or explicitly "
                "amend v2 to drop the two nonpromotive diagnostics; do not fabricate zero factors."
            ),
        ))
    if not hasattr(discovery, "SELECT_LEXICONS") or not hasattr(
        discovery, "FROZEN_SELECT_LEXICON_REGISTRY_SHA256"
    ):
        blockers.append(ReadinessBlocker(
            code="digit_lexicon_not_frozen",
            evidence="v1 freezes generic lexicon validation but no exact digit token IDs/registry hash.",
            cheapest_prospective_repair=(
                "Freeze the exact digit surface-form token IDs, ordered registry, and registry "
                "SHA256 in a prospective v2 amendment before any SELECT access."
            ),
        ))
    row_fields = ("SELECT_DOCUMENTS", "SELECT_ROW_SEED", "SELECT_DOCUMENT_INDEX_RULE")
    if not all(hasattr(discovery, name) for name in row_fields):
        blockers.append(ReadinessBlocker(
            code="fresh_select_rule_not_frozen",
            evidence="v1 permits any positive document count and freezes no deterministic fresh-row rule.",
            cheapest_prospective_repair=(
                "Prospectively freeze one registry-excluding deterministic SELECT document count, "
                "index/seed rule, tokenizer identity, and create-only row receipt."
            ),
        ))
    status = "GO" if not blockers else "PROSPECTIVE_NO_GO"
    return SelectV1Readiness(
        status=status,
        authority_allowed=not blockers,
        row_freeze_allowed=not blockers,
        model_forward_allowed=not blockers,
        blockers=tuple(blockers),
        frozen_arm_count=len(discovery.ARM_NAMES),
        promotive_arm_count=len(discovery.PROMOTIVE_ARMS),
    )


def require_v1_launch_ready() -> None:
    readiness = evaluate_v1_readiness()
    if readiness.blockers:
        codes = ",".join(blocker.code for blocker in readiness.blockers)
        raise RuntimeError(f"ordered-successor SELECT v1 is prospectively NO-GO: {codes}")


__all__ = (
    "ArmCellPoint", "CELL_NAMES", "IntegrityEvidence", "POWERED_CELLS",
    "PromotionDecision",
    "ReadinessBlocker", "SelectDocumentLedger", "SelectV1Readiness",
    "SOURCE_PATHS", "SimultaneousSelectScore", "decide_promotions",
    "evaluate_v1_readiness", "point_metric_table", "require_v1_launch_ready",
    "score_select_ledger",
)
