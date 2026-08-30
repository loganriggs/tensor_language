"""Pure-CPU document statistics for source-closed circuit campaigns.

This module deliberately owns no model, row, filesystem, or publication capability.
Callers provide CPU logits and support masks, receive document-level sufficient
statistics, and may evaluate a frozen family of higher-is-better coordinates with
one simultaneous document bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


class CoordinateKind(str, Enum):
    """Supported, higher-is-better campaign coordinate currencies."""

    TARGET_DAMAGE = "target_damage"
    SPECIFICITY = "specificity"
    COLLATERAL = "collateral"
    EXTRACTION_RECOVERY = "extraction_recovery"
    OOD_RETENTION = "ood_retention"
    KL = "kl"
    TOP1 = "top1"


@dataclass(frozen=True)
class ArmCellSums:
    arm: str
    nll_sum: float
    correct_count: int


@dataclass(frozen=True)
class DirectedKLSums:
    source_arm: str
    target_arm: str
    kl_sum: float


@dataclass(frozen=True)
class DocumentCellSums:
    """Sufficient statistics for one source document and one support cell."""

    n: int
    support_sha256: str
    arms: tuple[ArmCellSums, ...]
    directed_kls: tuple[DirectedKLSums, ...]


@dataclass(frozen=True)
class CoordinateSpec:
    """A frozen scientific coordinate.

    All kinds are oriented so a larger value is better:

    * target_damage: CE(candidate) - CE(native), on ``cell``.
    * specificity: target damage minus the same damage on ``comparison_cell``.
    * collateral: ``limit`` - target damage.
    * extraction_recovery / ood_retention:
      (CE(stake) - CE(candidate)) / (CE(stake) - CE(native)).
    * kl: ``limit`` - KL(source || candidate).
    * top1: accuracy(candidate) - accuracy(native) + ``limit``.  Thus ``limit``
      is the allowed absolute accuracy drop.

    ``draw_group`` controls document pairing. Coordinates in the same draw group
    must use roles with identical ordered document IDs and receive the same
    bootstrap multiplicities.
    """

    name: str
    kind: CoordinateKind
    role: str
    cell: str
    native_arm: str = "native"
    candidate_arm: str = "candidate"
    comparison_cell: str | None = None
    stake_arm: str | None = None
    source_arm: str | None = None
    limit: float | None = None
    draw_group: str | None = None

    def __post_init__(self) -> None:
        strings = (self.name, self.role, self.cell, self.native_arm, self.candidate_arm)
        if any(not isinstance(value, str) or not value for value in strings):
            raise ValueError("coordinate names, role, cell, and arms must be nonempty strings")
        if type(self.kind) is not CoordinateKind:
            raise ValueError("coordinate kind must be a CoordinateKind")
        if self.draw_group is not None and (
            not isinstance(self.draw_group, str) or not self.draw_group
        ):
            raise ValueError("draw_group must be None or a nonempty string")
        if self.kind is CoordinateKind.SPECIFICITY:
            if not isinstance(self.comparison_cell, str) or not self.comparison_cell:
                raise ValueError("specificity requires comparison_cell")
        elif self.comparison_cell is not None:
            raise ValueError("comparison_cell is legal only for specificity")
        if self.kind in (CoordinateKind.EXTRACTION_RECOVERY, CoordinateKind.OOD_RETENTION):
            if not isinstance(self.stake_arm, str) or not self.stake_arm:
                raise ValueError("recovery/retention requires stake_arm")
        elif self.stake_arm is not None:
            raise ValueError("stake_arm is legal only for recovery/retention")
        if self.kind is CoordinateKind.KL:
            if not isinstance(self.source_arm, str) or not self.source_arm:
                raise ValueError("KL requires source_arm")
        elif self.source_arm is not None:
            raise ValueError("source_arm is legal only for KL")
        needs_limit = self.kind in (
            CoordinateKind.COLLATERAL, CoordinateKind.KL, CoordinateKind.TOP1,
        )
        if needs_limit != (self.limit is not None):
            raise ValueError("limit is required exactly for collateral, KL, and top1")
        if self.limit is not None and (
            type(self.limit) is not float or not math.isfinite(self.limit) or self.limit < 0
        ):
            raise ValueError("coordinate limit must be a finite nonnegative float")


@dataclass(frozen=True)
class BootstrapResult:
    coordinate_names: tuple[str, ...]
    point_estimates: torch.Tensor
    simultaneous_lower_bounds: torch.Tensor
    critical_value: float
    confidence: float
    repetitions: int
    passed: bool


def _seed_value(seed: str, group: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{group}".encode()).digest()[:8], "little")


def _support_digest(rows: torch.Tensor, masks: torch.Tensor, cell: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"circuit-campaign-support-v1\0")
    digest.update(cell.encode())
    digest.update(b"\0")
    digest.update(str(rows.dtype).encode())
    digest.update(str(tuple(rows.shape)).encode())
    digest.update(rows.contiguous().numpy().tobytes(order="C"))
    digest.update(str(tuple(masks.shape)).encode())
    digest.update(masks.contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


def reduce_document_batch(
    logits_by_arm: Mapping[str, torch.Tensor],
    rows: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
    document_ids: Sequence[str],
    *,
    kl_pairs: Sequence[tuple[str, str]] = (),
) -> dict[str, dict[str, DocumentCellSums]]:
    """Reduce raw logits to document sufficient statistics without retaining logits.

    Rows from the same document are aggregated in their supplied order. Directed KL
    is computed only for explicitly requested ordered arm pairs.
    """

    if (
        not torch.is_tensor(rows) or rows.device.type != "cpu" or rows.dtype != torch.long
        or rows.ndim != 2 or len(rows) == 0 or rows.shape[1] < 2
        or len(document_ids) != len(rows)
        or any(not isinstance(value, str) or not value for value in document_ids)
        or not logits_by_arm or not masks
    ):
        raise ValueError("rows, document IDs, arms, or masks are malformed")
    if any(not isinstance(name, str) or not name for name in logits_by_arm) or any(
        not isinstance(name, str) or not name for name in masks
    ):
        raise ValueError("arm and cell names must be nonempty strings")
    arm_names = tuple(sorted(logits_by_arm))
    cell_names = tuple(sorted(masks))
    expected = (len(rows), rows.shape[1] - 1)
    if any(
        not torch.is_tensor(mask) or mask.device.type != "cpu" or mask.dtype != torch.bool
        or tuple(mask.shape) != expected
        for mask in masks.values()
    ):
        raise ValueError("support masks are malformed")
    logits_shape: tuple[int, int, int] | None = None
    for logits in logits_by_arm.values():
        if (
            not torch.is_tensor(logits) or logits.device.type != "cpu"
            or logits.ndim != 3 or tuple(logits.shape[:2]) != expected
            or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())
        ):
            raise ValueError("arm logits are malformed")
        shape = tuple(logits.shape)
        if logits_shape is not None and shape != logits_shape:
            raise ValueError("arm logits do not share an exact shape")
        logits_shape = shape
    assert logits_shape is not None
    targets = rows[:, 1:]
    if bool((rows < 0).any()) or int(rows.max()) >= logits_shape[2]:
        raise ValueError("row tokens exceed the arm vocabulary")

    pairs = tuple(kl_pairs)
    if len(set(pairs)) != len(pairs) or any(
        not isinstance(pair, tuple) or len(pair) != 2
        or pair[0] not in logits_by_arm or pair[1] not in logits_by_arm
        or pair[0] == pair[1]
        for pair in pairs
    ):
        raise ValueError("directed KL pairs are malformed")

    logprobs = {
        arm: F.log_softmax(logits_by_arm[arm].double(), dim=-1) for arm in arm_names
    }
    target_logprobs = {
        arm: value.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        for arm, value in logprobs.items()
    }
    correct = {arm: logits_by_arm[arm].argmax(-1) == targets for arm in arm_names}
    point_kls: dict[tuple[str, str], torch.Tensor] = {}
    for source, target in pairs:
        value = (
            logprobs[source].exp() * (logprobs[source] - logprobs[target])
        ).sum(-1)
        if float(value.min()) < -1e-12:
            raise RuntimeError("directed KL is numerically negative")
        point_kls[(source, target)] = value.clamp_min(0)

    document_order = tuple(dict.fromkeys(document_ids))
    output: dict[str, dict[str, DocumentCellSums]] = {}
    for document in document_order:
        row_indices = torch.tensor(
            [index for index, value in enumerate(document_ids) if value == document],
            dtype=torch.long,
        )
        output[document] = {}
        for cell in cell_names:
            selected_mask = masks[cell].index_select(0, row_indices)
            n = int(selected_mask.sum())
            arms = tuple(
                ArmCellSums(
                    arm=arm,
                    nll_sum=float(
                        -target_logprobs[arm].index_select(0, row_indices)[selected_mask].sum()
                    ),
                    correct_count=int(
                        correct[arm].index_select(0, row_indices)[selected_mask].sum()
                    ),
                )
                for arm in arm_names
            )
            kls = tuple(
                DirectedKLSums(
                    source_arm=source,
                    target_arm=target,
                    kl_sum=float(
                        point_kls[(source, target)].index_select(0, row_indices)[selected_mask].sum()
                    ),
                )
                for source, target in sorted(pairs)
            )
            output[document][cell] = DocumentCellSums(
                n=n,
                support_sha256=_support_digest(
                    rows.index_select(0, row_indices), selected_mask, cell,
                ),
                arms=arms,
                directed_kls=kls,
            )
    return output


def _validate_role_ledger(
    ledger: Mapping[str, Mapping[str, DocumentCellSums]],
) -> tuple[str, ...]:
    documents = tuple(sorted(ledger))
    if not documents:
        raise ValueError("document ledger is empty")
    schema: tuple[tuple[str, ...], tuple[tuple[str, str], ...], tuple[str, ...]] | None = None
    for document in documents:
        if not isinstance(document, str) or not document or not ledger[document]:
            raise ValueError("document ledger is malformed")
        cell_names = tuple(sorted(ledger[document]))
        for cell in cell_names:
            value = ledger[document][cell]
            if type(value) is not DocumentCellSums:
                raise ValueError("document cell must have exact DocumentCellSums type")
            arm_names = tuple(item.arm for item in value.arms)
            kl_names = tuple((item.source_arm, item.target_arm) for item in value.directed_kls)
            current = (arm_names, kl_names, cell_names)
            if schema is None:
                schema = current
            elif schema != current:
                raise ValueError("document ledgers do not share an exact schema")
            if (
                type(value.n) is not int or value.n < 0
                or not isinstance(value.support_sha256, str) or len(value.support_sha256) != 64
                or any(character not in "0123456789abcdef" for character in value.support_sha256)
                or len(set(arm_names)) != len(arm_names)
                or len(set(kl_names)) != len(kl_names)
            ):
                raise ValueError("document sufficient statistics are malformed")
            for item in value.arms:
                if (
                    type(item) is not ArmCellSums or not isinstance(item.arm, str) or not item.arm
                    or type(item.nll_sum) is not float or not math.isfinite(item.nll_sum)
                    or type(item.correct_count) is not int
                    or not 0 <= item.correct_count <= value.n
                ):
                    raise ValueError("arm sufficient statistics are malformed")
            for item in value.directed_kls:
                if (
                    type(item) is not DirectedKLSums
                    or item.source_arm not in arm_names or item.target_arm not in arm_names
                    or item.source_arm == item.target_arm
                    or type(item.kl_sum) is not float or not math.isfinite(item.kl_sum)
                    or item.kl_sum < 0
                ):
                    raise ValueError("KL sufficient statistics are malformed")
    return documents


def _weighted_cell(
    ledger: Mapping[str, Mapping[str, DocumentCellSums]], cell: str,
    multiplicities: Mapping[str, int],
) -> tuple[int, dict[str, float], dict[str, int], dict[tuple[str, str], float]]:
    total_n = 0
    nll: dict[str, float] = {}
    correct: dict[str, int] = {}
    kl: dict[tuple[str, str], float] = {}
    for document in sorted(ledger):
        if cell not in ledger[document]:
            raise ValueError(f"coordinate cell is absent: {cell}")
        value = ledger[document][cell]
        weight = multiplicities[document]
        total_n += weight * value.n
        for arm in value.arms:
            nll[arm.arm] = nll.get(arm.arm, 0.0) + weight * arm.nll_sum
            correct[arm.arm] = correct.get(arm.arm, 0) + weight * arm.correct_count
        for item in value.directed_kls:
            key = (item.source_arm, item.target_arm)
            kl[key] = kl.get(key, 0.0) + weight * item.kl_sum
    if total_n <= 0:
        raise ZeroDivisionError("coordinate has zero token denominator")
    return total_n, nll, correct, kl


def evaluate_coordinate(
    ledger: Mapping[str, Mapping[str, DocumentCellSums]], spec: CoordinateSpec,
    multiplicities: Mapping[str, int] | None = None,
) -> float:
    """Evaluate one coordinate from pooled document sufficient statistics."""

    documents = _validate_role_ledger(ledger)
    weights = {name: 1 for name in documents} if multiplicities is None else dict(multiplicities)
    if set(weights) != set(documents) or any(
        type(value) is not int or value < 0 for value in weights.values()
    ):
        raise ValueError("document multiplicities are malformed")

    n, nll, correct, kl = _weighted_cell(ledger, spec.cell, weights)
    required_arms = {spec.native_arm, spec.candidate_arm}
    if not required_arms <= set(nll):
        raise ValueError("coordinate references an absent arm")
    delta = (nll[spec.candidate_arm] - nll[spec.native_arm]) / n
    if spec.kind is CoordinateKind.TARGET_DAMAGE:
        value = delta
    elif spec.kind is CoordinateKind.SPECIFICITY:
        other_n, other_nll, _, _ = _weighted_cell(
            ledger, spec.comparison_cell or "", weights,
        )
        if not required_arms <= set(other_nll):
            raise ValueError("specificity references an absent arm")
        value = delta - (
            other_nll[spec.candidate_arm] - other_nll[spec.native_arm]
        ) / other_n
    elif spec.kind is CoordinateKind.COLLATERAL:
        assert spec.limit is not None
        value = spec.limit - delta
    elif spec.kind in (CoordinateKind.EXTRACTION_RECOVERY, CoordinateKind.OOD_RETENTION):
        assert spec.stake_arm is not None
        if spec.stake_arm not in nll:
            raise ValueError("recovery/retention references an absent stake arm")
        denominator = nll[spec.stake_arm] - nll[spec.native_arm]
        if denominator <= 0:
            raise ValueError("recovery/retention stake is not positive")
        value = (nll[spec.stake_arm] - nll[spec.candidate_arm]) / denominator
    elif spec.kind is CoordinateKind.KL:
        assert spec.source_arm is not None and spec.limit is not None
        pair = (spec.source_arm, spec.candidate_arm)
        if pair not in kl:
            raise ValueError("coordinate references an absent directed KL pair")
        value = spec.limit - kl[pair] / n
    elif spec.kind is CoordinateKind.TOP1:
        assert spec.limit is not None
        value = (correct[spec.candidate_arm] - correct[spec.native_arm]) / n + spec.limit
    else:  # pragma: no cover - exact enum validation makes this unreachable.
        raise AssertionError(spec.kind)
    if not math.isfinite(value):
        raise RuntimeError("coordinate evaluation is not finite")
    return float(value)


def evaluate_coordinates(
    role_ledgers: Mapping[str, Mapping[str, Mapping[str, DocumentCellSums]]],
    specs: Sequence[CoordinateSpec],
    multiplicities_by_group: Mapping[str, Mapping[str, int]] | None = None,
) -> torch.Tensor:
    """Evaluate an ordered coordinate family, optionally with shared draw weights."""

    specs = tuple(specs)
    if not specs or any(type(spec) is not CoordinateSpec for spec in specs):
        raise ValueError("coordinate family must contain exact CoordinateSpec values")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("coordinate family must be nonempty with unique names")
    if set(role_ledgers) != {spec.role for spec in specs}:
        raise ValueError("role ledgers must exactly equal coordinate roles")
    values = []
    for spec in specs:
        group = spec.draw_group or spec.role
        weights = None if multiplicities_by_group is None else multiplicities_by_group[group]
        values.append(evaluate_coordinate(role_ledgers[spec.role], spec, weights))
    return torch.tensor(values, dtype=torch.float64)


def _document_vectors(
    ledger: Mapping[str, Mapping[str, DocumentCellSums]], cell: str,
) -> tuple[
    torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor],
    dict[tuple[str, str], torch.Tensor],
]:
    documents = tuple(sorted(ledger))
    if any(cell not in ledger[document] for document in documents):
        raise ValueError(f"coordinate cell is absent: {cell}")
    values = [ledger[document][cell] for document in documents]
    counts = torch.tensor([value.n for value in values], dtype=torch.float64)
    arms = tuple(item.arm for item in values[0].arms)
    pairs = tuple(
        (item.source_arm, item.target_arm) for item in values[0].directed_kls
    )
    nll = {
        arm: torch.tensor([
            next(item.nll_sum for item in value.arms if item.arm == arm)
            for value in values
        ], dtype=torch.float64)
        for arm in arms
    }
    correct = {
        arm: torch.tensor([
            next(item.correct_count for item in value.arms if item.arm == arm)
            for value in values
        ], dtype=torch.float64)
        for arm in arms
    }
    kl = {
        pair: torch.tensor([
            next(
                item.kl_sum for item in value.directed_kls
                if (item.source_arm, item.target_arm) == pair
            )
            for value in values
        ], dtype=torch.float64)
        for pair in pairs
    }
    return counts, nll, correct, kl


def _bootstrap_coordinate(
    ledger: Mapping[str, Mapping[str, DocumentCellSums]],
    spec: CoordinateSpec,
    weights: torch.Tensor,
) -> torch.Tensor:
    counts, nll, correct, kl = _document_vectors(ledger, spec.cell)
    required_arms = {spec.native_arm, spec.candidate_arm}
    if not required_arms <= set(nll):
        raise ValueError("coordinate references an absent arm")
    denominator = weights @ counts
    if bool((denominator <= 0).any()):
        raise ZeroDivisionError("coordinate has zero token denominator")
    delta = (weights @ (nll[spec.candidate_arm] - nll[spec.native_arm])) / denominator
    if spec.kind is CoordinateKind.TARGET_DAMAGE:
        result = delta
    elif spec.kind is CoordinateKind.SPECIFICITY:
        other_counts, other_nll, _, _ = _document_vectors(
            ledger, spec.comparison_cell or "",
        )
        if not required_arms <= set(other_nll):
            raise ValueError("specificity references an absent arm")
        other_denominator = weights @ other_counts
        if bool((other_denominator <= 0).any()):
            raise ZeroDivisionError("coordinate has zero token denominator")
        result = delta - (
            weights @ (other_nll[spec.candidate_arm] - other_nll[spec.native_arm])
        ) / other_denominator
    elif spec.kind is CoordinateKind.COLLATERAL:
        assert spec.limit is not None
        result = spec.limit - delta
    elif spec.kind in (CoordinateKind.EXTRACTION_RECOVERY, CoordinateKind.OOD_RETENTION):
        assert spec.stake_arm is not None
        if spec.stake_arm not in nll:
            raise ValueError("recovery/retention references an absent stake arm")
        stake = weights @ (nll[spec.stake_arm] - nll[spec.native_arm])
        if bool((stake <= 0).any()):
            raise ValueError("recovery/retention stake is not positive")
        result = (
            weights @ (nll[spec.stake_arm] - nll[spec.candidate_arm])
        ) / stake
    elif spec.kind is CoordinateKind.KL:
        assert spec.source_arm is not None and spec.limit is not None
        pair = (spec.source_arm, spec.candidate_arm)
        if pair not in kl:
            raise ValueError("coordinate references an absent directed KL pair")
        result = spec.limit - (weights @ kl[pair]) / denominator
    elif spec.kind is CoordinateKind.TOP1:
        assert spec.limit is not None
        result = spec.limit + (
            weights @ (correct[spec.candidate_arm] - correct[spec.native_arm])
        ) / denominator
    else:  # pragma: no cover
        raise AssertionError(spec.kind)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("bootstrap coordinate evaluation is not finite")
    return result


def simultaneous_document_bootstrap(
    role_ledgers: Mapping[str, Mapping[str, Mapping[str, DocumentCellSums]]],
    specs: Sequence[CoordinateSpec],
    *,
    repetitions: int,
    seed: str,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Compute a shared-draw, one-sided simultaneous document confidence band."""

    specs = tuple(specs)
    if (
        type(repetitions) is not int or repetitions <= 0
        or not isinstance(seed, str) or not seed
        or type(confidence) is not float or not math.isfinite(confidence)
        or not 0 < confidence < 1
    ):
        raise ValueError("bootstrap configuration is malformed")
    point = evaluate_coordinates(role_ledgers, specs)

    groups: dict[str, tuple[str, ...]] = {}
    for spec in specs:
        group = spec.draw_group or spec.role
        documents = _validate_role_ledger(role_ledgers[spec.role])
        if group in groups and groups[group] != documents:
            raise ValueError("roles in one draw group must share exact document IDs")
        groups[group] = documents

    group_weights: dict[str, torch.Tensor] = {}
    for group, documents in groups.items():
        generator = torch.Generator().manual_seed(_seed_value(seed, group))
        draws = torch.randint(
            len(documents), (repetitions, len(documents)), generator=generator,
        )
        weights = torch.zeros(repetitions, len(documents), dtype=torch.float64)
        weights.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
        group_weights[group] = weights

    replicates = torch.stack([
        _bootstrap_coordinate(
            role_ledgers[spec.role], spec, group_weights[spec.draw_group or spec.role],
        )
        for spec in specs
    ], dim=1)
    maxima = (replicates - point).max(dim=1).values.sort().values
    critical = float(maxima[math.ceil(confidence * repetitions) - 1])
    lower = point - critical
    return BootstrapResult(
        coordinate_names=tuple(spec.name for spec in specs),
        point_estimates=point,
        simultaneous_lower_bounds=lower,
        critical_value=critical,
        confidence=confidence,
        repetitions=repetitions,
        passed=bool((lower > 0).all()),
    )


__all__ = (
    "ArmCellSums",
    "BootstrapResult",
    "CoordinateKind",
    "CoordinateSpec",
    "DirectedKLSums",
    "DocumentCellSums",
    "evaluate_coordinate",
    "evaluate_coordinates",
    "reduce_document_batch",
    "simultaneous_document_bootstrap",
)
