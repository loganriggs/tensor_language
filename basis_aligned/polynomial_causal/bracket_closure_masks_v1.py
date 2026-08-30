"""CPU-only score metadata for the L13H8 bracket-closure canary.

The stack below is deliberately absent from every executable program.  It labels
already-tokenized rows after selection so that a constant tensor edit can be scored
on compatible closers and typed controls.  It is not a parser-guided router.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import torch


class BracketDomain(str, Enum):
    PROSE = "prose"
    CODE = "code"


class ClosureCondition(str, Enum):
    COMPATIBLE = "compatible_closer"
    INCOMPATIBLE = "incompatible_closer"
    NO_OPENER = "no_opener"
    QUOTE_CONTROL = "quote_control"
    PUNCTUATION_CONTROL = "punctuation_control"


DEPTH_BINS = ("depth_1", "depth_2", "depth_3_plus")
DISTANCE_BINS = ("distance_1_8", "distance_9_32", "distance_33_128", "distance_129_plus")


@dataclass(frozen=True)
class DelimiterFamily:
    name: str
    opener_ids: tuple[int, ...]
    closer_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("delimiter family name must be nonempty")
        for label, values in (("opener", self.opener_ids), ("closer", self.closer_ids)):
            if type(values) is not tuple or not values or len(set(values)) != len(values) or any(
                type(value) is not int or value < 0 for value in values
            ):
                raise ValueError(f"{label} IDs must be unique nonnegative integers")
        if set(self.opener_ids) & set(self.closer_ids):
            raise ValueError("one token cannot be both opener and closer in a family")


@dataclass(frozen=True)
class DelimiterRegistry:
    families: tuple[DelimiterFamily, ...]
    quote_control_ids: tuple[int, ...]
    punctuation_control_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.families) is not tuple or len(self.families) < 2 or any(
            type(family) is not DelimiterFamily for family in self.families
        ):
            raise ValueError("registry needs at least two exact delimiter families")
        if len({family.name for family in self.families}) != len(self.families):
            raise ValueError("delimiter family names must be unique")
        groups = [
            set(values)
            for family in self.families
            for values in (family.opener_ids, family.closer_ids)
        ]
        for label, values in (
            ("quote", self.quote_control_ids),
            ("punctuation", self.punctuation_control_ids),
        ):
            if type(values) is not tuple or not values or len(set(values)) != len(values) or any(
                type(value) is not int or value < 0 for value in values
            ):
                raise ValueError(f"{label} controls must be unique nonnegative integer IDs")
            groups.append(set(values))
        seen: set[int] = set()
        for group in groups:
            if seen & group:
                raise ValueError("delimiter and control token groups must be disjoint")
            seen |= group


@dataclass(frozen=True)
class BracketMasks:
    compatible: torch.Tensor
    incompatible: torch.Tensor
    no_opener: torch.Tensor
    quote_control: torch.Tensor
    punctuation_control: torch.Tensor
    family_index: torch.Tensor
    depth: torch.Tensor
    distance: torch.Tensor
    domain_index: torch.Tensor

    def named_cells(self) -> Mapping[str, torch.Tensor]:
        return {
            "compatible_closer": self.compatible,
            "incompatible_closer": self.incompatible,
            "no_opener": self.no_opener,
            "quote_control": self.quote_control,
            "punctuation_control": self.punctuation_control,
        }

    def validate(self) -> None:
        cells = tuple(self.named_cells().values())
        if not cells or any(
            not torch.is_tensor(cell) or cell.device.type != "cpu" or cell.dtype != torch.bool
            or cell.shape != cells[0].shape for cell in cells
        ):
            raise ValueError("bracket masks must be same-shaped CPU booleans")
        if bool((torch.stack(cells).to(torch.int8).sum(0) > 1).any()):
            raise ValueError("bracket target/control cells overlap")
        shape = cells[0].shape
        for value, dtype in (
            (self.family_index, torch.int16), (self.depth, torch.int16),
            (self.distance, torch.int16), (self.domain_index, torch.int8),
        ):
            if value.device.type != "cpu" or value.dtype != dtype or value.shape != shape:
                raise ValueError("bracket metadata currency changed")
        closer = self.compatible | self.incompatible | self.no_opener
        if not torch.equal(self.family_index.ge(0), closer):
            raise ValueError("family index must identify exactly closer targets")
        has_stack = self.compatible | self.incompatible
        if not torch.equal(self.depth.gt(0), has_stack) or not torch.equal(
            self.distance.gt(0), has_stack,
        ):
            raise ValueError("depth/distance must identify exactly nonempty-stack closers")


def _depth_bin(value: int) -> str:
    return "depth_1" if value == 1 else "depth_2" if value == 2 else "depth_3_plus"


def _distance_bin(value: int) -> str:
    if value <= 8:
        return "distance_1_8"
    if value <= 32:
        return "distance_9_32"
    if value <= 128:
        return "distance_33_128"
    return "distance_129_plus"


def stratified_cells(masks: BracketMasks) -> Mapping[str, torch.Tensor]:
    """Typed natural cells; parser-derived metadata is score-only."""

    masks.validate()
    answer: dict[str, torch.Tensor] = dict(masks.named_cells())
    for domain_index, domain in enumerate(BracketDomain):
        domain_mask = masks.domain_index.eq(domain_index)
        for family in range(int(masks.family_index.max().item()) + 1):
            family_mask = masks.family_index.eq(family)
            for condition_name, condition in (
                ("compatible", masks.compatible), ("incompatible", masks.incompatible),
            ):
                for depth_name in DEPTH_BINS:
                    depth_mask = (
                        masks.depth.eq(1) if depth_name == "depth_1" else
                        masks.depth.eq(2) if depth_name == "depth_2" else masks.depth.ge(3)
                    )
                    for distance_name in DISTANCE_BINS:
                        distance_mask = (
                            masks.distance.le(8) if distance_name == "distance_1_8" else
                            (masks.distance.ge(9) & masks.distance.le(32))
                            if distance_name == "distance_9_32" else
                            (masks.distance.ge(33) & masks.distance.le(128))
                            if distance_name == "distance_33_128" else masks.distance.ge(129)
                        )
                        name = ":".join((domain.value, f"family_{family}", condition_name,
                                         depth_name, distance_name))
                        answer[name] = domain_mask & family_mask & condition & depth_mask & distance_mask
    return answer


def build_bracket_masks(
    rows: torch.Tensor,
    registry: DelimiterRegistry,
    domains: tuple[BracketDomain, ...],
    *,
    first_prediction: int = 64,
) -> BracketMasks:
    """Parse prefixes only to label next-token targets; never route a forward."""

    if not torch.is_tensor(rows) or rows.device.type != "cpu" or rows.dtype != torch.long or (
        rows.ndim != 2 or rows.shape[1] < 2 or not rows.is_contiguous()
    ):
        raise ValueError("rows must be contiguous CPU int64 [documents,tokens]")
    if bool((rows < 0).any()):
        raise ValueError("token IDs must be nonnegative")
    predictions = rows.shape[1] - 1
    if type(first_prediction) is not int or not 0 <= first_prediction < predictions:
        raise ValueError("first_prediction is outside row support")
    if type(domains) is not tuple or len(domains) != rows.shape[0] or any(
        type(domain) is not BracketDomain for domain in domains
    ):
        raise ValueError("every document needs one typed code/prose domain")

    shape = (rows.shape[0], predictions)
    compatible = torch.zeros(shape, dtype=torch.bool)
    incompatible = torch.zeros(shape, dtype=torch.bool)
    no_opener = torch.zeros(shape, dtype=torch.bool)
    quote = torch.zeros(shape, dtype=torch.bool)
    punctuation = torch.zeros(shape, dtype=torch.bool)
    family_index = torch.full(shape, -1, dtype=torch.int16)
    depth = torch.zeros(shape, dtype=torch.int16)
    distance = torch.zeros(shape, dtype=torch.int16)
    domain_index = torch.empty(shape, dtype=torch.int8)

    opener_owner = {token: index for index, family in enumerate(registry.families)
                    for token in family.opener_ids}
    closer_owner = {token: index for index, family in enumerate(registry.families)
                    for token in family.closer_ids}
    quotes = set(registry.quote_control_ids)
    punctuation_ids = set(registry.punctuation_control_ids)

    for document in range(rows.shape[0]):
        domain_index[document].fill_(tuple(BracketDomain).index(domains[document]))
        stack: list[tuple[int, int]] = []
        values = rows[document].tolist()
        for prediction in range(predictions):
            query = values[prediction]
            if query in opener_owner:
                stack.append((opener_owner[query], prediction))
            elif query in closer_owner and stack and stack[-1][0] == closer_owner[query]:
                stack.pop()
            if prediction < first_prediction:
                continue
            target = values[prediction + 1]
            if target in closer_owner:
                target_family = closer_owner[target]
                family_index[document, prediction] = target_family
                if not stack:
                    no_opener[document, prediction] = True
                else:
                    top_family, opener_position = stack[-1]
                    depth[document, prediction] = len(stack)
                    distance[document, prediction] = prediction + 1 - opener_position
                    if top_family == target_family:
                        compatible[document, prediction] = True
                    else:
                        incompatible[document, prediction] = True
            elif target in quotes:
                quote[document, prediction] = True
            elif target in punctuation_ids:
                punctuation[document, prediction] = True

    result = BracketMasks(
        compatible, incompatible, no_opener, quote, punctuation,
        family_index, depth, distance, domain_index,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class SyntheticCanaryCell:
    domain: BracketDomain
    condition: ClosureCondition
    family_index: int | None
    depth_bin: str | None
    distance_bin: str | None


def synthetic_canary_design(family_count: int) -> tuple[SyntheticCanaryCell, ...]:
    """Frozen balanced cell registry; token realization is a later bound artifact."""

    if type(family_count) is not int or family_count < 2:
        raise ValueError("synthetic design needs at least two delimiter families")
    cells: list[SyntheticCanaryCell] = []
    for domain in BracketDomain:
        for family in range(family_count):
            for condition in (ClosureCondition.COMPATIBLE, ClosureCondition.INCOMPATIBLE):
                for depth_bin in DEPTH_BINS:
                    for distance_bin in DISTANCE_BINS:
                        cells.append(SyntheticCanaryCell(
                            domain, condition, family, depth_bin, distance_bin,
                        ))
            cells.append(SyntheticCanaryCell(
                domain, ClosureCondition.NO_OPENER, family, None, None,
            ))
        for condition in (ClosureCondition.QUOTE_CONTROL, ClosureCondition.PUNCTUATION_CONTROL):
            cells.append(SyntheticCanaryCell(domain, condition, None, None, None))
    return tuple(cells)


__all__ = (
    "BracketDomain", "BracketMasks", "ClosureCondition", "DelimiterFamily",
    "DelimiterRegistry", "SyntheticCanaryCell", "build_bracket_masks",
    "stratified_cells", "synthetic_canary_design",
)
