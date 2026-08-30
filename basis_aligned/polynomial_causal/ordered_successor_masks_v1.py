"""Outcome-blind masks for an ordered-successor behavioral panel.

The masks deliberately distinguish remote ordered succession from literal copying
and from an immediately preceding source token.  They operate only on already
tokenized CPU rows and never load a model, tokenizer, or experiment artifact.

Prediction column ``p`` uses ``rows[:, p]`` as the query token and
``rows[:, p + 1]`` as the target.  A remote source is searched for only in
``rows[:, max(0, p-window):p]``; the query token is therefore not accidentally
counted as remote evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class OrderedLexicon:
    """An ordered semantic family with one or more token IDs per item."""

    name: str
    items: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("lexicon name must be nonempty")
        if len(self.items) < 2:
            raise ValueError("ordered lexicon needs at least two items")
        seen: set[int] = set()
        for item in self.items:
            if not item:
                raise ValueError("every lexicon item needs at least one token ID")
            if len(set(item)) != len(item):
                raise ValueError("duplicate token ID within a lexicon item")
            if any((not isinstance(token_id, int)) or token_id < 0 for token_id in item):
                raise ValueError("token IDs must be nonnegative integers")
            overlap = seen.intersection(item)
            if overlap:
                raise ValueError(f"token IDs cannot name two ordered items: {sorted(overlap)}")
            seen.update(item)


@dataclass(frozen=True)
class SuccessorMasks:
    """Disjoint primary/control cells plus audit metadata for one lexicon."""

    eligible_target: torch.Tensor
    positive_clean: torch.Tensor
    successor_copy_overlap: torch.Tensor
    copy_only: torch.Tensor
    wrong_source_clean: torch.Tensor
    no_source_clean: torch.Tensor
    excluded_local_or_ambiguous: torch.Tensor
    pair_index: torch.Tensor

    def named_cells(self) -> Mapping[str, torch.Tensor]:
        return {
            "positive_clean": self.positive_clean,
            "successor_copy_overlap": self.successor_copy_overlap,
            "copy_only": self.copy_only,
            "wrong_source_clean": self.wrong_source_clean,
            "no_source_clean": self.no_source_clean,
            "excluded_local_or_ambiguous": self.excluded_local_or_ambiguous,
        }

    def validate_partition(self) -> None:
        cells = list(self.named_cells().values())
        stacked = torch.stack(cells, dim=0).to(torch.int8)
        if not torch.equal(stacked.sum(0), self.eligible_target.to(torch.int8)):
            raise AssertionError("successor cells must exactly partition eligible targets")
        if not torch.equal(self.pair_index.ge(0), self.eligible_target):
            raise AssertionError("pair_index and eligible_target disagree")


def _membership(rows: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    answer = torch.zeros_like(rows, dtype=torch.bool)
    for token_id in token_ids:
        answer |= rows.eq(token_id)
    return answer


def _strict_window_any(member: torch.Tensor, prediction_count: int, window: int) -> torch.Tensor:
    """Whether a member occurs in [p-window, p), for each prediction p."""

    cumulative = torch.nn.functional.pad(member.to(torch.int64).cumsum(1), (1, 0))
    positions = torch.arange(prediction_count, device=member.device)
    starts = (positions - window).clamp(min=0)
    return (cumulative[:, positions] - cumulative[:, starts]) > 0


def _prefix_through_query_any(member: torch.Tensor, prediction_count: int) -> torch.Tensor:
    """Whether a member occurs in rows[:, :p+1], including the query token."""

    cumulative = member.to(torch.int64).cumsum(1)
    positions = torch.arange(prediction_count, device=member.device)
    return cumulative[:, positions] > 0


def build_ordered_successor_masks(
    rows: torch.Tensor,
    lexicon: OrderedLexicon,
    *,
    window: int = 128,
    first_prediction: int = 64,
) -> SuccessorMasks:
    """Build non-cyclic ordered-successor masks for one tokenized row panel.

    ``positive_clean`` requires the correct predecessor remotely, no lexicon
    token at the current query, and no previous occurrence of the target.  The
    same no-local-token rule makes ``wrong_source_clean`` an exact source-identity
    control rather than a disguised immediate-successor example.
    """

    if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
        raise TypeError("rows must be a rank-2 torch.Tensor")
    if rows.device.type != "cpu":
        raise ValueError("outcome-blind mask construction is CPU-only")
    if rows.dtype not in (torch.int32, torch.int64):
        raise TypeError("rows must contain integer token IDs")
    if rows.shape[1] < 2:
        raise ValueError("rows need at least one input and one target token")
    if window <= 0:
        raise ValueError("window must be positive")
    prediction_count = rows.shape[1] - 1
    if not 0 <= first_prediction < prediction_count:
        raise ValueError("first_prediction is outside the prediction columns")

    shape = (rows.shape[0], prediction_count)
    eligible = torch.zeros(shape, dtype=torch.bool)
    positive = torch.zeros(shape, dtype=torch.bool)
    successor_copy = torch.zeros(shape, dtype=torch.bool)
    copy_only = torch.zeros(shape, dtype=torch.bool)
    wrong_source = torch.zeros(shape, dtype=torch.bool)
    no_source = torch.zeros(shape, dtype=torch.bool)
    excluded = torch.zeros(shape, dtype=torch.bool)
    pair_index = torch.full(shape, -1, dtype=torch.int16)

    all_ids = tuple(token_id for item in lexicon.items for token_id in item)
    any_family = _membership(rows, all_ids)
    any_family_remote = _strict_window_any(any_family, prediction_count, window)
    any_family_local = any_family[:, :prediction_count]

    scored = torch.zeros(shape, dtype=torch.bool)
    scored[:, first_prediction:] = True

    for index, (source_ids, target_ids) in enumerate(zip(lexicon.items[:-1], lexicon.items[1:])):
        is_target = _membership(rows[:, 1:], target_ids) & scored
        if not bool(is_target.any()):
            continue
        if bool((eligible & is_target).any()):
            raise AssertionError("a target token maps to multiple successor pairs")

        source_member = _membership(rows, source_ids)
        target_member = _membership(rows, target_ids)
        source_remote = _strict_window_any(source_member, prediction_count, window)
        target_seen = _prefix_through_query_any(target_member, prediction_count)
        local_family = any_family_local
        other_remote = any_family_remote & ~source_remote

        clean_context = ~target_seen & ~local_family
        pos = is_target & clean_context & source_remote
        succ_copy = is_target & target_seen & source_remote & ~local_family
        copy = is_target & target_seen & ~source_remote & ~local_family
        wrong = is_target & clean_context & ~source_remote & other_remote
        none = is_target & clean_context & ~source_remote & ~other_remote
        assigned = pos | succ_copy | copy | wrong | none
        reject = is_target & ~assigned

        eligible |= is_target
        positive |= pos
        successor_copy |= succ_copy
        copy_only |= copy
        wrong_source |= wrong
        no_source |= none
        excluded |= reject
        pair_index[is_target] = index

    result = SuccessorMasks(
        eligible_target=eligible,
        positive_clean=positive,
        successor_copy_overlap=successor_copy,
        copy_only=copy_only,
        wrong_source_clean=wrong_source,
        no_source_clean=no_source,
        excluded_local_or_ambiguous=excluded,
        pair_index=pair_index,
    )
    result.validate_partition()
    return result


def support_by_cell(masks: SuccessorMasks) -> dict[str, dict[str, int]]:
    """Return position and distinct-document support for each disjoint cell."""

    answer: dict[str, dict[str, int]] = {}
    for name, mask in masks.named_cells().items():
        answer[name] = {
            "positions": int(mask.sum()),
            "documents": int(mask.any(dim=1).sum()),
        }
    return answer
