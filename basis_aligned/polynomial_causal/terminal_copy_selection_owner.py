"""Outcome-sealed batch owner for the frozen E4 attention-copy screen.

The owner runs one exact native forward and every frozen physical candidate on the
same rows, reduces each pair to document-level sufficient statistics, and returns no
logit tensor.  It loads neither rows, means, checkpoints, nor authorities itself.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
from terminal_copy_attention_dispatcher import PhysicalCandidateDispatcher
from terminal_copy_attention_owner import CandidateForwardOwner, CandidateOwnerClosure
from terminal_copy_streaming_statistics import (
    CELL_NAMES,
    FROZEN_CANDIDATES,
    DocumentCellSums,
    reduce_document_batch,
)


LAYER_COUNT = 18
MAX_HEAD_RECOMPOSITION_RELATIVE_ERROR = 0.003


@dataclass(frozen=True)
class SelectionBatchClosure:
    document_ids: tuple[str, ...]
    native_attention_calls: tuple[int, ...]
    native_mlp_calls: tuple[int, ...]
    native_unembedding_calls: int
    candidate_unembedding_calls: tuple[int, ...]
    candidate_closures: tuple[CandidateOwnerClosure, ...]
    raw_logits_returned: bool
    closed: bool


@dataclass(frozen=True)
class SelectionBatchResult:
    ledgers: Mapping[str, Mapping[str, Mapping[str, DocumentCellSums]]]
    closure: SelectionBatchClosure


@dataclass(frozen=True)
class MergedSelectionBatches:
    ledgers: Mapping[str, Mapping[str, Mapping[str, DocumentCellSums]]]
    batch_closures: tuple[SelectionBatchClosure, ...]
    ordered_document_ids: tuple[str, ...]


@dataclass(frozen=True)
class SyntheticPairEffect:
    item_id: str
    native_did: float
    candidate_did: float
    candidate_minus_native_did: float


@dataclass(frozen=True)
class SyntheticBatchClosure:
    item_ids: tuple[str, ...]
    native_attention_calls: tuple[int, ...]
    native_mlp_calls: tuple[int, ...]
    native_unembedding_calls: int
    candidate_unembedding_calls: tuple[int, ...]
    candidate_closures: tuple[CandidateOwnerClosure, ...]
    raw_logits_returned: bool
    closed: bool


@dataclass(frozen=True)
class SyntheticBatchResult:
    effects: Mapping[str, tuple[SyntheticPairEffect, ...]]
    closure: SyntheticBatchClosure


@dataclass(frozen=True)
class MergedSyntheticBatches:
    effects: Mapping[str, tuple[SyntheticPairEffect, ...]]
    batch_closures: tuple[SyntheticBatchClosure, ...]
    ordered_item_ids: tuple[str, ...]


def _seal_ledgers(
    value: Mapping[str, Mapping[str, Mapping[str, DocumentCellSums]]],
) -> Mapping[str, Mapping[str, Mapping[str, DocumentCellSums]]]:
    return MappingProxyType({
        candidate: MappingProxyType({
            document: MappingProxyType(dict(cells))
            for document, cells in documents.items()
        })
        for candidate, documents in value.items()
    })


class SelectionBatchOwner:
    """Own a single native-plus-eight-candidate batch reduction."""

    def __init__(self, dispatcher: PhysicalCandidateDispatcher) -> None:
        if not isinstance(dispatcher, PhysicalCandidateDispatcher):
            raise ValueError("selection owner requires the physical dispatcher")
        self._dispatcher: PhysicalCandidateDispatcher | None = copy.deepcopy(dispatcher)
        self._active = False
        self._closed = False
        self._failed = False

    def _require_open(self) -> PhysicalCandidateDispatcher:
        if self._failed:
            raise RuntimeError("selection batch owner is failed and cannot be reused")
        if self._closed or self._dispatcher is None:
            raise RuntimeError("selection batch owner is closed")
        return self._dispatcher

    @torch.no_grad()
    def run(
        self,
        model: torch.nn.Module,
        tokens: torch.Tensor,
        rows: torch.Tensor,
        masks: Mapping[str, torch.Tensor],
        document_ids: Sequence[str],
        *,
        require_production: bool = True,
    ) -> SelectionBatchResult:
        dispatcher = self._require_open()
        if self._active:
            raise RuntimeError("selection batch owner is not reentrant")
        documents = tuple(document_ids)
        if (
            not torch.is_tensor(rows) or rows.device.type != "cpu"
            or rows.dtype != torch.long or rows.ndim != 2
            or tuple(rows.shape[1:]) != (257,)
            or len(rows) != len(documents) or len(set(documents)) != len(documents)
            or set(masks) != set(CELL_NAMES)
            or any(
                not torch.is_tensor(mask) or mask.device.type != "cpu"
                or mask.dtype != torch.bool or tuple(mask.shape) != (len(rows), 256)
                for mask in masks.values()
            )
            or not torch.is_tensor(tokens) or tokens.dtype != torch.long
            or tuple(tokens.shape) != (len(rows), 256)
            or not torch.equal(tokens.detach().to("cpu"), rows[:, :256])
        ):
            raise ValueError("selection batch rows, masks, IDs, or model inputs are malformed")
        if require_production and len(rows) != 4:
            raise ValueError("production selection batches must contain exactly four documents")

        self._active = True
        success = False
        native_attention = [0] * LAYER_COUNT
        native_mlp = [0] * LAYER_COUNT
        candidate_closures: list[CandidateOwnerClosure] = []
        ledgers: dict[str, Mapping[str, Mapping[str, DocumentCellSums]]] = {}

        def native_attention_dispatch(event: facade.AttentionEvent):
            native_attention[event.site] += 1
            return event.block.attn(event.state, event.first_value)

        def native_mlp_dispatch(event: facade.EarlyMLPEvent):
            native_mlp[event.site] += 1
            return event.block.mlp(event.state)

        try:
            native_logits = facade.forward_with_dispatch(
                model, tokens, native_attention_dispatch, native_mlp_dispatch,
                require_production=require_production,
            ).detach().to("cpu")
            for candidate in FROZEN_CANDIDATES:
                candidate_owner = CandidateForwardOwner(
                    candidate=candidate, dispatcher=dispatcher,
                )
                candidate_logits = candidate_owner.run(
                    model, tokens, require_production=require_production,
                ).detach().to("cpu")
                ledgers[candidate] = reduce_document_batch(
                    native_logits, candidate_logits, rows, masks, documents,
                )
                del candidate_logits
                candidate_closures.append(candidate_owner.close())
            del native_logits
            closure = SelectionBatchClosure(
                document_ids=documents,
                native_attention_calls=tuple(native_attention),
                native_mlp_calls=tuple(native_mlp),
                native_unembedding_calls=1,
                candidate_unembedding_calls=(1,) * len(FROZEN_CANDIDATES),
                candidate_closures=tuple(candidate_closures),
                raw_logits_returned=False,
                closed=True,
            )
            result = SelectionBatchResult(ledgers=_seal_ledgers(ledgers), closure=closure)
            self._dispatcher = None
            self._closed = True
            success = True
            return result
        finally:
            self._active = False
            if not success:
                self._failed = True


def _synthetic_dids(
    logits: torch.Tensor,
    query_positions: Sequence[int],
    successor_y: Sequence[int],
    successor_z: Sequence[int],
) -> tuple[float, ...]:
    """Reduce alternating q->y/q->z histories to reciprocal association DiDs."""

    pair_count = len(query_positions)
    if (
        not torch.is_tensor(logits) or logits.device.type != "cpu"
        or logits.ndim != 3 or tuple(logits.shape[:2]) != (2 * pair_count, 256)
        or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())
        or len(successor_y) != pair_count or len(successor_z) != pair_count
        or any(type(position) is not int or not 64 <= position < 256 for position in query_positions)
        or any(
            type(token) is not int or not 0 <= token < logits.shape[-1]
            for token in (*successor_y, *successor_z)
        )
    ):
        raise ValueError("synthetic selection logits or coordinates are malformed")
    output = []
    for index, (position, token_y, token_z) in enumerate(
        zip(query_positions, successor_y, successor_z, strict=True)
    ):
        selected = logits[2 * index:2 * index + 2, position, :].double()
        logprob = F.log_softmax(selected, dim=-1)
        preference = logprob[:, token_y] - logprob[:, token_z]
        output.append(float(preference[0] - preference[1]))
    return tuple(output)


class SyntheticSelectionBatchOwner:
    """Own one descriptive reciprocal-crossover batch without releasing logits."""

    def __init__(self, dispatcher: PhysicalCandidateDispatcher) -> None:
        if not isinstance(dispatcher, PhysicalCandidateDispatcher):
            raise ValueError("synthetic selection owner requires the physical dispatcher")
        self._dispatcher: PhysicalCandidateDispatcher | None = copy.deepcopy(dispatcher)
        self._active = False
        self._closed = False
        self._failed = False

    def _require_open(self) -> PhysicalCandidateDispatcher:
        if self._failed:
            raise RuntimeError("synthetic selection owner is failed and cannot be reused")
        if self._closed or self._dispatcher is None:
            raise RuntimeError("synthetic selection owner is closed")
        return self._dispatcher

    @torch.no_grad()
    def run(
        self,
        model: torch.nn.Module,
        tokens: torch.Tensor,
        rows: torch.Tensor,
        item_ids: Sequence[str],
        query_positions: Sequence[int],
        successor_y: Sequence[int],
        successor_z: Sequence[int],
        *,
        require_production: bool = True,
    ) -> SyntheticBatchResult:
        dispatcher = self._require_open()
        if self._active:
            raise RuntimeError("synthetic selection owner is not reentrant")
        items = tuple(item_ids)
        pair_count = len(items)
        if (
            pair_count <= 0 or len(set(items)) != pair_count
            or any(not isinstance(item, str) or not item for item in items)
            or not torch.is_tensor(rows) or rows.device.type != "cpu"
            or rows.dtype != torch.long or tuple(rows.shape) != (2 * pair_count, 257)
            or not torch.is_tensor(tokens) or tokens.dtype != torch.long
            or tuple(tokens.shape) != (2 * pair_count, 256)
            or not torch.equal(tokens.detach().to("cpu"), rows[:, :256])
            or len(query_positions) != pair_count
            or len(successor_y) != pair_count or len(successor_z) != pair_count
        ):
            raise ValueError("synthetic selection rows, IDs, or coordinates are malformed")
        if require_production and pair_count != 2:
            raise ValueError("production synthetic batches must contain exactly two crossover pairs")

        self._active = True
        success = False
        native_attention = [0] * LAYER_COUNT
        native_mlp = [0] * LAYER_COUNT
        candidate_closures: list[CandidateOwnerClosure] = []
        effects: dict[str, tuple[SyntheticPairEffect, ...]] = {}

        def native_attention_dispatch(event: facade.AttentionEvent):
            native_attention[event.site] += 1
            return event.block.attn(event.state, event.first_value)

        def native_mlp_dispatch(event: facade.EarlyMLPEvent):
            native_mlp[event.site] += 1
            return event.block.mlp(event.state)

        try:
            native_logits = facade.forward_with_dispatch(
                model, tokens, native_attention_dispatch, native_mlp_dispatch,
                require_production=require_production,
            ).detach().to("cpu")
            native_dids = _synthetic_dids(
                native_logits, query_positions, successor_y, successor_z,
            )
            for candidate in FROZEN_CANDIDATES:
                candidate_owner = CandidateForwardOwner(
                    candidate=candidate, dispatcher=dispatcher,
                )
                candidate_logits = candidate_owner.run(
                    model, tokens, require_production=require_production,
                ).detach().to("cpu")
                candidate_dids = _synthetic_dids(
                    candidate_logits, query_positions, successor_y, successor_z,
                )
                effects[candidate] = tuple(
                    SyntheticPairEffect(
                        item_id=item,
                        native_did=native,
                        candidate_did=changed,
                        candidate_minus_native_did=changed - native,
                    )
                    for item, native, changed in zip(
                        items, native_dids, candidate_dids, strict=True,
                    )
                )
                del candidate_logits
                candidate_closures.append(candidate_owner.close())
            del native_logits
            closure = SyntheticBatchClosure(
                item_ids=items,
                native_attention_calls=tuple(native_attention),
                native_mlp_calls=tuple(native_mlp),
                native_unembedding_calls=1,
                candidate_unembedding_calls=(1,) * len(FROZEN_CANDIDATES),
                candidate_closures=tuple(candidate_closures),
                raw_logits_returned=False,
                closed=True,
            )
            self._dispatcher = None
            self._closed = True
            success = True
            return SyntheticBatchResult(
                effects=MappingProxyType(effects), closure=closure,
            )
        finally:
            self._active = False
            if not success:
                self._failed = True


def merge_selection_batches(
    results: Sequence[SelectionBatchResult],
    expected_document_ids: Sequence[str],
) -> MergedSelectionBatches:
    """Merge sealed batch statistics while enforcing the exact forward-call census."""

    if not results:
        raise ValueError("selection batch result bank is empty")
    combined: dict[str, dict[str, Mapping[str, DocumentCellSums]]] = {
        candidate: {} for candidate in FROZEN_CANDIDATES
    }
    expected_documents = tuple(expected_document_ids)
    if not expected_documents or len(set(expected_documents)) != len(expected_documents):
        raise ValueError("expected selection document order is malformed")
    seen_documents: set[str] = set()
    observed_documents: list[str] = []
    retained_closures: list[SelectionBatchClosure] = []
    for result in results:
        if not isinstance(result, SelectionBatchResult):
            raise ValueError("selection batch result has wrong type")
        closure = result.closure
        documents = closure.document_ids
        candidate_closures = closure.candidate_closures
        expected_plans = {
            candidate: PhysicalCandidateDispatcher.plan(candidate)
            for candidate in FROZEN_CANDIDATES
        }
        if (
            not closure.closed or closure.raw_logits_returned
            or closure.native_unembedding_calls != 1
            or closure.candidate_unembedding_calls != (1,) * len(FROZEN_CANDIDATES)
            or closure.native_attention_calls != (1,) * LAYER_COUNT
            or closure.native_mlp_calls != (1,) * LAYER_COUNT
            or tuple(result.ledgers) != FROZEN_CANDIDATES
            or len(candidate_closures) != len(FROZEN_CANDIDATES)
            or tuple(item.candidate for item in candidate_closures) != FROZEN_CANDIDATES
            or any(
                item.attempted_batch_calls != 1 or item.batch_calls != 1
                or item.document_calls != len(documents) or not item.closed
                or item.native_mlp_calls != (1,) * LAYER_COUNT
                or item.selected_layer_heads != expected_plans[item.candidate]
                or item.native_attention_calls != tuple(
                    0 if layer in dict(expected_plans[item.candidate]) else 1
                    for layer in range(LAYER_COUNT)
                )
                or item.adapter_attention_calls != tuple(
                    1 if layer in dict(expected_plans[item.candidate]) else 0
                    for layer in range(LAYER_COUNT)
                )
                or not math.isfinite(item.maximum_head_recomposition_abs_error)
                or not math.isfinite(item.maximum_head_recomposition_relative_error)
                or item.maximum_head_recomposition_abs_error < 0
                or item.maximum_head_recomposition_relative_error < 0
                or item.maximum_head_recomposition_relative_error
                    > MAX_HEAD_RECOMPOSITION_RELATIVE_ERROR
                for item in candidate_closures
            )
            or any(tuple(result.ledgers[candidate]) != documents for candidate in FROZEN_CANDIDATES)
            or seen_documents.intersection(documents)
        ):
            raise RuntimeError("selection batch closure or call census is malformed")
        seen_documents.update(documents)
        observed_documents.extend(documents)
        retained_closures.append(closure)
        reference = result.ledgers[FROZEN_CANDIDATES[0]]
        for candidate in FROZEN_CANDIDATES:
            for document in documents:
                for cell in CELL_NAMES:
                    left = reference[document][cell]
                    right = result.ledgers[candidate][document][cell]
                    if (
                        not isinstance(right, DocumentCellSums)
                        or right.n != left.n
                        or right.native_nll_sum != left.native_nll_sum
                        or right.native_correct_count != left.native_correct_count
                        or right.support_sha256 != left.support_sha256
                    ):
                        raise RuntimeError("selection candidates do not share one native baseline")
            combined[candidate].update(result.ledgers[candidate])
    if tuple(observed_documents) != expected_documents:
        raise RuntimeError("selection batches differ from exact ordered document bank")
    return MergedSelectionBatches(
        ledgers=_seal_ledgers(combined),
        batch_closures=tuple(retained_closures),
        ordered_document_ids=expected_documents,
    )


def merge_synthetic_batches(
    results: Sequence[SyntheticBatchResult],
    expected_item_ids: Sequence[str],
) -> MergedSyntheticBatches:
    """Merge descriptive crossover batches under the same exact call contract."""

    if not results:
        raise ValueError("synthetic selection batch result bank is empty")
    expected_items = tuple(expected_item_ids)
    if not expected_items or len(set(expected_items)) != len(expected_items):
        raise ValueError("expected synthetic item order is malformed")
    expected_plans = {
        candidate: PhysicalCandidateDispatcher.plan(candidate)
        for candidate in FROZEN_CANDIDATES
    }
    combined: dict[str, list[SyntheticPairEffect]] = {
        candidate: [] for candidate in FROZEN_CANDIDATES
    }
    observed_items: list[str] = []
    retained_closures: list[SyntheticBatchClosure] = []
    for result in results:
        if not isinstance(result, SyntheticBatchResult):
            raise ValueError("synthetic selection batch result has wrong type")
        closure = result.closure
        items = closure.item_ids
        candidate_closures = closure.candidate_closures
        if (
            not closure.closed or closure.raw_logits_returned
            or closure.native_unembedding_calls != 1
            or closure.candidate_unembedding_calls != (1,) * len(FROZEN_CANDIDATES)
            or closure.native_attention_calls != (1,) * LAYER_COUNT
            or closure.native_mlp_calls != (1,) * LAYER_COUNT
            or tuple(result.effects) != FROZEN_CANDIDATES
            or len(candidate_closures) != len(FROZEN_CANDIDATES)
            or tuple(item.candidate for item in candidate_closures) != FROZEN_CANDIDATES
            or any(
                item.attempted_batch_calls != 1 or item.batch_calls != 1
                or item.document_calls != 2 * len(items) or not item.closed
                or item.native_mlp_calls != (1,) * LAYER_COUNT
                or item.selected_layer_heads != expected_plans[item.candidate]
                or item.native_attention_calls != tuple(
                    0 if layer in dict(expected_plans[item.candidate]) else 1
                    for layer in range(LAYER_COUNT)
                )
                or item.adapter_attention_calls != tuple(
                    1 if layer in dict(expected_plans[item.candidate]) else 0
                    for layer in range(LAYER_COUNT)
                )
                or not math.isfinite(item.maximum_head_recomposition_abs_error)
                or not math.isfinite(item.maximum_head_recomposition_relative_error)
                or item.maximum_head_recomposition_abs_error < 0
                or item.maximum_head_recomposition_relative_error < 0
                or item.maximum_head_recomposition_relative_error
                    > MAX_HEAD_RECOMPOSITION_RELATIVE_ERROR
                for item in candidate_closures
            )
            or any(
                tuple(effect.item_id for effect in result.effects[candidate]) != items
                or any(
                    not all(math.isfinite(value) for value in (
                        effect.native_did, effect.candidate_did,
                        effect.candidate_minus_native_did,
                    ))
                    or effect.candidate_minus_native_did
                        != effect.candidate_did - effect.native_did
                    for effect in result.effects[candidate]
                )
                for candidate in FROZEN_CANDIDATES
            )
        ):
            raise RuntimeError("synthetic selection closure or call census is malformed")
        reference = result.effects[FROZEN_CANDIDATES[0]]
        for candidate in FROZEN_CANDIDATES:
            for left, right in zip(reference, result.effects[candidate], strict=True):
                if left.item_id != right.item_id or left.native_did != right.native_did:
                    raise RuntimeError("synthetic candidates do not share one native baseline")
            combined[candidate].extend(result.effects[candidate])
        observed_items.extend(items)
        retained_closures.append(closure)
    if tuple(observed_items) != expected_items:
        raise RuntimeError("synthetic batches differ from exact ordered item bank")
    return MergedSyntheticBatches(
        effects=MappingProxyType({
            candidate: tuple(values) for candidate, values in combined.items()
        }),
        batch_closures=tuple(retained_closures),
        ordered_item_ids=expected_items,
    )
