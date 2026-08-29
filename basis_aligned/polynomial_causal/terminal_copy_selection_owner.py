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
