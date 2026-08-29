"""Hook-free 18-site owner for one frozen E4 copy candidate.

The owner delegates only the candidate's registered attention layers to the physical
dispatcher.  Every other attention and every MLP is called natively through the
source-closed bilin18 facade.  It records calls and adapter integrity and revokes its
dispatcher reference on close.  It does not load rows, a model, or outcomes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

import bilin18_observed_model_facade as facade
from terminal_copy_attention_dispatcher import NAMED_LAYERS, PhysicalCandidateDispatcher


LAYER_COUNT = 18


@dataclass(frozen=True)
class CandidateOwnerClosure:
    candidate: str
    attempted_batch_calls: int
    batch_calls: int
    document_calls: int
    native_attention_calls: tuple[int, ...]
    adapter_attention_calls: tuple[int, ...]
    native_mlp_calls: tuple[int, ...]
    selected_layer_heads: tuple[tuple[int, tuple[int, ...]], ...]
    maximum_head_recomposition_abs_error: float
    maximum_head_recomposition_relative_error: float
    closed: bool


class CandidateForwardOwner:
    """Run one candidate on live counterfactual states without forward hooks."""

    def __init__(
        self, *, candidate: str, dispatcher: PhysicalCandidateDispatcher,
    ) -> None:
        if not isinstance(dispatcher, PhysicalCandidateDispatcher):
            raise ValueError("candidate owner requires the physical dispatcher")
        self._plan = dispatcher.plan(candidate)
        self._selected = dict(self._plan)
        self._candidate = candidate
        self._dispatcher: PhysicalCandidateDispatcher | None = copy.deepcopy(dispatcher)
        self._native_attention = [0] * LAYER_COUNT
        self._adapter_attention = [0] * LAYER_COUNT
        self._native_mlp = [0] * LAYER_COUNT
        self._batch_calls = 0
        self._attempted_batch_calls = 0
        self._document_calls = 0
        self._max_abs = 0.0
        self._max_relative = 0.0
        self._active = False
        self._closed = False
        self._failed = False
        self._closure: CandidateOwnerClosure | None = None

    def _require_open(self) -> PhysicalCandidateDispatcher:
        if self._failed:
            raise RuntimeError("candidate forward owner is failed and cannot be reused")
        if self._closed or self._dispatcher is None:
            raise RuntimeError("candidate forward owner is closed")
        return self._dispatcher

    @torch.no_grad()
    def run(
        self,
        model: torch.nn.Module,
        tokens: torch.Tensor,
        *,
        require_production: bool = True,
    ) -> torch.Tensor:
        dispatcher = self._require_open()
        if self._active:
            raise RuntimeError("candidate forward owner is not reentrant")
        try:
            blocks = model.transformer.h
        except AttributeError as error:
            raise ValueError("candidate model does not expose transformer blocks") from error
        if len(blocks) != LAYER_COUNT:
            raise ValueError("candidate owner requires exactly 18 transformer blocks")
        dispatcher.assert_matches_native({
            layer: blocks[layer].attn for layer in NAMED_LAYERS
        })
        self._active = True
        self._attempted_batch_calls += 1

        def attention(event: facade.AttentionEvent):
            if event.site in self._selected:
                result = dispatcher.dispatch(
                    candidate=self._candidate,
                    layer=event.site,
                    state=event.state,
                    first_value=event.first_value,
                    require_production=require_production,
                )
                self._adapter_attention[event.site] += 1
                self._max_abs = max(
                    self._max_abs,
                    result.closure.all_head_recomposition_max_abs_error,
                )
                self._max_relative = max(
                    self._max_relative,
                    result.closure.all_head_recomposition_relative_error,
                )
                return result.write, result.first_value_bus
            self._native_attention[event.site] += 1
            return event.block.attn(event.state, event.first_value)

        def mlp(event: facade.EarlyMLPEvent):
            self._native_mlp[event.site] += 1
            return event.block.mlp(event.state)

        success = False
        try:
            logits = facade.forward_with_dispatch(
                model, tokens, attention, mlp,
                require_production=require_production,
            )
            success = True
        finally:
            self._active = False
            if not success:
                self._failed = True
        self._batch_calls += 1
        self._document_calls += int(tokens.shape[0])
        return logits

    def close(self) -> CandidateOwnerClosure:
        self._require_open()
        if self._active:
            raise RuntimeError("cannot close an active candidate forward")
        self._closure = CandidateOwnerClosure(
            candidate=self._candidate,
            attempted_batch_calls=self._attempted_batch_calls,
            batch_calls=self._batch_calls,
            document_calls=self._document_calls,
            native_attention_calls=tuple(self._native_attention),
            adapter_attention_calls=tuple(self._adapter_attention),
            native_mlp_calls=tuple(self._native_mlp),
            selected_layer_heads=self._plan,
            maximum_head_recomposition_abs_error=self._max_abs,
            maximum_head_recomposition_relative_error=self._max_relative,
            closed=True,
        )
        self._dispatcher = None
        self._closed = True
        return self._closure

    @property
    def closure(self) -> CandidateOwnerClosure:
        if not self._closed or self._closure is None:
            raise RuntimeError("candidate forward owner is not closed")
        return self._closure
