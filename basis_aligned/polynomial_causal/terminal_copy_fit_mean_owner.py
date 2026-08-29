"""Outcome-blind native forward owner for collecting E4 fit head means.

The owner advances the real residual stream with original native attention and MLP
writes.  At only the five registered layers it independently decomposes the exact
same pre-attention state/value bus and sends the six named physical writes to the
deterministic accumulator.  It never calls the unembedding or computes logits/losses.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from typing import Sequence

import torch
import torch.nn.functional as F

from terminal_copy_attention_dispatcher import NAMED_LAYERS, PhysicalCandidateDispatcher
from terminal_copy_fit_head_means import FitHeadMeanAccumulator, FitHeadMeanBank


LAYER_COUNT = 18


def tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().to("cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode())
    digest.update(str(tuple(cpu.shape)).encode())
    # NumPy has no bfloat16 scalar type.  Hash the tensor's exact contiguous byte
    # representation through a uint8 view while retaining dtype and shape above.
    digest.update(cpu.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class FitMeanOwnerClosure:
    batch_calls: int
    document_calls: int
    native_attention_calls: tuple[int, ...]
    adapter_decomposition_calls: tuple[int, ...]
    native_mlp_calls: tuple[int, ...]
    native_unembedding_calls: int
    maximum_full_write_abs_error: float
    maximum_head_recomposition_abs_error: float
    maximum_head_recomposition_relative_error: float
    final_state_sha256s: tuple[str, ...]
    closed: bool


class FitMeanCollectionOwner:
    """Collect native physical means under an exact ordered-document transaction."""

    def __init__(
        self,
        *,
        dispatcher: PhysicalCandidateDispatcher,
        accumulator: FitHeadMeanAccumulator,
    ) -> None:
        if not isinstance(dispatcher, PhysicalCandidateDispatcher) or not isinstance(
            accumulator, FitHeadMeanAccumulator
        ):
            raise ValueError("fit mean owner requires dispatcher and accumulator")
        self._dispatcher: PhysicalCandidateDispatcher | None = copy.deepcopy(dispatcher)
        self._accumulator: FitHeadMeanAccumulator | None = copy.deepcopy(accumulator)
        self._native_attention = [0] * LAYER_COUNT
        self._adapter = [0] * LAYER_COUNT
        self._native_mlp = [0] * LAYER_COUNT
        self._batch_calls = 0
        self._document_calls = 0
        self._max_full_abs = 0.0
        self._max_head_abs = 0.0
        self._max_head_relative = 0.0
        self._final_state_hashes: list[str] = []
        self._active = False
        self._closed = False
        self._failed = False
        self._closure: FitMeanOwnerClosure | None = None

    def _require_open(
        self,
    ) -> tuple[PhysicalCandidateDispatcher, FitHeadMeanAccumulator]:
        if self._failed:
            raise RuntimeError("fit mean collection owner is failed and cannot be reused")
        if self._closed or self._dispatcher is None or self._accumulator is None:
            raise RuntimeError("fit mean collection owner is closed")
        return self._dispatcher, self._accumulator

    @torch.no_grad()
    def collect_batch(
        self,
        model: torch.nn.Module,
        tokens: torch.Tensor,
        document_ids: Sequence[str],
        *,
        require_production: bool = True,
    ) -> str:
        """Advance one native batch and return only its final-state hash, never logits."""

        dispatcher, accumulator = self._require_open()
        if self._active:
            raise RuntimeError("fit mean collection owner is not reentrant")
        documents = tuple(document_ids)
        if (
            not torch.is_tensor(tokens)
            or tokens.dtype != torch.long
            or tokens.ndim != 2
            or tokens.shape[0] != len(documents)
            or not documents
        ):
            raise ValueError("fit mean tokens and document IDs are malformed")
        try:
            blocks = model.transformer.h
            embedding = model.transformer.wte
        except AttributeError as error:
            raise ValueError("fit mean model schema changed") from error
        if len(blocks) != LAYER_COUNT:
            raise ValueError("fit mean owner requires exactly 18 transformer blocks")
        if require_production and tuple(tokens.shape[1:]) != (256,):
            raise ValueError("production fit mean rows must have 256 input tokens")
        if require_production and (
            not accumulator.production_contract
            or accumulator.source_dtype != torch.bfloat16
            or accumulator.published_dtype != torch.float32
        ):
            raise ValueError("production fit mean numeric dtypes are not frozen")
        dispatcher.assert_matches_native({
            layer: blocks[layer].attn for layer in NAMED_LAYERS
        })
        self._active = True
        success = False
        try:
            x = F.rms_norm(embedding(tokens), (dispatcher.width,))
            x0 = x
            first_value = None
            for site, block in enumerate(blocks):
                x = block.lambdas[0] * x + block.lambdas[1] * x0
                state = F.rms_norm(x, (dispatcher.width,))
                native_write, next_value = block.attn(state, first_value)
                self._native_attention[site] += 1
                if site in NAMED_LAYERS:
                    if require_production and first_value is None:
                        raise RuntimeError("registered late attention lost layer-0 value bus")
                    adapter = dispatcher.adapters[str(site)]
                    with adapter.begin(state, first_value) as transaction:
                        full = transaction.native_full_write()
                        bus = transaction.first_value_bus()
                        accumulator.consume_transaction(
                            layer=site,
                            document_ids=documents,
                            transaction=transaction,
                        )
                    self._adapter[site] += 1
                    full_error = float((full.float() - native_write.float()).abs().max())
                    self._max_full_abs = max(self._max_full_abs, full_error)
                    self._max_head_abs = max(
                        self._max_head_abs,
                        transaction.closure.all_head_recomposition_max_abs_error,
                    )
                    self._max_head_relative = max(
                        self._max_head_relative,
                        transaction.closure.all_head_recomposition_relative_error,
                    )
                    if not torch.equal(full, native_write) or not torch.equal(bus, next_value):
                        raise RuntimeError("owned adapter does not exactly replay native write/bus")
                first_value = next_value
                x = x + native_write
                x = x + block.mlp(F.rms_norm(x, (dispatcher.width,)))
                self._native_mlp[site] += 1
            final_sha256 = tensor_sha256(x)
            self._final_state_hashes.append(final_sha256)
            self._batch_calls += 1
            self._document_calls += len(documents)
            success = True
            return final_sha256
        finally:
            self._active = False
            if not success:
                self._failed = True

    def finalize(self) -> tuple[FitHeadMeanBank, FitMeanOwnerClosure]:
        dispatcher, accumulator = self._require_open()
        del dispatcher
        if self._active:
            raise RuntimeError("cannot finalize an active fit mean collection")
        bank = accumulator.finalize()
        closure = FitMeanOwnerClosure(
            batch_calls=self._batch_calls,
            document_calls=self._document_calls,
            native_attention_calls=tuple(self._native_attention),
            adapter_decomposition_calls=tuple(self._adapter),
            native_mlp_calls=tuple(self._native_mlp),
            native_unembedding_calls=0,
            maximum_full_write_abs_error=self._max_full_abs,
            maximum_head_recomposition_abs_error=self._max_head_abs,
            maximum_head_recomposition_relative_error=self._max_head_relative,
            final_state_sha256s=tuple(self._final_state_hashes),
            closed=True,
        )
        self._dispatcher = None
        self._accumulator = None
        self._closure = closure
        self._closed = True
        return bank, closure

    @property
    def closure(self) -> FitMeanOwnerClosure:
        if not self._closed or self._closure is None:
            raise RuntimeError("fit mean collection owner is not closed")
        return self._closure
