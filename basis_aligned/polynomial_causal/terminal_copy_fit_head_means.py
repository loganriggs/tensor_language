"""Deterministic fit-role head-write mean accumulator for E4 copy v1.

The accumulator is outcome-blind.  It accepts only the six preregistered physical
head writes, requires the frozen document order independently at every layer, and
adds documents one at a time on CPU float64 so changing batch boundaries cannot
change the result.  Finalization revokes the mutable sums.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping, Sequence

import torch

from terminal_copy_attention_adapter import HeadWriteTransaction
from terminal_copy_attention_dispatcher import NAMED_LAYERS
from terminal_copy_induction_v1 import NAMED_SIX_HEAD_FAMILY


def _named_heads_by_layer() -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = {layer: [] for layer in NAMED_LAYERS}
    for name in NAMED_SIX_HEAD_FAMILY:
        layer_text, head_text = name[1:].split("H", 1)
        grouped[int(layer_text)].append(int(head_text))
    return {layer: tuple(sorted(heads)) for layer, heads in grouped.items()}


NAMED_HEADS_BY_LAYER = _named_heads_by_layer()


def _document_digest(document_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for document_id in document_ids:
        encoded = document_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True)
class FitHeadMeanBank:
    per_head_position_means: Mapping[int, torch.Tensor]
    master_per_head_position_means: Mapping[int, torch.Tensor]
    document_count: int
    ordered_document_ids_sha256: str
    runtime_means_sha256: str
    master_means_sha256: str
    accumulator_dtype: str
    published_dtype: str

    def clone_means(self) -> dict[int, torch.Tensor]:
        return {
            layer: self.per_head_position_means[layer].clone()
            for layer in NAMED_LAYERS
        }

    def clone_master_means(self) -> dict[int, torch.Tensor]:
        return {
            layer: self.master_per_head_position_means[layer].clone()
            for layer in NAMED_LAYERS
        }


class FitHeadMeanAccumulator:
    """Accumulate native fit-role head writes in exact document order."""

    def __init__(
        self,
        *,
        ordered_document_ids: Sequence[str],
        sequence_length: int,
        n_head: int,
        width: int,
        published_dtype: torch.dtype = torch.float32,
    ) -> None:
        documents = tuple(ordered_document_ids)
        if (
            not documents
            or any(not isinstance(value, str) or not value for value in documents)
            or len(set(documents)) != len(documents)
        ):
            raise ValueError("fit document IDs must be unique nonempty strings")
        if any(type(value) is not int or value <= 0 for value in (
            sequence_length, n_head, width,
        )):
            raise ValueError("fit mean topology is malformed")
        if published_dtype not in (torch.float32, torch.float64):
            raise ValueError("published mean dtype must be float32 or float64")
        if any(head >= n_head for heads in NAMED_HEADS_BY_LAYER.values() for head in heads):
            raise ValueError("fit mean head topology cannot represent frozen heads")
        self._documents = documents
        self._sequence_length = sequence_length
        self._n_head = n_head
        self._width = width
        self._published_dtype = published_dtype
        self._next = {layer: 0 for layer in NAMED_LAYERS}
        self._sums: dict[int, torch.Tensor] | None = {
            layer: torch.zeros(
                sequence_length, len(NAMED_HEADS_BY_LAYER[layer]), width,
                dtype=torch.float64,
            )
            for layer in NAMED_LAYERS
        }
        self._finalized = False

    def _require_open(self) -> dict[int, torch.Tensor]:
        if self._finalized or self._sums is None:
            raise RuntimeError("fit head-mean accumulator is finalized")
        return self._sums

    def consume(
        self,
        *,
        layer: int,
        document_ids: Sequence[str],
        head_writes: Mapping[int, torch.Tensor],
    ) -> None:
        sums = self._require_open()
        if layer not in NAMED_LAYERS:
            raise ValueError("layer is outside the registered head-mean bank")
        documents = tuple(document_ids)
        if not documents:
            raise ValueError("fit mean batch is empty")
        start = self._next[layer]
        stop = start + len(documents)
        if stop > len(self._documents) or documents != self._documents[start:stop]:
            raise ValueError("fit documents are missing, duplicated, or out of frozen order")
        heads = NAMED_HEADS_BY_LAYER[layer]
        if set(head_writes) != set(heads):
            raise ValueError("fit head-write batch differs from frozen physical heads")
        expected_shape = (
            len(documents), self._sequence_length, self._width,
        )
        values: dict[int, torch.Tensor] = {}
        for head in heads:
            value = head_writes[head]
            if (
                not torch.is_tensor(value)
                or not value.is_floating_point()
                or value.shape != expected_shape
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("fit head-write tensor is malformed")
            values[head] = value.detach().to(device="cpu", dtype=torch.float64)
        # Preserve one addition per frozen document, independent of batch boundaries.
        for batch_index in range(len(documents)):
            for head_index, head in enumerate(heads):
                sums[layer][:, head_index, :].add_(values[head][batch_index])
        self._next[layer] = stop

    def consume_transaction(
        self,
        *,
        layer: int,
        document_ids: Sequence[str],
        transaction: HeadWriteTransaction,
    ) -> None:
        if not isinstance(transaction, HeadWriteTransaction):
            raise ValueError("fit mean collection requires an owned head-write transaction")
        self.consume(
            layer=layer,
            document_ids=document_ids,
            head_writes={
                head: transaction.select((head,))
                for head in NAMED_HEADS_BY_LAYER.get(layer, ())
            },
        )

    def finalize(self) -> FitHeadMeanBank:
        sums = self._require_open()
        incomplete = {
            layer: self._next[layer] for layer in NAMED_LAYERS
            if self._next[layer] != len(self._documents)
        }
        if incomplete:
            raise RuntimeError(f"fit head-mean layers are incomplete: {incomplete}")
        master_means: dict[int, torch.Tensor] = {}
        means: dict[int, torch.Tensor] = {}
        for layer in NAMED_LAYERS:
            master = (sums[layer] / len(self._documents)).contiguous()
            master_means[layer] = master
            means[layer] = master.to(self._published_dtype).contiguous()

        def digest_bank(bank: Mapping[int, torch.Tensor]) -> str:
            digest = hashlib.sha256()
            digest.update(_document_digest(self._documents).encode())
            for layer in NAMED_LAYERS:
                value = bank[layer]
                digest.update(layer.to_bytes(8, "little"))
                digest.update(str(value.dtype).encode())
                digest.update(str(tuple(value.shape)).encode())
                digest.update(value.numpy().tobytes(order="C"))
            return digest.hexdigest()

        bank = FitHeadMeanBank(
            per_head_position_means=means,
            master_per_head_position_means=master_means,
            document_count=len(self._documents),
            ordered_document_ids_sha256=_document_digest(self._documents),
            runtime_means_sha256=digest_bank(means),
            master_means_sha256=digest_bank(master_means),
            accumulator_dtype=str(torch.float64),
            published_dtype=str(self._published_dtype),
        )
        self._sums = None
        self._finalized = True
        return bank
