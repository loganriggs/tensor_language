"""Deterministic fit-role head-write mean accumulator for E4 copy v1.

The accumulator is outcome-blind.  It accepts only the six preregistered physical
head writes, requires the frozen document order independently at every layer, and
adds documents one at a time on CPU float64 so changing batch boundaries cannot
change the result.  Finalization revokes the mutable sums.
"""

from __future__ import annotations

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


def _mean_bank_digest(
    ordered_document_ids_sha256: str, bank: Mapping[int, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    digest.update(ordered_document_ids_sha256.encode())
    for layer in NAMED_LAYERS:
        value = bank[layer]
        digest.update(layer.to_bytes(8, "little"))
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


class FitHeadMeanBank:
    """Read-only-by-copy finalized mean artifact."""

    def __init__(
        self, *, per_head_position_means: Mapping[int, torch.Tensor],
        master_per_head_position_means: Mapping[int, torch.Tensor],
        document_count: int, ordered_document_ids_sha256: str,
        runtime_means_sha256: str, master_means_sha256: str,
        accumulator_dtype: str, published_dtype: str, source_dtype: str,
    ) -> None:
        self._runtime = {
            layer: per_head_position_means[layer].detach().clone()
            for layer in NAMED_LAYERS
        }
        self._master = {
            layer: master_per_head_position_means[layer].detach().clone()
            for layer in NAMED_LAYERS
        }
        self._document_count = document_count
        self._ordered_document_ids_sha256 = ordered_document_ids_sha256
        self._runtime_means_sha256 = runtime_means_sha256
        self._master_means_sha256 = master_means_sha256
        self._accumulator_dtype = accumulator_dtype
        self._published_dtype = published_dtype
        self._source_dtype = source_dtype

    @property
    def per_head_position_means(self) -> Mapping[int, torch.Tensor]:
        return self.clone_means()

    @property
    def master_per_head_position_means(self) -> Mapping[int, torch.Tensor]:
        return self.clone_master_means()

    @property
    def document_count(self) -> int:
        return self._document_count

    @property
    def ordered_document_ids_sha256(self) -> str:
        return self._ordered_document_ids_sha256

    @property
    def runtime_means_sha256(self) -> str:
        return self._runtime_means_sha256

    @property
    def master_means_sha256(self) -> str:
        return self._master_means_sha256

    @property
    def accumulator_dtype(self) -> str:
        return self._accumulator_dtype

    @property
    def published_dtype(self) -> str:
        return self._published_dtype

    @property
    def source_dtype(self) -> str:
        return self._source_dtype

    def clone_means(self) -> dict[int, torch.Tensor]:
        return {
            layer: self._runtime[layer].clone()
            for layer in NAMED_LAYERS
        }

    def clone_master_means(self) -> dict[int, torch.Tensor]:
        return {
            layer: self._master[layer].clone()
            for layer in NAMED_LAYERS
        }

    def verify_hashes(self) -> bool:
        return (
            _mean_bank_digest(self._ordered_document_ids_sha256, self._runtime)
            == self._runtime_means_sha256
            and _mean_bank_digest(self._ordered_document_ids_sha256, self._master)
            == self._master_means_sha256
        )


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
        source_dtype: torch.dtype | None = None,
        require_production: bool = False,
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
        if source_dtype is not None and source_dtype not in (
            torch.bfloat16, torch.float16, torch.float32, torch.float64,
        ):
            raise ValueError("source mean dtype must be floating or unspecified")
        if any(head >= n_head for heads in NAMED_HEADS_BY_LAYER.values() for head in heads):
            raise ValueError("fit mean head topology cannot represent frozen heads")
        if require_production and (
            len(documents) != 192
            or sequence_length != 256
            or n_head != 9
            or width != 1152
            or source_dtype != torch.bfloat16
            or published_dtype != torch.float32
        ):
            raise ValueError("production fit mean contract is not exact")
        self._documents = documents
        self._sequence_length = sequence_length
        self._n_head = n_head
        self._width = width
        self._published_dtype = published_dtype
        self._source_dtype = source_dtype
        self._production_contract = require_production
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
                or (self._source_dtype is not None and value.dtype != self._source_dtype)
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

        documents_sha256 = _document_digest(self._documents)
        bank = FitHeadMeanBank(
            per_head_position_means=means,
            master_per_head_position_means=master_means,
            document_count=len(self._documents),
            ordered_document_ids_sha256=documents_sha256,
            runtime_means_sha256=_mean_bank_digest(documents_sha256, means),
            master_means_sha256=_mean_bank_digest(documents_sha256, master_means),
            accumulator_dtype=str(torch.float64),
            published_dtype=str(self._published_dtype),
            source_dtype=str(self._source_dtype),
        )
        self._sums = None
        self._finalized = True
        return bank

    @property
    def source_dtype(self) -> torch.dtype | None:
        return self._source_dtype

    @property
    def published_dtype(self) -> torch.dtype:
        return self._published_dtype

    @property
    def production_contract(self) -> bool:
        return self._production_contract
