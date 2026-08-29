from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from terminal_copy_attention_adapter import OwnedPerHeadTensorAttention
from terminal_copy_attention_dispatcher import NAMED_LAYERS, PhysicalCandidateDispatcher
from terminal_copy_fit_head_means import (
    FitHeadMeanAccumulator,
    NAMED_HEADS_BY_LAYER,
)


def writes_for(layer: int, values: torch.Tensor) -> dict[int, torch.Tensor]:
    return {
        head: values + 10 * layer + head
        for head in NAMED_HEADS_BY_LAYER[layer]
    }


def complete(partitions: tuple[int, ...]):
    documents = tuple(f"doc-{index}" for index in range(sum(partitions)))
    accumulator = FitHeadMeanAccumulator(
        ordered_document_ids=documents, sequence_length=3, n_head=9, width=4,
    )
    start = 0
    base = torch.arange(len(documents) * 3 * 4, dtype=torch.float32).reshape(
        len(documents), 3, 4,
    )
    for size in partitions:
        stop = start + size
        for layer in NAMED_LAYERS:
            accumulator.consume(
                layer=layer,
                document_ids=documents[start:stop],
                head_writes=writes_for(layer, base[start:stop]),
            )
        start = stop
    return accumulator.finalize(), base


def test_means_are_exact_and_unnamed_heads_are_zero():
    bank, base = complete((2, 2))
    expected_base = base.double().mean(0).float()
    for layer in NAMED_LAYERS:
        value = bank.per_head_position_means[layer]
        assert value.shape == (3, len(NAMED_HEADS_BY_LAYER[layer]), 4)
        for head_index, head in enumerate(NAMED_HEADS_BY_LAYER[layer]):
            assert torch.equal(value[:, head_index], expected_base + 10 * layer + head)
        assert torch.equal(bank.master_per_head_position_means[layer].float(), value)
    assert bank.document_count == 4
    assert len(bank.ordered_document_ids_sha256) == 64
    assert len(bank.runtime_means_sha256) == 64
    assert len(bank.master_means_sha256) == 64


def test_result_is_bit_identical_across_batch_partitions():
    first, _ = complete((4,))
    second, _ = complete((1, 2, 1))
    assert first.runtime_means_sha256 == second.runtime_means_sha256
    assert first.master_means_sha256 == second.master_means_sha256
    for layer in NAMED_LAYERS:
        assert torch.equal(
            first.per_head_position_means[layer],
            second.per_head_position_means[layer],
        )


def test_document_order_duplicates_and_wrong_heads_fail_closed():
    documents = ("a", "b")
    accumulator = FitHeadMeanAccumulator(
        ordered_document_ids=documents, sequence_length=2, n_head=9, width=3,
    )
    tensor = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="order"):
        accumulator.consume(
            layer=5, document_ids=("b",), head_writes={5: tensor},
        )
    with pytest.raises(ValueError, match="physical heads"):
        accumulator.consume(
            layer=5, document_ids=("a",), head_writes={4: tensor},
        )
    with pytest.raises(ValueError, match="unique"):
        FitHeadMeanAccumulator(
            ordered_document_ids=("a", "a"), sequence_length=2, n_head=9, width=3,
        )


def test_nonfinite_shape_and_missing_layer_fail_closed():
    accumulator = FitHeadMeanAccumulator(
        ordered_document_ids=("a",), sequence_length=2, n_head=9, width=3,
    )
    with pytest.raises(ValueError, match="malformed"):
        accumulator.consume(
            layer=5, document_ids=("a",),
            head_writes={5: torch.full((1, 2, 3), float("nan"))},
        )
    with pytest.raises(ValueError, match="malformed"):
        accumulator.consume(
            layer=5, document_ids=("a",), head_writes={5: torch.zeros(1, 2, 4)},
        )
    with pytest.raises(RuntimeError, match="incomplete"):
        accumulator.finalize()


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(123)
        for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v", "c_proj"):
            setattr(self, name, nn.Linear(18, 18, bias=False))
        self.lamb = nn.Parameter(torch.tensor(0.25))
        self.n_head = 9
        self.rotary = SimpleNamespace(inv_freq=torch.ones(1))


def test_owned_transaction_collection_integrates_with_dispatcher_and_revokes():
    documents = ("a", "b")
    adapters = {
        layer: OwnedPerHeadTensorAttention.from_native(TinyAttention())
        for layer in NAMED_LAYERS
    }
    accumulator = FitHeadMeanAccumulator(
        ordered_document_ids=documents, sequence_length=3, n_head=9, width=18,
    )
    state = torch.randn(2, 3, 18)
    first_value = torch.randn(2, 3, 9, 2)
    for layer in NAMED_LAYERS:
        with adapters[layer].begin(state, first_value) as transaction:
            accumulator.consume_transaction(
                layer=layer, document_ids=documents, transaction=transaction,
            )
        assert transaction.closure.selected_head_sets == tuple(
            (head,) for head in NAMED_HEADS_BY_LAYER[layer]
        )
    bank = accumulator.finalize()
    dispatcher = PhysicalCandidateDispatcher(
        adapters=adapters, per_head_position_means=bank.clone_means(),
    )
    result = dispatcher.dispatch(
        candidate="L5H5", layer=5, state=state, first_value=first_value,
    )
    assert torch.isfinite(result.write).all()
    with pytest.raises(RuntimeError, match="finalized"):
        accumulator.finalize()


def test_bank_clone_does_not_alias_published_values():
    bank, _ = complete((2,))
    clone = bank.clone_means()
    clone[5].zero_()
    assert not torch.equal(clone[5], bank.per_head_position_means[5])
    master = bank.clone_master_means()
    master[5].zero_()
    assert not torch.equal(master[5], bank.master_per_head_position_means[5])
    exposed = bank.per_head_position_means
    exposed[5].zero_()
    assert bank.verify_hashes()
    assert not torch.equal(exposed[5], bank.per_head_position_means[5])


def test_production_accumulator_requires_exact_population_shape_and_dtypes():
    with pytest.raises(ValueError, match="production"):
        FitHeadMeanAccumulator(
            ordered_document_ids=("a",), sequence_length=256, n_head=9, width=1152,
            source_dtype=torch.bfloat16, require_production=True,
        )
    documents = tuple(f"doc-{index}" for index in range(192))
    accumulator = FitHeadMeanAccumulator(
        ordered_document_ids=documents, sequence_length=256, n_head=9, width=1152,
        source_dtype=torch.bfloat16, published_dtype=torch.float32,
        require_production=True,
    )
    assert accumulator.production_contract
