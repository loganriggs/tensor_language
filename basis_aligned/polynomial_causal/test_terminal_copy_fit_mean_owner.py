from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from terminal_copy_attention_dispatcher import NAMED_LAYERS, PhysicalCandidateDispatcher
from terminal_copy_attention_owner import LAYER_COUNT
from terminal_copy_fit_head_means import FitHeadMeanAccumulator
from terminal_copy_fit_mean_owner import FitMeanCollectionOwner
from test_terminal_copy_attention_owner import TinyModel


def native_final_state(model: TinyModel, tokens: torch.Tensor) -> torch.Tensor:
    x = F.rms_norm(model.transformer.wte(tokens), (18,))
    x0, first_value = x, None
    for block in model.transformer.h:
        x = block.lambdas[0] * x + block.lambdas[1] * x0
        attention, first_value = block.attn(F.rms_norm(x, (18,)), first_value)
        x = x + attention
        x = x + block.mlp(F.rms_norm(x, (18,)))
    return x


def make_owner(model: TinyModel, documents: tuple[str, ...], sequence: int):
    dispatcher = PhysicalCandidateDispatcher.from_native(
        attentions={layer: model.transformer.h[layer].attn for layer in NAMED_LAYERS},
        per_head_position_means={
            layer: torch.zeros(sequence, 2 if layer == 8 else 1, 18)
            for layer in NAMED_LAYERS
        },
    )
    accumulator = FitHeadMeanAccumulator(
        ordered_document_ids=documents,
        sequence_length=sequence,
        n_head=9,
        width=18,
    )
    return FitMeanCollectionOwner(dispatcher=dispatcher, accumulator=accumulator)


def test_collection_is_observational_and_never_calls_unembedding():
    model = TinyModel()
    reference = copy.deepcopy(model)
    tokens = torch.randint(0, 23, (2, 6))
    expected = native_final_state(reference, tokens)
    model.lm_head.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("fit collection called unembedding")
    )
    owner = make_owner(model, ("a", "b"), 6)
    observed = owner.collect_batch(
        model, tokens, ("a", "b"), require_production=False,
    )
    assert torch.equal(observed, expected)
    bank, closure = owner.finalize()
    assert closure.native_unembedding_calls == 0
    assert closure.native_attention_calls == (1,) * LAYER_COUNT
    assert closure.native_mlp_calls == (1,) * LAYER_COUNT
    assert closure.adapter_decomposition_calls == tuple(
        1 if layer in NAMED_LAYERS else 0 for layer in range(LAYER_COUNT)
    )
    assert closure.maximum_full_write_abs_error == 0
    assert bank.document_count == 2


def test_collection_is_deterministic_across_batches_and_tracks_hashes():
    model1, model2 = TinyModel(), TinyModel()
    model2.load_state_dict(model1.state_dict())
    documents = ("a", "b", "c")
    tokens = torch.randint(0, 23, (3, 5))
    one = make_owner(model1, documents, 5)
    one.collect_batch(model1, tokens, documents, require_production=False)
    bank1, closure1 = one.finalize()
    two = make_owner(model2, documents, 5)
    two.collect_batch(model2, tokens[:1], documents[:1], require_production=False)
    two.collect_batch(model2, tokens[1:], documents[1:], require_production=False)
    bank2, closure2 = two.finalize()
    assert bank1.master_means_sha256 == bank2.master_means_sha256
    assert bank1.runtime_means_sha256 == bank2.runtime_means_sha256
    assert closure1.document_calls == closure2.document_calls == 3
    assert closure1.batch_calls == 1 and closure2.batch_calls == 2
    assert len(closure1.final_state_sha256s) == 1
    assert len(closure2.final_state_sha256s) == 2


def test_collection_rejects_wrong_order_and_incomplete_finalize():
    model = TinyModel()
    owner = make_owner(model, ("a", "b"), 4)
    with pytest.raises(ValueError, match="order"):
        owner.collect_batch(
            model, torch.randint(0, 23, (1, 4)), ("b",), require_production=False,
        )
    with pytest.raises(RuntimeError, match="incomplete"):
        owner.finalize()


def test_production_collection_rejects_short_synthetic_rows():
    model = TinyModel()
    owner = make_owner(model, ("a",), 4)
    with pytest.raises(ValueError, match="256"):
        owner.collect_batch(
            model, torch.randint(0, 23, (1, 4)), ("a",), require_production=True,
        )
