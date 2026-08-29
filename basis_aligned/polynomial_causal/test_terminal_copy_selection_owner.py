from __future__ import annotations

import pytest
import torch

from terminal_copy_attention_dispatcher import NAMED_LAYERS, PhysicalCandidateDispatcher
import terminal_copy_selection_owner as selection_owner
from terminal_copy_selection_owner import SelectionBatchOwner, merge_selection_batches
from terminal_copy_streaming_statistics import CELL_NAMES, FROZEN_CANDIDATES
from test_terminal_copy_attention_owner import TinyModel


def make_case():
    model = TinyModel()
    rows = torch.randint(0, 23, (2, 257))
    tokens = rows[:, :256].clone()
    masks = {name: torch.zeros(2, 256, dtype=torch.bool) for name in CELL_NAMES}
    masks["positive"][:, 64:72] = True
    masks["matched_negative"][:, 72:80] = True
    masks["off_target"][:, 80:] = True
    dispatcher = PhysicalCandidateDispatcher.from_native(
        attentions={layer: model.transformer.h[layer].attn for layer in NAMED_LAYERS},
        per_head_position_means={
            layer: torch.zeros(256, 2 if layer == 8 else 1, 18)
            for layer in NAMED_LAYERS
        },
    )
    return model, rows, tokens, masks, dispatcher


def test_selection_owner_returns_only_sufficient_statistics_and_exact_calls():
    model, rows, tokens, masks, dispatcher = make_case()
    result = SelectionBatchOwner(dispatcher).run(
        model, tokens, rows, masks, ("a", "b"), require_production=False,
    )
    assert tuple(result.ledgers) == FROZEN_CANDIDATES
    assert result.closure.raw_logits_returned is False
    assert result.closure.native_unembedding_calls == 1
    assert result.closure.candidate_unembedding_calls == (1,) * len(FROZEN_CANDIDATES)
    assert result.closure.native_attention_calls == (1,) * 18
    assert result.closure.native_mlp_calls == (1,) * 18
    assert len(result.closure.candidate_closures) == len(FROZEN_CANDIDATES)
    assert all(closure.document_calls == 2 for closure in result.closure.candidate_closures)
    assert all(set(result.ledgers[candidate]) == {"a", "b"} for candidate in FROZEN_CANDIDATES)
    def contains_tensor(value):
        if torch.is_tensor(value):
            return True
        if isinstance(value, dict) or hasattr(value, "items"):
            return any(contains_tensor(key) or contains_tensor(item) for key, item in value.items())
        if isinstance(value, (tuple, list)):
            return any(contains_tensor(item) for item in value)
        if hasattr(value, "__dict__"):
            return contains_tensor(value.__dict__)
        return False

    assert not contains_tensor(result)
    with pytest.raises(TypeError):
        result.ledgers["new"] = {}  # type: ignore[index]


def test_batch_merge_enforces_unique_documents_and_call_census():
    model1, rows1, tokens1, masks1, dispatcher1 = make_case()
    first = SelectionBatchOwner(dispatcher1).run(
        model1, tokens1, rows1, masks1, ("a", "b"), require_production=False,
    )
    model2, rows2, tokens2, masks2, dispatcher2 = make_case()
    second = SelectionBatchOwner(dispatcher2).run(
        model2, tokens2, rows2, masks2, ("c", "d"), require_production=False,
    )
    merged = merge_selection_batches((first, second), ("a", "b", "c", "d"))
    assert len(merged.batch_closures) == 2
    assert all(tuple(merged.ledgers[candidate]) == ("a", "b", "c", "d") for candidate in FROZEN_CANDIDATES)
    with pytest.raises(RuntimeError, match="census"):
        merge_selection_batches((first, first), ("a", "b", "c", "d"))
    with pytest.raises(RuntimeError, match="ordered"):
        merge_selection_batches((first, second), ("a", "b", "d", "c"))


def test_selection_owner_rejects_input_row_mismatch_and_is_poisoned():
    model, rows, tokens, masks, dispatcher = make_case()
    owner = SelectionBatchOwner(dispatcher)
    bad = tokens.clone()
    bad[0, 0] = (bad[0, 0] + 1) % 23
    with pytest.raises(ValueError, match="malformed"):
        owner.run(model, bad, rows, masks, ("a", "b"), require_production=False)
    # Validation failures occur before the active transaction and do not poison it.
    result = owner.run(model, tokens, rows, masks, ("a", "b"), require_production=False)
    assert result.closure.closed
    with pytest.raises(RuntimeError, match="closed"):
        owner.run(model, tokens, rows, masks, ("a", "b"), require_production=False)


def test_selection_owner_poisoned_after_partial_forward(monkeypatch):
    model, rows, tokens, masks, dispatcher = make_case()
    owner = SelectionBatchOwner(dispatcher)
    original = model.transformer.h[0].mlp.forward
    model.transformer.h[0].mlp.forward = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("partial")
    )
    with pytest.raises(RuntimeError, match="partial"):
        owner.run(model, tokens, rows, masks, ("a", "b"), require_production=False)
    model.transformer.h[0].mlp.forward = original
    with pytest.raises(RuntimeError, match="failed"):
        owner.run(model, tokens, rows, masks, ("a", "b"), require_production=False)


def test_selection_owner_poisoned_after_reducer_failure(monkeypatch):
    model, rows, tokens, masks, dispatcher = make_case()
    owner = SelectionBatchOwner(dispatcher)
    monkeypatch.setattr(
        selection_owner, "reduce_document_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("reduce")),
    )
    with pytest.raises(RuntimeError, match="reduce"):
        owner.run(model, tokens, rows, masks, ("a", "b"), require_production=False)
    with pytest.raises(RuntimeError, match="failed"):
        owner.run(model, tokens, rows, masks, ("a", "b"), require_production=False)
