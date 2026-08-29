from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
from terminal_copy_attention_dispatcher import NAMED_LAYERS, PhysicalCandidateDispatcher
from terminal_copy_attention_owner import CandidateForwardOwner


class TinyAttention(nn.Module):
    def __init__(self, width: int = 18, heads: int = 9) -> None:
        super().__init__()
        for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v", "c_proj"):
            layer = nn.Linear(width, width, bias=False)
            nn.init.zeros_(layer.weight)
            setattr(self, name, layer)
        self.lamb = nn.Parameter(torch.tensor(0.25))
        self.n_head = heads
        self.rotary = SimpleNamespace(inv_freq=torch.ones(width // heads // 2))

    def forward(self, state, first_value=None):
        batch, sequence, width = state.shape
        value = self.c_v(state).view(batch, sequence, self.n_head, width // self.n_head)
        bus = value if first_value is None else first_value
        return torch.zeros_like(state), bus


class TinyBlock(nn.Module):
    def __init__(self, width: int, layer: int) -> None:
        super().__init__()
        self.lambdas = nn.Parameter(torch.tensor([0.9, 0.1]))
        self.attn = TinyAttention(width)
        self.mlp = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.mlp.weight)
        self.mlp.weight.data.mul_(0.002 * (layer + 1))


class TinyModel(nn.Module):
    def __init__(self, width: int = 18, vocab: int = 23) -> None:
        super().__init__()
        torch.manual_seed(77)
        self.transformer = SimpleNamespace(
            wte=nn.Embedding(vocab, width),
            h=nn.ModuleList([TinyBlock(width, layer) for layer in range(18)]),
        )
        # Register modules that SimpleNamespace itself would otherwise hide.
        self.wte = self.transformer.wte
        self.blocks = self.transformer.h
        self.lm_head = nn.Linear(width, vocab, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab)


def native_logits(model: TinyModel, tokens: torch.Tensor) -> torch.Tensor:
    def attention(event):
        return event.block.attn(event.state, event.first_value)

    def mlp(event):
        return event.block.mlp(event.state)

    return facade.forward_with_dispatch(
        model, tokens, attention, mlp, require_production=False,
    )


def make_owner(model: TinyModel, candidate="registered_four_head_set"):
    means = {
        layer: torch.zeros(8, 2 if layer == 8 else 1, 18)
        for layer in NAMED_LAYERS
    }
    dispatcher = PhysicalCandidateDispatcher.from_native(
        attentions={layer: model.transformer.h[layer].attn for layer in NAMED_LAYERS},
        per_head_position_means=means,
    )
    return CandidateForwardOwner(candidate=candidate, dispatcher=dispatcher)


def test_owner_replays_native_when_selected_writes_and_means_are_zero():
    model = TinyModel()
    reference = copy.deepcopy(model)
    tokens = torch.randint(0, 23, (3, 8))
    expected = native_logits(reference, tokens)
    owner = make_owner(model)
    # Selected-layer native attention forwards must never be called.
    for layer in (5, 7, 8):
        model.transformer.h[layer].attn.forward = lambda *_args, **_kwargs: (
            _ for _ in ()
        ).throw(AssertionError("selected native attention called"))
    observed = owner.run(model, tokens, require_production=False)
    assert torch.equal(observed, expected)
    closure = owner.close()
    assert closure.batch_calls == 1 and closure.document_calls == 3
    assert closure.selected_layer_heads == ((5, (5,)), (7, (3,)), (8, (3, 4)))
    assert closure.native_attention_calls == tuple(
        0 if layer in (5, 7, 8) else 1 for layer in range(18)
    )
    assert closure.adapter_attention_calls == tuple(
        1 if layer in (5, 7, 8) else 0 for layer in range(18)
    )
    assert closure.native_mlp_calls == (1,) * 18


def test_owner_accumulates_multiple_batches_and_revokes_on_close():
    model = TinyModel()
    owner = make_owner(model, candidate="L14H7")
    owner.run(model, torch.randint(0, 23, (2, 4)), require_production=False)
    owner.run(model, torch.randint(0, 23, (1, 4)), require_production=False)
    closure = owner.close()
    assert closure.batch_calls == 2 and closure.document_calls == 3
    assert closure.adapter_attention_calls[14] == 2
    assert sum(closure.adapter_attention_calls) == 2
    assert sum(closure.native_attention_calls) == 34
    assert sum(closure.native_mlp_calls) == 36
    assert owner.closure == closure
    with pytest.raises(RuntimeError, match="closed"):
        owner.run(model, torch.randint(0, 23, (1, 4)), require_production=False)
    with pytest.raises(RuntimeError, match="closed"):
        owner.close()


def test_owner_does_not_install_hooks_and_reports_adapter_integrity():
    model = TinyModel()
    owner = make_owner(model, candidate="registered_late_pair")
    before = tuple(
        (len(block.attn._forward_hooks), len(block.mlp._forward_hooks))
        for block in model.transformer.h
    )
    owner.run(model, torch.randint(0, 23, (2, 5)), require_production=False)
    after = tuple(
        (len(block.attn._forward_hooks), len(block.mlp._forward_hooks))
        for block in model.transformer.h
    )
    closure = owner.close()
    assert before == after == ((0, 0),) * 18
    assert closure.maximum_head_recomposition_abs_error == 0
    assert closure.maximum_head_recomposition_relative_error == 0


def test_owner_rejects_wrong_model_depth_and_unknown_candidate():
    model = TinyModel()
    with pytest.raises(ValueError, match="outside"):
        make_owner(model, candidate="unknown")
    owner = make_owner(model)
    model.transformer.h = model.transformer.h[:17]
    with pytest.raises(ValueError, match="18"):
        owner.run(model, torch.randint(0, 23, (1, 4)), require_production=False)
    owner.close()
