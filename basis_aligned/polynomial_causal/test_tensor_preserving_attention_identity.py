from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tensor_preserving_attention_identity import (
    AttentionNativePoison, deterministic_tokens, tensor_sha256,
)


class FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(2, 2, bias=False)


class FakeModel(nn.Module):
    def __init__(self, layers: int = 3) -> None:
        super().__init__()
        self.transformer = SimpleNamespace(
            h=nn.ModuleList([FakeBlock() for _ in range(layers)])
        )


def test_attention_poison_forbids_every_native_call_and_restores_dispatch() -> None:
    model = FakeModel()
    modules = tuple(block.attn for block in model.transformer.h)
    before = tuple(module.__dict__.get("forward") for module in modules)
    guard = AttentionNativePoison(model)
    with pytest.raises(RuntimeError, match="attention1"):
        with guard.scope():
            model.transformer.h[1].attn(torch.randn(1, 2))
    assert guard.calls == {0: 0, 1: 1, 2: 0}
    assert guard.restored and guard.inert
    assert tuple(block.attn for block in model.transformer.h) == modules
    assert tuple(module.__dict__.get("forward") for module in modules) == before
    assert all("forward" not in module.__dict__ for module in modules)


def test_attention_poison_is_one_use_and_restores_existing_instance_forward() -> None:
    model = FakeModel(layers=1)
    module = model.transformer.h[0].attn
    prior = module.forward
    module.forward = prior
    guard = AttentionNativePoison(model)
    with guard.scope():
        pass
    assert model.transformer.h[0].attn is module
    assert module.__dict__["forward"] is prior
    with pytest.raises(RuntimeError, match="one-use"):
        with guard.scope():
            pass


def test_role_free_fixture_has_exact_production_contract() -> None:
    tokens = deterministic_tokens(torch.device("cpu"))
    assert tokens.dtype == torch.long and tuple(tokens.shape) == (4, 256)
    assert int(tokens.min()) == 0 and int(tokens.max()) == 50_256
    assert torch.equal(tokens, deterministic_tokens(torch.device("cpu")))


def test_tensor_hash_supports_scalar_and_binds_shape() -> None:
    scalar = torch.tensor(0.5)
    vector = scalar.reshape(1)
    assert tensor_sha256(scalar) == tensor_sha256(torch.tensor(0.5))
    assert tensor_sha256(scalar) != tensor_sha256(vector)
