from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import bilin18_observed_adapter as observed
import bilin18_observed_model_facade as facade
import early_mlp_suffix_transport_v1_runtime as runtime
from test_bilin18_observed_model_facade import tiny_model


class FakeShip:
    production = False

    def __init__(self, *, call_native_at_zero: bool = False) -> None:
        self.call_native_at_zero = call_native_at_zero

    def attention(self, event):
        return event.block.attn(event.state, event.first_value)

    def mlp(self, event):
        if self.call_native_at_zero and event.site == 0:
            return event.block.mlp(event.state)
        return torch.zeros_like(event.state) + float(event.site + 1) / 100


class FakeHook:
    issuer_id = "a" * 64

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, site, state, handle, *, forward_nonce):
        self.calls.append((site, forward_nonce))
        return handle._consume(
            site=site, state=state, forward_nonce=forward_nonce,
            issuer_id=self.issuer_id,
        )


class FakeCapability:
    def __init__(self) -> None:
        self.bound = None

    def bind_outer_logits(self, logits):
        self.bound = logits


class FakeSession:
    def __init__(self) -> None:
        self.capability = FakeCapability()
        self.close_kwargs = None

    @contextmanager
    def forward_scope(self):
        yield self.capability

    def close(self, **kwargs):
        self.close_kwargs = kwargs
        return "step", "step-closure"


def test_student_adapter_closes_dispatch_and_literal_call_ledgers(monkeypatch) -> None:
    class SyntheticNWrite:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def _consume(self, *, site, state, forward_nonce, issuer_id):
            assert site == self.kwargs["site"]
            assert state is self.kwargs["state"]
            assert forward_nonce == self.kwargs["forward_nonce"]
            assert issuer_id == self.kwargs["issuer_id"]
            return self.kwargs["value"]

    monkeypatch.setattr(
        observed.runtime, "mint_deployed_n_write", lambda **kwargs: SyntheticNWrite(**kwargs),
    )
    model = tiny_model()
    adapter = observed.ObservedBilin18Adapter(model, FakeShip(), production=False)
    hook = FakeHook()
    session = FakeSession()
    tokens = torch.zeros((1, 2), dtype=torch.long)
    identity = SimpleNamespace(nonce="b" * 64)
    had_instance_forward = {
        site: "forward" in model.transformer.h[site].mlp.__dict__
        for site in observed.EARLY_SITES
    }

    step, closure, receipt = adapter.run_student(
        session=session, hook=hook, identity=identity, tokens=tokens,
    )

    assert (step, closure) == ("step", "step-closure")
    assert hook.calls == [(0, identity.nonce), (1, identity.nonce)]
    assert receipt.deployed_n_calls == ((0, 1), (1, 1), (2, 1))
    assert receipt.correction_calls == ((0, 1), (1, 1), (2, 0))
    assert receipt.literal_early_mlp_calls == ((0, 0), (1, 0), (2, 0))
    assert receipt.outer_forward_count == 1 and receipt.outer_returned
    assert session.capability.bound is not None
    assert session.close_kwargs == {
        "outer_forward_count": 1,
        "outer_returned": True,
        "hook_restored": True,
        "hook_inert": True,
    }
    assert all(
        ("forward" in model.transformer.h[site].mlp.__dict__)
        == had_instance_forward[site]
        for site in observed.EARLY_SITES
    )


def test_literal_native_call_fails_before_execution_and_restores_forward() -> None:
    model = tiny_model()
    adapter = observed.ObservedBilin18Adapter(
        model, FakeShip(call_native_at_zero=True), production=False,
    )
    before = {
        site: ("forward" in model.transformer.h[site].mlp.__dict__,
               model.transformer.h[site].mlp.__dict__.get("forward"))
        for site in observed.EARLY_SITES
    }

    with pytest.raises(RuntimeError, match="literal native MLP0 call is forbidden"):
        adapter.run_student(
            session=FakeSession(), hook=FakeHook(),
            identity=SimpleNamespace(nonce="c" * 64),
            tokens=torch.zeros((1, 2), dtype=torch.long),
        )

    after = {
        site: ("forward" in model.transformer.h[site].mlp.__dict__,
               model.transformer.h[site].mlp.__dict__.get("forward"))
        for site in observed.EARLY_SITES
    }
    assert after == before


def test_oon_forward_uses_gateway_for_zero_and_one_only() -> None:
    model = tiny_model()
    adapter = observed.ObservedBilin18Adapter(model, FakeShip(), production=False)

    class Gateway:
        def __init__(self):
            self.calls = []

        def call(self, site, state):
            self.calls.append(site)
            return model.transformer.h[site].mlp(state)

    gateway = Gateway()
    logits, closure = adapter._autonomous_oon_forward(
        gateway, torch.zeros((1, 2), dtype=torch.long),
    )
    assert logits.shape == (1, 2, facade.LOGIT_VOCAB)
    assert gateway.calls == [0, 1]
    assert closure["hook_calls"] == {0: 1, 1: 1, 2: 0}
