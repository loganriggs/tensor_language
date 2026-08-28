from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import bilin18_observed_adapter as observed
import bilin18_observed_model_facade as facade
import early_mlp_suffix_transport_v1_runtime as runtime
import early_mlp_suffix_transport_v1_capabilities as capabilities
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


def test_adapter_factory_owns_native_broker_binding() -> None:
    class WideNative(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, state):
            return state * self.scale

    class MinimalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.transformer = SimpleNamespace(h=[
                SimpleNamespace(mlp=WideNative()) for _ in range(3)
            ])

    model = MinimalModel()
    adapter = observed.ObservedBilin18Adapter(model, FakeShip(), production=False)
    hashes = {
        "inherited_snapshot_sha256": "1" * 64,
        "rows_receipt_sha256": "2" * 64,
        "fit_role_tensor_sha256": "3" * 64,
        "identity_teacher_mapping_sha256": "4" * 64,
    }
    context = capabilities.RunContext(source_commit="5" * 40, **hashes)
    bases = {site: torch.eye(runtime.D_MODEL)[:, :runtime.CODE_DIM] for site in (0, 1)}
    broker = adapter.make_capability_broker(
        issuer_id="6" * 64,
        coordinator=runtime.ScopeCoordinator(),
        run_context=context,
        bases=bases,
    )
    assert isinstance(broker, capabilities.CapabilityBroker)
    assert broker.issuer_id == "6" * 64
    guard = observed._EarlyNativePoison(model)
    state = torch.zeros(
        runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.D_MODEL,
    )
    with guard.scope():
        native = broker._CapabilityBroker__native_calls[0](state)
        torch.testing.assert_close(native, state)
        with pytest.raises(RuntimeError, match="literal native MLP0"):
            model.transformer.h[0].mlp(state)
    assert guard.restored and guard.inert

    class Authority(capabilities.MappedRunAuthority):
        @property
        def base_context(self):
            return context

        @property
        def sha256(self):
            return "7" * 64

        def require_source_identity(self, identity, student_inputs, student_indices):
            raise AssertionError("construction must not execute mapping authority")

        def require_identity(self, identity, **kwargs):
            raise AssertionError("construction must not execute mapping authority")

    mapped_broker = adapter.make_mapped_capability_broker(
        issuer_id="8" * 64, coordinator=runtime.ScopeCoordinator(),
        mapped_context=Authority(), bases=bases,
    )
    assert isinstance(mapped_broker, capabilities.CapabilityBroker)
    assert mapped_broker.ledger_snapshot.run_context_sha256 == "7" * 64


def test_adapter_delegates_mapped_teacher_without_releasing_logits() -> None:
    adapter = observed.ObservedBilin18Adapter(tiny_model(), FakeShip(), production=False)

    class Broker:
        def __init__(self):
            self.kwargs = None

        def run_mapped_oon_teacher(self, identity, step, **kwargs):
            self.kwargs = {"identity": identity, "step": step, **kwargs}
            return "sealed-result"

    broker = Broker()
    tensors = {
        "fit_rows": torch.zeros((4, 513), dtype=torch.long),
        "student_tokens": torch.ones((4, 256), dtype=torch.long),
        "teacher_tokens": torch.full((4, 256), 2, dtype=torch.long),
    }
    result = adapter.run_mapped_oon_teacher(
        broker=broker, identity="identity", step="step",
        student_indices=(0, 1, 2, 3), teacher_indices=(3, 2, 1, 0), **tensors,
    )
    assert result == "sealed-result"
    assert broker.kwargs == {
        "identity": "identity", "step": "step",
        "fit_rows": tensors["fit_rows"],
        "student_inputs": tensors["student_tokens"],
        "student_indices": (0, 1, 2, 3),
        "teacher_inputs": tensors["teacher_tokens"],
        "teacher_indices": (3, 2, 1, 0),
        "autonomous_forward": adapter._autonomous_oon_forward,
    }


def test_adapter_delegates_mapped_coordinate_target_trajectory() -> None:
    adapter = observed.ObservedBilin18Adapter(tiny_model(), FakeShip(), production=False)

    class Broker:
        def run_mapped_coordinate_teacher(self, identity, step, **kwargs):
            self.kwargs = {"identity": identity, "step": step, **kwargs}
            return "sealed-coordinate-result"

    broker = Broker()
    program = object()
    fit_rows = torch.zeros((4, 513), dtype=torch.long)
    source = torch.ones((4, 256), dtype=torch.long)
    teacher = torch.full((4, 256), 2, dtype=torch.long)
    result = adapter.run_mapped_coordinate_teacher(
        broker=broker, identity="identity", step="step", fit_rows=fit_rows,
        student_tokens=source, student_indices=(0, 1, 2, 3),
        teacher_tokens=teacher, teacher_indices=(3, 2, 1, 0), program=program,
    )
    assert result == "sealed-coordinate-result"
    assert broker.kwargs == {
        "identity": "identity", "step": "step", "fit_rows": fit_rows,
        "student_inputs": source, "student_indices": (0, 1, 2, 3),
        "teacher_inputs": teacher, "teacher_indices": (3, 2, 1, 0),
        "program": program,
        "autonomous_forward": adapter._autonomous_mapped_coordinate_forward,
    }


def test_mapped_coordinate_adapter_runs_p_p_n_and_poison_closes() -> None:
    model = tiny_model()
    adapter = observed.ObservedBilin18Adapter(model, FakeShip(), production=False)

    class Gateway:
        def __init__(self):
            self.calls = []

        def correct_and_label(self, site, state, deployed):
            self.calls.append((site, tuple(state.shape), tuple(deployed.shape)))
            return deployed

    gateway = Gateway()
    with torch.no_grad():
        closure = adapter._autonomous_mapped_coordinate_forward(
            gateway, torch.zeros((1, 2), dtype=torch.long),
        )
    assert [call[0] for call in gateway.calls] == [0, 1]
    assert closure == {
        "outer_forward_count": 1, "hook_calls": {0: 1, 1: 1, 2: 0},
        "outer_returned": True, "hook_restored": True, "hook_inert": True,
    }

    poisoned = observed.ObservedBilin18Adapter(
        model, FakeShip(call_native_at_zero=True), production=False,
    )
    with pytest.raises(RuntimeError, match="literal native MLP0"):
        with torch.no_grad():
            poisoned._autonomous_mapped_coordinate_forward(
                Gateway(), torch.zeros((1, 2), dtype=torch.long),
            )


def test_adapter_delegates_a_null_parent_and_source_teacher() -> None:
    adapter = observed.ObservedBilin18Adapter(tiny_model(), FakeShip(), production=False)

    class Broker:
        def prepare_mapped_parent(self, identity, **kwargs):
            self.parent_kwargs = {"identity": identity, **kwargs}
            return "parent", "parent-closure"

        def run_a_null_oon_teacher(self, identity, step, **kwargs):
            self.teacher_kwargs = {"identity": identity, "step": step, **kwargs}
            return "sealed-a-null-result"

    broker = Broker()
    program = object()
    fit_rows = torch.zeros((4, 513), dtype=torch.long)
    source = torch.ones((4, 256), dtype=torch.long)
    target = torch.full((4, 256), 2, dtype=torch.long)
    common = {
        "broker": broker, "identity": "identity", "fit_rows": fit_rows,
        "student_tokens": source, "student_indices": (0, 1, 2, 3),
        "teacher_tokens": target, "teacher_indices": (3, 2, 1, 0),
    }
    parent = adapter.prepare_mapped_parent(program=program, **common)
    result = adapter.run_a_null_oon_teacher(step="step", **common)

    assert parent == ("parent", "parent-closure")
    assert result == "sealed-a-null-result"
    assert broker.parent_kwargs == {
        "identity": "identity", "fit_rows": fit_rows,
        "student_inputs": source, "student_indices": (0, 1, 2, 3),
        "teacher_inputs": target, "teacher_indices": (3, 2, 1, 0),
        "program": program,
        "autonomous_forward": adapter._autonomous_mapped_parent_forward,
    }
    assert broker.teacher_kwargs == {
        "identity": "identity", "step": "step", "fit_rows": fit_rows,
        "student_inputs": source, "student_indices": (0, 1, 2, 3),
        "teacher_inputs": target, "teacher_indices": (3, 2, 1, 0),
        "autonomous_forward": adapter._autonomous_oon_forward,
    }


def test_mapped_parent_adapter_runs_native_free_p_p_n_and_poison_closes() -> None:
    model = tiny_model()
    adapter = observed.ObservedBilin18Adapter(model, FakeShip(), production=False)

    class Gateway:
        def __init__(self):
            self.calls = []

        def correct(self, site, state, deployed):
            self.calls.append((site, tuple(state.shape), tuple(deployed.shape)))
            return deployed

    gateway = Gateway()
    with torch.no_grad():
        closure = adapter._autonomous_mapped_parent_forward(
            gateway, torch.zeros((1, 2), dtype=torch.long),
        )
    assert [call[0] for call in gateway.calls] == [0, 1]
    assert closure == {
        "outer_forward_count": 1, "hook_calls": {0: 1, 1: 1, 2: 0},
        "outer_returned": True, "hook_restored": True, "hook_inert": True,
    }

    poisoned = observed.ObservedBilin18Adapter(
        model, FakeShip(call_native_at_zero=True), production=False,
    )
    with pytest.raises(RuntimeError, match="literal native MLP0"):
        with torch.no_grad():
            poisoned._autonomous_mapped_parent_forward(
                Gateway(), torch.zeros((1, 2), dtype=torch.long),
            )
