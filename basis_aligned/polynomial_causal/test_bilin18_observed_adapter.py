from __future__ import annotations

from contextlib import contextmanager
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

import bilin18_observed_adapter as observed
import bilin18_observed_model_facade as facade
import early_mlp_suffix_transport_v1_runtime as runtime
import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_final_actions as final_actions
import early_mlp_suffix_transport_v1_programs as programs
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


def test_final_exact_mlp2_background_is_the_only_allowed_literal_early_call(
    monkeypatch,
) -> None:
    class SyntheticNWrite:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def _consume(self, *, site, state, forward_nonce, issuer_id):
            return self.kwargs["value"]

    monkeypatch.setattr(
        observed.runtime, "mint_deployed_n_write", lambda **kwargs: SyntheticNWrite(**kwargs),
    )
    model = tiny_model()
    adapter = observed.ObservedBilin18Adapter(model, FakeShip(), production=False)
    identity = SimpleNamespace(
        nonce="d" * 64, phase="final",
        student_states=((0, "P"), (1, "P"), (2, "E")),
    )
    _, _, receipt = adapter.run_student(
        session=FakeSession(), hook=FakeHook(), identity=identity,
        tokens=torch.zeros((1, 2), dtype=torch.long),
    )
    assert receipt.deployed_n_calls == ((0, 1), (1, 1), (2, 0))
    assert receipt.literal_early_mlp_calls == ((0, 0), (1, 0), (2, 1))
    assert receipt.native_guard_restored and receipt.native_guard_inert


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


def _validation_program(route="R"):
    def site():
        return runtime.AffineCodeProgram(
            mean=torch.zeros(runtime.D_MODEL), scale=torch.ones(runtime.D_MODEL),
            weight=torch.zeros(runtime.D_MODEL, runtime.CODE_DIM),
            bias=torch.zeros(runtime.CODE_DIM),
        )
    return runtime.JointAffineProgram(site(), site(), route=route)


def _step_closure(scope, original_calls, ledger):
    return capabilities.StepClosure(
        identity_sha256="1" * 64, forward_nonce="2" * 64, scope=scope,
        producer_invocations=1, outer_forward_count=1,
        hook_calls=((0, 1), (1, 1), (2, 0)), original_calls=original_calls,
        outer_returned=True, hook_restored=True, hook_inert=True,
        output_shapes=((4, 192, 11),), output_dtypes=("torch.float32",),
        support="64:256", requires_grad=False, grad_fn_absent=True,
        consumed=True, output_sha256="3" * 64, ledger_sha256=ledger,
    )


def test_validation_adapter_consumes_reductions_into_collector_without_tensor_escape(
    monkeypatch,
) -> None:
    model = tiny_model()
    adapter = observed.ObservedBilin18Adapter(model, FakeShip(), production=False)
    program = _validation_program("R")
    rows = torch.arange(4 * 513, dtype=torch.long).view(4, 513) % 11
    identity = runtime.TraceIdentity.from_inputs(
        inputs=rows[:, :256].contiguous(), ordered_batch_indices=(0, 1, 2, 3),
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, fit_role_tensor_sha256="4" * 64,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256="5" * 64,
        role="early_mlp_suffix_transport_v1_validation", phase="validation",
        route="R", control="document_shuffle", teacher_kind="oon_logits",
        trial=0, epoch=0, optimizer_step=0, batch_ordinal=0,
        student_states=((0, "P"), (1, "P"), (2, "N")),
    )
    baseline_count = torch.ones(programs.VALIDATION_ROWS, dtype=torch.long)
    baseline = programs.ValidationBaselineSufficientStatistics(
        common_support_sha256="6" * 64,
        row_ce_sum=torch.ones(programs.VALIDATION_ROWS, dtype=torch.float64),
        row_ce_count=torch.full((programs.VALIDATION_ROWS,), 192, dtype=torch.long),
        row_copy_ce_sum=baseline_count.double(), row_copy_count=baseline_count,
        literal_early_mlp_calls=programs.ZERO_NATIVE_CALLS,
        native_guard_restored=True, native_guard_inert=True,
    )
    collector = programs.ValidationStatisticsCollector(
        route="R", program_sha256=identity.program_snapshot_sha256,
        common_support_sha256="6" * 64, baseline=baseline,
    )

    class Hook:
        def configure(self, **kwargs):
            self.configured = kwargs
        def clear_configuration(self):
            self.cleared = True

    hook = Hook()
    student_closure = _step_closure("student", capabilities.EXACT_ZERO_CALLS, "7" * 64)
    teacher_closure = _step_closure(
        "oon", capabilities.EXACT_EARLY_ORIGINAL_CALLS, "8" * 64,
    )
    adapter_closure = observed.ObservedClosure(
        scope="student", outer_forward_count=1, outer_returned=True,
        attention_dispatch_calls=tuple((site, 1) for site in range(18)),
        mlp_dispatch_calls=tuple((site, 1) for site in range(18)),
        deployed_n_calls=((0, 1), (1, 1), (2, 1)),
        correction_calls=((0, 1), (1, 1), (2, 0)),
        literal_early_mlp_calls=((0, 0), (1, 0), (2, 0)),
        native_guard_restored=True, native_guard_inert=True,
        logit_shape=(4, 256, 11), logit_dtype="torch.float32",
    )
    reductions = capabilities.ValidationBatchReductions(
        identity_sha256=identity.sha256, route="R",
        program_sha256=identity.program_snapshot_sha256,
        row_primary_sum=torch.ones(4, dtype=torch.float64),
        row_primary_count=torch.full((4,), 192, dtype=torch.long),
        row_ce_sum=torch.full((4,), 2.0, dtype=torch.float64),
        row_ce_count=torch.full((4,), 192, dtype=torch.long),
        row_copy_ce_sum=torch.full((4,), 0.5, dtype=torch.float64),
        row_copy_count=torch.ones(4, dtype=torch.long),
    )

    class Result:
        def consume_validation(self, supplied_rows):
            assert supplied_rows is rows
            return reductions, teacher_closure

    class Broker:
        def begin_student(self, supplied_identity, supplied_hook, inputs, indices):
            assert supplied_identity is identity and supplied_hook is hook
            assert tuple(indices) == (0, 1, 2, 3)
            return "session"
        def run_oon_teacher(self, supplied_identity, step, inputs, callback):
            assert supplied_identity is identity and step == "step"
            return Result()

    monkeypatch.setattr(
        adapter, "run_student",
        lambda **kwargs: ("step", student_closure, adapter_closure),
    )
    receipt = adapter.run_validation_batch(
        broker=Broker(), hook=hook, program=program, identity=identity,
        role_rows=rows, ordered_row_indices=(0, 1, 2, 3), collector=collector,
    )
    assert collector.completed_rows == 4 and receipt.control == "document_shuffle"
    assert all(not torch.is_tensor(getattr(receipt, field.name)) for field in fields(receipt))
    assert receipt.reduction_sha256 == runtime.logical_identity_sha256({
        name: runtime.tensor_identity_sha256(getattr(reductions, name))
        for name in (
            "row_primary_sum", "row_primary_count", "row_ce_sum", "row_ce_count",
            "row_copy_ce_sum", "row_copy_count",
        )
    })


def test_final_program_adapter_returns_only_typed_batch_reductions(monkeypatch) -> None:
    model = tiny_model()
    adapter = observed.ObservedBilin18Adapter(model, FakeShip(), production=False)
    program = _validation_program("R")
    rows = torch.arange(4 * 513, dtype=torch.long).view(4, 513) % 11
    identity = runtime.TraceIdentity.from_inputs(
        inputs=rows[:, :256].contiguous(), ordered_batch_indices=(0, 1, 2, 3),
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, fit_role_tensor_sha256="4" * 64,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256="5" * 64,
        role="early_mlp_suffix_transport_v1_final", phase="final",
        route="R", control="true", teacher_kind="oon_logits",
        trial=0, epoch=0, optimizer_step=0, batch_ordinal=0,
        student_states=((0, "P"), (1, "P"), (2, "N")),
    )

    class Hook:
        def configure(self, **kwargs):
            self.configured = kwargs
        def clear_configuration(self):
            self.cleared = True

    hook = Hook()
    student_closure = _step_closure("student", capabilities.EXACT_ZERO_CALLS, "7" * 64)
    teacher_closure = _step_closure(
        "oon", capabilities.EXACT_EARLY_ORIGINAL_CALLS, "8" * 64,
    )
    adapter_closure = observed.ObservedClosure(
        scope="student", outer_forward_count=1, outer_returned=True,
        attention_dispatch_calls=tuple((site, 1) for site in range(18)),
        mlp_dispatch_calls=tuple((site, 1) for site in range(18)),
        deployed_n_calls=((0, 1), (1, 1), (2, 1)),
        correction_calls=((0, 1), (1, 1), (2, 0)),
        literal_early_mlp_calls=((0, 0), (1, 0), (2, 0)),
        native_guard_restored=True, native_guard_inert=True,
        logit_shape=(4, 256, 11), logit_dtype="torch.float32",
    )
    reductions = capabilities.FinalBatchReductions(
        identity_sha256=identity.sha256, route="R",
        program_sha256=identity.program_snapshot_sha256,
        row_primary_sum=torch.ones(4, dtype=torch.float64),
        row_primary_count=torch.full((4,), 192, dtype=torch.long),
        row_ce_sum=torch.full((4,), 2.0, dtype=torch.float64),
        row_ce_count=torch.full((4,), 192, dtype=torch.long),
        row_copy_ce_sum=torch.full((4,), 0.5, dtype=torch.float64),
        row_copy_count=torch.ones(4, dtype=torch.long),
    )

    class Result:
        def consume_final(self, supplied_rows):
            assert supplied_rows is rows
            return reductions, teacher_closure

    class Broker:
        def begin_student(self, supplied_identity, supplied_hook, inputs, indices):
            assert supplied_identity is identity and supplied_hook is hook
            assert tuple(indices) == (0, 1, 2, 3)
            return "session"
        def run_oon_teacher(self, supplied_identity, step, inputs, callback):
            assert supplied_identity is identity and step == "step"
            return Result()

    monkeypatch.setattr(
        adapter, "run_student",
        lambda **kwargs: ("step", student_closure, adapter_closure),
    )
    returned, receipt = adapter.run_final_program_batch(
        broker=Broker(), hook=hook, program=program, identity=identity,
        role_rows=rows, ordered_row_indices=(0, 1, 2, 3),
    )
    assert returned is reductions
    assert receipt.control == "true" and receipt.route == "R"
    assert all(not torch.is_tensor(getattr(receipt, field.name)) for field in fields(receipt))
    assert all(
        torch.is_tensor(getattr(returned, name)) and getattr(returned, name).ndim == 1
        for name in (
            "row_primary_sum", "row_primary_count", "row_ce_sum", "row_ce_count",
            "row_copy_ce_sum", "row_copy_count",
        )
    )


def test_final_exact_mlp2_program_batch_is_ce_only_and_has_no_oon_teacher(
    monkeypatch,
) -> None:
    model = tiny_model()
    adapter = observed.ObservedBilin18Adapter(model, FakeShip(), production=False)
    program = _validation_program("R")
    rows = torch.arange(4 * 513, dtype=torch.long).view(4, 513) % 11
    identity = runtime.TraceIdentity.from_inputs(
        inputs=rows[:, :256].contiguous(), ordered_batch_indices=(0, 1, 2, 3),
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, fit_role_tensor_sha256="4" * 64,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256="5" * 64,
        role="early_mlp_suffix_transport_v1_final", phase="final",
        route="R", control="true", teacher_kind="oon_logits",
        trial=0, epoch=0, optimizer_step=0, batch_ordinal=0,
        student_states=((0, "P"), (1, "P"), (2, "E")),
    )

    class Hook:
        def configure(self, **kwargs):
            self.configured = kwargs
        def clear_configuration(self):
            self.cleared = True

    hook = Hook()
    student_closure = _step_closure("student", capabilities.EXACT_ZERO_CALLS, "7" * 64)
    ce_closure = _step_closure("final_ce", capabilities.EXACT_ZERO_CALLS, "8" * 64)
    adapter_closure = observed.ObservedClosure(
        scope="student", outer_forward_count=1, outer_returned=True,
        attention_dispatch_calls=tuple((site, 1) for site in range(18)),
        mlp_dispatch_calls=tuple((site, 1) for site in range(18)),
        deployed_n_calls=((0, 1), (1, 1), (2, 0)),
        correction_calls=((0, 1), (1, 1), (2, 0)),
        literal_early_mlp_calls=((0, 0), (1, 0), (2, 1)),
        native_guard_restored=True, native_guard_inert=True,
        logit_shape=(4, 256, 11), logit_dtype="torch.float32",
    )
    reductions = capabilities.FinalCEBatchReductions(
        identity_sha256=identity.sha256, route="R",
        program_sha256=identity.program_snapshot_sha256,
        row_ce_sum=torch.full((4,), 2.0, dtype=torch.float64),
        row_ce_count=torch.full((4,), 192, dtype=torch.long),
        row_copy_ce_sum=torch.full((4,), 0.5, dtype=torch.float64),
        row_copy_count=torch.ones(4, dtype=torch.long),
    )

    class Broker:
        def begin_student(self, supplied_identity, supplied_hook, inputs, indices):
            return "session"
        def consume_final_ce(self, supplied_identity, step, supplied_rows):
            assert supplied_identity is identity and step == "step" and supplied_rows is rows
            return reductions, ce_closure
        def run_oon_teacher(self, *args, **kwargs):
            raise AssertionError("exact-MLP2 E is CE-only and cannot construct OON")

    monkeypatch.setattr(
        adapter, "run_student",
        lambda **kwargs: ("step", student_closure, adapter_closure),
    )
    returned, receipt = adapter.run_final_program_batch(
        broker=Broker(), hook=hook, program=program, identity=identity,
        role_rows=rows, ordered_row_indices=(0, 1, 2, 3),
    )
    assert returned is reductions and receipt.control == "true"
    assert not hasattr(returned, "row_primary_sum")


def test_final_exact_mlp2_local_program_is_ce_only_and_rejects_denominators(
    monkeypatch,
) -> None:
    """E never smuggles L's coordinate statistic into a CE-only observation."""

    model = tiny_model()
    adapter = observed.ObservedBilin18Adapter(model, FakeShip(), production=False)
    program = _validation_program("L")
    rows = torch.arange(4 * 513, dtype=torch.long).view(4, 513) % 11
    identity = runtime.TraceIdentity.from_inputs(
        inputs=rows[:, :256].contiguous(), ordered_batch_indices=(0, 1, 2, 3),
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, fit_role_tensor_sha256="4" * 64,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256="5" * 64,
        role="early_mlp_suffix_transport_v1_final", phase="final",
        route="L", control="true", teacher_kind="coordinate_labels",
        trial=0, epoch=0, optimizer_step=0, batch_ordinal=0,
        student_states=((0, "P"), (1, "P"), (2, "E")),
    )

    class Hook:
        def configure(self, **kwargs):
            raise AssertionError("denominator validation must fail before execution")
        def clear_configuration(self):
            raise AssertionError("no hook was configured")

    with pytest.raises(RuntimeError, match="cannot receive denominators"):
        adapter.run_final_program_batch(
            broker=object(), hook=Hook(), program=program, identity=identity,
            role_rows=rows, ordered_row_indices=(0, 1, 2, 3),
            denominators=(2.0, 4.0),
        )


def test_validation_baseline_adapter_reduces_deployed_n_n_and_poison_closes(monkeypatch) -> None:
    class BaselineShip:
        production = False
        def __init__(self, malicious=False):
            self.malicious = malicious
        def attention(self, event):
            return torch.zeros_like(event.state), torch.zeros(1)
        def mlp(self, event):
            if self.malicious and event.site == 0:
                return event.block.mlp(event.state)
            return torch.zeros_like(event.state)

    model = tiny_model()
    rows = torch.arange(programs.VALIDATION_ROWS * 513, dtype=torch.long).view(
        programs.VALIDATION_ROWS, 513,
    ) % 17
    context = capabilities.ValidationRunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64,
        validation_role_tensor_sha256=runtime.tensor_identity_sha256(rows),
        identity_teacher_mapping_sha256="5" * 64,
    )
    identity = programs.make_validation_baseline_identity(
        context=context, role_rows=rows, batch_ordinal=0,
    )
    collector = programs.ValidationBaselineCollector(
        common_support_sha256=identity.common_support_sha256,
    )
    batch_rows = rows[:4].contiguous()

    def fake_forward(model, tokens, attention, mlp, *, require_production):
        state = torch.zeros(4, 256, 8)
        for site, block in enumerate(model.transformer.h):
            event = SimpleNamespace(site=site, block=block, state=state)
            attention(event)
            mlp(event)
        generator = torch.Generator().manual_seed(901)
        return torch.randn(4, 256, 17, generator=generator)

    monkeypatch.setattr(observed.facade, "forward_with_dispatch", fake_forward)
    adapter = observed.ObservedBilin18Adapter(model, BaselineShip(), production=False)
    receipt = adapter.run_validation_baseline_batch(
        identity=identity, role_rows=batch_rows,
        ordered_row_indices=(0, 1, 2, 3), collector=collector,
    )
    assert collector.completed_rows == 4 and receipt.identity_sha256 == identity.sha256
    assert all(not torch.is_tensor(getattr(receipt, field.name)) for field in fields(receipt))

    bad_collector = programs.ValidationBaselineCollector(
        common_support_sha256=identity.common_support_sha256,
    )
    malicious = observed.ObservedBilin18Adapter(
        model, BaselineShip(malicious=True), production=False,
    )
    with pytest.raises(RuntimeError, match="literal native MLP0"):
        malicious.run_validation_baseline_batch(
            identity=identity, role_rows=batch_rows,
            ordered_row_indices=(0, 1, 2, 3), collector=bad_collector,
        )
    assert bad_collector.completed_rows == 0


@pytest.mark.parametrize(
    ("execution_kind", "background", "deployed", "literal"),
    (
        ("deployed_baseline", "N", ((0, 1), (1, 1), (2, 1)), ((0, 0), (1, 0), (2, 0))),
        ("deployed_baseline", "E", ((0, 1), (1, 1), (2, 0)), ((0, 0), (1, 0), (2, 1))),
        ("native_baseline", "N", ((0, 0), (1, 0), (2, 1)), ((0, 1), (1, 1), (2, 0))),
        ("native_baseline", "E", ((0, 0), (1, 0), (2, 0)), ((0, 1), (1, 1), (2, 1))),
    ),
)
def test_final_baseline_physical_paths_close_exact_early_call_ledgers(
    execution_kind, background, deployed, literal,
) -> None:
    adapter = observed.ObservedBilin18Adapter(tiny_model(), FakeShip(), production=False)
    with torch.no_grad():
        logits, closure = adapter._run_final_baseline_forward(
            tokens=torch.zeros((1, 2), dtype=torch.long),
            execution_kind=execution_kind, background=background,
        )
    assert logits.shape == (1, 2, facade.LOGIT_VOCAB)
    assert closure.deployed_n_calls == deployed
    assert closure.literal_early_mlp_calls == literal
    assert closure.correction_calls == ((0, 0), (1, 0), (2, 0))
    assert closure.native_guard_restored and closure.native_guard_inert


def test_final_deployed_baseline_poison_rejects_hidden_native_call() -> None:
    adapter = observed.ObservedBilin18Adapter(
        tiny_model(), FakeShip(call_native_at_zero=True), production=False,
    )
    with pytest.raises(RuntimeError, match="literal native MLP0"):
        with torch.no_grad():
            adapter._run_final_baseline_forward(
                tokens=torch.zeros((1, 2), dtype=torch.long),
                execution_kind="deployed_baseline", background="N",
            )


def _materialized_baseline(arm: str, background: str):
    plan = final_actions.plan_for(arm, background)
    return final_actions.MaterializedFinalAction(
        plan=plan, source_bank_sha256="a" * 64,
        component_sha256s={"baseline": runtime.logical_identity_sha256({
            "execution_kind": plan.arm_plan.execution_kind,
        })},
        program=None,
    )


def _materialized_program(arm: str, background: str):
    plan = final_actions.plan_for(arm, background)
    route = "L" if plan.arm_plan.identity_route == "Q" else (
        plan.arm_plan.identity_route
    )
    program = _validation_program(route)
    return final_actions.MaterializedFinalAction(
        plan=plan, source_bank_sha256="a" * 64,
        component_sha256s={"synthetic_test_program": runtime.program_snapshot_sha256(program)},
        program=program,
    )


def _final_baseline_identity(materialized, rows):
    return final_actions.FinalActionBatchIdentity.from_role_rows(
        materialized=materialized, role_rows=rows,
        ordered_batch_indices=(0, 1, 2, 3), batch_ordinal=0,
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64,
        final_role_tensor_sha256="4" * 64, program_payload_sha256="5" * 64,
        common_support_sha256="6" * 64,
    )


@pytest.mark.parametrize(
    ("arm", "background", "expected_calls", "teacher_reused"),
    (
        ("n_n", "N", (("deployed_baseline", "N"), ("native_baseline", "N")), False),
        ("n_n", "E", (("deployed_baseline", "E"),), False),
        ("o_o", "N", (("native_baseline", "N"),), True),
        ("o_o", "E", (("native_baseline", "E"),), False),
    ),
)
def test_final_baseline_batch_reduces_all_four_paths_without_tensor_escape(
    monkeypatch, arm, background, expected_calls, teacher_reused,
) -> None:
    model = tiny_model()
    adapter = observed.ObservedBilin18Adapter(model, FakeShip(), production=False)
    rows = (torch.arange(4 * 513, dtype=torch.long).view(4, 513) % 11).contiguous()
    materialized = _materialized_baseline(arm, background)
    identity = _final_baseline_identity(materialized, rows)
    calls = []

    def fake_forward(*, tokens, execution_kind, background):
        calls.append((execution_kind, background))
        seed = 110 if execution_kind == "deployed_baseline" else 220
        logits = torch.randn(4, 256, 11, generator=torch.Generator().manual_seed(seed))
        exact = set((0, 1) if execution_kind == "native_baseline" else ())
        if background == "E":
            exact.add(2)
        closure = observed.ObservedClosure(
            scope=f"final_{execution_kind}_{background}",
            outer_forward_count=1, outer_returned=True,
            attention_dispatch_calls=tuple((site, 1) for site in range(4)),
            mlp_dispatch_calls=tuple((site, 1) for site in range(4)),
            deployed_n_calls=tuple(
                (site, 0 if site in exact else 1) for site in observed.EARLY_SITES
            ),
            correction_calls=((0, 0), (1, 0), (2, 0)),
            literal_early_mlp_calls=tuple(
                (site, 1 if site in exact else 0) for site in observed.EARLY_SITES
            ),
            native_guard_restored=True, native_guard_inert=True,
            logit_shape=tuple(logits.shape), logit_dtype=str(logits.dtype),
        )
        return logits, closure

    monkeypatch.setattr(adapter, "_run_final_baseline_forward", fake_forward)
    reductions, receipt = adapter.run_final_baseline_batch(
        materialized=materialized, identity=identity, role_rows=rows,
        ordered_row_indices=(0, 1, 2, 3),
    )
    assert tuple(calls) == expected_calls
    assert reductions.action_key == f"{arm}/{background}"
    assert (reductions.row_primary_sum is not None) == (background == "N")
    assert (receipt.observed_teacher_closure_sha256 is not None) == (background == "N")
    assert receipt.teacher_reused_student is teacher_reused
    assert all(not torch.is_tensor(getattr(receipt, field.name)) for field in fields(receipt))
    assert all(
        value is None or (torch.is_tensor(value) and value.device.type == "cpu" and value.ndim == 1)
        for value in (
            reductions.row_primary_sum, reductions.row_primary_count,
            reductions.row_ce_sum, reductions.row_ce_count,
            reductions.row_copy_ce_sum, reductions.row_copy_count,
        )
    )


def test_final_baseline_batch_rejects_target_substitution_before_forward(monkeypatch) -> None:
    adapter = observed.ObservedBilin18Adapter(tiny_model(), FakeShip(), production=False)
    rows = (torch.arange(4 * 513, dtype=torch.long).view(4, 513) % 11).contiguous()
    materialized = _materialized_baseline("n_n", "N")
    identity = _final_baseline_identity(materialized, rows)
    changed = rows.clone()
    changed[0, 256] = (changed[0, 256] + 1) % 11
    assert torch.equal(changed[:, :256], rows[:, :256])
    monkeypatch.setattr(
        adapter, "_run_final_baseline_forward",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("forward must not run")),
    )
    with pytest.raises(RuntimeError, match="differs from its sealed identity"):
        adapter.run_final_baseline_batch(
            materialized=materialized, identity=identity, role_rows=changed,
            ordered_row_indices=(0, 1, 2, 3),
        )


def test_materialized_final_wrapper_binds_hybrid_action_to_runtime_receipt(
    monkeypatch,
) -> None:
    adapter = observed.ObservedBilin18Adapter(tiny_model(), FakeShip(), production=False)
    rows = (torch.arange(4 * 513, dtype=torch.long).view(4, 513) % 11).contiguous()
    materialized = _materialized_program("s0_l1", "E")
    identity = final_actions.FinalActionBatchIdentity.from_role_rows(
        materialized=materialized, role_rows=rows,
        ordered_batch_indices=(0, 1, 2, 3), batch_ordinal=0,
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, final_role_tensor_sha256="4" * 64,
        program_payload_sha256="5" * 64, common_support_sha256="6" * 64,
    )
    context = capabilities.FinalRunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, final_role_tensor_sha256="4" * 64,
        identity_teacher_mapping_sha256="7" * 64,
    )
    bases = {
        site: torch.eye(runtime.D_MODEL)[:, :runtime.CODE_DIM] for site in (0, 1)
    }
    broker = adapter.make_capability_broker(
        issuer_id="8" * 64, coordinator=runtime.ScopeCoordinator(),
        run_context=context, bases=bases,
    )
    captured = {}
    returned = object()

    def lower(**kwargs):
        trace = kwargs["identity"]
        captured.update(kwargs)
        return returned, observed.ObservedFinalProgramBatchReceipt(
            identity_sha256=trace.sha256, route=trace.route,
            control=trace.control, batch_ordinal=trace.batch_ordinal,
            ordered_row_indices_sha256=runtime.logical_identity_sha256([0, 1, 2, 3]),
            reduction_sha256="9" * 64, student_ledger_sha256="b" * 64,
            teacher_ledger_sha256="c" * 64, observed_closure_sha256="d" * 64,
        )

    monkeypatch.setattr(adapter, "run_final_program_batch", lower)
    reductions, receipt = adapter.run_materialized_final_program_batch(
        broker=broker, hook=object(), materialized=materialized, identity=identity,
        final_context=context, role_rows=rows,
        ordered_row_indices=(0, 1, 2, 3),
    )
    trace = captured["identity"]
    assert reductions is returned
    assert trace.route == "S0" and trace.control == "hybrid_s0_l1"
    assert trace.student_states == ((0, "P"), (1, "P"), (2, "E"))
    assert runtime.program_snapshot_sha256(captured["program"]) == materialized.program_sha256
    assert receipt.action_key == "s0_l1/E"
    assert receipt.final_action_identity_sha256 == identity.sha256
    assert receipt.materialization_sha256 == materialized.sha256
    assert receipt.runtime_identity_sha256 == trace.sha256
    assert all(not torch.is_tensor(getattr(receipt, field.name)) for field in fields(receipt))


def test_materialized_final_wrapper_rejects_target_or_context_before_lower_forward(
    monkeypatch,
) -> None:
    adapter = observed.ObservedBilin18Adapter(tiny_model(), FakeShip(), production=False)
    rows = (torch.arange(4 * 513, dtype=torch.long).view(4, 513) % 11).contiguous()
    materialized = _materialized_program("qq", "N")
    identity = final_actions.FinalActionBatchIdentity.from_role_rows(
        materialized=materialized, role_rows=rows,
        ordered_batch_indices=(0, 1, 2, 3), batch_ordinal=0,
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, final_role_tensor_sha256="4" * 64,
        program_payload_sha256="5" * 64, common_support_sha256="6" * 64,
    )
    context = capabilities.FinalRunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, final_role_tensor_sha256="4" * 64,
        identity_teacher_mapping_sha256="7" * 64,
    )
    bases = {
        site: torch.eye(runtime.D_MODEL)[:, :runtime.CODE_DIM] for site in (0, 1)
    }
    broker = adapter.make_capability_broker(
        issuer_id="8" * 64, coordinator=runtime.ScopeCoordinator(),
        run_context=context, bases=bases,
    )
    monkeypatch.setattr(
        adapter, "run_final_program_batch",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("forward must not run")),
    )
    changed = rows.clone()
    changed[0, 256] = (changed[0, 256] + 1) % 11
    with pytest.raises(RuntimeError, match="differs from its sealed identity"):
        adapter.run_materialized_final_program_batch(
            broker=broker, hook=object(), materialized=materialized,
            identity=identity, final_context=context, role_rows=changed,
            ordered_row_indices=(0, 1, 2, 3), denominators=(1.0, 1.0),
        )

    other_context = capabilities.FinalRunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, final_role_tensor_sha256="4" * 64,
        identity_teacher_mapping_sha256="e" * 64,
    )
    with pytest.raises(RuntimeError, match="broker and action run context differ"):
        adapter.run_materialized_final_program_batch(
            broker=broker, hook=object(), materialized=materialized,
            identity=identity, final_context=other_context, role_rows=rows,
            ordered_row_indices=(0, 1, 2, 3), denominators=(1.0, 1.0),
        )
