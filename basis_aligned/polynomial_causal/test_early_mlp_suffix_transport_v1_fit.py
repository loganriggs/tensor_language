from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_fit as fit
import early_mlp_suffix_transport_v1_mapped as mapped
import early_mlp_suffix_transport_v1_runtime as runtime


def _basis() -> torch.Tensor:
    return torch.eye(runtime.D_MODEL)[:, :runtime.CODE_DIM]


def _state(seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "grammar": "affine", "interface": "state_complete_p",
        "mean": torch.zeros(runtime.D_MODEL), "scale": torch.ones(runtime.D_MODEL),
        "left": torch.randn(runtime.D_MODEL, runtime.CODE_DIM, generator=generator) / 100,
        "right": torch.randn(runtime.CODE_DIM, runtime.CODE_DIM, generator=generator) / 100,
        "bias": torch.zeros(runtime.CODE_DIM),
    }


def _program(route: str) -> runtime.JointAffineProgram:
    return runtime.JointAffineProgram.from_v21_states(
        {0: _state(1), 1: _state(2)}, route=route,
    )


def _context(rows: torch.Tensor) -> capabilities.RunContext:
    return capabilities.RunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64,
        fit_role_tensor_sha256=runtime.tensor_identity_sha256(rows),
        identity_teacher_mapping_sha256="5" * 64,
        fit_row_count=capabilities.FIT_ROW_COUNT,
    )


@dataclass(frozen=True)
class _Observed:
    scope: str = "student"
    outer_forward_count: int = 1


class _Hook:
    def __init__(self) -> None:
        self.program = None
        self.states = {}

    def configure(self, *, program, states, mapped_parent=None) -> None:
        self.program = program
        self.states = dict(states)
        self.mapped_parent = mapped_parent


class _Result:
    def __init__(self, *, program, denominator: bool, ledger: str) -> None:
        self.program = program
        self.denominator = denominator
        self.closure = SimpleNamespace(ledger_sha256=ledger)

    def consume_moments(self):
        assert self.denominator
        coordinates = torch.arange(runtime.CODE_DIM, dtype=torch.float32).view(1, 1, -1)
        row = torch.arange(runtime.BATCH_SIZE, dtype=torch.float32).view(-1, 1, 1)
        labels = (coordinates + row).expand(-1, runtime.SCORE_STOP - runtime.SCORE_START, -1)
        moments = runtime.MomentSufficientStatistics.from_labels(labels)
        return (moments, moments), self.closure

    def consume_loss(self, denominators=None):
        assert not self.denominator
        if self.program.route == "L":
            assert denominators is not None and len(denominators) == 2
        else:
            assert denominators is None
        loss = sum(
            parameter.square().mean()
            for parameter in self.program.parameters() if parameter.requires_grad
        )
        return loss, self.closure


class _Broker:
    def __init__(self, context) -> None:
        self.context = context
        self.program = None
        self.identities = []

    def begin_student(self, identity, hook, inputs, indices):
        self.context.require_identity(identity, inputs, indices)
        self.program = hook.program
        self.identities.append(identity)
        return object()

    def run_coordinate_teacher(self, identity, step):
        return _Result(
            program=self.program,
            denominator=identity.phase == "initial_denominator",
            ledger="b" * 64,
        )


class _Adapter:
    def run_student(self, *, session, hook, identity, tokens):
        del session, tokens
        assert hook.states == {0: "P", 1: "P"}
        return object(), SimpleNamespace(ledger_sha256="a" * 64), _Observed()

    def run_oon_teacher(self, *, broker, identity, step, tokens):
        del step, tokens
        return _Result(program=broker.program, denominator=False, ledger="c" * 64)


class _MappedBroker:
    def __init__(self, context) -> None:
        self.context = context
        self.program = None
        self.identities = []

    def begin_student(self, identity, hook, inputs, indices):
        self.context.require_source_identity(identity, inputs, indices)
        self.program = hook.program
        self.identities.append(identity)
        return object()


class _MappedAdapter(_Adapter):
    def __init__(self, context) -> None:
        self.context = context
        self.pairs = []

    def run_mapped_oon_teacher(
        self, *, broker, identity, step, fit_rows, student_tokens,
        student_indices, teacher_tokens, teacher_indices,
    ):
        del step
        self.context.require_identity(
            identity, fit_rows=fit_rows, student_inputs=student_tokens,
            student_indices=student_indices, teacher_inputs=teacher_tokens,
            teacher_indices=teacher_indices,
        )
        self.pairs.append((tuple(student_indices), tuple(teacher_indices)))
        return _Result(program=broker.program, denominator=False, ledger="d" * 64)

    def run_mapped_coordinate_teacher(
        self, *, broker, identity, step, fit_rows, student_tokens,
        student_indices, teacher_tokens, teacher_indices, program,
    ):
        del step
        assert program is broker.program and identity.route == "L"
        self.context.require_identity(
            identity, fit_rows=fit_rows, student_inputs=student_tokens,
            student_indices=student_indices, teacher_inputs=teacher_tokens,
            teacher_indices=teacher_indices,
        )
        self.pairs.append((tuple(student_indices), tuple(teacher_indices)))
        return _Result(program=broker.program, denominator=False, ledger="e" * 64)

    def prepare_mapped_parent(
        self, *, broker, identity, fit_rows, student_tokens, student_indices,
        teacher_tokens, teacher_indices, program,
    ):
        self.context.require_identity(
            identity, fit_rows=fit_rows, student_inputs=student_tokens,
            student_indices=student_indices, teacher_inputs=teacher_tokens,
            teacher_indices=teacher_indices,
        )
        assert program.route == "T"
        self.pairs.append(("parent", tuple(student_indices), tuple(teacher_indices)))
        return object(), SimpleNamespace(ledger_sha256="f" * 64)

    def run_a_null_oon_teacher(
        self, *, broker, identity, step, fit_rows, student_tokens,
        student_indices, teacher_tokens, teacher_indices,
    ):
        del step
        self.context.require_identity(
            identity, fit_rows=fit_rows, student_inputs=student_tokens,
            student_indices=student_indices, teacher_inputs=teacher_tokens,
            teacher_indices=teacher_indices,
        )
        self.pairs.append(("teacher", tuple(student_indices), tuple(teacher_indices)))
        return _Result(program=broker.program, denominator=False, ledger="9" * 64)


def test_schedule_and_identity_bind_program_before_forward(monkeypatch) -> None:
    monkeypatch.setattr(capabilities, "FIT_ROW_COUNT", 8)
    monkeypatch.setattr(capabilities, "FIT_BATCHES_PER_EPOCH", 2)
    rows = torch.arange(8 * 513, dtype=torch.long).view(8, 513)
    context = _context(rows)
    program = _program("R")
    indices = fit.scheduled_indices(phase="fit", trial=2, epoch=1, batch_ordinal=1)
    inputs = rows[torch.tensor(indices), :runtime.SEQUENCE_LENGTH]
    identity = fit.make_identity(
        context=context, program=program, inputs=inputs, indices=indices,
        phase="fit", route="R", trial=2, epoch=1, batch_ordinal=1,
    )
    assert identity.optimizer_step == 3
    assert identity.program_snapshot_sha256 == runtime.program_snapshot_sha256(program)
    with pytest.raises(RuntimeError, match="preregistered schedule"):
        fit.make_identity(
            context=context, program=program, inputs=inputs,
            indices=tuple(reversed(indices)), phase="fit", route="R", trial=2,
            epoch=1, batch_ordinal=1,
        )


def test_denominator_and_true_fit_share_exact_transaction_surface(monkeypatch) -> None:
    monkeypatch.setattr(capabilities, "FIT_ROW_COUNT", runtime.BATCH_SIZE)
    monkeypatch.setattr(capabilities, "FIT_BATCHES_PER_EPOCH", 1)
    monkeypatch.setattr(runtime, "EPOCHS", 1)
    rows = torch.arange(
        runtime.BATCH_SIZE * 513, dtype=torch.long,
    ).view(runtime.BATCH_SIZE, 513)
    context = _context(rows)
    adapter, hook = _Adapter(), _Hook()

    q_broker = _Broker(context)
    denominator = fit.run_initial_denominator_pass(
        rows=rows, context=context, program=_program("L"), adapter=adapter,
        broker=q_broker, hook=hook,
    )
    assert denominator.completed_steps == 1
    assert all(float(value) > 0 for value in denominator.denominators)
    assert q_broker.identities[0].route == "Q"

    r_broker = _Broker(context)
    candidate = fit.run_true_fit_trial(
        rows=rows, context=context, program=_program("R"), route="R", trial=0,
        denominators=None, adapter=adapter, broker=r_broker, hook=hook,
    )
    assert candidate.completed_steps == 1
    assert candidate.learning_rate == runtime.LEARNING_RATES[0]
    assert candidate.loss_min == candidate.loss_max == candidate.loss_sum
    assert r_broker.identities[0].teacher_kind == "oon_logits"
    assert set(candidate.state_dict) == {
        "site0.bias", "site0.mean", "site0.scale", "site0.weight",
        "site1.bias", "site1.mean", "site1.scale", "site1.weight",
    }

    t_broker = _Broker(context)
    transport = _program("L").independent_clone(route="T")
    transport_candidate = fit.run_true_fit_trial(
        rows=rows, context=context, program=transport, route="T", trial=1,
        denominators=None, adapter=adapter, broker=t_broker, hook=hook,
    )
    assert transport_candidate.completed_steps == 1
    assert t_broker.identities[0].route == "T"
    assert t_broker.identities[0].teacher_kind == "oon_logits"
    assert set(transport_candidate.state_dict) == {
        "cross", "site0.bias", "site0.mean", "site0.scale", "site0.weight",
        "site1.bias", "site1.mean", "site1.scale", "site1.weight",
    }


def test_fit_rows_and_loss_families_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(capabilities, "FIT_ROW_COUNT", runtime.BATCH_SIZE)
    monkeypatch.setattr(capabilities, "FIT_BATCHES_PER_EPOCH", 1)
    rows = torch.zeros((runtime.BATCH_SIZE, 513), dtype=torch.long)
    context = _context(rows)
    with pytest.raises(RuntimeError, match="sealed run context"):
        fit.validate_fit_rows(rows.clone().add_(1), context)
    with pytest.raises(ValueError, match="requires both"):
        fit.run_true_fit_trial(
            rows=rows, context=context, program=_program("L"), route="L", trial=0,
            denominators=None, adapter=_Adapter(), broker=_Broker(context), hook=_Hook(),
        )
    with pytest.raises(ValueError, match="must not receive"):
        fit.run_true_fit_trial(
            rows=rows, context=context, program=_program("S0"), route="S0", trial=0,
            denominators=(1.0, 1.0), adapter=_Adapter(), broker=_Broker(context), hook=_Hook(),
        )


def test_document_shuffle_fit_owns_exact_source_target_schedule(monkeypatch) -> None:
    monkeypatch.setattr(capabilities, "FIT_ROW_COUNT", runtime.BATCH_SIZE)
    monkeypatch.setattr(capabilities, "FIT_BATCHES_PER_EPOCH", 1)
    monkeypatch.setattr(runtime, "EPOCHS", 1)
    rows = torch.arange(
        runtime.BATCH_SIZE * 513, dtype=torch.long,
    ).view(runtime.BATCH_SIZE, 513)
    base = _context(rows)
    records = [{
        "document_id": f"doc-{index}", "dataset_document_index": index,
        "chunk_id": 0, "token_start": 0,
    } for index in range(runtime.BATCH_SIZE)]
    plan = mapped.build_document_block_plan(records, control="document_shuffle")
    context = mapped.MappedRunContext(base=base, plan=plan)
    adapter, broker = _MappedAdapter(context), _MappedBroker(context)

    candidate = fit.run_document_shuffle_fit_trial(
        rows=rows, context=context, program=_program("S1"), route="S1", trial=1,
        adapter=adapter, broker=broker, hook=_Hook(),
    )

    assert isinstance(candidate, fit.MappedFitCandidate)
    assert not isinstance(candidate, fit.FitCandidate)
    assert candidate.control == "document_shuffle"
    assert candidate.mapping_sha256 == plan.sha256
    assert candidate.completed_steps == 1
    identity = broker.identities[0]
    assert identity.control == "document_shuffle" and identity.route == "S1"
    assert identity.teacher_mapping_sha256 == plan.sha256
    source_indices, target_indices = adapter.pairs[0]
    assert target_indices == plan.target_indices(source_indices)
    assert all(
        plan.source_documents[source] != plan.source_documents[target]
        for source, target in zip(source_indices, target_indices, strict=True)
    )

    local_adapter, local_broker = _MappedAdapter(context), _MappedBroker(context)
    local = fit.run_document_shuffle_fit_trial(
        rows=rows, context=context, program=_program("L"), route="L", trial=0,
        adapter=local_adapter, broker=local_broker, hook=_Hook(),
        denominators=(1.0, 1.0),
    )
    assert local.route == "L" and local.completed_steps == 1
    assert local_broker.identities[0].teacher_kind == "coordinate_labels"
    assert local_adapter.pairs[0][1] == plan.target_indices(local_adapter.pairs[0][0])

    with pytest.raises(ValueError, match="route/trial/program"):
        fit.run_document_shuffle_fit_trial(
            rows=rows, context=context, program=_program("T"), route="T", trial=0,
            adapter=adapter, broker=broker, hook=_Hook(),
        )
    with pytest.raises(ValueError, match="requires both"):
        fit.run_document_shuffle_fit_trial(
            rows=rows, context=context, program=_program("L"), route="L", trial=0,
            adapter=adapter, broker=broker, hook=_Hook(),
        )


def test_a_null_fit_owns_false_parent_then_true_source_teacher(monkeypatch) -> None:
    monkeypatch.setattr(capabilities, "FIT_ROW_COUNT", runtime.BATCH_SIZE)
    monkeypatch.setattr(capabilities, "FIT_BATCHES_PER_EPOCH", 1)
    monkeypatch.setattr(runtime, "EPOCHS", 1)
    rows = torch.arange(
        runtime.BATCH_SIZE * 513, dtype=torch.long,
    ).view(runtime.BATCH_SIZE, 513)
    base = _context(rows)
    records = [{
        "document_id": f"doc-{index}", "dataset_document_index": index,
        "chunk_id": 0, "token_start": 0,
    } for index in range(runtime.BATCH_SIZE)]
    plan = mapped.build_document_block_plan(records, control="A_null_00")
    context = mapped.MappedRunContext(base=base, plan=plan)
    adapter, broker, hook = _MappedAdapter(context), _MappedBroker(context), _Hook()

    candidate = fit.run_a_null_fit_trial(
        rows=rows, context=context, program=_program("T"), trial=0,
        adapter=adapter, broker=broker, hook=hook,
    )

    assert isinstance(candidate, fit.MappedFitCandidate)
    assert candidate.control == "A_null_00" and candidate.route == "T"
    assert candidate.mapping_sha256 == plan.sha256 and candidate.completed_steps == 1
    assert set(candidate.state_dict) == {
        "cross", "site0.bias", "site0.mean", "site0.scale", "site0.weight",
        "site1.bias", "site1.mean", "site1.scale", "site1.weight",
    }
    identity = broker.identities[0]
    assert identity.control == "A_null_00" and identity.teacher_kind == "oon_logits"
    parent_tag, source_indices, target_indices = adapter.pairs[0]
    teacher_tag, teacher_source, teacher_target = adapter.pairs[1]
    assert parent_tag == "parent" and teacher_tag == "teacher"
    assert target_indices == plan.target_indices(source_indices)
    assert (teacher_source, teacher_target) == (source_indices, target_indices)
    assert hook.mapped_parent is not None
