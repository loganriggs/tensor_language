from __future__ import annotations

import pytest
import torch

import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_mapped as mapped
import early_mlp_suffix_transport_v1_runtime as runtime


def _records(spec=(2, 2, 1, 1, 1)):
    records = []
    for document_index, count in enumerate(spec):
        for chunk in range(count):
            records.append({
                "document_id": f"doc-{document_index}",
                "dataset_document_index": 100 + document_index,
                "chunk_id": chunk,
                "token_start": chunk * 256,
            })
    return records


def test_document_plan_is_deterministic_block_bijection_without_fixed_documents() -> None:
    records = _records()
    plan = mapped.build_document_block_plan(records, control="document_shuffle")
    replay = mapped.build_document_block_plan(records, control="document_shuffle")
    assert plan == replay and plan.seed == mapped.DOCUMENT_SHUFFLE_SEED
    assert sorted(plan.row_targets) == list(range(len(records)))
    assert all(
        source != target
        for source, target in zip(plan.source_documents, plan.target_documents, strict=True)
    )
    # Both rows from each two-row document move together and preserve within-doc order.
    assert plan.row_targets[1] == plan.row_targets[0] + 1
    assert plan.row_targets[3] == plan.row_targets[2] + 1
    assert {(item.rows_per_document, len(item.documents)) for item in plan.strata} == {
        (1, 3), (2, 2),
    }


def test_all_registered_control_seeds_and_degenerate_strata_fail_closed() -> None:
    records = _records()
    nulls = [
        mapped.build_document_block_plan(records, control=f"A_null_{index:02d}")
        for index in range(20)
    ]
    assert [plan.seed for plan in nulls] == list(range(2026083100, 2026083120))
    assert all(plan.control == f"A_null_{index:02d}" for index, plan in enumerate(nulls))
    with pytest.raises(ValueError, match="A_null"):
        mapped.control_seed("A_null_20")
    with pytest.raises(RuntimeError, match="at least two"):
        mapped.build_document_block_plan(_records((2, 1, 1)), control="document_shuffle")


def test_interleaved_documents_and_bad_provenance_are_rejected() -> None:
    records = _records((2, 2))
    interleaved = [records[0], records[2], records[1], records[3]]
    with pytest.raises(RuntimeError, match="not contiguous"):
        mapped.build_document_block_plan(interleaved, control="document_shuffle")
    malformed = [dict(record) for record in records]
    malformed[0]["extra"] = 1
    with pytest.raises(ValueError, match="schema"):
        mapped.build_document_block_plan(malformed, control="document_shuffle")


def _program(route: str) -> runtime.JointAffineProgram:
    state = {
        site: {
            "grammar": "affine", "interface": "state_complete_p",
            "mean": torch.zeros(runtime.D_MODEL), "scale": torch.ones(runtime.D_MODEL),
            "left": torch.zeros(runtime.D_MODEL, runtime.CODE_DIM),
            "right": torch.zeros(runtime.CODE_DIM, runtime.CODE_DIM),
            "bias": torch.zeros(runtime.CODE_DIM),
        }
        for site in (0, 1)
    }
    return runtime.JointAffineProgram.from_v21_states(state, route=route)


def test_mapped_context_binds_source_schedule_plan_and_target_tokens(monkeypatch) -> None:
    monkeypatch.setattr(capabilities, "FIT_ROW_COUNT", runtime.BATCH_SIZE)
    monkeypatch.setattr(capabilities, "FIT_BATCHES_PER_EPOCH", 1)
    rows = torch.arange(
        runtime.BATCH_SIZE * 513, dtype=torch.long,
    ).view(runtime.BATCH_SIZE, 513)
    records = _records((1, 1, 1, 1))
    plan = mapped.build_document_block_plan(records, control="document_shuffle")
    base = capabilities.RunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64,
        fit_role_tensor_sha256=runtime.tensor_identity_sha256(rows),
        identity_teacher_mapping_sha256="4" * 64,
        fit_row_count=runtime.BATCH_SIZE,
    )
    context = mapped.MappedRunContext(base=base, plan=plan)
    source_indices = tuple(int(value) for value in runtime.fit_permutations(4, 0)[0])
    source = rows[torch.tensor(source_indices), :runtime.SEQUENCE_LENGTH]
    targets = plan.target_indices(source_indices)
    teacher = rows[torch.tensor(targets), :runtime.SEQUENCE_LENGTH]
    program = _program("L")
    identity = runtime.TraceIdentity.from_inputs(
        inputs=source, ordered_batch_indices=source_indices,
        source_commit=base.source_commit,
        inherited_snapshot_sha256=base.inherited_snapshot_sha256,
        rows_receipt_sha256=base.rows_receipt_sha256,
        fit_role_tensor_sha256=base.fit_role_tensor_sha256,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256=plan.sha256, phase="fit", route="L",
        control="document_shuffle", teacher_kind="coordinate_labels", trial=0,
        epoch=0, optimizer_step=0, batch_ordinal=0,
        student_states=((0, "P"), (1, "P"), (2, "N")),
    )
    context.require_identity(
        identity, fit_rows=rows, student_inputs=source,
        student_indices=source_indices, teacher_inputs=teacher,
        teacher_indices=targets,
    )
    with pytest.raises(RuntimeError, match="teacher indices"):
        context.require_identity(
            identity, fit_rows=rows, student_inputs=source,
            student_indices=source_indices, teacher_inputs=teacher,
            teacher_indices=tuple(reversed(targets)),
        )
    changed = teacher.clone(); changed[0, 0] += 1
    with pytest.raises(RuntimeError, match="teacher tokens"):
        context.require_identity(
            identity, fit_rows=rows, student_inputs=source,
            student_indices=source_indices, teacher_inputs=changed,
            teacher_indices=targets,
        )


class _Native(torch.nn.Module):
    def __init__(self, scale: float, offset: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.offset = torch.nn.Parameter(torch.tensor(offset))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state * self.scale + self.offset


def _call_hook(
    hook: runtime.StudentCorrectionHook, site: int, state: torch.Tensor,
    deployed: torch.Tensor, nonce: str,
) -> torch.Tensor:
    handle = runtime.mint_deployed_n_write(
        site=site, state=state, value=deployed, forward_nonce=nonce,
        issuer_id=hook.issuer_id,
    )
    return hook(site, state, handle, forward_nonce=nonce)


def _autonomous_forward(gateway, inputs):
    state0 = inputs.float().unsqueeze(-1).expand(-1, -1, runtime.D_MODEL) / 1000
    exact0 = gateway.call(0, state0)
    state1 = state0 + exact0
    exact1 = gateway.call(1, state1)
    return exact1[..., :11], {
        "outer_forward_count": 1, "hook_calls": {0: 1, 1: 1, 2: 0},
        "outer_returned": True, "hook_restored": True, "hook_inert": True,
    }


def test_mapped_oon_broker_executes_only_exact_plan_target_before_spending(monkeypatch) -> None:
    monkeypatch.setattr(capabilities, "FIT_ROW_COUNT", runtime.BATCH_SIZE)
    monkeypatch.setattr(capabilities, "FIT_BATCHES_PER_EPOCH", 1)
    rows = torch.arange(
        runtime.BATCH_SIZE * 513, dtype=torch.long,
    ).view(runtime.BATCH_SIZE, 513)
    plan = mapped.build_document_block_plan(
        _records((1, 1, 1, 1)), control="document_shuffle",
    )
    base = capabilities.RunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64,
        fit_role_tensor_sha256=runtime.tensor_identity_sha256(rows),
        identity_teacher_mapping_sha256="4" * 64,
        fit_row_count=runtime.BATCH_SIZE,
    )
    context = mapped.MappedRunContext(base=base, plan=plan)
    source_indices = tuple(int(value) for value in runtime.fit_permutations(4, 0)[0])
    target_indices = plan.target_indices(source_indices)
    source = rows[torch.tensor(source_indices), :runtime.SEQUENCE_LENGTH]
    teacher = rows[torch.tensor(target_indices), :runtime.SEQUENCE_LENGTH]
    program = _program("R")
    issuer = "5" * 64
    coordinator = runtime.ScopeCoordinator()
    projection = torch.eye(runtime.D_MODEL)[:, :runtime.CODE_DIM]
    native0, native1 = _Native(2.0, 0.25), _Native(-0.5, 1.0)
    broker = capabilities.CapabilityBroker(
        issuer_id=issuer, coordinator=coordinator, run_context=base,
        bases={0: projection, 1: projection},
        native_calls={0: native0, 1: native1}, mapped_authority=context,
    )
    hook = runtime.StudentCorrectionHook(
        {0: projection, 1: projection}, issuer_id=issuer, coordinator=coordinator,
    )
    hook.configure(program=program, states={0: "P", 1: "P"})
    identity = runtime.TraceIdentity.from_inputs(
        inputs=source, ordered_batch_indices=source_indices,
        source_commit=base.source_commit,
        inherited_snapshot_sha256=base.inherited_snapshot_sha256,
        rows_receipt_sha256=base.rows_receipt_sha256,
        fit_role_tensor_sha256=base.fit_role_tensor_sha256,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256=plan.sha256, phase="fit", route="R",
        control="document_shuffle", teacher_kind="oon_logits", trial=0,
        epoch=0, optimizer_step=0, batch_ordinal=0,
        student_states=((0, "P"), (1, "P"), (2, "N")),
    )

    session = broker.begin_student(identity, hook, source, source_indices)
    with session.forward_scope() as capability:
        state0 = source.float().unsqueeze(-1).expand(
            -1, -1, runtime.D_MODEL,
        ) / 1000
        out0 = _call_hook(hook, 0, state0, torch.zeros_like(state0), identity.nonce)
        state1 = state0 + out0
        out1 = _call_hook(hook, 1, state1, torch.zeros_like(state1), identity.nonce)
        student_logits = out1[..., :11]
        student_logits.retain_grad()
        capability.bind_outer_logits(student_logits)
    step, _ = session.close(
        outer_forward_count=1, outer_returned=True,
        hook_restored=True, hook_inert=True,
    )

    assert broker.ledger_snapshot.run_context_sha256 == context.sha256
    with pytest.raises(RuntimeError, match="mapped OON entry point"):
        broker.run_oon_teacher(identity, step, source, _autonomous_forward)
    changed = teacher.clone()
    changed[0, 0] += 1
    with pytest.raises(RuntimeError, match="teacher tokens"):
        broker.run_mapped_oon_teacher(
            identity, step, fit_rows=rows, student_inputs=source,
            student_indices=source_indices, teacher_inputs=changed,
            teacher_indices=target_indices, autonomous_forward=_autonomous_forward,
        )

    result = broker.run_mapped_oon_teacher(
        identity, step, fit_rows=rows, student_inputs=source,
        student_indices=source_indices, teacher_inputs=teacher,
        teacher_indices=target_indices, autonomous_forward=_autonomous_forward,
    )
    loss, closure = result.consume_loss()
    target_state0 = teacher.float().unsqueeze(-1).expand(
        -1, -1, runtime.D_MODEL,
    ) / 1000
    expected_teacher = native1(target_state0 + native0(target_state0))[..., :11]
    expected_loss = runtime.teacher_student_kl(expected_teacher.detach(), student_logits)
    torch.testing.assert_close(loss, expected_loss)
    assert closure.original_calls == capabilities.EXACT_EARLY_ORIGINAL_CALLS
    assert broker.ledger_snapshot.completed_identity_count == 1
    loss.backward()
    assert student_logits.grad is not None
    assert native0.scale.grad is None and native1.scale.grad is None


def test_mapped_coordinate_broker_labels_target_p_trajectory_only(monkeypatch) -> None:
    monkeypatch.setattr(capabilities, "FIT_ROW_COUNT", runtime.BATCH_SIZE)
    monkeypatch.setattr(capabilities, "FIT_BATCHES_PER_EPOCH", 1)
    rows = torch.arange(
        runtime.BATCH_SIZE * 513, dtype=torch.long,
    ).view(runtime.BATCH_SIZE, 513)
    plan = mapped.build_document_block_plan(
        _records((1, 1, 1, 1)), control="document_shuffle",
    )
    base = capabilities.RunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64,
        fit_role_tensor_sha256=runtime.tensor_identity_sha256(rows),
        identity_teacher_mapping_sha256="4" * 64,
        fit_row_count=runtime.BATCH_SIZE,
    )
    context = mapped.MappedRunContext(base=base, plan=plan)
    source_indices = tuple(int(value) for value in runtime.fit_permutations(4, 0)[0])
    target_indices = plan.target_indices(source_indices)
    source = rows[torch.tensor(source_indices), :runtime.SEQUENCE_LENGTH]
    teacher = rows[torch.tensor(target_indices), :runtime.SEQUENCE_LENGTH]
    program = _program("L")
    issuer = "7" * 64
    coordinator = runtime.ScopeCoordinator()
    projection = torch.eye(runtime.D_MODEL)[:, :runtime.CODE_DIM]
    native0, native1 = _Native(2.0, 0.25), _Native(-0.5, 1.0)
    broker = capabilities.CapabilityBroker(
        issuer_id=issuer, coordinator=coordinator, run_context=base,
        bases={0: projection, 1: projection},
        native_calls={0: native0, 1: native1}, mapped_authority=context,
    )
    hook = runtime.StudentCorrectionHook(
        {0: projection, 1: projection}, issuer_id=issuer, coordinator=coordinator,
    )
    hook.configure(program=program, states={0: "P", 1: "P"})
    identity = runtime.TraceIdentity.from_inputs(
        inputs=source, ordered_batch_indices=source_indices,
        source_commit=base.source_commit,
        inherited_snapshot_sha256=base.inherited_snapshot_sha256,
        rows_receipt_sha256=base.rows_receipt_sha256,
        fit_role_tensor_sha256=base.fit_role_tensor_sha256,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256=plan.sha256, phase="fit", route="L",
        control="document_shuffle", teacher_kind="coordinate_labels", trial=0,
        epoch=0, optimizer_step=0, batch_ordinal=0,
        student_states=((0, "P"), (1, "P"), (2, "N")),
    )
    source_state0 = source.float().unsqueeze(-1).expand(
        -1, -1, runtime.D_MODEL,
    ) / 1000
    session = broker.begin_student(identity, hook, source, source_indices)
    with session.forward_scope() as capability:
        source_out0 = _call_hook(
            hook, 0, source_state0, torch.zeros_like(source_state0), identity.nonce,
        )
        source_state1 = source_state0 + source_out0
        source_out1 = _call_hook(
            hook, 1, source_state1, torch.zeros_like(source_state1), identity.nonce,
        )
        capability.bind_outer_logits(source_out1[..., :11])
    step, _ = session.close(
        outer_forward_count=1, outer_returned=True,
        hook_restored=True, hook_inert=True,
    )

    saved_gateways = []

    def mapped_target_forward(gateway, inputs):
        saved_gateways.append(gateway)
        target_state0 = inputs.float().unsqueeze(-1).expand(
            -1, -1, runtime.D_MODEL,
        ) / 1000
        target_out0 = gateway.correct_and_label(
            0, target_state0, torch.zeros_like(target_state0),
        )
        target_state1 = target_state0 + target_out0
        gateway.correct_and_label(1, target_state1, torch.zeros_like(target_state1))
        return {
            "outer_forward_count": 1, "hook_calls": {0: 1, 1: 1, 2: 0},
            "outer_returned": True, "hook_restored": True, "hook_inert": True,
        }

    changed = teacher.clone()
    changed[0, 0] += 1
    with pytest.raises(RuntimeError, match="teacher tokens"):
        broker.run_mapped_coordinate_teacher(
            identity, step, fit_rows=rows, student_inputs=source,
            student_indices=source_indices, teacher_inputs=changed,
            teacher_indices=target_indices, program=program,
            autonomous_forward=mapped_target_forward,
        )
    result = broker.run_mapped_coordinate_teacher(
        identity, step, fit_rows=rows, student_inputs=source,
        student_indices=source_indices, teacher_inputs=teacher,
        teacher_indices=target_indices, program=program,
        autonomous_forward=mapped_target_forward,
    )
    loss, closure = result.consume_loss((1.0, 1.0))
    target_state0 = teacher.float().unsqueeze(-1).expand(
        -1, -1, runtime.D_MODEL,
    ) / 1000
    target_state1 = target_state0 + runtime.JointAffineProgram.projected_replacement(
        torch.zeros_like(target_state0), program.site0_code(target_state0), projection,
    )
    labels = (
        runtime.scored_positions(native0(target_state0).detach() @ projection),
        runtime.scored_positions(native1(target_state1).detach() @ projection),
    )
    predictions = (
        program.site0_code(source_state0), program.site1_code(source_state1),
    )
    expected = runtime.normalized_local_loss(predictions, labels, (1.0, 1.0))
    torch.testing.assert_close(loss, expected)
    assert closure.scope == "mapped_coordinate" and closure.outer_forward_count == 1
    assert closure.original_calls == capabilities.EXACT_EARLY_ORIGINAL_CALLS
    with pytest.raises(RuntimeError, match="revoked"):
        saved_gateways[0].correct_and_label(
            0, target_state0, torch.zeros_like(target_state0),
        )
    loss.backward()
    assert program.site0.weight.grad is not None and program.site1.weight.grad is not None
    assert native0.scale.grad is None and native1.scale.grad is None


def test_ordinary_broker_cannot_be_repurposed_for_mapped_execution() -> None:
    base = capabilities.RunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, fit_role_tensor_sha256="4" * 64,
        identity_teacher_mapping_sha256="5" * 64,
    )
    projection = torch.eye(runtime.D_MODEL)[:, :runtime.CODE_DIM]
    broker = capabilities.CapabilityBroker(
        issuer_id="6" * 64, coordinator=runtime.ScopeCoordinator(),
        run_context=base, bases={0: projection, 1: projection},
        native_calls={0: _Native(1.0, 0.0), 1: _Native(1.0, 0.0)},
    )
    with pytest.raises(RuntimeError, match="ordinary broker"):
        broker.run_mapped_oon_teacher(
            None, None, fit_rows=None, student_inputs=None,
            student_indices=(), teacher_inputs=None, teacher_indices=(),
            autonomous_forward=_autonomous_forward,
        )


def test_a_null_uses_false_parent_only_for_cross_and_true_source_oon(monkeypatch) -> None:
    monkeypatch.setattr(capabilities, "FIT_ROW_COUNT", runtime.BATCH_SIZE)
    monkeypatch.setattr(capabilities, "FIT_BATCHES_PER_EPOCH", 1)
    rows = torch.arange(
        runtime.BATCH_SIZE * 513, dtype=torch.long,
    ).view(runtime.BATCH_SIZE, 513)
    plan = mapped.build_document_block_plan(
        _records((1, 1, 1, 1)), control="A_null_00",
    )
    base = capabilities.RunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64,
        fit_role_tensor_sha256=runtime.tensor_identity_sha256(rows),
        identity_teacher_mapping_sha256="4" * 64,
        fit_row_count=runtime.BATCH_SIZE,
    )
    context = mapped.MappedRunContext(base=base, plan=plan)
    source_indices = tuple(int(value) for value in runtime.fit_permutations(4, 0)[0])
    target_indices = plan.target_indices(source_indices)
    source = rows[torch.tensor(source_indices), :runtime.SEQUENCE_LENGTH]
    target = rows[torch.tensor(target_indices), :runtime.SEQUENCE_LENGTH]
    program = _program("T")
    with torch.no_grad():
        program.site0.weight[:runtime.CODE_DIM].copy_(torch.eye(runtime.CODE_DIM))
        program.cross.copy_(torch.eye(runtime.CODE_DIM))
    issuer = "8" * 64
    coordinator = runtime.ScopeCoordinator()
    projection = torch.eye(runtime.D_MODEL)[:, :runtime.CODE_DIM]
    native0, native1 = _Native(2.0, 0.25), _Native(-0.5, 1.0)
    broker = capabilities.CapabilityBroker(
        issuer_id=issuer, coordinator=coordinator, run_context=base,
        bases={0: projection, 1: projection},
        native_calls={0: native0, 1: native1}, mapped_authority=context,
    )
    hook = runtime.StudentCorrectionHook(
        {0: projection, 1: projection}, issuer_id=issuer, coordinator=coordinator,
    )
    identity = runtime.TraceIdentity.from_inputs(
        inputs=source, ordered_batch_indices=source_indices,
        source_commit=base.source_commit,
        inherited_snapshot_sha256=base.inherited_snapshot_sha256,
        rows_receipt_sha256=base.rows_receipt_sha256,
        fit_role_tensor_sha256=base.fit_role_tensor_sha256,
        program_snapshot_sha256=runtime.program_snapshot_sha256(program),
        teacher_mapping_sha256=plan.sha256, phase="fit", route="T",
        control="A_null_00", teacher_kind="oon_logits", trial=0,
        epoch=0, optimizer_step=0, batch_ordinal=0,
        student_states=((0, "P"), (1, "P"), (2, "N")),
    )

    saved_gateways = []

    def parent_forward(gateway, inputs):
        saved_gateways.append(gateway)
        state0 = inputs.float().unsqueeze(-1).expand(
            -1, -1, runtime.D_MODEL,
        ) / 1000
        out0 = gateway.correct(0, state0, torch.zeros_like(state0))
        state1 = state0 + out0
        gateway.correct(1, state1, torch.zeros_like(state1))
        return {
            "outer_forward_count": 1, "hook_calls": {0: 1, 1: 1, 2: 0},
            "outer_returned": True, "hook_restored": True, "hook_inert": True,
        }

    changed = target.clone(); changed[0, 0] += 1
    with pytest.raises(RuntimeError, match="teacher tokens"):
        broker.prepare_mapped_parent(
            identity, fit_rows=rows, student_inputs=source,
            student_indices=source_indices, teacher_inputs=changed,
            teacher_indices=target_indices, program=program,
            autonomous_forward=parent_forward,
        )
    parent, parent_closure = broker.prepare_mapped_parent(
        identity, fit_rows=rows, student_inputs=source,
        student_indices=source_indices, teacher_inputs=target,
        teacher_indices=target_indices, program=program,
        autonomous_forward=parent_forward,
    )
    prepared = broker.ledger_snapshot
    assert prepared.prepared_parent_identity_count == 1
    assert prepared.consumed_parent_identity_count == 0
    assert prepared.outstanding_parent_identity_sha256 == identity.sha256
    assert parent_closure.scope == "mapped_parent"
    assert parent_closure.original_calls == capabilities.EXACT_ZERO_CALLS
    hook.configure(program=program, states={0: "P", 1: "P"}, mapped_parent=parent)
    consumed = broker.ledger_snapshot
    assert consumed.consumed_parent_identity_count == 1
    assert consumed.outstanding_parent_identity_sha256 is None

    source_state0 = source.float().unsqueeze(-1).expand(
        -1, -1, runtime.D_MODEL,
    ) / 1000
    target_state0 = target.float().unsqueeze(-1).expand(
        -1, -1, runtime.D_MODEL,
    ) / 1000
    session = broker.begin_student(identity, hook, source, source_indices)
    with session.forward_scope() as capability:
        source_out0 = _call_hook(
            hook, 0, source_state0, torch.zeros_like(source_state0), identity.nonce,
        )
        source_state1 = source_state0 + source_out0
        source_out1 = _call_hook(
            hook, 1, source_state1, torch.zeros_like(source_state1), identity.nonce,
        )
        student_logits = source_out1[..., :11]
        capability.bind_outer_logits(student_logits)
    step, _ = session.close(
        outer_forward_count=1, outer_returned=True,
        hook_restored=True, hook_inert=True,
    )
    source_code = program.site0_code(source_state0)
    false_parent_code = program.site0_code(target_state0)
    torch.testing.assert_close(source_out0 @ projection, source_code)
    torch.testing.assert_close(source_out1 @ projection, false_parent_code)
    assert not torch.equal(source_code, false_parent_code)

    result = broker.run_a_null_oon_teacher(
        identity, step, fit_rows=rows, student_inputs=source,
        student_indices=source_indices, teacher_inputs=target,
        teacher_indices=target_indices, autonomous_forward=_autonomous_forward,
    )
    loss, closure = result.consume_loss()
    expected_teacher = native1(source_state0 + native0(source_state0))[..., :11]
    torch.testing.assert_close(
        loss, runtime.teacher_student_kl(expected_teacher.detach(), student_logits),
    )
    assert closure.original_calls == capabilities.EXACT_EARLY_ORIGINAL_CALLS
    loss.backward()
    assert program.cross.grad is not None
    assert program.site0.weight.grad is None and program.site1.weight.grad is None
    assert native0.scale.grad is None and native1.scale.grad is None
    with pytest.raises(RuntimeError, match="revoked"):
        saved_gateways[-1].correct(0, target_state0, torch.zeros_like(target_state0))
    with pytest.raises(RuntimeError, match="duplicated"):
        broker.prepare_mapped_parent(
            identity, fit_rows=rows, student_inputs=source,
            student_indices=source_indices, teacher_inputs=target,
            teacher_indices=target_indices, program=program,
            autonomous_forward=parent_forward,
        )
