from types import SimpleNamespace
import hashlib
from pathlib import Path

import pytest
import torch

import bilin18_observed_adapter as observed
import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_final_actions as final_actions
import early_mlp_suffix_transport_v1_programs as programs
import early_mlp_suffix_transport_v1_response_execution as execution
import early_mlp_suffix_transport_v1_response_plan as response_plan
import early_mlp_suffix_transport_v1_response_reductions as reductions
import early_mlp_suffix_transport_v1_runtime as runtime


def _geometry() -> programs.TransportInterventionGeometry:
    identity = torch.eye(runtime.CODE_DIM, dtype=torch.float64)
    signs = torch.ones(32, runtime.CODE_DIM, dtype=torch.long)
    signs[1::2].neg_()
    return programs.TransportInterventionGeometry(
        selected_l_program_sha256="1" * 64,
        fit_role_tensor_sha256="2" * 64,
        code_trajectory_sha256="3" * 64,
        code_count=capabilities.FIT_ROW_COUNT * (
            runtime.SCORE_STOP - runtime.SCORE_START
        ),
        mean=torch.zeros(runtime.CODE_DIM, dtype=torch.float64),
        covariance=identity.clone(),
        eigenvalues=torch.ones(runtime.CODE_DIM, dtype=torch.float64),
        eigenvectors=identity.clone(),
        clipped_eigenvalues=torch.ones(runtime.CODE_DIM, dtype=torch.float64),
        clip_floor=1e-12,
        natural_rms=2.0,
        raw_rademacher_signs=signs,
        normalized_directions=signs.double(),
    )


def _program_bank():
    calibration = programs.select_teacher_calibration({
        0.01: 0.011, 0.03: 0.04, 0.1: 0.11, 0.3: 0.30, 1.0: 1.0,
    })
    return {
        "transport_geometry": _geometry(),
        "teacher_calibration": calibration,
        "payload_sha256": "4" * 64,
    }


def _rows():
    return torch.arange(4 * 513, dtype=torch.long).reshape(4, 513).contiguous()


def _basis():
    return torch.eye(runtime.D_MODEL, dtype=torch.float32)[:, :runtime.CODE_DIM].contiguous()


def _materialized_ll():
    def site():
        return runtime.AffineCodeProgram(
            mean=torch.zeros(runtime.D_MODEL), scale=torch.ones(runtime.D_MODEL),
            weight=torch.zeros(runtime.D_MODEL, runtime.CODE_DIM),
            bias=torch.zeros(runtime.CODE_DIM),
        )
    program = runtime.JointAffineProgram(site(), site(), route="L")
    return final_actions.MaterializedFinalAction(
        plan=final_actions.plan_for("ll", "N"), source_bank_sha256="5" * 64,
        component_sha256s={"test": "6" * 64}, program=program,
    )


def _route_program(route):
    def site():
        return runtime.AffineCodeProgram(
            mean=torch.zeros(runtime.D_MODEL), scale=torch.ones(runtime.D_MODEL),
            weight=torch.zeros(runtime.D_MODEL, runtime.CODE_DIM),
            bias=torch.zeros(runtime.CODE_DIM),
        )
    return runtime.JointAffineProgram(site(), site(), route=route)


def _source_bank():
    routes = {
        "inherited_q": "L", "true/L": "L", "true/R": "R",
        "true/S0": "S0", "true/S1": "S1", "true/T": "T",
        "mapped/document_shuffle/L": "L",
        "mapped/document_shuffle/R": "R",
        **{f"mapped/A_null_{index:02d}/T": "T" for index in range(20)},
        "new_fit_mean": "L",
    }
    return final_actions.FinalProgramSourceBank({
        key: _route_program(routes[key]) for key in final_actions.SOURCE_PROGRAM_KEYS
    })


def _closure(scope, *, teacher):
    return observed.ObservedClosure(
        scope=scope, outer_forward_count=1, outer_returned=True,
        attention_dispatch_calls=tuple((site, 1) for site in range(18)),
        mlp_dispatch_calls=tuple((site, 1) for site in range(18)),
        deployed_n_calls=(
            ((0, 0), (1, 0), (2, 1)) if teacher
            else ((0, 1), (1, 1), (2, 1))
        ),
        correction_calls=(
            ((0, 0), (1, 0), (2, 0)) if teacher
            else ((0, 1), (1, 1), (2, 0))
        ),
        literal_early_mlp_calls=(
            ((0, 1), (1, 1), (2, 0)) if teacher
            else ((0, 0), (1, 0), (2, 0))
        ),
        native_guard_restored=True, native_guard_inert=True,
        logit_shape=(4, 192, 3), logit_dtype="torch.float32",
    )


def test_authority_derives_antithetic_code_and_matching_physical_edits():
    rows = _rows()
    basis = _basis()
    edits = {
        sign: execution.build_final_response_edit(
            validated_program_bank=_program_bank(), role_rows=rows,
            ordered_batch_indices=(0, 1, 2, 3), batch_ordinal=0,
            basis0=basis, edit_sign=sign,
        ) for sign in execution.EDIT_SIGNS
    }
    assert len({value.unit_identity_sha256 for value in edits.values()}) == 1
    assert torch.equal(edits[-1].code_edit, -edits[1].code_edit)
    assert torch.equal(edits[-1].physical_edit, -edits[1].physical_edit)
    assert edits[1].semantic_delta.dtype == torch.float64
    assert torch.equal(edits[-1].semantic_delta, -edits[1].semantic_delta)
    assert not bool(edits[0].code_edit.any())
    assert not bool(edits[0].physical_edit.any())
    assert torch.equal(edits[1].physical_edit, edits[1].code_edit @ basis.T)
    assignments = programs.intervention_assignments("final")
    assert torch.equal(edits[1].positions, assignments["positions"][:4])
    assert torch.equal(
        edits[1].direction_indices, assignments["direction_indices"][:4],
    )
    # The selected multiplier is .03 and natural RMS is 2.
    magnitude = edits[1].code_edit[
        torch.arange(4), edits[1].positions,
    ].abs().mean(dim=1)
    torch.testing.assert_close(magnitude, torch.full((4,), 0.06))


def test_response_execution_amendment_hash_is_pinned():
    path = Path(__file__).with_name(
        "EARLY_MLP_SUFFIX_TRANSPORT_V1_RESPONSE_EXECUTION_AMENDMENT.md"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        execution.RESPONSE_EXECUTION_AMENDMENT_SHA256
    )


def test_edit_rejects_wrong_rows_basis_schedule_and_post_mint_mutation():
    rows = _rows()
    basis = _basis()
    edit = execution.build_final_response_edit(
        validated_program_bank=_program_bank(), role_rows=rows,
        ordered_batch_indices=(0, 1, 2, 3), batch_ordinal=0,
        basis0=basis, edit_sign=1,
    )
    changed_rows = rows.clone()
    changed_rows[0, -1] += 1
    with pytest.raises(RuntimeError, match="rows or schedule"):
        edit.require_pristine(
            role_rows=changed_rows, ordered_batch_indices=(0, 1, 2, 3),
            basis0=basis,
        )
    changed_basis = basis.roll(1, dims=0)
    with pytest.raises(RuntimeError, match="basis changed"):
        edit.require_pristine(
            role_rows=rows, ordered_batch_indices=(0, 1, 2, 3),
            basis0=changed_basis,
        )
    edit.code_edit[0, edit.positions[0], 0] += 1
    with pytest.raises(RuntimeError, match="mutated"):
        edit.require_pristine(
            role_rows=rows, ordered_batch_indices=(0, 1, 2, 3),
            basis0=basis,
        )
    with pytest.raises(ValueError, match="canonical"):
        execution.build_final_response_edit(
            validated_program_bank=_program_bank(), role_rows=rows,
            ordered_batch_indices=(4, 5, 6, 7), batch_ordinal=0,
            basis0=basis, edit_sign=1,
        )


def test_response_runtime_nonce_binds_perturbation_and_actual_edit():
    rows = _rows()
    basis = _basis()
    materialized = _materialized_ll()
    identity = final_actions.FinalActionBatchIdentity.from_role_rows(
        materialized=materialized, role_rows=rows,
        ordered_batch_indices=(0, 1, 2, 3), batch_ordinal=0,
        source_commit="7" * 40, inherited_snapshot_sha256="8" * 64,
        rows_receipt_sha256="9" * 64, final_role_tensor_sha256="a" * 64,
        program_payload_sha256="4" * 64, common_support_sha256="b" * 64,
    )
    edits = {
        sign: execution.build_final_response_edit(
            validated_program_bank=_program_bank(), role_rows=rows,
            ordered_batch_indices=(0, 1, 2, 3), batch_ordinal=0,
            basis0=basis, edit_sign=sign,
        ) for sign in execution.EDIT_SIGNS
    }
    plan = response_plan.build_response_batch_plan(
        batch_ordinal=0,
        ordered_role_rows_sha256=runtime.tensor_identity_sha256(rows),
        intervention_unit_sha256=edits[0].unit_identity_sha256,
    )
    ll = {
        forward.perturbation: forward for forward in plan.forwards
        if forward.subject_key == "ll/N"
    }
    bindings = {
        perturbation: execution.bind_runtime_response_program_batch(
            materialized=materialized, final_action_identity=identity,
            forward_plan=ll[perturbation],
            edit=edits[{"baseline": 0, "positive": 1, "negative": -1}[perturbation]],
            role_rows=rows, ordered_batch_indices=(0, 1, 2, 3),
            teacher_mapping_sha256="c" * 64,
        ) for perturbation in response_plan.PERTURBATIONS
    }
    assert [bindings[name].runtime_identity.trial for name in response_plan.PERTURBATIONS] == [
        0, 1, 2,
    ]
    assert len({value.runtime_identity.sha256 for value in bindings.values()}) == 3
    assert len({value.execution_identity.sha256 for value in bindings.values()}) == 3
    assert all(
        value.execution_identity.response_execution_amendment_sha256
        == execution.RESPONSE_EXECUTION_AMENDMENT_SHA256
        for value in bindings.values()
    )
    with pytest.raises(RuntimeError, match="bindings disagree"):
        execution.bind_runtime_response_program_batch(
            materialized=materialized, final_action_identity=identity,
            forward_plan=ll["positive"], edit=edits[-1], role_rows=rows,
            ordered_batch_indices=(0, 1, 2, 3), teacher_mapping_sha256="c" * 64,
        )


class _Native(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, state):
        return state * self.scale


class _TeacherModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.transformer = SimpleNamespace(h=[
            SimpleNamespace(mlp=_Native(0.0 if site == 0 else 2.0))
            for site in range(18)
        ])


class _Ship:
    production = False

    def attention(self, event):
        return torch.zeros_like(event.state)

    def mlp(self, event):
        return torch.zeros_like(event.state)


def test_exact_teacher_applies_physical_edit_and_captures_raw_mlp1_write(monkeypatch):
    model = _TeacherModel()
    adapter = observed.ObservedBilin18Adapter(model, _Ship(), production=False)
    rows = _rows()
    basis = _basis()
    edit = execution.build_final_response_edit(
        validated_program_bank=_program_bank(), role_rows=rows,
        ordered_batch_indices=(0, 1, 2, 3), batch_ordinal=0,
        basis0=basis, edit_sign=1,
    )

    def fake_forward(model, tokens, attention, mlp, *, require_production):
        assert tuple(tokens.shape) == (runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH)
        state = torch.zeros(
            runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.D_MODEL,
        )
        writes = []
        for site, block in enumerate(model.transformer.h):
            event = SimpleNamespace(
                site=site, block=block, state=state,
                first_value=None, prior_writes=tuple(writes), tokens=tokens,
            )
            state = state + attention(event)
            event.state = state
            write = mlp(event)
            writes.append(write)
            state = state + write
        return state[..., :3].contiguous()

    monkeypatch.setattr(observed.facade, "forward_with_dispatch", fake_forward)
    result = adapter._run_final_response_teacher_forward(
        edit=edit, role_rows=rows, ordered_batch_indices=(0, 1, 2, 3),
        basis0=basis, basis1=basis,
    )
    expected_code = 2 * edit.code_edit[:, runtime.SCORE_START:]
    expected_logits = 3 * edit.code_edit[:, runtime.SCORE_START:, :3]
    torch.testing.assert_close(result.code1, expected_code)
    torch.testing.assert_close(result.logits, expected_logits)
    assert result.edit_sha256 == edit.sha256
    assert result.unit_identity_sha256 == edit.unit_identity_sha256
    assert result.observed_closure.deployed_n_calls == ((0, 0), (1, 0), (2, 1))
    assert result.observed_closure.literal_early_mlp_calls == (
        (0, 1), (1, 1), (2, 0),
    )
    assert result.observed_closure.native_guard_restored
    assert result.observed_closure.native_guard_inert


def test_response_student_uses_distinct_bound_trace_and_consumes_outputs(monkeypatch):
    model = _TeacherModel()
    adapter = observed.ObservedBilin18Adapter(model, _Ship(), production=False)
    rows = _rows()
    basis = _basis()
    materialized = _materialized_ll()
    final_context = capabilities.FinalRunContext(
        source_commit="7" * 40, inherited_snapshot_sha256="8" * 64,
        rows_receipt_sha256="9" * 64, final_role_tensor_sha256="a" * 64,
        identity_teacher_mapping_sha256="c" * 64,
    )
    identity = final_actions.FinalActionBatchIdentity.from_role_rows(
        materialized=materialized, role_rows=rows,
        ordered_batch_indices=(0, 1, 2, 3), batch_ordinal=0,
        source_commit=final_context.source_commit,
        inherited_snapshot_sha256=final_context.inherited_snapshot_sha256,
        rows_receipt_sha256=final_context.rows_receipt_sha256,
        final_role_tensor_sha256=final_context.final_role_tensor_sha256,
        program_payload_sha256="4" * 64, common_support_sha256="b" * 64,
    )
    edit = execution.build_final_response_edit(
        validated_program_bank=_program_bank(), role_rows=rows,
        ordered_batch_indices=(0, 1, 2, 3), batch_ordinal=0,
        basis0=basis, edit_sign=1,
    )
    plan = response_plan.build_response_batch_plan(
        batch_ordinal=0,
        ordered_role_rows_sha256=runtime.tensor_identity_sha256(rows),
        intervention_unit_sha256=edit.unit_identity_sha256,
    )
    positive = next(
        item for item in plan.forwards
        if item.subject_key == "ll/N" and item.perturbation == "positive"
    )
    binding = execution.bind_runtime_response_program_batch(
        materialized=materialized, final_action_identity=identity,
        forward_plan=positive, edit=edit, role_rows=rows,
        ordered_batch_indices=(0, 1, 2, 3),
        teacher_mapping_sha256=final_context.identity_teacher_mapping_sha256,
    )
    coordinator = runtime.ScopeCoordinator()
    hook = runtime.StudentCorrectionHook(
        {0: basis, 1: basis}, issuer_id="d" * 64, coordinator=coordinator,
    )
    broker = adapter.make_capability_broker(
        issuer_id="d" * 64, coordinator=coordinator,
        run_context=final_context, bases={0: basis, 1: basis},
    )

    def fake_forward(model, tokens, attention, mlp, *, require_production):
        state = torch.zeros(
            runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.D_MODEL,
        )
        writes = []
        for site, block in enumerate(model.transformer.h):
            event = SimpleNamespace(
                site=site, block=block, state=state,
                first_value=None, prior_writes=tuple(writes), tokens=tokens,
            )
            state = state + attention(event)
            event.state = state
            write = mlp(event)
            writes.append(write)
            state = state + write
        return state[..., :3].contiguous()

    monkeypatch.setattr(observed.facade, "forward_with_dispatch", fake_forward)
    result = adapter._run_final_response_student_forward(
        broker=broker, hook=hook, materialized=materialized,
        final_action_identity=identity, binding=binding, edit=edit,
        final_context=final_context, role_rows=rows,
        ordered_batch_indices=(0, 1, 2, 3), basis0=basis,
    )
    assert result.response_execution_identity_sha256 == binding.execution_identity.sha256
    assert result.edit_sha256 == edit.sha256
    assert result.unit_identity_sha256 == edit.unit_identity_sha256
    assert tuple(result.code1.shape) == (4, 192, 64)
    assert tuple(result.logits.shape) == (4, 192, 3)
    assert result.observed_closure.deployed_n_calls == ((0, 1), (1, 1), (2, 1))
    assert broker.ledger_snapshot.completed_identity_count == 1
    assert broker.ledger_snapshot.outstanding_identity_sha256 is None
    assert coordinator.idle


def test_atomic_batch_routes_69_forwards_and_binds_actual_receipt_triplets(monkeypatch):
    model = _TeacherModel()
    adapter = observed.ObservedBilin18Adapter(model, _Ship(), production=False)
    rows = _rows()
    basis = _basis()
    sources = _source_bank()
    final_context = capabilities.FinalRunContext(
        source_commit="7" * 40, inherited_snapshot_sha256="8" * 64,
        rows_receipt_sha256="9" * 64, final_role_tensor_sha256="a" * 64,
        identity_teacher_mapping_sha256="c" * 64,
    )

    class Inherited:
        authority = SimpleNamespace(snapshot_sha256=final_context.inherited_snapshot_sha256)

        def clone_bases(self):
            return {0: basis.clone(), 1: basis.clone()}

        def make_program(self, route):
            assert route == "L"
            return sources.clone("inherited_q")

    monkeypatch.setattr(
        observed.final_actions, "source_bank_from_validated",
        lambda validated, *, inherited_q: sources,
    )

    class FakeBroker:
        def __init__(self):
            self.count = 0

        @property
        def ledger_snapshot(self):
            count = self.count
            digest = runtime.logical_identity_sha256(list(range(count)))
            return capabilities.LedgerSnapshot(
                run_context_sha256=final_context.sha256,
                student_identity_count=count, teacher_identity_count=count,
                completed_identity_count=count,
                student_identities_sha256=digest,
                teacher_identities_sha256=digest,
                completed_identities_sha256=digest,
                prepared_parent_identity_count=0,
                prepared_parent_identities_sha256="0" * 64,
                consumed_parent_identity_count=0,
                consumed_parent_identities_sha256="0" * 64,
                outstanding_parent_identity_sha256=None,
                outstanding_identity_sha256=None,
                rolling_ledger_sha256=digest,
            )

    fake_broker = FakeBroker()
    monkeypatch.setattr(
        observed.ObservedBilin18Adapter, "make_capability_broker",
        lambda self, **kwargs: fake_broker,
    )
    teacher_plans = []
    student_plans = []

    def fake_teacher(self, *, edit, **kwargs):
        teacher_plans.append(edit.edit_sign)
        code = edit.code_edit[:, runtime.SCORE_START:].clone()
        logits = code[..., :3].clone()
        return observed._ObservedResponseTeacherForward(
            code1=code, logits=logits, edit_sha256=edit.sha256,
            unit_identity_sha256=edit.unit_identity_sha256,
            observed_closure=_closure("teacher", teacher=True),
        )

    def fake_student(self, *, broker, binding, edit, **kwargs):
        broker.count += 1
        student_plans.append((
            binding.execution_identity.action_key,
            binding.execution_identity.perturbation,
        ))
        code = 0.5 * edit.code_edit[:, runtime.SCORE_START:]
        logits = code[..., :3].clone()
        return observed._ObservedResponseStudentForward(
            code1=code, logits=logits,
            response_execution_identity_sha256=binding.execution_identity.sha256,
            edit_sha256=edit.sha256,
            unit_identity_sha256=edit.unit_identity_sha256,
            student_step_ledger_sha256=runtime.logical_identity_sha256({
                "student": broker.count,
            }),
            consumer_ledger_sha256=runtime.logical_identity_sha256({
                "consumer": broker.count,
            }),
            broker_ledger_sha256=runtime.logical_identity_sha256({
                "broker": broker.count,
            }),
            observed_closure=_closure("student", teacher=False),
        )

    monkeypatch.setattr(
        observed.ObservedBilin18Adapter,
        "_run_final_response_teacher_forward", fake_teacher,
    )
    monkeypatch.setattr(
        observed.ObservedBilin18Adapter,
        "_run_final_response_student_forward", fake_student,
    )
    result = adapter.run_final_response_batch(
        validated_program_bank=_program_bank(),
        inherited_initialization=Inherited(), final_context=final_context,
        role_rows=rows, ordered_batch_indices=(0, 1, 2, 3), batch_ordinal=0,
    )
    assert teacher_plans == [0, 1, -1]
    assert student_plans == [
        (action, perturbation)
        for action in response_plan.RESPONSE_ACTION_KEYS
        for perturbation in response_plan.PERTURBATIONS
    ]
    assert len(result.receipt.forward_receipt_sha256s) == 69
    assert result.receipt.teacher_forward_count == 3
    assert result.receipt.student_forward_count == 66
    assert result.receipt.atomic_complete
    assert len(result.arm_reductions) == 22
    assert result.arm_reductions[0].action_key == "ll/N"
    assert result.arm_reductions[1].action_key == "lt/N"
    assert result.arm_reductions[2].code_response is None
    # A half-sized student response gives NRE=.5 and therefore R2=.75.
    first = result.arm_reductions[0].code_response
    assert first is not None
    pooled_error = first.error_sum.sum() / first.teacher_sum.sum()
    assert float(pooled_error) == pytest.approx(0.25)
    teacher_receipts = result.receipt.forward_receipt_sha256s[:3]
    assert all(
        reduction.teacher_forward_receipt_sha256s == teacher_receipts
        for reduction in result.arm_reductions
    )
    assert len({
        receipt for reduction in result.arm_reductions
        for receipt in reduction.student_forward_receipt_sha256s
    }) == 66


def _synthetic_batch_result(batch_ordinal, *, final_context_sha256="3" * 64):
    unit = runtime.logical_identity_sha256({"unit": batch_ordinal})
    batch_plan = runtime.logical_identity_sha256({"plan": batch_ordinal})
    forwards = tuple(
        runtime.logical_identity_sha256({"batch": batch_ordinal, "forward": index})
        for index in range(69)
    )
    arm_reductions = []
    offset = 3
    for action_key in response_plan.RESPONSE_ACTION_KEYS:
        vector = reductions.BatchResponseReduction(
            error_sum=torch.ones(4, dtype=torch.float64),
            teacher_sum=torch.full((4,), 4.0, dtype=torch.float64),
            student_sum=torch.ones(4, dtype=torch.float64),
            dot_sum=torch.full((4,), 2.0, dtype=torch.float64),
            unit_identity=unit,
        )
        output_kl = reductions.BatchOutputKLReduction(
            numerator_sum=torch.ones(4, dtype=torch.float64),
            denominator_sum=torch.full((4,), 2.0, dtype=torch.float64),
            unit_identity=unit,
        )
        arm_reductions.append(execution.ObservedResponseArmReduction(
            action_key=action_key, batch_plan_sha256=batch_plan,
            teacher_forward_receipt_sha256s=forwards[:3],
            student_forward_receipt_sha256s=forwards[offset:offset + 3],
            code_response=vector if action_key in {"ll/N", "lt/N"} else None,
            logit_response=vector, output_kl_response=output_kl,
        ))
        offset += 3
    receipt = execution.ObservedResponseBatchReceipt(
        batch_ordinal=batch_ordinal, batch_plan_sha256=batch_plan,
        source_bank_sha256="1" * 64, program_payload_sha256="2" * 64,
        final_context_sha256=final_context_sha256,
        common_support_sha256="4" * 64,
        basis0_sha256="5" * 64, basis1_sha256="6" * 64,
        forward_receipt_sha256s=forwards,
        arm_reduction_sha256s=tuple(
            (value.action_key, value.sha256) for value in arm_reductions
        ),
        broker_ledger_sha256=runtime.logical_identity_sha256({
            "broker": batch_ordinal,
        }),
        teacher_forward_count=3, student_forward_count=66,
        atomic_complete=True,
    )
    return execution.ObservedResponseBatchResult(
        arm_reductions=tuple(arm_reductions), receipt=receipt,
    )


def test_run_accumulator_requires_48_canonical_batches_and_emits_exact_ledger():
    accumulator = execution.ObservedResponseRunAccumulator()
    with pytest.raises(RuntimeError, match="incomplete"):
        accumulator.finish()
    with pytest.raises(RuntimeError, match="canonical order"):
        accumulator.add(_synthetic_batch_result(1))
    for batch_ordinal in range(48):
        accumulator.add(_synthetic_batch_result(batch_ordinal))
    assert accumulator.batch_count == 48
    result = accumulator.finish()
    assert accumulator.batch_count == 0
    assert result.receipt.teacher_forward_count == 144
    assert result.receipt.student_forward_count == 3168
    assert result.receipt.row_count == 192
    assert result.receipt.atomic_complete
    assert len(result.receipt.batch_receipt_sha256s) == 48
    assert len(set(result.receipt.batch_receipt_sha256s)) == 48
    assert tuple(value.action_key for value in result.arm_reductions) == (
        response_plan.RESPONSE_ACTION_KEYS
    )
    first = result.arm_reductions[0]
    assert first.code_response is not None
    assert first.code_response.error_sum.shape == (192,)
    torch.testing.assert_close(
        first.code_response.error_sum, torch.ones(192, dtype=torch.float64),
    )
    assert len(first.code_response.unit_identity_sha256s) == 48
    assert len(set(first.code_response.unit_identity_sha256s)) == 48
    assert result.arm_reductions[2].code_response is None
    payload = result.to_final_statistics_payload()
    assert payload["response_run_receipt_sha256"] == result.receipt.sha256
    assert len(payload["logit_nulls"]) == 20
    assert len(payload["output_kl_nulls"]) == 20
    identities = {
        value["unit_identity"] for value in (
            payload["code_baseline"], payload["code_candidate"],
            payload["logit_baseline"], payload["logit_candidate"],
            *payload["logit_nulls"], payload["output_kl_baseline"],
            payload["output_kl_candidate"], *payload["output_kl_nulls"],
        )
    }
    assert identities == {payload["ordered_unit_identity_sha256"]}
    payload["code_baseline"]["error_sum"].zero_()
    assert bool(first.code_response.error_sum.all())
    with pytest.raises(RuntimeError, match="already closed"):
        accumulator.add(_synthetic_batch_result(0))
    with pytest.raises(RuntimeError, match="already closed"):
        accumulator.finish()


def test_role_runner_owns_all_48_batches_and_rejects_changed_role(monkeypatch):
    model = _TeacherModel()
    adapter = observed.ObservedBilin18Adapter(model, _Ship(), production=False)
    final_rows = torch.arange(192 * 513, dtype=torch.long).reshape(
        192, 513,
    ).contiguous()
    final_context = capabilities.FinalRunContext(
        source_commit="7" * 40, inherited_snapshot_sha256="8" * 64,
        rows_receipt_sha256="9" * 64,
        final_role_tensor_sha256=runtime.tensor_identity_sha256(final_rows),
        identity_teacher_mapping_sha256="c" * 64,
    )
    calls = []

    def fake_batch(self, *, role_rows, ordered_batch_indices, batch_ordinal, **kwargs):
        start = batch_ordinal * 4
        assert ordered_batch_indices == tuple(range(start, start + 4))
        assert torch.equal(role_rows, final_rows[start:start + 4])
        calls.append(batch_ordinal)
        return _synthetic_batch_result(
            batch_ordinal, final_context_sha256=final_context.sha256,
        )

    monkeypatch.setattr(
        observed.ObservedBilin18Adapter, "run_final_response_batch", fake_batch,
    )
    before = final_rows.clone()
    result = adapter.run_final_response_role(
        validated_program_bank=object(), inherited_initialization=object(),
        final_context=final_context, final_rows=final_rows,
    )
    assert calls == list(range(48))
    assert result.receipt.teacher_forward_count == 144
    assert result.receipt.student_forward_count == 3168
    assert torch.equal(final_rows, before)
    changed_context = capabilities.FinalRunContext(
        source_commit="7" * 40, inherited_snapshot_sha256="8" * 64,
        rows_receipt_sha256="9" * 64, final_role_tensor_sha256="d" * 64,
        identity_teacher_mapping_sha256="c" * 64,
    )
    with pytest.raises(RuntimeError, match="role authority"):
        adapter.run_final_response_role(
            validated_program_bank=object(), inherited_initialization=object(),
            final_context=changed_context, final_rows=final_rows,
        )
