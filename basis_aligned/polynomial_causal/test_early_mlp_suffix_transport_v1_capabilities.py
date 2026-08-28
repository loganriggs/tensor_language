from __future__ import annotations

import copy

import pytest
import torch

import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_runtime as runtime


def basis() -> torch.Tensor:
    return torch.eye(runtime.D_MODEL, dtype=torch.float32)[:, :runtime.CODE_DIM]


def state(seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "grammar": "affine", "interface": "state_complete_p",
        "mean": torch.zeros(runtime.D_MODEL), "scale": torch.ones(runtime.D_MODEL),
        "left": torch.randn(
            runtime.D_MODEL, runtime.CODE_DIM, generator=generator,
        ) / 50,
        "right": torch.randn(
            runtime.CODE_DIM, runtime.CODE_DIM, generator=generator,
        ) / 50,
        "bias": torch.randn(runtime.CODE_DIM, generator=generator) / 50,
    }


def program(route: str) -> runtime.JointAffineProgram:
    return runtime.JointAffineProgram.from_v21_states(
        {0: state(1), 1: state(2)}, route=route,
    )


class Native(torch.nn.Module):
    def __init__(self, scale: float, offset: float) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))
        self.offset = torch.nn.Parameter(torch.tensor(float(offset)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.scale + self.offset


def identity(
    hook: runtime.StudentCorrectionHook, inputs: torch.Tensor, *, step: int,
    route: str, teacher_kind: str, control: str = "true", phase: str = "fit",
) -> runtime.TraceIdentity:
    if phase == "initial_denominator":
        indices = tuple(range(step * runtime.BATCH_SIZE, (step + 1) * runtime.BATCH_SIZE))
    else:
        indices = tuple(int(value) for value in runtime.fit_permutations(384, 0)[0][
            step * runtime.BATCH_SIZE:(step + 1) * runtime.BATCH_SIZE
        ])
    return runtime.TraceIdentity.from_inputs(
        inputs=inputs, ordered_batch_indices=indices, source_commit="1" * 40,
        inherited_snapshot_sha256="2" * 64, rows_receipt_sha256="3" * 64,
        fit_role_tensor_sha256="4" * 64,
        program_snapshot_sha256=runtime.program_snapshot_sha256(hook.program),
        teacher_mapping_sha256="5" * 64, phase=phase, route=route,
        control=control, teacher_kind=teacher_kind, trial=0, epoch=0,
        optimizer_step=0 if phase == "initial_denominator" else step,
        batch_ordinal=step,
        student_states=tuple((site, hook.states.get(site, "N")) for site in (0, 1))
        + ((2, "N"),),
    )


def system(route: str, *, step: int = 0):
    issuer = "a" * 64
    coordinator = runtime.ScopeCoordinator()
    native0, native1 = Native(2.0, 0.25), Native(-0.5, 1.0)
    broker = capabilities.CapabilityBroker(
        issuer_id=issuer, coordinator=coordinator,
        run_context=capabilities.RunContext(
            source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
            rows_receipt_sha256="3" * 64, fit_role_tensor_sha256="4" * 64,
            identity_teacher_mapping_sha256="5" * 64,
        ), bases={0: basis(), 1: basis()},
        native_calls={0: native0, 1: native1},
    )
    hook = runtime.StudentCorrectionHook(
        {0: basis(), 1: basis()}, issuer_id=issuer, coordinator=coordinator,
    )
    program_route = "L" if route == "Q" else route
    prog = program(program_route)
    states = {0: "P", 1: "P"}
    hook.configure(program=prog, states=states)
    inputs = torch.arange(4 * 256, dtype=torch.long).view(4, 256) + step
    teacher_kind = "coordinate_labels" if route in {"Q", "L"} else "oon_logits"
    phase = "initial_denominator" if route == "Q" else "fit"
    ident = identity(
        hook, inputs, step=step, route=route, teacher_kind=teacher_kind, phase=phase,
    )
    return broker, hook, prog, native0, native1, inputs, ident


def scheduled_indices(ident):
    if ident.phase == "initial_denominator":
        return tuple(range(
            ident.batch_ordinal * runtime.BATCH_SIZE,
            (ident.batch_ordinal + 1) * runtime.BATCH_SIZE,
        ))
    permutation = runtime.fit_permutations(384, ident.trial)[ident.epoch]
    start = ident.batch_ordinal * runtime.BATCH_SIZE
    return tuple(int(value) for value in permutation[start:start + runtime.BATCH_SIZE])


def call_hook(hook, site, z, mo, nonce):
    deployed_n = runtime.mint_deployed_n_write(
        site=site, state=z, value=mo, forward_nonce=nonce,
        issuer_id=hook.issuer_id,
    )
    return hook(site, z, deployed_n, forward_nonce=nonce)


def run_student(broker, hook, inputs, ident, *, z1_shift: float = 0.0):
    generator = torch.Generator().manual_seed(10 + ident.optimizer_step)
    z0 = torch.randn(4, 256, runtime.D_MODEL, generator=generator)
    mo0 = torch.randn(4, 256, runtime.D_MODEL, generator=generator)
    session = broker.begin_student(ident, hook, inputs, scheduled_indices(ident))
    with session.forward_scope() as capability:
        out0 = call_hook(hook, 0, z0, mo0, ident.nonce)
        z1 = 0.3 * out0 + torch.randn(
            4, 256, runtime.D_MODEL, generator=generator,
        ) + z1_shift
        mo1 = torch.randn(4, 256, runtime.D_MODEL, generator=generator)
        out1 = call_hook(hook, 1, z1, mo1, ident.nonce)
        student_logits = out1[..., :11]
        student_logits.retain_grad()
        capability.bind_outer_logits(student_logits)
    step, closure = session.close(
        outer_forward_count=1, outer_returned=True,
        hook_restored=True, hook_inert=True,
    )
    return step, closure, z0, z1, student_logits


def test_coordinate_labels_use_current_student_states_and_detach() -> None:
    broker, hook, prog, native0, native1, inputs, ident = system("L")
    step, student_closure, z0, z1, _ = run_student(broker, hook, inputs, ident)
    assert student_closure.original_calls == capabilities.EXACT_ZERO_CALLS
    assert hook.program is None and hook.states == {}
    result = broker.run_coordinate_teacher(ident, step)
    predictions = (prog.site0_code(z0), prog.site1_code(z1))
    loss, closure = result.consume_loss((1.0, 1.0))
    manual_labels = (
        (native0(z0).detach() @ basis()),
        (native1(z1).detach() @ basis()),
    )
    expected = runtime.normalized_local_loss(predictions, manual_labels, (1.0, 1.0))
    torch.testing.assert_close(loss, expected)
    assert closure.original_calls == capabilities.EXACT_EARLY_ORIGINAL_CALLS
    assert closure.scope == "coordinate" and closure.outer_forward_count == 0
    loss.backward()
    assert prog.site0.weight.grad is not None and prog.site1.weight.grad is not None
    assert native0.scale.grad is None and native1.scale.grad is None
    with pytest.raises(RuntimeError, match="already consumed"):
        result.consume_loss((1.0, 1.0))


def test_coordinate_moments_are_exact_and_exclusive_with_loss() -> None:
    broker, hook, _, native0, native1, inputs, ident = system("Q", step=1)
    step, _, z0, z1, _ = run_student(broker, hook, inputs, ident, z1_shift=4.0)
    result = broker.run_coordinate_teacher(ident, step)
    moments, closure = result.consume_moments()
    manual = (
        runtime.MomentSufficientStatistics.from_labels(native0(z0) @ basis()),
        runtime.MomentSufficientStatistics.from_labels(native1(z1) @ basis()),
    )
    for observed, expected in zip(moments, manual, strict=True):
        torch.testing.assert_close(observed.mean, expected.mean)
        torch.testing.assert_close(observed.centered_m2, expected.centered_m2)
    assert closure.consumed is True
    with pytest.raises(RuntimeError, match="licensed only"):
        result.consume_loss((1.0, 1.0))


def autonomous_forward(gateway, inputs):
    z0 = inputs.float().unsqueeze(-1).expand(-1, -1, runtime.D_MODEL) / 1000
    exact0 = gateway.call(0, z0)
    z1 = z0 + exact0
    exact1 = gateway.call(1, z1)
    logits = exact1[..., :11]
    return logits, {
        "outer_forward_count": 1, "hook_calls": {0: 1, 1: 1, 2: 0},
        "outer_returned": True, "hook_restored": True, "hook_inert": True,
    }


def test_oon_is_autonomous_and_matches_manual_sequential_teacher() -> None:
    broker, hook, _, native0, native1, inputs, ident = system("R", step=2)
    step, _, _, student_z1, student_logits = run_student(
        broker, hook, inputs, ident, z1_shift=100.0,
    )
    result = broker.run_oon_teacher(ident, step, inputs, autonomous_forward)
    loss, closure = result.consume_loss()
    autonomous_z0 = inputs.float().unsqueeze(-1).expand(
        -1, -1, runtime.D_MODEL,
    ) / 1000
    manual = native1(autonomous_z0 + native0(autonomous_z0))[..., :11].detach()
    expected = runtime.teacher_student_kl(manual, student_logits)
    torch.testing.assert_close(loss, expected)
    assert not torch.allclose(manual[..., :1], native1(student_z1)[..., :1])
    assert closure.original_calls == capabilities.EXACT_EARLY_ORIGINAL_CALLS
    loss.backward()
    assert student_logits.grad is not None
    assert native0.scale.grad is None and native1.scale.grad is None


def test_gateway_is_revoked_and_exact_call_ledger_fails_closed() -> None:
    broker, hook, _, _, _, inputs, ident = system("R", step=3)
    step, _, *_ = run_student(broker, hook, inputs, ident)
    saved = []

    def missing_site(gateway, tokens):
        saved.append(gateway)
        z = torch.zeros(4, 256, runtime.D_MODEL)
        logits = gateway.call(0, z)[..., :7]
        return logits, {
            "outer_forward_count": 1, "hook_calls": {0: 1, 1: 1, 2: 0},
            "outer_returned": True, "hook_restored": True, "hook_inert": True,
        }

    with pytest.raises(RuntimeError, match="did not close exactly"):
        broker.run_oon_teacher(ident, step, inputs, missing_site)
    assert broker._CapabilityBroker__coordinator.idle
    with pytest.raises(RuntimeError, match="revoked"):
        saved[0].call(0, torch.zeros(4, 256, runtime.D_MODEL))
    with pytest.raises(RuntimeError, match="already consumed"):
        step._take(issuer_id="a" * 64, identity=ident)


def test_forbidden_mlp2_and_scope_overlap_restore_idle() -> None:
    broker, hook, _, _, _, inputs, ident = system("R", step=4)
    session = broker.begin_student(ident, hook, inputs, scheduled_indices(ident))
    with session.forward_scope() as capability:
        with pytest.raises(RuntimeError, match="overlaps active student"):
            with broker._CapabilityBroker__coordinator.enter("oon"):
                pass
        z = torch.zeros(4, 256, runtime.D_MODEL)
        mo = torch.zeros_like(z)
        out0 = call_hook(hook, 0, z, mo, ident.nonce)
        out1 = call_hook(hook, 1, out0, mo, ident.nonce)
        capability.bind_outer_logits(out1[..., :11])
    step, _ = session.close(
        outer_forward_count=1, outer_returned=True,
        hook_restored=True, hook_inert=True,
    )

    def calls_site2(gateway, tokens):
        gateway.call(2, torch.zeros(4, 256, runtime.D_MODEL))

    with pytest.raises(RuntimeError, match="exceeded|forbids"):
        broker.run_oon_teacher(ident, step, inputs, calls_site2)
    assert broker._CapabilityBroker__coordinator.idle


def test_student_original_call_and_bad_closure_mint_no_usable_trace() -> None:
    broker, hook, _, _, _, inputs, ident = system("L", step=5)
    session = broker.begin_student(ident, hook, inputs, scheduled_indices(ident))
    with pytest.raises(RuntimeError, match="exceeded"):
        with session.forward_scope() as monitor:
            monitor.record_original_call(0)
    assert hook.program is None and broker._CapabilityBroker__coordinator.idle
    with pytest.raises(RuntimeError, match="clean completed"):
        session.close(
            outer_forward_count=1, outer_returned=True,
            hook_restored=True, hook_inert=True,
        )

    broker2, hook2, _, _, _, inputs2, ident2 = system("L", step=6)
    trace_session = broker2.begin_student(ident2, hook2, inputs2, scheduled_indices(ident2))
    with trace_session.forward_scope() as capability:
        z = torch.zeros(4, 256, runtime.D_MODEL)
        call_hook(hook2, 0, z, z, ident2.nonce)
        call_hook(hook2, 1, z, z, ident2.nonce)
        capability.bind_outer_logits(z[..., :11])
    with pytest.raises(RuntimeError, match="closure failed"):
        trace_session.close(
            outer_forward_count=1, outer_returned=True,
            hook_restored=False, hook_inert=True,
        )
    assert hook2.program is None


def test_identity_replay_cross_broker_and_token_drift_reject() -> None:
    broker, hook, _, _, _, inputs, ident = system("R", step=7)
    step, _, *_ = run_student(broker, hook, inputs, ident)
    with pytest.raises(RuntimeError, match="duplicated"):
        broker.begin_student(ident, hook, inputs, scheduled_indices(ident))
    other = capabilities.CapabilityBroker(
        issuer_id="b" * 64, coordinator=runtime.ScopeCoordinator(),
        run_context=capabilities.RunContext(
            source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
            rows_receipt_sha256="3" * 64, fit_role_tensor_sha256="4" * 64,
            identity_teacher_mapping_sha256="5" * 64,
        ),
        bases={0: basis(), 1: basis()},
        native_calls={0: Native(1, 0), 1: Native(1, 0)},
    )
    with pytest.raises(RuntimeError, match="identity or issuer mismatch"):
        other.run_oon_teacher(ident, step, inputs, autonomous_forward)
    changed = inputs.clone(); changed[0, 0] += 1
    with pytest.raises(RuntimeError, match="differ"):
        broker.run_oon_teacher(ident, step, changed, autonomous_forward)
    result = broker.run_oon_teacher(ident, step, inputs, autonomous_forward)
    loss, _ = result.consume_loss()
    assert torch.isfinite(loss)
    with pytest.raises(RuntimeError, match="already consumed"):
        result.consume_loss()


def test_trace_is_forward_local_one_use_and_blocks_outstanding_reconfigure() -> None:
    broker, hook, prog, _, _, inputs, ident = system("L", step=8)
    step, _, z0, z1, _ = run_student(broker, hook, inputs, ident)
    with pytest.raises(RuntimeError, match="active trace"):
        hook.configure(program=prog, states={0: "P", 1: "P"})
    with pytest.raises(RuntimeError, match="no completed"):
        hook.pop_trace(ident)
    result = broker.run_coordinate_teacher(ident, step)
    result.consume_loss((1.0, 1.0))
    hook.configure(program=prog, states={0: "P", 1: "P"})
    second = identity(hook, torch.arange(4 * 256).view(4, 256), step=9,
                      route="L", teacher_kind="coordinate_labels")
    step2, *_ = run_student(
        broker, hook, torch.arange(4 * 256).view(4, 256), second, z1_shift=7.0,
    )
    assert step.output_sha256 != step2.output_sha256


def test_trace_identity_rejects_illegal_combinations_and_bool_integers() -> None:
    broker, hook, _, _, _, inputs, _ = system("L", step=10)
    kwargs = dict(
        inputs=inputs, ordered_batch_indices=range(4), source_commit="1" * 40,
        inherited_snapshot_sha256="2" * 64, rows_receipt_sha256="3" * 64,
        fit_role_tensor_sha256="4" * 64,
        program_snapshot_sha256=runtime.program_snapshot_sha256(hook.program),
        teacher_mapping_sha256="5" * 64, phase="fit", route="L", control="true",
        teacher_kind="coordinate_labels", trial=0, epoch=0, optimizer_step=0,
        batch_ordinal=0, student_states=((0, "P"), (1, "P"), (2, "N")),
    )
    with pytest.raises(ValueError, match="combination"):
        runtime.TraceIdentity.from_inputs(**{**kwargs, "teacher_kind": "oon_logits"})
    with pytest.raises(ValueError, match="integer"):
        runtime.TraceIdentity.from_inputs(**{**kwargs, "trial": True})
    with pytest.raises(ValueError, match=r"\[4,256\]"):
        runtime.TraceIdentity.from_inputs(**{**kwargs, "inputs": inputs[:, :-1]})
    with pytest.raises(ValueError, match="combination"):
        runtime.TraceIdentity.from_inputs(**{
            **kwargs, "route": "T", "teacher_kind": "oon_logits", "control": "zero_A",
        })


def test_student_rows_context_route_and_basis_are_exactly_bound() -> None:
    broker, hook, _, _, _, inputs, ident = system("L", step=11)
    changed = inputs.clone(); changed[0, 0] += 1
    with pytest.raises(RuntimeError, match="input tokens differ"):
        broker.begin_student(ident, hook, changed, scheduled_indices(ident))
    wrong_order = tuple(reversed(scheduled_indices(ident)))
    with pytest.raises(RuntimeError, match="batch indices differ"):
        broker.begin_student(ident, hook, inputs, wrong_order)

    mapped = identity(
        hook, inputs, step=11, route="L", teacher_kind="coordinate_labels",
        control="document_shuffle",
    )
    with pytest.raises(RuntimeError, match="mapped controls"):
        broker.begin_student(mapped, hook, inputs, scheduled_indices(mapped))

    issuer = "c" * 64
    coordinator = runtime.ScopeCoordinator()
    wrong_basis = basis()[:, torch.roll(torch.arange(runtime.CODE_DIM), 1)]
    wrong_hook = runtime.StudentCorrectionHook(
        {0: wrong_basis, 1: basis()}, issuer_id=issuer, coordinator=coordinator,
    )
    wrong_hook.configure(program=program("L"), states={0: "P", 1: "P"})
    wrong_ident = identity(
        wrong_hook, inputs, step=11, route="L", teacher_kind="coordinate_labels",
    )
    strict_broker = capabilities.CapabilityBroker(
        issuer_id=issuer, coordinator=coordinator,
        run_context=capabilities.RunContext(
            source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
            rows_receipt_sha256="3" * 64, fit_role_tensor_sha256="4" * 64,
            identity_teacher_mapping_sha256="5" * 64,
        ), bases={0: basis(), 1: basis()},
        native_calls={0: Native(1, 0), 1: Native(1, 0)},
    )
    with pytest.raises(RuntimeError, match="different bases"):
        strict_broker.begin_student(
            wrong_ident, wrong_hook, inputs, scheduled_indices(wrong_ident),
        )

    route_hook = runtime.StudentCorrectionHook(
        {0: basis(), 1: basis()}, issuer_id=issuer, coordinator=coordinator,
    )
    route_hook.configure(program=program("L"), states={0: "P", 1: "P"})
    relabeled = identity(route_hook, inputs, step=11, route="R", teacher_kind="oon_logits")
    with pytest.raises(RuntimeError, match="route/program"):
        strict_broker.begin_student(
            relabeled, route_hook, inputs, scheduled_indices(relabeled),
        )


def test_broker_is_sealed_and_ledger_closes_one_transaction() -> None:
    broker, hook, _, _, _, inputs, ident = system("L", step=12)
    with pytest.raises(RuntimeError, match="cannot be copied"):
        copy.copy(broker)
    with pytest.raises(AttributeError, match="sealed"):
        broker.issuer_id = "f" * 64
    step, student_closure, *_ = run_student(broker, hook, inputs, ident)
    assert student_closure.consumed is False
    with pytest.raises(RuntimeError, match="active trace"):
        # A different identity cannot start while the exact teacher is pending.
        hook.configure(program=program("L"), states={0: "P", 1: "P"})
    with pytest.raises(RuntimeError, match="cannot be copied"):
        copy.copy(step)
    result = broker.run_coordinate_teacher(ident, step)
    result.consume_loss((1.0, 1.0))
    snapshot = broker.ledger_snapshot
    assert (snapshot.student_identity_count, snapshot.teacher_identity_count) == (1, 1)
    assert snapshot.completed_identity_count == 1
    assert snapshot.outstanding_identity_sha256 is None


def test_denominator_schedule_and_student_output_mutation_fail_closed() -> None:
    broker, hook, _, _, _, inputs, _ = system("L", step=13)
    denominator = runtime.TraceIdentity.from_inputs(
        inputs=inputs, ordered_batch_indices=range(4), source_commit="1" * 40,
        inherited_snapshot_sha256="2" * 64, rows_receipt_sha256="3" * 64,
        fit_role_tensor_sha256="4" * 64,
        program_snapshot_sha256=runtime.program_snapshot_sha256(hook.program),
        teacher_mapping_sha256="5" * 64, phase="initial_denominator", route="Q",
        control="true", teacher_kind="coordinate_labels", trial=1, epoch=0,
        optimizer_step=0, batch_ordinal=0,
        student_states=((0, "P"), (1, "P"), (2, "N")),
    )
    with pytest.raises(RuntimeError, match="denominator schedule"):
        broker.begin_student(denominator, hook, inputs, range(4))

    broker2, hook2, _, _, _, inputs2, ident2 = system("R", step=14)
    step, *_ = run_student(broker2, hook2, inputs2, ident2)
    result = broker2.run_oon_teacher(ident2, step, inputs2, autonomous_forward)
    outputs = result._TeacherResult__student_outputs
    with torch.no_grad():
        outputs._StudentOutputs__values["logits"].add_(1)
    with pytest.raises(RuntimeError, match="mutated"):
        result.consume_loss()
    assert broker2.ledger_snapshot.outstanding_identity_sha256 is None


def test_route_trainability_and_coordinate_phase_are_typed() -> None:
    broker, hook, _, _, _, inputs, ident = system("S0", step=15)
    hook.program.site1.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="trainable tensor set changed"):
        broker.begin_student(ident, hook, inputs, scheduled_indices(ident))
    assert broker.ledger_snapshot.outstanding_identity_sha256 is None

    broker2, hook2, _, _, _, inputs2, ident2 = system("L", step=16)
    step, *_ = run_student(broker2, hook2, inputs2, ident2)
    result = broker2.run_coordinate_teacher(ident2, step)
    with pytest.raises(RuntimeError, match="initial Q"):
        result.consume_moments()
    loss, _ = result.consume_loss((1.0, 1.0))
    assert torch.isfinite(loss)


def test_detached_or_mutated_preclose_outputs_fail_closed() -> None:
    broker, hook, _, _, _, inputs, ident = system("R", step=17)
    generator = torch.Generator().manual_seed(101)
    z = torch.randn(4, 256, runtime.D_MODEL, generator=generator)
    session = broker.begin_student(ident, hook, inputs, scheduled_indices(ident))
    with session.forward_scope() as capability:
        out0 = call_hook(hook, 0, z, z, ident.nonce)
        out1 = call_hook(hook, 1, out0, z, ident.nonce)
        capability.bind_outer_logits(out1[..., :11].detach())
    with pytest.raises(RuntimeError, match="detached from their graph"):
        session.close(
            outer_forward_count=1, outer_returned=True,
            hook_restored=True, hook_inert=True,
        )
    assert hook.program is None and hook.states == {}
    assert broker.ledger_snapshot.outstanding_identity_sha256 is None

    broker2, hook2, _, _, _, inputs2, ident2 = system("R", step=18)
    session2 = broker2.begin_student(ident2, hook2, inputs2, scheduled_indices(ident2))
    with session2.forward_scope() as capability:
        out0 = call_hook(hook2, 0, z, z, ident2.nonce)
        out1 = call_hook(hook2, 1, out0, z, ident2.nonce)
        logits = out1[..., :11]
        capability.bind_outer_logits(logits)
    with torch.no_grad():
        logits.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="malformed"):
        session2.close(
            outer_forward_count=1, outer_returned=True,
            hook_restored=True, hook_inert=True,
        )
    assert hook2.program is None and hook2.states == {}
    assert broker2.ledger_snapshot.outstanding_identity_sha256 is None


def test_mutated_output_cannot_block_teacher_failure_cleanup() -> None:
    broker, hook, _, _, _, inputs, ident = system("R", step=19)
    step, *_ = run_student(broker, hook, inputs, ident)
    outputs = step._StudentStep__outputs
    with torch.no_grad():
        outputs._StudentOutputs__values["logits"].fill_(float("nan"))

    def fails(gateway, tokens):
        raise ValueError("synthetic producer failure")

    with pytest.raises(ValueError, match="synthetic producer failure"):
        broker.run_oon_teacher(ident, step, inputs, fails)
    assert broker.ledger_snapshot.outstanding_identity_sha256 is None
    assert broker._CapabilityBroker__coordinator.idle
    with pytest.raises(RuntimeError, match="already consumed"):
        step._take(issuer_id="a" * 64, identity=ident)


@pytest.mark.parametrize("route,step_index", [("R", 20), ("S0", 21), ("S1", 22), ("T", 23)])
def test_suffix_loss_graph_must_reach_exact_route_parameters(route, step_index) -> None:
    broker, hook, prog, _, _, inputs, ident = system(route, step=step_index)
    z = torch.randn(4, 256, runtime.D_MODEL)
    session = broker.begin_student(ident, hook, inputs, scheduled_indices(ident))
    with session.forward_scope() as capability:
        out0 = call_hook(hook, 0, z, z, ident.nonce)
        call_hook(hook, 1, out0, z, ident.nonce)
        fake = torch.zeros(4, 256, 11, requires_grad=True) * 1.0
        capability.bind_outer_logits(fake)
    with pytest.raises(RuntimeError, match="disconnected from route parameters"):
        session.close(
            outer_forward_count=1, outer_returned=True,
            hook_restored=True, hook_inert=True,
        )
    assert all(parameter.grad is None for parameter in prog.parameters())
    assert broker.ledger_snapshot.outstanding_identity_sha256 is None
    assert hook.program is None and hook.states == {}


def test_close_without_forward_aborts_and_clears_configuration() -> None:
    broker, hook, _, _, _, inputs, ident = system("L", step=24)
    session = broker.begin_student(ident, hook, inputs, scheduled_indices(ident))
    with pytest.raises(RuntimeError, match="lacks a clean completed forward"):
        session.close(
            outer_forward_count=1, outer_returned=True,
            hook_restored=True, hook_inert=True,
        )
    assert broker.ledger_snapshot.outstanding_identity_sha256 is None
    assert hook.program is None and hook.states == {}


def test_stale_step_rejection_preserves_fresh_teacher_transaction() -> None:
    broker, hook, prog, _, _, inputs, ident = system("L", step=25)
    old_step, *_ = run_student(broker, hook, inputs, ident)
    first = broker.run_coordinate_teacher(ident, old_step)
    first.consume_loss((1.0, 1.0))

    hook.configure(program=prog, states={0: "P", 1: "P"})
    inputs2 = inputs + 1
    ident2 = identity(
        hook, inputs2, step=26, route="L", teacher_kind="coordinate_labels",
    )
    fresh_step, *_ = run_student(broker, hook, inputs2, ident2)
    before = broker.ledger_snapshot
    with pytest.raises(RuntimeError, match="already consumed|identity"):
        broker.run_coordinate_teacher(ident2, old_step)
    after = broker.ledger_snapshot
    assert after.teacher_identity_count == before.teacher_identity_count
    assert after.outstanding_identity_sha256 == ident2.sha256
    valid = broker.run_coordinate_teacher(ident2, fresh_step)
    loss, _ = valid.consume_loss((1.0, 1.0))
    assert torch.isfinite(loss)
    assert broker.ledger_snapshot.outstanding_identity_sha256 is None
