import pytest
import torch

import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_final_capability as capability
import early_mlp_suffix_transport_v1_response_execution as response_execution
import early_mlp_suffix_transport_v1_runtime as runtime


SHA = "a" * 64


def _row(value: float = 1.0) -> capability.RowReduction:
    return capability.RowReduction(
        row_sum=torch.full((192,), value, dtype=torch.float64),
        row_count=torch.ones(192, dtype=torch.long),
    )


def _response() -> capability.ResponseReduction:
    return capability.ResponseReduction(
        error_sum=torch.ones(192, dtype=torch.float64),
        teacher_sum=torch.full((192,), 4.0, dtype=torch.float64),
        student_sum=torch.ones(192, dtype=torch.float64),
        dot_sum=torch.full((192,), 2.0, dtype=torch.float64),
        unit_identity="b" * 64,
    )


def _output_kl() -> capability.OutputKLReduction:
    return capability.OutputKLReduction(
        numerator_sum=torch.ones(192, dtype=torch.float64),
        denominator_sum=torch.full((192,), 2.0, dtype=torch.float64),
        unit_identity="b" * 64,
    )


def _frequency(index: int) -> capability.FrequencyRowReduction:
    return capability.FrequencyRowReduction(
        row_sum=torch.ones(192, dtype=torch.float64) if index == 0 else (
            torch.zeros(192, dtype=torch.float64)
        ),
        row_count=torch.ones(192, dtype=torch.long) if index == 0 else (
            torch.zeros(192, dtype=torch.long)
        ),
    )


def _observation(action: capability.FinalAction, **changes):
    response = action.background == "N" and (
        action.arm == "ll" or action.arm == "lt" or action.arm.startswith("a_null_")
    )
    code = action.background == "N" and action.arm in {"ll", "lt"}
    values = {
        "action": action,
        "common_support_sha256": SHA,
        "ce": _row(),
        "teacher_kl": _row() if action.background == "N" else None,
        "copy_ce": _row(),
        "frequency_ce": tuple(_frequency(index) for index in range(9)),
        "code_response": _response() if code else None,
        "logit_response": _response() if response else None,
        "output_kl_response": _output_kl() if response else None,
        "consumer_norm_ratio": tuple(_row() for _ in range(18)),
        "execution_closure_sha256": "c" * 64,
    }
    values.update(changes)
    return capability.FinalArmObservation(**values)


def test_capability_executes_exact_canonical_lattice_once():
    calls = []

    def execute(action):
        calls.append(action.key)
        return _observation(action)

    owned = capability.mint_final_action_capability(
        issuer_id="d" * 64, common_support_sha256=SHA, executor=execute,
    )
    bundle = owned.execute_all()
    assert tuple(calls) == capability.CANONICAL_ACTION_KEYS
    assert len(bundle.observations) == len(capability.BASE_ARMS) * 2 == 68
    assert bundle.common_support_sha256 == SHA
    assert owned.spent is True and owned.failed is False
    with pytest.raises(RuntimeError, match="already closed"):
        owned.execute_all()


def test_wrong_action_or_support_poison_closes_capability():
    def wrong_action(action):
        if action.key == capability.CANONICAL_ACTION_KEYS[1]:
            return _observation(capability.CANONICAL_ACTIONS[0])
        return _observation(action)

    owned = capability.mint_final_action_capability(
        issuer_id="d" * 64, common_support_sha256=SHA, executor=wrong_action,
    )
    with pytest.raises(RuntimeError, match="wrong typed action"):
        owned.execute_all()
    assert owned.spent is True and owned.failed is True
    with pytest.raises(RuntimeError, match="already closed"):
        owned.execute_all()

    mixed = capability.mint_final_action_capability(
        issuer_id="d" * 64, common_support_sha256=SHA,
        executor=lambda action: _observation(action, common_support_sha256="e" * 64),
    )
    with pytest.raises(RuntimeError, match="mixed scored support"):
        mixed.execute_all()


def test_response_and_background_types_follow_registered_semantics():
    with pytest.raises(ValueError, match="requires teacher-KL"):
        _observation(capability.FinalAction("qq", "N"), teacher_kl=None)
    with pytest.raises(ValueError, match="CE-only"):
        _observation(capability.FinalAction("qq", "E"), teacher_kl=_row())
    with pytest.raises(ValueError, match="response reductions"):
        _observation(capability.FinalAction("lt", "N"), logit_response=None)
    with pytest.raises(ValueError, match="response reductions"):
        _observation(capability.FinalAction("qq", "N"), logit_response=_response())
    with pytest.raises(ValueError, match="response reductions"):
        _observation(capability.FinalAction("lt", "N"), output_kl_response=None)


def test_raw_or_graph_bearing_reductions_fail_closed():
    with pytest.raises(ValueError, match="allowed final row reduction"):
        capability.RowReduction(
            row_sum=torch.ones(192, 1, dtype=torch.float64),
            row_count=torch.ones(192, dtype=torch.long),
        )
    graph = torch.ones(192, dtype=torch.float64, requires_grad=True) * 2
    with pytest.raises(ValueError, match="allowed final row reduction"):
        capability.RowReduction(
            row_sum=graph, row_count=torch.ones(192, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="all nine"):
        _observation(capability.FinalAction("qq", "N"), frequency_ce=tuple())
    with pytest.raises(ValueError, match="all live-consumer"):
        _observation(capability.FinalAction("qq", "N"), consumer_norm_ratio=tuple())


def test_response_inner_products_are_consistency_checked_and_cloned():
    source = torch.ones(192, dtype=torch.float64)
    response = capability.ResponseReduction(
        error_sum=source, teacher_sum=4 * source, student_sum=source,
        dot_sum=2 * source, unit_identity="b" * 64,
    )
    source.fill_(9)
    assert torch.equal(response.error_sum, torch.ones(192, dtype=torch.float64))
    with pytest.raises(ValueError, match="inconsistent"):
        capability.ResponseReduction(
            error_sum=2 * torch.ones(192, dtype=torch.float64),
            teacher_sum=4 * torch.ones(192, dtype=torch.float64),
            student_sum=torch.ones(192, dtype=torch.float64),
            dot_sum=2 * torch.ones(192, dtype=torch.float64),
            unit_identity="b" * 64,
        )


def test_output_kl_reductions_are_nonnegative_and_cloned():
    source = torch.ones(192, dtype=torch.float64)
    reduction = capability.OutputKLReduction(
        numerator_sum=source, denominator_sum=2 * source,
        unit_identity="b" * 64,
    )
    source.fill_(9)
    assert torch.equal(reduction.numerator_sum, torch.ones(192, dtype=torch.float64))
    with pytest.raises(ValueError, match="nonnegative"):
        capability.OutputKLReduction(
            numerator_sum=-torch.ones(192, dtype=torch.float64),
            denominator_sum=torch.ones(192, dtype=torch.float64),
            unit_identity="b" * 64,
        )


def test_direct_construction_without_mint_token_is_forbidden():
    with pytest.raises(TypeError, match="must be minted"):
        capability.FinalActionCapability(
            _token=object(), issuer_id="d" * 64,
            common_support_sha256=SHA, executor=lambda action: _observation(action),
        )


def _run_response(value: float, units, marker: str):
    student = torch.full((192,), value, dtype=torch.float64)
    teacher = torch.full((192,), 4.0, dtype=torch.float64)
    dot = torch.sqrt(student * teacher)
    error = student + teacher - 2 * dot
    return response_execution.ObservedRunResponseReduction(
        error_sum=error, teacher_sum=teacher, student_sum=student, dot_sum=dot,
        unit_identity_sha256s=units,
        batch_reduction_sha256s=tuple(
            runtime.logical_identity_sha256({"response": marker, "batch": index})
            for index in range(48)
        ),
    )


def _run_output(units, marker: str):
    return response_execution.ObservedRunOutputKLReduction(
        numerator_sum=torch.ones(192, dtype=torch.float64),
        denominator_sum=torch.full((192,), 2.0, dtype=torch.float64),
        unit_identity_sha256s=units,
        batch_reduction_sha256s=tuple(
            runtime.logical_identity_sha256({"output": marker, "batch": index})
            for index in range(48)
        ),
    )


def _response_run_result():
    units = tuple(
        runtime.logical_identity_sha256({"unit": index}) for index in range(48)
    )
    arms = tuple(
        response_execution.ObservedResponseRunArmReduction(
            action_key=key,
            code_response=(
                _run_response(1.0, units, f"{key}/code")
                if key in {"ll/N", "lt/N"} else None
            ),
            logit_response=_run_response(1.0, units, f"{key}/logit"),
            output_kl_response=_run_output(units, key),
        ) for key in response_execution.response_plan.RESPONSE_ACTION_KEYS
    )
    ordered = runtime.logical_identity_sha256({
        "kind": "early_mlp_suffix_transport_v1_response_units",
        "ordered_batch_unit_sha256s": list(units),
    })
    receipt = response_execution.ObservedResponseRunReceipt(
        final_context_sha256="1" * 64, source_bank_sha256="2" * 64,
        program_payload_sha256="3" * 64, common_support_sha256=SHA,
        basis0_sha256="4" * 64, basis1_sha256="5" * 64,
        ordered_unit_identity_sha256=ordered,
        batch_receipt_sha256s=tuple(
            runtime.logical_identity_sha256({"receipt": index}) for index in range(48)
        ),
        batch_plan_sha256s=tuple(
            runtime.logical_identity_sha256({"plan": index}) for index in range(48)
        ),
        arm_reduction_sha256s=tuple((value.action_key, value.sha256) for value in arms),
        teacher_forward_count=144, student_forward_count=3168,
        row_count=192, atomic_complete=True,
    )
    return response_execution.ObservedResponseRunResult(
        arm_reductions=arms, receipt=receipt,
    )


def _observation_bundle(run=None, *, changed_action: str | None = None):
    run = _response_run_result() if run is None else run
    by_action = {value.action_key: value for value in run.arm_reductions}
    observations = []
    unit = run.receipt.ordered_unit_identity_sha256
    for action in capability.CANONICAL_ACTIONS:
        changes = {}
        if action.key in by_action:
            arm = by_action[action.key]
            changes = {
                "code_response": (
                    None if arm.code_response is None else capability.ResponseReduction(
                        **arm.code_response.as_statistics(unit)
                    )
                ),
                "logit_response": capability.ResponseReduction(
                    **arm.logit_response.as_statistics(unit)
                ),
                "output_kl_response": capability.OutputKLReduction(
                    **arm.output_kl_response.as_statistics(unit)
                ),
            }
        if action.key == changed_action:
            changes["logit_response"] = capability.ResponseReduction(
                error_sum=torch.ones(192, dtype=torch.float64),
                teacher_sum=torch.ones(192, dtype=torch.float64),
                student_sum=torch.zeros(192, dtype=torch.float64),
                dot_sum=torch.zeros(192, dtype=torch.float64),
                unit_identity=unit,
            )
        observations.append(_observation(action, **changes))
    plan_sha256 = runtime.logical_identity_sha256(
        list(capability.CANONICAL_ACTION_KEYS)
    )
    return capability.FinalObservationBundle(
        common_support_sha256=SHA, observations=tuple(observations),
        action_plan_sha256=plan_sha256,
        bundle_sha256=runtime.logical_identity_sha256({
            "plan": plan_sha256,
            "observations": [value.sha256 for value in observations],
        }),
    )


def test_observational_bundle_joins_one_complete_response_run() -> None:
    run = _response_run_result()
    bundle = _observation_bundle(run)
    receipt = capability.join_observations_with_response_run(bundle, run)
    assert receipt.observation_bundle_sha256 == bundle.bundle_sha256
    assert receipt.response_run_receipt_sha256 == run.receipt.sha256
    assert receipt.program_payload_sha256 == run.receipt.program_payload_sha256
    assert receipt.common_support_sha256 == SHA
    assert tuple(key for key, _left, _right in receipt.response_action_matches) == (
        capability.RESPONSE_ACTION_KEYS
    )
    assert all(left == right for _key, left, right in receipt.response_action_matches)


def test_evidence_join_rejects_substitution_and_mixed_support() -> None:
    run = _response_run_result()
    changed = _observation_bundle(run, changed_action="a_null_07/N")
    with pytest.raises(RuntimeError, match="a_null_07/N observation differs"):
        capability.join_observations_with_response_run(changed, run)

    mixed = response_execution.ObservedResponseRunReceipt(
        **{
            field: ("9" * 64 if field == "common_support_sha256" else getattr(
                run.receipt, field
            )) for field in run.receipt.__dataclass_fields__
        }
    )
    mixed_run = response_execution.ObservedResponseRunResult(
        arm_reductions=run.arm_reductions, receipt=mixed,
    )
    with pytest.raises(RuntimeError, match="support"):
        capability.join_observations_with_response_run(_observation_bundle(run), mixed_run)


def test_evidence_join_receipt_is_self_authenticating() -> None:
    receipt = capability.join_observations_with_response_run(
        _observation_bundle(), _response_run_result(),
    )
    values = {
        field: getattr(receipt, field) for field in receipt.__dataclass_fields__
    }
    values["ordered_unit_identity_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="identity changed"):
        capability.FinalEvidenceJoinReceipt(**values)


def _final_trace(*, phase="final", role="early_mlp_suffix_transport_v1_final"):
    rows = torch.arange(4 * 513, dtype=torch.long).view(4, 513)
    inputs = rows[:, :256].contiguous()
    identity = runtime.TraceIdentity.from_inputs(
        inputs=inputs, ordered_batch_indices=(8, 9, 10, 11),
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, fit_role_tensor_sha256="4" * 64,
        program_snapshot_sha256="5" * 64, teacher_mapping_sha256="6" * 64,
        role=role, phase=phase, route="R", control="true",
        teacher_kind="oon_logits", trial=0, epoch=0, optimizer_step=2,
        batch_ordinal=2, student_states=((0, "P"), (1, "P"), (2, "N")),
    )
    return identity, inputs


def test_final_trace_and_context_are_distinct_from_validation_authority():
    identity, inputs = _final_trace()
    context = capabilities.FinalRunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, final_role_tensor_sha256="4" * 64,
        identity_teacher_mapping_sha256="6" * 64,
    )
    context.require_identity(identity, inputs, (8, 9, 10, 11))
    validation = capabilities.ValidationRunContext(
        source_commit="1" * 40, inherited_snapshot_sha256="2" * 64,
        rows_receipt_sha256="3" * 64, validation_role_tensor_sha256="4" * 64,
        identity_teacher_mapping_sha256="6" * 64,
    )
    with pytest.raises(RuntimeError, match="validation identity"):
        validation.require_identity(identity, inputs, (8, 9, 10, 11))
    with pytest.raises(ValueError, match="role and execution phase"):
        _final_trace(role="early_mlp_suffix_transport_v1_validation")
