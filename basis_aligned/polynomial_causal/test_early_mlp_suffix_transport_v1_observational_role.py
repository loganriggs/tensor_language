from __future__ import annotations

import pytest
import torch

import bilin18_observed_adapter as observed
import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_final_capability as capability
import early_mlp_suffix_transport_v1_observational_role as role
import early_mlp_suffix_transport_v1_runtime as runtime


SUPPORT = "a" * 64


def _batch(
    action: capability.FinalAction, ordinal: int, *, support: str = SUPPORT,
    receipt_action: str | None = None,
) -> role.FinalObservationalBatch:
    ce_sum = torch.full((4,), 192.0, dtype=torch.float64)
    ce_count = torch.full((4,), 192, dtype=torch.long)
    frequency_sum = torch.zeros(4, 9, dtype=torch.float64)
    frequency_count = torch.zeros(4, 9, dtype=torch.long)
    for row in range(4):
        index = (ordinal * 4 + row) % 9
        frequency_sum[row, index] = ce_sum[row]
        frequency_count[row, index] = ce_count[row]
    identity = {"action": action.key, "batch": ordinal}
    receipt_identity = {"action": receipt_action or action.key, "batch": ordinal}
    return role.FinalObservationalBatch(
        action=action, batch_ordinal=ordinal, common_support_sha256=support,
        action_identity_sha256=runtime.logical_identity_sha256(identity),
        backend_receipt_sha256=runtime.logical_identity_sha256(receipt_identity),
        frequency_assignment_sha256=runtime.tensor_identity_sha256(
            torch.tensor(
                [[(ordinal * 4 + row) % 9] * 192 for row in range(4)],
                dtype=torch.long,
            )
        ),
        row_primary_sum=(
            torch.ones(4, dtype=torch.float64) if action.background == "N" else None
        ),
        row_primary_count=(
            torch.full((4,), 192, dtype=torch.long)
            if action.background == "N" else None
        ),
        row_ce_sum=ce_sum, row_ce_count=ce_count,
        row_copy_ce_sum=torch.ones(4, dtype=torch.float64),
        row_copy_count=torch.ones(4, dtype=torch.long),
        row_frequency_ce_sum=frequency_sum,
        row_frequency_count=frequency_count,
    )


def _response(identity: str):
    return capability.ResponseReduction(
        error_sum=torch.ones(192, dtype=torch.float64),
        teacher_sum=torch.full((192,), 4.0, dtype=torch.float64),
        student_sum=torch.ones(192, dtype=torch.float64),
        dot_sum=torch.full((192,), 2.0, dtype=torch.float64),
        unit_identity=identity,
    )


def _output(identity: str):
    return capability.OutputKLReduction(
        numerator_sum=torch.ones(192, dtype=torch.float64),
        denominator_sum=torch.full((192,), 2.0, dtype=torch.float64),
        unit_identity=identity,
    )


def _complete(core: role.FinalObservationalActionCore):
    response = core.action.background == "N" and (
        core.action.arm in capability.final_actions.RESPONSE_ARMS
    )
    code = core.action.background == "N" and (
        core.action.arm in capability.final_actions.CODE_RESPONSE_ARMS
    )
    unit = "b" * 64
    consumer = tuple(
        capability.RowReduction(
            row_sum=torch.ones(192, dtype=torch.float64),
            row_count=torch.ones(192, dtype=torch.long),
        ) for _ in range(18)
    )
    return capability.FinalArmObservation(
        action=core.action, common_support_sha256=core.common_support_sha256,
        ce=core.ce, teacher_kl=core.primary, copy_ce=core.copy_ce,
        frequency_ce=core.frequency_ce,
        code_response=_response(unit) if code else None,
        logit_response=_response(unit) if response else None,
        output_kl_response=_output(unit) if response else None,
        consumer_norm_ratio=consumer, execution_closure_sha256="c" * 64,
    )


def test_role_owner_executes_68_actions_by_48_batches_and_mints_capability() -> None:
    calls = []

    def execute(action, ordinal):
        calls.append((action.key, ordinal))
        return _batch(action, ordinal)

    owner = role.FinalObservationalRoleOwner(
        issuer_id="d" * 64, common_support_sha256=SUPPORT,
        batch_executor=execute,
    )
    bundle = owner.mint_action_capability(completer=_complete).execute_all()
    assert len(calls) == 68 * 48
    assert tuple(calls[index * 48] for index in range(68)) == tuple(
        (key, 0) for key in capability.CANONICAL_ACTION_KEYS
    )
    assert bundle.common_support_sha256 == SUPPORT
    receipt = owner.receipt
    assert tuple(key for key, _value in receipt.action_core_sha256s) == (
        capability.CANONICAL_ACTION_KEYS
    )
    assert len(receipt.backend_receipt_sha256s) == 68 * 48
    assert receipt.call_ledgers_sha256 == runtime.logical_identity_sha256(
        capability.final_actions.expected_observational_action_call_ledgers()
    )
    assert len(receipt.frequency_plan_sha256) == 64


@pytest.mark.parametrize("bad_ordinal", (1, 47))
def test_action_accumulator_rejects_skipped_or_reordered_batch(bad_ordinal) -> None:
    action = capability.CANONICAL_ACTIONS[0]
    accumulator = role.FinalObservationalActionAccumulator(action, SUPPORT)
    with pytest.raises(RuntimeError, match="skipped, reordered"):
        accumulator.add(_batch(action, bad_ordinal))
    with pytest.raises(RuntimeError, match="incomplete"):
        accumulator.finish()


def test_action_accumulator_rejects_duplicate_receipt_and_mixed_support() -> None:
    action = capability.CANONICAL_ACTIONS[0]
    accumulator = role.FinalObservationalActionAccumulator(action, SUPPORT)
    accumulator.add(_batch(action, 0))
    duplicate = _batch(action, 1)
    object.__setattr__(
        duplicate, "backend_receipt_sha256",
        _batch(action, 0).backend_receipt_sha256,
    )
    with pytest.raises(RuntimeError, match="duplicated"):
        accumulator.add(duplicate)

    mixed = role.FinalObservationalActionAccumulator(action, SUPPORT)
    with pytest.raises(RuntimeError, match="support-mixed"):
        mixed.add(_batch(action, 0, support="9" * 64))


def test_role_owner_rejects_receipt_replay_across_actions() -> None:
    def execute(action, ordinal):
        # The second action maliciously reuses the first action's backend receipts.
        receipt_action = capability.CANONICAL_ACTIONS[0].key
        return _batch(action, ordinal, receipt_action=receipt_action)

    owner = role.FinalObservationalRoleOwner(
        issuer_id="d" * 64, common_support_sha256=SUPPORT,
        batch_executor=execute,
    )
    owned = owner.mint_action_capability(completer=_complete)
    with pytest.raises(RuntimeError, match="replayed across actions"):
        owned.execute_all()
    with pytest.raises(RuntimeError, match="unavailable before completion"):
        _ = owner.receipt


def test_role_owner_rejects_reordered_or_duplicated_actions() -> None:
    owner = role.FinalObservationalRoleOwner(
        issuer_id="d" * 64, common_support_sha256=SUPPORT,
        batch_executor=lambda action, ordinal: _batch(action, ordinal),
    )
    with pytest.raises(RuntimeError, match="skipped, reordered, or duplicated"):
        owner._execute_core(capability.CANONICAL_ACTIONS[1])

    duplicate = role.FinalObservationalRoleOwner(
        issuer_id="d" * 64, common_support_sha256=SUPPORT,
        batch_executor=lambda action, ordinal: _batch(action, ordinal),
    )
    duplicate._execute_core(capability.CANONICAL_ACTIONS[0])
    with pytest.raises(RuntimeError, match="skipped, reordered, or duplicated"):
        duplicate._execute_core(capability.CANONICAL_ACTIONS[0])


def test_role_owner_rejects_frequency_support_drift_across_actions() -> None:
    def execute(action, ordinal):
        value = _batch(action, ordinal)
        if action == capability.CANONICAL_ACTIONS[1] and ordinal == 17:
            object.__setattr__(value, "frequency_assignment_sha256", "9" * 64)
        return value

    owner = role.FinalObservationalRoleOwner(
        issuer_id="d" * 64, common_support_sha256=SUPPORT,
        batch_executor=execute,
    )
    owned = owner.mint_action_capability(completer=_complete)
    with pytest.raises(RuntimeError, match="mixed frequency support"):
        owned.execute_all()


def test_role_owner_rejects_completer_that_changes_owned_reductions() -> None:
    owner = role.FinalObservationalRoleOwner(
        issuer_id="d" * 64, common_support_sha256=SUPPORT,
        batch_executor=lambda action, ordinal: _batch(action, ordinal),
    )

    def malicious(core):
        observation = _complete(core)
        changed = capability.RowReduction(
            row_sum=observation.ce.row_sum + 1,
            row_count=observation.ce.row_count,
        )
        values = {
            field: getattr(observation, field)
            for field in observation.__dataclass_fields__
        }
        values["ce"] = changed
        values["frequency_ce"] = tuple(
            capability.FrequencyRowReduction(
                row_sum=value.row_sum + (value.row_count > 0).double(),
                row_count=value.row_count,
            ) for value in observation.frequency_ce
        )
        return capability.FinalArmObservation(**values)

    with pytest.raises(RuntimeError, match="changed owned reductions"):
        owner.mint_action_capability(completer=malicious).execute_all()


def test_frequency_partition_allows_empty_cells_but_not_fabricated_mass() -> None:
    empty = capability.FrequencyRowReduction(
        row_sum=torch.zeros(192, dtype=torch.float64),
        row_count=torch.zeros(192, dtype=torch.long),
    )
    assert int(empty.row_count.sum()) == 0
    with pytest.raises(ValueError, match="nonzero CE"):
        capability.FrequencyRowReduction(
            row_sum=torch.ones(192, dtype=torch.float64),
            row_count=torch.zeros(192, dtype=torch.long),
        )


@pytest.mark.parametrize("background", ("N", "E"))
def test_existing_program_backend_result_joins_exact_reduction_receipt(background) -> None:
    action = capability.FinalAction("rr", background)
    synthetic = _batch(action, 0)
    fields = {
        "identity_sha256": synthetic.action_identity_sha256,
        "route": "R", "program_sha256": "1" * 64,
        "row_ce_sum": synthetic.row_ce_sum,
        "row_ce_count": synthetic.row_ce_count,
        "row_copy_ce_sum": synthetic.row_copy_ce_sum,
        "row_copy_count": synthetic.row_copy_count,
        "row_frequency_ce_sum": synthetic.row_frequency_ce_sum,
        "row_frequency_count": synthetic.row_frequency_count,
    }
    if background == "N":
        fields.update(
            row_primary_sum=synthetic.row_primary_sum,
            row_primary_count=synthetic.row_primary_count,
        )
        reductions = capabilities.FinalBatchReductions(**fields)
        order = (
            "row_primary_sum", "row_primary_count", "row_ce_sum", "row_ce_count",
            "row_copy_ce_sum", "row_copy_count", "row_frequency_ce_sum",
            "row_frequency_count",
        )
    else:
        reductions = capabilities.FinalCEBatchReductions(**fields)
        order = (
            "row_ce_sum", "row_ce_count", "row_copy_ce_sum", "row_copy_count",
            "row_frequency_ce_sum", "row_frequency_count",
        )
    reduction_sha256 = runtime.logical_identity_sha256({
        name: runtime.tensor_identity_sha256(getattr(reductions, name)) for name in order
    })
    receipt = observed.ObservedMaterializedFinalProgramBatchReceipt(
        action_key=action.key,
        final_action_identity_sha256=synthetic.action_identity_sha256,
        materialization_sha256="2" * 64, binding_sha256="3" * 64,
        runtime_identity_sha256="4" * 64, runtime_receipt_sha256="5" * 64,
        reduction_sha256=reduction_sha256,
        frequency_assignment_sha256="6" * 64, batch_ordinal=0,
    )
    joined = role.observational_batch_from_backend(
        action=action, common_support_sha256=SUPPORT,
        reductions=reductions, receipt=receipt,
    )
    assert joined.action == action and joined.batch_ordinal == 0
    assert joined.row_frequency_count.sum().item() == 4 * 192

    forged = observed.ObservedMaterializedFinalProgramBatchReceipt(
        **{
            field: ("9" * 64 if field == "reduction_sha256" else getattr(receipt, field))
            for field in receipt.__dataclass_fields__
        }
    )
    with pytest.raises(RuntimeError, match="differs from its receipt"):
        role.observational_batch_from_backend(
            action=action, common_support_sha256=SUPPORT,
            reductions=reductions, receipt=forged,
        )
