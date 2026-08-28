from __future__ import annotations

import inspect

import pytest
import torch

import early_mlp_suffix_transport_v1_consumer_norms as consumer
import early_mlp_suffix_transport_v1_diagnostic_integration as integration
import early_mlp_suffix_transport_v1_final_capability as final_capability
import early_mlp_suffix_transport_v1_observational_role as observational_role
import early_mlp_suffix_transport_v1_response_execution as response_execution
import early_mlp_suffix_transport_v1_response_plan as response_plan
import early_mlp_suffix_transport_v1_response_reductions as response_reductions
import early_mlp_suffix_transport_v1_runtime as runtime


SUPPORT = "4" * 64
MODEL = "5" * 64
COMPONENTS = "6" * 64
PROGRAM = "2" * 64


def _sha(value) -> str:
    return runtime.logical_identity_sha256(value)


def _observational_batch(
    action: final_capability.FinalAction, ordinal: int, *, stream: str,
    frequency_override: str | None = None,
) -> observational_role.FinalObservationalBatch:
    identity = _sha([stream, "identity", action.key, ordinal])
    ce_sum = torch.full((4,), 1.0 + ordinal / 100.0, dtype=torch.float64)
    ce_count = torch.full((4,), 192, dtype=torch.long)
    frequency_sum = torch.zeros((4, 9), dtype=torch.float64)
    frequency_count = torch.zeros((4, 9), dtype=torch.long)
    frequency_sum[:, 0] = ce_sum
    frequency_count[:, 0] = ce_count
    return observational_role.FinalObservationalBatch(
        action=action, batch_ordinal=ordinal,
        common_support_sha256=SUPPORT,
        action_identity_sha256=identity,
        backend_receipt_sha256=_sha([stream, "backend", action.key, ordinal]),
        frequency_assignment_sha256=(
            frequency_override or _sha(["frequency", ordinal])
        ),
        row_primary_sum=(
            torch.full((4,), 0.5, dtype=torch.float64)
            if action.background == "N" else None
        ),
        row_primary_count=(
            torch.full((4,), 192, dtype=torch.long)
            if action.background == "N" else None
        ),
        row_ce_sum=ce_sum, row_ce_count=ce_count,
        row_copy_ce_sum=torch.full((4,), 0.25, dtype=torch.float64),
        row_copy_count=torch.full((4,), 192, dtype=torch.long),
        row_frequency_ce_sum=frequency_sum,
        row_frequency_count=frequency_count,
    )


def _direct_capture(
    batch: observational_role.FinalObservationalBatch, magnitude: float,
) -> consumer._CapturedConsumerMagnitudes:
    magnitudes = torch.full((18, 4), magnitude, dtype=torch.float64)
    receipt = consumer.ConsumerCaptureReceipt(
        action_key=batch.action.key, action_sha256=batch.action.sha256,
        action_identity_sha256=batch.action_identity_sha256,
        model_identity_sha256=MODEL, component_identity_sha256=COMPONENTS,
        common_support_sha256=batch.common_support_sha256,
        batch_ordinal=batch.batch_ordinal,
        magnitude_sha256=runtime.tensor_identity_sha256(magnitudes),
        hook_calls=tuple((layer, 1) for layer in range(18)),
        hooks_removed=True, hooks_inert=True,
    )
    return consumer._CapturedConsumerMagnitudes(
        token=consumer._MINT_TOKEN, action=batch.action,
        magnitudes=magnitudes, receipt=receipt,
    )


def _captured(
    action: final_capability.FinalAction, ordinal: int, *, stream: str,
    magnitude: float, frequency_override: str | None = None,
) -> integration.CapturedObservationalBatch:
    batch = _observational_batch(
        action, ordinal, stream=stream,
        frequency_override=frequency_override,
    )
    return integration.CapturedObservationalBatch(
        _token=integration._MINT_TOKEN, batch=batch,
        capture=_direct_capture(batch, magnitude),
        final_context_sha256="3" * 64, program_payload_sha256=PROGRAM,
    )


def _native_magnitude(background: str, ordinal: int) -> float:
    return (2.0 if background == "N" else 4.0) + ordinal / 100.0


def _response_batch(ordinal: int) -> response_execution.ObservedResponseBatchResult:
    unit = _sha(["unit", ordinal])
    plan = _sha(["plan", ordinal])
    forwards = tuple(_sha(["forward", ordinal, index]) for index in range(69))
    arms = []
    offset = 3
    for action_key in response_plan.RESPONSE_ACTION_KEYS:
        vector = response_reductions.BatchResponseReduction(
            error_sum=torch.ones(4, dtype=torch.float64),
            teacher_sum=torch.full((4,), 4.0, dtype=torch.float64),
            student_sum=torch.ones(4, dtype=torch.float64),
            dot_sum=torch.full((4,), 2.0, dtype=torch.float64),
            unit_identity=unit,
        )
        output = response_reductions.BatchOutputKLReduction(
            numerator_sum=torch.ones(4, dtype=torch.float64),
            denominator_sum=torch.full((4,), 2.0, dtype=torch.float64),
            unit_identity=unit,
        )
        arms.append(response_execution.ObservedResponseArmReduction(
            action_key=action_key, batch_plan_sha256=plan,
            teacher_forward_receipt_sha256s=forwards[:3],
            student_forward_receipt_sha256s=forwards[offset:offset + 3],
            code_response=(vector if action_key in {"ll/N", "lt/N"} else None),
            logit_response=vector, output_kl_response=output,
        ))
        offset += 3
    receipt = response_execution.ObservedResponseBatchReceipt(
        batch_ordinal=ordinal, batch_plan_sha256=plan,
        source_bank_sha256="1" * 64, program_payload_sha256=PROGRAM,
        final_context_sha256="3" * 64, common_support_sha256=SUPPORT,
        basis0_sha256="7" * 64, basis1_sha256="8" * 64,
        forward_receipt_sha256s=forwards,
        arm_reduction_sha256s=tuple((value.action_key, value.sha256) for value in arms),
        broker_ledger_sha256=_sha(["broker", ordinal]),
        teacher_forward_count=3, student_forward_count=66, atomic_complete=True,
    )
    return response_execution.ObservedResponseBatchResult(
        arm_reductions=tuple(arms), receipt=receipt,
    )


def _response_run() -> response_execution.ObservedResponseRunResult:
    accumulator = response_execution.ObservedResponseRunAccumulator()
    for ordinal in range(48):
        accumulator.add(_response_batch(ordinal))
    return accumulator.finish()


class _Projection(torch.nn.Module):
    def forward(self, value):
        return value.expand(-1, -1, consumer.MODEL_WIDTH)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for _layer in range(18):
            block = torch.nn.Module()
            block.attn = torch.nn.Module()
            block.attn.c_proj = _Projection()
            blocks.append(block)
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList(blocks)

    def forward(self, value):
        for block in self.transformer.h:
            block.attn.c_proj(value)


def test_capture_binding_wraps_exactly_one_authorized_forward() -> None:
    model = _Model()
    action = final_capability.FinalAction("qq", "N")
    batch = _observational_batch(action, 0, stream="bound")
    state = {"action": batch.action_identity_sha256, "model": MODEL}
    capture = consumer.AttentionConsumerOutputCapture(
        model=model, action=action, batch_ordinal=0,
        common_support_sha256=SUPPORT,
        expected_action_identity_sha256=state["action"],
        action_identity_reader=lambda: state["action"],
        expected_model_identity_sha256=MODEL,
        model_identity_reader=lambda: state["model"],
    )
    calls = []

    def forward():
        calls.append("forward")
        model(torch.ones(4, 256, 1))
        return batch

    bound = integration.bind_consumer_capture_context(
        capture=capture, forward=forward,
        final_context_sha256="3" * 64, program_payload_sha256=PROGRAM,
    )
    assert calls == ["forward"]
    assert bound.batch.sha256 == batch.sha256
    assert all(not block.attn.c_proj._forward_hooks for block in model.transformer.h)
    with pytest.raises(RuntimeError, match="already consumed"):
        bound._take_capture(object())


def test_native_cache_is_exact_96_forward_background_batch_schedule() -> None:
    calls = []

    def execute(action, ordinal):
        calls.append((action.key, ordinal))
        return _captured(
            action, ordinal, stream="native",
            magnitude=_native_magnitude(action.background, ordinal),
        )

    cache = integration.build_native_denominator_cache(
        common_support_sha256=SUPPORT, final_context_sha256="3" * 64,
        program_payload_sha256=PROGRAM, executor=execute,
    )
    expected = [
        (f"o_o/{background}", ordinal)
        for background in ("N", "E") for ordinal in range(48)
    ]
    assert calls == expected
    assert cache.receipt.forward_count == 96
    assert cache.receipt.schedule == tuple(expected)
    assert cache.receipt.authorized_for_selection is False
    assert cache._entry(final_capability.FinalAction("qq", "N"), 17).capture.receipt == (
        cache._entry(final_capability.FinalAction("ll", "N"), 17).capture.receipt
    )
    assert cache._entry(final_capability.FinalAction("qq", "N"), 17).capture.receipt != (
        cache._entry(final_capability.FinalAction("qq", "E"), 17).capture.receipt
    )


def test_full_owner_reuses_cache_preserves_canonical_order_and_joins_response() -> None:
    native_calls = []
    action_calls = []
    action_index = {
        action.key: index for index, action in enumerate(final_capability.CANONICAL_ACTIONS)
    }

    def native_executor(action, ordinal):
        native_calls.append((action.key, ordinal))
        return _captured(
            action, ordinal, stream="native",
            magnitude=_native_magnitude(action.background, ordinal),
        )

    def action_executor(action, ordinal):
        action_calls.append((action.key, ordinal))
        factor = 1.0 if action.arm == "o_o" else 1.1 + action_index[action.key] / 100.0
        return _captured(
            action, ordinal, stream="action",
            magnitude=_native_magnitude(action.background, ordinal) * factor,
        )

    owner = integration.IntegratedDiagnosticOwner(
        issuer_id="9" * 64, common_support_sha256=SUPPORT,
        native_executor=native_executor, action_executor=action_executor,
        response_run=_response_run(),
    )
    result = owner.execute_all()
    assert native_calls == [
        (f"o_o/{background}", ordinal)
        for background in ("N", "E") for ordinal in range(48)
    ]
    assert action_calls == [
        (action.key, ordinal) for action in final_capability.CANONICAL_ACTIONS
        for ordinal in range(48)
    ]
    assert result.receipt.native_forward_count == 96
    assert result.receipt.action_forward_count == 68 * 48
    assert result.receipt.final_role_load_authorized is False
    assert len(result.observations.observations) == 68
    by_action = {value.action.key: value for value in result.observations.observations}
    assert torch.allclose(
        by_action["qq/N"].consumer_norm_ratio[0].row_sum,
        torch.full((192,), 1.1, dtype=torch.float64), atol=1e-12,
    )
    assert torch.equal(
        by_action["o_o/E"].consumer_norm_ratio[17].row_sum,
        torch.ones(192, dtype=torch.float64),
    )
    assert by_action["ll/N"].code_response is not None
    assert by_action["a_null_19/N"].logit_response is not None
    assert by_action["ll/E"].code_response is None
    assert result.evidence_join.response_run_receipt_sha256 == (
        result.receipt.response_run_receipt_sha256
    )
    with pytest.raises(RuntimeError, match="already closed"):
        owner.execute_all()


def test_mixed_native_support_poison_closes_owner() -> None:
    calls = 0

    def native_executor(action, ordinal):
        nonlocal calls
        calls += 1
        changed = 1 if calls == 7 else ordinal
        return _captured(
            action, changed, stream="native",
            magnitude=_native_magnitude(action.background, changed),
        )

    owner = integration.IntegratedDiagnosticOwner(
        issuer_id="9" * 64, common_support_sha256=SUPPORT,
        native_executor=native_executor,
        action_executor=lambda *_args: (_ for _ in ()).throw(AssertionError()),
        response_run=_response_run(),
    )
    with pytest.raises(RuntimeError, match="schedule, support, or authority"):
        owner.execute_all()
    assert calls == 7
    with pytest.raises(RuntimeError, match="already closed"):
        owner.execute_all()


def test_action_frequency_mismatch_fails_before_second_action_batch() -> None:
    action_calls = []

    def native_executor(action, ordinal):
        return _captured(
            action, ordinal, stream="native",
            magnitude=_native_magnitude(action.background, ordinal),
        )

    def action_executor(action, ordinal):
        action_calls.append((action.key, ordinal))
        return _captured(
            action, ordinal, stream="action", magnitude=3.0,
            frequency_override=_sha("wrong-frequency"),
        )

    owner = integration.IntegratedDiagnosticOwner(
        issuer_id="9" * 64, common_support_sha256=SUPPORT,
        native_executor=native_executor, action_executor=action_executor,
        response_run=_response_run(),
    )
    with pytest.raises(RuntimeError, match="support changed"):
        owner.execute_all()
    assert action_calls == [("qq/N", 0)]


def test_integration_source_has_no_role_loader_or_open_data_authority() -> None:
    source = inspect.getsource(integration)
    assert "load_roles(" not in source
    assert "torch.load(" not in source
    assert "load_bilin18(" not in source
    assert "authorized_for_selection: bool = False" in source
    assert "final_role_load_authorized: bool = False" in source
