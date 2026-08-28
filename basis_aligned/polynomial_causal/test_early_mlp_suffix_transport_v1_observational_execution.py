from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

import bilin18_observed_adapter as observed
import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_consumer_norms as consumer_norms
import early_mlp_suffix_transport_v1_diagnostic_integration as diagnostic_integration
import early_mlp_suffix_transport_v1_final_actions as final_actions
import early_mlp_suffix_transport_v1_final_capability as final_capability
import early_mlp_suffix_transport_v1_fit as fit
import early_mlp_suffix_transport_v1_inherited as inherited
import early_mlp_suffix_transport_v1_observational_execution as execution
import early_mlp_suffix_transport_v1_runtime as runtime


def _program(route: str, marker: float) -> runtime.JointAffineProgram:
    def site(value: float) -> runtime.AffineCodeProgram:
        weight = torch.zeros(runtime.D_MODEL, runtime.CODE_DIM)
        weight[0, 0] = value
        return runtime.AffineCodeProgram(
            mean=torch.zeros(runtime.D_MODEL), scale=torch.ones(runtime.D_MODEL),
            weight=weight, bias=torch.full((runtime.CODE_DIM,), value / 100),
        )

    value = runtime.JointAffineProgram(site(marker), site(marker + 0.5), route=route)
    if route == "T":
        with torch.no_grad():
            value.cross.fill_(marker / 1000)
    return value


def _sources() -> final_actions.FinalProgramSourceBank:
    routes = {
        "inherited_q": "L", "true/L": "L", "true/R": "R",
        "true/S0": "S0", "true/S1": "S1", "true/T": "T",
        "mapped/document_shuffle/L": "L", "mapped/document_shuffle/R": "R",
        **{f"mapped/A_null_{index:02d}/T": "T" for index in range(20)},
        "new_fit_mean": "L",
    }
    return final_actions.FinalProgramSourceBank({
        key: _program(routes[key], float(index + 1))
        for index, key in enumerate(final_actions.SOURCE_PROGRAM_KEYS)
    })


def _initialization() -> inherited.LoadedInitialization:
    bases = {
        site: torch.eye(runtime.D_MODEL)[:, :runtime.CODE_DIM].contiguous()
        for site in (0, 1)
    }
    states = {
        site: {
            "grammar": "affine", "interface": "state_complete_p",
            "mean": torch.zeros(runtime.D_MODEL),
            "scale": torch.ones(runtime.D_MODEL),
            "left": torch.zeros(runtime.D_MODEL, runtime.CODE_DIM),
            "right": torch.eye(runtime.CODE_DIM),
            "bias": torch.zeros(runtime.CODE_DIM),
        } for site in (0, 1)
    }
    authority = inherited.ValidatedInherited(
        bindings={}, ship_realization_sha256="1" * 64,
        compiler_source_commit="2" * 40, basis_source_commit="3" * 40,
        snapshot_sha256=inherited._tensor_tree_hash(bases, states),
        full_product_sha256={
            site: inherited.raw_tensor_sha256(
                states[site]["left"] @ states[site]["right"]
            ) for site in (0, 1)
        },
    )
    return inherited.LoadedInitialization(bases, states, authority)


def _denominator_pass() -> fit.DenominatorPass:
    count = capabilities.FIT_ROW_COUNT * (
        runtime.SCORE_STOP - runtime.SCORE_START
    )
    records = []
    for site, denominator in enumerate((2.0, 4.0)):
        records.append({
            "count": count,
            "coordinate_sum": torch.zeros(runtime.CODE_DIM, dtype=torch.float64),
            "coordinate_square_sum": torch.ones(runtime.CODE_DIM, dtype=torch.float64),
            "mean": torch.zeros(runtime.CODE_DIM, dtype=torch.float64),
            "centered_sum_of_squares": torch.tensor(1.0 + site, dtype=torch.float64),
            "raw_sum_square_replay": torch.tensor(1.0 + site, dtype=torch.float64),
            "denominator": torch.tensor(denominator, dtype=torch.float64),
            "ordered_support_sha256": "4" * 64,
        })
    return fit.DenominatorPass(
        site_records=tuple(records), transaction_history_sha256="5" * 64,
        completed_steps=capabilities.FIT_BATCHES_PER_EPOCH,
    )


def _rows() -> torch.Tensor:
    return (torch.arange(execution.FINAL_ROW_COUNT * execution.FINAL_ROW_WIDTH)
            .view(execution.FINAL_ROW_COUNT, execution.FINAL_ROW_WIDTH)
            % 97).long().contiguous()


def _frequency(rows: torch.Tensor) -> execution.FinalFrequencyPlan:
    counts = torch.zeros(execution.TOKEN_VOCAB, dtype=torch.long)
    counts[:97] = torch.arange(97, dtype=torch.long)
    return execution.FinalFrequencyPlan(
        fit_token_counts=counts,
        fit_token_counts_sha256=runtime.tensor_identity_sha256(counts),
        source_authority_sha256="6" * 64, final_rows=rows,
        final_role_tensor_sha256=runtime.tensor_identity_sha256(rows),
    )


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Linear(1, 1)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([_Block(), _Block()])


class _Ship:
    def attention(self, event):
        return event

    def mlp(self, event):
        return event


def _validated(denominator_sha256: str) -> dict:
    return {
        "true_programs": {}, "mapped_programs": {},
        "new_fit_mean": SimpleNamespace(fit_moments_sha256=denominator_sha256),
        "payload_sha256": "7" * 64, "validation_baseline": None,
        "validation_execution": None, "transport_geometry": None,
        "teacher_calibration": None,
    }


def _advance_broker(broker, identity_sha256: str) -> None:
    for name in (
        "_CapabilityBroker__student_identities",
        "_CapabilityBroker__teacher_identities",
        "_CapabilityBroker__completed_identities",
    ):
        object.__getattribute__(broker, name).add(identity_sha256)
    previous = broker.ledger_snapshot.rolling_ledger_sha256
    object.__setattr__(
        broker, "_CapabilityBroker__rolling_ledger_sha256",
        runtime.logical_identity_sha256([previous, identity_sha256]),
    )


def _reduction_and_receipt(kwargs, calls):
    identity = kwargs["identity"]
    materialized = kwargs["materialized"]
    bins = kwargs["frequency_bins"]
    action_key = identity.action_key
    calls.append((action_key, identity.batch_ordinal, kwargs.get("denominators")))
    runtime_identity = runtime.logical_identity_sha256({
        "final_action": identity.sha256, "test_runtime": True,
    })
    _advance_broker(kwargs["broker"], runtime_identity)
    frequency_count = torch.stack([
        torch.bincount(row, minlength=9) for row in bins
    ]).long()
    frequency_sum = frequency_count.double()
    common = {
        "identity_sha256": runtime_identity,
        "route": materialized.make_program().route,
        "program_sha256": materialized.program_sha256,
        "row_ce_sum": torch.full((4,), 192.0, dtype=torch.float64),
        "row_ce_count": torch.full((4,), 192, dtype=torch.long),
        "row_copy_ce_sum": torch.ones(4, dtype=torch.float64),
        "row_copy_count": torch.ones(4, dtype=torch.long),
        "row_frequency_ce_sum": frequency_sum,
        "row_frequency_count": frequency_count,
    }
    if action_key.endswith("/N"):
        reductions = capabilities.FinalBatchReductions(
            row_primary_sum=torch.ones(4, dtype=torch.float64),
            row_primary_count=torch.ones(4, dtype=torch.long), **common,
        )
        fields = (
            "row_primary_sum", "row_primary_count", "row_ce_sum", "row_ce_count",
            "row_copy_ce_sum", "row_copy_count", "row_frequency_ce_sum",
            "row_frequency_count",
        )
    else:
        reductions = capabilities.FinalCEBatchReductions(**common)
        fields = (
            "row_ce_sum", "row_ce_count", "row_copy_ce_sum", "row_copy_count",
            "row_frequency_ce_sum", "row_frequency_count",
        )
    reduction_sha256 = runtime.logical_identity_sha256({
        name: runtime.tensor_identity_sha256(getattr(reductions, name))
        for name in fields
    })
    receipt = observed.ObservedMaterializedFinalProgramBatchReceipt(
        action_key=action_key, final_action_identity_sha256=identity.sha256,
        materialization_sha256=materialized.sha256, binding_sha256="8" * 64,
        runtime_identity_sha256=runtime_identity, runtime_receipt_sha256="9" * 64,
        reduction_sha256=reduction_sha256,
        frequency_assignment_sha256=runtime.tensor_identity_sha256(bins),
        batch_ordinal=identity.batch_ordinal,
    )
    return reductions, receipt


def _captured_magnitudes(identity, support, magnitude=1.0):
    action = final_capability.FinalAction(*identity.action_key.split("/"))
    values = torch.full((18, 4), magnitude, dtype=torch.float64)
    receipt = consumer_norms.ConsumerCaptureReceipt(
        action_key=action.key, action_sha256=action.sha256,
        action_identity_sha256=identity.sha256,
        model_identity_sha256="d" * 64, component_identity_sha256="e" * 64,
        common_support_sha256=support, batch_ordinal=identity.batch_ordinal,
        magnitude_sha256=runtime.tensor_identity_sha256(values),
        hook_calls=tuple((layer, 1) for layer in range(18)),
        hooks_removed=True, hooks_inert=True,
    )
    return consumer_norms._CapturedConsumerMagnitudes(
        token=consumer_norms._MINT_TOKEN, action=action,
        magnitudes=values, receipt=receipt,
    )


def _baseline_reduction_receipt(kwargs, calls, *, captured):
    identity = kwargs["identity"]
    bins = kwargs["frequency_bins"]
    calls.append((identity.action_key, identity.batch_ordinal))
    frequency_count = torch.stack([
        torch.bincount(row, minlength=9) for row in bins
    ]).long()
    frequency_sum = frequency_count.double()
    n_background = identity.action_key.endswith("/N")
    reductions = observed.ObservedFinalBaselineBatchReductions(
        identity_sha256=identity.sha256, action_key=identity.action_key,
        row_primary_sum=(torch.ones(4, dtype=torch.float64) if n_background else None),
        row_primary_count=(torch.ones(4, dtype=torch.long) if n_background else None),
        row_ce_sum=torch.full((4,), 192.0, dtype=torch.float64),
        row_ce_count=torch.full((4,), 192, dtype=torch.long),
        row_copy_ce_sum=torch.ones(4, dtype=torch.float64),
        row_copy_count=torch.ones(4, dtype=torch.long),
        row_frequency_ce_sum=frequency_sum,
        row_frequency_count=frequency_count,
    )
    fields = (
        "row_primary_sum", "row_primary_count", "row_ce_sum", "row_ce_count",
        "row_copy_ce_sum", "row_copy_count", "row_frequency_ce_sum",
        "row_frequency_count",
    ) if n_background else (
        "row_ce_sum", "row_ce_count", "row_copy_ce_sum", "row_copy_count",
        "row_frequency_ce_sum", "row_frequency_count",
    )
    reduction_sha256 = runtime.logical_identity_sha256({
        "action_key": identity.action_key,
        **{
            name: runtime.tensor_identity_sha256(getattr(reductions, name))
            for name in fields
        },
    })
    receipt = observed.ObservedFinalBaselineBatchReceipt(
        identity_sha256=identity.sha256, action_key=identity.action_key,
        batch_ordinal=identity.batch_ordinal,
        ordered_row_indices_sha256=runtime.logical_identity_sha256(
            list(kwargs["ordered_row_indices"])
        ),
        reduction_sha256=reduction_sha256,
        frequency_assignment_sha256=runtime.tensor_identity_sha256(bins),
        observed_student_closure_sha256="f" * 64,
        observed_teacher_closure_sha256=("1" * 64 if n_background else None),
        teacher_reused_student=n_background,
    )
    if not captured:
        return reductions, receipt
    return reductions, receipt, _captured_magnitudes(
        identity, identity.common_support_sha256,
    )


def _executor(monkeypatch):
    rows = _rows()
    initialization = _initialization()
    denominator = _denominator_pass()
    sources = _sources()
    adapter = observed.ObservedBilin18Adapter(_TinyModel(), _Ship(), production=False)
    calls = []
    monkeypatch.setattr(
        final_actions, "source_bank_from_validated",
        lambda validated, *, inherited_q: sources,
    )
    adapter.run_materialized_final_program_batch = MethodType(
        lambda self, **kwargs: _reduction_and_receipt(kwargs, calls), adapter,
    )
    context = capabilities.FinalRunContext(
        source_commit="a" * 40,
        inherited_snapshot_sha256=initialization.authority.snapshot_sha256,
        rows_receipt_sha256="b" * 64,
        final_role_tensor_sha256=runtime.tensor_identity_sha256(rows),
        identity_teacher_mapping_sha256="c" * 64,
    )
    result = execution.FinalObservationalBatchExecutor(
        adapter=adapter,
        validated_program_bank=_validated(denominator.sha256),
        inherited_initialization=initialization, final_context=context,
        final_rows=rows, denominator_pass=denominator,
        frequency_plan=_frequency(rows),
    )
    return result, calls, rows, denominator, initialization, adapter, context


def test_frequency_plan_uses_exact_target_columns_and_right_boundaries() -> None:
    rows = torch.zeros(execution.FINAL_ROW_COUNT, execution.FINAL_ROW_WIDTH, dtype=torch.long)
    rows[:, 64] = 96
    rows[:, 65:257] = torch.tensor(list(range(192))) % 9
    rows[:, 257] = 96
    rows = rows.contiguous()
    counts = torch.zeros(execution.TOKEN_VOCAB, dtype=torch.long)
    counts[:9] = torch.tensor([0, 1, 2, 4, 8, 16, 32, 64, 128])
    plan = execution.FinalFrequencyPlan(
        fit_token_counts=counts,
        fit_token_counts_sha256=runtime.tensor_identity_sha256(counts),
        source_authority_sha256="d" * 64, final_rows=rows,
        final_role_tensor_sha256=runtime.tensor_identity_sha256(rows),
    )
    first = plan.batch(0)
    assert first[0, :9].tolist() == list(range(9))
    assert first[0, 191] == 2
    first.zero_()
    assert plan.batch(0)[0, :9].tolist() == list(range(9))


def test_frequency_plan_rejects_hash_role_and_vocabulary_substitution() -> None:
    rows = _rows()
    counts = torch.ones(execution.TOKEN_VOCAB, dtype=torch.long)
    with pytest.raises(ValueError, match="count authority"):
        execution.FinalFrequencyPlan(
            fit_token_counts=counts, fit_token_counts_sha256="e" * 64,
            source_authority_sha256="f" * 64, final_rows=rows,
            final_role_tensor_sha256=runtime.tensor_identity_sha256(rows),
        )
    changed = rows.clone()
    changed[0, 65] = execution.TOKEN_VOCAB
    with pytest.raises(ValueError, match="vocabulary"):
        execution.FinalFrequencyPlan(
            fit_token_counts=counts,
            fit_token_counts_sha256=runtime.tensor_identity_sha256(counts),
            source_authority_sha256="f" * 64, final_rows=changed,
            final_role_tensor_sha256=runtime.tensor_identity_sha256(changed),
        )


def test_real_executor_binds_n_denominators_e_absence_and_runtime_identity(monkeypatch) -> None:
    executor, calls, *_ = _executor(monkeypatch)
    for ordinal in range(execution.FINAL_BATCH_COUNT):
        batch = executor(final_capability.CANONICAL_ACTIONS[0], ordinal)
        assert batch.action.key == "qq/N" and batch.batch_ordinal == ordinal
    for ordinal in range(execution.FINAL_BATCH_COUNT):
        batch = executor(final_capability.CANONICAL_ACTIONS[1], ordinal)
        assert batch.action.key == "qq/E" and batch.batch_ordinal == ordinal
    assert len(calls) == 96
    assert all(value[2] is not None for value in calls[:48])
    assert all(value[2] is None for value in calls[48:])


def test_real_executor_poison_closes_on_batch_reorder(monkeypatch) -> None:
    executor, *_ = _executor(monkeypatch)
    action = final_capability.CANONICAL_ACTIONS[0]
    executor(action, 0)
    with pytest.raises(RuntimeError, match="order changed"):
        executor(action, 2)
    with pytest.raises(RuntimeError, match="order changed"):
        executor(action, 1)


def test_source_closed_native_prepass_is_exact_96_and_main_action_is_captured(
    monkeypatch,
) -> None:
    executor, program_calls, *_rest, adapter, _context = _executor(monkeypatch)
    native_calls = []
    adapter.run_final_baseline_batch_captured = MethodType(
        lambda self, **kwargs: _baseline_reduction_receipt(
            kwargs, native_calls, captured=True,
        ), adapter,
    )

    native = executor.make_native_denominator_executor()
    observed_schedule = []
    for action, ordinal in diagnostic_integration.NATIVE_SCHEDULE:
        value = native(action, ordinal)
        observed_schedule.append((value.batch.action.key, value.batch.batch_ordinal))
    assert observed_schedule == [
        (action.key, ordinal)
        for action, ordinal in diagnostic_integration.NATIVE_SCHEDULE
    ]
    assert native_calls == observed_schedule and len(native_calls) == 96

    def captured_program(self, **kwargs):
        reductions, receipt = _reduction_and_receipt(kwargs, program_calls)
        return reductions, receipt, _captured_magnitudes(
            kwargs["identity"], kwargs["identity"].common_support_sha256,
            magnitude=1.5,
        )

    adapter.run_materialized_final_program_batch_captured = MethodType(
        captured_program, adapter,
    )
    first = executor.run_captured(final_capability.CANONICAL_ACTIONS[0], 0)
    assert first.batch.action.key == "qq/N" and first.batch.batch_ordinal == 0
    assert first.final_context_sha256 == executor._context.sha256
    assert first.program_payload_sha256 == "7" * 64
    assert program_calls == [("qq/N", 0, executor._denominators)]


def test_real_executor_rejects_context_frequency_and_denominator_mismatch(monkeypatch) -> None:
    executor, calls, rows, denominator, initialization, adapter, context = _executor(monkeypatch)
    assert executor.common_support_sha256 and not calls
    bad_context = capabilities.FinalRunContext(
        source_commit=context.source_commit,
        inherited_snapshot_sha256=context.inherited_snapshot_sha256,
        rows_receipt_sha256=context.rows_receipt_sha256,
        final_role_tensor_sha256="1" * 64,
        identity_teacher_mapping_sha256=context.identity_teacher_mapping_sha256,
    )
    with pytest.raises(RuntimeError, match="sealed context"):
        execution.FinalObservationalBatchExecutor(
            adapter=adapter,
            validated_program_bank=_validated(denominator.sha256),
            inherited_initialization=initialization, final_context=bad_context,
            final_rows=rows, denominator_pass=denominator,
            frequency_plan=_frequency(rows),
        )
    with pytest.raises(RuntimeError, match="frozen fit moments"):
        execution.FinalObservationalBatchExecutor(
            adapter=adapter, validated_program_bank=_validated("2" * 64),
            inherited_initialization=initialization, final_context=context,
            final_rows=rows, denominator_pass=denominator,
            frequency_plan=_frequency(rows),
        )
