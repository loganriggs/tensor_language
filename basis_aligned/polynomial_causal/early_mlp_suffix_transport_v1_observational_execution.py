"""Source-closed observed executor for the canonical suffix final lattice.

This module performs no artifact, row, model, or checkpoint loading.  It accepts the
already validated program/inherited authorities, the one-shot final tensor, the
frozen fit denominator pass, and a hash-bound token-count authority.  It alone joins
those objects to the existing observed program/baseline backends one canonical batch
at a time.  Only :class:`FinalObservationalBatch` values cross the boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch

import bilin18_observed_adapter as observed
import early_mlp_suffix_transport_v1_capabilities as capabilities
import early_mlp_suffix_transport_v1_diagnostic_integration as diagnostic_integration
import early_mlp_suffix_transport_v1_final_actions as final_actions
import early_mlp_suffix_transport_v1_final_capability as final_capability
import early_mlp_suffix_transport_v1_fit as fit
import early_mlp_suffix_transport_v1_inherited as inherited
import early_mlp_suffix_transport_v1_observational_role as observational_role
import early_mlp_suffix_transport_v1_runtime as runtime


FINAL_ROW_COUNT = capabilities.FINAL_ROW_COUNT
FINAL_ROW_WIDTH = 513
FINAL_BATCH_COUNT = capabilities.FINAL_BATCH_COUNT
TOKEN_VOCAB = 50257
# The preregistration inherits the already prospective v2.1 nine-bin definition.
TOKEN_FREQUENCY_BOUNDARIES = (1, 2, 4, 8, 16, 32, 64, 128)
PROGRAM_ACTION_COUNT = sum(
    plan.arm_plan.execution_kind in {"projected_program", "mean_program"}
    for plan in final_actions.CANONICAL_ACTION_PLANS
)
PROGRAM_BATCH_COUNT = PROGRAM_ACTION_COUNT * FINAL_BATCH_COUNT
BASELINE_BATCH_COUNT = (
    len(final_actions.CANONICAL_ACTION_PLANS) - PROGRAM_ACTION_COUNT
) * FINAL_BATCH_COUNT


def _sha256(name: str, value: Any) -> str:
    if not runtime._sha256_text(value):
        raise ValueError(f"{name} must be a SHA-256 identity")
    return value


class FinalFrequencyPlan:
    """Immutable final assignment derived from a hash-bound fit count vector.

    The count-vector loader remains outside this no-I/O module.  Supplying its
    expected hash and source-authority hash prevents an arbitrary assignment tensor
    from being substituted at final execution.  Final targets are fixed by the
    amended common support: raw row columns 65:257.
    """

    __slots__ = (
        "__assignments", "__assignment_sha256", "__bin_counts",
        "__final_role_tensor_sha256", "__fit_token_counts_sha256",
        "__source_authority_sha256",
    )

    def __init__(
        self, *, fit_token_counts: torch.Tensor,
        fit_token_counts_sha256: str, source_authority_sha256: str,
        final_rows: torch.Tensor, final_role_tensor_sha256: str,
    ) -> None:
        _sha256("fit token-count identity", fit_token_counts_sha256)
        _sha256("frequency source authority", source_authority_sha256)
        _sha256("frequency final role", final_role_tensor_sha256)
        if not torch.is_tensor(fit_token_counts) or fit_token_counts.dtype != torch.long or (
            tuple(fit_token_counts.shape) != (TOKEN_VOCAB,)
        ) or fit_token_counts.device.type != "cpu" or fit_token_counts.requires_grad or (
            bool((fit_token_counts < 0).any())
        ) or int(fit_token_counts.sum()) <= 0 or runtime.tensor_identity_sha256(
            fit_token_counts
        ) != fit_token_counts_sha256:
            raise ValueError("frequency fit token-count authority changed")
        if not torch.is_tensor(final_rows) or final_rows.dtype != torch.long or tuple(
            final_rows.shape
        ) != (FINAL_ROW_COUNT, FINAL_ROW_WIDTH) or final_rows.device.type != "cpu" or (
            not final_rows.is_contiguous()
        ) or runtime.tensor_identity_sha256(final_rows) != final_role_tensor_sha256:
            raise ValueError("frequency final-role authority changed")
        targets = final_rows[:, runtime.SCORE_START + 1:runtime.SCORE_STOP + 1]
        if bool((targets < 0).any()) or bool((targets >= TOKEN_VOCAB).any()):
            raise ValueError("frequency final target is outside the GPT-2 vocabulary")
        counts = fit_token_counts.detach().clone().contiguous()
        assignments = torch.bucketize(
            counts.index_select(0, targets.flatten()),
            torch.tensor(TOKEN_FREQUENCY_BOUNDARIES, dtype=torch.long), right=True,
        ).view(FINAL_ROW_COUNT, runtime.SCORE_STOP - runtime.SCORE_START).contiguous()
        object.__setattr__(self, "_FinalFrequencyPlan__assignments", assignments)
        object.__setattr__(
            self, "_FinalFrequencyPlan__assignment_sha256",
            runtime.tensor_identity_sha256(assignments),
        )
        object.__setattr__(
            self, "_FinalFrequencyPlan__bin_counts",
            tuple(int(value) for value in torch.bincount(
                assignments.flatten(), minlength=len(TOKEN_FREQUENCY_BOUNDARIES) + 1,
            )),
        )
        object.__setattr__(
            self, "_FinalFrequencyPlan__fit_token_counts_sha256",
            fit_token_counts_sha256,
        )
        object.__setattr__(
            self, "_FinalFrequencyPlan__source_authority_sha256",
            source_authority_sha256,
        )
        object.__setattr__(
            self, "_FinalFrequencyPlan__final_role_tensor_sha256",
            final_role_tensor_sha256,
        )

    def __copy__(self):
        raise RuntimeError("final frequency plans cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("final frequency plans cannot be copied")

    def __reduce__(self):
        raise RuntimeError("final frequency plans cannot be serialized")

    @property
    def assignment_sha256(self) -> str:
        return self.__assignment_sha256

    @property
    def final_role_tensor_sha256(self) -> str:
        return self.__final_role_tensor_sha256

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "rule": "fit target counts; final targets at raw columns 65:257; "
                    "torch.bucketize(right=True)",
            "boundaries": list(TOKEN_FREQUENCY_BOUNDARIES),
            "fit_token_counts_sha256": self.__fit_token_counts_sha256,
            "source_authority_sha256": self.__source_authority_sha256,
            "final_role_tensor_sha256": self.__final_role_tensor_sha256,
            "assignment_sha256": self.__assignment_sha256,
            "bin_counts": list(self.__bin_counts),
        })

    def batch(self, ordinal: int) -> torch.Tensor:
        if type(ordinal) is not int or not 0 <= ordinal < FINAL_BATCH_COUNT:
            raise ValueError("frequency batch ordinal changed")
        start = ordinal * runtime.BATCH_SIZE
        value = self.__assignments[start:start + runtime.BATCH_SIZE].clone().contiguous()
        if runtime.tensor_identity_sha256(self.__assignments) != self.__assignment_sha256:
            raise RuntimeError("final frequency plan mutated")
        return value


@dataclass(frozen=True, slots=True)
class FinalObservationalExecutionReceipt:
    final_context_sha256: str
    source_bank_sha256: str
    program_payload_sha256: str
    common_support_sha256: str
    denominator_pass_sha256: str
    frequency_plan_sha256: str
    batch_result_sha256s_sha256: str
    broker_ledger_sha256: str
    program_batch_count: int
    baseline_batch_count: int

    def __post_init__(self) -> None:
        for name in (
            "final_context_sha256", "source_bank_sha256", "program_payload_sha256",
            "common_support_sha256", "denominator_pass_sha256",
            "frequency_plan_sha256", "batch_result_sha256s_sha256",
            "broker_ledger_sha256",
        ):
            _sha256(name, getattr(self, name))
        if self.program_batch_count != PROGRAM_BATCH_COUNT or (
            self.baseline_batch_count != BASELINE_BATCH_COUNT
        ):
            raise ValueError("observational execution receipt batch ledger changed")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256(asdict(self))


class FinalNativeDenominatorBatchExecutor:
    """Independent exact O/O/N then O/O/E consumer-denominator prepass."""

    def __init__(
        self, *, adapter: observed.ObservedBilin18Adapter,
        final_context: capabilities.FinalRunContext, final_rows: torch.Tensor,
        frequency_plan: FinalFrequencyPlan,
        sources: final_actions.FinalProgramSourceBank,
        program_payload_sha256: str, common_support_sha256: str,
    ) -> None:
        if not isinstance(adapter, observed.ObservedBilin18Adapter) or not isinstance(
            final_context, capabilities.FinalRunContext
        ) or not isinstance(frequency_plan, FinalFrequencyPlan) or not isinstance(
            sources, final_actions.FinalProgramSourceBank
        ):
            raise TypeError("native denominator prepass requires typed authorities")
        if not torch.is_tensor(final_rows) or final_rows.dtype != torch.long or tuple(
            final_rows.shape
        ) != (FINAL_ROW_COUNT, FINAL_ROW_WIDTH) or final_rows.device.type != "cpu" or (
            not final_rows.is_contiguous()
        ) or runtime.tensor_identity_sha256(final_rows) != final_context.final_role_tensor_sha256:
            raise RuntimeError("native denominator prepass rows changed authority")
        self._payload = _sha256("native denominator program payload", program_payload_sha256)
        self._support = _sha256("native denominator support", common_support_sha256)
        self._adapter = adapter
        self._context = final_context
        self._rows = final_rows.detach().clone().contiguous()
        self._frequency = frequency_plan
        self._sources = sources
        self._next = 0
        self._failed = False
        self._closed = False

    def _poison(self) -> None:
        self._failed = True
        self._adapter = self._context = self._rows = None
        self._frequency = self._sources = None

    def __call__(
        self, action: final_capability.FinalAction, batch_ordinal: int,
    ) -> diagnostic_integration.CapturedObservationalBatch:
        if self._failed or self._closed or self._next >= len(
            diagnostic_integration.NATIVE_SCHEDULE
        ) or (action, batch_ordinal) != diagnostic_integration.NATIVE_SCHEDULE[self._next]:
            self._poison()
            raise RuntimeError("native denominator prepass order changed")
        try:
            materialized = final_actions.materialize(
                final_actions.plan_for(action.arm, action.background), self._sources,
            )
            start = batch_ordinal * runtime.BATCH_SIZE
            indices = tuple(range(start, start + runtime.BATCH_SIZE))
            rows = self._rows[start:start + runtime.BATCH_SIZE].contiguous()
            frequency = self._frequency.batch(batch_ordinal)
            identity = final_actions.FinalActionBatchIdentity.from_role_rows(
                materialized=materialized, role_rows=rows,
                ordered_batch_indices=indices, batch_ordinal=batch_ordinal,
                source_commit=self._context.source_commit,
                inherited_snapshot_sha256=self._context.inherited_snapshot_sha256,
                rows_receipt_sha256=self._context.rows_receipt_sha256,
                final_role_tensor_sha256=self._context.final_role_tensor_sha256,
                program_payload_sha256=self._payload,
                common_support_sha256=self._support,
            )
            reductions, backend_receipt, consumer_capture = (
                self._adapter.run_final_baseline_batch_captured(
                    materialized=materialized, identity=identity, role_rows=rows,
                    ordered_row_indices=indices, frequency_bins=frequency,
                )
            )
            result = observational_role.observational_batch_from_backend(
                action=action, common_support_sha256=self._support,
                reductions=reductions, receipt=backend_receipt,
            )
            captured = diagnostic_integration.bind_completed_consumer_capture(
                batch=result, capture=consumer_capture,
                final_context_sha256=self._context.sha256,
                program_payload_sha256=self._payload,
            )
        except BaseException:
            self._poison()
            raise
        self._next += 1
        if self._next == len(diagnostic_integration.NATIVE_SCHEDULE):
            self._closed = True
            self._adapter = self._context = self._rows = None
            self._frequency = self._sources = None
        return captured


class FinalObservationalBatchExecutor:
    """One-shot real adapter join consumed by ``FinalObservationalRoleOwner``."""

    def __init__(
        self, *, adapter: observed.ObservedBilin18Adapter,
        validated_program_bank: Mapping[str, Any],
        inherited_initialization: inherited.LoadedInitialization,
        final_context: capabilities.FinalRunContext, final_rows: torch.Tensor,
        denominator_pass: fit.DenominatorPass,
        frequency_plan: FinalFrequencyPlan,
    ) -> None:
        if not isinstance(adapter, observed.ObservedBilin18Adapter) or not isinstance(
            inherited_initialization, inherited.LoadedInitialization
        ) or not isinstance(final_context, capabilities.FinalRunContext) or not isinstance(
            denominator_pass, fit.DenominatorPass
        ) or not isinstance(frequency_plan, FinalFrequencyPlan):
            raise TypeError("observational execution requires typed authorities")
        if not torch.is_tensor(final_rows) or final_rows.dtype != torch.long or tuple(
            final_rows.shape
        ) != (FINAL_ROW_COUNT, FINAL_ROW_WIDTH) or final_rows.device.type != "cpu" or (
            not final_rows.is_contiguous()
        ) or runtime.tensor_identity_sha256(final_rows) != final_context.final_role_tensor_sha256:
            raise RuntimeError("observational final role differs from its sealed context")
        if inherited_initialization.authority.snapshot_sha256 != (
            final_context.inherited_snapshot_sha256
        ) or frequency_plan.final_role_tensor_sha256 != final_context.final_role_tensor_sha256:
            raise RuntimeError("observational inherited/frequency authorities disagree")
        if not isinstance(validated_program_bank, Mapping):
            raise TypeError("observational program bank was not replay-validated")
        payload_sha256 = validated_program_bank.get("payload_sha256")
        _sha256("observational program payload", payload_sha256)
        mean = validated_program_bank.get("new_fit_mean")
        denominator_sha256 = denominator_pass.sha256
        if getattr(mean, "fit_moments_sha256", None) != denominator_sha256:
            raise RuntimeError("final local denominators differ from frozen fit moments")
        denominators = tuple(
            value.detach().cpu().double().contiguous().clone()
            for value in denominator_pass.denominators
        )
        if len(denominators) != 2 or any(
            value.numel() != 1 or not bool(torch.isfinite(value)) or float(value) <= 0
            for value in denominators
        ):
            raise RuntimeError("final local denominators are not positive frozen scalars")
        bases = inherited_initialization.clone_bases()
        inherited_q = inherited_initialization.make_program("L")
        sources = final_actions.source_bank_from_validated(
            validated_program_bank, inherited_q=inherited_q,
        )
        common_support_sha256 = runtime.logical_identity_sha256({
            "role": "early_mlp_suffix_transport_v1_final",
            "final_role_tensor_sha256": final_context.final_role_tensor_sha256,
            "rows_receipt_sha256": final_context.rows_receipt_sha256,
            "row_count": final_context.final_row_count,
            "score_start": runtime.SCORE_START,
            "score_stop": runtime.SCORE_STOP,
        })
        issuer_id = runtime.logical_identity_sha256({
            "kind": "final_observational_role",
            "final_context_sha256": final_context.sha256,
            "source_bank_sha256": sources.sha256,
            "program_payload_sha256": payload_sha256,
            "denominator_pass_sha256": denominator_sha256,
            "frequency_plan_sha256": frequency_plan.sha256,
            "common_support_sha256": common_support_sha256,
        })
        coordinator = runtime.ScopeCoordinator()
        hook = runtime.StudentCorrectionHook(
            bases, issuer_id=issuer_id, coordinator=coordinator,
        )
        broker = adapter.make_capability_broker(
            issuer_id=issuer_id, coordinator=coordinator,
            run_context=final_context, bases=bases,
        )
        if not isinstance(broker, capabilities.CapabilityBroker) or (
            broker.ledger_snapshot.run_context_sha256 != final_context.sha256
        ) or any(getattr(broker.ledger_snapshot, name) != 0 for name in (
            "student_identity_count", "teacher_identity_count", "completed_identity_count",
        )) or broker.ledger_snapshot.outstanding_identity_sha256 is not None:
            raise RuntimeError("observational broker did not start empty and context-bound")
        self._adapter = adapter
        self._context = final_context
        self._rows = final_rows.detach().clone().contiguous()
        self._frequency = frequency_plan
        self._denominators = denominators
        self._sources = sources
        self._payload_sha256 = payload_sha256
        self._support_sha256 = common_support_sha256
        self._denominator_sha256 = denominator_sha256
        self._issuer_id = issuer_id
        self._coordinator = coordinator
        self._hook = hook
        self._broker = broker
        self._next_action = 0
        self._next_batch = 0
        self._materialized = None
        self._batch_sha256s: list[str] = []
        self._failed = False
        self._receipt: FinalObservationalExecutionReceipt | None = None
        self._native_executor_minted = False

    @property
    def issuer_id(self) -> str:
        return self._issuer_id

    @property
    def common_support_sha256(self) -> str:
        return self._support_sha256

    @property
    def receipt(self) -> FinalObservationalExecutionReceipt:
        if self._receipt is None:
            raise RuntimeError("observational execution receipt is unavailable")
        return self._receipt

    def make_native_denominator_executor(self) -> FinalNativeDenominatorBatchExecutor:
        """Mint one prepass owner before any canonical action has executed."""

        if self._failed or self._receipt is not None or self._native_executor_minted or (
            self._next_action != 0 or self._next_batch != 0
        ):
            self._poison()
            raise RuntimeError("native denominator prepass was minted late or twice")
        self._native_executor_minted = True
        return FinalNativeDenominatorBatchExecutor(
            adapter=self._adapter, final_context=self._context,
            final_rows=self._rows, frequency_plan=self._frequency,
            sources=self._sources, program_payload_sha256=self._payload_sha256,
            common_support_sha256=self._support_sha256,
        )

    def make_integrated_diagnostic_owner(
        self, response_run: Any,
    ) -> diagnostic_integration.IntegratedDiagnosticOwner:
        """Join physical callbacks without loading or authorizing the final role."""

        import early_mlp_suffix_transport_v1_response_execution as response_execution

        if type(response_run) is not response_execution.ObservedResponseRunResult or (
            response_run.receipt.final_context_sha256 != self._context.sha256
        ) or response_run.receipt.program_payload_sha256 != self._payload_sha256 or (
            response_run.receipt.common_support_sha256 != self._support_sha256
        ):
            self._poison()
            raise RuntimeError("integrated response run differs from observational authority")
        native = self.make_native_denominator_executor()

        def native_callback(action, batch_ordinal):
            try:
                return native(action, batch_ordinal)
            except BaseException:
                self._poison()
                raise

        return diagnostic_integration.IntegratedDiagnosticOwner(
            issuer_id=self._issuer_id,
            common_support_sha256=self._support_sha256,
            native_executor=native_callback, action_executor=self.run_captured,
            response_run=response_run,
        )

    def _poison(self) -> None:
        self._failed = True
        self._adapter = self._rows = self._frequency = self._denominators = None
        self._sources = self._hook = self._broker = self._materialized = None
        self._batch_sha256s.clear()

    def __call__(
        self, action: final_capability.FinalAction, batch_ordinal: int,
    ) -> observational_role.FinalObservationalBatch:
        return self._execute(action, batch_ordinal, captured=False)

    def run_captured(
        self, action: final_capability.FinalAction, batch_ordinal: int,
    ) -> diagnostic_integration.CapturedObservationalBatch:
        """Run the canonical action once with the live-consumer output hooks."""

        return self._execute(action, batch_ordinal, captured=True)

    def _execute(
        self, action: final_capability.FinalAction, batch_ordinal: int, *, captured: bool,
    ) -> Any:
        if self._failed or self._receipt is not None or self._next_action >= len(
            final_capability.CANONICAL_ACTIONS
        ) or action != final_capability.CANONICAL_ACTIONS[self._next_action] or (
            batch_ordinal != self._next_batch
        ):
            self._poison()
            raise RuntimeError("observational executor action/batch order changed")
        try:
            if batch_ordinal == 0:
                self._materialized = final_actions.materialize(
                    final_actions.plan_for(action.arm, action.background), self._sources,
                )
            start = batch_ordinal * runtime.BATCH_SIZE
            indices = tuple(range(start, start + runtime.BATCH_SIZE))
            rows = self._rows[start:start + runtime.BATCH_SIZE].contiguous()
            frequency = self._frequency.batch(batch_ordinal)
            identity = final_actions.FinalActionBatchIdentity.from_role_rows(
                materialized=self._materialized, role_rows=rows,
                ordered_batch_indices=indices, batch_ordinal=batch_ordinal,
                source_commit=self._context.source_commit,
                inherited_snapshot_sha256=self._context.inherited_snapshot_sha256,
                rows_receipt_sha256=self._context.rows_receipt_sha256,
                final_role_tensor_sha256=self._context.final_role_tensor_sha256,
                program_payload_sha256=self._payload_sha256,
                common_support_sha256=self._support_sha256,
            )
            before = self._broker.ledger_snapshot
            kind = self._materialized.plan.arm_plan.execution_kind
            if kind in {"projected_program", "mean_program"}:
                program = self._materialized.make_program()
                local_denominators = self._denominators if (
                    action.background == "N" and program.route == "L"
                ) else None
                backend = (
                    self._adapter.run_materialized_final_program_batch_captured
                    if captured else self._adapter.run_materialized_final_program_batch
                )
                backend_value = backend(
                    broker=self._broker, hook=self._hook,
                    materialized=self._materialized, identity=identity,
                    final_context=self._context, role_rows=rows,
                    ordered_row_indices=indices, denominators=local_denominators,
                    frequency_bins=frequency,
                )
                if captured:
                    reductions, backend_receipt, consumer_capture = backend_value
                else:
                    reductions, backend_receipt = backend_value
                    consumer_capture = None
                after = self._broker.ledger_snapshot
                if any(getattr(after, name) != getattr(before, name) + 1 for name in (
                    "student_identity_count", "teacher_identity_count",
                    "completed_identity_count",
                )) or after.outstanding_identity_sha256 is not None or (
                    after.rolling_ledger_sha256 == before.rolling_ledger_sha256
                ):
                    raise RuntimeError("observational program broker transaction did not close")
            else:
                backend = (
                    self._adapter.run_final_baseline_batch_captured
                    if captured else self._adapter.run_final_baseline_batch
                )
                backend_value = backend(
                    materialized=self._materialized, identity=identity,
                    role_rows=rows, ordered_row_indices=indices,
                    frequency_bins=frequency,
                )
                if captured:
                    reductions, backend_receipt, consumer_capture = backend_value
                else:
                    reductions, backend_receipt = backend_value
                    consumer_capture = None
                after = self._broker.ledger_snapshot
                if asdict(after) != asdict(before):
                    raise RuntimeError("observational baseline spent the program broker")
            result = observational_role.observational_batch_from_backend(
                action=action, common_support_sha256=self._support_sha256,
                reductions=reductions, receipt=backend_receipt,
            )
            self._batch_sha256s.append(result.sha256)
            captured_result = (
                diagnostic_integration.bind_completed_consumer_capture(
                    batch=result, capture=consumer_capture,
                    final_context_sha256=self._context.sha256,
                    program_payload_sha256=self._payload_sha256,
                ) if captured else None
            )
        except BaseException:
            self._poison()
            raise
        self._next_batch += 1
        if self._next_batch == FINAL_BATCH_COUNT:
            self._next_batch = 0
            self._next_action += 1
            self._materialized = None
        if self._next_action == len(final_capability.CANONICAL_ACTIONS):
            snapshot = self._broker.ledger_snapshot
            if snapshot.student_identity_count != PROGRAM_BATCH_COUNT or (
                snapshot.teacher_identity_count != PROGRAM_BATCH_COUNT
            ) or snapshot.completed_identity_count != PROGRAM_BATCH_COUNT or (
                snapshot.outstanding_identity_sha256 is not None
            ) or not self._coordinator.idle:
                self._poison()
                raise RuntimeError("observational role broker ledger did not close")
            self._receipt = FinalObservationalExecutionReceipt(
                final_context_sha256=self._context.sha256,
                source_bank_sha256=self._sources.sha256,
                program_payload_sha256=self._payload_sha256,
                common_support_sha256=self._support_sha256,
                denominator_pass_sha256=self._denominator_sha256,
                frequency_plan_sha256=self._frequency.sha256,
                batch_result_sha256s_sha256=runtime.logical_identity_sha256(
                    self._batch_sha256s
                ),
                broker_ledger_sha256=runtime.logical_identity_sha256(asdict(snapshot)),
                program_batch_count=PROGRAM_BATCH_COUNT,
                baseline_batch_count=BASELINE_BATCH_COUNT,
            )
            self._adapter = self._rows = self._frequency = self._denominators = None
            self._sources = self._hook = self._broker = self._materialized = None
            self._batch_sha256s.clear()
        return captured_result if captured else result
