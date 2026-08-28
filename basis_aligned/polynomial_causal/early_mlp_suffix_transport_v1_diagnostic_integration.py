"""CPU-side owner joining observational, consumer, and response evidence.

This module has no row, model, checkpoint, or artifact loader.  It accepts only
already-authorized observed callbacks.  It runs a fixed 96-forward O/O consumer
denominator prepass, preserves the independent canonical 68-by-48 observational
schedule, completes each action with typed consumer/response reductions, and joins
the resulting bundle to one completed response-run receipt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

import early_mlp_suffix_transport_v1_consumer_norms as consumer
import early_mlp_suffix_transport_v1_final_capability as final_capability
import early_mlp_suffix_transport_v1_observational_role as observational_role
import early_mlp_suffix_transport_v1_response_execution as response_execution
import early_mlp_suffix_transport_v1_runtime as runtime


NATIVE_SCHEDULE = tuple(
    (final_capability.FinalAction(arm="o_o", background=background), batch)
    for background in final_capability.BACKGROUNDS
    for batch in range(observational_role.BATCH_COUNT)
)
ACTION_BATCH_COUNT = len(final_capability.CANONICAL_ACTIONS) * (
    observational_role.BATCH_COUNT
)
_MINT_TOKEN = object()


def _sha256(name: str, value: Any) -> str:
    if not runtime._sha256_text(value):
        raise ValueError(f"{name} must be a SHA-256 identity")
    return value


class CapturedObservationalBatch:
    """One authorized observational result bound to its in-forward capture."""

    __slots__ = (
        "__batch", "__capture", "__final_context_sha256",
        "__program_payload_sha256",
    )

    def __init__(
        self, *, _token: object,
        batch: observational_role.FinalObservationalBatch,
        capture: consumer._CapturedConsumerMagnitudes,
        final_context_sha256: str, program_payload_sha256: str,
    ) -> None:
        if _token is not _MINT_TOKEN or type(batch) is not (
            observational_role.FinalObservationalBatch
        ) or not isinstance(capture, consumer._CapturedConsumerMagnitudes):
            raise TypeError("captured observational batches require the binding owner")
        receipt = capture.receipt
        if capture.action != batch.action or receipt.action_identity_sha256 != (
            batch.action_identity_sha256
        ) or receipt.common_support_sha256 != batch.common_support_sha256 or (
            receipt.batch_ordinal != batch.batch_ordinal
        ):
            raise RuntimeError("consumer capture differs from observational action forward")
        self.__final_context_sha256 = _sha256(
            "captured final context", final_context_sha256,
        )
        self.__program_payload_sha256 = _sha256(
            "captured program payload", program_payload_sha256,
        )
        self.__batch = batch
        self.__capture = capture

    def __copy__(self):
        raise RuntimeError("captured observational batches cannot be copied")

    def __deepcopy__(self, memo):
        raise RuntimeError("captured observational batches cannot be copied")

    def __reduce__(self):
        raise RuntimeError("captured observational batches cannot be serialized")

    @property
    def batch(self) -> observational_role.FinalObservationalBatch:
        return self.__batch

    @property
    def final_context_sha256(self) -> str:
        return self.__final_context_sha256

    @property
    def program_payload_sha256(self) -> str:
        return self.__program_payload_sha256

    def _take_capture(self, token: object) -> consumer._CapturedConsumerMagnitudes:
        if token is not _MINT_TOKEN or self.__capture is None:
            raise RuntimeError("captured observational batch was already consumed")
        value = self.__capture
        self.__capture = None
        return value


def bind_consumer_capture_context(
    *, capture: consumer.AttentionConsumerOutputCapture,
    forward: Callable[[], observational_role.FinalObservationalBatch],
    final_context_sha256: str, program_payload_sha256: str,
) -> CapturedObservationalBatch:
    """Run exactly one authorized forward inside its output-capture context."""

    if not isinstance(capture, consumer.AttentionConsumerOutputCapture) or not callable(
        forward
    ):
        raise TypeError("consumer binding requires a typed capture and forward")
    calls = 0
    with capture:
        calls += 1
        batch = forward()
    captured = capture.finish()
    if calls != 1 or type(batch) is not observational_role.FinalObservationalBatch:
        raise RuntimeError("consumer binding did not execute one typed action forward")
    return CapturedObservationalBatch(
        _token=_MINT_TOKEN, batch=batch, capture=captured,
        final_context_sha256=final_context_sha256,
        program_payload_sha256=program_payload_sha256,
    )


@dataclass(frozen=True, slots=True)
class NativeDenominatorCacheReceipt:
    common_support_sha256: str
    final_context_sha256: str
    program_payload_sha256: str
    schedule: tuple[tuple[str, int], ...]
    observational_batch_sha256s: tuple[str, ...]
    capture_receipt_sha256s: tuple[str, ...]
    backend_receipt_sha256s: tuple[str, ...]
    forward_count: int
    metric_role: str = "integrity_denominator_only"
    authorized_for_selection: bool = False

    def __post_init__(self) -> None:
        _sha256("native denominator support", self.common_support_sha256)
        _sha256("native denominator final context", self.final_context_sha256)
        _sha256("native denominator program payload", self.program_payload_sha256)
        expected = tuple((action.key, batch) for action, batch in NATIVE_SCHEDULE)
        if self.schedule != expected or self.forward_count != len(NATIVE_SCHEDULE):
            raise ValueError("native consumer denominator schedule changed")
        for name in (
            "observational_batch_sha256s", "capture_receipt_sha256s",
            "backend_receipt_sha256s",
        ):
            values = getattr(self, name)
            if len(values) != len(NATIVE_SCHEDULE) or len(set(values)) != len(values) or any(
                not runtime._sha256_text(value) for value in values
            ):
                raise ValueError(f"native denominator {name} is incomplete or replayed")
        if self.metric_role != "integrity_denominator_only" or (
            self.authorized_for_selection is not False
        ):
            raise ValueError("native denominator cache gained selection authority")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class _NativeEntry:
    action: final_capability.FinalAction
    batch_ordinal: int
    frequency_assignment_sha256: str
    observational_batch_sha256: str
    backend_receipt_sha256: str
    capture: consumer._CapturedConsumerMagnitudes


class NativeDenominatorCache:
    """Immutable scalar cache built before canonical action execution begins."""

    def __init__(
        self, *, _token: object, entries: tuple[_NativeEntry, ...],
        receipt: NativeDenominatorCacheReceipt,
    ) -> None:
        if _token is not _MINT_TOKEN or type(receipt) is not (
            NativeDenominatorCacheReceipt
        ) or len(entries) != len(NATIVE_SCHEDULE):
            raise TypeError("native denominator cache was not validly minted")
        expected = tuple((action.key, batch) for action, batch in NATIVE_SCHEDULE)
        if tuple((entry.action.key, entry.batch_ordinal) for entry in entries) != expected:
            raise RuntimeError("native denominator entries changed schedule")
        self._entries = {
            (entry.action.background, entry.batch_ordinal): entry for entry in entries
        }
        self._receipt = receipt

    @property
    def receipt(self) -> NativeDenominatorCacheReceipt:
        return self._receipt

    def _entry(
        self, action: final_capability.FinalAction, batch_ordinal: int,
    ) -> _NativeEntry:
        if type(action) is not final_capability.FinalAction:
            raise TypeError("native denominator lookup requires a typed action")
        try:
            return self._entries[(action.background, batch_ordinal)]
        except KeyError as error:
            raise RuntimeError("native denominator lookup changed support") from error


def build_native_denominator_cache(
    *, common_support_sha256: str,
    final_context_sha256: str, program_payload_sha256: str,
    executor: Callable[
        [final_capability.FinalAction, int], CapturedObservationalBatch
    ],
) -> NativeDenominatorCache:
    """Execute exactly O/O/N[0:48], then O/O/E[0:48], before action order."""

    _sha256("native denominator support", common_support_sha256)
    _sha256("native denominator final context", final_context_sha256)
    _sha256("native denominator program payload", program_payload_sha256)
    if not callable(executor):
        raise TypeError("native denominator executor must be callable")
    entries: list[_NativeEntry] = []
    try:
        for action, batch_ordinal in NATIVE_SCHEDULE:
            value = executor(action, batch_ordinal)
            if not isinstance(value, CapturedObservationalBatch):
                raise RuntimeError("native denominator executor returned an unbound batch")
            batch = value.batch
            capture = value._take_capture(_MINT_TOKEN)
            if batch.action != action or batch.batch_ordinal != batch_ordinal or (
                batch.common_support_sha256 != common_support_sha256
            ) or value.final_context_sha256 != final_context_sha256 or (
                value.program_payload_sha256 != program_payload_sha256
            ):
                raise RuntimeError(
                    "native denominator executor changed schedule, support, or authority"
                )
            entries.append(_NativeEntry(
                action=action, batch_ordinal=batch_ordinal,
                frequency_assignment_sha256=batch.frequency_assignment_sha256,
                observational_batch_sha256=batch.sha256,
                backend_receipt_sha256=batch.backend_receipt_sha256,
                capture=capture,
            ))
    except BaseException:
        entries.clear()
        raise
    receipt = NativeDenominatorCacheReceipt(
        common_support_sha256=common_support_sha256,
        final_context_sha256=final_context_sha256,
        program_payload_sha256=program_payload_sha256,
        schedule=tuple((action.key, batch) for action, batch in NATIVE_SCHEDULE),
        observational_batch_sha256s=tuple(
            entry.observational_batch_sha256 for entry in entries
        ),
        capture_receipt_sha256s=tuple(
            entry.capture.receipt.sha256 for entry in entries
        ),
        backend_receipt_sha256s=tuple(
            entry.backend_receipt_sha256 for entry in entries
        ),
        forward_count=len(entries),
    )
    return NativeDenominatorCache(
        _token=_MINT_TOKEN, entries=tuple(entries), receipt=receipt,
    )


class ConsumerIntegratedBatchExecutor:
    """Canonical batch wrapper that pairs each action with the frozen O/O cache."""

    def __init__(
        self, *, common_support_sha256: str, native_cache: NativeDenominatorCache,
        final_context_sha256: str, program_payload_sha256: str,
        executor: Callable[
            [final_capability.FinalAction, int], CapturedObservationalBatch
        ],
    ) -> None:
        _sha256("consumer integrated support", common_support_sha256)
        if not isinstance(native_cache, NativeDenominatorCache) or (
            native_cache.receipt.common_support_sha256 != common_support_sha256
        ) or native_cache.receipt.final_context_sha256 != final_context_sha256 or (
            native_cache.receipt.program_payload_sha256 != program_payload_sha256
        ) or not callable(executor):
            raise TypeError("integrated executor requires typed cache and callback")
        self._support = common_support_sha256
        self._final_context = _sha256(
            "consumer integrated final context", final_context_sha256,
        )
        self._program_payload = _sha256(
            "consumer integrated program payload", program_payload_sha256,
        )
        self._cache = native_cache
        self._executor = executor
        self._next_action = 0
        self._next_batch = 0
        self._batches: list[consumer.ConsumerNormBatchResult] = []
        self._completed: consumer.ConsumerNormActionResult | None = None
        self._action_result_sha256s: list[tuple[str, str]] = []
        self._paired_batch_sha256s: list[str] = []
        self._failed = False
        self._closed = False

    def _poison(self) -> None:
        self._failed = True
        self._executor = self._cache = None
        self._batches.clear()
        self._completed = None

    def __call__(
        self, action: final_capability.FinalAction, batch_ordinal: int,
    ) -> observational_role.FinalObservationalBatch:
        if self._failed or self._closed or self._completed is not None or (
            self._next_action >= len(final_capability.CANONICAL_ACTIONS)
        ) or action != final_capability.CANONICAL_ACTIONS[self._next_action] or (
            batch_ordinal != self._next_batch
        ):
            self._poison()
            raise RuntimeError("consumer-integrated action/batch order changed")
        try:
            value = self._executor(action, batch_ordinal)
            if not isinstance(value, CapturedObservationalBatch):
                raise RuntimeError("action executor returned an unbound consumer batch")
            batch = value.batch
            action_capture = value._take_capture(_MINT_TOKEN)
            native = self._cache._entry(action, batch_ordinal)
            if batch.action != action or batch.batch_ordinal != batch_ordinal or (
                batch.common_support_sha256 != self._support
            ) or value.final_context_sha256 != self._final_context or (
                value.program_payload_sha256 != self._program_payload
            ) or batch.frequency_assignment_sha256 != native.frequency_assignment_sha256:
                raise RuntimeError("action/native consumer support changed")
            reduced = consumer.reduce_consumer_norm_batch(
                action_capture=action_capture, native_capture=native.capture,
            )
            self._batches.append(reduced)
            self._paired_batch_sha256s.append(runtime.logical_identity_sha256({
                "observational_batch_sha256": batch.sha256,
                "consumer_batch_sha256": reduced.sha256,
                "native_observational_batch_sha256": native.observational_batch_sha256,
                "final_context_sha256": self._final_context,
                "program_payload_sha256": self._program_payload,
            }))
        except BaseException:
            self._poison()
            raise
        self._next_batch += 1
        if self._next_batch == observational_role.BATCH_COUNT:
            self._completed = consumer.aggregate_consumer_norm_action(
                action, tuple(self._batches),
            )
            self._batches.clear()
            self._next_batch = 0
            self._next_action += 1
        return batch

    def take_completed_for_core(
        self, core: observational_role.FinalObservationalActionCore,
    ) -> consumer.ConsumerNormActionResult:
        if self._failed or type(core) is not observational_role.FinalObservationalActionCore or (
            self._completed is None
        ) or self._completed.action != core.action:
            self._poison()
            raise RuntimeError("consumer result does not match completed observational core")
        value = self._completed
        self._completed = None
        self._action_result_sha256s.append((core.action.key, value.result_sha256))
        if self._next_action == len(final_capability.CANONICAL_ACTIONS):
            self._closed = True
            self._executor = None
        return value

    @property
    def action_result_sha256s(self) -> tuple[tuple[str, str], ...]:
        expected = final_capability.CANONICAL_ACTION_KEYS
        if not self._closed or self._failed or tuple(
            key for key, _value in self._action_result_sha256s
        ) != expected or len(self._paired_batch_sha256s) != ACTION_BATCH_COUNT:
            raise RuntimeError("consumer-integrated execution is incomplete")
        return tuple(self._action_result_sha256s)

    @property
    def paired_batch_ledger_sha256(self) -> str:
        self.action_result_sha256s
        return runtime.logical_identity_sha256(self._paired_batch_sha256s)


class FinalDiagnosticCompleter:
    """Pure typed join from one observational core to its complete observation."""

    def __init__(
        self, *, consumer_executor: ConsumerIntegratedBatchExecutor,
        response_run: response_execution.ObservedResponseRunResult,
        common_support_sha256: str,
    ) -> None:
        if not isinstance(consumer_executor, ConsumerIntegratedBatchExecutor) or type(
            response_run
        ) is not response_execution.ObservedResponseRunResult or (
            response_run.receipt.common_support_sha256 != common_support_sha256
        ):
            raise TypeError("diagnostic completer requires complete typed owners")
        self._consumer = consumer_executor
        self._response_run = response_run
        self._support = _sha256("diagnostic completer support", common_support_sha256)
        unit = response_run.receipt.ordered_unit_identity_sha256
        self._responses = {
            value.action_key: (
                None if value.code_response is None else final_capability.ResponseReduction(
                    **value.code_response.as_statistics(unit)
                ),
                final_capability.ResponseReduction(
                    **value.logit_response.as_statistics(unit)
                ),
                final_capability.OutputKLReduction(
                    **value.output_kl_response.as_statistics(unit)
                ),
                value.sha256,
            ) for value in response_run.arm_reductions
        }
        self._closure_sha256s: list[tuple[str, str]] = []
        self._failed = False

    def __call__(
        self, core: observational_role.FinalObservationalActionCore,
    ) -> final_capability.FinalArmObservation:
        if self._failed or type(core) is not observational_role.FinalObservationalActionCore or (
            core.common_support_sha256 != self._support
        ):
            self._failed = True
            raise RuntimeError("diagnostic completer core changed support")
        try:
            consumer_result = self._consumer.take_completed_for_core(core)
            response = self._responses.get(core.action.key)
            closure = runtime.logical_identity_sha256({
                "observational_core_sha256": core.core_sha256,
                "consumer_action_result_sha256": consumer_result.result_sha256,
                "native_denominator_cache_sha256": self._consumer._cache.receipt.sha256,
                "response_run_receipt_sha256": self._response_run.receipt.sha256,
                "response_arm_reduction_sha256": (
                    None if response is None else response[3]
                ),
            })
            observation = final_capability.FinalArmObservation(
                action=core.action, common_support_sha256=self._support,
                ce=core.ce, teacher_kl=core.primary, copy_ce=core.copy_ce,
                frequency_ce=core.frequency_ce,
                code_response=None if response is None else response[0],
                logit_response=None if response is None else response[1],
                output_kl_response=None if response is None else response[2],
                consumer_norm_ratio=consumer_result.reductions,
                execution_closure_sha256=closure,
            )
            self._closure_sha256s.append((core.action.key, closure))
            return observation
        except BaseException:
            self._failed = True
            self._responses.clear()
            raise

    @property
    def closure_sha256s(self) -> tuple[tuple[str, str], ...]:
        if self._failed or tuple(key for key, _value in self._closure_sha256s) != (
            final_capability.CANONICAL_ACTION_KEYS
        ):
            raise RuntimeError("diagnostic completer is incomplete")
        return tuple(self._closure_sha256s)


@dataclass(frozen=True, slots=True)
class IntegratedObservationReceipt:
    native_cache_sha256: str
    observational_role_receipt_sha256: str
    observation_bundle_sha256: str
    response_run_receipt_sha256: str
    evidence_join_receipt_sha256: str
    consumer_action_result_sha256s: tuple[tuple[str, str], ...]
    action_closure_sha256s: tuple[tuple[str, str], ...]
    paired_batch_ledger_sha256: str
    native_forward_count: int
    action_forward_count: int
    final_role_load_authorized: bool = False

    def __post_init__(self) -> None:
        for name in (
            "native_cache_sha256", "observational_role_receipt_sha256",
            "observation_bundle_sha256", "response_run_receipt_sha256",
            "evidence_join_receipt_sha256", "paired_batch_ledger_sha256",
        ):
            _sha256(name, getattr(self, name))
        expected = final_capability.CANONICAL_ACTION_KEYS
        if tuple(key for key, _value in self.consumer_action_result_sha256s) != expected or tuple(
            key for key, _value in self.action_closure_sha256s
        ) != expected or any(
            not runtime._sha256_text(value)
            for collection in (
                self.consumer_action_result_sha256s, self.action_closure_sha256s,
            ) for _key, value in collection
        ) or self.native_forward_count != len(NATIVE_SCHEDULE) or (
            self.action_forward_count != ACTION_BATCH_COUNT
        ) or self.final_role_load_authorized is not False:
            raise ValueError("integrated observation receipt changed coverage or authority")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class IntegratedObservationResult:
    observations: final_capability.FinalObservationBundle
    evidence_join: final_capability.FinalEvidenceJoinReceipt
    receipt: IntegratedObservationReceipt

    def __post_init__(self) -> None:
        if type(self.observations) is not final_capability.FinalObservationBundle or type(
            self.evidence_join
        ) is not final_capability.FinalEvidenceJoinReceipt or type(
            self.receipt
        ) is not IntegratedObservationReceipt or self.receipt.observation_bundle_sha256 != (
            self.observations.bundle_sha256
        ) or self.receipt.evidence_join_receipt_sha256 != self.evidence_join.sha256:
            raise ValueError("integrated observation result differs from its receipt")


class IntegratedDiagnosticOwner:
    """One-shot callback owner; it cannot load or authorize the final role."""

    def __init__(
        self, *, issuer_id: str, common_support_sha256: str,
        native_executor: Callable[
            [final_capability.FinalAction, int], CapturedObservationalBatch
        ],
        action_executor: Callable[
            [final_capability.FinalAction, int], CapturedObservationalBatch
        ],
        response_run: response_execution.ObservedResponseRunResult,
    ) -> None:
        self._issuer = _sha256("integrated diagnostic issuer", issuer_id)
        self._support = _sha256("integrated diagnostic support", common_support_sha256)
        if not callable(native_executor) or not callable(action_executor) or type(
            response_run
        ) is not response_execution.ObservedResponseRunResult or (
            response_run.receipt.common_support_sha256 != self._support
        ):
            raise TypeError("integrated diagnostic owner requires typed callbacks and response")
        self._native_executor = native_executor
        self._action_executor = action_executor
        self._response_run = response_run
        self._spent = False
        self._failed = False

    def execute_all(self) -> IntegratedObservationResult:
        if self._spent or self._failed:
            raise RuntimeError("integrated diagnostic owner is already closed")
        self._spent = True
        try:
            native_cache = build_native_denominator_cache(
                common_support_sha256=self._support,
                final_context_sha256=self._response_run.receipt.final_context_sha256,
                program_payload_sha256=self._response_run.receipt.program_payload_sha256,
                executor=self._native_executor,
            )
            integrated = ConsumerIntegratedBatchExecutor(
                common_support_sha256=self._support, native_cache=native_cache,
                final_context_sha256=self._response_run.receipt.final_context_sha256,
                program_payload_sha256=self._response_run.receipt.program_payload_sha256,
                executor=self._action_executor,
            )
            completer = FinalDiagnosticCompleter(
                consumer_executor=integrated, response_run=self._response_run,
                common_support_sha256=self._support,
            )
            role = observational_role.FinalObservationalRoleOwner(
                issuer_id=self._issuer, common_support_sha256=self._support,
                batch_executor=integrated,
            )
            capability = role.mint_action_capability(completer=completer)
            observations = capability.execute_all()
            evidence_join = final_capability.join_observations_with_response_run(
                observations, self._response_run,
            )
            receipt = IntegratedObservationReceipt(
                native_cache_sha256=native_cache.receipt.sha256,
                observational_role_receipt_sha256=role.receipt.sha256,
                observation_bundle_sha256=observations.bundle_sha256,
                response_run_receipt_sha256=self._response_run.receipt.sha256,
                evidence_join_receipt_sha256=evidence_join.sha256,
                consumer_action_result_sha256s=integrated.action_result_sha256s,
                action_closure_sha256s=completer.closure_sha256s,
                paired_batch_ledger_sha256=integrated.paired_batch_ledger_sha256,
                native_forward_count=len(NATIVE_SCHEDULE),
                action_forward_count=ACTION_BATCH_COUNT,
            )
        except BaseException:
            self._failed = True
            self._native_executor = self._action_executor = self._response_run = None
            raise
        self._native_executor = self._action_executor = self._response_run = None
        return IntegratedObservationResult(
            observations=observations, evidence_join=evidence_join, receipt=receipt,
        )
