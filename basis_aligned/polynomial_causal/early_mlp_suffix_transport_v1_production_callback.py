"""Fail-closed assembly of the suffix-transport observed final callback.

This module performs no I/O and owns no model, checkpoint, or role loader.  It joins
the already-initialized observational factory to the typed response and integrated
diagnostic owners.  Scientific gate semantics and the expensive scored-row
gauge/SVD/DiD replays remain behind one explicit capability: this assembler never
invents comparator choices or zero replay residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

import bilin18_observed_adapter as observed
import early_mlp_suffix_transport_v1_diagnostic_integration as integration
import early_mlp_suffix_transport_v1_final as final_owner
import early_mlp_suffix_transport_v1_final_actions as final_actions
import early_mlp_suffix_transport_v1_final_capability as final_capability
import early_mlp_suffix_transport_v1_final_execution as final_execution
import early_mlp_suffix_transport_v1_observational_authority as authority
import early_mlp_suffix_transport_v1_observational_execution as observational_execution
import early_mlp_suffix_transport_v1_response_execution as response_execution
import early_mlp_suffix_transport_v1_runtime as runtime
import early_mlp_suffix_transport_v1_statistics as statistics


_MINT_TOKEN = object()


def _sha256(name: str, value: Any) -> str:
    if not runtime._sha256_text(value):
        raise ValueError(f"{name} must be a SHA-256 identity")
    return value


def _replay_vector(name: str, value: Any) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 1 or value.numel() == 0 or (
        value.numel() > 4096
    ) or value.device.type != "cpu" or value.requires_grad or not bool(
        torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} replay evidence is malformed")
    return value.detach().double().contiguous().clone()


def _gate_map(value: Any, names: Sequence[str], label: str) -> dict[str, bool]:
    if not isinstance(value, Mapping) or set(value) != set(names) or any(
        type(value[name]) is not bool for name in names
    ):
        raise ValueError(f"{label} gate evidence is incomplete")
    return {name: value[name] for name in names}


@dataclass(frozen=True, slots=True)
class FinalDecisionReplayEvidence:
    """Only reviewed scientific decisions and replay residuals may fill this type."""

    final_context_sha256: str
    program_payload_sha256: str
    common_support_sha256: str
    observation_bundle_sha256: str
    response_run_receipt_sha256: str
    integrated_receipt_sha256: str
    objective_gates: Mapping[str, bool]
    transport_observational_gates: Mapping[str, bool]
    numerical_payload: Mapping[str, Any]
    outer_model_returned: bool
    hooks_restored: bool
    hooks_inert: bool
    component_tree_before_sha256: str
    component_tree_after_sha256: str
    student_poison_closed: bool
    gauge_replay_differences: Sequence[torch.Tensor]
    svd_replay_difference: torch.Tensor
    difference_in_differences_replay_difference: torch.Tensor

    def __post_init__(self) -> None:
        for name in (
            "final_context_sha256", "program_payload_sha256",
            "common_support_sha256", "observation_bundle_sha256",
            "response_run_receipt_sha256", "integrated_receipt_sha256",
            "component_tree_before_sha256", "component_tree_after_sha256",
        ):
            _sha256(name, getattr(self, name))
        for name in (
            "outer_model_returned", "hooks_restored", "hooks_inert",
            "student_poison_closed",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"{name} closure evidence is not literal true")
        if self.component_tree_after_sha256 != self.component_tree_before_sha256:
            raise ValueError("decision/replay producer observed component-tree drift")
        object.__setattr__(
            self, "objective_gates",
            _gate_map(self.objective_gates, final_owner.OBJECTIVE_GATES, "objective"),
        )
        object.__setattr__(
            self, "transport_observational_gates",
            _gate_map(
                self.transport_observational_gates,
                statistics.TRANSPORT_OBSERVATIONAL_GATES, "transport observational",
            ),
        )
        # Reuse the final boundary's tensor-free scalar-tree validator.  In
        # particular, no row, logit, code, state, or replay tensor can hide here.
        object.__setattr__(
            self, "numerical_payload",
            final_execution._scalar_tree(self.numerical_payload),
        )
        gauges = self.gauge_replay_differences
        if not isinstance(gauges, (tuple, list)) or len(gauges) != 8:
            raise ValueError("decision/replay evidence requires exactly eight gauges")
        object.__setattr__(
            self, "gauge_replay_differences",
            tuple(_replay_vector(f"gauge {index}", value)
                  for index, value in enumerate(gauges)),
        )
        object.__setattr__(
            self, "svd_replay_difference",
            _replay_vector("SVD", self.svd_replay_difference),
        )
        object.__setattr__(
            self, "difference_in_differences_replay_difference",
            _replay_vector(
                "difference-in-differences",
                self.difference_in_differences_replay_difference,
            ),
        )


@dataclass(frozen=True, slots=True)
class FinalDecisionReplayAuthority:
    """Prospective binding for one reviewed decision/replay producer."""

    final_context_sha256: str
    program_payload_sha256: str
    common_support_sha256: str
    producer_source_sha256: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _sha256(name, getattr(self, name))

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            name: getattr(self, name) for name in self.__dataclass_fields__
        })


class FinalDecisionReplayCapability:
    """Single-use reducer whose executor receives reductions, never rows/model.

    The executor is intentionally not implemented in this module.  A separately
    reviewed observed producer must close over any physical replay state, mint this
    capability before final execution, and return all gates and residuals explicitly.
    """

    def __init__(
        self, *, _token: object, binding: FinalDecisionReplayAuthority,
        executor: Callable[..., FinalDecisionReplayEvidence],
    ) -> None:
        if _token is not _MINT_TOKEN or type(binding) is not (
            FinalDecisionReplayAuthority
        ) or not callable(executor):
            raise TypeError("decision/replay capability requires its reviewed producer")
        self._binding = binding
        self._executor = executor
        self._spent = False
        self._failed = False

    @property
    def binding(self) -> FinalDecisionReplayAuthority:
        return self._binding

    def reduce(
        self, *, observations: final_capability.FinalObservationBundle,
        final_records: Sequence[Mapping[str, Any]],
        response_receipt: response_execution.ObservedResponseRunReceipt,
        integrated_receipt: integration.IntegratedObservationReceipt,
    ) -> FinalDecisionReplayEvidence:
        if self._spent or self._failed:
            raise RuntimeError("decision/replay capability is already closed")
        if type(observations) is not final_capability.FinalObservationBundle or type(
            response_receipt
        ) is not response_execution.ObservedResponseRunReceipt or type(
            integrated_receipt
        ) is not integration.IntegratedObservationReceipt or not isinstance(
            final_records, (tuple, list)
        ) or len(final_records) != final_execution.FINAL_ROW_COUNT or any(
            not isinstance(record, Mapping) for record in final_records
        ):
            self._failed = True
            raise TypeError("decision/replay capability requires complete typed evidence")
        expected = self._binding
        if observations.common_support_sha256 != expected.common_support_sha256 or (
            response_receipt.final_context_sha256 != expected.final_context_sha256
        ) or response_receipt.program_payload_sha256 != expected.program_payload_sha256 or (
            response_receipt.common_support_sha256 != expected.common_support_sha256
        ) or integrated_receipt.response_run_receipt_sha256 != response_receipt.sha256 or (
            integrated_receipt.observation_bundle_sha256 != observations.bundle_sha256
        ):
            self._failed = True
            raise RuntimeError("typed decision/replay inputs differ from bound authority")
        self._spent = True
        try:
            value = self._executor(
                observations=observations,
                final_records=tuple(dict(record) for record in final_records),
                response_receipt=response_receipt,
                integrated_receipt=integrated_receipt,
            )
            if type(value) is not FinalDecisionReplayEvidence:
                raise RuntimeError("decision/replay producer returned an untyped result")
            if value.final_context_sha256 != expected.final_context_sha256 or (
                value.program_payload_sha256 != expected.program_payload_sha256
            ) or value.common_support_sha256 != expected.common_support_sha256 or (
                value.observation_bundle_sha256 != observations.bundle_sha256
            ) or value.response_run_receipt_sha256 != response_receipt.sha256 or (
                value.integrated_receipt_sha256 != integrated_receipt.sha256
            ):
                raise RuntimeError("decision/replay result differs from its bound run")
        except BaseException:
            self._failed = True
            self._executor = None
            raise
        self._executor = None
        return value


def mint_final_decision_replay_capability(
    *, binding: FinalDecisionReplayAuthority,
    executor: Callable[..., FinalDecisionReplayEvidence],
) -> FinalDecisionReplayCapability:
    """Producer-side mint; inclusion here grants no scientific replay authority."""

    return FinalDecisionReplayCapability(
        _token=_MINT_TOKEN, binding=binding, executor=executor,
    )


class FinalProductionCallback:
    """One-shot callback passed directly to ``final_execution.execute_final``."""

    def __init__(
        self, *, executor_factory: authority.FinalObservationalExecutorFactory,
        decision_replay: FinalDecisionReplayCapability,
    ) -> None:
        if not isinstance(
            executor_factory, authority.FinalObservationalExecutorFactory
        ) or type(decision_replay) is not FinalDecisionReplayCapability:
            raise TypeError("production callback requires initialized typed owners")
        self._factory = executor_factory
        self._decision_replay = decision_replay
        self._spent = False
        self._failed = False

    def __call__(
        self, *, adapter: observed.ObservedBilin18Adapter,
        final_rows: torch.Tensor, final_records: Sequence[Mapping[str, Any]],
        program_bank: Mapping[str, Any],
    ) -> final_execution.FinalObservedReductions:
        if self._spent or self._failed:
            raise RuntimeError("production final callback is already closed")
        if not isinstance(adapter, observed.ObservedBilin18Adapter):
            self._failed = True
            raise TypeError("production callback requires the observed adapter")
        self._spent = True
        try:
            response_run, observational = self._factory.build_with_response(
                adapter=adapter, final_rows=final_rows,
                validated_program_bank=program_bank,
            )
            if type(response_run) is not response_execution.ObservedResponseRunResult or type(
                observational
            ) is not observational_execution.FinalObservationalBatchExecutor:
                raise RuntimeError("initialized factory returned an untyped execution owner")
            integrated = observational.make_integrated_diagnostic_owner(
                response_run
            ).execute_all()
            if type(integrated) is not integration.IntegratedObservationResult:
                raise RuntimeError("integrated diagnostic execution did not close")
            observational_receipt = observational.receipt
            evidence = self._decision_replay.reduce(
                observations=integrated.observations,
                final_records=final_records,
                response_receipt=response_run.receipt,
                integrated_receipt=integrated.receipt,
            )
            response = response_run.to_final_statistics_payload()
            if response["response_run_receipt_sha256"] != response_run.receipt.sha256:
                raise RuntimeError("response statistics escaped their run receipt")
            if "execution_receipts" in evidence.numerical_payload:
                raise RuntimeError("decision payload reserved execution receipt namespace")
            numerical_payload = {
                **evidence.numerical_payload,
                "execution_receipts": {
                    "decision_replay_authority_sha256": (
                        self._decision_replay.binding.sha256
                    ),
                    "observational_execution_receipt_sha256": (
                        observational_receipt.sha256
                    ),
                    "integrated_observation_receipt_sha256": integrated.receipt.sha256,
                    "observation_bundle_sha256": integrated.observations.bundle_sha256,
                },
            }
            closure = {
                "outer_model_returned": evidence.outer_model_returned,
                "hooks_restored": evidence.hooks_restored,
                "hooks_inert": evidence.hooks_inert,
                "component_tree_before_sha256": (
                    evidence.component_tree_before_sha256
                ),
                "component_tree_after_sha256": evidence.component_tree_after_sha256,
                "student_poison_closed": evidence.student_poison_closed,
                "program_payload_sha256": evidence.program_payload_sha256,
                "common_support_sha256": evidence.common_support_sha256,
                "arm_support_sha256s": {
                    action: evidence.common_support_sha256
                    for action in final_capability.CANONICAL_ACTION_KEYS
                },
                "observational_action_call_ledgers": (
                    final_actions.expected_observational_action_call_ledgers()
                ),
                "gauge_replay_differences": evidence.gauge_replay_differences,
                "svd_replay_difference": evidence.svd_replay_difference,
                "difference_in_differences_replay_difference": (
                    evidence.difference_in_differences_replay_difference
                ),
                "row_count": final_execution.FINAL_ROW_COUNT,
                "scored_tokens_per_row": final_execution.SCORED_TOKENS_PER_ROW,
                "scored_token_count": (
                    final_execution.FINAL_ROW_COUNT
                    * final_execution.SCORED_TOKENS_PER_ROW
                ),
            }
            result = final_execution.FinalObservedReductions(
                objective_gates=evidence.objective_gates,
                transport_observational_gates=(
                    evidence.transport_observational_gates
                ),
                code_baseline=response["code_baseline"],
                code_candidate=response["code_candidate"],
                logit_baseline=response["logit_baseline"],
                logit_candidate=response["logit_candidate"],
                logit_nulls=response["logit_nulls"],
                output_kl_baseline=response["output_kl_baseline"],
                output_kl_candidate=response["output_kl_candidate"],
                output_kl_nulls=response["output_kl_nulls"],
                numerical_payload=numerical_payload, closure_evidence=closure,
                response_run_receipt=response_run.receipt,
                evidence_join_receipt=integrated.evidence_join,
            )
        except BaseException:
            self._failed = True
            self._factory = self._decision_replay = None
            raise
        self._factory = self._decision_replay = None
        return result
