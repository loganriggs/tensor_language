"""Canonical 68-action observational role owner for suffix final execution.

This module performs no model, checkpoint, row, or artifact I/O.  A source-closed
adapter supplies one typed four-row batch result at a time.  The owner fixes action
and batch order, aggregates CE/copy/frequency/primary row reductions, closes exact
receipt and call ledgers, and can mint the existing one-shot action capability only
through an explicit completer for still-separate response and consumer diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch

import early_mlp_suffix_transport_v1_final_actions as final_actions
import early_mlp_suffix_transport_v1_final_capability as final_capability
import early_mlp_suffix_transport_v1_runtime as runtime


BATCH_COUNT = final_actions.OBSERVATIONAL_BATCH_COUNT
ROW_COUNT = BATCH_COUNT * runtime.BATCH_SIZE
FREQUENCY_BIN_COUNT = final_capability.FREQUENCY_BIN_COUNT


def _sha256(name: str, value: Any) -> str:
    if not runtime._sha256_text(value):
        raise ValueError(f"{name} must be a SHA-256 identity")
    return value


def _batch_vector(name: str, value: Any, *, count: bool) -> torch.Tensor:
    dtype = torch.long if count else torch.float64
    if not torch.is_tensor(value) or tuple(value.shape) != (
        runtime.BATCH_SIZE,
    ) or value.dtype != dtype or value.device.type != "cpu" or value.requires_grad or (
        not bool(torch.isfinite(value).all())
    ) or bool((value < 0).any()):
        raise ValueError(f"{name} is not a permitted observational batch reduction")
    return value.detach().clone().contiguous()


def _frequency_matrix(name: str, value: Any, *, count: bool) -> torch.Tensor:
    dtype = torch.long if count else torch.float64
    if not torch.is_tensor(value) or tuple(value.shape) != (
        runtime.BATCH_SIZE, FREQUENCY_BIN_COUNT
    ) or value.dtype != dtype or value.device.type != "cpu" or value.requires_grad or (
        not bool(torch.isfinite(value).all())
    ) or bool((value < 0).any()):
        raise ValueError(f"{name} is not a permitted frequency reduction")
    return value.detach().clone().contiguous()


@dataclass(frozen=True, slots=True)
class FinalObservationalBatch:
    action: final_capability.FinalAction
    batch_ordinal: int
    common_support_sha256: str
    action_identity_sha256: str
    backend_receipt_sha256: str
    frequency_assignment_sha256: str
    row_primary_sum: torch.Tensor | None
    row_primary_count: torch.Tensor | None
    row_ce_sum: torch.Tensor
    row_ce_count: torch.Tensor
    row_copy_ce_sum: torch.Tensor
    row_copy_count: torch.Tensor
    row_frequency_ce_sum: torch.Tensor
    row_frequency_count: torch.Tensor

    def __post_init__(self) -> None:
        if type(self.action) is not final_capability.FinalAction or type(
            self.batch_ordinal
        ) is not int or not 0 <= self.batch_ordinal < BATCH_COUNT:
            raise ValueError("observational batch action or ordinal changed")
        for name in (
            "common_support_sha256", "action_identity_sha256",
            "backend_receipt_sha256", "frequency_assignment_sha256",
        ):
            _sha256(name, getattr(self, name))
        primary = self.action.background == "N"
        if primary != (self.row_primary_sum is not None) or primary != (
            self.row_primary_count is not None
        ):
            raise ValueError("observational primary reduction changed N/E semantics")
        if primary:
            object.__setattr__(
                self, "row_primary_sum",
                _batch_vector("row_primary_sum", self.row_primary_sum, count=False),
            )
            object.__setattr__(
                self, "row_primary_count",
                _batch_vector("row_primary_count", self.row_primary_count, count=True),
            )
        for name in ("row_ce_sum", "row_ce_count", "row_copy_ce_sum", "row_copy_count"):
            object.__setattr__(
                self, name, _batch_vector(name, getattr(self, name), count=name.endswith("count")),
            )
        object.__setattr__(
            self, "row_frequency_ce_sum",
            _frequency_matrix("row_frequency_ce_sum", self.row_frequency_ce_sum, count=False),
        )
        object.__setattr__(
            self, "row_frequency_count",
            _frequency_matrix("row_frequency_count", self.row_frequency_count, count=True),
        )
        if not torch.equal(self.row_frequency_count.sum(dim=1), self.row_ce_count) or not (
            torch.allclose(self.row_frequency_ce_sum.sum(dim=1), self.row_ce_sum, atol=1e-10)
        ):
            raise ValueError("observational frequency bins do not partition CE")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "action_sha256": self.action.sha256,
            "batch_ordinal": self.batch_ordinal,
            "common_support_sha256": self.common_support_sha256,
            "action_identity_sha256": self.action_identity_sha256,
            "backend_receipt_sha256": self.backend_receipt_sha256,
            "frequency_assignment_sha256": self.frequency_assignment_sha256,
            **{
                name: (
                    None if getattr(self, name) is None
                    else runtime.tensor_identity_sha256(getattr(self, name))
                ) for name in (
                    "row_primary_sum", "row_primary_count", "row_ce_sum",
                    "row_ce_count", "row_copy_ce_sum", "row_copy_count",
                    "row_frequency_ce_sum", "row_frequency_count",
                )
            },
        })


def observational_batch_from_backend(
    *, action: final_capability.FinalAction, common_support_sha256: str,
    reductions: Any, receipt: Any,
) -> FinalObservationalBatch:
    """Bind one existing observed program/baseline backend result to the owner."""

    import bilin18_observed_adapter as observed
    import early_mlp_suffix_transport_v1_capabilities as capabilities

    if type(action) is not final_capability.FinalAction:
        raise TypeError("observational backend join requires a typed action")
    program = action.arm not in {"n_n", "o_o"}
    expected_reduction = (
        capabilities.FinalBatchReductions if action.background == "N"
        else capabilities.FinalCEBatchReductions
    ) if program else observed.ObservedFinalBaselineBatchReductions
    expected_receipt = (
        observed.ObservedMaterializedFinalProgramBatchReceipt
        if program else observed.ObservedFinalBaselineBatchReceipt
    )
    if type(reductions) is not expected_reduction or type(receipt) is not expected_receipt:
        raise TypeError("observational backend returned the wrong typed result")
    if receipt.action_key != action.key or reductions.identity_sha256 != (
        receipt.final_action_identity_sha256 if program else receipt.identity_sha256
    ) or (not program and reductions.action_key != action.key):
        raise RuntimeError("observational backend action identity changed")
    scalar_fields = (
        "row_ce_sum", "row_ce_count", "row_copy_ce_sum", "row_copy_count",
        "row_frequency_ce_sum", "row_frequency_count",
    )
    reduction_identity = {
        name: runtime.tensor_identity_sha256(getattr(reductions, name))
        for name in scalar_fields
    }
    if action.background == "N":
        reduction_identity = {
            "row_primary_sum": runtime.tensor_identity_sha256(
                reductions.row_primary_sum
            ),
            "row_primary_count": runtime.tensor_identity_sha256(
                reductions.row_primary_count
            ),
            **reduction_identity,
        }
    if not program:
        reduction_identity = {"action_key": action.key, **reduction_identity}
    if runtime.logical_identity_sha256(reduction_identity) != receipt.reduction_sha256:
        raise RuntimeError("observational backend reduction differs from its receipt")
    return FinalObservationalBatch(
        action=action, batch_ordinal=receipt.batch_ordinal,
        common_support_sha256=common_support_sha256,
        action_identity_sha256=(
            receipt.final_action_identity_sha256 if program else receipt.identity_sha256
        ),
        backend_receipt_sha256=runtime.logical_identity_sha256(asdict(receipt)),
        frequency_assignment_sha256=receipt.frequency_assignment_sha256,
        row_primary_sum=(
            reductions.row_primary_sum if action.background == "N" else None
        ),
        row_primary_count=(
            reductions.row_primary_count if action.background == "N" else None
        ),
        **{name: getattr(reductions, name) for name in scalar_fields},
    )


@dataclass(frozen=True, slots=True)
class FinalObservationalActionCore:
    action: final_capability.FinalAction
    common_support_sha256: str
    primary: final_capability.RowReduction | None
    ce: final_capability.RowReduction
    copy_ce: final_capability.RowReduction
    frequency_ce: tuple[final_capability.FrequencyRowReduction, ...]
    batch_sha256s: tuple[str, ...]
    backend_receipt_sha256s: tuple[str, ...]
    action_identity_sha256s: tuple[str, ...]
    frequency_assignment_sha256s: tuple[str, ...]
    call_ledger: Mapping[str, Any]
    core_sha256: str

    def __post_init__(self) -> None:
        if type(self.action) is not final_capability.FinalAction or not _sha256(
            "action support", self.common_support_sha256
        ) or (self.primary is not None) != (self.action.background == "N") or type(
            self.ce
        ) is not final_capability.RowReduction or type(
            self.copy_ce
        ) is not final_capability.RowReduction or not isinstance(
            self.frequency_ce, tuple
        ) or len(self.frequency_ce) != FREQUENCY_BIN_COUNT or any(
            type(value) is not final_capability.FrequencyRowReduction
            for value in self.frequency_ce
        ):
            raise ValueError("observational action core schema changed")
        for name in (
            "batch_sha256s", "backend_receipt_sha256s", "action_identity_sha256s",
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != BATCH_COUNT or len(
                set(values)
            ) != BATCH_COUNT or any(not runtime._sha256_text(value) for value in values):
                raise ValueError(f"observational action {name} is incomplete or duplicated")
        if len(self.frequency_assignment_sha256s) != BATCH_COUNT or any(
            not runtime._sha256_text(value) for value in self.frequency_assignment_sha256s
        ):
            raise ValueError("observational frequency assignment plan is incomplete")
        expected = final_actions.expected_early_call_pattern(
            final_actions.plan_for(self.action.arm, self.action.background)
        ).totals(BATCH_COUNT)
        if dict(self.call_ledger) != expected:
            raise ValueError("observational action call ledger changed")
        body = self._identity_body()
        if not _sha256("observational action core", self.core_sha256) or (
            runtime.logical_identity_sha256(body) != self.core_sha256
        ):
            raise ValueError("observational action core identity changed")

    def _identity_body(self) -> dict[str, Any]:
        return {
            "action_sha256": self.action.sha256,
            "common_support_sha256": self.common_support_sha256,
            "primary_sha256": None if self.primary is None else self.primary.sha256,
            "ce_sha256": self.ce.sha256,
            "copy_ce_sha256": self.copy_ce.sha256,
            "frequency_ce_sha256s": [value.sha256 for value in self.frequency_ce],
            "batch_sha256s": list(self.batch_sha256s),
            "backend_receipt_sha256s": list(self.backend_receipt_sha256s),
            "action_identity_sha256s": list(self.action_identity_sha256s),
            "frequency_assignment_sha256s": list(self.frequency_assignment_sha256s),
            "call_ledger": dict(self.call_ledger),
        }


class FinalObservationalActionAccumulator:
    def __init__(self, action: final_capability.FinalAction, support: str) -> None:
        if type(action) is not final_capability.FinalAction:
            raise TypeError("observational accumulator requires a typed action")
        self._action = action
        self._support = _sha256("observational support", support)
        self._batches: list[FinalObservationalBatch] = []
        self._closed = False

    def add(self, batch: FinalObservationalBatch) -> None:
        if self._closed:
            raise RuntimeError("observational action accumulator is closed")
        if type(batch) is not FinalObservationalBatch or batch.action != self._action or (
            batch.common_support_sha256 != self._support
        ) or batch.batch_ordinal != len(self._batches):
            self._closed = True
            self._batches.clear()
            raise RuntimeError("observational batch is skipped, reordered, or support-mixed")
        if batch.backend_receipt_sha256 in {
            value.backend_receipt_sha256 for value in self._batches
        } or batch.action_identity_sha256 in {
            value.action_identity_sha256 for value in self._batches
        }:
            self._closed = True
            self._batches.clear()
            raise RuntimeError("observational batch or action identity was duplicated")
        self._batches.append(batch)

    def finish(self) -> FinalObservationalActionCore:
        if self._closed or len(self._batches) != BATCH_COUNT:
            self._closed = True
            self._batches.clear()
            raise RuntimeError("observational action is incomplete")
        self._closed = True

        def row(sum_name: str, count_name: str) -> final_capability.RowReduction:
            return final_capability.RowReduction(
                row_sum=torch.cat([getattr(value, sum_name) for value in self._batches]),
                row_count=torch.cat([getattr(value, count_name) for value in self._batches]),
            )

        primary = None
        if self._action.background == "N":
            primary = row("row_primary_sum", "row_primary_count")
        ce = row("row_ce_sum", "row_ce_count")
        copy_ce = row("row_copy_ce_sum", "row_copy_count")
        frequency = tuple(
            final_capability.FrequencyRowReduction(
                row_sum=torch.cat([
                    value.row_frequency_ce_sum[:, index] for value in self._batches
                ]),
                row_count=torch.cat([
                    value.row_frequency_count[:, index] for value in self._batches
                ]),
            ) for index in range(FREQUENCY_BIN_COUNT)
        )
        body = {
            "action_sha256": self._action.sha256,
            "common_support_sha256": self._support,
            "primary_sha256": None if primary is None else primary.sha256,
            "ce_sha256": ce.sha256, "copy_ce_sha256": copy_ce.sha256,
            "frequency_ce_sha256s": [value.sha256 for value in frequency],
            "batch_sha256s": [value.sha256 for value in self._batches],
            "backend_receipt_sha256s": [
                value.backend_receipt_sha256 for value in self._batches
            ],
            "action_identity_sha256s": [
                value.action_identity_sha256 for value in self._batches
            ],
            "frequency_assignment_sha256s": [
                value.frequency_assignment_sha256 for value in self._batches
            ],
            "call_ledger": final_actions.expected_early_call_pattern(
                final_actions.plan_for(self._action.arm, self._action.background)
            ).totals(BATCH_COUNT),
        }
        result = FinalObservationalActionCore(
            action=self._action, common_support_sha256=self._support,
            primary=primary, ce=ce, copy_ce=copy_ce, frequency_ce=frequency,
            batch_sha256s=tuple(body["batch_sha256s"]),
            backend_receipt_sha256s=tuple(body["backend_receipt_sha256s"]),
            action_identity_sha256s=tuple(body["action_identity_sha256s"]),
            frequency_assignment_sha256s=tuple(
                body["frequency_assignment_sha256s"]
            ),
            call_ledger=body["call_ledger"],
            core_sha256=runtime.logical_identity_sha256(body),
        )
        self._batches.clear()
        return result


@dataclass(frozen=True, slots=True)
class FinalObservationalRoleReceipt:
    common_support_sha256: str
    action_core_sha256s: tuple[tuple[str, str], ...]
    backend_receipt_sha256s: tuple[str, ...]
    frequency_plan_sha256: str
    call_ledgers_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not _sha256("role support", self.common_support_sha256) or tuple(
            key for key, _value in self.action_core_sha256s
        ) != final_capability.CANONICAL_ACTION_KEYS or any(
            not runtime._sha256_text(value) for _key, value in self.action_core_sha256s
        ) or len(self.backend_receipt_sha256s) != len(
            final_capability.CANONICAL_ACTION_KEYS
        ) * BATCH_COUNT or len(set(self.backend_receipt_sha256s)) != len(
            self.backend_receipt_sha256s
        ) or any(not runtime._sha256_text(value) for value in self.backend_receipt_sha256s):
            raise ValueError("observational role receipt is incomplete or duplicated")
        _sha256("role call ledgers", self.call_ledgers_sha256)
        _sha256("role frequency plan", self.frequency_plan_sha256)
        body = {
            "common_support_sha256": self.common_support_sha256,
            "action_core_sha256s": [list(value) for value in self.action_core_sha256s],
            "backend_receipt_sha256s": list(self.backend_receipt_sha256s),
            "frequency_plan_sha256": self.frequency_plan_sha256,
            "call_ledgers_sha256": self.call_ledgers_sha256,
        }
        if not _sha256("observational role receipt", self.receipt_sha256) or (
            runtime.logical_identity_sha256(body) != self.receipt_sha256
        ):
            raise ValueError("observational role receipt identity changed")

    @property
    def sha256(self) -> str:
        return self.receipt_sha256


class FinalObservationalRoleOwner:
    """One-shot canonical owner around a source-closed four-row batch executor."""

    def __init__(
        self, *, issuer_id: str, common_support_sha256: str,
        batch_executor: Callable[[final_capability.FinalAction, int], FinalObservationalBatch],
    ) -> None:
        self._issuer = _sha256("observational issuer", issuer_id)
        self._support = _sha256("observational support", common_support_sha256)
        if not callable(batch_executor):
            raise TypeError("observational role requires a batch executor")
        self._batch_executor = batch_executor
        self._next_action = 0
        self._cores: list[FinalObservationalActionCore] = []
        self._failed = False
        self._receipt: FinalObservationalRoleReceipt | None = None
        self._capability_minted = False

    @property
    def receipt(self) -> FinalObservationalRoleReceipt:
        if self._receipt is None:
            raise RuntimeError("observational role receipt is unavailable before completion")
        return self._receipt

    def _poison(self) -> None:
        self._failed = True
        self._batch_executor = None
        self._receipt = None
        self._cores.clear()

    def _execute_core(
        self, action: final_capability.FinalAction,
    ) -> FinalObservationalActionCore:
        if self._failed or self._receipt is not None or self._next_action >= len(
            final_capability.CANONICAL_ACTIONS
        ) or action != final_capability.CANONICAL_ACTIONS[self._next_action]:
            self._poison()
            raise RuntimeError("observational actions are skipped, reordered, or duplicated")
        accumulator = FinalObservationalActionAccumulator(action, self._support)
        try:
            for ordinal in range(BATCH_COUNT):
                accumulator.add(self._batch_executor(action, ordinal))
            core = accumulator.finish()
        except BaseException:
            self._poison()
            raise
        prior = {
            receipt for value in self._cores for receipt in value.backend_receipt_sha256s
        }
        if any(receipt in prior for receipt in core.backend_receipt_sha256s):
            self._poison()
            raise RuntimeError("observational backend receipt replayed across actions")
        if self._cores and core.frequency_assignment_sha256s != self._cores[
            0
        ].frequency_assignment_sha256s:
            self._poison()
            raise RuntimeError("observational actions mixed frequency support")
        self._cores.append(core)
        self._next_action += 1
        if self._next_action == len(final_capability.CANONICAL_ACTIONS):
            ledgers = {
                value.action.key: dict(value.call_ledger) for value in self._cores
            }
            body = {
                "common_support_sha256": self._support,
                "action_core_sha256s": [
                    [value.action.key, value.core_sha256] for value in self._cores
                ],
                "backend_receipt_sha256s": [
                    receipt for value in self._cores
                    for receipt in value.backend_receipt_sha256s
                ],
                "frequency_plan_sha256": runtime.logical_identity_sha256(
                    list(self._cores[0].frequency_assignment_sha256s)
                ),
                "call_ledgers_sha256": runtime.logical_identity_sha256(ledgers),
            }
            self._receipt = FinalObservationalRoleReceipt(
                common_support_sha256=self._support,
                action_core_sha256s=tuple(
                    (value.action.key, value.core_sha256) for value in self._cores
                ),
                backend_receipt_sha256s=tuple(body["backend_receipt_sha256s"]),
                frequency_plan_sha256=body["frequency_plan_sha256"],
                call_ledgers_sha256=body["call_ledgers_sha256"],
                receipt_sha256=runtime.logical_identity_sha256(body),
            )
            self._batch_executor = None
        return core

    def mint_action_capability(
        self, *, completer: Callable[[FinalObservationalActionCore], final_capability.FinalArmObservation],
    ) -> final_capability.FinalActionCapability:
        if not callable(completer):
            raise TypeError("observational role requires an explicit diagnostic completer")
        if self._capability_minted or self._failed or self._receipt is not None:
            raise RuntimeError("observational role capability was already minted or closed")
        self._capability_minted = True

        def execute(action: final_capability.FinalAction):
            core = self._execute_core(action)
            try:
                observation = completer(core)
            except BaseException:
                self._poison()
                raise
            if type(observation) is not final_capability.FinalArmObservation or (
                observation.action != action
            ) or observation.common_support_sha256 != self._support or observation.ce.sha256 != (
                core.ce.sha256
            ) or observation.copy_ce.sha256 != core.copy_ce.sha256 or tuple(
                value.sha256 for value in observation.frequency_ce
            ) != tuple(value.sha256 for value in core.frequency_ce) or (
                None if observation.teacher_kl is None else observation.teacher_kl.sha256
            ) != (None if core.primary is None else core.primary.sha256):
                self._poison()
                raise RuntimeError("observational completer changed owned reductions")
            return observation

        return final_capability.mint_final_action_capability(
            issuer_id=self._issuer, common_support_sha256=self._support,
            executor=execute,
        )
