"""Pure call plan and receipt binding for the final paired-response transaction.

This module performs no I/O and cannot run a model.  It closes the scheduling layer
between the canonical 22 response actions and a future observed adapter transaction:

* one shared exact-teacher baseline/positive/negative triplet per four-row batch;
* one student baseline/positive/negative triplet for LL, LT, and each indexed null;
* exact action, perturbation, row, intervention-unit, and expected-call bindings;
* a tensor-free receipt that rejects reordered, duplicated, or relabelled forwards.

The observed backend must execute this plan atomically and reduce raw teacher/student
states internally.  A valid receipt proves scheduling and provenance, not scientific
success and not that the numerical reductions are correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import early_mlp_suffix_transport_v1_final_actions as final_actions
import early_mlp_suffix_transport_v1_runtime as runtime


PERTURBATIONS = ("baseline", "positive", "negative")
RESPONSE_ACTION_KEYS = tuple(
    f"{arm}/N" for arm in final_actions.BASE_ARMS
    if arm in final_actions.RESPONSE_ARMS
)
TEACHER_KEY = "exact_teacher/N"
EXPECTED_BATCHES = final_actions.OBSERVATIONAL_BATCH_COUNT
TEACHER_FORWARDS_PER_BATCH = len(PERTURBATIONS)
STUDENT_FORWARDS_PER_BATCH = len(RESPONSE_ACTION_KEYS) * len(PERTURBATIONS)
TOTAL_TEACHER_FORWARDS = EXPECTED_BATCHES * TEACHER_FORWARDS_PER_BATCH
TOTAL_STUDENT_FORWARDS = EXPECTED_BATCHES * STUDENT_FORWARDS_PER_BATCH


def _sha256(name: str, value: Any) -> str:
    if not runtime._sha256_text(value):
        raise ValueError(f"{name} is not a SHA-256 identity")
    return value


def _pattern_sha256(pattern: final_actions.FinalEarlyCallPattern) -> str:
    return runtime.logical_identity_sha256({
        name: list(getattr(pattern, name)) for name in pattern.__dataclass_fields__
    })


TEACHER_CALL_PATTERN = final_actions.FinalEarlyCallPattern(
    deployed_n_calls=((0, 0), (1, 0), (2, 1)),
    correction_calls=((0, 0), (1, 0), (2, 0)),
    literal_early_mlp_calls=((0, 1), (1, 1), (2, 0)),
)


@dataclass(frozen=True, slots=True)
class ResponseForwardPlan:
    """Identity of one physical forward inside a paired-response batch."""

    subject_key: str
    perturbation: str
    edit_sign: int
    batch_ordinal: int
    ordered_role_rows_sha256: str
    intervention_unit_sha256: str
    action_plan_sha256: str
    expected_call_pattern_sha256: str
    shared_teacher: bool

    def __post_init__(self) -> None:
        if self.subject_key != TEACHER_KEY and self.subject_key not in (
            RESPONSE_ACTION_KEYS
        ):
            raise ValueError("response forward subject is outside the registered plan")
        if self.perturbation not in PERTURBATIONS or self.edit_sign != {
            "baseline": 0, "positive": 1, "negative": -1,
        }[self.perturbation]:
            raise ValueError("response perturbation/sign binding changed")
        if type(self.batch_ordinal) is not int or not 0 <= self.batch_ordinal < (
            EXPECTED_BATCHES
        ):
            raise ValueError("response batch ordinal changed")
        _sha256("ordered response rows", self.ordered_role_rows_sha256)
        _sha256("response intervention unit", self.intervention_unit_sha256)
        _sha256("response action plan", self.action_plan_sha256)
        _sha256("response call pattern", self.expected_call_pattern_sha256)
        teacher = self.subject_key == TEACHER_KEY
        if self.shared_teacher is not teacher:
            raise ValueError("response teacher sharing flag changed")
        if teacher:
            expected_action = runtime.logical_identity_sha256({
                "subject": TEACHER_KEY, "early_sites": "exact",
                "mlp2_background": "N",
            })
            expected_pattern = _pattern_sha256(TEACHER_CALL_PATTERN)
        else:
            arm, background = self.subject_key.split("/")
            action = final_actions.plan_for(arm, background)
            expected_action = action.sha256
            expected_pattern = _pattern_sha256(
                final_actions.expected_early_call_pattern(action)
            )
        if self.action_plan_sha256 != expected_action or (
            self.expected_call_pattern_sha256 != expected_pattern
        ):
            raise ValueError("response forward physical action binding changed")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            name: getattr(self, name) for name in self.__dataclass_fields__
        })


@dataclass(frozen=True, slots=True)
class ResponseBatchPlan:
    """The only licensed ordering of the 69 forwards in one response batch."""

    batch_ordinal: int
    ordered_role_rows_sha256: str
    intervention_unit_sha256: str
    forwards: tuple[ResponseForwardPlan, ...]

    def __post_init__(self) -> None:
        if type(self.batch_ordinal) is not int or not 0 <= self.batch_ordinal < (
            EXPECTED_BATCHES
        ):
            raise ValueError("response batch plan ordinal changed")
        _sha256("response batch rows", self.ordered_role_rows_sha256)
        _sha256("response batch intervention unit", self.intervention_unit_sha256)
        expected = _forward_plan(
            batch_ordinal=self.batch_ordinal,
            ordered_role_rows_sha256=self.ordered_role_rows_sha256,
            intervention_unit_sha256=self.intervention_unit_sha256,
        )
        if not isinstance(self.forwards, tuple) or self.forwards != expected:
            raise ValueError("response batch forwards are incomplete or reordered")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "batch_ordinal": self.batch_ordinal,
            "ordered_role_rows_sha256": self.ordered_role_rows_sha256,
            "intervention_unit_sha256": self.intervention_unit_sha256,
            "forward_sha256s": [value.sha256 for value in self.forwards],
        })


def _subject(
    *, subject_key: str, perturbation: str, batch_ordinal: int,
    ordered_role_rows_sha256: str, intervention_unit_sha256: str,
) -> ResponseForwardPlan:
    teacher = subject_key == TEACHER_KEY
    if teacher:
        action_sha256 = runtime.logical_identity_sha256({
            "subject": TEACHER_KEY, "early_sites": "exact",
            "mlp2_background": "N",
        })
        pattern = TEACHER_CALL_PATTERN
    else:
        arm, background = subject_key.split("/")
        action = final_actions.plan_for(arm, background)
        action_sha256 = action.sha256
        pattern = final_actions.expected_early_call_pattern(action)
    return ResponseForwardPlan(
        subject_key=subject_key, perturbation=perturbation,
        edit_sign={"baseline": 0, "positive": 1, "negative": -1}[perturbation],
        batch_ordinal=batch_ordinal,
        ordered_role_rows_sha256=ordered_role_rows_sha256,
        intervention_unit_sha256=intervention_unit_sha256,
        action_plan_sha256=action_sha256,
        expected_call_pattern_sha256=_pattern_sha256(pattern),
        shared_teacher=teacher,
    )


def _forward_plan(
    *, batch_ordinal: int, ordered_role_rows_sha256: str,
    intervention_unit_sha256: str,
) -> tuple[ResponseForwardPlan, ...]:
    subjects = (TEACHER_KEY, *RESPONSE_ACTION_KEYS)
    return tuple(
        _subject(
            subject_key=subject, perturbation=perturbation,
            batch_ordinal=batch_ordinal,
            ordered_role_rows_sha256=ordered_role_rows_sha256,
            intervention_unit_sha256=intervention_unit_sha256,
        )
        for subject in subjects for perturbation in PERTURBATIONS
    )


def build_response_batch_plan(
    *, batch_ordinal: int, ordered_role_rows_sha256: str,
    intervention_unit_sha256: str,
) -> ResponseBatchPlan:
    forwards = _forward_plan(
        batch_ordinal=batch_ordinal,
        ordered_role_rows_sha256=ordered_role_rows_sha256,
        intervention_unit_sha256=intervention_unit_sha256,
    )
    return ResponseBatchPlan(
        batch_ordinal=batch_ordinal,
        ordered_role_rows_sha256=ordered_role_rows_sha256,
        intervention_unit_sha256=intervention_unit_sha256,
        forwards=forwards,
    )


@dataclass(frozen=True, slots=True)
class ResponseForwardReceipt:
    """Tensor-free proof that an observed forward followed one planned identity."""

    forward_plan_sha256: str
    subject_key: str
    perturbation: str
    batch_ordinal: int
    observed_call_pattern_sha256: str
    physical_edit_sha256: str
    observed_closure_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "forward_plan_sha256", "observed_call_pattern_sha256",
            "physical_edit_sha256", "observed_closure_sha256",
        ):
            _sha256(name, getattr(self, name))
        if self.subject_key != TEACHER_KEY and self.subject_key not in (
            RESPONSE_ACTION_KEYS
        ) or self.perturbation not in PERTURBATIONS or type(
            self.batch_ordinal
        ) is not int:
            raise ValueError("response forward receipt header is malformed")


@dataclass(frozen=True, slots=True)
class ResponseBatchReceipt:
    """Exact, ordered closure of one atomic paired-response batch."""

    batch_plan_sha256: str
    forward_receipts: tuple[ResponseForwardReceipt, ...]
    reduction_sha256s: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _sha256("response batch plan", self.batch_plan_sha256)
        if not isinstance(self.forward_receipts, tuple) or len(
            self.forward_receipts
        ) != TEACHER_FORWARDS_PER_BATCH + STUDENT_FORWARDS_PER_BATCH:
            raise ValueError("response batch receipt has the wrong forward count")
        if not isinstance(self.reduction_sha256s, tuple) or tuple(
            key for key, _value in self.reduction_sha256s
        ) != RESPONSE_ACTION_KEYS or any(
            not runtime._sha256_text(value) for _key, value in self.reduction_sha256s
        ):
            raise ValueError("response batch reductions are incomplete or reordered")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "batch_plan_sha256": self.batch_plan_sha256,
            "forward_receipts": [
                {
                    name: getattr(value, name)
                    for name in value.__dataclass_fields__
                }
                for value in self.forward_receipts
            ],
            "reduction_sha256s": list(self.reduction_sha256s),
        })


def seal_response_batch(
    *, plan: ResponseBatchPlan,
    forward_receipts: Sequence[ResponseForwardReceipt],
    reduction_sha256s: Mapping[str, str],
) -> ResponseBatchReceipt:
    """Bind observed receipts to the plan; reject arm/sign/row fabrication."""

    if not isinstance(plan, ResponseBatchPlan):
        raise TypeError("response sealing requires a typed batch plan")
    supplied = tuple(forward_receipts)
    if len(supplied) != len(plan.forwards):
        raise RuntimeError("response forward receipt count changed")
    for expected, observed in zip(plan.forwards, supplied, strict=True):
        if not isinstance(observed, ResponseForwardReceipt) or (
            observed.forward_plan_sha256 != expected.sha256
        ) or observed.subject_key != expected.subject_key or (
            observed.perturbation != expected.perturbation
        ) or observed.batch_ordinal != expected.batch_ordinal or (
            observed.observed_call_pattern_sha256
            != expected.expected_call_pattern_sha256
        ):
            raise RuntimeError("response receipt differs from its planned forward")
    if not isinstance(reduction_sha256s, Mapping) or tuple(
        reduction_sha256s
    ) != RESPONSE_ACTION_KEYS:
        raise RuntimeError("response reduction labels are incomplete or reordered")
    return ResponseBatchReceipt(
        batch_plan_sha256=plan.sha256,
        forward_receipts=supplied,
        reduction_sha256s=tuple(reduction_sha256s.items()),
    )


def expected_full_call_ledger() -> dict[str, Any]:
    """Registered full-run forward counts, independent of numerical outcomes."""

    return {
        "batches": EXPECTED_BATCHES,
        "response_actions": len(RESPONSE_ACTION_KEYS),
        "teacher_forwards": TOTAL_TEACHER_FORWARDS,
        "student_forwards": TOTAL_STUDENT_FORWARDS,
        "teacher_triplets": EXPECTED_BATCHES,
        "student_triplets": EXPECTED_BATCHES * len(RESPONSE_ACTION_KEYS),
        "forwards_per_batch": TEACHER_FORWARDS_PER_BATCH + STUDENT_FORWARDS_PER_BATCH,
    }
