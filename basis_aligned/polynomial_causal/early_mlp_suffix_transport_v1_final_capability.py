"""Typed, fail-closed action/observation boundary for suffix final execution.

The final semantic owner must never receive a model, role rows, logits, residual
states, or a dispatcher.  A future observed adapter instead closes over those objects
and mints :class:`FinalActionCapability` with an executor that accepts one immutable
action and returns only the reductions defined here.

This module performs no I/O and has no model/data loader.  It fixes canonical action
coverage, sufficient-statistic shapes, response-arm semantics, and single-use failure
closure before a final adapter backend is authorized.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

import early_mlp_suffix_transport_v1_final_actions as final_actions
import early_mlp_suffix_transport_v1_runtime as runtime
import early_mlp_suffix_transport_v1_statistics as statistics


FINAL_ROW_COUNT = 192
SCORED_TOKENS_PER_ROW = 192
FREQUENCY_BIN_COUNT = 9
MODEL_LAYER_COUNT = 18

BASE_ARMS = final_actions.BASE_ARMS
BACKGROUNDS = final_actions.BACKGROUNDS
CANONICAL_ACTION_KEYS = final_actions.CANONICAL_ACTION_KEYS
_RESPONSE_ARMS = final_actions.RESPONSE_ARMS
_CODE_RESPONSE_ARMS = final_actions.CODE_RESPONSE_ARMS
RESPONSE_ACTION_KEYS = (
    "ll/N", "lt/N", *(f"a_null_{index:02d}/N" for index in range(20)),
)
_MINT_TOKEN = object()


def _sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} is not a SHA-256 identity")
    return value


def _row_vector(name: str, value: Any, *, count: bool) -> torch.Tensor:
    expected_dtype = torch.long if count else torch.float64
    if (
        not torch.is_tensor(value)
        or value.ndim != 1
        or tuple(value.shape) != (FINAL_ROW_COUNT,)
        or value.dtype != expected_dtype
        or value.device.type != "cpu"
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
        or bool((value < 0).any())
    ):
        raise ValueError(f"{name} is not an allowed final row reduction")
    return value.detach().clone().contiguous()


@dataclass(frozen=True, slots=True)
class FinalAction:
    arm: str
    background: str

    def __post_init__(self) -> None:
        if self.arm not in BASE_ARMS or self.background not in BACKGROUNDS:
            raise ValueError("final action is outside the registered lattice")

    @property
    def key(self) -> str:
        return f"{self.arm}/{self.background}"

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "arm": self.arm, "background": self.background,
        })


CANONICAL_ACTIONS = tuple(
    FinalAction(arm=plan.arm_plan.arm, background=plan.background)
    for plan in final_actions.CANONICAL_ACTION_PLANS
)


@dataclass(frozen=True, slots=True)
class RowReduction:
    """Per-final-row numerator and count for a pooled scalar statistic."""

    row_sum: torch.Tensor
    row_count: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "row_sum", _row_vector("row sum", self.row_sum, count=False),
        )
        object.__setattr__(
            self, "row_count", _row_vector("row count", self.row_count, count=True),
        )
        if bool((self.row_count <= 0).any()):
            raise ValueError("final row reduction has empty row support")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "row_sum": runtime.tensor_identity_sha256(self.row_sum),
            "row_count": runtime.tensor_identity_sha256(self.row_count),
        })


@dataclass(frozen=True, slots=True)
class ResponseReduction:
    """Per-row vector-response inner products; never the response vectors."""

    error_sum: torch.Tensor
    teacher_sum: torch.Tensor
    student_sum: torch.Tensor
    dot_sum: torch.Tensor
    unit_identity: str

    def __post_init__(self) -> None:
        payload = {
            "error_sum": self.error_sum,
            "teacher_sum": self.teacher_sum,
            "student_sum": self.student_sum,
            "dot_sum": self.dot_sum,
            "unit_identity": self.unit_identity,
        }
        checked = statistics.validate_response_sufficient_statistics(
            payload, length=FINAL_ROW_COUNT,
        )
        _sha256("response unit identity", self.unit_identity)
        for name, value in checked.items():
            object.__setattr__(self, name, value.detach().clone().contiguous())

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            name: runtime.tensor_identity_sha256(getattr(self, name))
            for name in statistics.RESPONSE_KEYS
        } | {"unit_identity": self.unit_identity})


@dataclass(frozen=True, slots=True)
class OutputKLReduction:
    """Per-row sums for output KL relative to the exact teacher edit size."""

    numerator_sum: torch.Tensor
    denominator_sum: torch.Tensor
    unit_identity: str

    def __post_init__(self) -> None:
        payload = {
            "numerator_sum": self.numerator_sum,
            "denominator_sum": self.denominator_sum,
            "unit_identity": self.unit_identity,
        }
        checked = statistics.validate_output_kl_sufficient_statistics(
            payload, length=FINAL_ROW_COUNT,
        )
        _sha256("output-KL unit identity", self.unit_identity)
        for name, value in checked.items():
            object.__setattr__(self, name, value.detach().clone().contiguous())

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            name: runtime.tensor_identity_sha256(getattr(self, name))
            for name in statistics.OUTPUT_KL_KEYS
        } | {"unit_identity": self.unit_identity})


@dataclass(frozen=True, slots=True)
class FinalArmObservation:
    """The complete permitted output of one final action."""

    action: FinalAction
    common_support_sha256: str
    ce: RowReduction
    teacher_kl: RowReduction | None
    copy_ce: RowReduction
    frequency_ce: tuple[RowReduction, ...]
    code_response: ResponseReduction | None
    logit_response: ResponseReduction | None
    output_kl_response: OutputKLReduction | None
    consumer_norm_ratio: tuple[RowReduction, ...]
    execution_closure_sha256: str

    def __post_init__(self) -> None:
        if type(self.action) is not FinalAction:
            raise ValueError("final observation action is not typed")
        _sha256("common support", self.common_support_sha256)
        _sha256("execution closure", self.execution_closure_sha256)
        if type(self.ce) is not RowReduction or type(self.copy_ce) is not RowReduction:
            raise ValueError("final CE reductions are not typed")
        if self.action.background == "N":
            if type(self.teacher_kl) is not RowReduction:
                raise ValueError("deployed-MLP2 action requires teacher-KL reductions")
        elif self.teacher_kl is not None:
            raise ValueError("exact-MLP2 background is CE-only")
        if not isinstance(self.frequency_ce, tuple) or len(self.frequency_ce) != (
            FREQUENCY_BIN_COUNT
        ) or any(type(value) is not RowReduction for value in self.frequency_ce):
            raise ValueError("final observation requires all nine frequency bins")
        if not isinstance(self.consumer_norm_ratio, tuple) or len(
            self.consumer_norm_ratio
        ) != MODEL_LAYER_COUNT or any(
            type(value) is not RowReduction for value in self.consumer_norm_ratio
        ):
            raise ValueError("final observation requires all live-consumer norm ratios")

        response_expected = self.action.background == "N" and self.action.arm in (
            _RESPONSE_ARMS
        )
        code_expected = self.action.background == "N" and self.action.arm in (
            _CODE_RESPONSE_ARMS
        )
        if (self.logit_response is not None) != response_expected or (
            self.code_response is not None
        ) != code_expected or (self.output_kl_response is not None) != response_expected:
            raise ValueError("response reductions do not match the registered action")
        if self.logit_response is not None and type(
            self.logit_response
        ) is not ResponseReduction:
            raise ValueError("logit response is not typed")
        if self.code_response is not None and type(
            self.code_response
        ) is not ResponseReduction:
            raise ValueError("code response is not typed")
        if self.output_kl_response is not None and type(
            self.output_kl_response
        ) is not OutputKLReduction:
            raise ValueError("output-KL response is not typed")
        if self.code_response is not None and self.logit_response is not None and (
            self.code_response.unit_identity != self.logit_response.unit_identity
        ):
            raise ValueError("code and logit responses use different intervention units")
        if self.output_kl_response is not None and self.logit_response is not None and (
            self.output_kl_response.unit_identity != self.logit_response.unit_identity
        ):
            raise ValueError("output-KL and vector responses use different intervention units")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "action_sha256": self.action.sha256,
            "common_support_sha256": self.common_support_sha256,
            "ce_sha256": self.ce.sha256,
            "teacher_kl_sha256": (
                None if self.teacher_kl is None else self.teacher_kl.sha256
            ),
            "copy_ce_sha256": self.copy_ce.sha256,
            "frequency_ce_sha256s": [value.sha256 for value in self.frequency_ce],
            "code_response_sha256": (
                None if self.code_response is None else self.code_response.sha256
            ),
            "logit_response_sha256": (
                None if self.logit_response is None else self.logit_response.sha256
            ),
            "output_kl_response_sha256": (
                None if self.output_kl_response is None
                else self.output_kl_response.sha256
            ),
            "consumer_norm_ratio_sha256s": [
                value.sha256 for value in self.consumer_norm_ratio
            ],
            "execution_closure_sha256": self.execution_closure_sha256,
        })


@dataclass(frozen=True, slots=True)
class FinalObservationBundle:
    common_support_sha256: str
    observations: tuple[FinalArmObservation, ...]
    action_plan_sha256: str
    bundle_sha256: str

    def __post_init__(self) -> None:
        _sha256("bundle common support", self.common_support_sha256)
        _sha256("action plan", self.action_plan_sha256)
        _sha256("bundle", self.bundle_sha256)
        if tuple(value.action for value in self.observations) != CANONICAL_ACTIONS:
            raise ValueError("final observation bundle is incomplete or reordered")
        if any(
            value.common_support_sha256 != self.common_support_sha256
            for value in self.observations
        ):
            raise ValueError("final observation bundle mixes scored support")


@dataclass(frozen=True, slots=True)
class FinalEvidenceJoinReceipt:
    """Tensor-free proof that observational and paired-response owners agree.

    The observational capability and response runner intentionally own different
    transactions.  This receipt prevents a final callback from combining the 68
    observational arms from one run with response statistics from another.
    """

    observation_bundle_sha256: str
    response_run_receipt_sha256: str
    program_payload_sha256: str
    common_support_sha256: str
    ordered_unit_identity_sha256: str
    response_action_matches: tuple[tuple[str, str, str], ...]
    response_statistics_sha256: str
    join_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "observation_bundle_sha256", "response_run_receipt_sha256",
            "program_payload_sha256", "common_support_sha256",
            "ordered_unit_identity_sha256", "response_statistics_sha256",
            "join_sha256",
        ):
            _sha256(name, getattr(self, name))
        if not isinstance(self.response_action_matches, tuple) or tuple(
            key for key, _observation, _run in self.response_action_matches
        ) != RESPONSE_ACTION_KEYS or any(
            not _sha256("response observation", observation)
            or not _sha256("response run reduction", run)
            or observation != run
            for _key, observation, run in self.response_action_matches
        ):
            raise ValueError("final evidence join does not match every response action")
        body = {
            name: (
                [list(value) for value in self.response_action_matches]
                if name == "response_action_matches" else getattr(self, name)
            )
            for name in self.__dataclass_fields__ if name != "join_sha256"
        }
        if runtime.logical_identity_sha256(body) != self.join_sha256:
            raise ValueError("final evidence join identity changed")

    @property
    def sha256(self) -> str:
        return self.join_sha256


def _response_statistics_identity(payload: Mapping[str, Any]) -> str:
    """Hash exactly the final-owner response roles through their typed schemas."""

    keys = {
        "response_run_receipt_sha256", "ordered_unit_identity_sha256",
        "code_baseline", "code_candidate", "logit_baseline", "logit_candidate",
        "logit_nulls", "output_kl_baseline", "output_kl_candidate",
        "output_kl_nulls",
    }
    if not isinstance(payload, Mapping) or set(payload) != keys or not _sha256(
        "response run receipt", payload["response_run_receipt_sha256"]
    ) or not _sha256("ordered response units", payload["ordered_unit_identity_sha256"]):
        raise ValueError("final response statistics payload schema changed")
    if not isinstance(payload["logit_nulls"], tuple) or len(
        payload["logit_nulls"]
    ) != 20 or not isinstance(payload["output_kl_nulls"], tuple) or len(
        payload["output_kl_nulls"]
    ) != 20:
        raise ValueError("final response null payload changed")
    typed = {
        "code_baseline": ResponseReduction(**payload["code_baseline"]),
        "code_candidate": ResponseReduction(**payload["code_candidate"]),
        "logit_baseline": ResponseReduction(**payload["logit_baseline"]),
        "logit_candidate": ResponseReduction(**payload["logit_candidate"]),
        "output_kl_baseline": OutputKLReduction(**payload["output_kl_baseline"]),
        "output_kl_candidate": OutputKLReduction(**payload["output_kl_candidate"]),
    }
    logit_nulls = tuple(ResponseReduction(**value) for value in payload["logit_nulls"])
    output_nulls = tuple(
        OutputKLReduction(**value) for value in payload["output_kl_nulls"]
    )
    identities = {
        value.unit_identity for value in (*typed.values(), *logit_nulls, *output_nulls)
    }
    if identities != {payload["ordered_unit_identity_sha256"]}:
        raise ValueError("final response statistics mix ordered units")
    return runtime.logical_identity_sha256({
        "response_run_receipt_sha256": payload["response_run_receipt_sha256"],
        "ordered_unit_identity_sha256": payload["ordered_unit_identity_sha256"],
        **{name: value.sha256 for name, value in typed.items()},
        "logit_null_sha256s": [value.sha256 for value in logit_nulls],
        "output_kl_null_sha256s": [value.sha256 for value in output_nulls],
    })


def join_observations_with_response_run(
    observations: FinalObservationBundle, response_run: Any,
) -> FinalEvidenceJoinReceipt:
    """Bind the canonical action bundle to one completed response-run result."""

    import early_mlp_suffix_transport_v1_response_execution as response_execution

    if type(observations) is not FinalObservationBundle or type(
        response_run
    ) is not response_execution.ObservedResponseRunResult:
        raise TypeError("final evidence join requires typed complete owners")
    expected_plan = runtime.logical_identity_sha256(list(CANONICAL_ACTION_KEYS))
    if observations.action_plan_sha256 != expected_plan or response_run.receipt.atomic_complete is not (
        True
    ) or observations.common_support_sha256 != response_run.receipt.common_support_sha256:
        raise RuntimeError("final evidence owners differ in plan, completion, or support")
    payload = response_run.to_final_statistics_payload()
    if payload["response_run_receipt_sha256"] != response_run.receipt.sha256:
        raise RuntimeError("final response payload escaped its run receipt")
    by_action = {value.action.key: value for value in observations.observations}
    by_run = {value.action_key: value for value in response_run.arm_reductions}
    matches = []
    for action_key in RESPONSE_ACTION_KEYS:
        observation = by_action[action_key]
        run = by_run[action_key]
        run_identity = runtime.logical_identity_sha256({
            "code_response_sha256": (
                None if run.code_response is None else ResponseReduction(
                    **run.code_response.as_statistics(
                        response_run.receipt.ordered_unit_identity_sha256
                    )
                ).sha256
            ),
            "logit_response_sha256": ResponseReduction(
                **run.logit_response.as_statistics(
                    response_run.receipt.ordered_unit_identity_sha256
                )
            ).sha256,
            "output_kl_response_sha256": OutputKLReduction(
                **run.output_kl_response.as_statistics(
                    response_run.receipt.ordered_unit_identity_sha256
                )
            ).sha256,
        })
        observation_identity = runtime.logical_identity_sha256({
            "code_response_sha256": (
                None if observation.code_response is None
                else observation.code_response.sha256
            ),
            "logit_response_sha256": observation.logit_response.sha256,
            "output_kl_response_sha256": observation.output_kl_response.sha256,
        })
        if observation_identity != run_identity:
            raise RuntimeError(f"{action_key} observation differs from response run")
        matches.append((action_key, observation_identity, run_identity))
    body = {
        "observation_bundle_sha256": observations.bundle_sha256,
        "response_run_receipt_sha256": response_run.receipt.sha256,
        "program_payload_sha256": response_run.receipt.program_payload_sha256,
        "common_support_sha256": observations.common_support_sha256,
        "ordered_unit_identity_sha256": response_run.receipt.ordered_unit_identity_sha256,
        "response_action_matches": [list(value) for value in matches],
        "response_statistics_sha256": _response_statistics_identity(payload),
    }
    return FinalEvidenceJoinReceipt(
        **{key: value for key, value in body.items() if key != "response_action_matches"},
        response_action_matches=tuple(matches),
        join_sha256=runtime.logical_identity_sha256(body),
    )


class FinalActionCapability:
    """Single-use canonical executor; callers cannot choose or repeat final actions."""

    def __init__(
        self, *, _token: object, issuer_id: str, common_support_sha256: str,
        executor: Callable[[FinalAction], FinalArmObservation],
    ) -> None:
        if _token is not _MINT_TOKEN or not callable(executor):
            raise TypeError("final action capability must be minted by the observed boundary")
        self.__issuer_id = _sha256("final capability issuer", issuer_id)
        self.__support = _sha256("final capability support", common_support_sha256)
        self.__executor = executor
        self.__spent = False
        self.__failed = False

    @property
    def spent(self) -> bool:
        return self.__spent

    @property
    def failed(self) -> bool:
        return self.__failed

    def execute_all(self) -> FinalObservationBundle:
        if self.__spent or self.__failed:
            raise RuntimeError("final action capability is already closed")
        object.__setattr__(self, "_FinalActionCapability__spent", True)
        observations: list[FinalArmObservation] = []
        try:
            for action in CANONICAL_ACTIONS:
                value = self.__executor(action)
                if type(value) is not FinalArmObservation or value.action != action:
                    raise RuntimeError("final executor returned the wrong typed action")
                if value.common_support_sha256 != self.__support:
                    raise RuntimeError("final executor mixed scored support")
                observations.append(value)
        except BaseException:
            observations.clear()
            object.__setattr__(self, "_FinalActionCapability__failed", True)
            object.__setattr__(self, "_FinalActionCapability__executor", None)
            raise
        object.__setattr__(self, "_FinalActionCapability__executor", None)
        plan_sha256 = runtime.logical_identity_sha256(list(CANONICAL_ACTION_KEYS))
        observation_sha256s = [value.sha256 for value in observations]
        bundle_sha256 = runtime.logical_identity_sha256({
            "issuer_id": self.__issuer_id,
            "common_support_sha256": self.__support,
            "action_plan_sha256": plan_sha256,
            "observation_sha256s": observation_sha256s,
        })
        return FinalObservationBundle(
            common_support_sha256=self.__support,
            observations=tuple(observations),
            action_plan_sha256=plan_sha256,
            bundle_sha256=bundle_sha256,
        )


def mint_final_action_capability(
    *, issuer_id: str, common_support_sha256: str,
    executor: Callable[[FinalAction], FinalArmObservation],
) -> FinalActionCapability:
    """Adapter-side factory; the executor must close over model/rows and return reductions."""
    return FinalActionCapability(
        _token=_MINT_TOKEN, issuer_id=issuer_id,
        common_support_sha256=common_support_sha256, executor=executor,
    )
