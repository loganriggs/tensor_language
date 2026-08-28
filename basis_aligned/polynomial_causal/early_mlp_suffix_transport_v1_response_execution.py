"""Authority-derived physical edits for paired finite-response execution.

This module performs no model I/O.  It derives the only licensed four-row MLP0 edit
from the validated canonical program bank, final role rows, canonical batch indices,
and the fitted MLP0 output basis.  The observed adapter consumes the result; callers
cannot supply positions, directions, amplitude, or a residual-stream write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

import early_mlp_suffix_transport_v1 as contract
import early_mlp_suffix_transport_v1_final_actions as final_actions
import early_mlp_suffix_transport_v1_programs as programs
import early_mlp_suffix_transport_v1_response_plan as response_plan
import early_mlp_suffix_transport_v1_response_reductions as response_reductions
import early_mlp_suffix_transport_v1_runtime as runtime


FINAL_ROW_WIDTH = 513
EDIT_SIGNS = (-1, 0, 1)
_MINT_TOKEN = object()
PERTURBATION_TRIAL = {"baseline": 0, "positive": 1, "negative": 2}
RESPONSE_EXECUTION_AMENDMENT_SHA256 = (
    "8c5162ce1c621850b7e96a42f64205f803dd4fd772b55f349b53a655da2ce6ba"
)


def _sha256(name: str, value: Any) -> str:
    if not runtime._sha256_text(value):
        raise ValueError(f"{name} must be a SHA-256 identity")
    return value


@dataclass(frozen=True, slots=True)
class FinalResponseEdit:
    """One immutable, authority-bound code edit and matching physical write."""

    _mint_token: Any
    batch_ordinal: int
    edit_sign: int
    unit_identity_sha256: str
    ordered_role_rows_sha256: str
    ordered_input_tokens_sha256: str
    program_payload_sha256: str
    geometry_sha256: str
    calibration_sha256: str
    basis0_sha256: str
    positions_sha256: str
    direction_indices_sha256: str
    semantic_delta_sha256: str
    code_edit_sha256: str
    physical_edit_sha256: str
    positions: torch.Tensor
    direction_indices: torch.Tensor
    semantic_delta: torch.Tensor
    code_edit: torch.Tensor
    physical_edit: torch.Tensor

    def __post_init__(self) -> None:
        if self._mint_token is not _MINT_TOKEN:
            raise ValueError("final response edits can only be authority-minted")
        if type(self.batch_ordinal) is not int or not 0 <= self.batch_ordinal < 48 or (
            self.edit_sign not in EDIT_SIGNS
        ):
            raise ValueError("final response edit schedule is malformed")
        for name in (
            "unit_identity_sha256", "ordered_role_rows_sha256",
            "ordered_input_tokens_sha256", "program_payload_sha256",
            "geometry_sha256", "calibration_sha256", "basis0_sha256",
            "positions_sha256", "direction_indices_sha256", "code_edit_sha256",
            "physical_edit_sha256", "semantic_delta_sha256",
        ):
            _sha256(name, getattr(self, name))
        tensors = {
            "positions": (self.positions, (runtime.BATCH_SIZE,), torch.long),
            "direction_indices": (
                self.direction_indices, (runtime.BATCH_SIZE,), torch.long,
            ),
            "semantic_delta": (
                self.semantic_delta,
                (runtime.BATCH_SIZE, runtime.CODE_DIM), torch.float64,
            ),
            "code_edit": (
                self.code_edit,
                (runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.CODE_DIM),
                torch.float32,
            ),
            "physical_edit": (
                self.physical_edit,
                (runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.D_MODEL),
                torch.float32,
            ),
        }
        for name, (value, shape, dtype) in tensors.items():
            if not torch.is_tensor(value) or tuple(value.shape) != shape or value.dtype != (
                dtype
            ) or value.device.type != "cpu" or value.requires_grad or value.grad_fn is not (
                None
            ) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"final response {name} is malformed")
            object.__setattr__(self, name, value.detach().clone().contiguous())
        expected_tensor_hashes = {
            "positions_sha256": runtime.tensor_identity_sha256(self.positions),
            "direction_indices_sha256": runtime.tensor_identity_sha256(
                self.direction_indices,
            ),
            "semantic_delta_sha256": runtime.tensor_identity_sha256(
                self.semantic_delta,
            ),
            "code_edit_sha256": runtime.tensor_identity_sha256(self.code_edit),
            "physical_edit_sha256": runtime.tensor_identity_sha256(self.physical_edit),
        }
        if any(getattr(self, name) != value for name, value in expected_tensor_hashes.items()):
            raise ValueError("final response edit tensor identity changed during minting")
        if bool((self.positions < runtime.SCORE_START).any()) or bool((
            self.positions >= runtime.SCORE_STOP
        ).any()) or bool((self.direction_indices < 0).any()) or bool((
            self.direction_indices >= 32
        ).any()):
            raise ValueError("final response edit assignment is out of range")
        nonzero_positions = self.code_edit.abs().sum(dim=-1) > 0
        expected_nonzero = torch.zeros(
            runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, dtype=torch.bool,
        )
        if self.edit_sign:
            expected_nonzero[torch.arange(runtime.BATCH_SIZE), self.positions] = True
        if not torch.equal(nonzero_positions, expected_nonzero):
            raise ValueError("final response code edit support changed")
        if self.edit_sign == 0 and (
            bool(self.code_edit.any()) or bool(self.physical_edit.any())
        ):
            raise ValueError("baseline response edit is not exactly zero")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            name: getattr(self, name) for name in (
                "batch_ordinal", "edit_sign", "unit_identity_sha256",
                "ordered_role_rows_sha256", "ordered_input_tokens_sha256",
                "program_payload_sha256", "geometry_sha256",
                "calibration_sha256", "basis0_sha256", "positions_sha256",
                "direction_indices_sha256", "semantic_delta_sha256", "code_edit_sha256",
                "physical_edit_sha256",
            )
        } | {
            name: runtime.tensor_identity_sha256(getattr(self, name))
            for name in (
                "positions", "direction_indices", "semantic_delta", "code_edit",
                "physical_edit",
            )
        })

    def require_pristine(
        self, *, role_rows: torch.Tensor, ordered_batch_indices: Sequence[int],
        basis0: torch.Tensor,
    ) -> None:
        indices = tuple(ordered_batch_indices)
        expected_indices = tuple(range(
            self.batch_ordinal * runtime.BATCH_SIZE,
            (self.batch_ordinal + 1) * runtime.BATCH_SIZE,
        ))
        if not torch.is_tensor(role_rows) or role_rows.dtype != torch.long or tuple(
            role_rows.shape
        ) != (runtime.BATCH_SIZE, FINAL_ROW_WIDTH) or role_rows.device.type != (
            "cpu"
        ) or not role_rows.is_contiguous() or indices != expected_indices or (
            runtime.tensor_identity_sha256(role_rows) != self.ordered_role_rows_sha256
        ) or runtime.tensor_identity_sha256(
            role_rows[:, :runtime.SEQUENCE_LENGTH].contiguous()
        ) != self.ordered_input_tokens_sha256:
            raise RuntimeError("final response edit rows or schedule changed")
        checked_basis = basis0.detach().cpu().float().contiguous()
        contract.validate_orthonormal_basis("response basis0", checked_basis)
        if runtime.tensor_identity_sha256(checked_basis) != self.basis0_sha256:
            raise RuntimeError("final response edit basis changed")
        self.require_content_pristine()
        replay = torch.matmul(self.code_edit, checked_basis.T)
        if not torch.equal(replay, self.physical_edit):
            raise RuntimeError("final response physical write differs from code edit")

    def require_content_pristine(self) -> None:
        """Reject post-mint mutation without accepting any caller-selected basis."""

        current_hashes = {
            "positions_sha256": runtime.tensor_identity_sha256(self.positions),
            "direction_indices_sha256": runtime.tensor_identity_sha256(
                self.direction_indices,
            ),
            "semantic_delta_sha256": runtime.tensor_identity_sha256(
                self.semantic_delta,
            ),
            "code_edit_sha256": runtime.tensor_identity_sha256(self.code_edit),
            "physical_edit_sha256": runtime.tensor_identity_sha256(self.physical_edit),
        }
        if any(getattr(self, name) != value for name, value in current_hashes.items()):
            raise RuntimeError("final response edit mutated after minting")


def build_final_response_edit(
    *, validated_program_bank: Mapping[str, Any], role_rows: torch.Tensor,
    ordered_batch_indices: Sequence[int], batch_ordinal: int,
    basis0: torch.Tensor, edit_sign: int,
) -> FinalResponseEdit:
    """Derive one baseline/positive/negative edit from frozen final authority."""

    if not isinstance(validated_program_bank, Mapping) or not isinstance(
        validated_program_bank.get("transport_geometry"),
        programs.TransportInterventionGeometry,
    ) or not isinstance(validated_program_bank.get("teacher_calibration"), Mapping):
        raise TypeError("response edit requires a validated canonical program bank")
    if edit_sign not in EDIT_SIGNS:
        raise ValueError("response edit sign must be -1, 0, or 1")
    if not torch.is_tensor(role_rows) or role_rows.dtype != torch.long or tuple(
        role_rows.shape
    ) != (runtime.BATCH_SIZE, FINAL_ROW_WIDTH) or role_rows.device.type != "cpu" or not (
        role_rows.is_contiguous()
    ):
        raise ValueError("response edit requires one contiguous CPU final batch")
    indices = tuple(ordered_batch_indices)
    expected_indices = tuple(range(
        batch_ordinal * runtime.BATCH_SIZE,
        (batch_ordinal + 1) * runtime.BATCH_SIZE,
    ))
    if indices != expected_indices:
        raise ValueError("response edit batch indices are not canonical")
    geometry = validated_program_bank["transport_geometry"]
    calibration = validated_program_bank["teacher_calibration"]
    payload_sha256 = _sha256(
        "canonical program payload", validated_program_bank.get("payload_sha256"),
    )
    expected_calibration = programs.select_teacher_calibration(
        calibration.get("teacher_median_kls", {}),
    )
    if dict(calibration) != dict(expected_calibration):
        raise RuntimeError("response edit calibration did not replay")
    selected_multiplier = float(calibration["selected_amplitude_multiplier"])
    amplitude = selected_multiplier * float(geometry.natural_rms)
    if not torch.isfinite(torch.tensor(amplitude)) or amplitude <= 0:
        raise RuntimeError("response edit amplitude is invalid")
    checked_basis = basis0.detach().cpu().float().contiguous()
    contract.validate_orthonormal_basis("response basis0", checked_basis)
    assignments = programs.intervention_assignments("final")
    row_index = torch.tensor(indices, dtype=torch.long)
    positions = assignments["positions"].index_select(0, row_index).contiguous()
    direction_indices = assignments["direction_indices"].index_select(
        0, row_index,
    ).contiguous()
    directions = geometry.normalized_directions.index_select(
        0, direction_indices,
    ).double().contiguous()
    semantic_delta = (
        float(edit_sign) * amplitude * directions
        if edit_sign else torch.zeros_like(directions)
    ).contiguous()
    code_edit = torch.zeros(
        runtime.BATCH_SIZE, runtime.SEQUENCE_LENGTH, runtime.CODE_DIM,
        dtype=torch.float32,
    )
    if edit_sign:
        code_edit[
            torch.arange(runtime.BATCH_SIZE), positions,
        ] = semantic_delta.float()
    physical_edit = torch.matmul(code_edit, checked_basis.T).contiguous()
    rows_sha256 = runtime.tensor_identity_sha256(role_rows)
    inputs_sha256 = runtime.tensor_identity_sha256(
        role_rows[:, :runtime.SEQUENCE_LENGTH].contiguous()
    )
    basis_sha256 = runtime.tensor_identity_sha256(checked_basis)
    calibration_identity = {
        key: calibration[key] for key in (
            "selected_amplitude_multiplier", "selected_teacher_median_kl",
            "geometric_center", "calibration_passed",
        )
    } | {
        "teacher_median_kls": {
            str(amplitude): float(calibration["teacher_median_kls"][amplitude])
            for amplitude in programs.INTERVENTION_AMPLITUDES
        },
    }
    calibration_sha256 = runtime.logical_identity_sha256(calibration_identity)
    unit_identity = runtime.logical_identity_sha256({
        "batch_ordinal": batch_ordinal,
        "ordered_batch_indices": list(indices),
        "ordered_role_rows_sha256": rows_sha256,
        "program_payload_sha256": payload_sha256,
        "geometry_sha256": geometry.sha256,
        "calibration_sha256": calibration_sha256,
        "basis0_sha256": basis_sha256,
        "positions_sha256": runtime.tensor_identity_sha256(positions),
        "direction_indices_sha256": runtime.tensor_identity_sha256(direction_indices),
        "support": "64:256",
    })
    result = FinalResponseEdit(
        _mint_token=_MINT_TOKEN,
        batch_ordinal=batch_ordinal, edit_sign=edit_sign,
        unit_identity_sha256=unit_identity,
        ordered_role_rows_sha256=rows_sha256,
        ordered_input_tokens_sha256=inputs_sha256,
        program_payload_sha256=payload_sha256,
        geometry_sha256=geometry.sha256,
        calibration_sha256=calibration_sha256,
        basis0_sha256=basis_sha256,
        positions_sha256=runtime.tensor_identity_sha256(positions),
        direction_indices_sha256=runtime.tensor_identity_sha256(direction_indices),
        semantic_delta_sha256=runtime.tensor_identity_sha256(semantic_delta),
        code_edit_sha256=runtime.tensor_identity_sha256(code_edit),
        physical_edit_sha256=runtime.tensor_identity_sha256(physical_edit),
        positions=positions, direction_indices=direction_indices,
        semantic_delta=semantic_delta,
        code_edit=code_edit, physical_edit=physical_edit,
    )
    result.require_pristine(
        role_rows=role_rows, ordered_batch_indices=indices, basis0=checked_basis,
    )
    return result


@dataclass(frozen=True, slots=True)
class ResponseExecutionIdentity:
    """Outer nonce binding one student trace to its actual finite edit."""

    action_key: str
    perturbation: str
    trial: int
    response_execution_amendment_sha256: str
    response_forward_plan_sha256: str
    final_action_identity_sha256: str
    materialization_sha256: str
    edit_sha256: str
    unit_identity_sha256: str
    code_edit_sha256: str
    physical_edit_sha256: str
    runtime_identity_sha256: str

    def __post_init__(self) -> None:
        if self.action_key not in response_plan.RESPONSE_ACTION_KEYS or self.perturbation not in (
            PERTURBATION_TRIAL
        ) or self.trial != PERTURBATION_TRIAL[self.perturbation]:
            raise ValueError("response execution action/perturbation identity changed")
        if self.response_execution_amendment_sha256 != (
            RESPONSE_EXECUTION_AMENDMENT_SHA256
        ):
            raise ValueError("response execution amendment identity changed")
        for name in (
            "response_forward_plan_sha256", "final_action_identity_sha256",
            "materialization_sha256", "edit_sha256", "unit_identity_sha256",
            "code_edit_sha256", "physical_edit_sha256", "runtime_identity_sha256",
        ):
            _sha256(name, getattr(self, name))

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            name: getattr(self, name) for name in self.__dataclass_fields__
        })


@dataclass(frozen=True, slots=True)
class ResponseProgramBatchBinding:
    """The response-specific outer identity and its broker runtime trace."""

    execution_identity: ResponseExecutionIdentity
    runtime_identity: runtime.TraceIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.execution_identity, ResponseExecutionIdentity) or not isinstance(
            self.runtime_identity, runtime.TraceIdentity
        ) or self.runtime_identity.sha256 != self.execution_identity.runtime_identity_sha256:
            raise ValueError("response program binding identity changed")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "execution_identity_sha256": self.execution_identity.sha256,
            "runtime_identity_sha256": self.runtime_identity.sha256,
        })


def bind_runtime_response_program_batch(
    *, materialized: final_actions.MaterializedFinalAction,
    final_action_identity: final_actions.FinalActionBatchIdentity,
    forward_plan: response_plan.ResponseForwardPlan,
    edit: FinalResponseEdit, role_rows: torch.Tensor,
    ordered_batch_indices: Sequence[int], teacher_mapping_sha256: str,
) -> ResponseProgramBatchBinding:
    """Mint distinct baseline/+/- student traces from sealed response authority."""

    if not isinstance(materialized, final_actions.MaterializedFinalAction) or not isinstance(
        final_action_identity, final_actions.FinalActionBatchIdentity
    ) or not isinstance(forward_plan, response_plan.ResponseForwardPlan) or not isinstance(
        edit, FinalResponseEdit
    ) or not runtime._sha256_text(teacher_mapping_sha256):
        raise TypeError("response runtime binding requires typed authorities")
    indices = tuple(ordered_batch_indices)
    final_action_identity.require_role_rows(
        materialized=materialized, role_rows=role_rows,
        ordered_batch_indices=indices,
    )
    edit.require_content_pristine()
    # Full row/basis replay is repeated by the observed adapter, which owns B0.
    if forward_plan.subject_key != final_action_identity.action_key or (
        materialized.plan.key != final_action_identity.action_key
    ) or forward_plan.batch_ordinal != final_action_identity.batch_ordinal or (
        forward_plan.ordered_role_rows_sha256
        != final_action_identity.ordered_role_rows_sha256
    ) or forward_plan.intervention_unit_sha256 != edit.unit_identity_sha256 or (
        forward_plan.edit_sign != edit.edit_sign
    ) or forward_plan.perturbation not in PERTURBATION_TRIAL or (
        forward_plan.action_plan_sha256 != materialized.plan.sha256
    ) or edit.program_payload_sha256 != final_action_identity.program_payload_sha256 or (
        edit.ordered_role_rows_sha256 != final_action_identity.ordered_role_rows_sha256
    ) or edit.ordered_input_tokens_sha256 != final_action_identity.ordered_input_tokens_sha256 or (
        edit.batch_ordinal != final_action_identity.batch_ordinal
    ):
        raise RuntimeError("response runtime authority bindings disagree")
    program = materialized.make_program()
    arm = materialized.plan.arm_plan
    if arm.identity_control is None or materialized.program_sha256 != (
        runtime.program_snapshot_sha256(program)
    ):
        raise RuntimeError("response materialization lost its physical program")
    trace = runtime.TraceIdentity.from_inputs(
        inputs=role_rows[:, :runtime.SEQUENCE_LENGTH].contiguous(),
        ordered_batch_indices=indices,
        source_commit=final_action_identity.source_commit,
        inherited_snapshot_sha256=final_action_identity.inherited_snapshot_sha256,
        rows_receipt_sha256=final_action_identity.rows_receipt_sha256,
        fit_role_tensor_sha256=final_action_identity.final_role_tensor_sha256,
        program_snapshot_sha256=materialized.program_sha256,
        teacher_mapping_sha256=teacher_mapping_sha256,
        role="early_mlp_suffix_transport_v1_final", phase="final",
        route=program.route, control=arm.identity_control,
        teacher_kind=("coordinate_labels" if program.route == "L" else "oon_logits"),
        trial=PERTURBATION_TRIAL[forward_plan.perturbation], epoch=0,
        optimizer_step=final_action_identity.batch_ordinal,
        batch_ordinal=final_action_identity.batch_ordinal,
        student_states=((0, "P"), (1, "P"), (2, "N")),
    )
    execution_identity = ResponseExecutionIdentity(
        action_key=final_action_identity.action_key,
        perturbation=forward_plan.perturbation,
        trial=PERTURBATION_TRIAL[forward_plan.perturbation],
        response_execution_amendment_sha256=(
            RESPONSE_EXECUTION_AMENDMENT_SHA256
        ),
        response_forward_plan_sha256=forward_plan.sha256,
        final_action_identity_sha256=final_action_identity.sha256,
        materialization_sha256=materialized.sha256,
        edit_sha256=edit.sha256,
        unit_identity_sha256=edit.unit_identity_sha256,
        code_edit_sha256=edit.code_edit_sha256,
        physical_edit_sha256=edit.physical_edit_sha256,
        runtime_identity_sha256=trace.sha256,
    )
    return ResponseProgramBatchBinding(
        execution_identity=execution_identity, runtime_identity=trace,
    )


@dataclass(frozen=True, slots=True)
class ObservedResponseForwardReceipt:
    """Tensor-free identity of one actually executed response forward."""

    forward_plan_sha256: str
    subject_key: str
    perturbation: str
    batch_ordinal: int
    execution_identity_sha256: str
    final_action_identity_sha256: str | None
    materialization_sha256: str | None
    edit_sha256: str
    semantic_delta_sha256: str
    code_edit_sha256: str
    physical_edit_sha256: str
    code1_sha256: str
    logits_sha256: str
    observed_closure_sha256: str
    student_step_ledger_sha256: str | None
    consumer_ledger_sha256: str | None
    broker_ledger_sha256: str | None

    def __post_init__(self) -> None:
        if self.subject_key != response_plan.TEACHER_KEY and self.subject_key not in (
            response_plan.RESPONSE_ACTION_KEYS
        ) or self.perturbation not in response_plan.PERTURBATIONS or type(
            self.batch_ordinal
        ) is not int or not 0 <= self.batch_ordinal < response_plan.EXPECTED_BATCHES:
            raise ValueError("observed response forward header is malformed")
        for name in (
            "forward_plan_sha256", "execution_identity_sha256", "edit_sha256",
            "semantic_delta_sha256", "code_edit_sha256", "physical_edit_sha256",
            "code1_sha256", "logits_sha256", "observed_closure_sha256",
        ):
            _sha256(name, getattr(self, name))
        optional = (
            self.final_action_identity_sha256, self.materialization_sha256,
            self.student_step_ledger_sha256, self.consumer_ledger_sha256,
            self.broker_ledger_sha256,
        )
        teacher = self.subject_key == response_plan.TEACHER_KEY
        if teacher != all(value is None for value in optional) or (
            not teacher and any(not runtime._sha256_text(value) for value in optional)
        ):
            raise ValueError("observed response teacher/student provenance changed")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            name: getattr(self, name) for name in self.__dataclass_fields__
        })


@dataclass(frozen=True, slots=True)
class ObservedResponseArmReduction:
    """One action's actual triplets and its typed row-scalar reductions."""

    action_key: str
    batch_plan_sha256: str
    teacher_forward_receipt_sha256s: tuple[str, str, str]
    student_forward_receipt_sha256s: tuple[str, str, str]
    code_response: response_reductions.BatchResponseReduction | None
    logit_response: response_reductions.BatchResponseReduction
    output_kl_response: response_reductions.BatchOutputKLReduction

    def __post_init__(self) -> None:
        if self.action_key not in response_plan.RESPONSE_ACTION_KEYS:
            raise ValueError("observed response reduction action is unregistered")
        _sha256("observed response batch plan", self.batch_plan_sha256)
        for values in (
            self.teacher_forward_receipt_sha256s,
            self.student_forward_receipt_sha256s,
        ):
            if not isinstance(values, tuple) or len(values) != 3 or any(
                not runtime._sha256_text(value) for value in values
            ):
                raise ValueError("observed response receipt triplet is malformed")
        code_expected = self.action_key in {"ll/N", "lt/N"}
        if (self.code_response is not None) != code_expected or (
            self.code_response is not None and not isinstance(
                self.code_response, response_reductions.BatchResponseReduction,
            )
        ) or not isinstance(
            self.logit_response, response_reductions.BatchResponseReduction,
        ) or not isinstance(
            self.output_kl_response, response_reductions.BatchOutputKLReduction,
        ):
            raise ValueError("observed response reduction modalities changed")
        identities = {
            self.logit_response.unit_identity,
            self.output_kl_response.unit_identity,
        }
        if self.code_response is not None:
            identities.add(self.code_response.unit_identity)
        if len(identities) != 1:
            raise ValueError("observed response reduction units differ")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            "action_key": self.action_key,
            "batch_plan_sha256": self.batch_plan_sha256,
            "teacher_forward_receipt_sha256s": list(
                self.teacher_forward_receipt_sha256s
            ),
            "student_forward_receipt_sha256s": list(
                self.student_forward_receipt_sha256s
            ),
            "code_response_sha256": (
                None if self.code_response is None else self.code_response.sha256
            ),
            "logit_response_sha256": self.logit_response.sha256,
            "output_kl_response_sha256": self.output_kl_response.sha256,
        })


@dataclass(frozen=True, slots=True)
class ObservedResponseBatchReceipt:
    """Atomic closure of one ordered 69-forward four-row response batch."""

    batch_plan_sha256: str
    source_bank_sha256: str
    program_payload_sha256: str
    final_context_sha256: str
    common_support_sha256: str
    basis0_sha256: str
    basis1_sha256: str
    forward_receipt_sha256s: tuple[str, ...]
    arm_reduction_sha256s: tuple[tuple[str, str], ...]
    broker_ledger_sha256: str
    teacher_forward_count: int
    student_forward_count: int
    atomic_complete: bool

    def __post_init__(self) -> None:
        for name in (
            "batch_plan_sha256", "source_bank_sha256", "program_payload_sha256",
            "final_context_sha256", "common_support_sha256", "basis0_sha256",
            "basis1_sha256", "broker_ledger_sha256",
        ):
            _sha256(name, getattr(self, name))
        if not isinstance(self.forward_receipt_sha256s, tuple) or len(
            self.forward_receipt_sha256s
        ) != 69 or any(
            not runtime._sha256_text(value) for value in self.forward_receipt_sha256s
        ) or not isinstance(self.arm_reduction_sha256s, tuple) or tuple(
            key for key, _value in self.arm_reduction_sha256s
        ) != response_plan.RESPONSE_ACTION_KEYS or any(
            not runtime._sha256_text(value) for _key, value in self.arm_reduction_sha256s
        ) or self.teacher_forward_count != 3 or self.student_forward_count != 66 or (
            self.atomic_complete is not True
        ):
            raise ValueError("observed response batch did not close atomically")

    @property
    def sha256(self) -> str:
        return runtime.logical_identity_sha256({
            name: (
                list(getattr(self, name)) if name in {
                    "forward_receipt_sha256s", "arm_reduction_sha256s",
                } else getattr(self, name)
            ) for name in self.__dataclass_fields__
        })


@dataclass(frozen=True, slots=True)
class ObservedResponseBatchResult:
    arm_reductions: tuple[ObservedResponseArmReduction, ...]
    receipt: ObservedResponseBatchReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.arm_reductions, tuple) or tuple(
            value.action_key for value in self.arm_reductions
        ) != response_plan.RESPONSE_ACTION_KEYS or any(
            not isinstance(value, ObservedResponseArmReduction)
            for value in self.arm_reductions
        ) or not isinstance(self.receipt, ObservedResponseBatchReceipt) or tuple(
            (value.action_key, value.sha256) for value in self.arm_reductions
        ) != self.receipt.arm_reduction_sha256s:
            raise ValueError("observed response batch result differs from its receipt")
