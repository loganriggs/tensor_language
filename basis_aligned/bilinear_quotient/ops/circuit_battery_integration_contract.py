"""Thin typed contract connecting a task battery to the circuit framework.

The task generator and model-facing intervention kernel remain separate.  This
module only freezes the boundary they must share with circuit_experiment_spec,
circuit_artifact_package, and circuit_managed_entry.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Sequence

import circuit_experiment_spec as framework


PHASES = ("FIT", "SELECT", "TEST")
TRANSFORMS = ("A1", "A2", "P", "C")


class BatteryContractError(ValueError):
    """A battery task would weaken split, evidence, or pricing guarantees."""


@dataclass(frozen=True)
class TransformSpec:
    transform_id: Literal["A1", "A2", "P", "C"]
    generator_role: str
    answer_changes: bool
    expected_effect: Literal["toward_donor", "invariant", "registered_active"]


@dataclass(frozen=True)
class JointArmSpec:
    arm_id: str
    member_arms: tuple[str, ...]


@dataclass(frozen=True)
class BatteryTaskSpec:
    task_id: str
    generator_role: str
    answer_role: str
    transforms: tuple[TransformSpec, ...]
    joint_arms: tuple[JointArmSpec, ...] = ()
    group_id_field: str = "group_id"
    row_id_field: str = "row_id"


@dataclass(frozen=True)
class PhasedArtifact:
    artifact: framework.ArtifactRef
    visibility: Literal["PROTOCOL", "FIT", "SELECT", "TEST"]


@dataclass(frozen=True)
class ExactPhasePrice:
    phase: Literal["FIT", "SELECT", "TEST"]
    forward_calls: int
    example_evaluations: int
    backward_calls: int
    model_updates: int
    evidence_bytes: int


@dataclass(frozen=True)
class PhaseReceipt:
    task_id: str
    phase: Literal["FIT", "SELECT", "TEST"]
    decision: Literal["pass", "fail", "invalid"]
    selection_sha256: str
    result_sha256: str
    price: ExactPhasePrice


def _sha(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 \
            or any(character not in "0123456789abcdef" for character in value):
        raise BatteryContractError(f"{label} must be a lowercase SHA-256")


def validate_task(task: BatteryTaskSpec) -> None:
    if not task.task_id or not task.generator_role or not task.answer_role:
        raise BatteryContractError("task roles must be explicit")
    if any(not isinstance(item.generator_role, str) or not item.generator_role
           or type(item.answer_changes) is not bool for item in task.transforms):
        raise BatteryContractError("transform generator roles and answer semantics must be typed")
    by_id = {item.transform_id: item for item in task.transforms}
    if tuple(item.transform_id for item in task.transforms) != TRANSFORMS or len(by_id) != 4:
        raise BatteryContractError("task must declare ordered A1/A2/P/C transforms")
    if len({item.generator_role for item in task.transforms}) != 4:
        raise BatteryContractError("A1/A2/P/C require independent generator roles")
    if not by_id["A1"].answer_changes or not by_id["A2"].answer_changes:
        raise BatteryContractError("A1 and A2 must independently change the answer")
    if by_id["P"].answer_changes:
        raise BatteryContractError("P must preserve the answer")
    expected = {"A1": "toward_donor", "A2": "toward_donor",
                "P": "invariant", "C": "registered_active"}
    if any(by_id[name].expected_effect != effect for name, effect in expected.items()):
        raise BatteryContractError("transform expected effects do not match the protocol")
    arm_ids = [arm.arm_id for arm in task.joint_arms]
    if len(arm_ids) != len(set(arm_ids)) or any(not arm_id for arm_id in arm_ids):
        raise BatteryContractError("joint arm IDs must be nonempty and unique")
    if any(len(arm.member_arms) < 2 or len(set(arm.member_arms)) != len(arm.member_arms)
           or arm.arm_id in arm.member_arms for arm in task.joint_arms):
        raise BatteryContractError("joint arms require at least two distinct singleton members")


def validate_rows(task: BatteryTaskSpec, rows: Sequence[Mapping[str, object]],
                  required_phases: Sequence[str] = PHASES) -> str:
    """Validate one A1/A2/P/C panel per split-disjoint generated group."""
    validate_task(task)
    if not rows:
        raise BatteryContractError("task rows are empty")
    panels: dict[tuple[str, str], list[str]] = {}
    group_splits: dict[str, set[str]] = {}
    row_ids: list[str] = []
    transforms = {item.transform_id: item for item in task.transforms}
    for row in rows:
        try:
            row_id = row[task.row_id_field]
            group_id = row[task.group_id_field]
            split = row["split"]
            transform_id = row["transform_id"]
            task_id = row["task_id"]
            answer_changes = row["answer_changes"]
        except KeyError as error:
            raise BatteryContractError(f"row lacks required field: {error.args[0]}") from error
        if not all(isinstance(value, str) and value for value in
                   (row_id, group_id, split, transform_id, task_id)):
            raise BatteryContractError("row identities must be nonempty strings")
        if task_id != task.task_id or split not in required_phases or transform_id not in transforms:
            raise BatteryContractError("row task, split, or transform is outside authority")
        if type(answer_changes) is not bool or answer_changes != transforms[transform_id].answer_changes:
            raise BatteryContractError("row answer-change semantics differ from its transform")
        row_ids.append(row_id)
        panels.setdefault((split, group_id), []).append(transform_id)
        group_splits.setdefault(group_id, set()).add(split)
    if len(row_ids) != len(set(row_ids)):
        raise BatteryContractError("row IDs are duplicated")
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise BatteryContractError("a generated group crosses split boundaries")
    if set(split for split, _ in panels) != set(required_phases):
        raise BatteryContractError("one or more required split panels are absent")
    if any(tuple(sorted(members)) != tuple(sorted(TRANSFORMS)) for members in panels.values()):
        raise BatteryContractError("every group must contain exactly one A1/A2/P/C row")
    return framework.canonical_sha256([dict(row) for row in rows])


def authorize_phase(task: BatteryTaskSpec, requested_phase: str,
                    prior_receipts: Sequence[PhaseReceipt]) -> None:
    """Require a receipt-complete prefix before a later split can be opened."""
    validate_task(task)
    if requested_phase not in PHASES:
        raise BatteryContractError("requested phase is invalid")
    expected = PHASES[:PHASES.index(requested_phase)]
    if tuple(receipt.phase for receipt in prior_receipts) != expected:
        raise BatteryContractError("phase receipts are not the exact required prefix")
    if any(receipt.task_id != task.task_id or receipt.decision != "pass"
           for receipt in prior_receipts):
        raise BatteryContractError("later split cannot open after fail, invalid, or another task")
    for receipt in prior_receipts:
        if receipt.price.phase != receipt.phase:
            raise BatteryContractError("receipt price is for another phase")
        _sha(receipt.selection_sha256, "selection fingerprint")
        _sha(receipt.result_sha256, "result fingerprint")
    if len({receipt.selection_sha256 for receipt in prior_receipts}) > 1:
        raise BatteryContractError("FIT selection changed after it was frozen")


def phase_artifacts(phase: str, artifacts: Sequence[PhasedArtifact]) \
        -> tuple[framework.ArtifactRef, ...]:
    """Return an invocation closure containing no future-split bytes."""
    if phase not in PHASES:
        raise BatteryContractError("phase is invalid")
    allowed = {"PROTOCOL", *PHASES[:PHASES.index(phase) + 1]}
    if any(item.visibility not in allowed for item in artifacts):
        raise BatteryContractError("managed invocation includes a future split artifact")
    if any(item.visibility == "PROTOCOL" and item.artifact.kind == "outcome" for item in artifacts):
        raise BatteryContractError("an outcome cannot be relabeled as protocol input")
    references = tuple(item.artifact for item in artifacts)
    framework.validate_spec(framework.CircuitExperimentSpec(
        experiment_id="artifact-closure-check", rung=0, artifacts=references,
        phases=(), authority_tables=(), calls=(),
    ))
    return references


def validate_price(compiled: Mapping[str, object], price: ExactPhasePrice) -> None:
    """Bind the literal call manifest to exact calls and example evaluations."""
    integers = (price.forward_calls, price.example_evaluations, price.backward_calls,
                price.model_updates, price.evidence_bytes)
    if any(type(value) is not int or value < 0 for value in integers) or price.model_updates != 0:
        raise BatteryContractError("price fields must be exact nonnegative integers and zero updates")
    calls = list(compiled.get("call_manifest", ()))
    if any(call.get("split") != price.phase for call in calls):
        raise BatteryContractError("price phase differs from the compiled manifest")
    if len(calls) != price.forward_calls:
        raise BatteryContractError("forward-call price differs from the compiled manifest")
    evaluations = sum(int(call["logical_batch_size"]) for call in calls)
    if evaluations != price.example_evaluations:
        raise BatteryContractError("example-evaluation price differs from the compiled manifest")


def validate_price_receipt(expected: ExactPhasePrice, observed: ExactPhasePrice) -> None:
    if observed != expected:
        raise BatteryContractError("measured phase price differs from its exact declaration")


def validate_joint_arm_evidence(task: BatteryTaskSpec, rows: Sequence[Mapping[str, object]],
                                compiled: Mapping[str, object], evidence: Sequence[Mapping[str, object]],
                                tolerance: float = 1e-12) -> None:
    """Require separately executed joint effects and save singleton interaction."""
    validate_task(task)
    calls = {str(call["call_id"]): call for call in compiled.get("call_manifest", ())}
    row_groups = {str(row[task.row_id_field]): str(row[task.group_id_field]) for row in rows}
    expected: set[tuple[str, str]] = set()
    arms = {arm.arm_id: arm for arm in task.joint_arms}
    for arm in task.joint_arms:
        for call in calls.values():
            if call.get("arm") == arm.arm_id:
                expected.update((arm.arm_id, str(row_id)) for row_id in call["row_ids"])
        if not any(call.get("arm") == arm.arm_id for call in calls.values()):
            raise BatteryContractError("declared joint arm is absent from the physical manifest")
    observed: set[tuple[str, str]] = set()
    for record in evidence:
        arm_id, row_id = str(record.get("joint_arm_id")), str(record.get("row_id"))
        key = (arm_id, row_id)
        if key in observed or key not in expected or arm_id not in arms:
            raise BatteryContractError("joint evidence membership is duplicated or unauthorized")
        observed.add(key)
        if record.get("group_id") != row_groups.get(row_id):
            raise BatteryContractError("joint evidence uses the wrong statistical group")
        joint_call = calls.get(str(record.get("joint_call_id")))
        member_calls = record.get("member_call_ids")
        singleton_effects = record.get("singleton_effects")
        if not isinstance(member_calls, Mapping) or not isinstance(singleton_effects, Mapping):
            raise BatteryContractError("joint evidence lacks typed singleton maps")
        members = arms[arm_id].member_arms
        if set(member_calls) != set(members) or set(singleton_effects) != set(members):
            raise BatteryContractError("joint evidence member arms changed")
        if joint_call is None or joint_call.get("arm") != arm_id or row_id not in joint_call["row_ids"]:
            raise BatteryContractError("joint effect lacks its own physical model call")
        for member in members:
            call = calls.get(str(member_calls[member]))
            if call is None or call.get("arm") != member or call["row_ids"] != joint_call["row_ids"]:
                raise BatteryContractError("singleton and joint calls do not share an exact batch")
        values = [record.get("joint_effect"), record.get("interaction"), *singleton_effects.values()]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise BatteryContractError("joint evidence effects must be finite numbers")
        interaction = float(record["joint_effect"]) - sum(float(singleton_effects[m]) for m in members)
        if not math.isclose(float(record["interaction"]), interaction, rel_tol=0.0, abs_tol=tolerance):
            raise BatteryContractError("joint interaction was not reconstructed from primitive effects")
    if observed != expected:
        raise BatteryContractError("joint-arm primitive evidence is incomplete")
