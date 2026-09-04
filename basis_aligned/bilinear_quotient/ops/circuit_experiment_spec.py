"""Pure-CPU declarative contracts for high-quality circuit experiments.

This module compiles authority tables and literal model-call schedules.  It
deliberately does not know how to construct a circuit intervention or score a
scientific gate; those remain pinned experiment-specific functions.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Literal, Mapping, Sequence

import result_contract


JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class SpecError(ValueError):
    """The declarative experiment contract is inconsistent."""


def canonical_json_bytes(value: object) -> bytes:
    result_contract.validate_standard_json(value)
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def spec_json(spec: "CircuitExperimentSpec") -> dict[str, object]:
    """Return the frozen dataclass contract as literal JSON, not Python tuples."""
    def convert(value: object) -> object:
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, Mapping):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    converted = convert(asdict(spec))
    if not isinstance(converted, dict):  # pragma: no cover - dataclass invariant
        raise SpecError("compiled spec is not a JSON object")
    return converted


@dataclass(frozen=True)
class ArtifactRef:
    role: str
    path: str
    sha256: str
    kind: Literal["source", "prereg", "authority", "outcome"]
    executable: bool = False
    dryrun_access: bool = False


@dataclass(frozen=True)
class AuthorityTableSpec:
    name: str
    identity_fields: tuple[str, ...]
    split_field: str | None = "split"
    group_fields: tuple[str, ...] = ()
    expected_counts: Mapping[str, int] = field(default_factory=dict)
    expected_total: int | None = None


@dataclass(frozen=True)
class CallFamilySpec:
    name: str
    split: str
    arms: tuple[str, ...]
    batch_size: int
    call_kind: str
    guard: str
    call_id_template: str
    sequence_field: str = "ids"
    row_id_field: str = "row_id"
    filters: tuple[tuple[str, tuple[object, ...]], ...] = ()
    arm_call_kinds: tuple[tuple[str, str], ...] = ()
    arm_batch_limits: tuple[tuple[str, int], ...] = ()
    axis_order: Literal["batch_arm", "arm_batch"] = "arm_batch"
    sort_policy: Literal["canonical_json", "legacy_python_repr"] = "canonical_json"
    batch_limit: int | None = None
    shape_validation_mode: str = "dynamic_batched_token_matrix_exact_common_length_v1"
    checkpoint_validation: str = "facade_verified_sha256"
    model_structure_validation: str = "facade_bilin18_structure"


@dataclass(frozen=True)
class ArraySpec:
    name: str
    call_kinds: tuple[str, ...]
    dtype: str
    shape: tuple[str | int, ...]
    retained: bool
    finite_policy: Literal["always", "final_nonfinite_diagnostic"] = "always"


@dataclass(frozen=True)
class PredicateSpec:
    predicate_id: str
    phase: str
    priority: int
    evaluator_role: str
    required_arrays: tuple[str, ...]
    disposition: Literal["diagnostic", "hard_abort"]


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    opens_after: str | None = None
    forbidden_splits: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScienceProjectionSpec:
    projector_role: str
    decision_role: str
    allowed_terminals: tuple[str, ...]
    output_types: Mapping[str, str]


@dataclass(frozen=True)
class CircuitExperimentSpec:
    experiment_id: str
    rung: int
    artifacts: tuple[ArtifactRef, ...]
    phases: tuple[PhaseSpec, ...]
    authority_tables: tuple[AuthorityTableSpec, ...]
    calls: tuple[CallFamilySpec, ...]
    arrays: tuple[ArraySpec, ...] = ()
    predicates: tuple[PredicateSpec, ...] = ()
    science: ScienceProjectionSpec | None = None


def validate_spec(spec: CircuitExperimentSpec) -> None:
    result_contract.validate_standard_json(spec_json(spec))
    if not spec.experiment_id or spec.rung < 0:
        raise SpecError("experiment identity is invalid")
    for label, values in {
        "artifact roles": [item.role for item in spec.artifacts],
        "artifact paths": [item.path for item in spec.artifacts],
        "phase names": [item.name for item in spec.phases],
        "authority table names": [item.name for item in spec.authority_tables],
        "call family names": [item.name for item in spec.calls],
        "array names": [item.name for item in spec.arrays],
        "predicate IDs": [item.predicate_id for item in spec.predicates],
    }.items():
        if len(values) != len(set(values)) or any(not value for value in values):
            raise SpecError(f"{label} must be nonempty and unique")
    phases = {phase.name for phase in spec.phases}
    if any(call.split not in phases for call in spec.calls):
        raise SpecError("call family names an undeclared phase")
    if any(predicate.phase not in phases for predicate in spec.predicates):
        raise SpecError("predicate names an undeclared phase")
    priorities = [predicate.priority for predicate in spec.predicates]
    if len(priorities) != len(set(priorities)):
        raise SpecError("predicate priorities must be unique")
    kind_order = {"instrument": 0, "authority": 1, "evidence": 2, "science": 3}
    typed_predicates = [item for item in spec.predicates if hasattr(item, "kind")]
    if any(getattr(item, "kind") not in kind_order for item in typed_predicates):
        raise SpecError("predicate kind must be typed")
    ordered_kinds = [kind_order[getattr(item, "kind")] for item in sorted(
        typed_predicates, key=lambda item: item.priority
    )]
    if ordered_kinds != sorted(ordered_kinds):
        raise SpecError("predicate priority must put instrument checks before science")
    if any(item.predicate_id.startswith("pred_") for item in spec.predicates):
        raise SpecError("pred_ names are reserved for registered science outputs")
    arrays = {array.name: array for array in spec.arrays}
    for predicate in spec.predicates:
        missing = set(predicate.required_arrays) - set(arrays)
        if missing:
            raise SpecError(f"predicate {predicate.predicate_id} lacks arrays: {sorted(missing)}")
        if predicate.disposition == "diagnostic" and any(
            not arrays[name].retained for name in predicate.required_arrays
        ):
            raise SpecError("a diagnostic predicate depends on unretained evidence")
    for artifact in spec.artifacts:
        if len(artifact.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in artifact.sha256):
            raise SpecError(f"artifact {artifact.role} has invalid SHA-256")
        if artifact.kind == "outcome" and artifact.dryrun_access:
            raise SpecError("model-free dry run may not access an outcome artifact")
    for call in spec.calls:
        if not hasattr(call, "arm_specs"):
            continue  # legacy shadow specifications remain byte-compatible
        arm_specs = tuple(getattr(call, "arm_specs"))
        if tuple(getattr(item, "name", None) for item in arm_specs) != call.arms:
            raise SpecError("typed arm names differ from call-family arms")
        if len(call.arms) != len(set(call.arms)):
            raise SpecError("typed arm names must be unique")
        if any(getattr(item, "role", None) not in {
            "native", "counterfactual", "control", "null"
        } for item in arm_specs):
            raise SpecError("every typed arm requires a valid role")
        if any(getattr(item, "direction", None) not in {
            "undirected", "forward", "reverse"
        } for item in arm_specs):
            raise SpecError("every typed arm requires a valid direction")
        for item in arm_specs:
            if item.role == "counterfactual" and not any(
                peer.role == "native" and peer.direction == item.direction
                for peer in arm_specs
            ):
                raise SpecError("counterfactual arm requires a direction-matched native role")


def _identity(record: Mapping[str, object], fields: tuple[str, ...]) -> object:
    if not fields or any(field not in record for field in fields):
        raise SpecError(f"authority identity fields are missing: {fields}")
    values = [record[field] for field in fields]
    return values[0] if len(values) == 1 else values


def compile_authority_tables(
    table_specs: Sequence[AuthorityTableSpec],
    tables: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    """Compile exact ordered IDs, counts and hashes from pure authority tables."""
    output: dict[str, object] = {}
    for table_spec in table_specs:
        if table_spec.name not in tables:
            raise SpecError(f"missing authority table: {table_spec.name}")
        records = [dict(record) for record in tables[table_spec.name]]
        result_contract.validate_standard_json(records)
        identities = [_identity(record, table_spec.identity_fields) for record in records]
        identity_keys = [canonical_json_bytes(value) for value in identities]
        if len(identity_keys) != len(set(identity_keys)):
            raise SpecError(f"authority table {table_spec.name} has duplicate identities")
        for index, record in enumerate(records):
            for group_field in table_spec.group_fields:
                if not isinstance(record.get(group_field), str) or not record[group_field]:
                    raise SpecError(
                        f"authority table {table_spec.name} row {index} has invalid {group_field}"
                    )
        counts: dict[str, int] = {}
        if table_spec.split_field is not None:
            split_values = [record.get(table_spec.split_field) for record in records]
            if any(not isinstance(value, str) or not value for value in split_values):
                raise SpecError(f"authority table {table_spec.name} has invalid split")
            counts = dict(sorted(Counter(split_values).items()))
        if table_spec.expected_total is not None and len(records) != table_spec.expected_total:
            raise SpecError(f"authority table {table_spec.name} total changed")
        if table_spec.expected_counts and counts != dict(table_spec.expected_counts):
            raise SpecError(f"authority table {table_spec.name} split counts changed: {counts}")
        expected_digest = getattr(table_spec, "expected_records_sha256", None)
        if expected_digest is not None and canonical_sha256(records) != expected_digest:
            raise SpecError(f"authority table {table_spec.name} split/content digest changed")
        output[table_spec.name] = {
            "count": len(records),
            "counts_by_split": counts,
            "ordered_identities": identities,
            "ordered_identities_sha256": canonical_sha256(identities),
            "records_sha256": canonical_sha256(records),
        }
    extra = set(tables) - {spec.name for spec in table_specs}
    if extra:
        raise SpecError(f"undeclared authority tables: {sorted(extra)}")
    return output


def _selected_records(
    records: Sequence[Mapping[str, object]], family: CallFamilySpec
) -> list[dict[str, object]]:
    selected = [dict(record) for record in records if record.get("split") == family.split]
    for field_name, allowed in family.filters:
        selected = [record for record in selected if record.get(field_name) in allowed]
    if not selected:
        raise SpecError(f"call family {family.name} has no source records")
    ids = [record.get(family.row_id_field) for record in selected]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise SpecError(f"call family {family.name} has duplicate or invalid source IDs")
    return selected


def _batches(
    records: Sequence[Mapping[str, object]], family: CallFamilySpec
) -> list[list[dict[str, object]]]:
    if family.batch_size <= 0 or not family.arms:
        raise SpecError(f"call family {family.name} has invalid batching or arms")
    for record in records:
        sequence = record.get(family.sequence_field)
        if not isinstance(sequence, list) or not sequence:
            raise SpecError(f"call family {family.name} has invalid token sequence")
    if family.sort_policy == "legacy_python_repr":
        key = lambda record: (len(record[family.sequence_field]), str(record))
    else:
        key = lambda record: (
            len(record[family.sequence_field]), canonical_json_bytes(record)
        )
    ordered = sorted(records, key=key)
    batches: list[list[dict[str, object]]] = []
    cursor = 0
    while cursor < len(ordered):
        length = len(ordered[cursor][family.sequence_field])
        end = cursor
        while (
            end < len(ordered)
            and len(ordered[end][family.sequence_field]) == length
            and end - cursor < family.batch_size
        ):
            end += 1
        batches.append(ordered[cursor:end])
        cursor = end
    if family.batch_limit is not None:
        if family.batch_limit <= 0:
            raise SpecError("batch limit must be positive")
        batches = batches[: family.batch_limit]
    return batches


def _call_record(
    family: CallFamilySpec, arm: str, batch_index: int,
    batch: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    lengths = {len(record[family.sequence_field]) for record in batch}
    if len(lengths) != 1:
        raise SpecError("compiled call mixes sequence lengths")
    call_kinds = dict(family.arm_call_kinds)
    record = {
        "call_id": family.call_id_template.format(
            split=family.split, family=family.name, arm=arm, batch=batch_index
        ),
        "split": family.split,
        "guard": family.guard,
        "call_kind": call_kinds.get(arm, family.call_kind),
        "arm": arm,
        "logical_batch_size": len(batch),
        "padded_sequence_length": next(iter(lengths)),
        "row_ids": [str(record[family.row_id_field]) for record in batch],
        "shape_validation_mode": family.shape_validation_mode,
        "checkpoint_validation": family.checkpoint_validation,
        "model_structure_validation": family.model_structure_validation,
    }
    if hasattr(family, "arm_specs"):
        typed = {item.name: item for item in getattr(family, "arm_specs")}[arm]
        record.update(arm_role=typed.role, arm_direction=typed.direction)
    return record


def compile_call_manifest(
    records: Sequence[Mapping[str, object]],
    families: Sequence[CallFamilySpec],
) -> list[dict[str, object]]:
    """Compile a deterministic ordered call manifest from declared families."""
    calls: list[dict[str, object]] = []
    for family in families:
        selected = _selected_records(records, family)
        batches = _batches(selected, family)
        for label, pairs in {
            "call-kind": family.arm_call_kinds,
            "batch-limit": family.arm_batch_limits,
        }.items():
            keys = [key for key, _ in pairs]
            if len(keys) != len(set(keys)):
                raise SpecError(f"call family {family.name} has duplicate {label} overrides")
        if any(not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
               for _, limit in family.arm_batch_limits):
            raise SpecError(f"call family {family.name} has invalid arm batch limit")
        unknown_overrides = (
            set(dict(family.arm_call_kinds)) | set(dict(family.arm_batch_limits))
        ) - set(family.arms)
        if unknown_overrides:
            raise SpecError(f"call family {family.name} overrides unknown arms")
        limits = dict(family.arm_batch_limits)
        pairs = (
            (batch_index, batch, arm)
            for batch_index, batch in enumerate(batches)
            for arm in family.arms
            if batch_index < limits.get(arm, len(batches))
        ) if family.axis_order == "batch_arm" else (
            (batch_index, batch, arm)
            for arm in family.arms
            for batch_index, batch in enumerate(batches)
            if batch_index < limits.get(arm, len(batches))
        )
        calls.extend(
            _call_record(family, arm, batch_index, batch)
            for batch_index, batch, arm in pairs
        )
    call_ids = [record["call_id"] for record in calls]
    if len(call_ids) != len(set(call_ids)):
        raise SpecError("compiled call IDs are not unique")
    result_contract.validate_standard_json(calls)
    return calls


def summarize_call_manifest(calls: Sequence[Mapping[str, object]]) -> dict[str, object]:
    shapes = Counter(
        f"{call['logical_batch_size']}x{call['padded_sequence_length']}" for call in calls
    )
    return {
        "call_count": len(calls),
        "call_kind_counts": dict(sorted(Counter(call["call_kind"] for call in calls).items())),
        "guard_counts": dict(sorted(Counter(call["guard"] for call in calls).items())),
        "shape_counts": dict(sorted(shapes.items())),
        "maximum_batch_size": max(int(call["logical_batch_size"]) for call in calls),
        "manifest_sha256": canonical_sha256(list(calls)),
    }


def compile_experiment(
    spec: CircuitExperimentSpec,
    *,
    authority_tables: Mapping[str, Sequence[Mapping[str, object]]],
    call_source_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    validate_spec(spec)
    authority = compile_authority_tables(spec.authority_tables, authority_tables)
    calls = compile_call_manifest(call_source_records, spec.calls)
    compiled = {
        "schema": "circuit_experiment_compiled_contract_v1",
        "experiment_id": spec.experiment_id,
        "rung": spec.rung,
        "spec_sha256": canonical_sha256(spec_json(spec)),
        "authority": authority,
        "call_manifest": calls,
        "call_summary": summarize_call_manifest(calls),
        "predicate_order": [
            predicate.predicate_id
            for predicate in sorted(spec.predicates, key=lambda item: item.priority)
        ],
    }
    result_contract.validate_standard_json(compiled)
    return compiled
