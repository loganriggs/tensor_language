"""Pure-CPU declarative compiler for the reusable FIT causal screen.

This module contains no model executor.  It composes the stable generic
``circuit_experiment_spec`` compiler with the battery integration contract,
then adds the small amount of physical information needed by one common
model-facing runner: paired row/token bindings, one-position donor-to-recipient
interventions, the fixed residual/module ceiling grid, and an optional
nine-head expansion selected by a preregistered FIT-only rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_experiment_spec as framework
import circuit_fast_screen_kernel as kernel


TRANSFORMS = battery.TRANSFORMS
RESIDUAL_BOUNDARIES = tuple(range(19))
MODULE_LAYERS = tuple(range(18))
HEAD_INDICES = tuple(range(9))
RESIDUAL_SITE_IDS = tuple(f"resid:{boundary:02d}" for boundary in RESIDUAL_BOUNDARIES)
MODULE_SITE_IDS = tuple(
    f"{kind}:{layer:02d}" for layer in MODULE_LAYERS for kind in ("attn", "mlp")
)
ATTENTION_SITE_IDS = tuple(f"attn:{layer:02d}" for layer in MODULE_LAYERS)
CEILING_SITE_IDS = RESIDUAL_SITE_IDS + MODULE_SITE_IDS

ANSWER_SCORE = "recipient_answer_minus_foil_logit"
INTERVENTION_DIRECTION = "donor_to_recipient"
INTERVENTION_SCOPE = "single_semantic_position"
INTERVENTION_VALUE = "exact_replace"
TIE_BREAK = (
    "descending_target_recovery_then_ascending_p_invariance_then_"
    "ascending_c_absolute_recovery_then_site_id"
)
TERMINAL_SCHEMA = {
    "screen": {
        "meaning": "native capability and at least one selective causal site pass",
        "selected_site_required": True,
    },
    "null": {
        "meaning": "valid native incapability or no selective causal site",
        "selected_site_required": False,
    },
    "invalid": {
        "meaning": "authority, instrument, evidence, or execution contract invalid",
        "selected_site_required": False,
    },
}
_ARRAY_CONTRACTS = (
    {
        "name": "token_a_logit",
        "dtype": "float32_le",
        "shape": ["logical_batch_size"],
        "finite_policy": "always",
    },
    {
        "name": "token_b_logit",
        "dtype": "float32_le",
        "shape": ["logical_batch_size"],
        "finite_policy": "always",
    },
)


class FastScreenSpecError(ValueError):
    """The reusable screen specification or authority is inconsistent."""


@dataclass(frozen=True)
class CandidateHypothesis:
    behavior: str
    answer_score: Literal["recipient_answer_minus_foil_logit"]
    information_read: str
    proposed_operation: str
    proposed_write: str
    candidate_sites: tuple[str, ...]
    alternative_explanation: str
    circuit_prediction: str
    opposing_null_prediction: str


@dataclass(frozen=True)
class SemanticPositionSpec:
    role: str
    recipient_field: str
    donor_field: str
    element_index: int | None = None


@dataclass(frozen=True)
class AuthorityFieldSpec:
    recipient_sequence_field: str = "base_ids"
    donor_sequence_field: str = "donor_ids"
    recipient_answer_field: str = "base_answer_id"
    recipient_foil_field: str = "base_foil_id"
    donor_answer_field: str = "donor_answer_id"
    donor_foil_field: str = "donor_foil_id"


ScreenBars = kernel.FixedBars


@dataclass(frozen=True)
class InterventionSpec:
    direction: Literal["donor_to_recipient"] = "donor_to_recipient"
    scope: Literal["single_semantic_position"] = "single_semantic_position"
    value: Literal["exact_replace"] = "exact_replace"
    recipient_sequence_role: Literal["base"] = "base"
    donor_sequence_role: Literal["donor"] = "donor"


@dataclass(frozen=True)
class HeadSelectionSpec:
    eligible_parent_site_ids: tuple[str, ...] = ATTENTION_SITE_IDS
    head_indices: tuple[int, ...] = HEAD_INDICES
    score_fields: tuple[str, ...] = (
        "a1_mean_recovery", "a2_mean_recovery",
        "a1_direction_fraction", "a2_direction_fraction",
        "p_invariance_effect", "c_absolute_recovery",
    )
    tie_break: str = TIE_BREAK


@dataclass(frozen=True)
class TerminalSpec:
    allowed: tuple[Literal["screen", "null", "invalid"], ...] = (
        "screen", "null", "invalid"
    )
    screen_reason: str = "selective_causal_site"
    null_reasons: tuple[str, ...] = (
        "native_behavior_incapable", "no_selective_causal_site"
    )
    invalid_reasons: tuple[str, ...] = (
        "authority_invalid", "instrument_invalid", "evidence_invalid",
        "execution_invalid",
    )


@dataclass(frozen=True)
class CircuitFastScreenSpec:
    experiment_id: str
    hypothesis: CandidateHypothesis
    task: battery.BatteryTaskSpec
    authority_sha256: str
    expected_fit_rows: int
    batch_size: int
    semantic_position: SemanticPositionSpec
    fields: AuthorityFieldSpec
    bars: kernel.FixedBars
    declared_max_price: battery.ExactPhasePrice
    intervention: InterventionSpec = InterventionSpec()
    head_selection: HeadSelectionSpec = HeadSelectionSpec()
    terminals: TerminalSpec = TerminalSpec()


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def spec_json(spec: CircuitFastScreenSpec) -> dict[str, object]:
    value = _json_value(asdict(spec))
    if not isinstance(value, dict):  # pragma: no cover - dataclass invariant
        raise FastScreenSpecError("screen spec is not a JSON object")
    framework.canonical_json_bytes(value)
    return value


def _require_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise FastScreenSpecError(f"{label} must be nonempty text")


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise FastScreenSpecError(f"{label} must be a lowercase SHA-256")


def validate_spec(spec: CircuitFastScreenSpec) -> None:
    """Validate the complete prospective screen contract without authority bytes."""
    spec_json(spec)
    for label, value in {
        "experiment_id": spec.experiment_id,
        "behavior": spec.hypothesis.behavior,
        "information_read": spec.hypothesis.information_read,
        "proposed_operation": spec.hypothesis.proposed_operation,
        "proposed_write": spec.hypothesis.proposed_write,
        "alternative_explanation": spec.hypothesis.alternative_explanation,
        "circuit_prediction": spec.hypothesis.circuit_prediction,
        "opposing_null_prediction": spec.hypothesis.opposing_null_prediction,
        "semantic role": spec.semantic_position.role,
        "recipient position field": spec.semantic_position.recipient_field,
        "donor position field": spec.semantic_position.donor_field,
    }.items():
        _require_text(value, label)
    if spec.hypothesis.behavior != spec.task.task_id:
        raise FastScreenSpecError("hypothesis behavior differs from the battery task")
    if spec.hypothesis.answer_score != ANSWER_SCORE:
        raise FastScreenSpecError("answer score is not the frozen paired-logit score")
    if spec.hypothesis.circuit_prediction == spec.hypothesis.opposing_null_prediction:
        raise FastScreenSpecError("opposing predictions must be distinct")
    if spec.hypothesis.candidate_sites != CEILING_SITE_IDS:
        raise FastScreenSpecError("candidate site grid must be the exact 19+36 ceiling grid")
    if spec.semantic_position.element_index is not None \
            and (type(spec.semantic_position.element_index) is not int
                 or spec.semantic_position.element_index < 0):
        raise FastScreenSpecError("semantic-position element index must be nonnegative")
    field_values = tuple(asdict(spec.fields).values())
    if len(field_values) != len(set(field_values)) or any(
        not isinstance(value, str) or not value for value in field_values
    ):
        raise FastScreenSpecError("authority field bindings must be nonempty and distinct")
    if type(spec.expected_fit_rows) is not int or spec.expected_fit_rows <= 0 \
            or spec.expected_fit_rows % len(TRANSFORMS):
        raise FastScreenSpecError("FIT row census must contain complete A1/A2/P/C panels")
    if type(spec.batch_size) is not int or spec.batch_size <= 0:
        raise FastScreenSpecError("batch size must be a positive integer")
    if spec.bars != kernel.FIXED_BARS:
        raise FastScreenSpecError("screen bars differ from circuit_fast_screen_kernel.FIXED_BARS")
    if spec.terminals != TerminalSpec():
        raise FastScreenSpecError("terminal schema must remain screen/null/invalid")
    if spec.intervention != InterventionSpec():
        raise FastScreenSpecError("intervention must be exact one-position donor-to-recipient replacement")
    if spec.head_selection != HeadSelectionSpec():
        raise FastScreenSpecError("head expansion and tie-break must remain the exact conditional nine-head rule")
    _validate_sha256(spec.authority_sha256, "authority digest")
    battery.validate_task(spec.task)
    # C is a registered active negative control.  Some tasks use an unrelated
    # answer-changing endpoint; others, such as agreement-attractor tests, keep
    # the answer fixed while changing a distractor.  The task declaration is
    # the authority and the producer scores the two cases differently.
    price = spec.declared_max_price
    if price.phase != "FIT" or price.backward_calls != 0 or price.model_updates != 0:
        raise FastScreenSpecError("maximum price must be FIT-only with zero backward/update calls")


def _token_sequence(row: Mapping[str, object], field: str, label: str) -> list[int]:
    value = row.get(field)
    if not isinstance(value, list) or not value or any(
        type(token) is not int or token < 0 for token in value
    ):
        raise FastScreenSpecError(f"{label} token sequence is invalid")
    return value


def _token_id(row: Mapping[str, object], field: str, label: str) -> int:
    value = row.get(field)
    if type(value) is not int or value < 0:
        raise FastScreenSpecError(f"{label} token ID is invalid")
    return value


def _position(row: Mapping[str, object], field: str, index: int | None,
              length: int, label: str) -> int:
    value = row.get(field)
    if index is None:
        position = value
    else:
        if not isinstance(value, list) or index >= len(value):
            raise FastScreenSpecError(f"{label} semantic-position vector is invalid")
        position = value[index]
    if type(position) is not int or not 0 <= position < length:
        raise FastScreenSpecError(f"{label} semantic position is outside its sequence")
    return position


def validate_fit_authority(
    spec: CircuitFastScreenSpec, rows: Sequence[Mapping[str, object]]
) -> dict[str, dict[str, object]]:
    """Validate linked FIT panels and exact paired token/position bindings."""
    validate_spec(spec)
    materialized = [dict(row) for row in rows]
    if len(materialized) != spec.expected_fit_rows:
        raise FastScreenSpecError("FIT authority row census changed")
    try:
        observed_sha256 = battery.validate_rows(spec.task, materialized, required_phases=("FIT",))
    except battery.BatteryContractError as error:
        raise FastScreenSpecError(str(error)) from error
    if observed_sha256 != spec.authority_sha256:
        raise FastScreenSpecError("FIT authority digest changed")
    by_id: dict[str, dict[str, object]] = {}
    for row in materialized:
        row_id = str(row[spec.task.row_id_field])
        recipient = _token_sequence(
            row, spec.fields.recipient_sequence_field, "recipient"
        )
        donor = _token_sequence(row, spec.fields.donor_sequence_field, "donor")
        base_answer = _token_id(
            row, spec.fields.recipient_answer_field, "recipient answer"
        )
        base_foil = _token_id(row, spec.fields.recipient_foil_field, "recipient foil")
        donor_answer = _token_id(row, spec.fields.donor_answer_field, "donor answer")
        donor_foil = _token_id(row, spec.fields.donor_foil_field, "donor foil")
        if base_answer == base_foil or donor_answer == donor_foil \
                or {base_answer, base_foil} != {donor_answer, donor_foil}:
            raise FastScreenSpecError("recipient/donor answer-plus-foil pair is not jointly aligned")
        answer_changes = row["answer_changes"]
        if answer_changes != (base_answer == donor_foil and base_foil == donor_answer):
            raise FastScreenSpecError("answer-change metadata differs from paired answer/foil IDs")
        recipient_position = _position(
            row, spec.semantic_position.recipient_field,
            spec.semantic_position.element_index, len(recipient), "recipient",
        )
        donor_position = _position(
            row, spec.semantic_position.donor_field,
            spec.semantic_position.element_index, len(donor), "donor",
        )
        enriched = dict(row)
        enriched["_recipient_position"] = recipient_position
        enriched["_donor_position"] = donor_position
        by_id[row_id] = enriched
    return by_id


def _framework_spec(spec: CircuitFastScreenSpec) -> framework.CircuitExperimentSpec:
    calls = []
    for side in ("base", "donor"):
        sequence_field = (
            spec.fields.recipient_sequence_field if side == "base"
            else spec.fields.donor_sequence_field
        )
        for transform in TRANSFORMS:
            name = f"native_{side}_{transform}"
            calls.append(framework.CallFamilySpec(
                name=name,
                split="FIT",
                arms=(name,),
                batch_size=spec.batch_size,
                call_kind="native_paired_logits",
                guard="fit_always",
                call_id_template=f"FIT:native:{side}:{transform}:{{batch}}",
                arm_specs=(framework.ArmSpec(name, "native", "undirected"),),
                sequence_field=sequence_field,
                row_id_field=spec.task.row_id_field,
                filters=(("transform_id", (transform,)),),
                shape_validation_mode=framework.RIGHT_PADDED_LENGTH,
            ))
    arrays = tuple(framework.ArraySpec(
        name=contract["name"], call_kinds=("native_paired_logits",),
        dtype=contract["dtype"], shape=("logical_batch_size",), retained=True,
    ) for contract in _ARRAY_CONTRACTS)
    return framework.CircuitExperimentSpec(
        experiment_id=spec.experiment_id,
        rung=0,
        artifacts=(),
        phases=(framework.PhaseSpec("FIT", forbidden_splits=("SELECT", "TEST", "OOD")),),
        authority_tables=(framework.AuthorityTableSpec(
            name="fit_rows",
            identity_fields=(spec.task.row_id_field,),
            expected_records_sha256=spec.authority_sha256,
            group_fields=(spec.task.group_id_field,),
            expected_counts={"FIT": spec.expected_fit_rows},
            expected_total=spec.expected_fit_rows,
        ),),
        calls=tuple(calls),
        arrays=arrays,
    )


def _metric_binding(spec: CircuitFastScreenSpec, row: Mapping[str, object]) -> dict[str, object]:
    return {
        "row_id": str(row[spec.task.row_id_field]),
        "token_a_id": int(row[spec.fields.recipient_answer_field]),
        "token_b_id": int(row[spec.fields.recipient_foil_field]),
        "recipient_answer_id": int(row[spec.fields.recipient_answer_field]),
        "donor_answer_id": int(row[spec.fields.donor_answer_field]),
        "answer_changes": bool(row["answer_changes"]),
    }


def _semantic_binding(spec: CircuitFastScreenSpec, row: Mapping[str, object]) -> dict[str, object]:
    return {
        "row_id": str(row[spec.task.row_id_field]),
        "semantic_role": spec.semantic_position.role,
        "recipient_position": int(row["_recipient_position"]),
        "donor_position": int(row["_donor_position"]),
    }


def _site_descriptor(site_id: str) -> dict[str, object]:
    kind, number = site_id.split(":")
    if kind == "resid":
        return {"site_id": site_id, "site_kind": "residual_state",
                "boundary": int(number)}
    return {"site_id": site_id, "site_kind": "module_output",
            "module_kind": kind, "layer": int(number)}


def _head_descriptor(parent: str, head: int) -> dict[str, object]:
    layer = int(parent.split(":")[1])
    return {
        "site_id": f"{parent}:head:{head:02d}",
        "site_kind": "attention_head_pre_projection",
        "parent_site_id": parent,
        "layer": layer,
        "head": head,
    }


def _patch_call(
    spec: CircuitFastScreenSpec,
    source_call: Mapping[str, object],
    rows: Mapping[str, Mapping[str, object]],
    site: Mapping[str, object],
    stage: str,
) -> dict[str, object]:
    row_ids = [str(row_id) for row_id in source_call["row_ids"]]
    transform = str(source_call["transform_id"])
    return {
        "call_id": (
            f"FIT:{stage}:{site['site_id']}:{transform}:{source_call['batch_index']}"
        ),
        "split": "FIT",
        "guard": "native_capability_pass",
        "call_kind": "single_position_donor_to_recipient_patch",
        "call_family": stage,
        "arm": str(site["site_id"]),
        "arm_role": "counterfactual",
        "arm_direction": "forward",
        "stage": stage,
        "transform_id": transform,
        "logical_batch_size": int(source_call["logical_batch_size"]),
        "padded_sequence_length": int(source_call["padded_sequence_length"]),
        "row_ids": row_ids,
        "site": dict(site),
        "intervention": {
            "direction": spec.intervention.direction,
            "scope": spec.intervention.scope,
            "value": spec.intervention.value,
            "recipient_sequence_role": spec.intervention.recipient_sequence_role,
            "donor_sequence_role": spec.intervention.donor_sequence_role,
            "recipient_sequence_field": spec.fields.recipient_sequence_field,
            "donor_sequence_field": spec.fields.donor_sequence_field,
        },
        "semantic_bindings": [_semantic_binding(spec, rows[row_id]) for row_id in row_ids],
        "metric_bindings": [_metric_binding(spec, rows[row_id]) for row_id in row_ids],
        "array_contracts": [dict(contract) for contract in _ARRAY_CONTRACTS],
        "shape_validation_mode": source_call["shape_validation_mode"],
        "checkpoint_validation": source_call["checkpoint_validation"],
        "model_structure_validation": source_call["model_structure_validation"],
    }


def _active_price(calls: Sequence[Mapping[str, object]]) -> battery.ExactPhasePrice:
    evaluations = sum(int(call["logical_batch_size"]) for call in calls)
    return battery.ExactPhasePrice(
        phase="FIT", forward_calls=len(calls), example_evaluations=evaluations,
        backward_calls=0, model_updates=0, evidence_bytes=8 * evaluations,
    )


def _attention_selection(
    spec: CircuitFastScreenSpec,
    scores: Mapping[str, Mapping[str, float]] | None,
) -> tuple[str, str | None, str | None]:
    if scores is None:
        return "pending", None, None
    if set(scores) != set(spec.head_selection.eligible_parent_site_ids):
        raise FastScreenSpecError("head selection requires scores for all 18 attention modules")
    normalized: dict[str, dict[str, float]] = {}
    required_fields = set(spec.head_selection.score_fields)
    for site_id, record in scores.items():
        if set(record) != required_fields:
            raise FastScreenSpecError("attention score record has unknown or missing fields")
        if any(type(record[field]) not in (int, float)
               or isinstance(record[field], bool)
               or not math.isfinite(float(record[field])) for field in required_fields):
            raise FastScreenSpecError("attention scores must be finite literal values")
        normalized[site_id] = {
            field: float(record[field]) for field in spec.head_selection.score_fields
        }
        if any(normalized[site_id][field] < 0.0 for field in (
            "a1_direction_fraction", "a2_direction_fraction",
            "p_invariance_effect", "c_absolute_recovery",
        )) or any(normalized[site_id][field] > 1.0 for field in (
            "a1_direction_fraction", "a2_direction_fraction",
        )):
            raise FastScreenSpecError("attention direction/control scores are out of range")
    eligible = [site_id for site_id in spec.head_selection.eligible_parent_site_ids if (
        normalized[site_id]["a1_mean_recovery"]
        >= spec.bars.minimum_target_family_recovery
        and normalized[site_id]["a2_mean_recovery"]
        >= spec.bars.minimum_target_family_recovery
        and normalized[site_id]["a1_direction_fraction"]
        >= spec.bars.minimum_target_direction_fraction
        and normalized[site_id]["a2_direction_fraction"]
        >= spec.bars.minimum_target_direction_fraction
        and normalized[site_id]["p_invariance_effect"]
        <= spec.bars.maximum_p_invariance_effect
        and normalized[site_id]["c_absolute_recovery"]
        <= spec.bars.maximum_c_absolute_recovery
    )]
    digest = framework.canonical_sha256(normalized)
    if not eligible:
        return "skipped_no_passing_attention_module", None, digest
    order = {site_id: index for index, site_id in enumerate(CEILING_SITE_IDS)}
    target_recovery = lambda site_id: (
        normalized[site_id]["a1_mean_recovery"]
        + normalized[site_id]["a2_mean_recovery"]
    ) / 2.0
    selected = min(
        eligible,
        key=lambda site_id: (
            -target_recovery(site_id),
            normalized[site_id]["p_invariance_effect"],
            normalized[site_id]["c_absolute_recovery"],
            order[site_id],
        ),
    )
    return "expanded", selected, digest


def _augment_native_calls(
    spec: CircuitFastScreenSpec,
    calls: Sequence[Mapping[str, object]],
    rows: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    counters: dict[tuple[str, str], int] = {}
    output = []
    for source in calls:
        call = dict(source)
        family = str(call["call_family"])
        _, side, transform = family.split("_", 2)
        key = (side, transform)
        batch_index = counters.get(key, 0)
        counters[key] = batch_index + 1
        call.update(
            stage="native", side=side, transform_id=transform,
            batch_index=batch_index,
            sequence_field=(
                spec.fields.recipient_sequence_field if side == "base"
                else spec.fields.donor_sequence_field
            ),
            metric_bindings=[_metric_binding(spec, rows[str(row_id)])
                             for row_id in call["row_ids"]],
        )
        output.append(call)
    return output


def compile_screen(
    spec: CircuitFastScreenSpec,
    rows: Sequence[Mapping[str, object]],
    *,
    attention_module_scores: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, object]:
    """Compile the exact active manifest and its prospective maximum price.

    With ``attention_module_scores=None`` the head stage remains unopened.  If
    scores are supplied, all 18 attention modules must be present.  Exactly one
    passing parent is selected by the frozen tie-break and expanded into nine
    head calls per recipient batch; if none passes, the head stage is skipped.
    """
    by_id = validate_fit_authority(spec, rows)
    materialized = [dict(row) for row in rows]
    framework_spec = _framework_spec(spec)
    base = framework.compile_experiment(
        framework_spec,
        authority_tables={"fit_rows": materialized},
        call_source_records=materialized,
    )
    native = _augment_native_calls(spec, base["call_manifest"], by_id)
    recipient_calls = [call for call in native if call["side"] == "base"]
    ceiling = [
        _patch_call(spec, source, by_id, _site_descriptor(site_id), "ceiling")
        for site_id in CEILING_SITE_IDS
        for source in recipient_calls
    ]
    status, parent, score_sha256 = _attention_selection(spec, attention_module_scores)
    heads = [] if parent is None else [
        _patch_call(spec, source, by_id, _head_descriptor(parent, head), "head")
        for head in spec.head_selection.head_indices
        for source in recipient_calls
    ]
    calls = native + ceiling + heads
    price = _active_price(calls)
    maximum_price = _active_price(native + ceiling + [
        _patch_call(spec, source, by_id, _head_descriptor("attn:00", head), "head")
        for head in spec.head_selection.head_indices
        for source in recipient_calls
    ])
    if maximum_price != spec.declared_max_price:
        raise FastScreenSpecError("declared maximum price differs from the compiled maximum")
    call_only = {"call_manifest": calls}
    try:
        battery.validate_price(call_only, price)
    except battery.BatteryContractError as error:  # pragma: no cover - internal invariant
        raise FastScreenSpecError(str(error)) from error
    compiled = {
        "schema": "circuit_fast_screen_compiled_v1",
        "experiment_id": spec.experiment_id,
        "screen_spec_sha256": framework.canonical_sha256(spec_json(spec)),
        "framework_spec_sha256": base["spec_sha256"],
        "authority": base["authority"],
        "hypothesis": spec_json(spec)["hypothesis"],
        "score_contract": {
            "answer_score": ANSWER_SCORE,
            "family_roles": {
                "A1": "answer_changing_target",
                "A2": "answer_changing_target",
                "P": "same_answer_invariance_control",
                "C": (
                    "answer_changing_unrelated_behavior_control"
                    if next(
                        item for item in spec.task.transforms
                        if item.transform_id == "C"
                    ).answer_changes
                    else "same_answer_active_negative_control"
                ),
            },
            "bars": _json_value(asdict(spec.bars)),
            "tie_break": TIE_BREAK,
        },
        "intervention_contract": {
            **_json_value(asdict(spec.intervention)),
            "authority_fields": _json_value(asdict(spec.fields)),
            "semantic_position": _json_value(asdict(spec.semantic_position)),
        },
        "terminal_schema": _json_value(TERMINAL_SCHEMA),
        "conditional_head_plan": {
            "status": status,
            "eligible_parent_site_ids": list(spec.head_selection.eligible_parent_site_ids),
            "head_indices": list(spec.head_selection.head_indices),
            "selection_rule": spec.head_selection.tie_break,
            "selected_parent_site_id": parent,
            "attention_scores_sha256": score_sha256,
        },
        "call_manifest": calls,
        "call_summary": framework.summarize_call_manifest(calls),
        "price": _json_value(asdict(price)),
        "max_price": _json_value(asdict(maximum_price)),
    }
    framework.canonical_json_bytes(compiled)
    compiled["compiled_sha256"] = framework.canonical_sha256(compiled)
    return compiled


def validate_compiled_screen(
    spec: CircuitFastScreenSpec,
    rows: Sequence[Mapping[str, object]],
    compiled: Mapping[str, object],
    *,
    attention_module_scores: Mapping[str, Mapping[str, float]] | None = None,
) -> None:
    expected = compile_screen(
        spec, rows, attention_module_scores=attention_module_scores
    )
    if framework.canonical_json_bytes(dict(compiled)) != framework.canonical_json_bytes(expected):
        raise FastScreenSpecError("compiled screen differs from exact deterministic recompilation")


def validate_terminal_record(record: Mapping[str, object]) -> None:
    """Validate the common scorer's small screen/null/invalid output surface."""
    if set(record) != {"terminal", "reason", "selected_site_id"}:
        raise FastScreenSpecError("terminal record has unknown or missing fields")
    terminal = record["terminal"]
    reason = record["reason"]
    selected = record["selected_site_id"]
    if terminal == "screen":
        valid_head_sites = {
            f"{parent}:head:{head:02d}"
            for parent in ATTENTION_SITE_IDS for head in HEAD_INDICES
        }
        if reason != TerminalSpec().screen_reason \
                or selected not in set(CEILING_SITE_IDS) | valid_head_sites:
            raise FastScreenSpecError("screen terminal requires one selected causal site")
    elif terminal == "null":
        if reason not in TerminalSpec().null_reasons or selected is not None:
            raise FastScreenSpecError("null terminal must use a declared reason and no site")
    elif terminal == "invalid":
        if reason not in TerminalSpec().invalid_reasons or selected is not None:
            raise FastScreenSpecError("invalid terminal must use a declared reason and no site")
    else:
        raise FastScreenSpecError("terminal must be screen, null, or invalid")
    framework.canonical_json_bytes(dict(record))


def compile_dryrun(
    spec: CircuitFastScreenSpec, rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Return a deterministic model-free receipt for the unopened head stage."""
    compiled = compile_screen(spec, rows)
    return {
        "schema": "circuit_fast_screen_dryrun_v1",
        "experiment_id": spec.experiment_id,
        "compiled_sha256": compiled["compiled_sha256"],
        "call_manifest_sha256": compiled["call_summary"]["manifest_sha256"],
        "authority_sha256": compiled["authority"]["fit_rows"]["records_sha256"],
        "active_price": compiled["price"],
        "max_price": compiled["max_price"],
        "head_stage": compiled["conditional_head_plan"]["status"],
        "model_loaded": False,
        "gpu_accessed": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "model_updates": 0,
        "queue_touched": False,
    }
