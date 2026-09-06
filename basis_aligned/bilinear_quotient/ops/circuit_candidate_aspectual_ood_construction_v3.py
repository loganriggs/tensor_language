"""Outcome-sealed syntax-and-lexicon OOD panel for the donor-free aspectual actuator."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_aspectual as parent
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder


TASK_ID = parent.TASK_ID
TASK_SPEC = parent.TASK_SPEC
SCHEMA = builder.SCHEMA
SPLIT = builder.SPLIT
SEED = 20260913
GROUPS = 16
ASPECT = parent.ASPECT
CONTROL = (" morning", " evening")
CandidateBankError = builder.CandidateBankError
canonical_sha256 = builder.canonical_sha256

_AGENTS = (
    "accountant", "chemist", "director", "engineer", "farmer", "gardener",
    "judge", "lawyer", "manager", "nurse", "officer", "plumber",
    "researcher", "teacher", "vendor", "writer",
)
_PERIODS = (
    "quarter", "season", "voyage", "term", "contract", "trial", "survey",
    "repair", "initiative", "operation", "review", "residency", "workshop",
    "festival", "series", "summit",
)


def _answer(present: bool) -> str:
    return ASPECT[0] if present else ASPECT[1]


def _temporal(agent: str, period: str, present: bool) -> str:
    if present:
        return f"Over the past {period} the {agent}"
    return f"Before the last {period} ended the {agent}"


def _embedded(agent: str, period: str, present: bool) -> str:
    return f"The archive notes that {_temporal(agent, period, present).lower()}"


def _control(agent: str, alternate: str, forward: bool) -> tuple[str, str]:
    primary = f"After sunrise the {agent} began the shift early in the"
    secondary = f"At dawn the {alternate} started the work early in the"
    return (primary, secondary) if forward else (secondary, primary)


def _panel(group_number: int) -> list[dict[str, Any]]:
    agent, period = _AGENTS[group_number], _PERIODS[group_number]
    alternate = _AGENTS[(group_number + 5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'ood_construction_v3', SEED, group_number])[:24]}"
    sentence_types = ("present_perfect" if base_present else "past_perfect", "present_perfect" if donor_present else "past_perfect")
    common_no_vocab = dict(
        seed=SEED, task_id=TASK_ID, group_number=group_number, group_id=group_id,
        reporter=agent, alternate_reporter=alternate, adjective="ood_v3",
        object_name=period, spec=TASK_SPEC,
    )
    common = dict(common_no_vocab, vocabulary=ASPECT)
    suffix = f" the {agent}"
    control_base, control_donor = _control(agent, alternate, forward)
    return [
        builder._row(
            **common, transform_id="A1", construction_id="past_span_vs_completed_span_ood",
            direction_id=direction, matched_suffix=suffix,
            base_text=_temporal(agent, period, base_present), donor_text=_temporal(agent, period, donor_present),
            base_answer=_answer(base_present), donor_answer=_answer(donor_present), sentence_types=sentence_types,
        ),
        builder._row(
            **common, transform_id="A2", construction_id="archive_embedded_span_ood",
            direction_id=direction, matched_suffix=suffix,
            base_text=_embedded(agent, period, base_present), donor_text=_embedded(agent, period, donor_present),
            base_answer=_answer(base_present), donor_answer=_answer(donor_present), sentence_types=sentence_types,
        ),
        builder._row(
            **common, transform_id="P", construction_id="past_span_noun_rewrite_ood",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=suffix,
            base_text=_temporal(agent, period, base_present),
            donor_text=_temporal(agent, _PERIODS[(group_number + 7) % GROUPS], base_present),
            base_answer=_answer(base_present), donor_answer=_answer(base_present),
            sentence_types=(sentence_types[0], sentence_types[0]),
        ),
        builder._row(
            **common_no_vocab, vocabulary=CONTROL, transform_id="C",
            construction_id="sunrise_morning_same_answer_ood",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix="early in the", base_text=control_base, donor_text=control_donor,
            base_answer=CONTROL[0], donor_answer=CONTROL[0],
            sentence_types=("morning_completion_control", "morning_completion_control"),
        ),
    ]


def _build() -> list[dict[str, Any]]:
    return [row for group_number in range(GROUPS) for row in _panel(group_number)]


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed OOD-construction-v3 authority")
    try:
        digest = battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    if len(materialized) != GROUPS * 4 or len({str(row["row_id"]) for row in materialized}) != len(materialized):
        raise CandidateBankError("OOD-construction-v3 count or uniqueness changed")
    if any(not all(row["construction_checks"].values()) for row in materialized):
        raise CandidateBankError("a construction check is false")
    return digest


def build_rows(task_id: str = TASK_ID) -> list[dict[str, Any]]:
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    rows = _build()
    _validate(rows)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID) -> str:
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    return _validate(rows)


def authority_sha256(task_id: str = TASK_ID) -> str:
    return validate_rows(build_rows(task_id), task_id=task_id)


if __name__ == "__main__":
    rows = build_rows()
    print("authority:", authority_sha256())
    for family in ("A1", "A2", "P", "C"):
        row = next(item for item in rows if item["family"] == family)
        print(f"{family}: {row['base_text']!r} -> {row['base_answer']!r}")
        print(f"    {row['donor_text']!r} -> {row['donor_answer']!r}")
