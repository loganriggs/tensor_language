"""Outcome-sealed fresh constructions for the aspectual-anchor circuit."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_aspectual as parent
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_canonical_control_v2 as canonical


TASK_ID = parent.TASK_ID
TASK_SPEC = parent.TASK_SPEC
SCHEMA = builder.SCHEMA
SPLIT = builder.SPLIT
SEED = 20260906
GROUPS = 16
ASPECT = parent.ASPECT
CandidateBankError = builder.CandidateBankError
canonical_sha256 = builder.canonical_sha256

_AGENTS = (
    "analyst", "baker", "captain", "dentist", "editor", "farmer", "gardener",
    "historian", "inspector", "jeweler", "keeper", "librarian", "mechanic",
    "nurse", "operator", "painter",
)
_PERIODS = (
    "decade", "semester", "campaign", "expedition", "assignment", "project",
    "investigation", "renovation", "construction", "deployment", "migration",
    "recovery", "transition", "experiment", "tournament", "conference",
)
if len(_AGENTS) != GROUPS or len(_PERIODS) != GROUPS:
    raise RuntimeError("fresh lexical tables changed size")


def _answer(present: bool) -> str:
    return ASPECT[0] if present else ASPECT[1]


def _ledger(agent: str, period: str, present: bool) -> str:
    cue = "since" if present else "by"
    return f"In the ledger, {cue} last {period}, the {agent}"


def _memory(agent: str, period: str, present: bool) -> str:
    cue = "since" if present else "by"
    return f"The witness remembers that {cue} last {period} the {agent}"


def _panel(group_number: int) -> list[dict[str, Any]]:
    agent = _AGENTS[group_number]
    period = _PERIODS[group_number]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'fresh_construction_v1', SEED, group_number])[:24]}"
    sentence_types = (
        "present_perfect" if base_present else "past_perfect",
        "present_perfect" if donor_present else "past_perfect",
    )
    common_no_vocab = dict(
        seed=SEED,
        task_id=TASK_ID,
        group_number=group_number,
        group_id=group_id,
        reporter=agent,
        alternate_reporter=_AGENTS[(group_number + 5) % GROUPS],
        adjective="fresh",
        object_name=period,
        spec=TASK_SPEC,
    )
    common = dict(common_no_vocab, vocabulary=ASPECT)
    suffix = f" the {agent}"
    return [
        builder._row(
            **common,
            transform_id="A1",
            construction_id="ledger_parenthetical_anchor",
            direction_id=direction,
            matched_suffix=suffix,
            base_text=_ledger(agent, period, base_present),
            donor_text=_ledger(agent, period, donor_present),
            base_answer=_answer(base_present),
            donor_answer=_answer(donor_present),
            sentence_types=sentence_types,
        ),
        builder._row(
            **common,
            transform_id="A2",
            construction_id="memory_embedded_anchor",
            direction_id=direction,
            matched_suffix=suffix,
            base_text=_memory(agent, period, base_present),
            donor_text=_memory(agent, period, donor_present),
            base_answer=_answer(base_present),
            donor_answer=_answer(donor_present),
            sentence_types=sentence_types,
        ),
        builder._row(
            **common,
            transform_id="P",
            construction_id=(
                "ledger_parenthetical_present_perfect"
                if base_present else "ledger_parenthetical_past_perfect"
            ),
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=suffix,
            base_text=_ledger(agent, period, base_present),
            donor_text=_ledger(
                agent, _PERIODS[(group_number + 7) % GROUPS], base_present
            ),
            base_answer=_answer(base_present),
            donor_answer=_answer(base_present),
            sentence_types=(sentence_types[0], sentence_types[0]),
        ),
        builder._row(
            **common_no_vocab,
            transform_id="C",
            **canonical.row_kwargs(group_number, forward),
        ),
    ]


def _build() -> list[dict[str, Any]]:
    return [row for group_number in range(GROUPS) for row in _panel(group_number)]


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from the sealed fresh construction authority")
    try:
        digest = battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    if len(materialized) != GROUPS * 4:
        raise CandidateBankError("fresh row count changed")
    if len({str(row["row_id"]) for row in materialized}) != len(materialized):
        raise CandidateBankError("fresh row IDs are not unique")
    for row in materialized:
        if not all(row["construction_checks"].values()):
            raise CandidateBankError("a construction check is false")
    return digest


def build_rows(task_id: str = TASK_ID) -> list[dict[str, Any]]:
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    rows = _build()
    _validate(rows)
    return rows


def validate_rows(
    rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID
) -> str:
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
