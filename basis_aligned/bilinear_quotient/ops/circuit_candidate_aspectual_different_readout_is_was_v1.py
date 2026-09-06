"""Sealed different-readout panel for has/had circuit transfer to is/was."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
import circuit_fast_screen_canonical_control_v2 as canonical


SCHEMA = builder.SCHEMA
SPLIT = builder.SPLIT
SEED = 20260917
GROUPS = 16
TASK_ID = "tense_auxiliary.is_vs_was"
READOUT = (" is", " was")
canonical_sha256 = builder.canonical_sha256
CandidateBankError = builder.CandidateBankError

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_linked_present_past_auxiliary_panels",
    answer_role="score_jointly_tokenized_is_versus_was",
    transforms=(
        battery.TransformSpec("A1", "today_yesterday_tense_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "bulletin_embedded_tense_swap", True, "toward_donor"),
        battery.TransformSpec("P", "same_tense_temporal_paraphrase", False, "invariant"),
        canonical.TRANSFORM,
    ),
)

_AGENTS = (
    "accountant", "chemist", "director", "engineer", "farmer", "gardener",
    "judge", "lawyer", "manager", "nurse", "officer", "plumber",
    "researcher", "teacher", "vendor", "writer",
)


def _answer(present: bool) -> str:
    return READOUT[0] if present else READOUT[1]


def _plain(agent: str, present: bool) -> str:
    return f"{'Today' if present else 'Yesterday'} the {agent}"


def _embedded(agent: str, present: bool) -> str:
    return f"The bulletin says that {'today' if present else 'yesterday'} the {agent}"


def _panel(group_number: int) -> list[dict[str, Any]]:
    agent = _AGENTS[group_number]
    alternate = _AGENTS[(group_number + 5) % GROUPS]
    forward = group_number % 2 == 0
    base_present, donor_present = (True, False) if forward else (False, True)
    direction = "present_to_past" if forward else "past_to_present"
    group_id = f"FIT:{canonical_sha256([SCHEMA, TASK_ID, 'different_readout_v1', SEED, group_number])[:24]}"
    sentence_types = ("present_progressive" if base_present else "past_progressive", "present_progressive" if donor_present else "past_progressive")
    common_no_vocab = dict(
        seed=SEED, task_id=TASK_ID, group_number=group_number, group_id=group_id,
        reporter=agent, alternate_reporter=alternate, adjective="different_readout_v1",
        object_name="time", spec=TASK_SPEC,
    )
    common = dict(common_no_vocab, vocabulary=READOUT)
    suffix = f" the {agent}"
    return [
        builder._row(
            **common, transform_id="A1", construction_id="today_yesterday_auxiliary",
            direction_id=direction, matched_suffix=suffix,
            base_text=_plain(agent, base_present), donor_text=_plain(agent, donor_present),
            base_answer=_answer(base_present), donor_answer=_answer(donor_present),
            sentence_types=sentence_types,
        ),
        builder._row(
            **common, transform_id="A2", construction_id="bulletin_embedded_today_yesterday_auxiliary",
            direction_id=direction, matched_suffix=suffix,
            base_text=_embedded(agent, base_present), donor_text=_embedded(agent, donor_present),
            base_answer=_answer(base_present), donor_answer=_answer(donor_present),
            sentence_types=sentence_types,
        ),
        builder._row(
            **common, transform_id="P", construction_id="today_now_same_auxiliary",
            direction_id="primary_to_alternative" if forward else "alternative_to_primary",
            matched_suffix=suffix,
            base_text=_plain(agent, base_present),
            donor_text=(f"Now the {agent}" if base_present else f"Earlier the {agent}"),
            base_answer=_answer(base_present), donor_answer=_answer(base_present),
            sentence_types=(sentence_types[0], sentence_types[0]),
        ),
        builder._row(**common_no_vocab, transform_id="C", **canonical.row_kwargs(group_number, forward)),
    ]


def _build() -> list[dict[str, Any]]:
    return [row for group_number in range(GROUPS) for row in _panel(group_number)]


def _validate(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build():
        raise CandidateBankError("rows differ from sealed different-readout authority")
    try:
        digest = battery.validate_rows(TASK_SPEC, materialized, required_phases=(SPLIT,))
    except battery.BatteryContractError as error:
        raise CandidateBankError(str(error)) from error
    if len(materialized) != GROUPS * 4 or len({str(row["row_id"]) for row in materialized}) != len(materialized):
        raise CandidateBankError("different-readout count or uniqueness changed")
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
