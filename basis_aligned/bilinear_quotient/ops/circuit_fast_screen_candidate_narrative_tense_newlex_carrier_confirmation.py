#!/usr/bin/env python3
"""Sixteen genuinely new lexical panels for narrative-tense carrier confirmation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_battery_task14 as task14
import circuit_fast_screen_candidate_narrative_tense as old
import circuit_fast_screen_candidate_narrative_tense_fresh_unchanged_carrier as prior
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder


canonical_sha256 = builder.canonical_sha256
CandidateBankError = builder.CandidateBankError
SCHEMA = builder.SCHEMA
SPLIT = "lexical_confirmation"
TASK_ID = "narrative_tense.past_vs_present_newlex_carrier_confirmation"
DEFAULT_GROUPS = 16
DEFAULT_SEED = 20260905
TENSE = (" was", " is")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_newlex_narrative_tense_carrier_confirmation_panels",
    answer_role="score_jointly_tokenized_past_versus_present_copula",
    transforms=(
        battery.TransformSpec("A1", "licensed_served_one_purpose_tense_swap", True,
                              "toward_donor"),
        battery.TransformSpec("A2", "new_report_frame_three_cue_tense_swap", True,
                              "toward_donor"),
        battery.TransformSpec("P", "same_tense_subject_rewrite", False, "invariant"),
        battery.TransformSpec("C", "past_subject_and_location_rewrite", False,
                              "registered_active"),
    ),
)

OLD_LEXEMES = set(old._SUBJECTS + old._ALTERNATES + old._PLACES + old._FOCUS) \
    | set(prior.FRESH_VOCABULARY)
ELIGIBLE_SINGULARS = tuple(pair[0] for pair in task14.NOUN_PAIRS if pair[0] not in OLD_LEXEMES)
if len(ELIGIBLE_SINGULARS) != 40:
    raise RuntimeError("new lexical pool changed")
USED_SINGULARS = ELIGIBLE_SINGULARS[:32]
SUBJECTS = USED_SINGULARS[:16]
ALTERNATES = USED_SINGULARS[16:]
TAIL_LOCATION_POOL = ELIGIBLE_SINGULARS[32:]
TAILS = tuple(TAIL_LOCATION_POOL[index % 8] for index in range(16))
LOCATIONS = tuple(TAIL_LOCATION_POOL[(index + 3) % 8] for index in range(16))
LEXICAL_TUPLES = tuple(zip(SUBJECTS, ALTERNATES, TAILS, LOCATIONS))
if len(set(LEXICAL_TUPLES)) != 16 or any(len(set(items)) != 4 for items in LEXICAL_TUPLES):
    raise RuntimeError("new lexical tuples are not uniquely deranged")


def _a1(subject: str, tail: str, past: bool) -> str:
    lead, verb = (("Yesterday", "served") if past else ("Today", "serves"))
    return f"{lead} the {subject} {verb} one purpose. The central purpose of the {tail}"


def _a2(subject: str, tail: str, past: bool) -> str:
    verb, when, attract = (("served", "yesterday", "attracted") if past
                           else ("serves", "today", "attracts"))
    return (f"The {subject} that {verb} one purpose {when} {attract} attention. "
            f"The central purpose of the {tail}")


def _p(subject: str, tail: str, past: bool) -> str:
    lead, verb = (("Yesterday", "served") if past else ("Today", "serves"))
    return f"{lead} the {subject} {verb} one purpose. The practical purpose of the {tail}"


def _control(subject: str, location: str, tail: str, past: bool) -> str:
    lead, verb = (("Yesterday", "worked") if past else ("Today", "works"))
    return (f"{lead} the {subject} {verb} near the {location}. "
            f"The central purpose of the {tail}")


def _answer(past: bool) -> str:
    return " was" if past else " is"


_IDENTITY_FIELDS = (
    "schema", "task_id", "split", "seed", "group_number", "group_id",
    "transform_id", "construction_id", "direction_id", "capability_cell_id",
    "reporter", "alternate_reporter", "adjective", "object_name", "base_text",
    "donor_text", "base_answer", "donor_answer", "base_foil", "donor_foil",
    "base_prediction_position", "donor_prediction_position",
)


def _row(**kwargs) -> dict[str, Any]:
    row = builder._row(**kwargs)
    row["split"] = SPLIT
    row["row_id"] = canonical_sha256({key: row[key] for key in _IDENTITY_FIELDS})
    return row


def _panel(seed: int, group: int) -> list[dict[str, Any]]:
    subject, alternate, tail, location = LEXICAL_TUPLES[group]
    forward = group % 2 == 0
    base_past, donor_past = ((True, False) if forward else (False, True))
    direction = "past_to_present" if forward else "present_to_past"
    group_id = f"LEXCONF:{canonical_sha256([SCHEMA, TASK_ID, seed, group])[:24]}"
    common = dict(seed=seed, task_id=TASK_ID, group_number=group, group_id=group_id,
                  reporter=subject, alternate_reporter=alternate, adjective="documented",
                  object_name=tail, vocabulary=TENSE, spec=TASK_SPEC)
    p_past = group % 4 < 2
    p_forward = group % 2 == 0
    p_base, p_donor = ((subject, alternate) if p_forward else (alternate, subject))
    c_past = group % 4 < 2
    c_forward = group % 2 == 0
    c_base_subject, c_donor_subject = ((subject, alternate) if c_forward
                                       else (alternate, subject))
    donor_location = LOCATIONS[(group + 1) % DEFAULT_GROUPS]
    if len({subject, alternate, tail, location, donor_location}) != 5:
        raise CandidateBankError("C lexical roles are not distinct within the row")
    return [
        _row(**common, transform_id="A1", construction_id="served_one_purpose_direct",
             direction_id=direction, matched_suffix=f"The central purpose of the {tail}",
             base_text=_a1(subject, tail, base_past), donor_text=_a1(subject, tail, donor_past),
             base_answer=_answer(base_past), donor_answer=_answer(donor_past),
             sentence_types=(("past" if base_past else "present"),
                             ("past" if donor_past else "present"))),
        _row(**common, transform_id="A2", construction_id="new_relative_three_cue",
             direction_id=direction, matched_suffix=f"The central purpose of the {tail}",
             base_text=_a2(subject, tail, base_past), donor_text=_a2(subject, tail, donor_past),
             base_answer=_answer(base_past), donor_answer=_answer(donor_past),
             sentence_types=(("past" if base_past else "present"),
                             ("past" if donor_past else "present"))),
        _row(**common, transform_id="P",
             construction_id=f"served_one_purpose_practical_{'past' if p_past else 'present'}",
             direction_id=(f"{'past' if p_past else 'present'}_"
                           f"{'primary_to_alternative' if p_forward else 'alternative_to_primary'}"),
             matched_suffix=f"The practical purpose of the {tail}",
             base_text=_p(p_base, tail, p_past), donor_text=_p(p_donor, tail, p_past),
             base_answer=_answer(p_past), donor_answer=_answer(p_past),
             sentence_types=(("past" if p_past else "present"),) * 2),
        _row(**common, transform_id="C", construction_id="same_tense_subject_location_control",
             direction_id=(f"{'past' if c_past else 'present'}_"
                           f"{'primary_to_alternative' if c_forward else 'alternative_to_primary'}"),
             matched_suffix=f"The central purpose of the {tail}",
             base_text=_control(c_base_subject, location, tail, c_past),
             donor_text=_control(c_donor_subject, donor_location, tail, c_past),
             base_answer=_answer(c_past), donor_answer=_answer(c_past),
             sentence_types=(("past" if c_past else "present"),) * 2),
    ]


def _build(groups: int, seed: int):
    return [row for group in range(groups) for row in _panel(seed, group)]


def _validate(rows, groups: int, seed: int) -> str:
    if groups != DEFAULT_GROUPS or seed != DEFAULT_SEED:
        raise CandidateBankError("this evidential authority is frozen at 16 groups and one seed")
    materialized = [dict(row) for row in rows]
    if materialized != _build(groups, seed):
        raise CandidateBankError("rows differ from the deterministic lexical authority")
    battery.validate_task(TASK_SPEC)
    if set(USED_SINGULARS + TAIL_LOCATION_POOL) & OLD_LEXEMES:
        raise CandidateBankError("new lexical pool overlaps a prior narrative authority")
    if set(USED_SINGULARS) & set(TAIL_LOCATION_POOL):
        raise CandidateBankError("subject and tail/location role vocabularies overlap")
    expected = {"A1": (0, 3), "A2": (3, 6, 7), "P": (2,), "C": (2, 6)}
    panels, cells = {}, {}
    endpoints = []
    for row in materialized:
        if row["split"] != SPLIT or row["task_id"] != TASK_ID:
            raise CandidateBankError("task or split label changed")
        base, donor = row["base_ids"], row["donor_ids"]
        changed = tuple(i for i, values in enumerate(zip(base, donor)) if values[0] != values[1])
        if len(base) != len(donor) or base[-1] != donor[-1] \
                or changed != expected[row["transform_id"]]:
            raise CandidateBankError("paired token alignment changed")
        if not all(row["construction_checks"].values()):
            raise CandidateBankError("a construction check failed")
        panels.setdefault(row["group_id"], []).append(row["transform_id"])
        endpoints.extend((row["base_text"], row["donor_text"]))
        cell = (row["transform_id"], row["direction_id"])
        cells[cell] = cells.get(cell, 0) + 1
    if any(tuple(sorted(panel)) != ("A1", "A2", "C", "P") for panel in panels.values()):
        raise CandidateBankError("a panel is incomplete")
    if any(cells.get((family, direction)) != 8 for family in ("A1", "A2")
           for direction in ("past_to_present", "present_to_past")):
        raise CandidateBankError("target directions are unbalanced")
    for family in ("P", "C"):
        for tense in ("past", "present"):
            for direction in ("primary_to_alternative", "alternative_to_primary"):
                if cells.get((family, f"{tense}_{direction}")) != 4:
                    raise CandidateBankError("P/C tense-by-direction cells are unbalanced")
    if len(set(row["row_id"] for row in materialized)) != len(materialized) \
            or len(set(endpoints)) != len(endpoints):
        raise CandidateBankError("new row IDs or prompt endpoints are duplicated")
    prior_rows = old.build_rows() + prior.build_rows()
    old_text = {row[side] for row in prior_rows for side in ("base_text", "donor_text")}
    old_tokens = {tuple(row[side]) for row in prior_rows for side in ("base_ids", "donor_ids")}
    if set(endpoints) & old_text or any(tuple(builder.ENCODING.encode(text)) in old_tokens
                                        for text in endpoints):
        raise CandidateBankError("a new prompt endpoint overlaps a prior narrative authority")
    return canonical_sha256(materialized)


def build_rows(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
               seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    rows = _build(groups, seed)
    _validate(rows, groups, seed)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID,
                  groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED) -> str:
    if task_id != TASK_ID:
        raise CandidateBankError("task ID changed")
    return _validate(rows, groups, seed)


def authority_sha256() -> str:
    return validate_rows(build_rows())


if __name__ == "__main__":
    print(authority_sha256())
