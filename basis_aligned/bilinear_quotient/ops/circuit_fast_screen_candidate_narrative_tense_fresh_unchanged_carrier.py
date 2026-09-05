#!/usr/bin/env python3
"""Fresh confirmation authority for the narrative-tense L11H3 carrier test."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_battery_integration_contract as battery
import circuit_fast_screen_candidate_narrative_tense as old
import circuit_fast_screen_candidate_pronoun as source
import circuit_fast_screen_candidate_sentence_terminal_context_control as builder


canonical_sha256 = builder.canonical_sha256
CandidateBankError = builder.CandidateBankError
SCHEMA = builder.SCHEMA
SPLIT = "fresh_confirmation"
TASK_ID = "narrative_tense.past_vs_present_fresh_unchanged_carrier"
DEFAULT_GROUPS = 32
DEFAULT_SEED = 20260905
TENSE = (" was", " is")

TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="generate_fresh_narrative_tense_unchanged_carrier_confirmation_panels",
    answer_role="score_jointly_tokenized_past_versus_present_copula",
    transforms=(
        battery.TransformSpec("A1", "fresh_direct_tense_swap", True, "toward_donor"),
        battery.TransformSpec("A2", "fresh_relative_tense_swap", True, "toward_donor"),
        battery.TransformSpec("P", "fresh_subject_rewrite_same_tense", False, "invariant"),
        battery.TransformSpec("C", "fresh_past_subject_and_location_rewrite", False,
                              "registered_active"),
    ),
)


def _flatten(values):
    for value in values:
        if isinstance(value, tuple):
            yield from _flatten(value)
        else:
            yield value


_OLD_WORDS = set(_flatten((old._SUBJECTS, old._ALTERNATES, old._PLACES, old._FOCUS)))
_SOURCE_WORDS = tuple(_flatten((source._OBJECTS, source._LOCATIONS)))
_fresh = []
for _word in _SOURCE_WORDS:
    if _word not in _OLD_WORDS and _word not in _fresh \
            and len(builder.ENCODING.encode(" " + _word)) == 1:
        _fresh.append(_word)
FRESH_VOCABULARY = tuple(_fresh)
if len(FRESH_VOCABULARY) != 82:
    raise RuntimeError("fresh vocabulary or GPT-2 tokenization changed")
SUBJECTS = FRESH_VOCABULARY[:32]
ALTERNATES = FRESH_VOCABULARY[32:64]
TAILS = FRESH_VOCABULARY[50:82]


def _direct(subject: str, tail: str, past: bool) -> str:
    lead, verb = ("Yesterday", "remained") if past else ("Today", "remains")
    return f"{lead} the {subject} {verb} indoors. The central purpose of the {tail}"


def _relative(subject: str, tail: str, past: bool) -> str:
    verb, when, draw = (("remained", "yesterday", "drew") if past
                        else ("remains", "today", "draws"))
    return (f"The {subject} that {verb} indoors {when} {draw} notice. "
            f"The central purpose of the {tail}")


def _control(subject: str, tail: str) -> str:
    return (f"In earlier years the {subject} occupied the {tail}. "
            "The original purpose of the shelter")


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
    """Reuse established token checks, then bind the fresh-confirmation identity."""
    row = builder._row(**kwargs)
    row["split"] = SPLIT
    row["row_id"] = canonical_sha256({key: row[key] for key in _IDENTITY_FIELDS})
    return row


def _panel(seed: int, group_number: int) -> list[dict[str, Any]]:
    subject, alternate, tail = (SUBJECTS[group_number], ALTERNATES[group_number],
                                TAILS[group_number])
    forward = group_number % 2 == 0
    base_past, donor_past = ((True, False) if forward else (False, True))
    direction = "past_to_present" if forward else "present_to_past"
    group_id = f"FRESH:{canonical_sha256([SCHEMA, TASK_ID, seed, group_number])[:24]}"
    common = dict(
        seed=seed, task_id=TASK_ID, group_number=group_number, group_id=group_id,
        reporter=subject, alternate_reporter=alternate, adjective="central",
        object_name=tail, vocabulary=TENSE, spec=TASK_SPEC,
    )
    direct_suffix = f"The central purpose of the {tail}"
    p_subject, p_alternate = alternate, ALTERNATES[(group_number + 1) % DEFAULT_GROUPS]
    p_base, p_donor = ((p_subject, p_alternate) if forward else (p_alternate, p_subject))
    c_tail, c_donor_tail = tail, TAILS[(group_number + 7) % DEFAULT_GROUPS]
    c_base_subject, c_donor_subject = ((subject, alternate) if forward
                                       else (alternate, subject))
    return [
        _row(**common, transform_id="A1", construction_id="fresh_direct_narration",
             direction_id=direction, matched_suffix=direct_suffix,
             base_text=_direct(subject, tail, base_past),
             donor_text=_direct(subject, tail, donor_past),
             base_answer=_answer(base_past), donor_answer=_answer(donor_past),
             sentence_types=(("past" if base_past else "present"),
                             ("past" if donor_past else "present"))),
        _row(**common, transform_id="A2", construction_id="fresh_relative_clause",
             direction_id=direction, matched_suffix=direct_suffix,
             base_text=_relative(subject, tail, base_past),
             donor_text=_relative(subject, tail, donor_past),
             base_answer=_answer(base_past), donor_answer=_answer(donor_past),
             sentence_types=(("past" if base_past else "present"),
                             ("past" if donor_past else "present"))),
        _row(**common, transform_id="P",
             construction_id=f"fresh_direct_{'past' if base_past else 'present'}",
             direction_id=("primary_to_alternative" if forward
                           else "alternative_to_primary"), matched_suffix=direct_suffix,
             base_text=_direct(p_base, tail, base_past),
             donor_text=_direct(p_donor, tail, base_past),
             base_answer=_answer(base_past), donor_answer=_answer(base_past),
             sentence_types=(("past" if base_past else "present"),) * 2),
        _row(**common, transform_id="C", construction_id="fresh_past_context_control",
             direction_id="base_to_donor" if forward else "donor_to_base",
             matched_suffix="The original purpose of the shelter",
             base_text=_control(c_base_subject, c_tail),
             donor_text=_control(c_donor_subject, c_donor_tail),
             base_answer=" was", donor_answer=" was", sentence_types=("past", "past")),
    ]


def _build(groups: int, seed: int) -> list[dict[str, Any]]:
    return [row for group_number in range(groups) for row in _panel(seed, group_number)]


def _validate(rows, groups: int, seed: int) -> str:
    if type(groups) is not int or not 2 <= groups <= DEFAULT_GROUPS or groups % 2:
        raise CandidateBankError("groups must be an even integer from 2 through 32")
    materialized = [dict(row) for row in rows]
    if materialized != _build(groups, seed):
        raise CandidateBankError("rows differ from the deterministic fresh authority")
    battery.validate_task(TASK_SPEC)
    if any(row["split"] != SPLIT or row["task_id"] != TASK_ID for row in materialized):
        raise CandidateBankError("row task or fresh-confirmation label changed")
    expected_diffs = {"A1": (0, 3), "A2": (3, 5, 6), "P": (2,), "C": (4, 7)}
    if set(FRESH_VOCABULARY) & _OLD_WORDS:
        raise CandidateBankError("fresh and original narrative vocabularies overlap")
    tuples = {(SUBJECTS[i], ALTERNATES[i], TAILS[i]) for i in range(groups)}
    if len(tuples) != groups or any(len(set(item)) != 3 for item in tuples):
        raise CandidateBankError("fresh lexical tuples are not unique within and across groups")
    if any(len(builder.ENCODING.encode(" " + word)) != 1 for word in FRESH_VOCABULARY):
        raise CandidateBankError("a fresh content word is not one leading-space token")
    if any(len(builder.ENCODING.encode(answer)) != 1 for answer in TENSE):
        raise CandidateBankError("a tense answer is not one token")
    old_rows = old.build_rows()
    old_texts = {str(row[side]) for row in old_rows for side in ("base_text", "donor_text")}
    old_tokens = {tuple(row[side]) for row in old_rows for side in ("base_ids", "donor_ids")}
    cells = {}
    for row in materialized:
        if not all(row["construction_checks"].values()):
            raise CandidateBankError("a stored construction check is false")
        if len(row["base_ids"]) != len(row["donor_ids"]):
            raise CandidateBankError("paired prompt lengths differ")
        if row["base_text"] in old_texts or row["donor_text"] in old_texts \
                or tuple(row["base_ids"]) in old_tokens or tuple(row["donor_ids"]) in old_tokens:
            raise CandidateBankError("a fresh endpoint duplicates the original authority")
        changed = tuple(i for i, pair in enumerate(zip(row["base_ids"], row["donor_ids"]))
                        if pair[0] != pair[1])
        if changed != expected_diffs[row["transform_id"]]:
            raise CandidateBankError(f"source token roles changed for {row['transform_id']}")
        key = (row["transform_id"], row["direction_id"])
        cells[key] = cells.get(key, 0) + 1
    half = groups // 2
    for family in ("A1", "A2"):
        for direction in ("past_to_present", "present_to_past"):
            if cells.get((family, direction)) != half:
                raise CandidateBankError("target directions are unbalanced")
    panels = {}
    for row in materialized:
        panels.setdefault(row["group_id"], []).append(row["transform_id"])
    if any(tuple(sorted(panel)) != ("A1", "A2", "C", "P") for panel in panels.values()):
        raise CandidateBankError("every fresh group must contain one A1/A2/P/C panel")
    row_ids = [str(row["row_id"]) for row in materialized]
    endpoints = [(str(row[side + "_text"]), tuple(row[side + "_ids"]))
                 for row in materialized for side in ("base", "donor")]
    if len(row_ids) != len(set(row_ids)) or len(endpoints) != len(set(endpoints)):
        raise CandidateBankError("fresh row IDs or endpoints are duplicated")
    return canonical_sha256(materialized)


def build_rows(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
               seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    if task_id != TASK_ID:
        raise CandidateBankError("task_id differs from the frozen authority")
    rows = _build(groups, seed)
    _validate(rows, groups, seed)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID,
                  groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED) -> str:
    if task_id != TASK_ID:
        raise CandidateBankError("task_id differs from the frozen authority")
    return _validate(rows, groups, seed)


def authority_sha256() -> str:
    return validate_rows(build_rows())


if __name__ == "__main__":
    print(authority_sha256())
