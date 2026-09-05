#!/usr/bin/env python3
"""Frozen native-only A2/C selector authority for narrative tense.

This module defines text and token identities only.  It deliberately contains
no model loader, causal intervention, result writer, queue operation, or claim.
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import tiktoken

import circuit_fast_screen_candidate_narrative_tense as original_narrative
import circuit_fast_screen_candidate_narrative_tense_fresh_unchanged_carrier as fresh_v1
import circuit_fast_screen_candidate_narrative_tense_newlex_carrier_confirmation as newlex_v1


SCHEMA = "narrative_tense_a2_c_native_capability_authority_v1"
TASK_ID = "narrative_tense.past_vs_present"
CAPABILITY_ID = "narrative_tense.a2_c_native_capability_select_holdout_v1"
GROUPS = 16
FIT_GROUPS = tuple(range(8))
HOLDOUT_GROUPS = tuple(range(8, 16))
ENCODING = tiktoken.get_encoding("gpt2")
ANSWERS = (" was", " is")

# These 64 nouns are absent from the original narrative authority, the first
# fresh authority, and the failed 16-group new-lexicon authority.  Each is one
# GPT-2 token when preceded by a space.
SUBJECT_PAIRS = (
    ("boy", "girl"), ("uncle", "aunt"), ("son", "daughter"),
    ("waiter", "boss"), ("chief", "prince"), ("poet", "monk"),
    ("priest", "spy"), ("cop", "dean"), ("scout", "fan"),
    ("host", "patron"), ("master", "lord"), ("lady", "hero"),
    ("victim", "client"), ("cook", "dancer"), ("keeper", "tutor"),
    ("listener", "king"),
)
TAILS = (
    "desk", "bench", "roof", "wall", "floor", "wheel", "bell", "drum",
    "flag", "coin", "bowl", "jar", "pipe", "rope", "chain", "seed",
)
LOCATIONS = (
    "yard", "cave", "pond", "trail", "path", "beach", "city", "shop",
    "home", "barn", "tent", "bay", "shore", "woods", "coast", "pier",
)

A2_TEMPLATE_ORDER = ("record_coordination", "while_observers", "reported_frame")
C_TEMPLATE_ORDER = ("explicit_period", "years_nowadays", "back_then_right_now")


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _answer(past: bool) -> str:
    return " was" if past else " is"


def _a1(subject: str, tail: str, past: bool) -> str:
    lead, verb = (("Yesterday", "served") if past else ("Today", "serves"))
    return f"{lead} the {subject} {verb} one purpose. The central purpose of the {tail}"


def _p(subject: str, tail: str, past: bool) -> str:
    lead, verb = (("Yesterday", "served") if past else ("Today", "serves"))
    return f"{lead} the {subject} {verb} one purpose. The practical purpose of the {tail}"


def _a2(template_id: str, subject: str, tail: str, past: bool) -> str:
    if template_id == "record_coordination":
        verb, when, report = (("served", "yesterday", "made") if past
                              else ("serves", "today", "makes"))
        return (f"The {subject} {verb} one purpose {when} and {report} a record. "
                f"The central purpose of the {tail}")
    if template_id == "while_observers":
        verb, when, observe = (("served", "yesterday", "took") if past
                               else ("serves", "today", "take"))
        return (f"While the {subject} {verb} one purpose {when}, observers {observe} notes. "
                f"The central purpose of the {tail}")
    if template_id == "reported_frame":
        report, verb, when = (("said", "served", "yesterday") if past
                              else ("say", "serves", "today"))
        return (f"Reports {report} the {subject} {verb} one purpose {when}. "
                f"The central purpose of the {tail}")
    raise ValueError(f"unknown A2 template: {template_id}")


def _c(template_id: str, subject: str, location: str, tail: str, past: bool) -> str:
    if template_id == "explicit_period":
        prefix, verb = (("In the past", "worked") if past else ("At present", "works"))
    elif template_id == "years_nowadays":
        prefix, verb = (("Years ago", "worked") if past else ("Nowadays", "works"))
    elif template_id == "back_then_right_now":
        prefix, verb = (("Back then", "worked") if past else ("Right now", "works"))
    else:
        raise ValueError(f"unknown C template: {template_id}")
    return (f"{prefix}, the {subject} {verb} near the {location}. "
            f"The central purpose of the {tail}")


def _joint(prompt: str, answer: str) -> tuple[tuple[int, ...], int]:
    prompt_ids = tuple(ENCODING.encode(prompt))
    complete = tuple(ENCODING.encode(prompt + answer))
    if complete[:len(prompt_ids)] != prompt_ids or len(complete) != len(prompt_ids) + 1:
        raise ValueError("answer is not one stable continuation token")
    if len(ENCODING.encode(answer)) != 1:
        raise ValueError("answer vocabulary changed tokenization")
    return prompt_ids, complete[-1]


def _row(*, group: int, family: str, template_id: str, direction: str,
         base_text: str, donor_text: str, base_answer: str, donor_answer: str) -> dict:
    base_ids, base_answer_id = _joint(base_text, base_answer)
    donor_ids, donor_answer_id = _joint(donor_text, donor_answer)
    base_foil = ANSWERS[1] if base_answer == ANSWERS[0] else ANSWERS[0]
    donor_foil = ANSWERS[1] if donor_answer == ANSWERS[0] else ANSWERS[0]
    _, base_foil_id = _joint(base_text, base_foil)
    _, donor_foil_id = _joint(donor_text, donor_foil)
    identity = [SCHEMA, group, family, template_id, direction, base_text, donor_text,
                base_answer, donor_answer]
    return {
        "schema": SCHEMA, "task_id": TASK_ID, "capability_id": CAPABILITY_ID,
        "phase": "FIT" if group in FIT_GROUPS else "HOLDOUT",
        "group_number": group, "row_id": canonical_sha256(identity),
        "family": family, "template_id": template_id, "direction_id": direction,
        "base_text": base_text, "donor_text": donor_text,
        "base_ids": list(base_ids), "donor_ids": list(donor_ids),
        "base_answer": base_answer, "donor_answer": donor_answer,
        "base_foil": base_foil, "donor_foil": donor_foil,
        "base_answer_id": base_answer_id, "donor_answer_id": donor_answer_id,
        "base_foil_id": base_foil_id, "donor_foil_id": donor_foil_id,
    }


def _panel(group: int) -> list[dict]:
    primary, alternate = SUBJECT_PAIRS[group]
    tail = TAILS[group]
    location = LOCATIONS[group]
    donor_location = LOCATIONS[(group + 5) % GROUPS]
    target_forward = group % 2 == 0
    base_past, donor_past = ((True, False) if target_forward else (False, True))
    target_direction = "past_to_present" if target_forward else "present_to_past"
    control_past = group % 4 < 2
    control_forward = group % 2 == 0
    control_direction = (f"{'past' if control_past else 'present'}_"
                         f"{'primary_to_alternative' if control_forward else 'alternative_to_primary'}")
    base_subject, donor_subject = ((primary, alternate) if control_forward
                                   else (alternate, primary))
    rows = [
        _row(group=group, family="A1", template_id="served_one_purpose",
             direction=target_direction,
             base_text=_a1(primary, tail, base_past),
             donor_text=_a1(primary, tail, donor_past),
             base_answer=_answer(base_past), donor_answer=_answer(donor_past)),
        _row(group=group, family="P", template_id="served_one_purpose_practical",
             direction=control_direction,
             base_text=_p(base_subject, tail, control_past),
             donor_text=_p(donor_subject, tail, control_past),
             base_answer=_answer(control_past), donor_answer=_answer(control_past)),
    ]
    rows.extend(
        _row(group=group, family="A2", template_id=template_id,
             direction=target_direction,
             base_text=_a2(template_id, primary, tail, base_past),
             donor_text=_a2(template_id, primary, tail, donor_past),
             base_answer=_answer(base_past), donor_answer=_answer(donor_past))
        for template_id in A2_TEMPLATE_ORDER
    )
    rows.extend(
        _row(group=group, family="C", template_id=template_id,
             direction=control_direction,
             base_text=_c(template_id, base_subject, location, tail, control_past),
             donor_text=_c(template_id, donor_subject, donor_location, tail, control_past),
             base_answer=_answer(control_past), donor_answer=_answer(control_past))
        for template_id in C_TEMPLATE_ORDER
    )
    return rows


def build_rows() -> list[dict]:
    rows = [row for group in range(GROUPS) for row in _panel(group)]
    validate_rows(rows)
    return rows


def build_package(a2_template_id: str, c_template_id: str) -> list[dict]:
    """Return one of the nine predeclared complete A1/A2/P/C packages."""
    if a2_template_id not in A2_TEMPLATE_ORDER or c_template_id not in C_TEMPLATE_ORDER:
        raise ValueError("package template lies outside the frozen candidate bank")
    return [row for row in build_rows()
            if row["family"] in {"A1", "P"}
            or (row["family"] == "A2" and row["template_id"] == a2_template_id)
            or (row["family"] == "C" and row["template_id"] == c_template_id)]


def package_sha256(a2_template_id: str, c_template_id: str) -> str:
    package = build_package(a2_template_id, c_template_id)
    if len(package) != 64:
        raise ValueError("selected package must contain 64 paired rows")
    return canonical_sha256(package)


def _changed(row: Mapping[str, object]) -> tuple[int, ...]:
    return tuple(index for index, pair in enumerate(zip(row["base_ids"], row["donor_ids"]))
                 if pair[0] != pair[1])


def validate_rows(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    expected = [row for group in range(GROUPS) for row in _panel(group)]
    if materialized != expected or len(materialized) != 128:
        raise ValueError("authority rows differ from the frozen 128-row design")
    vocabulary = tuple(word for pair in SUBJECT_PAIRS for word in pair) + TAILS + LOCATIONS
    if len(vocabulary) != len(set(vocabulary)) or any(
            len(ENCODING.encode(" " + word)) != 1 for word in vocabulary):
        raise ValueError("authority vocabulary is not unique one-token material")
    prior_words = set(
        original_narrative._SUBJECTS + original_narrative._ALTERNATES
        + original_narrative._PLACES + original_narrative._FOCUS
    ) | set(fresh_v1.FRESH_VOCABULARY) | set(newlex_v1.ELIGIBLE_SINGULARS)
    if set(vocabulary) & prior_words:
        raise ValueError("authority content vocabulary overlaps a prior narrative authority")
    row_ids, endpoints = set(), set()
    counts: dict[tuple[str, str, str, str], int] = {}
    for row in materialized:
        if row["row_id"] in row_ids:
            raise ValueError("duplicate row ID")
        row_ids.add(row["row_id"])
        for side in ("base", "donor"):
            endpoint = tuple(row[f"{side}_ids"])
            if endpoint in endpoints:
                raise ValueError("duplicate prompt endpoint")
            endpoints.add(endpoint)
        if len(row["base_ids"]) != len(row["donor_ids"]) \
                or row["base_ids"][-1] != row["donor_ids"][-1]:
            raise ValueError("paired prompts are not source-aligned")
        family, template = row["family"], row["template_id"]
        changes = _changed(row)
        if family == "A1" and changes != (0, 3):
            raise ValueError("A1 token changes moved")
        if family == "P" and changes != (2,):
            raise ValueError("P token changes moved")
        if family == "A2":
            expected_a2 = {
                "record_coordination": (2, 5, 7),
                "while_observers": (3, 6, 9),
                "reported_frame": (1, 4, 7),
            }[template]
            if changes != expected_a2:
                raise ValueError("A2 token changes moved")
        if family == "C" and len(changes) != 2:
            raise ValueError("C must change only subject and location")
        key = (row["phase"], family, template, row["direction_id"])
        counts[key] = counts.get(key, 0) + 1
    for phase in ("FIT", "HOLDOUT"):
        for direction in ("past_to_present", "present_to_past"):
            if counts.get((phase, "A1", "served_one_purpose", direction)) != 4:
                raise ValueError("A1 half-direction balance changed")
            for template in A2_TEMPLATE_ORDER:
                if counts.get((phase, "A2", template, direction)) != 4:
                    raise ValueError("A2 half-direction balance changed")
        for family, templates in (("P", ("served_one_purpose_practical",)),
                                  ("C", C_TEMPLATE_ORDER)):
            for template in templates:
                for tense in ("past", "present"):
                    for direction in ("primary_to_alternative", "alternative_to_primary"):
                        if counts.get((phase, family, template, f"{tense}_{direction}")) != 2:
                            raise ValueError("control half-cell balance changed")
    prior_rows = (original_narrative.build_rows() + fresh_v1.build_rows()
                  + newlex_v1.build_rows())
    prior_texts = {row[f"{side}_text"] for row in prior_rows for side in ("base", "donor")}
    prior_tokens = {tuple(row[f"{side}_ids"]) for row in prior_rows for side in ("base", "donor")}
    current_texts = {row[f"{side}_text"] for row in materialized for side in ("base", "donor")}
    if current_texts & prior_texts or endpoints & prior_tokens:
        raise ValueError("authority overlaps a prior narrative prompt endpoint")
    return canonical_sha256(materialized)


def authority_sha256() -> str:
    return validate_rows(build_rows())


if __name__ == "__main__":
    print(authority_sha256())
