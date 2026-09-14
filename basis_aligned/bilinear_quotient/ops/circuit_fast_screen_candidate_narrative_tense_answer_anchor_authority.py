#!/usr/bin/env python3
"""Frozen answer-clause-anchor narrative-tense capability authority."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import tiktoken

import circuit_fast_screen_candidate_narrative_tense as original
import circuit_fast_screen_candidate_narrative_tense_fresh_unchanged_carrier as fresh
import circuit_fast_screen_candidate_narrative_tense_newlex_carrier_confirmation as newlex
import circuit_fast_screen_candidate_narrative_tense_a2_c_native_capability as a2c
import circuit_fast_screen_candidate_narrative_tense_newlex_a1_a2_authority as fresh_v2


SCHEMA = "narrative_tense_answer_anchor_authority_v1"
TASK_ID = "narrative_tense.past_vs_present"
CAPABILITY_ID = "narrative_tense.answer_anchor_native_capability_v1"
CAUSAL_CANDIDATE_ID = "narrative_tense.attn11_head3_answer_anchor_carrier_v1"
ENCODING = tiktoken.get_encoding("gpt2")
GROUPS = 16
FIT_GROUPS = tuple(range(8))
HOLDOUT_GROUPS = tuple(range(8, 16))
ANSWERS = (" was", " is")

SUBJECT_PAIRS = (
    ("snake", "turtle"), ("lizard", "rabbit"), ("rat", "fox"),
    ("wolf", "cow"), ("pig", "duck"), ("hen", "bull"), ("calf", "lamb"),
    ("mask", "badge"), ("medal", "stamp"), ("label", "sign"),
    ("frame", "screen"), ("globe", "statue"), ("bush", "grass"),
    ("leaf", "root"), ("fruit", "grain"), ("soap", "brush"),
)
TAILS = ("flour", "sugar", "salt", "spice", "tea", "coffee", "milk", "cream",
         "butter", "egg", "honey", "jam", "meal", "lunch", "dinner", "breakfast")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def _answer(past: bool) -> str:
    return ANSWERS[0 if past else 1]


def _a1(subject: str, tail: str, past: bool) -> str:
    lead, verb = (("Yesterday", "served") if past else ("Today", "serves"))
    return f"{lead} the {subject} {verb} one role. {lead}, the central role of the {tail}"


def _a2(subject: str, tail: str, past: bool) -> str:
    lead, verb = (("Yesterday", "described") if past else ("Today", "describe"))
    return (f"{lead}, records {verb} the {subject} in one role. "
            f"{lead}, the central role of the {tail}")


def _p(subject: str, tail: str, past: bool, wording: str) -> str:
    lead, verb = (("Yesterday", "served") if past else ("Today", "serves"))
    return f"{lead} the {subject} {verb} one role. {lead}, the {wording} role of the {tail}"


def _c(subject: str, tail: str, past: bool, wording: str) -> str:
    lead, verb = (("Yesterday", "described") if past else ("Today", "describe"))
    return (f"{lead}, records {verb} the {subject} in one role. "
            f"{lead}, the {wording} role of the {tail}")


def _joint(text: str, answer: str) -> tuple[list[int], int]:
    ids = ENCODING.encode(text)
    complete = ENCODING.encode(text + answer)
    if ENCODING.decode(ids) != text or complete[:-1] != ids \
            or len(ENCODING.encode(answer)) != 1:
        raise ValueError("unstable joint tokenization")
    return ids, complete[-1]


def _row(group: int, family: str, direction: str, base_text: str,
         donor_text: str, base_past: bool, donor_past: bool) -> dict:
    base_answer, donor_answer = _answer(base_past), _answer(donor_past)
    base_ids, base_answer_id = _joint(base_text, base_answer)
    donor_ids, donor_answer_id = _joint(donor_text, donor_answer)
    base_foil = ANSWERS[1] if base_past else ANSWERS[0]
    donor_foil = ANSWERS[1] if donor_past else ANSWERS[0]
    _, base_foil_id = _joint(base_text, base_foil)
    _, donor_foil_id = _joint(donor_text, donor_foil)
    identity = [SCHEMA, group, family, direction, base_text, donor_text,
                base_answer, donor_answer]
    return {
        "schema": SCHEMA, "task_id": TASK_ID, "capability_id": CAPABILITY_ID,
        "phase": "FIT" if group in FIT_GROUPS else "HOLDOUT",
        "group_number": group, "row_id": canonical_sha256(identity),
        "family": family, "template_id": "direct_answer_anchor" if family in {"A1", "P"}
        else "reported_answer_anchor", "direction_id": direction,
        "base_text": base_text, "donor_text": donor_text,
        "base_ids": base_ids, "donor_ids": donor_ids,
        "base_answer": base_answer, "donor_answer": donor_answer,
        "base_foil": base_foil, "donor_foil": donor_foil,
        "base_answer_id": base_answer_id, "donor_answer_id": donor_answer_id,
        "base_foil_id": base_foil_id, "donor_foil_id": donor_foil_id,
    }


def _panel(group: int) -> list[dict]:
    primary, alternate = SUBJECT_PAIRS[group]
    tail = TAILS[group]
    forward = group % 2 == 0
    base_past, donor_past = ((True, False) if forward else (False, True))
    target_direction = "past_to_present" if forward else "present_to_past"
    control_past = group % 4 < 2
    control_direction = ("primary_to_alternative" if forward
                         else "alternative_to_primary")
    base_subject, donor_subject = ((primary, alternate) if forward
                                   else (alternate, primary))
    return [
        _row(group, "A1", target_direction,
             _a1(primary, tail, base_past), _a1(primary, tail, donor_past),
             base_past, donor_past),
        _row(group, "A2", target_direction,
             _a2(primary, tail, base_past), _a2(primary, tail, donor_past),
             base_past, donor_past),
        _row(group, "P", control_direction,
             _p(primary, tail, control_past, "formal"),
             _p(primary, tail, control_past, "public"), control_past, control_past),
        _row(group, "C", control_direction,
             _c(primary, tail, control_past, "official"),
             _c(primary, tail, control_past, "recorded"), control_past, control_past),
    ]


def build_rows() -> list[dict]:
    rows = [row for group in range(GROUPS) for row in _panel(group)]
    validate_rows(rows)
    return rows


def validate_rows(rows: Sequence[Mapping[str, object]]) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != [row for group in range(GROUPS) for row in _panel(group)] \
            or len(materialized) != 64:
        raise ValueError("authority differs from frozen design")
    vocab = tuple(x for pair in SUBJECT_PAIRS for x in pair) + TAILS
    if len(vocab) != 48 or len(set(vocab)) != 48 \
            or any(len(ENCODING.encode(" " + word)) != 1 for word in vocab):
        raise ValueError("fresh vocabulary must be 64 unique one-token words")
    used = set(original._SUBJECTS + original._ALTERNATES + original._PLACES + original._FOCUS) \
        | set(fresh.FRESH_VOCABULARY) | set(newlex.ELIGIBLE_SINGULARS) \
        | set(x for pair in a2c.SUBJECT_PAIRS for x in pair) \
        | set(a2c.TAILS) | set(a2c.LOCATIONS) | set(x for pair in fresh_v2.SUBJECT_PAIRS for x in pair) | set(fresh_v2.TAILS)
    if set(vocab) & used:
        raise ValueError("content vocabulary overlaps a prior narrative authority")
    row_ids, endpoints = set(), set()
    counts = {}
    for row in materialized:
        if row["row_id"] in row_ids:
            raise ValueError("duplicate row")
        row_ids.add(row["row_id"])
        if len(row["base_ids"]) != len(row["donor_ids"]) \
                or row["base_ids"][-1] != row["donor_ids"][-1]:
            raise ValueError("paired prompts are not source aligned")
        for side in ("base", "donor"):
            endpoint = tuple(row[f"{side}_ids"])
            if endpoint in endpoints:
                raise ValueError("duplicate prompt endpoint")
            endpoints.add(endpoint)
        key = (row["phase"], row["family"], row["direction_id"])
        counts[key] = counts.get(key, 0) + 1
    for phase in ("FIT", "HOLDOUT"):
        for family in ("A1", "A2"):
            for direction in ("past_to_present", "present_to_past"):
                if counts.get((phase, family, direction)) != 4:
                    raise ValueError("target balance changed")
        for family in ("P", "C"):
            for direction in ("primary_to_alternative", "alternative_to_primary"):
                if counts.get((phase, family, direction)) != 4:
                    raise ValueError("control balance changed")
    prior_rows = (original.build_rows() + fresh.build_rows() + newlex.build_rows()
                  + a2c.build_rows() + fresh_v2.build_rows())
    prior_endpoints = {tuple(row[f"{side}_ids"]) for row in prior_rows for side in ("base", "donor")}
    if endpoints & prior_endpoints:
        raise ValueError("prompt endpoint overlaps prior narrative evidence")
    return canonical_sha256(materialized)


def authority_sha256() -> str:
    return validate_rows(build_rows())


if __name__ == "__main__":
    print(authority_sha256())
