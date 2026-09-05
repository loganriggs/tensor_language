#!/usr/bin/env python3
"""Frozen native-only authority for a fresh matched-natural Task14 screen.

The three endpoints in a row share an exact prefix.  The opposite endpoint
changes only the grammatical subject's number; the lexical-control endpoint
changes only its lemma while preserving number.  This module contains no model
loading, causal intervention, result access, queue operation, or claim.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Mapping, Sequence

import circuit_battery_task14 as old_task14
import circuit_fast_screen_candidate_task14_fresh_fronted_natural_qk_number_specificity as fresh_v1


SCHEMA = "task14_fresh_matched_natural_native_capability_authority_v1"
TASK_ID = "subject_verb.number_agreement"
CAPABILITY_ID = "subject_verb.number_agreement.fresh_matched_natural_native_capability_v1"
CAUSAL_CANDIDATE_ID = \
    "subject_verb.number_agreement.head11_3_fresh_matched_natural_qk_factorial_v1"
FIT_GROUPS = tuple(range(8))
HOLDOUT_GROUPS = tuple(range(8, 16))
ROLES = ("recipient", "opposite_same_lemma", "same_number_different_lemma")
TEMPLATES = (
    ("across_beside", "Across the {a1} beside the {a2}, the {subject}"),
    ("between_below", "Between the {a1} below the {a2}, the {subject}"),
)
NOUN_PAIRS = (
    ("agent", "agents"), ("artist", "artists"),
    ("author", "authors"), ("driver", "drivers"),
    ("farmer", "farmers"), ("nurse", "nurses"),
    ("pilot", "pilots"), ("reader", "readers"),
    ("rider", "riders"), ("sailor", "sailors"),
    ("singer", "singers"), ("writer", "writers"),
    ("owner", "owners"), ("leader", "leaders"),
    ("guard", "guards"), ("judge", "judges"),
)
EXPECTED_AUTHORITY_SHA256 = "8862dc84c10a28c0857cb7b201adab46e72f2e7063069fcf1a09f50ec21947d7"


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _phase(group: int) -> str:
    return "FIT" if group in FIT_GROUPS else "HOLDOUT"


def _answer_id(number: int) -> int:
    return 318 if number == 0 else 389


def _render(template: str, a1: str, a2: str, subject: str) -> tuple[str, list[int]]:
    text = template.format(a1=a1, a2=a2, subject=subject)
    ids = old_task14.ENCODING.encode(text)
    if len(ids) != 9:
        raise ValueError(f"prompt is not nine GPT-2 tokens: {text!r} -> {ids}")
    return text, ids


def _build_unvalidated() -> list[dict]:
    rows = []
    for group in range(16):
        phase_start = 0 if group < 8 else 8
        local = group - phase_start
        recipient_number = 0 if local < 4 else 1
        direction = "singular_to_plural" if recipient_number == 0 \
            else "plural_to_singular"
        state_index = local % 4
        attractor_state = (state_index // 2, state_index % 2)
        alternative_group = phase_start + ((local + 4) % 8)
        a1_group = phase_start + ((local + 1) % 8)
        a2_group = phase_start + ((local + 2) % 8)
        a1 = NOUN_PAIRS[a1_group][attractor_state[0]]
        a2 = NOUN_PAIRS[a2_group][attractor_state[1]]
        subjects = {
            "recipient": NOUN_PAIRS[group][recipient_number],
            "opposite_same_lemma": NOUN_PAIRS[group][1-recipient_number],
            "same_number_different_lemma": NOUN_PAIRS[alternative_group][recipient_number],
        }
        numbers = {
            "recipient": recipient_number,
            "opposite_same_lemma": 1-recipient_number,
            "same_number_different_lemma": recipient_number,
        }
        for template_id, template in TEMPLATES:
            endpoints = {}
            for role in ROLES:
                text, ids = _render(template, a1, a2, subjects[role])
                answer = _answer_id(numbers[role])
                endpoints[role] = {
                    "text": text, "ids": ids, "subject": subjects[role],
                    "subject_number": "singular" if numbers[role] == 0 else "plural",
                    "answer_id": answer, "foil_id": 389 if answer == 318 else 318,
                }
            identity = [SCHEMA, group, template_id, endpoints]
            rows.append({
                "schema": SCHEMA, "task_id": TASK_ID, "capability_id": CAPABILITY_ID,
                "phase": _phase(group), "group_number": group,
                "row_id": canonical_sha256(identity), "template_id": template_id,
                "direction_id": direction, "attractor_state": list(attractor_state),
                "subject_position": 8, "recipient_subject_pair_index": group,
                "lexical_control_subject_pair_index": alternative_group,
                "endpoints": endpoints,
            })
    return rows


def _prior_material() -> tuple[set[str], set[str], set[tuple[int, ...]]]:
    old_authority, _digest = old_task14.build_authority()
    vocabulary, prompts, tokens = set(), set(), set()
    for row in old_authority:
        for key in ("head_pair", "attractor_pair", "second_head_pair",
                    "second_attractor_pair", "surface_attractor_pair"):
            vocabulary.update(row[key])
        for side in ("base", "donor"):
            prompts.add(row[f"{side}_text"])
            tokens.add(tuple(row[f"{side}_ids"]))
    vocabulary.update(word for pair in fresh_v1.NOUN_PAIRS for word in pair)
    for row in fresh_v1.build_rows():
        for role in ("base", "same", "opposite"):
            prompts.add(row[f"{role}_text"])
            tokens.add(tuple(row[f"{role}_ids"]))
    return vocabulary, prompts, tokens


def validate_rows(rows: Sequence[Mapping[str, object]], *, verify_hash: bool = True) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build_unvalidated() or len(materialized) != 32:
        raise ValueError("authority differs from the frozen 32-row design")
    if len({row["row_id"] for row in materialized}) != 32:
        raise ValueError("authority row IDs are not unique")
    forms = {word for pair in NOUN_PAIRS for word in pair}
    prior_vocabulary, prior_prompts, prior_tokens = _prior_material()
    if len(forms) != 32 or forms & prior_vocabulary:
        raise ValueError("fresh noun forms are duplicated or overlap prior Task14")
    if any(len(old_task14.ENCODING.encode(" " + word)) != 1 for word in forms):
        raise ValueError("fresh noun form is not one GPT-2 token after a space")
    counts = Counter()
    prompts, tokens = [], []
    phase_forms = {"FIT": set(), "HOLDOUT": set()}
    for row in materialized:
        endpoints = row["endpoints"]
        if tuple(endpoints) != ROLES or row["subject_position"] != 8:
            raise ValueError("endpoint roles or subject position changed")
        recipient = endpoints["recipient"]
        opposite = endpoints["opposite_same_lemma"]
        lexical = endpoints["same_number_different_lemma"]
        if recipient["subject_number"] == opposite["subject_number"] \
                or recipient["subject_number"] != lexical["subject_number"]:
            raise ValueError("endpoint number relation changed")
        if recipient["subject"] == lexical["subject"]:
            raise ValueError("lexical control did not change the subject lemma")
        for alternate in (opposite, lexical):
            differences = [index for index, pair in enumerate(
                zip(recipient["ids"], alternate["ids"])) if pair[0] != pair[1]]
            if differences != [8]:
                raise ValueError("matched endpoint changes outside subject token 8")
        for role in ROLES:
            endpoint = endpoints[role]
            answer = " is" if endpoint["answer_id"] == 318 else " are"
            if old_task14.ENCODING.encode(endpoint["text"] + answer) \
                    != endpoint["ids"] + [endpoint["answer_id"]]:
                raise ValueError("answer continuation tokenization changed")
            prompts.append(endpoint["text"])
            tokens.append(tuple(endpoint["ids"]))
            phase_forms[row["phase"]].add(endpoint["subject"])
            counts[(row["phase"], row["template_id"], row["direction_id"], role)] += 1
    if len(prompts) != 96 or len(set(prompts)) != 96 or len(set(tokens)) != 96:
        raise ValueError("authority must contain 96 unique prompt endpoints")
    if set(prompts) & prior_prompts or set(tokens) & prior_tokens:
        raise ValueError("fresh prompt endpoint overlaps prior Task14")
    if phase_forms["FIT"] & phase_forms["HOLDOUT"]:
        raise ValueError("FIT and HOLDOUT subject vocabularies overlap")
    if len(counts) != 24 or set(counts.values()) != {4}:
        raise ValueError(f"capability cells are not 24 x 4: {counts}")
    digest = canonical_sha256(materialized)
    if verify_hash and digest != EXPECTED_AUTHORITY_SHA256:
        raise ValueError(f"authority digest changed: {digest}")
    return digest


def build_rows() -> list[dict]:
    rows = _build_unvalidated()
    validate_rows(rows)
    return rows


def compile_plan() -> dict:
    rows = build_rows()
    return {
        "schema": SCHEMA, "task_id": TASK_ID, "capability_id": CAPABILITY_ID,
        "rows": len(rows), "prompt_endpoints": len(rows) * len(ROLES),
        "fit_groups": list(FIT_GROUPS), "holdout_groups": list(HOLDOUT_GROUPS),
        "roles": list(ROLES), "authority_sha256": canonical_sha256(rows),
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
