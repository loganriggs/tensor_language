#!/usr/bin/env python3
"""Frozen third-corpus authority for direction-cardinality prototype transfer."""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import circuit_battery_task14 as old_task14
import circuit_fast_screen_candidate_task14_fixed_reader_transfer as prior


SCHEMA = "task14_cardinality_prototype_transfer_authority_v1"
TASK_ID = "subject_verb.number_agreement"
CAPABILITY_ID = "subject_verb.number_agreement.cardinality_prototype_transfer_capability_v1"
CAUSAL_CANDIDATE_ID = "subject_verb.number_agreement.mlp6_7_fixed_direction_cardinality_upstream_program_v1"
SUBJECT_POSITION = 8
ROLES = ("recipient", "opposite_same_lemma", "same_number_different_lemma")
TEMPLATES = (
    ("near_beyond", "Near the {a1} beyond the {a2}, the {subject}"),
    ("beyond_near", "Beyond the {a1} near the {a2}, the {subject}"),
)
NOUN_PAIRS = (
    ("member", "members"), ("neighbor", "neighbors"),
    ("partner", "partners"), ("commander", "commanders"),
    ("critic", "critics"), ("dealer", "dealers"),
    ("fisherman", "fishermen"), ("governor", "governors"),
    ("journalist", "journalists"), ("merchant", "merchants"),
    ("pastor", "pastors"), ("trainer", "trainers"),
    ("warrior", "warriors"), ("apprentice", "apprentices"),
    ("athlete", "athletes"), ("attendant", "attendants"),
)
EXPECTED_AUTHORITY_SHA256 = "f1c0e7cf386c223352c220a0a2ae620b8b126bf739d9f21cd2f8e889263201cb"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _render(template: str, a1: str, a2: str, subject: str) -> tuple[str, list[int]]:
    text = template.format(a1=a1, a2=a2, subject=subject)
    ids = old_task14.ENCODING.encode(text)
    if len(ids) != 9:
        raise ValueError(f"not nine tokens: {text!r} {ids}")
    return text, ids


def _build_unvalidated() -> list[dict[str, object]]:
    rows = []
    half = len(NOUN_PAIRS) // 2
    for group in range(len(NOUN_PAIRS)):
        recipient_number = 0 if group < half else 1
        within = group % half
        direction = "singular_to_plural" if recipient_number == 0 else "plural_to_singular"
        lexical_group = (0 if recipient_number == 0 else half) + (within + 5) % half
        a1_group, a2_group = (group + 3) % len(NOUN_PAIRS), (group + 7) % len(NOUN_PAIRS)
        attractor_state = ((within % 4) // 2, within % 2)
        a1 = NOUN_PAIRS[a1_group][attractor_state[0]]
        a2 = NOUN_PAIRS[a2_group][attractor_state[1]]
        for template_id, template in TEMPLATES:
            subjects = {
                "recipient": NOUN_PAIRS[group][recipient_number],
                "opposite_same_lemma": NOUN_PAIRS[group][1 - recipient_number],
                "same_number_different_lemma": NOUN_PAIRS[lexical_group][recipient_number],
            }
            endpoints = {}
            for role in ROLES:
                number = 1 - recipient_number if role == "opposite_same_lemma" else recipient_number
                text, ids = _render(template, a1, a2, subjects[role])
                answer = 318 if number == 0 else 389
                endpoints[role] = {
                    "text": text, "ids": ids, "subject": subjects[role],
                    "subject_number": "singular" if number == 0 else "plural",
                    "answer_id": answer, "foil_id": 389 if answer == 318 else 318,
                }
            identity = [SCHEMA, group, template_id, attractor_state, endpoints]
            rows.append({
                "schema": SCHEMA, "task_id": TASK_ID, "capability_id": CAPABILITY_ID,
                "causal_candidate_id": CAUSAL_CANDIDATE_ID, "phase": "PROSPECTIVE",
                "group_number": group, "row_id": canonical_sha256(identity),
                "template_id": template_id, "direction_id": direction,
                "attractor_state": list(attractor_state), "subject_position": SUBJECT_POSITION,
                "endpoints": endpoints,
            })
    return rows


def _prior_material() -> tuple[set[str], set[str], set[tuple[int, ...]]]:
    vocabulary, prompts, tokens = prior.prior._prior_material()
    for authority in (prior.prior, prior):
        vocabulary.update(word for pair in authority.NOUN_PAIRS for word in pair)
        for row in authority.build_rows():
            for endpoint in row["endpoints"].values():
                prompts.add(endpoint["text"])
                tokens.add(tuple(endpoint["ids"]))
    return vocabulary, prompts, tokens


def validate_rows(rows: list[dict[str, object]], verify_hash: bool = True) -> str:
    rows = list(rows)
    if rows != _build_unvalidated() or len(rows) != 32 or len({x["row_id"] for x in rows}) != 32:
        raise ValueError("authority changed")
    if sorted(Counter((x["direction_id"], x["template_id"]) for x in rows).values()) != [8] * 4:
        raise ValueError("direction/template balance changed")
    prior_vocabulary, prior_prompts, prior_tokens = _prior_material()
    forms = {word for pair in NOUN_PAIRS for word in pair}
    if len(forms) != 32 or forms & prior_vocabulary or any(len(old_task14.ENCODING.encode(" " + word)) != 1 for word in forms):
        raise ValueError("noun novelty/tokenization failed")
    prompts, tokens = [], []
    for row in rows:
        recipient, opposite, lexical = (row["endpoints"][role] for role in ROLES)
        if recipient["subject_number"] == opposite["subject_number"] or recipient["subject_number"] != lexical["subject_number"] or recipient["subject"] == lexical["subject"]:
            raise ValueError("endpoint relation failed")
        for alternate in (opposite, lexical):
            if [index for index, pair in enumerate(zip(recipient["ids"], alternate["ids"])) if pair[0] != pair[1]] != [SUBJECT_POSITION]:
                raise ValueError("endpoint delta failed")
        for endpoint in row["endpoints"].values():
            answer = " is" if endpoint["answer_id"] == 318 else " are"
            if old_task14.ENCODING.encode(endpoint["text"] + answer) != endpoint["ids"] + [endpoint["answer_id"]]:
                raise ValueError("continuation failed")
            prompts.append(endpoint["text"])
            tokens.append(tuple(endpoint["ids"]))
    if len(set(prompts)) != 96 or len(set(tokens)) != 96 or set(prompts) & prior_prompts or set(tokens) & prior_tokens:
        raise ValueError("prompt novelty failed")
    digest = canonical_sha256(rows)
    if verify_hash and digest != EXPECTED_AUTHORITY_SHA256:
        raise ValueError(f"hash changed {digest}")
    return digest


def build_rows() -> list[dict[str, object]]:
    rows = _build_unvalidated()
    validate_rows(rows)
    return rows


def compile_plan() -> dict[str, object]:
    rows = build_rows()
    return {
        "schema": SCHEMA, "task_id": TASK_ID, "capability_id": CAPABILITY_ID,
        "causal_candidate_id": CAUSAL_CANDIDATE_ID, "row_count": len(rows),
        "prompt_endpoints": 96, "templates": [x[0] for x in TEMPLATES],
        "authority_sha256": canonical_sha256(rows),
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
