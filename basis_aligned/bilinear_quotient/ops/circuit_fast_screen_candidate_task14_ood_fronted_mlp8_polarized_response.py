#!/usr/bin/env python3
"""Frozen Task14 OOD-fronted authority for the prospective MLP8 split.

The underlying OOD text and whole-head outcomes have already been opened.  The
authority is therefore intervention-held-out, not a pristine OOD test.  No
MLP8 polarized-response intervention has been evaluated on these rows.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Mapping, Sequence

import circuit_battery_task14 as task14


SCHEMA = "task14_ood_fronted_mlp8_polarized_response_authority_v1"
TASK_ID = "subject_verb.number_agreement"
SPLIT = "OOD_TEXT_REUSE_NEW_MLP8_INTERVENTION"
CAPABILITY_ID = "subject_verb.number_agreement.ood_fronted_mlp8_native_capability_v1"
CAUSAL_CANDIDATE_ID = \
    "subject_verb.number_agreement.head11_3_ood_fronted_subject_mlp8_polarized_response_factorial_v1"
SOURCE_GROUP_NUMBERS = tuple(range(0, 8)) + tuple(range(16, 24))
ROLES = ("recipient", "opposite_same_lemma", "same_number_different_lemma")
SUBJECT_POSITION = 8
EXPECTED_AUTHORITY_SHA256 = "935412b14f145917fd65973779669eedce53f5001b865a64f4a018c4ea8dc02a"


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_rows() -> list[dict]:
    authority, _digest = task14.build_authority()
    return sorted(
        [row for row in authority
         if row["split"] == "OOD" and row["transform_id"] == "A1"
         and row["group_number"] in SOURCE_GROUP_NUMBERS],
        key=lambda row: int(row["group_number"]),
    )


def _endpoint(row: Mapping[str, object], side: str) -> dict:
    answer_id = int(row[f"{side}_answer_id"])
    return {
        "text": str(row[f"{side}_text"]),
        "ids": list(row[f"{side}_ids"]),
        "subject": str(row[f"{side}_text"]).rsplit(" ", 1)[-1],
        "subject_number": str(row[f"{side}_subject_number"]),
        "answer_id": answer_id,
        "foil_id": 389 if answer_id == 318 else 318,
    }


def _build_unvalidated() -> list[dict]:
    source = _source_rows()
    by_direction: dict[str, list[dict]] = {"singular_to_plural": [],
                                           "plural_to_singular": []}
    for row in source:
        direction = f"{row['base_subject_number']}_to_{row['donor_subject_number']}"
        by_direction[direction].append(row)
    lexical_source = {}
    for rows in by_direction.values():
        ordered = sorted(rows, key=lambda row: int(row["group_number"]))
        for index, row in enumerate(ordered):
            lexical_source[int(row["group_number"])] = ordered[(index + 4) % len(ordered)]

    output = []
    for row in source:
        group = int(row["group_number"])
        recipient = _endpoint(row, "base")
        opposite = _endpoint(row, "donor")
        lexical_row = lexical_source[group]
        lexical_token = int(lexical_row["base_ids"][SUBJECT_POSITION])
        lexical_ids = list(recipient["ids"])
        lexical_ids[SUBJECT_POSITION] = lexical_token
        lexical_text = task14.ENCODING.decode(lexical_ids)
        lexical = {
            "text": lexical_text,
            "ids": lexical_ids,
            "subject": lexical_text.rsplit(" ", 1)[-1],
            "subject_number": recipient["subject_number"],
            "answer_id": recipient["answer_id"],
            "foil_id": recipient["foil_id"],
        }
        direction = f"{recipient['subject_number']}_to_{opposite['subject_number']}"
        endpoints = {
            "recipient": recipient,
            "opposite_same_lemma": opposite,
            "same_number_different_lemma": lexical,
        }
        identity = [SCHEMA, row["row_id"], lexical_row["row_id"], endpoints]
        output.append({
            "schema": SCHEMA,
            "task_id": TASK_ID,
            "split": SPLIT,
            "capability_id": CAPABILITY_ID,
            "row_id": canonical_sha256(identity),
            "source_row_id": row["row_id"],
            "source_group_id": row["group_id"],
            "group_number": group,
            "lexical_source_group_number": int(lexical_row["group_number"]),
            "template_id": "ood_fronted_two_attractors",
            "direction_id": direction,
            "attractor_state": [bool(row["base_attractor_plural"]),
                                bool(row["base_second_attractor_plural"])],
            "subject_position": SUBJECT_POSITION,
            "endpoints": endpoints,
        })
    return output


def validate_rows(rows: Sequence[Mapping[str, object]], *, verify_hash: bool = True) -> str:
    materialized = [dict(row) for row in rows]
    if materialized != _build_unvalidated() or len(materialized) != 16:
        raise ValueError("authority differs from the frozen 16-row OOD design")
    if len({row["row_id"] for row in materialized}) != 16:
        raise ValueError("authority row IDs are not unique")
    balance = Counter((row["direction_id"], tuple(row["attractor_state"]))
                      for row in materialized)
    if len(balance) != 8 or set(balance.values()) != {2}:
        raise ValueError(f"direction/attractor balance changed: {balance}")
    prompts, token_tuples = [], []
    for row in materialized:
        if row["split"] != SPLIT or row["subject_position"] != SUBJECT_POSITION:
            raise ValueError("scope or subject position changed")
        endpoint = row["endpoints"]
        if tuple(endpoint) != ROLES:
            raise ValueError("endpoint roles changed")
        recipient = endpoint["recipient"]
        opposite = endpoint["opposite_same_lemma"]
        lexical = endpoint["same_number_different_lemma"]
        if recipient["subject_number"] == opposite["subject_number"] \
                or recipient["subject_number"] != lexical["subject_number"]:
            raise ValueError("number relation changed")
        if recipient["subject"] == lexical["subject"]:
            raise ValueError("lexical control retained the recipient lemma")
        for alternate in (opposite, lexical):
            differences = [index for index, pair in enumerate(
                zip(recipient["ids"], alternate["ids"])) if pair[0] != pair[1]]
            if differences != [SUBJECT_POSITION]:
                raise ValueError("matched endpoint changes outside subject token 8")
        for role in ROLES:
            value = endpoint[role]
            if len(value["ids"]) != 9:
                raise ValueError("OOD endpoint is not exactly nine tokens")
            answer = " is" if value["answer_id"] == 318 else " are"
            if task14.ENCODING.encode(value["text"] + answer) \
                    != value["ids"] + [value["answer_id"]]:
                raise ValueError("prompt-plus-answer tokenization changed")
            prompts.append(value["text"])
            token_tuples.append(tuple(value["ids"]))
    if len(prompts) != 48 or len(set(prompts)) != 48 \
            or len(set(token_tuples)) != 48:
        raise ValueError("authority endpoints are not 48 unique prompts")
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
        "schema": SCHEMA,
        "task_id": TASK_ID,
        "split": SPLIT,
        "row_count": len(rows),
        "roles": list(ROLES),
        "source_group_numbers": list(SOURCE_GROUP_NUMBERS),
        "authority_sha256": canonical_sha256(rows),
        "direction_attractor_cells": {
            f"{direction}__a{int(state[0])}{int(state[1])}": count
            for (direction, state), count in sorted(Counter(
                (row["direction_id"], tuple(row["attractor_state"]))
                for row in rows).items())
        },
        "data_status": "OOD text and whole-head outcomes open; MLP8 intervention held out",
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
