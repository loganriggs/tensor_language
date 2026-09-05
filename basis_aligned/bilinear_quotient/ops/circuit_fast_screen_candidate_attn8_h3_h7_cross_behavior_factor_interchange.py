#!/usr/bin/env python3
# BQLANE: cpu
"""Frozen paired list/digit authority for L8H3/H7 cross-behavior factors."""

from __future__ import annotations

from collections import Counter
import hashlib
import json

import tiktoken


SCHEMA = "attn8_h3_h7_cross_behavior_factor_interchange_authority_v1"
ENC = tiktoken.get_encoding("gpt2")
LEXICAL = (
    ("ember", "reef", "orchard"), ("harp", "meadow", "lantern"),
    ("canyon", "violin", "anchor"), ("copper", "planet", "basket"),
    ("velvet", "castle", "pencil"), ("silver", "tunnel", "camera"),
    ("winter", "rocket", "pillow"), ("yellow", "ocean", "mirror"),
)
EXPECTED_ROWS_SHA256 = "2de29587089cdef69bfe95c60f6cb49026d4b40114df68f587a7ea9f1ca3e019"


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _positions(ids, text, format_id):
    if format_id == "list":
        positions, prefix = [], ""
        for line in text.splitlines(keepends=True):
            positions.append(len(ENC.encode(prefix))); prefix += line
        if prefix != text or len(positions) != 3 or ids[-1] != ENC.encode("\n")[0]:
            raise ValueError("list semantic positions changed")
    else:
        commas = [i for i, token in enumerate(ids) if token == ENC.encode(",")[0]]
        positions = [i-1 for i in commas[-3:]]
        if len(positions) != 3 or commas[-1] != len(ids)-1:
            raise ValueError("digit semantic positions changed")
    return positions


def _target(format_id, words, values):
    if format_id == "list":
        text = "".join(f"{value}. {word}\n" for value, word in zip(values, words))
        answer_text = str(values[-1]+1)
    else:
        text = f"For the {words[0]}, the numbers are {values[0]}, {values[1]}, {values[2]},"
        answer_text = " " + str(values[-1]+1)
    ids = ENC.encode(text); answer_ids = ENC.encode(answer_text)
    if len(answer_ids) != 1:
        raise ValueError("target answer is not one token")
    return {"text": text, "ids": ids, "source_positions": _positions(ids, text, format_id),
            "query_position": len(ids)-1, "answer_text": answer_text,
            "answer_id": answer_ids[0]}


def _control(control_id, words, values):
    if control_id == "repeated_list_copy":
        repeated = (values[-1],) * 3
        endpoint = _target("list", words, repeated)
        answer_text = str(values[-1]); endpoint["answer_text"] = answer_text
        endpoint["answer_id"] = ENC.encode(answer_text)[0]
        endpoint["preference_foil_id"] = ENC.encode(str(values[-1]+1))[0]
    elif control_id == "digit_copy":
        repeated = (values[-1],) * 3
        endpoint = _target("digit", words, repeated)
        answer_text = " " + str(values[-1]); endpoint["answer_text"] = answer_text
        endpoint["answer_id"] = ENC.encode(answer_text)[0]
        endpoint["preference_foil_id"] = ENC.encode(" "+str(values[-1]+1))[0]
    elif control_id == "step_two":
        stepped = (values[0], values[0]+2, values[0]+4)
        endpoint = _target("digit", words, stepped)
        answer_text = " " + str(values[0]+6); endpoint["answer_text"] = answer_text
        endpoint["answer_id"] = ENC.encode(answer_text)[0]
        endpoint["preference_foil_id"] = ENC.encode(" "+str(values[0]+5))[0]
    else:
        raise ValueError(control_id)
    return endpoint


def _build_unvalidated():
    output = []
    for group_index, words in enumerate(LEXICAL):
        split = "FIT" if group_index < 4 else "SELECT"
        base_values = (8+2*group_index, 9+2*group_index, 10+2*group_index)
        donor_values = tuple(value+1 for value in base_values)
        endpoints = {format_id: {"base": _target(format_id, words, base_values),
                                 "donor": _target(format_id, words, donor_values)}
                     for format_id in ("list", "digit")}
        controls = {direction: {control: _control(control, words, values)
                                for control in ("repeated_list_copy", "digit_copy", "step_two")}
                    for direction, values in (("base_to_donor", base_values),
                                              ("donor_to_base", donor_values))}
        for format_id in ("list", "digit"):
            for direction in ("base_to_donor", "donor_to_base"):
                recipient = "base" if direction == "base_to_donor" else "donor"
                donor = "donor" if recipient == "base" else "base"
                identity = [SCHEMA, group_index, split, format_id, direction]
                output.append({"schema": SCHEMA, "row_id": canonical_sha256(identity),
                    "group_id": f"numeric-pair-{group_index:02d}", "split": split,
                    "recipient_format": format_id, "direction": direction,
                    "visible_base_values": list(base_values), "visible_donor_values": list(donor_values),
                    "recipient": endpoints[format_id][recipient],
                    "within_donor": endpoints[format_id][donor],
                    "cross_same": endpoints["digit" if format_id == "list" else "list"][recipient],
                    "cross_opposite": endpoints["digit" if format_id == "list" else "list"][donor],
                    "controls": controls[direction]})
    return output


def validate_rows(rows, verify_frozen_hash=True):
    rows = list(rows)
    if len(rows) != 32 or len({row["row_id"] for row in rows}) != 32:
        raise ValueError("paired authority must have 32 unique directional rows")
    counts = Counter((row["split"], row["recipient_format"], row["direction"]) for row in rows)
    if len(counts) != 8 or set(counts.values()) != {4}:
        raise ValueError(f"split/format/direction balance changed: {counts}")
    for row in rows:
        recipient, within, same, cross = (row[key] for key in
            ("recipient", "within_donor", "cross_same", "cross_opposite"))
        if row["recipient_format"] == "list":
            if not (recipient["answer_text"].isdigit() and within["answer_text"].isdigit()
                    and same["answer_text"].startswith(" ") and cross["answer_text"].startswith(" ")):
                raise ValueError("list/digit answer surfaces changed")
        else:
            if not (recipient["answer_text"].startswith(" ") and within["answer_text"].startswith(" ")
                    and same["answer_text"].isdigit() and cross["answer_text"].isdigit()):
                raise ValueError("digit/list answer surfaces changed")
        recipient_values = row["visible_base_values"] if row["direction"] == "base_to_donor" else row["visible_donor_values"]
        donor_values = row["visible_donor_values"] if row["direction"] == "base_to_donor" else row["visible_base_values"]
        if recipient["answer_text"].strip() != str(recipient_values[-1]+1) \
                or same["answer_text"].strip() != str(recipient_values[-1]+1) \
                or within["answer_text"].strip() != str(donor_values[-1]+1) \
                or cross["answer_text"].strip() != str(donor_values[-1]+1):
            raise ValueError("visible states or semantic next-label answers are not matched")
        for endpoint in (recipient, within, same, cross, *row["controls"].values()):
            if len(endpoint["source_positions"]) != 3 \
                    or endpoint["query_position"] != len(endpoint["ids"])-1:
                raise ValueError("semantic source/query mapping changed")
    digest = canonical_sha256(rows)
    if verify_frozen_hash and digest != EXPECTED_ROWS_SHA256:
        raise ValueError(f"paired authority digest changed: {digest}")
    return digest


def build_rows():
    rows = _build_unvalidated(); validate_rows(rows); return rows


def compile_plan():
    rows = build_rows()
    return {"schema": SCHEMA, "row_count": len(rows), "authority_sha256": validate_rows(rows),
            "cells": {"|".join(key): value for key, value in Counter(
                (row["split"], row["recipient_format"], row["direction"]) for row in rows).items()},
            "model_loaded": False, "outcomes_opened": []}


if __name__ == "__main__": print(json.dumps(compile_plan(), indent=2, sort_keys=True))
