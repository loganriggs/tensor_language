#!/usr/bin/env python3
"""Frozen same-sequence temporal plus is-was command capability authority."""

from __future__ import annotations

from collections import Counter
import hashlib, json

import circuit_battery_task14 as tokenizer_source

SCHEMA = "temporal_iswas_dual_command_authority_v1"
TASK_ID = "cross_task.temporal_iswas.dual_command"
CAPABILITY_ID = "cross_task.temporal_iswas.dual_command_native_capability_v1"
CELLS = ("00", "10", "01", "11")
TEMPLATES = ("schedule_semicolon", "forecast_while")
AGENTS = ("broker", "pilot", "teacher", "doctor", "sailor", "baker", "driver", "artist",
          "author", "farmer", "nurse", "reader", "rider", "singer", "writer", "guard")
PARTNERS = ("analyst", "chemist", "editor", "judge", "leader", "manager", "owner", "planner",
            "reporter", "scholar", "tailor", "visitor", "worker", "buyer", "seller", "trainer")
ENCODING = tokenizer_source.ENCODING


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _single(text):
    ids = ENCODING.encode(text)
    if len(ids) != 1:
        raise ValueError(f"expected one token for {text!r}: {ids}")
    return ids[0]


TEMPORAL = {0: ("Next year", " will", " submit"), 1: ("Last year", " had", " submitted")}
ISWAS = {0: ("today", " is"), 1: ("yesterday", " was")}


def _render(template, agent, partner, temporal_bit, iswas_bit):
    temporal_cue, temporal_answer, temporal_verb = TEMPORAL[temporal_bit]
    iswas_cue, iswas_answer = ISWAS[iswas_bit]
    if template == "schedule_semicolon":
        prefix = f"The schedule confirms: {temporal_cue}, the {agent}"
        middle = f"{temporal_verb} the report; {iswas_cue}, the {partner}"
        suffix = " prepared."
    elif template == "forecast_while":
        prefix = f"The forecast states that {temporal_cue} the {agent}"
        middle = f"{temporal_verb} the plan, while {iswas_cue} the {partner}"
        suffix = " available."
    else:
        raise ValueError("unknown template")
    text = prefix + temporal_answer + middle + iswas_answer + suffix
    ids = ENCODING.encode(text)
    temporal_position = len(ENCODING.encode(prefix))
    iswas_position = len(ENCODING.encode(prefix + temporal_answer + middle))
    temporal_id, iswas_id = _single(temporal_answer), _single(iswas_answer)
    if ids[temporal_position] != temporal_id or ids[iswas_position] != iswas_id:
        raise ValueError("answer token position did not align under full encoding")
    return {"text": text, "ids": ids, "temporal_position": temporal_position,
            "iswas_position": iswas_position, "temporal_answer_id": temporal_id,
            "temporal_foil_id": _single(" had" if temporal_bit == 0 else " will"),
            "iswas_answer_id": iswas_id,
            "iswas_foil_id": _single(" was" if iswas_bit == 0 else " is")}


def _build_unvalidated():
    rows = []
    for group, (agent, partner) in enumerate(zip(AGENTS, PARTNERS)):
        for template in TEMPLATES:
            endpoints = {cell: _render(template, agent, partner, int(cell[0]), int(cell[1]))
                         for cell in CELLS}
            row = {"schema": SCHEMA, "task_id": TASK_ID, "capability_id": CAPABILITY_ID,
                   "phase": "FIT" if group < 8 else "HOLDOUT", "group_number": group,
                   "template_id": template, "agent": agent, "partner": partner,
                   "endpoints": endpoints}
            row["row_id"] = canonical_sha256(row)
            rows.append(row)
    return rows


def validate_rows(rows, *, verify_hash=True):
    materialized = [dict(row) for row in rows]
    if materialized != _build_unvalidated() or len(materialized) != 32:
        raise ValueError("dual-command rows differ from frozen design")
    if len({row["row_id"] for row in materialized}) != 32:
        raise ValueError("dual-command row IDs are not unique")
    counts = Counter((row["phase"], row["template_id"]) for row in materialized)
    if counts != Counter({(phase, template): 8 for phase in ("FIT", "HOLDOUT")
                          for template in TEMPLATES}):
        raise ValueError("phase/template balance changed")
    for row in materialized:
        if set(row["endpoints"]) != set(CELLS):
            raise ValueError("joint cell inventory changed")
        for cell, endpoint in row["endpoints"].items():
            if endpoint["temporal_answer_id"] != _single(TEMPORAL[int(cell[0])][1]):
                raise ValueError("temporal answer/cell mismatch")
            if endpoint["iswas_answer_id"] != _single(ISWAS[int(cell[1])][1]):
                raise ValueError("is-was answer/cell mismatch")
    digest = canonical_sha256(materialized)
    if verify_hash and digest != EXPECTED_AUTHORITY_SHA256:
        raise ValueError(f"dual-command logical hash changed: {digest}")
    return digest


def build_rows():
    rows = _build_unvalidated()
    validate_rows(rows)
    return rows


EXPECTED_AUTHORITY_SHA256 = "05f04d6438a0eb8d503cedeba752cb4196a20f935782b9a6f65f7539e28b0e8e"


if __name__ == "__main__":
    candidate = _build_unvalidated()
    print(validate_rows(candidate, verify_hash=False))
    for cell, endpoint in candidate[0]["endpoints"].items():
        print(cell, endpoint["text"], endpoint["temporal_position"], endpoint["iswas_position"])
