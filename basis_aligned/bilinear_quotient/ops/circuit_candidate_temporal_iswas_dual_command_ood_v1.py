#!/usr/bin/env python3
"""Fresh-template/lexicon dual-command authority for H4 OOD confirmation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import circuit_candidate_temporal_auxiliary_fresh_cues_v12 as temporal
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v16 as iswas

SCHEMA = "temporal_iswas_dual_command_ood_authority_v1"
TASK_ID = "cross_task.temporal_iswas.dual_command_ood_v1"
CAPABILITY_ID = "cross_task.temporal_iswas.dual_command_ood_native_capability_v1"
CELLS = ("00", "10", "01", "11")
TEMPLATES = ("registry_right_now", "field_these_days")
ENCODING = temporal.lex.ENCODING


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _single(text):
    ids = ENCODING.encode(text)
    if len(ids) != 1: raise ValueError(f"expected one token for {text!r}: {ids}")
    return ids[0]


def _by_group(rows, panel):
    return {row["group_number"]: row for row in rows if row["transform_id"] == panel}


def _side(row, answer):
    for side in ("base", "donor"):
        if row[f"{side}_answer"] == answer: return row[f"{side}_text"]
    raise ValueError("requested answer absent")


def _render(temporal_row, iswas_row, temporal_bit, iswas_bit):
    temporal_answer = " will" if temporal_bit == 0 else " had"
    iswas_answer = " is" if iswas_bit == 0 else " was"
    temporal_prefix = _side(temporal_row, temporal_answer)
    iswas_prefix = _side(iswas_row, iswas_answer)
    bridge = " complete the audit. " if temporal_bit == 0 else " completed the audit. "
    text = temporal_prefix + temporal_answer + bridge + iswas_prefix + iswas_answer + " prepared."
    ids = ENCODING.encode(text)
    temporal_position = len(ENCODING.encode(temporal_prefix))
    iswas_position = len(ENCODING.encode(temporal_prefix + temporal_answer + bridge + iswas_prefix))
    if ids[temporal_position] != _single(temporal_answer) or ids[iswas_position] != _single(iswas_answer):
        raise ValueError("answer positions changed under complete encoding")
    if ids[:temporal_position] != ENCODING.encode(temporal_prefix):
        raise ValueError("fresh temporal prefix was not preserved exactly")
    return {"text": text, "ids": ids, "temporal_prefix": temporal_prefix,
            "temporal_prefix_ids": ids[:temporal_position],
            "temporal_position": temporal_position, "iswas_position": iswas_position,
            "temporal_answer_id": _single(temporal_answer),
            "temporal_foil_id": _single(" had" if temporal_bit == 0 else " will"),
            "iswas_answer_id": _single(iswas_answer),
            "iswas_foil_id": _single(" was" if iswas_bit == 0 else " is")}


def _build_unvalidated():
    temporal_rows = temporal.build_rows(groups=16)
    iswas_rows = iswas.build_rows()
    pairings = {
        "registry_right_now": (_by_group(temporal_rows, "A1"), _by_group(iswas_rows, "A1")),
        "field_these_days": (_by_group(temporal_rows, "A2"), _by_group(iswas_rows, "A2")),
    }
    rows = []
    for group in range(16):
        for template, (temporal_bank, iswas_bank) in pairings.items():
            endpoints = {cell: _render(temporal_bank[group], iswas_bank[group],
                                       int(cell[0]), int(cell[1])) for cell in CELLS}
            row = {"schema": SCHEMA, "task_id": TASK_ID, "capability_id": CAPABILITY_ID,
                   "phase": "FIT" if group < 8 else "HOLDOUT", "group_number": group,
                   "template_id": template,
                   "temporal_source_row_id": temporal_bank[group]["row_id"],
                   "iswas_source_row_id": iswas_bank[group]["row_id"],
                   "endpoints": endpoints}
            row["row_id"] = canonical_sha256(row); rows.append(row)
    return rows


def validate_rows(rows, *, verify_hash=True):
    materialized = [dict(row) for row in rows]
    if materialized != _build_unvalidated() or len(materialized) != 32:
        raise ValueError("OOD rows differ from frozen construction")
    if len({row["row_id"] for row in materialized}) != 32:
        raise ValueError("OOD row IDs are not unique")
    counts = Counter((row["phase"], row["template_id"]) for row in materialized)
    if counts != Counter({(phase, template): 8 for phase in ("FIT", "HOLDOUT")
                          for template in TEMPLATES}):
        raise ValueError("OOD phase/template balance changed")
    for row in materialized:
        if set(row["endpoints"]) != set(CELLS): raise ValueError("joint cells changed")
        lengths = {len(endpoint["ids"]) for endpoint in row["endpoints"].values()}
        temporal_positions = {endpoint["temporal_position"] for endpoint in row["endpoints"].values()}
        iswas_positions = {endpoint["iswas_position"] for endpoint in row["endpoints"].values()}
        if len(lengths) != 1 or len(temporal_positions) != 1 or len(iswas_positions) != 1:
            raise ValueError("OOD paired alignment changed")
        for endpoint in row["endpoints"].values():
            if endpoint["ids"][:endpoint["temporal_position"]] != endpoint["temporal_prefix_ids"]:
                raise ValueError("OOD temporal prefix identity failed")
    digest = canonical_sha256(materialized)
    if verify_hash and digest != EXPECTED_AUTHORITY_SHA256:
        raise ValueError(f"OOD logical authority changed: {digest}")
    return digest


def build_rows():
    rows = _build_unvalidated(); validate_rows(rows); return rows


EXPECTED_AUTHORITY_SHA256 = "19cc9aebdb7c2b0cef1b93ea889056f721423d4e15ff37bc2c0a5beae285e591"


if __name__ == "__main__":
    rows = _build_unvalidated(); print(validate_rows(rows, verify_hash=False))
    for cell, endpoint in rows[0]["endpoints"].items():
        print(cell, endpoint["text"], endpoint["temporal_position"], endpoint["iswas_position"])
