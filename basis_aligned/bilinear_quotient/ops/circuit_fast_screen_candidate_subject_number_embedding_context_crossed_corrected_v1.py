#!/usr/bin/env python3
"""Corrected crossed authority with verified opposite-number noun attractors."""
from __future__ import annotations

import hashlib
import json

import circuit_battery_task14 as task14
import circuit_fast_screen_candidate_subject_number_embedding_removal_fresh as removal


SCHEMA = "subject_number_embedding_context_crossed_corrected_authority_v1"
PAIRS = removal.PAIRS
TEMPLATES = (
    ("near", "Near the {attractor}, the {subject}"),
    ("behind", "Behind the {attractor}, the {subject}"),
    ("under", "Under the {attractor}, the {subject}"),
    ("above", "Above the {attractor}, the {subject}"),
)
EXPECTED_AUTHORITY_SHA256 = "cf100fa31eb6131df23837b8da54328676a3e22e81c5922157059047f1b80886"


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def build_rows(verify_hash=True):
    encoding = task14.ENCODING
    answer_ids = {"singular": encoding.encode(" is")[0],
                  "plural": encoding.encode(" are")[0]}
    rows = []
    for pair_index, (singular, plural, stratum) in enumerate(PAIRS):
        attractor_pair = PAIRS[(pair_index + 7) % len(PAIRS)]
        for number_index, (number, subject) in enumerate((("singular", singular),
                                                          ("plural", plural))):
            attractor_index = 1 - number_index
            attractor = attractor_pair[attractor_index]
            attractor_number = "singular" if attractor_index == 0 else "plural"
            if attractor_number == number or attractor in ("regular", "irregular"):
                raise ValueError("attractor is not an opposite-number noun")
            for context_index, (template_id, template) in enumerate(TEMPLATES):
                text = template.format(attractor=attractor, subject=subject)
                token_ids = encoding.encode(text)
                subject_id, attractor_id = encoding.encode(" " + subject), encoding.encode(" " + attractor)
                if len(subject_id) != 1 or len(attractor_id) != 1 \
                        or token_ids[-1] != subject_id[0] \
                        or token_ids[2] != attractor_id[0]:
                    raise ValueError("noun atomicity or position failed")
                answer_id = answer_ids[number]
                if encoding.encode(text + (" is" if number == "singular" else " are")) \
                        != token_ids + [answer_id]:
                    raise ValueError("answer continuation failed")
                subject_index = 2 * pair_index + number_index
                identity = [SCHEMA, subject_index, template_id, text, token_ids]
                rows.append({"schema": SCHEMA, "row_id": canonical(identity),
                             "pair_index": pair_index, "subject_index": subject_index,
                             "context_index": context_index, "stratum": stratum,
                             "template_id": template_id, "number": number,
                             "subject": subject, "attractor": attractor,
                             "attractor_number": attractor_number, "text": text,
                             "token_ids": token_ids, "subject_position": len(token_ids) - 1,
                             "native_answer_id": answer_id})
    if len(rows) != 128 or len({row["row_id"] for row in rows}) != 128 \
            or len({len(row["token_ids"]) for row in rows}) != 1 \
            or any(row["number"] == row["attractor_number"] for row in rows):
        raise ValueError("authority invariant failed")
    for subject_index in range(32):
        subset = [row for row in rows if row["subject_index"] == subject_index]
        if [row["template_id"] for row in subset] != [name for name, _ in TEMPLATES] \
                or len({row["subject"] for row in subset}) != 1 \
                or len({row["attractor"] for row in subset}) != 1:
            raise ValueError("crossed-factor invariant failed")
    digest = canonical(rows)
    if verify_hash and EXPECTED_AUTHORITY_SHA256 != "TO_BE_FROZEN" \
            and digest != EXPECTED_AUTHORITY_SHA256:
        raise ValueError("authority changed")
    return rows


def compile_plan():
    rows = build_rows()
    return {"schema": SCHEMA, "rows": len(rows), "subjects": 32,
            "pairs": len(PAIRS), "contexts": [name for name, _ in TEMPLATES],
            "sequence_length": len(rows[0]["token_ids"]),
            "opposite_number_attractor_rows": sum(row["number"] != row["attractor_number"]
                                                   for row in rows),
            "authority_sha256": canonical(rows), "model_loaded": False,
            "behavior_opened": False}


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
