#!/usr/bin/env python3
"""Fourth disjoint lexical authority for the small L11H3 mediation hypothesis."""
from __future__ import annotations

import hashlib
import json

import circuit_battery_task14 as task14
import circuit_fast_screen_candidate_subject_number_embedding_decoder_fresh as decoder_fresh
import circuit_fast_screen_candidate_subject_number_embedding_removal_fresh as removal
import run_subject_number_embedding_decoder_v1 as discovery


SCHEMA = "subject_number_embedding_mediation_fresh_authority_v1"
PAIRS = (
    ("foot", "feet", "irregular"),
    ("tooth", "teeth", "irregular"),
    ("thief", "thieves", "irregular"),
    ("leaf", "leaves", "irregular"),
    ("calf", "calves", "irregular"),
    ("half", "halves", "irregular"),
    ("life", "lives", "irregular"),
    ("elf", "elves", "irregular"),
    ("engineer", "engineers", "regular"),
    ("professor", "professors", "regular"),
    ("researcher", "researchers", "regular"),
    ("scientist", "scientists", "regular"),
    ("worker", "workers", "regular"),
    ("student", "students", "regular"),
    ("soldier", "soldiers", "regular"),
    ("director", "directors", "regular"),
)
TEMPLATES = (
    ("under", "Under the {attractor}, the {subject}"),
    ("above", "Above the {attractor}, the {subject}"),
)
EXPECTED_AUTHORITY_SHA256 = "6cb0416fc007a0870174a4d3022a969cb1d510bd1c904735382522b374d2c50c"


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def build_rows(verify_hash=True):
    encoding = task14.ENCODING
    opened = {word for pairs in discovery.FAMILIES.values() for pair in pairs for word in pair}
    opened |= {word for singular, plural, _ in decoder_fresh.PAIRS
               for word in (singular, plural)}
    opened |= {word for singular, plural, _ in removal.PAIRS
               for word in (singular, plural)}
    if any(word in opened for singular, plural, _ in PAIRS for word in (singular, plural)):
        raise ValueError("mediation vocabulary overlaps earlier authorities")
    answer_ids = {"singular": encoding.encode(" is")[0],
                  "plural": encoding.encode(" are")[0]}
    rows = []
    for pair_index, (singular, plural, stratum) in enumerate(PAIRS):
        attractor_pair = PAIRS[(pair_index + 5) % len(PAIRS)]
        for number_index, (number, subject) in enumerate((("singular", singular),
                                                          ("plural", plural))):
            attractor = attractor_pair[2 - number_index]
            for template_id, template in TEMPLATES:
                text = template.format(attractor=attractor, subject=subject)
                token_ids = encoding.encode(text)
                if len(encoding.encode(" " + subject)) != 1 \
                        or len(encoding.encode(" " + attractor)) != 1 \
                        or token_ids[-1] != encoding.encode(" " + subject)[0]:
                    raise ValueError("noun atomicity or subject position failed")
                opposite = "plural" if number == "singular" else "singular"
                answer_id = answer_ids[number]
                suffix = " is" if number == "singular" else " are"
                if encoding.encode(text + suffix) != token_ids + [answer_id]:
                    raise ValueError("answer continuation failed")
                identity = [SCHEMA, pair_index, number, template_id, text, token_ids]
                rows.append({
                    "schema": SCHEMA, "row_id": canonical(identity),
                    "pair_index": pair_index, "stratum": stratum,
                    "template_id": template_id, "number": number,
                    "subject": subject, "attractor": attractor, "text": text,
                    "token_ids": token_ids, "subject_position": len(token_ids) - 1,
                    "native_answer_id": answer_id,
                    "opposite_answer_id": answer_ids[opposite],
                })
    if len(rows) != 64 or len({row["row_id"] for row in rows}) != 64 \
            or len({len(row["token_ids"]) for row in rows}) != 1:
        raise ValueError("authority cardinality or equal length failed")
    digest = canonical(rows)
    if verify_hash and EXPECTED_AUTHORITY_SHA256 != "TO_BE_FROZEN" \
            and digest != EXPECTED_AUTHORITY_SHA256:
        raise ValueError("authority changed")
    return rows


def compile_plan():
    rows = build_rows()
    return {
        "schema": SCHEMA, "rows": len(rows), "pairs": len(PAIRS),
        "sequence_length": len(rows[0]["token_ids"]),
        "strata": {name: sum(row["stratum"] == name for row in rows)
                   for name in ("irregular", "regular")},
        "numbers": {name: sum(row["number"] == name for row in rows)
                    for name in ("singular", "plural")},
        "templates": [name for name, _ in TEMPLATES],
        "authority_sha256": canonical(rows),
        "model_loaded": False, "behavior_opened": False,
    }


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
