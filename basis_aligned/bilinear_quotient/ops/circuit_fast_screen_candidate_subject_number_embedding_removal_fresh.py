#!/usr/bin/env python3
"""Prospective fresh-text authority for embedding-number coordinate removal."""
from __future__ import annotations

import hashlib
import json

import circuit_battery_task14 as task14
import circuit_fast_screen_candidate_subject_number_embedding_decoder_fresh as fresh
import run_subject_number_embedding_decoder_v1 as discovery


SCHEMA = "subject_number_embedding_removal_fresh_authority_v1"
PAIRS = (
    ("fungus", "fungi", "irregular"),
    ("analysis", "analyses", "irregular"),
    ("crisis", "crises", "irregular"),
    ("phenomenon", "phenomena", "irregular"),
    ("wolf", "wolves", "irregular"),
    ("knife", "knives", "irregular"),
    ("wife", "wives", "irregular"),
    ("shelf", "shelves", "irregular"),
    ("musician", "musicians", "regular"),
    ("surgeon", "surgeons", "regular"),
    ("expert", "experts", "regular"),
    ("tourist", "tourists", "regular"),
    ("architect", "architects", "regular"),
    ("priest", "priests", "regular"),
    ("artist", "artists", "regular"),
    ("visitor", "visitors", "regular"),
)
TEMPLATES = (
    ("near", "Near the {attractor}, the {subject}"),
    ("behind", "Behind the {attractor}, the {subject}"),
)
EXPECTED_AUTHORITY_SHA256 = "fe373615d98fcf85bdb58bd257db626ea94a2b45a08129b8ac6a453b867937d8"


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def build_rows(verify_hash=True):
    encoding = task14.ENCODING
    opened = {word for pairs in discovery.FAMILIES.values() for pair in pairs for word in pair}
    opened |= {word for singular, plural, _ in fresh.PAIRS for word in (singular, plural)}
    if any(word in opened for singular, plural, _ in PAIRS for word in (singular, plural)):
        raise ValueError("removal vocabulary overlaps decoder authorities")
    answer_ids = {"singular": encoding.encode(" is")[0],
                  "plural": encoding.encode(" are")[0]}
    rows = []
    for pair_index, (singular, plural, stratum) in enumerate(PAIRS):
        attractor_pair = PAIRS[(pair_index + 7) % len(PAIRS)]
        for number_index, (number, subject) in enumerate((("singular", singular),
                                                          ("plural", plural))):
            attractor = attractor_pair[2 - number_index]
            for template_id, template in TEMPLATES:
                text = template.format(attractor=attractor, subject=subject)
                token_ids = encoding.encode(text)
                subject_id = encoding.encode(" " + subject)
                attractor_id = encoding.encode(" " + attractor)
                if len(subject_id) != 1 or len(attractor_id) != 1 \
                        or token_ids[-1] != subject_id[0]:
                    raise ValueError("noun atomicity or subject position failed")
                answer_id = answer_ids[number]
                opposite = "plural" if number == "singular" else "singular"
                if encoding.encode(text + (" is" if number == "singular" else " are")) \
                        != token_ids + [answer_id]:
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
                    "control_token_ids": {"can": encoding.encode(" can")[0],
                                          "will": encoding.encode(" will")[0]},
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
        "schema": SCHEMA, "rows": len(rows),
        "pairs": len(PAIRS), "sequence_length": len(rows[0]["token_ids"]),
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
