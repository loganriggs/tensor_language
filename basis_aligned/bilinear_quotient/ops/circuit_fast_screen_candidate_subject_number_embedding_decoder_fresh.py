#!/usr/bin/env python3
"""Frozen post-decoder lexical authority for native embedding-number OOD."""
from __future__ import annotations

import hashlib
import json

import circuit_battery_task14 as task14
import run_subject_number_embedding_decoder_v1 as discovery


SCHEMA = "subject_number_embedding_decoder_fresh_authority_v1"
DECODER_SHA256 = "f2356b8a1c336d011e36059e571117a0d61c4104a3570521bd59838a29f40cc1"
PAIRS = (
    ("man", "men", "irregular"),
    ("woman", "women", "irregular"),
    ("child", "children", "irregular"),
    ("person", "people", "irregular"),
    ("mouse", "mice", "irregular"),
    ("glass", "glasses", "irregular"),
    ("doctor", "doctors", "regular"),
    ("teacher", "teachers", "regular"),
    ("farmer", "farmers", "regular"),
    ("lawyer", "lawyers", "regular"),
    ("poet", "poets", "regular"),
    ("dancer", "dancers", "regular"),
    ("singer", "singers", "regular"),
    ("driver", "drivers", "regular"),
    ("guard", "guards", "regular"),
    ("captain", "captains", "regular"),
)
EXPECTED_AUTHORITY_SHA256 = "13a69cbe042382ff572e75019bb3981cda9ec7335747bff510d9b196309d849c"


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def build_rows():
    training = {word for pairs in discovery.FAMILIES.values() for pair in pairs for word in pair}
    rows = []
    for pair_index, (singular, plural, stratum) in enumerate(PAIRS):
        if singular in training or plural in training:
            raise ValueError("fresh form overlaps decoder discovery")
        token_ids = []
        for word in (singular, plural):
            ids = task14.ENCODING.encode(" " + word)
            if len(ids) != 1:
                raise ValueError(f"form is not one token: {word}")
            token_ids.append(ids[0])
        rows.append({"schema": SCHEMA, "pair_index": pair_index,
                     "singular": singular, "plural": plural, "stratum": stratum,
                     "singular_token_id": token_ids[0], "plural_token_id": token_ids[1],
                     "row_id": canonical([SCHEMA, pair_index, singular, plural, stratum, token_ids])})
    if len(rows) != 16 or len({row["row_id"] for row in rows}) != 16 \
            or len({row["singular_token_id"] for row in rows}
                   | {row["plural_token_id"] for row in rows}) != 32:
        raise ValueError("authority cardinality failed")
    digest = canonical(rows)
    if EXPECTED_AUTHORITY_SHA256 != "TO_BE_FROZEN" and digest != EXPECTED_AUTHORITY_SHA256:
        raise ValueError(f"authority changed: {digest}")
    return rows


def compile_plan():
    rows = build_rows()
    return {"schema": SCHEMA, "decoder_sha256": DECODER_SHA256,
            "rows": len(rows), "forms": 2 * len(rows),
            "strata": {name: sum(row["stratum"] == name for row in rows)
                        for name in ("irregular", "regular")},
            "authority_sha256": canonical(rows), "model_loaded": False,
            "decoder_scores_opened": False}


if __name__ == "__main__":
    print(json.dumps(compile_plan(), indent=2, sort_keys=True))
