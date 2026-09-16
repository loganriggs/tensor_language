#!/usr/bin/env python3
"""Freeze prospective two-site prompts for the token-decoded subject writer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent / "bilinear_quotient/ops"
sys.path.insert(0, str(OPS))
import circuit_battery_task14 as task14  # noqa: E402
import circuit_fast_screen_candidate_subject_number_embedding_decoder_fresh as lexical  # noqa: E402


OUT = HERE / "SUBJECT_NUMBER_EMBEDDING_WRITER_TWO_SITE_V1_ROWS.json"
SCHEMA = "subject_number_embedding_writer_two_site_v1_rows"
DECODER_RESULT_SHA256 = "d151c1b47eccbf29e3b5ebac7ade8c7090029f1f0ee56ee2a28b49e0bd3fba12"
TEMPLATES = (
    ("behind_before", "Behind the {a1}, the {s1} {v1} calm; before the {a2}, the {s2}"),
    ("along_beneath", "Along the {a1}, the {s1} {v1} ready; beneath the {a2}, the {s2}"),
)
EXPECTED_ROW_MANIFEST_SHA256 = "016e2a28af3c9cb38ff31a4eefc2ce82c89e63f2a5897e7839e39a14e26972bc"


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def build_rows(verify_hash=True):
    encoding = task14.ENCODING
    pairs = [(singular, plural) for singular, plural, _ in lexical.PAIRS]
    answer_ids = {"singular": encoding.encode(" is")[0],
                  "plural": encoding.encode(" are")[0]}
    rows = []
    for group in range(8):
        number1, number2 = ((0, 0), (0, 1), (1, 0), (1, 1))[group % 4]
        pair1, pair2 = pairs[2 * group], pairs[2 * group + 1]
        a1_pair, a2_pair = pairs[(2 * group + 5) % len(pairs)], pairs[(2 * group + 11) % len(pairs)]
        for template_id, template in TEMPLATES:
            s1, s2 = pair1[number1], pair2[number2]
            a1, a2 = a1_pair[1 - number1], a2_pair[1 - number2]
            v1 = "is" if number1 == 0 else "are"
            text = template.format(a1=a1, s1=s1, v1=v1, a2=a2, s2=s2)
            ids = encoding.encode(text)
            prefix1 = template.split("{s1}", 1)[0].format(a1=a1)
            position1 = len(encoding.encode(prefix1 + s1)) - 1
            position2 = len(ids) - 1
            sites = []
            for position, number, subject in ((position1, number1, s1),
                                               (position2, number2, s2)):
                native = "singular" if number == 0 else "plural"
                opposite = "plural" if number == 0 else "singular"
                if ids[position] != encoding.encode(" " + subject)[0]:
                    raise ValueError("subject position mismatch")
                sites.append({"position": position, "subject": subject,
                              "native_number": native,
                              "direction": f"{native}_to_{opposite}",
                              "native_answer_id": answer_ids[native],
                              "opposite_answer_id": answer_ids[opposite]})
            if ids[position1 + 1] != sites[0]["native_answer_id"]:
                raise ValueError("first continuation mismatch")
            suffix = " is" if number2 == 0 else " are"
            if encoding.encode(text + suffix) != ids + [sites[1]["native_answer_id"]]:
                raise ValueError("second continuation mismatch")
            rows.append({"schema": SCHEMA,
                         "row_id": canonical([SCHEMA, group, template_id, text, sites]),
                         "group": group, "template_id": template_id,
                         "number_pair": [sites[0]["native_number"], sites[1]["native_number"]],
                         "text": text, "token_ids": ids, "sites": sites,
                         "control_token_ids": {"can": encoding.encode(" can")[0],
                                               "will": encoding.encode(" will")[0]}})
    if len(rows) != 16 or len({len(row["token_ids"]) for row in rows}) != 1:
        raise ValueError("row count or equal-length batch failed")
    manifest = canonical(rows)
    if verify_hash and EXPECTED_ROW_MANIFEST_SHA256 != "TO_BE_FROZEN" \
            and manifest != EXPECTED_ROW_MANIFEST_SHA256:
        raise ValueError("row manifest changed")
    return rows


def compile_plan():
    rows = build_rows()
    return {"schema": SCHEMA, "rows": len(rows), "sites": 2 * len(rows),
            "sequence_length": len(rows[0]["token_ids"]),
            "templates": [item[0] for item in TEMPLATES],
            "number_pair_counts": {key: sum("|".join(row["number_pair"]) == key for row in rows)
                                   for key in ("singular|singular", "singular|plural",
                                               "plural|singular", "plural|plural")},
            "row_manifest_sha256": canonical(rows),
            "decoder_result_sha256": DECODER_RESULT_SHA256,
            "model_loaded": False, "behavior_opened": False}


def main():
    plan = compile_plan()
    rows = build_rows()
    payload = {
        **plan,
        "selection_rule": (
            "all sixteen noun pairs from the prospectively frozen decoder authority, "
            "assigned before behavioral outcomes to eight groups crossing all four "
            "number pairs twice and two fixed two-clause templates"
        ),
        "rows": rows,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(OUT), **plan}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
