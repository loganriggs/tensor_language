#!/usr/bin/env python3
"""Freeze a fresh two-clause authority for subject-number axis composition."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OPS = HERE.parent / "bilinear_quotient/ops"
sys.path.insert(0, str(OPS))
import circuit_battery_task14 as task14  # noqa: E402

OUT = HERE / "SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V2_ROWS.json"
SCHEMA = "subject_number_two_site_composition_v2_rows"
NOUN_PAIRS = (
    ("manager", "managers"), ("sailor", "sailors"),
    ("hunter", "hunters"), ("owner", "owners"),
    ("leader", "leaders"), ("player", "players"),
    ("writer", "writers"), ("buyer", "buyers"),
    ("seller", "sellers"), ("maker", "makers"),
    ("runner", "runners"), ("rider", "riders"),
    ("voter", "voters"), ("clerk", "clerks"),
    ("judge", "judges"), ("nurse", "nurses"),
)

TEMPLATES = (
    ("above_below", "Above the {a1}, the {s1} {v1} ready; below the {a2}, the {s2}"),
    ("across_outside", "Across the {a1}, the {s1} {v1} waiting; outside the {a2}, the {s2}"),
)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main():
    encoding = task14.ENCODING
    for pair in NOUN_PAIRS:
        for word in pair:
            if len(encoding.encode(" " + word)) != 1:
                raise RuntimeError(f"noun is not one token: {word}")
    answer_ids = {"singular": encoding.encode(" is")[0], "plural": encoding.encode(" are")[0]}
    if any(len(encoding.encode(text)) != 1 for text in (" is", " are", " can", " will")):
        raise RuntimeError("answer or control token is not atomic")

    rows = []
    for group in range(8):
        number1, number2 = (0, 0)
        for template_id, template in TEMPLATES:
            pair1 = NOUN_PAIRS[2 * group]
            pair2 = NOUN_PAIRS[2 * group + 1]
            a1_pair = NOUN_PAIRS[(2 * group + 5) % len(NOUN_PAIRS)]
            a2_pair = NOUN_PAIRS[(2 * group + 11) % len(NOUN_PAIRS)]
            s1, s2 = pair1[number1], pair2[number2]
            a1, a2 = a1_pair[1 - number1], a2_pair[1 - number2]
            v1 = "is" if number1 == 0 else "are"
            text = template.format(a1=a1, s1=s1, v1=v1, a2=a2, s2=s2)
            ids = encoding.encode(text)
            prefix1 = template.split("{s1}", 1)[0].format(a1=a1)
            site1 = len(encoding.encode(prefix1 + s1)) - 1
            site2 = len(ids) - 1
            if ids[site1] != encoding.encode(" " + s1)[0] or ids[site2] != encoding.encode(" " + s2)[0]:
                raise RuntimeError("subject position mismatch")
            if ids[site1 + 1] != answer_ids["singular" if number1 == 0 else "plural"]:
                raise RuntimeError("first-clause verb mismatch")
            expected2 = answer_ids["singular" if number2 == 0 else "plural"]
            if encoding.encode(text + (" is" if number2 == 0 else " are")) != ids + [expected2]:
                raise RuntimeError("second-clause continuation mismatch")
            sites = []
            for site, number, subject in ((site1, number1, s1), (site2, number2, s2)):
                native = "singular" if number == 0 else "plural"
                opposite = "plural" if number == 0 else "singular"
                sites.append({
                    "position": site,
                    "subject": subject,
                    "native_number": native,
                    "direction": f"{native}_to_{opposite}",
                    "native_answer_id": answer_ids[native],
                    "opposite_answer_id": answer_ids[opposite],
                })
            row = {
                "schema": SCHEMA,
                "row_id": canonical([SCHEMA, group, template_id, text, sites]),
                "group": group,
                "template_id": template_id,
                "number_pair": [sites[0]["native_number"], sites[1]["native_number"]],
                "text": text,
                "token_ids": ids,
                "sites": sites,
                "control_token_ids": {"can": encoding.encode(" can")[0], "will": encoding.encode(" will")[0]},
            }
            rows.append(row)

    payload = {
        "schema": SCHEMA,
        "selection_rule": "eight fresh noun-pair groups with the independently strong singular-to-plural direction at both sites, crossed with two fresh fixed two-clause templates; rank-one axis and cardinality-four coefficient fixed independently of rows",
        "outcomes_opened": False,
        "model_loaded": False,
        "row_count": len(rows),
        "site_count": 2 * len(rows),
        "templates": [item[0] for item in TEMPLATES],
        "number_pair_counts": {
            key: sum("|".join(row["number_pair"]) == key for row in rows)
            for key in ("singular|singular", "singular|plural", "plural|singular", "plural|plural")
        },
        "rows": rows,
    }
    payload["row_manifest_sha256"] = canonical(rows)
    payload["builder_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(OUT), "rows": len(rows), "manifest": payload["row_manifest_sha256"], "counts": payload["number_pair_counts"]}, indent=2))


if __name__ == "__main__":
    main()
