#!/usr/bin/env python3
"""Freeze a third, nested-pending construction without loading the model."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import tiktoken


HERE = Path(__file__).resolve().parent
OUT = HERE / "BRACKET_NESTED_PENDING_OOD_V1_ROWS.json"
MARKS = {
    "parenthesis": {"open": "(", "close": ")"},
    "square": {"open": "[", "close": "]"},
    "quote": {"open": '"', "close": '"'},
}
UNORDERED = (("parenthesis", "square"), ("parenthesis", "quote"), ("square", "quote"))
GROUPS = (
    ("laboratory notebook", "sample", "estimate", "conclusion"),
    ("museum catalogue", "portrait", "archive", "annotation"),
    ("shipping manifest", "parcel", "invoice", "destination"),
    ("court transcript", "witness", "exhibit", "finding"),
    ("field journal", "species", "habitat", "observation"),
    ("engineering memo", "sensor", "threshold", "calibration"),
    ("medical record", "patient", "dosage", "assessment"),
    ("music review", "movement", "motif", "cadence"),
    ("budget report", "revenue", "expense", "forecast"),
    ("travel diary", "station", "route", "arrival"),
    ("recipe draft", "ingredient", "mixture", "serving"),
    ("research abstract", "premise", "method", "result"),
)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    tok = tiktoken.get_encoding("gpt2")
    token = tok.encode
    answer_ids = {name: token(spec["close"])[0] for name, spec in MARKS.items()}
    opener_ids = {"parenthesis": 357, "square": 685, "quote": 366}
    assert all(len(token(spec["open"])) == len(token(spec["close"])) == 1 for spec in MARKS.values())
    rows = []
    for group_index, (document, a, b, c) in enumerate(GROUPS):
        for left, right in UNORDERED:
            outer = next(name for name in MARKS if name not in (left, right))
            def target_text(inner):
                return (f"The {document} began {MARKS[outer]['open']} a preliminary section, "
                        f"then nested {MARKS[inner]['open']} the {a}, the {b}, and the {c} remained unfinished")
            base, donor = target_text(left), target_text(right)
            base_ids, donor_ids = token(base), token(donor)
            assert len(base_ids) == len(donor_ids)
            bo = max(i for i, value in enumerate(base_ids) if value == opener_ids[left])
            do = max(i for i, value in enumerate(donor_ids) if value == opener_ids[right])
            assert bo == do and bo > 0
            row = {
                "family_id": "nested_two_pending_stack_top",
                "program_role": "target",
                "group_index": group_index,
                "base_text": base, "donor_text": donor,
                "base_ids": base_ids, "donor_ids": donor_ids,
                "base_answer_id": answer_ids[left], "donor_answer_id": answer_ids[right],
                "base_open_position": bo, "donor_open_position": do,
                "base_type": left, "donor_type": right, "outer_type": outer,
            }
            row["row_id"] = canonical(row)
            rows.append(row)

        for inner in MARKS:
            outers = [name for name in MARKS if name != inner]
            def control_text(outer):
                return (f"The {document} began {MARKS[outer]['open']} a preliminary section, "
                        f"then nested {MARKS[inner]['open']} the {a}, the {b}, and the {c} remained unfinished")
            base, donor = control_text(outers[0]), control_text(outers[1])
            base_ids, donor_ids = token(base), token(donor)
            assert len(base_ids) == len(donor_ids)
            bo = max(i for i, value in enumerate(base_ids) if value == opener_ids[inner])
            do = max(i for i, value in enumerate(donor_ids) if value == opener_ids[inner])
            assert bo == do and bo > 0
            row = {
                "family_id": "outer_pending_type_change_inner_fixed",
                "program_role": "control",
                "group_index": group_index,
                "base_text": base, "donor_text": donor,
                "base_ids": base_ids, "donor_ids": donor_ids,
                "base_answer_id": answer_ids[inner], "donor_answer_id": answer_ids[inner],
                "base_open_position": bo, "donor_open_position": do,
                "inner_type": inner, "base_outer_type": outers[0], "donor_outer_type": outers[1],
            }
            row["row_id"] = canonical(row)
            rows.append(row)
    counts = Counter(row["program_role"] for row in rows)
    pair_counts = Counter()
    for row in rows:
        if row["program_role"] == "target":
            pair_counts[(row["base_answer_id"], row["donor_answer_id"])] += 1
            pair_counts[(row["donor_answer_id"], row["base_answer_id"])] += 1
    assert len(rows) == 72 and counts == {"target": 36, "control": 36}
    assert len(pair_counts) == 6 and set(pair_counts.values()) == {12}
    payload = {
        "schema": "bracket_nested_pending_ood_v1_rows",
        "model_loaded": False,
        "outcomes_opened": [],
        "row_count": len(rows),
        "endpoint_count": 2 * len(rows),
        "counts": dict(counts),
        "ordered_pair_counts": {f"{a}->{b}": n for (a, b), n in sorted(pair_counts.items())},
        "rows": rows,
    }
    payload["row_manifest_sha256"] = canonical(rows)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("row_count", "endpoint_count", "counts", "ordered_pair_counts", "row_manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
