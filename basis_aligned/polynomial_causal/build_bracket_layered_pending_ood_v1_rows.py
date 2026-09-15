#!/usr/bin/env python3
"""Freeze a fresh fifth layered-pending bracket construction without model access."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import tiktoken


HERE = Path(__file__).resolve().parent
OUT = HERE / "BRACKET_LAYERED_PENDING_OOD_V1_ROWS.json"
MARKS = {
    "parenthesis": {"open": "(", "close": ")", "open_id": 357},
    "square": {"open": "[", "close": "]", "open_id": 685},
    "quote": {"open": '"', "close": '"', "open_id": 366},
}
UNORDERED = (("parenthesis", "square"), ("parenthesis", "quote"), ("square", "quote"))
GROUPS = (
    ("planetarium guide", "projection", "constellation", "narration"),
    ("bakery schedule", "starter", "dough", "batch"),
    ("forestry report", "canopy", "watershed", "restoration"),
    ("studio checklist", "canvas", "pigment", "exhibition"),
    ("aviation brief", "runway", "altitude", "approach"),
    ("library circular", "collection", "catalogue", "lending"),
    ("marine chart", "current", "channel", "anchorage"),
    ("festival program", "ensemble", "venue", "finale"),
    ("factory worksheet", "component", "assembly", "inspection"),
    ("ecology lecture", "wetland", "pollinator", "conservation"),
    ("market analysis", "retailer", "demand", "inventory"),
    ("training manual", "procedure", "example", "assessment"),
)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_rows(token):
    answer_ids = {name: token(spec["close"])[0] for name, spec in MARKS.items()}
    rows = []
    for group_index, (document, a, b, c) in enumerate(GROUPS):
        for left, right in UNORDERED:
            middle = next(name for name in MARKS if name not in (left, right))

            def target_text(inner):
                return (f"During the {document}, {{ a prefatory aside continued; afterward "
                        f"{MARKS[middle]['open']} a supplementary passage began, and finally "
                        f"{MARKS[inner]['open']} the {a}, the {b}, and the {c} stayed incomplete")

            base, donor = target_text(left), target_text(right)
            base_ids, donor_ids = token(base), token(donor)
            assert len(base_ids) == len(donor_ids)
            bo = max(i for i, value in enumerate(base_ids) if value == MARKS[left]["open_id"])
            do = max(i for i, value in enumerate(donor_ids) if value == MARKS[right]["open_id"])
            assert bo == do and [i for i, pair in enumerate(zip(base_ids, donor_ids)) if pair[0] != pair[1]] == [bo]
            row = {
                "family_id": "layered_pending_stack_top", "program_role": "target",
                "group_index": group_index, "base_text": base, "donor_text": donor,
                "base_ids": base_ids, "donor_ids": donor_ids,
                "base_answer_id": answer_ids[left], "donor_answer_id": answer_ids[right],
                "base_open_position": bo, "donor_open_position": do,
                "base_type": left, "donor_type": right,
                "outer_type": "brace", "middle_type": middle,
            }
            row["row_id"] = canonical(row)
            rows.append(row)

        for inner in MARKS:
            middles = [name for name in MARKS if name != inner]

            def control_text(middle):
                return (f"During the {document}, {{ a prefatory aside continued; afterward "
                        f"{MARKS[middle]['open']} a supplementary passage began, and finally "
                        f"{MARKS[inner]['open']} the {a}, the {b}, and the {c} stayed incomplete")

            base, donor = control_text(middles[0]), control_text(middles[1])
            base_ids, donor_ids = token(base), token(donor)
            assert len(base_ids) == len(donor_ids)
            bo = max(i for i, value in enumerate(base_ids) if value == MARKS[inner]["open_id"])
            do = max(i for i, value in enumerate(donor_ids) if value == MARKS[inner]["open_id"])
            assert bo == do and len([i for i, pair in enumerate(zip(base_ids, donor_ids)) if pair[0] != pair[1]]) == 1
            row = {
                "family_id": "layered_middle_change_inner_fixed", "program_role": "control",
                "group_index": group_index, "base_text": base, "donor_text": donor,
                "base_ids": base_ids, "donor_ids": donor_ids,
                "base_answer_id": answer_ids[inner], "donor_answer_id": answer_ids[inner],
                "base_open_position": bo, "donor_open_position": do,
                "inner_type": inner, "outer_type": "brace",
                "base_middle_type": middles[0], "donor_middle_type": middles[1],
            }
            row["row_id"] = canonical(row)
            rows.append(row)
    return rows


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = build_rows(tiktoken.get_encoding("gpt2").encode)
    counts = Counter(row["program_role"] for row in rows)
    pairs = Counter()
    for row in rows:
        if row["program_role"] == "target":
            pairs[(row["base_answer_id"], row["donor_answer_id"])] += 1
            pairs[(row["donor_answer_id"], row["base_answer_id"])] += 1
    assert len(rows) == 72 and counts == {"target": 36, "control": 36}
    assert len(pairs) == 6 and set(pairs.values()) == {12}
    payload = {
        "schema": "bracket_layered_pending_ood_v1_rows", "model_loaded": False,
        "outcomes_opened": [], "row_count": len(rows), "endpoint_count": 2 * len(rows),
        "counts": dict(counts),
        "ordered_pair_counts": {f"{a}->{b}": n for (a, b), n in sorted(pairs.items())},
        "rows": rows,
    }
    payload["row_manifest_sha256"] = canonical(rows)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "row_count", "endpoint_count", "counts", "ordered_pair_counts", "row_manifest_sha256"
    )}, indent=2))


if __name__ == "__main__":
    main()
