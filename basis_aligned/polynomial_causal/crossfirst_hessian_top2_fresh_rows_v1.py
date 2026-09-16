#!/usr/bin/env python3
"""Build a score-blind fresh panel for the frozen CrossFirst Hessian top-two rule."""
import json
from pathlib import Path

import tiktoken

from regional_cue_row_check_v1 import validate
from scalar_new_endpoints_rows_v1 import PAIRS

P = Path(__file__).resolve().parent
OUT = P / "CROSSFIRST_HESSIAN_TOP2_FRESH_V1_ROWS.json"
if OUT.exists():
    raise FileExistsError(OUT)

encoder = tiktoken.get_encoding("gpt2")
seen = set()
for path in P.glob("*ROWS.json"):
    try:
        value = json.loads(path.read_text())
    except (ValueError, OSError):
        continue
    lists = [value] if isinstance(value, list) else [item for item in value.values() if isinstance(item, list)] if isinstance(value, dict) else []
    for items in lists:
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("ids"), list) and all(isinstance(token, int) for token in item["ids"]):
                seen.add(tuple(item["ids"]))

templates = [
    "Before the copy deadline, an editor based in {city} added this line to the feature: {stem}",
    "The archive identifies the author as a longtime resident of {city}; the surviving sentence reads: {stem}",
    "During a proofreading workshop in {city}, a participant completed the example with the words {stem}",
    "This passage was transcribed exactly from a columnist in {city}. It ends with {stem}",
]
city_pairs = [("Bristol", "Austin"), ("Oxford", "Denver"), ("Glasgow", "Phoenix"), ("Cardiff", "Detroit")]
rows = []
for family, template in enumerate(templates):
    uk_city, us_city = city_pairs[family]
    for concept, (uk, us, stem) in enumerate(PAIRS):
        for side, cue in enumerate(("British", "American")):
            city = (uk_city, us_city)[side]
            text = template.format(city=city, stem=stem)
            ids = encoder.encode(text)
            if tuple(ids) in seen:
                raise ValueError("full-prefix overlap")
            seen.add(tuple(ids))
            uk_ids, us_ids = encoder.encode(uk), encoder.encode(us)
            if len(uk_ids) != 1 or len(us_ids) != 1:
                raise ValueError("endpoint is not one token")
            rows.append({
                "row_id": len(rows),
                "donor_id": len(rows) ^ 1,
                "family": family,
                "family_name": ("copy_deadline", "archive_author", "proofreading_workshop", "columnist_transcript")[family],
                "concept": concept,
                "cue": cue,
                "city": city,
                "uk_token": uk,
                "us_token": us,
                "uk_id": uk_ids[0],
                "us_id": us_ids[0],
                "control_ids": [670, 3946],
                "text": text,
                "ids": ids,
            })

checks = validate(rows)
payload = {
    "rows": rows,
    "checks": checks,
    "scope": "96 score-blind full prefixes absent from every repository *ROWS.json at construction time; four new templates and four single-token city pairs; six existing spelling endpoints; no model call, capability filter, or corpus-OOD claim.",
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"rows": len(rows), "pairs": len(rows) // 2, "maximum_tokens": max(len(row["ids"]) for row in rows), "checks": checks}, indent=2))
