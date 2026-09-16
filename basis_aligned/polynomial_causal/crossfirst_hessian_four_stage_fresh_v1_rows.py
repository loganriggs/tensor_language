#!/usr/bin/env python3
"""Build a score-blind new-endpoint panel for the frozen four-stage correction."""
import json
from pathlib import Path

import tiktoken

from regional_cue_row_check_v1 import validate

P = Path(__file__).resolve().parent
OUT = P / "CROSSFIRST_HESSIAN_FOUR_STAGE_FRESH_V1_ROWS.json"
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

endpoints = [
    (" centre", " center", "The committee placed the memorial at the civic"),
    (" metre", " meter", "The carpenter measured exactly one"),
    (" defence", " defense", "The barrister prepared a detailed legal"),
    (" licence", " license", "The council renewed the operating"),
    (" travelling", " traveling", "They spent the entire summer"),
    (" cancelled", " canceled", "Without warning, the airline"),
]
templates = [
    "An editor working in {city} checked the regional style guide before completing this sentence: {stem}",
    "The manuscript came from a publishing office in {city}. Its final uncorrected line was: {stem}",
    "At a language seminar in {city}, the instructor asked for the locally conventional ending to: {stem}",
    "A newspaper archived in {city} contains the following unfinished quotation: \"{stem}",
]
city_pairs = [("Leeds", "Dallas"), ("Belfast", "Miami"), ("Edinburgh", "Houston"), ("Swansea", "Portland")]
rows = []
for family, template in enumerate(templates):
    uk_city, us_city = city_pairs[family]
    for concept, (uk, us, stem) in enumerate(endpoints):
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
                "family_name": ("style_guide", "publishing_office", "language_seminar", "archived_newspaper")[family],
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
    "scope": "48 score-blind full prefixes absent from every repository *ROWS.json at construction; four new templates, four city pairs, and six single-token UK/US endpoints absent from the preceding CrossFirst Hessian panels; no model call, capability filter, or corpus-OOD claim.",
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"rows": len(rows), "pairs": len(rows) // 2, "maximum_tokens": max(len(row["ids"]) for row in rows), "checks": checks}, indent=2))
