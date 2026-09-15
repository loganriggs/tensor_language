#!/usr/bin/env python3
"""Author a score-blind panel for the head9.8 routing/value composition test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import tiktoken

from regional_city_article_check_v1 import validate_city_articles
from regional_cue_row_check_v1 import validate


TEMPLATES = (
    'A librarian in {city} archived an original manuscript. Preserve the regional spelling when continuing: "The newly',
    'An exhibit from {city} retained its writer\'s exact wording. Complete the quoted sentence faithfully: "The newly',
)
CITY_PAIRS = (("Edinburgh", "Chicago"), ("Liverpool", "Denver"))
ENDPOINTS = (
    ("colour", "color"), ("centre", "center"), ("travelled", "traveled"),
    ("theatre", "theater"), ("organise", "organize"), ("neighbour", "neighbor"),
)


def canonical_row(row):
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode()


def main():
    root = Path(__file__).resolve().parent
    out = root / "SETTING2_REGIONAL_QK1_VALUE_COMPOSITION_FRESH_V1_ROWS.json"
    if out.exists():
        raise FileExistsError(out)
    encoder = tiktoken.get_encoding("gpt2")
    rows = []
    for family, template in enumerate(TEMPLATES):
        for pair, (uk_city, us_city) in enumerate(CITY_PAIRS):
            for endpoint, (uk_word, us_word) in enumerate(ENDPOINTS):
                targets = [encoder.encode(" " + word) for word in (uk_word, us_word)]
                if any(len(ids) != 1 for ids in targets):
                    raise ValueError("Every endpoint must be one GPT2 token")
                for cue, city in (("British", uk_city), ("American", us_city)):
                    text = template.format(city=city)
                    row = {
                        "family": family, "template": family, "pair": pair,
                        "endpoint": endpoint, "cue": cue, "city": city,
                        "text": text, "ids": encoder.encode(text),
                        "uk_id": targets[0][0], "us_id": targets[1][0],
                        "control_ids": [3797, 3290],
                    }
                    row["row_id"] = hashlib.sha256(canonical_row(row)).hexdigest()
                    rows.append(row)
    if len(rows) != 48:
        raise ValueError("Expected 48 rows")
    checks = validate(rows)
    validate_city_articles(rows)
    prior_ids = set()
    for path in root.glob("*ROWS.json"):
        if path == out:
            continue
        try:
            payload = json.loads(path.read_text())
            prior_ids.update(tuple(row.get("ids", ())) for row in payload.get("rows", ()))
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    overlap = sum(tuple(row["ids"]) in prior_ids for row in rows)
    if overlap:
        raise ValueError(f"Found {overlap} rows with previously used token contexts")
    payload = {
        "schema": "setting2_regional_qk1_value_composition_fresh_rows_v1",
        "selection": "Templates, cities, and endpoints fixed without model execution, activations, or scores.",
        "templates": TEMPLATES, "city_pairs": CITY_PAIRS, "endpoints": ENDPOINTS,
        "prior_context_overlap": overlap, "row_checks": checks, "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "pairs": len(rows)//2,
                      "lengths": sorted({len(row['ids']) for row in rows}),
                      "prior_context_overlap": overlap}))


if __name__ == "__main__":
    main()
