#!/usr/bin/env python3
"""Build outcome-blind fresh rows for the frozen head17.2 factor corner."""
import hashlib
import json
from pathlib import Path

import tiktoken

from regional_city_article_check_v1 import validate_city_articles
from regional_cue_row_check_v1 import validate


TEMPLATES = (
    'An estate office in {city} found a box of correspondence. Copy the next word using the writer\'s regional conventions: "The committee debated the',
    'A language historian from {city} transcribed an old interview. Continue the quotation in the speaker\'s spelling: "They described the',
)
CITY_PAIRS = (("Norwich", "Portland"), ("Cardiff", "Denver"))
ENDPOINTS = (
    ("behaviour", "behavior"),
    ("licence", "license"),
    ("analyse", "analyze"),
    ("recognise", "recognize"),
    ("fulfil", "fulfill"),
    ("cancelled", "canceled"),
)


def main():
    root = Path(__file__).resolve().parent
    out = root / "SETTING2_REGIONAL_ATTENTION17H2_FACTOR_CORNER_FRESH_V1_ROWS.json"
    if out.exists():
        raise FileExistsError(out)
    enc = tiktoken.get_encoding("gpt2")
    rows = []
    for family, template in enumerate(TEMPLATES):
        for pair, (uk_city, us_city) in enumerate(CITY_PAIRS):
            for endpoint, (uk_word, us_word) in enumerate(ENDPOINTS):
                targets = [enc.encode(" " + word) for word in (uk_word, us_word)]
                if any(len(ids) != 1 for ids in targets):
                    raise ValueError("endpoint must be one GPT-2 token")
                for cue, city in (("British", uk_city), ("American", us_city)):
                    row = {
                        "family": family,
                        "template": family,
                        "pair": pair,
                        "endpoint": endpoint,
                        "cue": cue,
                        "city": city,
                        "text": template.format(city=city),
                        "uk_id": targets[0][0],
                        "us_id": targets[1][0],
                        "control_ids": [670, 3946],
                    }
                    row["ids"] = enc.encode(row["text"])
                    row["row_id"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    rows.append(row)
    if len(rows) != 48:
        raise ValueError("expected 48 rows")
    checks = validate(rows)
    validate_city_articles(rows)
    prior = set()
    for path in root.glob("*ROWS.json"):
        if path == out:
            continue
        try:
            prior.update(tuple(row.get("ids", ())) for row in json.loads(path.read_text()).get("rows", ()))
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    overlap = sum(tuple(row["ids"]) in prior for row in rows)
    if overlap:
        raise ValueError(f"{overlap} prior contexts")
    out.write_text(json.dumps({
        "schema": "setting2_regional_attention17h2_factor_corner_fresh_rows_v1",
        "selection": "Templates, cities, and endpoints authored and token-checked without model execution, activations, logits, or scores.",
        "templates": TEMPLATES,
        "city_pairs": CITY_PAIRS,
        "endpoints": ENDPOINTS,
        "prior_context_overlap": overlap,
        "row_checks": checks,
        "rows": rows,
    }, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "pairs": len(rows) // 2, "lengths": sorted({len(row["ids"]) for row in rows}), "prior_context_overlap": overlap}))


if __name__ == "__main__":
    main()
