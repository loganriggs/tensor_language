"""Build an outcome-blind fresh panel for the inherited-city relay chain."""
import hashlib
import json
from pathlib import Path

import tiktoken
from regional_cue_row_check_v1 import validate
from regional_city_article_check_v1 import validate_city_articles

TEMPLATES = [
    'A museum in {city} stored a handwritten diary. Copy the next line without changing its spelling: "Our new',
    'An editor based in {city} kept the contributor\'s original spelling. Continue the quotation exactly: "Our new',
]
CITY_PAIRS = [("Bristol", "Boston"), ("Oxford", "Austin")]
ENDPOINTS = [
    ("colour", "color"), ("centre", "center"), ("travelled", "traveled"),
    ("favourite", "favorite"), ("honour", "honor"), ("behaviour", "behavior"),
]


def main():
    root = Path(__file__).resolve().parent
    out = root / "ODD_ATTENTION8H2_CHAIN_FRESH_V1_ROWS.json"
    if out.exists():
        raise FileExistsError(out)
    tokenizer = tiktoken.get_encoding("gpt2")
    rows = []
    for family, template in enumerate(TEMPLATES):
        for pair, (uk_city, us_city) in enumerate(CITY_PAIRS):
            for endpoint, (uk_word, us_word) in enumerate(ENDPOINTS):
                targets = [tokenizer.encode(" " + word) for word in (uk_word, us_word)]
                if any(len(ids) != 1 for ids in targets):
                    raise ValueError("Endpoint must be one token")
                for cue, city in (("British", uk_city), ("American", us_city)):
                    text = template.format(city=city)
                    row = {
                        "family": family, "template": family, "pair": pair,
                        "endpoint": endpoint, "cue": cue, "city": city,
                        "text": text, "ids": tokenizer.encode(text),
                        "uk_id": targets[0][0], "us_id": targets[1][0],
                        "control_ids": [670, 3946],
                    }
                    row["row_id"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    rows.append(row)
    assert len(rows) == 48
    validate(rows)
    validate_city_articles(rows)
    prior = set()
    for path in root.glob("*ROWS.json"):
        if path == out:
            continue
        try:
            prior.update(tuple(row.get("ids", [])) for row in json.loads(path.read_text()).get("rows", []))
        except Exception:
            continue
    assert not any(tuple(row["ids"]) in prior for row in rows)
    assert all(row["ids"].count(366) == 1 for row in rows)
    result = {
        "schema": "odd_attention8h2_chain_fresh.rows.v1",
        "selection": "Templates, cities and unseen endpoints frozen without model execution or scores. All city and endpoint strings are single GPT2 tokens.",
        "templates": TEMPLATES, "city_pairs": CITY_PAIRS, "endpoints": ENDPOINTS,
        "rows": rows,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "pairs": len(rows) // 2, "lengths": sorted(set(len(row["ids"]) for row in rows)), "quote_positions": sorted(set(row["ids"].index(366) for row in rows))}))


if __name__ == "__main__":
    main()
