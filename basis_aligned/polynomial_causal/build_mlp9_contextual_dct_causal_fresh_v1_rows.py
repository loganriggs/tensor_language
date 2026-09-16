#!/usr/bin/env python3
"""Build a score-blind fresh panel for the frozen DCT causal intervention."""
import json
from pathlib import Path

import tiktoken

P = Path(__file__).resolve().parent
OUT = P / "MLP9_CONTEXTUAL_DCT_CAUSAL_FRESH_V1_ROWS.json"
if OUT.exists():
    raise FileExistsError(OUT)
encoder = tiktoken.get_encoding("gpt2")
seen = set()
for path in P.glob("*ROWS.json"):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    lists = [value] if isinstance(value, list) else [item for item in value.values() if isinstance(item, list)] if isinstance(value, dict) else []
    for items in lists:
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("ids"), list) and all(isinstance(token, int) for token in item["ids"]):
                seen.add(tuple(item["ids"]))

clauses = [
    "the harbor authority replaced three damaged navigation buoys",
    "a revised catalog listed the manuscripts in chronological order",
    "the bakery donated its unsold bread to the neighborhood pantry",
    "both research teams reproduced the unexpected measurement",
    "the council postponed debate until the engineering report arrived",
    "several migratory birds returned earlier than local observers expected",
    "the conservator removed a layer of varnish from the landscape",
    "a temporary bridge carried pedestrians across the construction site",
    "the warehouse installed new sensors beside every loading bay",
    "each volunteer received a map and an emergency contact card",
    "the telescope recorded a faint object beyond the known cluster",
    "all four proposals included funding for routine maintenance",
    "the pilot changed course after receiving the updated forecast",
    "an external reviewer verified the calculations in the appendix",
    "the library opened a quiet room for oral-history recordings",
    "the afternoon workshop ended with a practical demonstration",
]
templates = [
    "A newly circulated operations memo notes that {clause}. Follow-up:",
    "In a later public statement, the coordinator confirmed that {clause}, before adding",
    "Field bulletin {number} records that {clause}. Next action:",
    "Once the evidence had been reviewed, the committee concluded that {clause}; nevertheless,",
]
names = ("operations_memo", "public_statement", "field_bulletin", "committee_review")
rows = []
for family, template in enumerate(templates):
    for variant, clause in enumerate(clauses):
        text = template.format(clause=clause, number=variant + 301)
        ids = encoder.encode(text)
        if tuple(ids) in seen:
            raise ValueError("full-prefix overlap")
        seen.add(tuple(ids))
        rows.append({"row_id": len(rows), "family": family, "family_name": names[family], "variant": variant, "text": text, "ids": ids})
payload = {"schema": "mlp9_contextual_dct_causal_fresh_v1_rows", "rows": rows, "scope": "64 score-blind fresh cross-domain prefixes created after direction, scale, reader, and controls were frozen; full token sequences absent from all repository *ROWS.json at construction; no model call or outcome filtering."}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"rows": len(rows), "families": 4, "minimum_tokens": min(len(row["ids"]) for row in rows), "maximum_tokens": max(len(row["ids"]) for row in rows)}, indent=2))
