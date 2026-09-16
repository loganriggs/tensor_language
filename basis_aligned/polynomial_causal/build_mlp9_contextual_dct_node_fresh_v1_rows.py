#!/usr/bin/env python3
"""Build score-blind cross-domain contexts for the extracted MLP9 DCT node."""
import json
from pathlib import Path

import tiktoken

P = Path(__file__).resolve().parent
OUT = P / "MLP9_CONTEXTUAL_DCT_NODE_FRESH_V1_ROWS.json"
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
    "the northern route reopened shortly after sunrise",
    "three ceramic samples remained intact after the trial",
    "the revised estimate omitted several maintenance costs",
    "a quiet courtyard connected the library and the gallery",
    "each volunteer received a map and a numbered badge",
    "the rainfall record showed an unusually dry spring",
    "two backup generators started without manual intervention",
    "the oldest manuscript used a different system of punctuation",
    "the research vessel changed course near the outer reef",
    "every sealed container passed the overnight pressure test",
    "the final rehearsal ended before the audience arrived",
    "several witnesses described the same blue delivery van",
    "the replacement sensor reported a stable temperature",
    "an unexpected delay shifted the launch into early autumn",
    "the workshop produced six prototypes in a single afternoon",
    "no additional permits were required for the temporary bridge",
]
templates = [
    "The quarterly operations memo records that {clause}. The next observation is",
    "A museum guide answered the visitor carefully: \"It appears that {clause}.\" She then added",
    "Field notebook entry {number}: {clause}. Follow-up:",
    "Given the available evidence, the review panel concluded {clause}; nevertheless,",
]

rows = []
for family, template in enumerate(templates):
    for variant, clause in enumerate(clauses):
        text = template.format(clause=clause, number=variant + 1)
        ids = encoder.encode(text)
        if tuple(ids) in seen:
            raise ValueError("full-prefix overlap")
        seen.add(tuple(ids))
        rows.append({
            "row_id": len(rows),
            "family": family,
            "family_name": ("operations_memo", "museum_dialogue", "field_notebook", "review_panel")[family],
            "variant": variant,
            "text": text,
            "ids": ids,
        })

payload = {
    "schema": "mlp9_contextual_dct_node_fresh_v1_rows",
    "rows": rows,
    "scope": "64 score-blind cross-domain prefixes absent from every repository *ROWS.json at construction; four prose structures and sixteen fixed clauses; no model call, outcome, capability filter, endpoint, or corpus-OOD claim.",
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"rows": len(rows), "families": 4, "minimum_tokens": min(len(row["ids"]) for row in rows), "maximum_tokens": max(len(row["ids"]) for row in rows)}, indent=2))
