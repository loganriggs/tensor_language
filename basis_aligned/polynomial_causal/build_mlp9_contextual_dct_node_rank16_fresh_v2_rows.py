#!/usr/bin/env python3
"""Build the second untouched cross-domain context panel for frozen rank 16."""
import json
from pathlib import Path

import tiktoken

P = Path(__file__).resolve().parent
OUT = P / "MLP9_CONTEXTUAL_DCT_NODE_RANK16_FRESH_V2_ROWS.json"
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
    "the coastal survey identified four previously unmapped inlets",
    "a software update restored access to the archived records",
    "the orchard produced fewer apples than the previous season",
    "both train services resumed after the signal inspection",
    "the medical team revised its schedule for the mobile clinic",
    "several roof panels shifted during the overnight windstorm",
    "the translation preserved the ambiguity of the original phrase",
    "a narrow footpath continued beyond the stone boundary wall",
    "the finance office approved the corrected invoice on Tuesday",
    "each classroom received two boxes of laboratory equipment",
    "the excavation revealed a second foundation beneath the tower",
    "all remaining tickets were distributed through local schools",
    "the navigation system selected a safer route around the pass",
    "an independent audit confirmed the revised inventory totals",
    "the community garden added a shelter for tools and seedlings",
    "the evening broadcast included a detailed weather advisory",
]
templates = [
    "An internal planning note states that {clause}. Pending item:",
    "During the recorded interview, the speaker explained that {clause}, and continued",
    "Technical log {number} reports: {clause}. Status:",
    "After comparing the documents, the investigators agreed that {clause}; however,",
]
rows = []
for family, template in enumerate(templates):
    for variant, clause in enumerate(clauses):
        text = template.format(clause=clause, number=variant + 101)
        ids = encoder.encode(text)
        if tuple(ids) in seen:
            raise ValueError("full-prefix overlap")
        seen.add(tuple(ids))
        rows.append({"row_id": len(rows), "family": family, "family_name": ("planning_note", "recorded_interview", "technical_log", "document_review")[family], "variant": variant, "text": text, "ids": ids})
payload = {"schema": "mlp9_contextual_dct_node_rank16_fresh_v2_rows", "rows": rows, "scope": "64 second-panel score-blind cross-domain prefixes absent from every repository *ROWS.json at construction; no model call, outcome, filtering, endpoint, or corpus-OOD claim."}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"rows": len(rows), "families": 4, "minimum_tokens": min(len(row["ids"]) for row in rows), "maximum_tokens": max(len(row["ids"]) for row in rows)}, indent=2))
