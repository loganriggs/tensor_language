#!/usr/bin/env python3
# BQGATE: LIBRARY -- generic outcome-blind natural-text row miner (CPU, no model) for the DoD natural-row panels.
"""`mine(cues, labels, source, out, schema, exclude=(), cue_window=12, context=24, per_cell=16, doc_budget=20000)`: scan FineWeb
(sample-10BT streaming) or NeelNanda/pile-10k in stream order for positions whose NEXT token is one of `labels` (token id -> label)
and whose CONTEXT tokens contain exactly one cue token from `cues` (token id -> cue name) within `cue_window` tokens before the
target, and none of `exclude` (token ids). Cells = (cue name, label); rows taken in stream order until `per_cell` each or the budget
ends (partial cells are recorded, not padded). No model score enters selection; the cue filter is any-sense (stated by callers).
Receipt: {"schema", "source", "docs_scanned", "per_cell", "rows": [{doc_index, position, cue, label, cue_offset, ids, text}], "rows_sha256"}.
`dod_natural_line.rows_from_receipt` reads it (label field = "label", construction from cue + label)."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("gpt2")
ROOT = Path(__file__).resolve().parent.parent


def ids_of(forms):
    out = {}
    for text, name in forms.items():
        ids = ENC.encode(text)
        if len(ids) == 1:
            out[ids[0]] = name
    return out


def mine(cues, labels, source, out, schema, exclude=(), cue_window=12, context=24, per_cell=16, doc_budget=20000, show=8):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True) if source == "fineweb" else load_dataset("NeelNanda/pile-10k", split="train")
    cells = {(c, l): [] for c in sorted(set(cues.values())) for l in sorted(set(labels.values()))}
    seen = 0
    for doc_index, doc in enumerate(itertools.islice(ds, doc_budget)):
        ids = ENC.encode(doc["text"]); seen += 1
        for t in range(context, len(ids) - 1):
            nxt = ids[t + 1]
            if nxt not in labels:
                continue
            start = t + 1 - context; ctx = ids[start:t + 1]
            if any(tid in exclude for tid in ctx):
                continue
            found = [(i, cues[tid]) for i, tid in enumerate(ctx) if tid in cues]
            if len(found) != 1 or found[0][0] < context - cue_window:
                continue
            offset, cue = found[0]; label = labels[nxt]
            if len(cells[(cue, label)]) >= per_cell:
                continue
            cells[(cue, label)].append({"doc_index": doc_index, "position": t, "cue": cue, "label": label, "cue_offset": offset, "ids": ctx, "text": ENC.decode(ctx)})
        if all(len(v) >= per_cell for v in cells.values()):
            break
    rows = [r for key in sorted(cells) for r in cells[key]]
    payload = {"schema": schema, "source": "HuggingFaceFW/fineweb sample-10BT streaming" if source == "fineweb" else "NeelNanda/pile-10k (stream order)", "docs_scanned": seen,
               "per_cell": {f"{c}/{l}": len(v) for (c, l), v in cells.items()}, "rows": rows}
    payload["rows_sha256"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    Path(out).write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}))
    for r in rows[::max(1, len(rows) // show)][:show]:
        print(r["cue"], r["label"], repr(r["text"][-70:]))
    return payload
