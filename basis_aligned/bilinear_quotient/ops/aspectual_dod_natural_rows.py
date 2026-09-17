#!/usr/bin/env python3
# BQGATE: LIBRARY -- outcome-blind natural-text row miner for the aspectual DoD battery (CPU, no model).
"""Mine FineWeb (sample-10BT, streaming, first DOC_BUDGET documents in stream order) for natural rows:
a position whose NEXT token is ' has' or ' had', preceded within CUE_WINDOW tokens by a cue token
(' since'/'Since' -> cue 'since'; ' by'/'By' -> cue 'by'), with a plain lowercase word token right
before the target. The row context is the preceding CONTEXT tokens (must contain the cue). No model
score enters selection; rows are taken in stream order until 16 per (cue, label) cell = 64 rows.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("gpt2")
DOC_BUDGET, CUE_WINDOW, CONTEXT, PER_CELL = 6000, 10, 24, 16
HAS, HAD = ENC.encode(" has")[0], ENC.encode(" had")[0]
CUES = {ENC.encode(" since")[0]: "since", ENC.encode("Since")[0]: "since", ENC.encode(" by")[0]: "by", ENC.encode("By")[0]: "by"}
OUT = Path(__file__).resolve().parent.parent / "circuits/followups/aspectual_anchor_dod_natural_rows_v20.json"


def plain_word(tid: int) -> bool:
    t = ENC.decode([tid])
    return t.startswith(" ") and t[1:].isalpha() and t[1:].islower() and len(t) > 2


def mine():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    cells = {(c, l): [] for c in ("since", "by") for l in ("has", "had")}
    seen = 0
    for doc_index, doc in enumerate(itertools.islice(ds, DOC_BUDGET)):
        ids = ENC.encode(doc["text"])
        seen += 1
        for t in range(CONTEXT, len(ids) - 1):
            nxt = ids[t + 1]
            if nxt not in (HAS, HAD) or not plain_word(ids[t]):
                continue
            window = ids[max(0, t - CUE_WINDOW):t]
            cue_positions = [i for i, tid in enumerate(ids[t - CUE_WINDOW + 1:t + 1], start=t - CUE_WINDOW + 1) if tid in CUES]
            if not cue_positions:
                continue
            cue_pos = cue_positions[-1]
            cue = CUES[ids[cue_pos]]
            label = "has" if nxt == HAS else "had"
            if len(cells[(cue, label)]) >= PER_CELL:
                continue
            start = t + 1 - CONTEXT
            ctx = ids[start:t + 1]
            if any(tid in (HAS, HAD) for tid in ctx[:-1]):
                continue  # no earlier has/had inside the context
            cells[(cue, label)].append({"doc_index": doc_index, "position": t, "cue": cue, "label": label,
                                        "cue_offset": cue_pos - start, "ids": ctx, "text": ENC.decode(ctx)})
        if all(len(v) >= PER_CELL for v in cells.values()):
            break
    return cells, seen


def main():
    cells, seen = mine()
    rows = [r for key in sorted(cells) for r in cells[key]]
    payload = {"schema": "aspectual_dod_natural_rows_v20", "source": "HuggingFaceFW/fineweb sample-10BT streaming", "docs_scanned": seen,
               "per_cell": {f"{c}/{l}": len(v) for (c, l), v in cells.items()}, "rows": rows}
    payload["rows_sha256"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=1))
    for r in rows[:2] + rows[16:18] + rows[32:34] + rows[48:50]:
        print(r["cue"], r["label"], repr(r["text"][-90:]))


if __name__ == "__main__":
    main()
