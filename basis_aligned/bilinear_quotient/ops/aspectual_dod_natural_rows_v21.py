#!/usr/bin/env python3
# BQGATE: LIBRARY -- outcome-blind TEMPORAL-`by` natural row miner (CPU, no model). Same stream, same rules as
# aspectual_dod_natural_rows.py, but a `by` cue counts only when followed within 3 tokens by ' the', ' then',
# ' now', ' that' or a 4-digit year, and only 'by'-cued rows are collected (16 per next-token label).
from __future__ import annotations
import hashlib, itertools, json, re
from pathlib import Path
import tiktoken
import aspectual_dod_natural_rows as base
ENC = base.ENC
OUT = Path(__file__).resolve().parent.parent / "circuits/followups/aspectual_anchor_dod_natural_rows_v21.json"
BY = {ENC.encode(" by")[0], ENC.encode("By")[0]}
FOLLOW = {ENC.encode(t)[0] for t in (" the", " then", " now", " that")}


def temporal_by(ids, i):
    nxt = ids[i + 1:i + 4]
    if any(t in FOLLOW for t in nxt[:1]):
        return True
    return bool(re.fullmatch(r" ?(1[89]\d\d|20\d\d)", ENC.decode(nxt[:2]).rstrip()))


def mine():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    cells = {"has": [], "had": []}
    seen = 0
    for doc_index, doc in enumerate(itertools.islice(ds, base.DOC_BUDGET * 3)):
        ids = ENC.encode(doc["text"]); seen += 1
        for t in range(base.CONTEXT, len(ids) - 1):
            nxt = ids[t + 1]
            if nxt not in (base.HAS, base.HAD) or not base.plain_word(ids[t]):
                continue
            cue_positions = [i for i in range(t - base.CUE_WINDOW + 1, t + 1) if ids[i] in BY and temporal_by(ids, i)]
            if not cue_positions:
                continue
            label = "has" if nxt == base.HAS else "had"
            if len(cells[label]) >= base.PER_CELL:
                continue
            start = t + 1 - base.CONTEXT
            ctx = ids[start:t + 1]
            if any(tid in (base.HAS, base.HAD) for tid in ctx[:-1]):
                continue
            cells[label].append({"doc_index": doc_index, "position": t, "cue": "by", "label": label,
                                 "cue_offset": cue_positions[-1] - start, "ids": ctx, "text": ENC.decode(ctx)})
        if all(len(v) >= base.PER_CELL for v in cells.values()):
            break
    return cells, seen


def main():
    cells, seen = mine()
    rows = [r for key in sorted(cells) for r in cells[key]]
    payload = {"schema": "aspectual_dod_natural_rows_v21_temporal_by", "docs_scanned": seen,
               "per_cell": {k: len(v) for k, v in cells.items()}, "rows": rows,
               "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=1))
    for r in rows[:3] + rows[16:19]:
        print(r["label"], repr(r["text"][-90:]))


if __name__ == "__main__":
    main()
