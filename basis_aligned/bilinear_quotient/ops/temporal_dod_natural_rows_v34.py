#!/usr/bin/env python3
# BQGATE: LIBRARY -- outcome-blind natural row miner for the temporal will/had line (CPU, no model).
# Same rules as aspectual_dod_natural_rows.py with cues tomorrow/earlier and targets ' will'/' had';
# one panel from FineWeb (training corpus, in-distribution) and one from the Pile mirror (OOD).
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
import aspectual_dod_natural_rows as base
ENC = base.ENC
OUT = Path(__file__).resolve().parent.parent / "circuits/followups/temporal_auxiliary_dod_natural_rows_v34.json"
WILL, HAD = ENC.encode(" will")[0], ENC.encode(" had")[0]
CUES = {ENC.encode(" tomorrow")[0]: "tomorrow", ENC.encode("Tomorrow")[0]: "tomorrow", ENC.encode(" earlier")[0]: "earlier", ENC.encode("Earlier")[0]: "earlier"}
DOC_BUDGET = 30000


def mine(ds):
    cells = {(c, l): [] for c in ("tomorrow", "earlier") for l in ("will", "had")}
    seen = 0
    for doc_index, doc in enumerate(itertools.islice(ds, DOC_BUDGET)):
        ids = ENC.encode(doc["text"]); seen += 1
        for t in range(base.CONTEXT, len(ids) - 1):
            nxt = ids[t + 1]
            if nxt not in (WILL, HAD) or not base.plain_word(ids[t]):
                continue
            cue_positions = [i for i in range(t - base.CUE_WINDOW + 1, t + 1) if ids[i] in CUES]
            if not cue_positions:
                continue
            cue_pos = cue_positions[-1]; cue = CUES[ids[cue_pos]]; label = "will" if nxt == WILL else "had"
            if len(cells[(cue, label)]) >= base.PER_CELL:
                continue
            start = t + 1 - base.CONTEXT; ctx = ids[start:t + 1]
            if any(tid in (WILL, HAD) for tid in ctx[:-1]):
                continue
            cells[(cue, label)].append({"doc_index": doc_index, "position": t, "cue": cue, "label": label, "cue_offset": cue_pos - start, "ids": ctx, "text": ENC.decode(ctx)})
        if all(len(v) >= base.PER_CELL for v in cells.values()):
            break
    return cells, seen


def main():
    from datasets import load_dataset
    panels = {}
    for name, ds in (("fineweb", load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)),
                     ("pile", load_dataset("monology/pile-uncopyrighted", split="train", streaming=True))):
        cells, seen = mine(ds)
        rows = [r for key in sorted(cells) for r in cells[key]]
        panels[name] = {"docs_scanned": seen, "per_cell": {f"{c}/{l}": len(v) for (c, l), v in cells.items()}, "rows": rows,
                        "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}
        print(name, panels[name]["docs_scanned"], panels[name]["per_cell"])
    OUT.write_text(json.dumps({"schema": "temporal_dod_natural_rows_v34", "panels": panels}, indent=1) + "\n")
    for name in panels:
        for r in panels[name]["rows"][:2]:
            print(name, r["cue"], r["label"], repr(r["text"][-80:]))


if __name__ == "__main__":
    main()
