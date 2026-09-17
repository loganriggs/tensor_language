#!/usr/bin/env python3
# BQGATE: LIBRARY -- outcome-blind OOD row miner on the Pile mirror (monology/pile-uncopyrighted, streaming,
# stream order), same rules as aspectual_dod_natural_rows.py (any-sense since/by cells, 16 per (cue, label))
# plus the temporal-`by` cells of aspectual_dod_natural_rows_v21.py (16 per label). CPU, no model.
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
import aspectual_dod_natural_rows as base
import aspectual_dod_natural_rows_v21 as tb
ENC = base.ENC
OUT = Path(__file__).resolve().parent.parent / "circuits/followups/aspectual_anchor_dod_pile_rows_v26.json"
DOC_BUDGET = 20000


def main():
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    any_cells = {(c, l): [] for c in ("since", "by") for l in ("has", "had")}
    tb_cells = {"has": [], "had": []}
    seen = 0
    for doc_index, doc in enumerate(itertools.islice(ds, DOC_BUDGET)):
        ids = ENC.encode(doc["text"]); seen += 1
        for t in range(base.CONTEXT, len(ids) - 1):
            nxt = ids[t + 1]
            if nxt not in (base.HAS, base.HAD) or not base.plain_word(ids[t]):
                continue
            label = "has" if nxt == base.HAS else "had"
            start = t + 1 - base.CONTEXT
            ctx = ids[start:t + 1]
            if any(tid in (base.HAS, base.HAD) for tid in ctx[:-1]):
                continue
            cue_positions = [i for i in range(t - base.CUE_WINDOW + 1, t + 1) if ids[i] in base.CUES]
            if cue_positions:
                cue_pos = cue_positions[-1]; cue = base.CUES[ids[cue_pos]]
                if len(any_cells[(cue, label)]) < base.PER_CELL:
                    any_cells[(cue, label)].append({"doc_index": doc_index, "position": t, "cue": cue, "label": label,
                                                    "cue_offset": cue_pos - start, "ids": ctx, "text": ENC.decode(ctx),
                                                    "meta": str(doc.get("meta", ""))[:80]})
            tbp = [i for i in range(t - base.CUE_WINDOW + 1, t + 1) if ids[i] in tb.BY and tb.temporal_by(ids, i)]
            if tbp and len(tb_cells[label]) < base.PER_CELL:
                tb_cells[label].append({"doc_index": doc_index, "position": t, "cue": "by", "label": label,
                                        "cue_offset": tbp[-1] - start, "ids": ctx, "text": ENC.decode(ctx), "meta": str(doc.get("meta", ""))[:80]})
        if all(len(v) >= base.PER_CELL for v in any_cells.values()) and all(len(v) >= base.PER_CELL for v in tb_cells.values()):
            break
    any_rows = [r for key in sorted(any_cells) for r in any_cells[key]]
    tb_rows = [r for key in sorted(tb_cells) for r in tb_cells[key]]
    payload = {"schema": "aspectual_dod_pile_rows_v26", "source": "monology/pile-uncopyrighted streaming", "docs_scanned": seen,
               "per_cell": {**{f"any/{c}/{l}": len(v) for (c, l), v in any_cells.items()}, **{f"temporal_by/{l}": len(v) for l, v in tb_cells.items()}},
               "any_rows": any_rows, "temporal_by_rows": tb_rows,
               "any_rows_sha256": hashlib.sha256(json.dumps(any_rows, sort_keys=True).encode()).hexdigest(),
               "temporal_by_rows_sha256": hashlib.sha256(json.dumps(tb_rows, sort_keys=True).encode()).hexdigest()}
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if not k.endswith("rows")}, indent=1))
    for r in any_rows[:2] + tb_rows[:2]:
        print(r["cue"], r["label"], repr(r["text"][-80:]))


if __name__ == "__main__":
    main()
