#!/usr/bin/env python3
# BQGATE: LIBRARY -- outcome-blind natural-text row miner for the pronoun-gender DoD battery (CPU, no model).
"""Mine FineWeb (sample-10BT, streaming, first DOC_BUDGET documents in stream order) for natural rows: a position whose
NEXT token is ' he' or ' she', preceded within CUE_WINDOW tokens by a gendered noun from the fixed list below (the
module's 16 pairs + the v71 10 pairs, lowercase-with-space and sentence-initial forms), with no other gendered noun and no
third-person pronoun (he/she/his/her/him/hers/himself/herself) anywhere in the CONTEXT tokens before the target. Cell =
(noun gender, next token); rows are taken in stream order until PER_CELL per cell. No model score enters selection.
Congruent cells (male/he, female/she) test the readout; incongruent cells (male/she, female/he) are natural counter-cases."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("gpt2")
DOC_BUDGET, CUE_WINDOW, CONTEXT, PER_CELL = 20000, 12, 24, 16
HE, SHE = ENC.encode(" he")[0], ENC.encode(" she")[0]
PAIRS = (("king", "queen"), ("father", "mother"), ("brother", "sister"), ("uncle", "aunt"), ("son", "daughter"), ("husband", "wife"),
         ("boy", "girl"), ("man", "woman"), ("prince", "princess"), ("grandfather", "grandmother"), ("nephew", "niece"), ("actor", "actress"),
         ("waiter", "waitress"), ("duke", "duchess"), ("lord", "lady"), ("monk", "nun"),
         ("hero", "heroine"), ("god", "goddess"), ("grandson", "granddaughter"), ("boyfriend", "girlfriend"), ("spokesman", "spokeswoman"),
         ("dad", "mom"), ("bull", "cow"), ("groom", "bride"), ("male", "female"), ("guy", "gal"))
NOUNS = {}
for m, f in PAIRS:
    for w, g in ((m, "male"), (f, "female")):
        for form in (" " + w, w.capitalize(), " " + w.capitalize()):
            ids = ENC.encode(form)
            if len(ids) == 1:
                NOUNS[ids[0]] = (w, g)
PRONOUNS = {ENC.encode(t)[0] for t in (" he", " she", " his", " her", " him", " hers", " himself", " herself", "He", "She", " He", " She")
            if len(ENC.encode(t)) == 1}
import sys
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "fineweb"   # "fineweb" (training corpus, v73) or "pile" (NeelNanda/pile-10k, OOD, v75)
OUT = Path(__file__).resolve().parent.parent / ("circuits/followups/pronoun_gender_dod_natural_rows_v73.json" if SOURCE == "fineweb" else "circuits/followups/pronoun_gender_dod_pile_rows_v75.json")


def mine():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True) if SOURCE == "fineweb" else load_dataset("NeelNanda/pile-10k", split="train")
    cells = {(g, l): [] for g in ("male", "female") for l in ("he", "she")}
    seen = 0
    for doc_index, doc in enumerate(itertools.islice(ds, DOC_BUDGET)):
        ids = ENC.encode(doc["text"])
        seen += 1
        for t in range(CONTEXT, len(ids) - 1):
            nxt = ids[t + 1]
            if nxt not in (HE, SHE):
                continue
            start = t + 1 - CONTEXT
            ctx = ids[start:t + 1]
            if any(tid in PRONOUNS for tid in ctx):
                continue
            nouns = [(i, NOUNS[tid]) for i, tid in enumerate(ctx) if tid in NOUNS]
            if len(nouns) != 1 or nouns[0][0] < CONTEXT - CUE_WINDOW:
                continue
            (offset, (word, gender)) = nouns[0]
            label = "he" if nxt == HE else "she"
            if len(cells[(gender, label)]) >= PER_CELL:
                continue
            cells[(gender, label)].append({"doc_index": doc_index, "position": t, "noun": word, "gender": gender, "label": label,
                                           "cue_offset": offset, "ids": ctx, "text": ENC.decode(ctx)})
        if all(len(v) >= PER_CELL for v in cells.values()):
            break
    return cells, seen


def main():
    cells, seen = mine()
    rows = [r for key in sorted(cells) for r in cells[key]]
    payload = {"schema": "pronoun_gender_dod_natural_rows_v73" if SOURCE == "fineweb" else "pronoun_gender_dod_pile_rows_v75", "source": "HuggingFaceFW/fineweb sample-10BT streaming" if SOURCE == "fineweb" else "NeelNanda/pile-10k (stream order)", "docs_scanned": seen,
               "per_cell": {f"{g}/{l}": len(v) for (g, l), v in cells.items()}, "rows": rows}
    payload["rows_sha256"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=1))
    for r in rows[:2] + rows[16:18] + rows[32:34] + rows[48:50]:
        print(r["gender"], r["label"], r["noun"], repr(r["text"][-90:]))


if __name__ == "__main__":
    main()
