#!/usr/bin/env python3
# BQGATE: LIBRARY -- outcome-blind natural-text row miner for the pronoun-number DoD battery (CPU, no model).
"""Mine FineWeb (sample-10BT streaming; `fineweb`, v77) or NeelNanda/pile-10k (`pile`, v78) for natural rows: a position whose
NEXT token is ' they' or ' he', preceded within CUE_WINDOW tokens by exactly one agent noun from the fixed list (the v76 fresh
candidates + the corpus reporter/agent lists), in singular (' noun') or plural (' nouns') form, with no other list noun and no
third-person pronoun (he/she/they/his/her/their/him/them/himself/themselves) in the CONTEXT tokens before the target. Cell =
(noun number, next token); 16 per cell in stream order. No model score enters selection; the noun filter is any-sense."""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import tiktoken

import aspectual_dod_lib as L
import circuit_fast_screen_candidates as lex
import run_pronoun_number_dod_battery_v76 as v76

ENC = tiktoken.get_encoding("gpt2")
SOURCE = sys.argv[1] if len(sys.argv) > 1 else "fineweb"
DOC_BUDGET, CUE_WINDOW, CONTEXT, PER_CELL = 20000, 12, 24, 16
THEY, HE = ENC.encode(" they")[0], ENC.encode(" he")[0]
WORDS = sorted(set(v76.AGENT_CANDIDATES) | {p[0] for p in lex._REPORTERS} | {p[1] for p in lex._REPORTERS} | set(L.AGENTS))
NOUNS = {}
for w in WORDS:
    for form, number in ((" " + w, "singular"), (" " + w + "s", "plural"), (" " + w.capitalize(), "singular"), (" " + w.capitalize() + "s", "plural")):
        ids = ENC.encode(form)
        if len(ids) == 1:
            NOUNS[ids[0]] = (w, number)
PRONOUNS = {ENC.encode(t)[0] for t in (" he", " she", " they", " his", " her", " their", " him", " them", " himself", " themselves", "He", "She", "They", " He", " She", " They")
            if len(ENC.encode(t)) == 1}
OUT = Path(__file__).resolve().parent.parent / ("circuits/followups/pronoun_number_dod_natural_rows_v77.json" if SOURCE == "fineweb" else "circuits/followups/pronoun_number_dod_pile_rows_v78.json")


def mine():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True) if SOURCE == "fineweb" else load_dataset("NeelNanda/pile-10k", split="train")
    cells = {(g, l): [] for g in ("singular", "plural") for l in ("they", "he")}
    seen = 0
    for doc_index, doc in enumerate(itertools.islice(ds, DOC_BUDGET)):
        ids = ENC.encode(doc["text"])
        seen += 1
        for t in range(CONTEXT, len(ids) - 1):
            nxt = ids[t + 1]
            if nxt not in (THEY, HE):
                continue
            start = t + 1 - CONTEXT
            ctx = ids[start:t + 1]
            if any(tid in PRONOUNS for tid in ctx):
                continue
            nouns = [(i, NOUNS[tid]) for i, tid in enumerate(ctx) if tid in NOUNS]
            if len(nouns) != 1 or nouns[0][0] < CONTEXT - CUE_WINDOW:
                continue
            (offset, (word, number)) = nouns[0]
            label = "they" if nxt == THEY else "he"
            if len(cells[(number, label)]) >= PER_CELL:
                continue
            cells[(number, label)].append({"doc_index": doc_index, "position": t, "noun": word, "number": number, "label": label,
                                           "cue_offset": offset, "ids": ctx, "text": ENC.decode(ctx)})
        if all(len(v) >= PER_CELL for v in cells.values()):
            break
    return cells, seen


def main():
    cells, seen = mine()
    rows = [r for key in sorted(cells) for r in cells[key]]
    payload = {"schema": "pronoun_number_dod_natural_rows_v77" if SOURCE == "fineweb" else "pronoun_number_dod_pile_rows_v78",
               "source": "HuggingFaceFW/fineweb sample-10BT streaming" if SOURCE == "fineweb" else "NeelNanda/pile-10k (stream order)", "docs_scanned": seen,
               "per_cell": {f"{g}/{l}": len(v) for (g, l), v in cells.items()}, "rows": rows}
    payload["rows_sha256"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=1))
    for r in rows[:2] + rows[16:18] + rows[32:34] + rows[48:50]:
        print(r["number"], r["label"], r["noun"], repr(r["text"][-80:]))


if __name__ == "__main__":
    main()
