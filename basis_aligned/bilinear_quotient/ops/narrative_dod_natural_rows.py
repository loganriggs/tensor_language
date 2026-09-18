#!/usr/bin/env python3
# BQGATE: LIBRARY -- natural-row config for the narrative was/is line (CPU, no model). Usage: narrative_dod_natural_rows.py <fineweb|pile>
"""The narrative line's panel cue is a tensed verb + 'last winter' / 'every winter' (spread over a clause), which the single-token miner cannot
match; the natural analogue used here is a single temporal adverb: past = yesterday / ago / formerly / previously / earlier, present = today / now /
currently / nowadays / presently, within 12 tokens of a next token ' was' / ' is'; other be-forms excluded from the context. Congruent cells
past/was and present/is; the others are corpus counter-cases (a past adverb with a present-tense clause, etc.). Any-sense filter stated."""
from __future__ import annotations
import sys
import dod_natural_miner as M

SOURCE = sys.argv[1]
past = {" yesterday": "past", "Yesterday": "past", " ago": "past", " formerly": "past", "Formerly": "past", " previously": "past", "Previously": "past", " earlier": "past", "Earlier": "past"}
present = {" today": "present", "Today": "present", " now": "present", "Now": "present", " currently": "present", "Currently": "present", " nowadays": "present", "Nowadays": "present", " presently": "present"}
cues = M.ids_of({**past, **present})
labels = M.ids_of({" was": "was", " is": "is"})
exclude = set(M.ids_of({t: 1 for t in (" was", " is", " were", " are", " been", " being", "Was", "Is", " had", " has", " have")}).keys())
VERSION = {"fineweb": 151, "pile": 152}[SOURCE]
OUT = M.ROOT / f"circuits/followups/narrative_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}.json"

if __name__ == "__main__":
    M.mine(cues, labels, SOURCE, OUT, f"narrative_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}", exclude=exclude)
