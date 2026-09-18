#!/usr/bin/env python3
# BQGATE: LIBRARY -- natural-row config for the lexical were/was line (CPU, no model). Usage: lexical_number_dod_natural_rows.py <fineweb|pile>
"""Cue: an agent noun from the pronoun-number miner's list in singular / plural form within 12 tokens; next token ' were' / ' was'. Excluded from
the context: other list nouns, pronouns, and was / were / is / are themselves. Congruent cells plural/were, singular/was; the others are counter-cases.
Any-sense filter stated (a far noun may not be the subject)."""
from __future__ import annotations
import sys
import dod_natural_miner as M
import pronoun_number_dod_natural_rows as P

SOURCE = sys.argv[1]
cues = {tid: number for tid, (w, number) in P.NOUNS.items()}
labels = M.ids_of({" were": "were", " was": "was"})
exclude = set(P.PRONOUNS) | set(M.ids_of({" was": 1, " were": 1, " is": 1, " are": 1, "Was": 1, "Were": 1}).keys())
VERSION = {"fineweb": 138, "pile": 139}[SOURCE]
OUT = M.ROOT / f"circuits/followups/lexical_number_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}.json"

if __name__ == "__main__":
    M.mine(cues, labels, SOURCE, OUT, f"lexical_number_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}", exclude=exclude)
