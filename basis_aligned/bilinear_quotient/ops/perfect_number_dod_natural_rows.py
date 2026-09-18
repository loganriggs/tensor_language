#!/usr/bin/env python3
# BQGATE: LIBRARY -- natural-row config for the perfect have/has line (CPU, no model). Usage: perfect_number_dod_natural_rows.py <fineweb|pile>
"""Cue: an agent noun from the pronoun-number miner's word list in singular (' noun') or plural (' nouns') form within 12 tokens; next token
' have' / ' has'. Excluded from the context: other list nouns and the pronouns (he/she/they/…) and the auxiliaries have/has themselves, so
the cue noun is the only number-marked subject candidate. Any-sense filter stated (a far noun may not be the subject). Cells (number, label):
plural/have and singular/has are congruent; the others are corpus counter-cases (e.g. 'the students' teacher has')."""
from __future__ import annotations
import sys
import dod_natural_miner as M
import pronoun_number_dod_natural_rows as P

SOURCE = sys.argv[1]
cues = {tid: number for tid, (w, number) in P.NOUNS.items()}
labels = M.ids_of({" have": "have", " has": "has"})
exclude = set(P.PRONOUNS) | set(M.ids_of({" have": 1, " has": 1, " had": 1, "Have": 1, "Has": 1}).keys())
VERSION = {"fineweb": 100, "pile": 101}[SOURCE]
OUT = M.ROOT / f"circuits/followups/perfect_number_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}.json"

if __name__ == "__main__":
    M.mine(cues, labels, SOURCE, OUT, f"perfect_number_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}", exclude=exclude)
