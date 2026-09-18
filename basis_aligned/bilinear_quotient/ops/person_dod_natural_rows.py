#!/usr/bin/env python3
# BQGATE: LIBRARY -- natural-row config for the reflexive-person line (CPU, no model). Usage: person_dod_natural_rows.py <fineweb|pile>
"""Cue: a first- or second-person pronoun token (I / you, sentence-initial forms too) within 12 tokens; next token ' myself' / ' yourself'.
Other person pronouns (me, we, us, my, your, he, she, they, him, her, them) excluded from the context so the cue is the only person-marked
antecedent candidate. Cells (cue, label): I/myself and you/yourself are congruent; I/yourself and you/myself are corpus counter-cases
(a quoted or embedded speaker). Any-sense filter stated."""
from __future__ import annotations
import sys
import dod_natural_miner as M

SOURCE = sys.argv[1]
cues = M.ids_of({" I": "I", "I": "I", " you": "you", "You": "you", " You": "you"})
labels = M.ids_of({" myself": "myself", " yourself": "yourself"})
exclude = set(M.ids_of({t: 1 for t in (" me", " we", " us", " my", " your", " he", " she", " they", " him", " her", " them", " our", " mine", " yours", "We", " We", "He", " He", "She", " She", "They", " They", "My", " My", "Your", " Your")}).keys())
VERSION = {"fineweb": 110, "pile": 111}[SOURCE]
OUT = M.ROOT / f"circuits/followups/person_reflexive_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}.json"

if __name__ == "__main__":
    M.mine(cues, labels, SOURCE, OUT, f"person_reflexive_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}", exclude=exclude)
