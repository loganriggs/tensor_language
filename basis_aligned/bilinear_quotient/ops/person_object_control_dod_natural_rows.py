#!/usr/bin/env python3
# BQGATE: LIBRARY -- natural-row config for the object-control me/you line (CPU, no model; two-token cue). Usage: person_object_control_dod_natural_rows.py <fineweb|pile>
"""Cue: an object pronoun ' me' (first person) or ' you' (second) within 12 tokens, followed by ' to' before the target (the control infinitive:
'told me to trust myself'); next token ' myself' / ' yourself'. Other person pronouns excluded from the context. Congruent cells me/myself and
you/yourself; the others are corpus counter-cases (a quoted speaker). Any-sense filter stated ('you' is also generic)."""
from __future__ import annotations
import sys
import dod_natural_miner as M

SOURCE = sys.argv[1]
cues = M.ids_of({" me": "me", " you": "you"})
second = {"me": set(M.ids_of({" to": 1}).keys()), "you": set(M.ids_of({" to": 1}).keys())}
labels = M.ids_of({" myself": "myself", " yourself": "yourself"})
exclude = set(M.ids_of({t: 1 for t in (" I", "I", " we", " us", " my", " your", " he", " she", " they", " him", " her", " them", " our", " mine", " yours", "You", " You", "We", " We")}).keys())
VERSION = {"fineweb": 155, "pile": 156}[SOURCE]
OUT = M.ROOT / f"circuits/followups/person_object_control_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}.json"

if __name__ == "__main__":
    M.mine(cues, labels, SOURCE, OUT, f"person_object_control_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}", exclude=exclude, second=second)
