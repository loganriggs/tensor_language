#!/usr/bin/env python3
# BQGATE: LIBRARY -- natural-row config for the modal would/will line (CPU, no model; two-token cue). Usage: modal_dod_natural_rows.py <fineweb|pile>
"""Cue: a conditional / temporal conjunction (If / if -> 'remote', When / when / Once / once -> 'present') within 12 tokens, plus a required second
token between it and the target: ' had' for remote, ' has' / ' have' for present; next token ' would' / ' will'. Other modals excluded from the
context. Congruent cells remote/would and present/will; the others are corpus counter-cases. Any-sense filter stated (an 'if … had … will' row can be
a real conditional of a different type)."""
from __future__ import annotations
import sys
import dod_natural_miner as M

SOURCE = sys.argv[1]
cues = M.ids_of({" If": "remote", "If": "remote", " if": "remote", " When": "present", "When": "present", " when": "present", " Once": "present", "Once": "present", " once": "present"})
second = {"remote": set(M.ids_of({" had": 1, " hadn": 1}).keys()), "present": set(M.ids_of({" has": 1, " have": 1}).keys())}
labels = M.ids_of({" would": "would", " will": "will"})
exclude = set(M.ids_of({t: 1 for t in (" would", " will", " could", " should", " might", " shall", " won", " wouldn")}).keys())
VERSION = {"fineweb": 153, "pile": 154}[SOURCE]
OUT = M.ROOT / f"circuits/followups/modal_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}.json"

if __name__ == "__main__":
    M.mine(cues, labels, SOURCE, OUT, f"modal_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}", exclude=exclude, second=second)
