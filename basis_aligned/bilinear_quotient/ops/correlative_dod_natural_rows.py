#!/usr/bin/env python3
# BQGATE: LIBRARY -- natural-row configs for the two correlative lines (CPU, no model). Usage: correlative_dod_natural_rows.py <eithernot|bothneither> <fineweb|pile>
"""either/not: cue tokens either / not within 12 tokens, next token ' or' / ' but' (congruent either/or, not/but). both/neither: cue tokens both / neither,
next token ' and' / ' nor' (congruent both/and, neither/nor). No exclusions beyond the single-cue rule; any-sense filter stated ('not ... but' is
common in natural text, 'either ... but' rare). Receipts under circuits/followups/correlative_*_rows_v12x.json."""
from __future__ import annotations
import sys
import dod_natural_miner as M

LINE, SOURCE = sys.argv[1], sys.argv[2]
CONF = {"eithernot": (M.ids_of({" either": "either", "Either": "either", " Either": "either", " not": "not", "Not": "not", " Not": "not"}), M.ids_of({" or": "or", " but": "but"}), "correlative_either_not"),
        "bothneither": (M.ids_of({" both": "both", "Both": "both", " Both": "both", " neither": "neither", "Neither": "neither", " Neither": "neither"}), M.ids_of({" and": "and", " nor": "nor"}), "correlative_both_neither")}
cues, labels, stem = CONF[LINE]
VERSION = {("eithernot", "fineweb"): 125, ("eithernot", "pile"): 126, ("bothneither", "fineweb"): 127, ("bothneither", "pile"): 128}[(LINE, SOURCE)]
OUT = M.ROOT / f"circuits/followups/{stem}_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}.json"

if __name__ == "__main__":
    M.mine(cues, labels, SOURCE, OUT, f"{stem}_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}")
