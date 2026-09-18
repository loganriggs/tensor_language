#!/usr/bin/env python3
# BQGATE: LIBRARY -- natural-row configs for the two selection lines (CPU, no model). Usage: selection_dod_natural_rows.py <inof|updown> <fineweb|pile>
"""in/of: cue tokens interested / afraid (lowercase-with-space and sentence-initial forms) within 12 tokens, next token ' in' / ' of'.
up/down: cue tokens woke / calmed (and woken / calms / wakes / calm?) -- kept to the module's pair woke / calmed, next ' up' / ' down'.
Excluded tokens: none beyond the single-cue rule (any-sense: 'afraid' + 'of' is near-deterministic; the incongruent cells are
whatever the corpus offers and may stay short). Receipts under circuits/followups/selection_*_rows_v9x.json."""
from __future__ import annotations
import sys
import dod_natural_miner as M

LINE, SOURCE = sys.argv[1], sys.argv[2]
CONF = {"inof": (M.ids_of({" interested": "interested", "Interested": "interested", " Interested": "interested", " afraid": "afraid", "Afraid": "afraid", " Afraid": "afraid"}),
                 M.ids_of({" in": "in", " of": "of"}), "selection_inof"),
        "updown": (M.ids_of({" woke": "woke", "Woke": "woke", " Woke": "woke", " calmed": "calmed", "Calmed": "calmed", " Calmed": "calmed"}),
                   M.ids_of({" up": "up", " down": "down"}), "selection_updown")}
cues, labels, stem = CONF[LINE]
VERSION = {("inof", "fineweb"): 91, ("inof", "pile"): 92, ("updown", "fineweb"): 93, ("updown", "pile"): 94}[(LINE, SOURCE)]
OUT = M.ROOT / f"circuits/followups/{stem}_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}.json"

if __name__ == "__main__":
    M.mine(cues, labels, SOURCE, OUT, f"{stem}_dod_{'natural' if SOURCE == 'fineweb' else 'pile'}_rows_v{VERSION}")
