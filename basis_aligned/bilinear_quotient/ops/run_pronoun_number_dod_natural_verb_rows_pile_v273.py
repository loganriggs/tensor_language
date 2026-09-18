#!/usr/bin/env python3
# BQGATE: LIBRARY -- CPU natural-row miner run (no model): pronoun-number rows WITH a verb annotation (review 28).
# BQLANE: cpu
"""Pronoun number they/he: natural rows with a VERB annotation (v273, pile). Same cells and filters as v77 / v78 (a position whose next token is ' they' or ' he',
preceded within 12 tokens by exactly one list noun in singular or plural form, no third-person pronoun in the 24-token context), plus the generic miner's
`second` requirement: a past-tense verb from a fixed 71-word single-token list must occur BETWEEN the noun and the target, and its position is recorded as
`second_offset`. Purpose: the verb-site units (MLP-6 {2483, 2826, 4131}, MLP-5 {1036, 715}) were undetectable at cue + 1 on the v77 / v78 rows (v248), where the
token after the noun is not reliably the verb. Outcome-blind: no model score enters selection."""
from __future__ import annotations
from pathlib import Path
import dod_natural_miner as M
import pronoun_number_dod_natural_rows as base
import aspectual_dod_lib as L

VERBS = ['said','was','were','had','made','took','went','came','saw','got','told','found','gave','left','felt','put','kept','began','asked','called','knew','thought','wanted','used','worked','looked','played','moved','lived','turned','seemed','became','opened','walked','won','lost','bought','sold','held','ran','met','paid','sent','built','wrote','read','spoke','stood','sat','brought','decided','needed','tried','started','showed','helped','died','watched','remained','stayed','returned','arrived','agreed','joined','served','led','fell','grew','chose','drove','heard']
VERB_IDS = set()
for w in VERBS:
    try: VERB_IDS.add(L._single(" " + w))
    except Exception: pass
OUT = Path(__file__).resolve().parent.parent / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"

if __name__ == "__main__":
    cues = {tid: number for tid, (w, number) in base.NOUNS.items()}
    labels = {base.THEY: "they", base.HE: "he"}
    M.mine(cues, labels, "pile", OUT, "pronoun_number_dod_natural_verb_rows_v273", exclude=base.PRONOUNS, cue_window=base.CUE_WINDOW, context=base.CONTEXT, per_cell=base.PER_CELL,
           doc_budget=base.DOC_BUDGET, second={"plural": VERB_IDS, "singular": VERB_IDS})
