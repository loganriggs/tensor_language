#!/usr/bin/env python3
"""verb_preposition_fc_by_from.swore_vs_withdrew -- does a site carry a verb's PREPOSITION SELECTION (swore -> `by`, withdrew -> `from`) in the FRAME-C sentence shape?

`It turned out the pilot swore, apparently,` obliges ` by`; `... withdrew, apparently,` obliges ` from`. Frame-C copy of a counted frame-A preposition circuit's cue mappings, authored under the measured (frame, mapping) account and verified against its own donor.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-08, v353 preposition frame batch).
Capability is checked on CPU before the battery: rows where the donor does not beat the base on the
donor axis are dropped by `g.prepare(valid_only=True)` and counted as `n_dropped` in the receipt.
"""
from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidates as lex

R, ADJ, OBJ = lex._REPORTERS, lex._ADJECTIVES, lex._OBJECTS
_agent = lambda i: R[i][0]
_alt = lambda i: R[i][1]
_adj = lambda i: ADJ[i]
_adj2 = lambda i: ADJ[(i + 1) % len(ADJ)]
_obj = lambda i: OBJ[i]
_obj2 = lambda i: OBJ[(i + 1) % len(OBJ)]
GENDER = (("king", "queen"), ("father", "mother"), ("brother", "sister"), ("uncle", "aunt"), ("son", "daughter"),
          ("husband", "wife"), ("boy", "girl"), ("man", "woman"), ("prince", "princess"), ("grandfather", "grandmother"),
          ("nephew", "niece"), ("actor", "actress"), ("waiter", "waitress"), ("duke", "duchess"), ("lord", "lady"),
          ("monk", "nun"))
_g = lambda i, male: GENDER[i % len(GENDER)][0 if male else 1]

SPEC = bs.BehaviourSpec(
    task_id="verb_preposition_fc_by_from.swore_vs_withdrew",
    vocabulary=(' by', ' from'),
    generator_role="generate_linked_verb_preposition_fc_by_from_fit_panels",
    answer_role="score_jointly_tokenized_by_versus_from",
    a1=bs.Family("turnout_frame", "turnout_frame_verb_swap", lambda i, pos: f"It turned out the {_agent(i)} {'swore' if pos else 'withdrew'}, apparently,"),
    a2=bs.Family("rumour_frame", "rumour_frame_verb_swap", lambda i, pos: f"The {_adj(i)} rumour was the {_agent(i)} {'swore' if pos else 'withdrew'}, apparently,"),
    p_donor=lambda i, pos: f"It turned out the {_adj2(i)} {_agent(i)} {'swore' if pos else 'withdrew'}, apparently,",
    a1_suffix=lambda i: ", apparently,",
    a2_suffix=lambda i: ", apparently,",
    directions=('swore_to_withdrew', 'withdrew_to_swore'),
    kinds=('swore', 'withdrew'),
    p_generator_role="object_lexical_rewrite",
)

TASK_ID = SPEC.task_id
TASK_SPEC = SPEC.battery_spec()
build_rows, validate_rows, authority_sha256 = SPEC.api()

if __name__ == "__main__":
    rows = build_rows()
    print("authority:", authority_sha256())
    for f in ("A1", "A2", "P", "C"):
        r = next(x for x in rows if x["family"] == f)
        print(f"  {f} base : {r['base_text']!r} -> {r['base_answer']!r}")
        print(f"     donor: {r['donor_text']!r} -> {r['donor_answer']!r}")
