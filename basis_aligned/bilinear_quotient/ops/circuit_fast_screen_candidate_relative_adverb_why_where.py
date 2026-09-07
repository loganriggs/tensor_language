#!/usr/bin/env python3
"""relative_adverb_why_where.reason_vs_place -- does a site carry a head noun's SEMANTIC CLASS (reason -> `why`, place -> `where`) across a parenthetical to the relative adverb?

`The pilot near the lantern recalled the reason, of course,` obliges ` why`; `... recalled the place, of course,` obliges ` where`. NEW family (relative-adverb selection by a head noun; the wh family reads argument/adjunct and animacy, never the reason/place choice).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v235 batch).
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
    task_id="relative_adverb_why_where.reason_vs_place",
    vocabulary=(' why', ' where'),
    generator_role="generate_linked_relative_adverb_why_where_fit_panels",
    answer_role="score_jointly_tokenized_why_versus_where",
    a1=bs.Family("bare_frame", "bare_frame_head_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} recalled the {'reason' if pos else 'place'}, of course,"),
    a2=bs.Family("told_frame", "told_frame_head_swap", lambda i, pos: f"Nobody told the {_adj(i)} {_agent(i)} the {'reason' if pos else 'place'}, of course,"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_adj2(i)} {_obj(i)} recalled the {'reason' if pos else 'place'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('reason_to_place', 'place_to_reason'),
    kinds=('reason', 'place'),
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
