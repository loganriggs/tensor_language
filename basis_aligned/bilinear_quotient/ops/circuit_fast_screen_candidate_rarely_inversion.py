#!/usr/bin/env python3
"""rarely_inversion.rarely_vs_often -- does a site carry a fronted adverb's NEGATIVE force (Rarely vs Often) past a parenthetical to subject-auxiliary inversion?

`Near the lantern, rarely, of course,` obliges ` did`; `often, of course,` obliges ` the`. Inversion-family sibling of negative_inversion (never/often before).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v215 batch).
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
    task_id="rarely_inversion.rarely_vs_often",
    vocabulary=(' did', ' the'),
    generator_role="generate_linked_rarely_inversion_fit_panels",
    answer_role="score_jointly_tokenized_did_versus_the",
    a1=bs.Family("bare_frame", "bare_frame_adverb_swap", lambda i, pos: f"Near the {_obj(i)}, {'rarely' if pos else 'often'}, of course,"),
    a2=bs.Family("years_frame", "years_frame_adverb_swap", lambda i, pos: f"{'Rarely' if pos else 'Often'} in all the {_adj(i)} years, of course,"),
    p_donor=lambda i, pos: f"Near the {_obj2(i)}, {'rarely' if pos else 'often'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('negative_to_positive', 'positive_to_negative'),
    kinds=('negative', 'positive'),
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
