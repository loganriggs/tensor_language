#!/usr/bin/env python3
"""gender_possessive_his_her.he_vs_she -- does a site carry a subject pronoun's GENDER (he -> `his`, she -> `her`) across a verb phrase and a parenthetical to the possessive determiner?

`Near the lantern he kept the crate, of course, in` obliges ` his`; `Near the lantern she kept the crate, of course, in` obliges ` her`. Gender readout (himself/herself is the only screened gender pair); NEW pair his/her.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v247 batch).
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
    task_id="gender_possessive_his_her.he_vs_she",
    vocabulary=(' his', ' her'),
    generator_role="generate_linked_gender_possessive_his_her_fit_panels",
    answer_role="score_jointly_tokenized_his_versus_her",
    a1=bs.Family("bare_frame", "bare_frame_swap", lambda i, pos: f"Near the {_obj(i)} {'he' if pos else 'she'} kept the crate, of course, in"),
    a2=bs.Family("paid_frame", "paid_frame_swap", lambda i, pos: f"After {'he' if pos else 'she'} paid the {_adj(i)} {_agent(i)}, of course, for"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} {'he' if pos else 'she'} kept the crate, of course, in",
    a1_suffix=lambda i: " kept the crate, of course, in",
    a2_suffix=lambda i: f" paid the {_adj(i)} {_agent(i)}, of course, for",
    directions=('masculine_to_feminine', 'feminine_to_masculine'),
    kinds=('masculine', 'feminine'),
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
