#!/usr/bin/env python3
"""reflexive_number_itself_themselves.it_vs_they -- does a site carry an inanimate pronoun subject's NUMBER (it -> `itself`, they -> `themselves`) across a verb and a parenthetical to the reflexive?

`Near the lantern it stood, of course, by` obliges ` itself`; `Near the lantern they stood, of course, by` obliges ` themselves`. Number family; NEW readout pair itself/themselves (himself/itself and the person-family themselves pairs exist).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v241 batch).
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
    task_id="reflexive_number_itself_themselves.it_vs_they",
    vocabulary=(' itself', ' themselves'),
    generator_role="generate_linked_reflexive_number_itself_themselves_fit_panels",
    answer_role="score_jointly_tokenized_itself_versus_themselves",
    a1=bs.Family("bare_frame", "bare_frame_swap", lambda i, pos: f"Near the {_obj(i)} {'it' if pos else 'they'} stood, of course, by"),
    a2=bs.Family("fell_frame", "fell_frame_swap", lambda i, pos: f"Because {'it' if pos else 'they'} fell on the {_adj(i)} {_agent(i)}, of course, and hurt"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} {'it' if pos else 'they'} stood, of course, by",
    a1_suffix=lambda i: " stood, of course, by",
    a2_suffix=lambda i: f" fell on the {_adj(i)} {_agent(i)}, of course, and hurt",
    directions=('singular_to_plural', 'plural_to_singular'),
    kinds=('singular', 'plural'),
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
