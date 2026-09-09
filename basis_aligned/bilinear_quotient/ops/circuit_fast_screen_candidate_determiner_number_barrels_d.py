#!/usr/bin/env python3
"""determiner_number_barrels_d.many_vs_each -- does a site carry a DETERMINER's number selection (many -> `barrels`, each -> `barrel`) across an intervening adjective to the noun?

`Near the lantern the pilot inspected many cracked` obliges ` barrels`; `... each cracked` obliges ` barrel`. SECOND ATTEMPT at a sibling for determiner_number_crates, and the change is principled rather than cosmetic: the pair that WORKED (several / each) contrasts a plural quantifier with a DISTRIBUTIVE singular, while the two that failed at v405 (many / one, numerous / another) used non-distributive singulars -- `one` and `another` are discourse-anaphoric as much as they are number-marking, so the base and donor may differ in more than number. These keep the distributive contrast and change only the nouns.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-08, v409 distributive sibling batch).
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
    task_id="determiner_number_barrels_d.many_vs_each",
    vocabulary=(' barrels', ' barrel'),
    generator_role="generate_linked_determiner_number_barrels_d_fit_panels",
    answer_role="score_jointly_tokenized_barrels_versus_barrel",
    a1=bs.Family("inspect_frame", "inspect_frame_det_swap", lambda i, pos: f"Near the {_obj(i)} the {_agent(i)} inspected {'many' if pos else 'each'} cracked"),
    a2=bs.Family("storm_frame", "storm_frame_det_swap", lambda i, pos: f"After the {_adj(i)} storm the {_agent(i)} inspected {'many' if pos else 'each'} cracked"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the {_agent(i)} inspected {'many' if pos else 'each'} cracked",
    a1_suffix=lambda i: " cracked",
    a2_suffix=lambda i: " cracked",
    directions=('many_to_each', 'each_to_many'),
    kinds=('many', 'each'),
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
