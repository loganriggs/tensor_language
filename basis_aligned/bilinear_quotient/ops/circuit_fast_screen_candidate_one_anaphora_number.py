#!/usr/bin/env python3
"""one_anaphora_number.crate_vs_crates -- does a site carry an antecedent noun's NUMBER (crate -> one, crates -> ones) across `, not the ADJ` to the anaphoric pro-form?

`The pilot near the lantern wanted the heavy crate, not the light` obliges ` one`; `heavy crates, not the light` obliges ` ones`. Number family, NEW construction: anaphoric one/ones agreeing with a contrasted antecedent.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v219 batch).
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
    task_id="one_anaphora_number.crate_vs_crates",
    vocabulary=(' one', ' ones'),
    generator_role="generate_linked_one_anaphora_number_fit_panels",
    answer_role="score_jointly_tokenized_one_versus_ones",
    a1=bs.Family("bare_frame", "bare_frame_noun_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} wanted the {_adj(i)} {'crate' if pos else 'crates'}, not the {_adj2(i)}"),
    a2=bs.Family("dawn_frame", "dawn_frame_noun_swap", lambda i, pos: f"Before dawn the {_agent(i)} chose the {_adj(i)} {'crate' if pos else 'crates'}, not the {_adj2(i)}"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_obj2(i)} wanted the {_adj(i)} {'crate' if pos else 'crates'}, not the {_adj2(i)}",
    a1_suffix=lambda i: f", not the {_adj2(i)}",
    a2_suffix=lambda i: f", not the {_adj2(i)}",
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
