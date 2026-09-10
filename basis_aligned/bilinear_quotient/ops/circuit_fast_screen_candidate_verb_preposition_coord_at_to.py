#!/usr/bin/env python3
"""verb_preposition_coord_at_to.aimedlast_vs_appealedlast -- lexeme PRESENCE or GOVERNANCE?

`Near the lantern the pilot appealed and aimed, of course,` obliges ` at`;
`Near the lantern the pilot aimed and appealed, of course,` obliges ` to`.

BOTH SIDES CONTAIN THE IDENTICAL TOKEN MULTISET. Only the ORDER of the two coordinated cue verbs differs, so a
direction that encodes "which verb lexemes are present in the recent window" has nothing to separate and must score
at complement level; a direction that encodes which verb GOVERNS the upcoming PP carries it. Every other cell in
this family has exactly one verb in the window, always both nearest and governing, so presence and governance have
never been separated anywhere in the corpus -- and this project's own note records these circuits as recency
readers, which is a governance-adjacent claim that has never been tested against presence.
Registered in advance: base and donor differ in TWO positions here rather than one, because a permutation is the
only way to hold the multiset fixed. That is a deliberate departure from the family's one-word convention and it is
the point of the cell, not an oversight.
The red team that proposed this design flagged it as the highest capability risk in its report: coordinated cue
verbs are untested here and the model may attach the FIRST verb's preposition, or prefer one answer on both sides.
A capability failure is therefore a real outcome and is registered as such rather than treated as unperformable.
Authored 2026-09-10; capability screened on CPU before any battery.
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
    task_id="verb_preposition_coord_at_to.aimedlast_vs_appealedlast",
    vocabulary=(' at', ' to'),
    generator_role="generate_linked_verb_preposition_coord_at_to_fit_panels",
    answer_role="score_jointly_tokenized_at_versus_to",
    a1=bs.Family("coord_frame", "coord_frame_order_swap", lambda i, pos: f"Near the {_obj(i)} the {_agent(i)} {'appealed and aimed' if pos else 'aimed and appealed'}, of course,"),
    a2=bs.Family("coord_storm_frame", "coord_storm_frame_order_swap", lambda i, pos: f"After the {_adj(i)} storm the {_agent(i)} {'appealed and aimed' if pos else 'aimed and appealed'}, of course,"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the {_agent(i)} {'appealed and aimed' if pos else 'aimed and appealed'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('aimedlast_to_appealedlast', 'appealedlast_to_aimedlast'),
    kinds=('aimedlast', 'appealedlast'),
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
