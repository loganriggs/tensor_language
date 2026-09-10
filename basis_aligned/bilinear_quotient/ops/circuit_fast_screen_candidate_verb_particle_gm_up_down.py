#!/usr/bin/env python3
"""verb_particle_gm_up_down.woke_vs_calmed -- the same mapping in a GRAMMATICAL particle slot.

`Near the lantern the pilot woke back` obliges ` up`; `... calmed back` obliges ` down`.

An adversarial audit of this family reported that in all 40 cells the target continuation is
`VERB , <adverbial> , PARTICLE` -- a slot where an English particle cannot appear (*`the pilot woke, of course, up`).
If that is right, every measurement in verb_particle is taken off-distribution, in a position where no particle
selection is syntactically licensed and only the verb-particle collocation prior can act. That matters beyond this
family: v467 used verb_particle as the SECOND CONSTRUCTION validating the pooled distinctness rule that R2, R7 and
the count of 139 rest on.
This cell keeps the mapping, the frame and the lexicon and changes only the slot: ` back` is the shared matched
suffix on both sides, so the particle now lands where `wake back up` and `calm back down` are ordinary English.
Authored 2026-09-10 in response to the audit; capability is screened on CPU before any battery.
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
    task_id="verb_particle_gm_up_down.woke_vs_calmed",
    vocabulary=(' up', ' down'),
    generator_role="generate_linked_verb_particle_gm_up_down_fit_panels",
    answer_role="score_jointly_tokenized_up_versus_down",
    a1=bs.Family("grammatical_frame", "grammatical_frame_verb_swap", lambda i, pos: f"Near the {_obj(i)} the {_agent(i)} {'woke' if pos else 'calmed'} back"),
    a2=bs.Family("grammatical_storm_frame", "grammatical_storm_frame_verb_swap", lambda i, pos: f"After the {_adj(i)} storm the {_agent(i)} {'woke' if pos else 'calmed'} back"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the {_agent(i)} {'woke' if pos else 'calmed'} back",
    a1_suffix=lambda i: " back",
    a2_suffix=lambda i: " back",
    directions=('woke_to_calmed', 'calmed_to_woke'),
    kinds=('woke', 'calmed'),
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
