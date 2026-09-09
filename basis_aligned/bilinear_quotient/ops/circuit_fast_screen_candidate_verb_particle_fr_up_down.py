#!/usr/bin/env python3
"""verb_particle_fr_up_down.woke_vs_calmed -- does a site carry the woke/calmed selection under this structural stress?

`The pilot who trained the cadets woke, of course,` obliges ` up`. STRESS cell: six pooled rungs have found no shape a single direction cannot cover, so these vary axes the corpus has NEVER varied -- a relative clause inside the subject, a conjoined subject, and a full subordinate CLAUSE between the cue and the readout instead of a parenthetical phrase. A cell that a pooled direction cannot cover is the point of authoring it.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-08, v383 stress batch).
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
    task_id="verb_particle_fr_up_down.woke_vs_calmed",
    vocabulary=(' up', ' down'),
    generator_role="generate_linked_verb_particle_fr_up_down_fit_panels",
    answer_role="score_jointly_tokenized_up_versus_down",
    a1=bs.Family("relclause_frame", "relclause_frame_verb_swap", lambda i, pos: f"The {_agent(i)} who trained the cadets {'woke' if pos else 'calmed'}, of course,"),
    a2=bs.Family("relclause_adj_frame", "relclause_adj_frame_verb_swap", lambda i, pos: f"The {_adj(i)} {_agent(i)} who trained the cadets {'woke' if pos else 'calmed'}, of course,"),
    p_donor=lambda i, pos: f"The {_adj2(i)} {_agent(i)} who trained the cadets {'woke' if pos else 'calmed'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
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
