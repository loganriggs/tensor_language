#!/usr/bin/env python3
"""verb_particle_fp_up_down.woke_vs_calmed -- does a site carry the woke/calmed selection across a LONG parenthetical?

`Near the lantern the pilot woke, as everyone standing in the yard already knew,` obliges ` up`. LENGTH probe: every cell in this corpus separates the cue from the readout by a three-token parenthetical; this one uses eight to ten tokens, so the pooled breadth test can vary sentence LENGTH and not only wording.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-08, v377 length batch).
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
    task_id="verb_particle_fp_up_down.woke_vs_calmed",
    vocabulary=(' up', ' down'),
    generator_role="generate_linked_verb_particle_fp_up_down_fit_panels",
    answer_role="score_jointly_tokenized_up_versus_down",
    a1=bs.Family("long_paren_frame", "long_paren_frame_verb_swap", lambda i, pos: f"Near the {_obj(i)} the {_agent(i)} {'woke' if pos else 'calmed'}, as everyone standing in the yard already knew,"),
    a2=bs.Family("long_paren_storm_frame", "long_paren_storm_frame_verb_swap", lambda i, pos: f"After the {_adj(i)} storm the {_agent(i)} {'woke' if pos else 'calmed'}, as everyone standing in the yard already knew,"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the {_agent(i)} {'woke' if pos else 'calmed'}, as everyone standing in the yard already knew,",
    a1_suffix=lambda i: ", as everyone standing in the yard already knew,",
    a2_suffix=lambda i: ", as everyone standing in the yard already knew,",
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
