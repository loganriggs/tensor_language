#!/usr/bin/env python3
"""verb_particle_fd_up_down.woke_vs_calmed -- does a site carry the woke/calmed selection in the FRAME-D sentence shape?

`Nobody doubted the pilot woke, honestly,` obliges ` up`; `... calmed, honestly,` obliges ` down`. FRAME-CAPACITY test: this mapping pair already has counted circuits in frames A, B and C, and frames D and E ask how many distinct frames one pair can carry before the copies start fusing with each other.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-08, v361 frame-capacity batch).
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
    task_id="verb_particle_fd_up_down.woke_vs_calmed",
    vocabulary=(' up', ' down'),
    generator_role="generate_linked_verb_particle_fd_up_down_fit_panels",
    answer_role="score_jointly_tokenized_up_versus_down",
    a1=bs.Family("doubt_frame", "doubt_frame_verb_swap", lambda i, pos: f"Nobody doubted the {_agent(i)} {'woke' if pos else 'calmed'}, honestly,"),
    a2=bs.Family("witness_frame", "witness_frame_verb_swap", lambda i, pos: f"The {_adj(i)} witness swore the {_agent(i)} {'woke' if pos else 'calmed'}, honestly,"),
    p_donor=lambda i, pos: f"Nobody doubted the {_adj2(i)} {_agent(i)} {'woke' if pos else 'calmed'}, honestly,",
    a1_suffix=lambda i: ", honestly,",
    a2_suffix=lambda i: ", honestly,",
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
