#!/usr/bin/env python3
"""verb_particle_away_down_r.shied_vs_settled -- does a site carry a verb's PARTICLE SELECTION (shied -> `away`, settled -> `down`) across a parenthetical to the particle?

`Near the lantern the pilot shied, of course,` obliges ` away`; `... settled, of course,` obliges ` down`. DESIGNED-TO-FUSE cell: both cue -> token mappings are REUSED VERBATIM from counted circuits (shied -> ` away` from away_up and settled -> ` down` from out_down), while the readout PAIR is recombined. The cue-overlap account predicts this fuses with both donors; the token-pair gate predicts it is a fresh behaviour. It is the counterpart of the v319/v321 twins, which hold the pair fixed and change the cues.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-08, v325 batch).
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
    task_id="verb_particle_away_down_r.shied_vs_settled",
    vocabulary=(' away', ' down'),
    generator_role="generate_linked_verb_particle_away_down_r_fit_panels",
    answer_role="score_jointly_tokenized_away_versus_down",
    a1=bs.Family("bare_frame", "bare_frame_verb_swap", lambda i, pos: f"Near the {_obj(i)} the {_agent(i)} {'shied' if pos else 'settled'}, of course,"),
    a2=bs.Family("storm_frame", "storm_frame_verb_swap", lambda i, pos: f"After the {_adj(i)} storm the {_agent(i)} {'shied' if pos else 'settled'}, of course,"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the {_agent(i)} {'shied' if pos else 'settled'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('shied_to_settled', 'settled_to_shied'),
    kinds=('shied', 'settled'),
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
