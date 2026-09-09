#!/usr/bin/env python3
"""verb_particle_fc_down_in.hunkered_vs_caved -- does a site carry a verb's PARTICLE SELECTION (hunkered -> `down`, caved -> `in`) in the FRAME-C sentence shape?

`It turned out the pilot hunkered, apparently,` obliges ` down`; `... caved, apparently,` obliges ` in`. Frame-C copy of a counted frame-A circuit's cue mappings, authored under the measured (frame, mapping) account.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-08, v349 frame-C batch 2).
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
    task_id="verb_particle_fc_down_in.hunkered_vs_caved",
    vocabulary=(' down', ' in'),
    generator_role="generate_linked_verb_particle_fc_down_in_fit_panels",
    answer_role="score_jointly_tokenized_down_versus_in",
    a1=bs.Family("turnout_frame", "turnout_frame_verb_swap", lambda i, pos: f"It turned out the {_agent(i)} {'hunkered' if pos else 'caved'}, apparently,"),
    a2=bs.Family("rumour_frame", "rumour_frame_verb_swap", lambda i, pos: f"The {_adj(i)} rumour was the {_agent(i)} {'hunkered' if pos else 'caved'}, apparently,"),
    p_donor=lambda i, pos: f"It turned out the {_adj2(i)} {_agent(i)} {'hunkered' if pos else 'caved'}, apparently,",
    a1_suffix=lambda i: ", apparently,",
    a2_suffix=lambda i: ", apparently,",
    directions=('hunkered_to_caved', 'caved_to_hunkered'),
    kinds=('hunkered', 'caved'),
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
