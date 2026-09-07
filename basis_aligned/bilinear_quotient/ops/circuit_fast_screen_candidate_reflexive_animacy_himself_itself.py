#!/usr/bin/env python3
"""reflexive_animacy_himself_itself.pilot_vs_crate -- does a site carry a subject's ANIMACY (pilot -> `himself`, crate -> `itself`) across a PP, a verb phrase and a parenthetical to the reflexive after `all by`?

`The pilot near the lantern stood there, of course, all by` obliges ` himself`; `The crate near the lantern stood there, of course, all by` obliges ` itself`. Animacy family (he/it, who/which exist), NEW readout token pair himself/itself (reflexive rather than nominative/relative).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v233 batch).
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
    task_id="reflexive_animacy_himself_itself.pilot_vs_crate",
    vocabulary=(' himself', ' itself'),
    generator_role="generate_linked_reflexive_animacy_himself_itself_fit_panels",
    answer_role="score_jointly_tokenized_himself_versus_itself",
    a1=bs.Family("bare_frame", "bare_frame_subject_swap", lambda i, pos: f"The {_agent(i) if pos else 'crate'} near the {_obj(i)} stood there, of course, all by"),
    a2=bs.Family("alone_frame", "alone_frame_subject_swap", lambda i, pos: f"Left alone, the {_adj(i)} {_agent(i) if pos else 'crate'} sat, of course, by"),
    p_donor=lambda i, pos: f"The {_agent(i) if pos else 'crate'} near the {_adj2(i)} {_obj(i)} stood there, of course, all by",
    a1_suffix=lambda i: ", of course, all by",
    a2_suffix=lambda i: ", of course, by",
    directions=('animate_to_inanimate', 'inanimate_to_animate'),
    kinds=('animate', 'inanimate'),
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
