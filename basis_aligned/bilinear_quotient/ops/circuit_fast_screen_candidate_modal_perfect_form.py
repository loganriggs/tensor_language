#!/usr/bin/env python3
"""modal_perfect_form.can_vs_has -- does a site carry a modal (can -> bare) vs perfect auxiliary (has -> participle) past an adverb to the verb form?

`The pilot near the lantern can surely` obliges ` carry`; `has surely` obliges ` carried`. Verb-form sibling of did_has_negation with a modal cue.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v213 batch).
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
    task_id="modal_perfect_form.can_vs_has",
    vocabulary=(' carry', ' carried'),
    generator_role="generate_linked_modal_perfect_form_fit_panels",
    answer_role="score_jointly_tokenized_carry_versus_carried",
    a1=bs.Family("bare_frame", "bare_frame_aux_swap", lambda i, pos: f"The {_agent(i)} near the {_obj(i)} {'can' if pos else 'has'} surely"),
    a2=bs.Family("now_frame", "now_frame_aux_swap", lambda i, pos: f"By now the {_adj(i)} {_agent(i)} {'can' if pos else 'has'} surely"),
    p_donor=lambda i, pos: f"The {_agent(i)} near the {_obj2(i)} {'can' if pos else 'has'} surely",
    a1_suffix=lambda i: " surely",
    a2_suffix=lambda i: " surely",
    directions=('bare_to_participle', 'participle_to_bare'),
    kinds=('bare', 'participle'),
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
