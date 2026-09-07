#!/usr/bin/env python3
"""reciprocal.two_vs_one -- does a site carry a plural subject to the reciprocal versus a reflexive?

`The two pilots near the lantern greeted` obliges ` each` (other); `The pilot ... greeted` obliges ` himself`.

    A1  "The two pilots near the lantern greeted" -> " each" / "The pilot near the lantern greeted" -> " himself"
    A2  "In the end the two pilots had warmly thanked" -> " each"

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, the ninth standing lesson).
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
    task_id="reciprocal.two_vs_one",
    vocabulary=(" each", " himself"),
    generator_role="generate_linked_reciprocal_fit_panels",
    answer_role="score_jointly_tokenized_each_versus_himself",
    a1=bs.Family("bare_frame", "bare_frame_number_swap", lambda i, pos: f"The {'two ' if pos else ''}{_agent(i)}{'s' if pos else ''} near the {_obj(i)} greeted"),
    a2=bs.Family("end_frame", "end_frame_number_swap", lambda i, pos: f"In the end the {'two ' if pos else ''}{_agent(i)}{'s' if pos else ''} had warmly thanked"),
    p_donor=lambda i, pos: f"The {'two ' if pos else ''}{_alt(i)}{'s' if pos else ''} near the {_obj(i)} greeted",
    a1_suffix=lambda i: " greeted",
    a2_suffix=lambda i: " thanked",
    directions=("plural_to_singular", "singular_to_plural"),
    kinds=("two_subjects", "one_subject"),
    p_generator_role="agent_lexical_rewrite",
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
