#!/usr/bin/env python3
"""demonstrative_number.these_vs_this -- does a site carry a demonstrative's number across an adjective to the pro-form?

`these bright` obliges ` ones`, `this bright` obliges ` one`; the cue is the determiner two tokens back and the answer is read at the adjective.

    A1  "The pilot chose these bright" -> " ones" / "... this bright" -> " one"
    A2  "Of all the lanterns the pilot liked those bright" -> " ones"

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
    task_id="demonstrative_number.these_vs_this",
    vocabulary=(" ones", " one"),
    generator_role="generate_linked_demonstrative_number_fit_panels",
    answer_role="score_jointly_tokenized_ones_versus_one",
    a1=bs.Family("bare_frame", "bare_frame_demonstrative_swap", lambda i, pos: f"The {_agent(i)} chose {'these' if pos else 'this'} {_adj(i)}"),
    a2=bs.Family("partitive_frame", "partitive_frame_demonstrative_swap", lambda i, pos: f"Of all the {_obj(i)}s the {_agent(i)} liked {'those' if pos else 'that'} {_adj(i)}"),
    p_donor=lambda i, pos: f"The {_alt(i)} chose {'these' if pos else 'this'} {_adj(i)}",
    a1_suffix=lambda i: f" {_adj(i)}",
    a2_suffix=lambda i: f" {_adj(i)}",
    directions=("plural_to_singular", "singular_to_plural"),
    kinds=("plural_demonstrative", "singular_demonstrative"),
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
