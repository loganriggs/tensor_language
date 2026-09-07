#!/usr/bin/env python3
"""agreement_lifts_lift.he_vs_they -- does a site carry a pronoun subject's NUMBER (he -> `lifts`, they -> `lift`) across a parenthetical and an adverb to the lexical verb's agreement suffix?

`Near the lantern he, of course, always` obliges ` lifts`; `Near the lantern they, of course, always` obliges ` lift`. Number family (is/are, was/were, has/have exist as agreement readouts), NEW readout token pair lifts/lift (lexical-verb -s agreement).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v239 batch).
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
    task_id="agreement_lifts_lift.he_vs_they",
    vocabulary=(' lifts', ' lift'),
    generator_role="generate_linked_agreement_lifts_lift_fit_panels",
    answer_role="score_jointly_tokenized_lifts_versus_lift",
    a1=bs.Family("bare_frame", "bare_frame_pronoun_swap", lambda i, pos: f"Near the {_obj(i)} {'he' if pos else 'they'}, of course, always"),
    a2=bs.Family("tired_frame", "tired_frame_pronoun_swap", lambda i, pos: f"Because the {_adj(i)} {_agent(i)} is tired, {'he' if pos else 'they'}, of course, rarely"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} {'he' if pos else 'they'}, of course, always",
    a1_suffix=lambda i: ", of course, always",
    a2_suffix=lambda i: ", of course, rarely",
    directions=('singular_to_plural', 'plural_to_singular'),
    kinds=('singular', 'plural'),
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
