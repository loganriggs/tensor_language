#!/usr/bin/env python3
"""coordination_agreement.and_vs_or -- the CONNECTIVE sets agreement, not the nouns.

Both coordinated nouns are SINGULAR in every row. Only the connective changes, and it alone
decides the auxiliary: `and` obliges a plural, `or` obliges a singular.

    A1  "The pilot and the sailor"                -> " were"  /  "... or ..."  -> " was"
    A2  "In the report the pilot and the guard"   -> " were"  /  "... or ..."  -> " was"
    P   swap the FIRST noun; the final token (the second noun) is untouched

This is deliberately distinguished from two neighbours.

`subject_verb_number_agreement` varies the NUMBER OF THE HEAD NOUN; here every noun is singular
and the number of the subject is constructed by the connective. `correlative_pair.both_vs_neither`
and its siblings put the connective on the ANSWER side -- the model predicts ` and` / ` or`. Here
the connective is the CUE and agreement is the answer, so the inference runs the other way.

The cue also sits four tokens back from the read position, so a site that transfers this is
carrying a constructed number feature across the second noun phrase rather than reading a local
morphological cue -- there is no local cue to read, since `sailor` is singular either way.

Authored through `circuit_fast_screen_behaviour_spec`.
"""
from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidates as lex

R = lex._REPORTERS
_first = lambda i: R[i][0]
_alt_first = lambda i: R[i][1]
_second = lambda i: R[(i + 13) % len(R)][0]
_second_b = lambda i: R[(i + 23) % len(R)][0]

SPEC = bs.BehaviourSpec(
    task_id="coordination_agreement.and_vs_or",
    vocabulary=(" were", " was"),
    generator_role="generate_linked_coordination_agreement_fit_panels",
    answer_role="score_jointly_tokenized_were_versus_was",
    a1=bs.Family("bare_frame", "bare_frame_connective_swap",
                 lambda i, pos: f"The {_first(i)} {'and' if pos else 'or'} the {_second(i)}"),
    a2=bs.Family("report_frame", "report_frame_connective_swap",
                 lambda i, pos: f"In the report the {_first(i)} {'and' if pos else 'or'} the {_second_b(i)}"),
    p_donor=lambda i, pos: f"The {_alt_first(i)} {'and' if pos else 'or'} the {_second(i)}",
    a1_suffix=lambda i: f" {_second(i)}",
    a2_suffix=lambda i: f" {_second_b(i)}",
    directions=("and_to_or", "or_to_and"),
    kinds=("and_plural", "or_singular"),
    p_generator_role="first_noun_lexical_rewrite",
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
