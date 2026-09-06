#!/usr/bin/env python3
"""verb_preposition.relied_vs_objected -- is the open subcategorization frame carried?

Every behaviour in this corpus so far reads a variable that is grammatical (number, tense,
polarity, aspect) or discourse-structural (list index, bracket depth). This one reads a LEXICAL
selection: which preposition a particular verb subcategorizes for. It is a different kind of
variable, which is the point of running it -- breadth, not another negation screen.

The cue is deliberately non-local. A bare "The clerk relied" would put the verb in the final
position and any site would trivially carry it; an intervening temporal adverbial that fits both
verbs pushes the cue two tokens back and shares the final token across both sides:

    A1  "The clerk relied for years"   -> " on"   /  "The clerk objected for years"   -> " to"
    A2  "In the hearing the clerk had relied for months" -> " on"  /  "... objected ..." -> " to"
    P   agent rewrite, same final token " years"

If a site transfers this, it is carrying the frame across the adverbial rather than reading the
verb locally.

Authored through `circuit_fast_screen_behaviour_spec`.
"""
from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidates as lex

R = lex._REPORTERS
_agent = lambda i: R[i][0]
_alt = lambda i: R[i][1]

SPEC = bs.BehaviourSpec(
    task_id="verb_preposition.relied_vs_objected",
    vocabulary=(" on", " to"),
    generator_role="generate_linked_verb_preposition_fit_panels",
    answer_role="score_jointly_tokenized_on_versus_to",
    a1=bs.Family("bare_frame", "bare_frame_verb_swap",
                 lambda i, pos: f"The {_agent(i)} {'relied' if pos else 'objected'} for years"),
    a2=bs.Family("report_frame", "report_frame_verb_swap",
                 lambda i, pos: f"In the hearing the {_agent(i)} had {'relied' if pos else 'objected'} for months"),
    p_donor=lambda i, pos: f"The {_alt(i)} {'relied' if pos else 'objected'} for years",
    a1_suffix=lambda i: " years",
    a2_suffix=lambda i: " months",
    directions=("relied_to_objected", "objected_to_relied"),
    kinds=("relied", "objected"),
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
