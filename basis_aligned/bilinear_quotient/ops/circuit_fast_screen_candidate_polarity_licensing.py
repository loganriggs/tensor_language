#!/usr/bin/env python3
"""polarity_licensing.never_vs_often -- does a site carry which polarity context is open?

This lane resolved a shared correlative direction as a `neither` axis rather than correlative
state (`das_correlative_neither_axis_test_v1`, transfer 0.036 / 0.020). That left one question
open on the board: is that direction specific to `neither`, or is it a general negation axis that
would also carry other negative-polarity behaviour?

Nothing in the corpus tests polarity licensing, so the question could not be asked. This is the
missing precondition. A negative licensor obliges an NPI, a frequency adverb obliges its positive
counterpart, and both frames end on the SAME verb token:

    A1  "The pilot has never noticed"    -> " anything"  /  "... has often noticed ..."   -> " something"
    A2  "In the report the pilot had never admitted" -> " anything"  /  "... often ..."   -> " something"

The licensor sits four tokens back and the answer is read at the verb, so a site that transfers
this is carrying the polarity context across intervening material rather than reading a local cue.

If this screens selective, the DAS follow-up is a transfer test: does the jointly-fitted
correlative direction carry polarity here? Carrying it means `neither` was a negation feature all
along; near-zero means the `neither` axis is narrower than negation. Both readings are
informative, which is why the boring branch is registered with the interesting one.

Authored through `circuit_fast_screen_behaviour_spec`.
"""
from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidates as lex

R = lex._REPORTERS
_agent = lambda i: R[i][0]
_alt = lambda i: R[i][1]

SPEC = bs.BehaviourSpec(
    task_id="polarity_licensing.never_vs_often",
    vocabulary=(" anything", " something"),
    generator_role="generate_linked_polarity_licensing_fit_panels",
    answer_role="score_jointly_tokenized_anything_versus_something",
    a1=bs.Family("bare_frame", "bare_frame_licensor_swap",
                 lambda i, pos: f"The {_agent(i)} has {'never' if pos else 'often'} noticed"),
    a2=bs.Family("report_frame", "report_frame_licensor_swap",
                 lambda i, pos: f"In the report the {_agent(i)} had {'never' if pos else 'often'} admitted"),
    p_donor=lambda i, pos: f"The {_alt(i)} has {'never' if pos else 'often'} noticed",
    a1_suffix=lambda i: " noticed",
    a2_suffix=lambda i: " admitted",
    directions=("negative_to_positive", "positive_to_negative"),
    kinds=("negative_licensor", "positive_licensor"),
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
