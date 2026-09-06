#!/usr/bin/env python3
"""correlative_pair.both_vs_either -- a correlative contrast with NO negative member.

Authored to break a confound in this lane's own DAS result. `correlative_pair.both_vs_neither`
and `correlative_state.either_vs_neither` are both selective at resid:18, and one rank-1
direction serves both at 0.975 / 0.963 — but both pairs put `neither` (answering ` nor`) on their
donor side, so that direction may encode `neither` rather than correlative state.

This pair has no negative member: `both` obliges ` and`, `either` obliges ` or`.

    A1  "The leader praised both the guide"   -> " and"   /  "... either ..." -> " or"
    A2  "In the notes the leader named both the judge" -> " and"  /  "... either ..." -> " or"

Under the `neither`-axis reading, a rank-1 direction separating {both, either} from {neither}
places `both` and `either` on the SAME side, so the jointly-fitted direction should transfer here
at near zero. That is the discriminating prediction; the screen below establishes the behaviour
is carried at all, which is the precondition for testing it.

Authored through `circuit_fast_screen_behaviour_spec`.
"""
from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidates as lex

R = lex._REPORTERS
_agent = lambda i: R[i][0]
_alt = lambda i: R[i][1]
_obj1 = lambda i: R[(i + 11) % len(R)][0]
_obj2 = lambda i: R[(i + 19) % len(R)][0]

SPEC = bs.BehaviourSpec(
    task_id="correlative_pair.both_vs_either",
    vocabulary=(" and", " or"),
    generator_role="generate_linked_both_either_fit_panels",
    answer_role="score_jointly_tokenized_and_versus_or",
    a1=bs.Family("bare_frame", "bare_frame_conjunction_swap",
                 lambda i, pos: f"The {_agent(i)} praised {'both' if pos else 'either'} the {_obj1(i)}"),
    a2=bs.Family("report_frame", "report_frame_conjunction_swap",
                 lambda i, pos: f"In the notes the {_agent(i)} named {'both' if pos else 'either'} the {_obj2(i)}"),
    p_donor=lambda i, pos: f"The {_alt(i)} praised {'both' if pos else 'either'} the {_obj1(i)}",
    a1_suffix=lambda i: f" {_obj1(i)}",
    a2_suffix=lambda i: f" {_obj2(i)}",
    directions=("both_to_either", "either_to_both"),
    kinds=("both", "either"),
    p_generator_role="noun_lexical_rewrite",
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
