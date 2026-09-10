#!/usr/bin/env python3
"""correlative_disjoint_either_not.either_vs_not -- a correlative cell whose readout pair is DISJOINT.

`The pilot praised either the lantern` obliges ` or`; `The pilot praised not the lantern` obliges ` but`.

WHY THIS CELL EXISTS. v499 screened six disjoint controls across four constructions under the correlative_pair fit
and every one missed the units floor (best 0.787): that head set reaches its own family and nothing else. So a
control for this family has to come FROM this family -- and v499 also found that every existing correlative cell
shares a readout token with (` and`, ` nor`): but_and and or_and share ` and`, state and both canonicals share ` nor`.
There was therefore no cell in the corpus that is both REACHABLE and DISJOINT, which left v497's shared-token leak
of 0.255 as an upper bound rather than an estimate.
This cell is built to be exactly that: the same construction and the same frame shape as correlative_pair, so the
same sites should be engaged, with a readout pair (` or`, ` but`) that shares NO token with (` and`, ` nor`). Both
cues are single GPT-2 tokens, as are both answers, so base and donor are length-matched -- the check the shared row
builder does not perform (verified missing 2026-09-10; polarity_anyone_someone and polarity_ever_never are
mismatched on 32/32 rows for want of it).
`praised not the lantern but ...` is the contrastive-negation frame and is ordinary English, but it is less frequent
than the both/and pattern, so a capability failure is a registered outcome and is screened on CPU before use.
Authored 2026-09-10 to satisfy the precondition v499 registered.
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
    task_id="correlative_disjoint_either_not.either_vs_not",
    vocabulary=(' or', ' but'),
    generator_role="generate_linked_correlative_disjoint_either_not_fit_panels",
    answer_role="score_jointly_tokenized_or_versus_but",
    a1=bs.Family("praise_frame", "praise_frame_correlative_swap", lambda i, pos: f"The {_agent(i)} praised {'either' if pos else 'not'} the {_obj(i)}"),
    a2=bs.Family("named_frame", "named_frame_correlative_swap", lambda i, pos: f"In the notes the {_agent(i)} named {'either' if pos else 'not'} the {_obj(i)}"),
    p_donor=lambda i, pos: f"The {_adj2(i)} {_agent(i)} praised {'either' if pos else 'not'} the {_obj(i)}",
    a1_suffix=lambda i: f" the {_obj(i)}",
    a2_suffix=lambda i: f" the {_obj(i)}",
    directions=('either_to_not', 'not_to_either'),
    kinds=('either', 'not'),
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
