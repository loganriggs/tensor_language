#!/usr/bin/env python3
"""possessive_disjoint_my_your.I_vs_you -- a possessive cell whose readout pair is DISJOINT.

`Whenever I checked` obliges ` my`; `Whenever you checked` obliges ` your`.

WHY THIS CELL EXISTS. v507 carried the four-hypothesis protocol to possessive_number.adjacent_antecedent and got
three of four: A1 0.988, A2 1.016, P 0.0814. Row 4 was NOT establishable, because none of five cross-construction
control candidates was reachable -- units 0.474, 0.465, 0.395, 0.329, 0.322, all below the floor and all below the
correlative case's best of 0.787. That made a pattern with v499: a fitted head set reaches only its OWN
construction, so the controls that are DISJOINT in vocabulary are exactly the ones it does not reach, and a
near-zero reading from an unreachable control says nothing.
The fix that worked for correlative was to author a same-construction cell with a disjoint readout pair (v501:
either/not -> ` or`/` but`, reached at units 0.973, inert at -0.021). It does not port directly, because every
possessive cell in the corpus shares ` their` or ` his`. This cell supplies the missing one by changing the cue from
NUMBER to PERSON: the same possessive-determiner slot, the same frame shape as possessive_adjacent, a shared matched
suffix, and a readout pair (` my`, ` your`) that shares no token with (` their`, ` his`). All four tokens are single
GPT-2 tokens and both cues are single tokens, so base and donor are length-matched -- the check the shared row
builder does not perform.
Whether the possessive head set REACHES it is the open question and is not assumed here; that is what the rung after
this one asks, exactly as v501 was screened before v503 relied on it.
Authored 2026-09-10 to satisfy the requirement v507 registered; capability screened on CPU first.
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
    task_id="possessive_disjoint_my_your.I_vs_you",
    vocabulary=(' my', ' your'),
    generator_role="generate_linked_possessive_disjoint_my_your_fit_panels",
    answer_role="score_jointly_tokenized_my_versus_your",
    a1=bs.Family("whenever_frame", "whenever_frame_person_swap", lambda i, pos: f"Whenever {'I' if pos else 'you'} checked"),
    a2=bs.Family("notes_person_frame", "notes_person_frame_person_swap", lambda i, pos: f"In the notes {'I' if pos else 'you'} listed"),
    p_donor=lambda i, pos: f"Whenever {'I' if pos else 'you'} carefully checked",
    a1_suffix=lambda i: " checked",
    a2_suffix=lambda i: " listed",
    directions=('I_to_you', 'you_to_I'),
    kinds=('first', 'second'),
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
