#!/usr/bin/env python3
"""reciprocal_lenmatched.two_vs_lone -- the reciprocal task with clean and corrupted the SAME token length.

`The two leaders near the window greeted` obliges ` each`; `The lone leader near the window greeted` obliges
` himself`. Both sides are 7 GPT-2 tokens.

WHY. A corpus-wide sweep found 102 of 465 tasks with a clean/corrupted token-length mismatch, and NINE of those are
counted. reciprocal is the most severe: mismatched on 32 of 32 rows, and BY CONSTRUCTION rather than by accident --
its cue is number, so the plural side spends an extra token (`The two leaders` against `The leader`). Patching
happens at the final token, which is aligned, but every position before it is shifted by one on every row.
Position alignment between clean and corrupted prompts is standard practice, and an older lane in this repo already
carries an `equal_token_length` flag that the fast-screen builder dropped, so this is a regression rather than a
discovery. This cell is the repair: the singular side takes ` lone`, which costs exactly the token the plural side
spends on ` two`, so the two sides match at 7 tokens with the cue and the answer pair unchanged.
Authored 2026-09-10 to measure what the misalignment actually costs; capability screened on CPU first.
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
    task_id="reciprocal_lenmatched.two_vs_lone",
    vocabulary=(' each', ' himself'),
    generator_role="generate_linked_reciprocal_lenmatched_fit_panels",
    answer_role="score_jointly_tokenized_each_versus_himself",
    a1=bs.Family("greeted_frame", "greeted_frame_number_swap", lambda i, pos: f"The {'two ' + _agent(i) + 's' if pos else 'lone ' + _agent(i)} near the {_obj(i)} greeted"),
    a2=bs.Family("notes_frame", "notes_frame_number_swap", lambda i, pos: f"In the notes the {'two ' + _agent(i) + 's' if pos else 'lone ' + _agent(i)} praised"),
    p_donor=lambda i, pos: f"The {'two ' + _agent(i) + 's' if pos else 'lone ' + _agent(i)} near the {_obj2(i)} greeted",
    a1_suffix=lambda i: " greeted",
    a2_suffix=lambda i: " praised",
    directions=('two_to_lone', 'lone_to_two'),
    kinds=('two', 'lone'),
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
