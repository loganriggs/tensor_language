#!/usr/bin/env python3
"""additive_scope_lenmatched.notonly_vs_adverb -- the additive-scope task with both sides the SAME token length.

`The leader not only wrote the transcript` obliges ` but`; `The leader warmly wrote the transcript` obliges ` and`.
Both sides are 7 GPT-2 tokens.

WHY. The corpus sweep found additive_scope mismatched on 28 of 32 rows and it IS counted -- the second-worst counted
case after reciprocal, which v541 priced at zero. I registered that pricing as ONE measurement rather than a result
covering all nine counted mismatched tasks, so this is the second.
The mismatch has a simple cause: the cue ` not only` is two GPT-2 tokens while most of the adverbs on the other side
are one, so the four rows that happen to use a two-token adverb are already matched and the other twenty-eight are
not. The repair keeps the design exactly and draws the adverb only from two-token options -- warmly, bravely,
sternly, coolly, deftly, primly, flatly -- so every row matches at 7 tokens with the cue contrast and the answer pair
(` but`, ` and`) unchanged. P varies the agent noun rather than inserting an adjective, so the P pair stays matched
too.
Authored 2026-09-10; capability screened on CPU before use.
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
_ADV = ('warmly', 'bravely', 'sternly', 'coolly', 'deftly', 'primly', 'flatly')
GENDER = (("king", "queen"), ("father", "mother"), ("brother", "sister"), ("uncle", "aunt"), ("son", "daughter"),
          ("husband", "wife"), ("boy", "girl"), ("man", "woman"), ("prince", "princess"), ("grandfather", "grandmother"),
          ("nephew", "niece"), ("actor", "actress"), ("waiter", "waitress"), ("duke", "duchess"), ("lord", "lady"),
          ("monk", "nun"))
_g = lambda i, male: GENDER[i % len(GENDER)][0 if male else 1]

SPEC = bs.BehaviourSpec(
    task_id="additive_scope_lenmatched.notonly_vs_adverb",
    vocabulary=(' but', ' and'),
    generator_role="generate_linked_additive_scope_lenmatched_fit_panels",
    answer_role="score_jointly_tokenized_but_versus_and",
    a1=bs.Family("wrote_frame", "wrote_frame_scope_swap", lambda i, pos: f"The {_agent(i)} {'not only' if pos else _ADV[i % len(_ADV)]} wrote the {_obj(i)}"),
    a2=bs.Family("signed_frame", "signed_frame_scope_swap", lambda i, pos: f"In the notes the {_agent(i)} {'not only' if pos else _ADV[i % len(_ADV)]} signed the {_obj(i)}"),
    p_donor=lambda i, pos: f"The {_agent((i + 7) % 32)} {'not only' if pos else _ADV[i % len(_ADV)]} wrote the {_obj(i)}",
    a1_suffix=lambda i: f" wrote the {_obj(i)}",
    a2_suffix=lambda i: f" signed the {_obj(i)}",
    directions=('notonly_to_adverb', 'adverb_to_notonly'),
    kinds=('notonly', 'adverb'),
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
