#!/usr/bin/env python3
"""infinitive_subject_agreement.infinitive_vs_plural -- does a site carry an INFINITIVAL subject's singular agreement (`To lift the crates ... is`) against its plural object attractor versus the bare plural subject?

`To lift the crates near the lantern, of course,` obliges ` is`; `The crates near the lantern, of course,` obliges ` are`. Sibling of gerund_subject_agreement (infinitive instead of gerund).

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v225 batch).
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
    task_id="infinitive_subject_agreement.infinitive_vs_plural",
    vocabulary=(' is', ' are'),
    generator_role="generate_linked_infinitive_subject_agreement_fit_panels",
    answer_role="score_jointly_tokenized_is_versus_are",
    a1=bs.Family("bare_frame", "bare_frame_subject_swap", lambda i, pos: f"{'To lift the' if pos else 'The'} {_obj(i)}s near the {_obj2(i)}, of course,"),
    a2=bs.Family("notes_frame", "notes_frame_subject_swap", lambda i, pos: f"In the notes {'to lift the' if pos else 'the'} {_adj(i)} {_obj(i)}s, of course,"),
    p_donor=lambda i, pos: f"{'To lift the' if pos else 'The'} {_obj(i)}s near the {_adj2(i)} {_obj2(i)}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('singular_to_plural', 'plural_to_singular'),
    kinds=('infinitive', 'plural'),
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
