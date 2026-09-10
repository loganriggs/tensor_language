#!/usr/bin/env python3
"""correlative_open_rec.both_vs_neither -- a DISCHARGED correlative sits nearer than the open one.

`The pilot praised both the crate holding either the rope or the lamp` obliges ` and`; with `neither`, ` nor`.

An adversarial audit reported that every cell in the correlative family contains exactly ONE correlative word, so
"the nearest correlative" and "the still-open correlative" are the same token on 100% of rows -- a head that copies
the paired conjunction of the most recent member of {both, either, neither} reproduces the whole family. Here an
`either ... or` is interposed and DISCHARGED by ` or the lamp`, so the nearest correlative is not the live one and a
recency reader answers from `either` on BOTH sides. Base and donor differ in exactly ONE word.
v489 found the model IS a recency reader in noun_preposition (32/32 rows dropped when a competing cue noun was
interposed) while v475 found it RESISTING an attractor in possessive_number, so the corpus baseline covers neither
outcome here and either result is a finding. A capability collapse is the registered recency outcome, and the
correlative-free control cell is what separates it from "this frame is too heavy".
Authored 2026-09-10 from the audit; screened on CPU before any battery.
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
    task_id="correlative_open_rec.both_vs_neither",
    vocabulary=(' and', ' nor'),
    generator_role="generate_linked_correlative_open_rec_fit_panels",
    answer_role="score_jointly_tokenized_and_versus_nor",
    a1=bs.Family("open_frame", "open_frame_correlative_swap", lambda i, pos: f"The {_agent(i)} praised {'both' if pos else 'neither'} the crate holding either the rope or the lamp"),
    a2=bs.Family("noted_frame", "noted_frame_correlative_swap", lambda i, pos: f"In the notes the {_agent(i)} named {'both' if pos else 'neither'} the crate holding either the rope or the lamp"),
    p_donor=lambda i, pos: f"The {_adj2(i)} {_agent(i)} praised {'both' if pos else 'neither'} the crate holding either the rope or the lamp",
    a1_suffix=lambda i: " the crate holding either the rope or the lamp",
    a2_suffix=lambda i: " the crate holding either the rope or the lamp",
    directions=('both_to_neither', 'neither_to_both'),
    kinds=('both', 'neither'),
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
