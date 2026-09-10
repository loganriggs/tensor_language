#!/usr/bin/env python3
"""noun_preposition_rec_interest.interest_vs_respect -- the NEAREST noun is the WRONG one, on purpose.

`The pilot showed interest, despite the ongoing respect, of course,` obliges ` in`;
`The pilot showed respect, despite the ongoing interest, of course,` obliges ` for`.

An adversarial audit reported that in all 9 cells of this family the cue is the LINEARLY NEAREST noun to the readout,
at a constant offset of -5, so "the noun five back" and "the noun that governs the PP" have never been separated
anywhere in the family. Here the OTHER member of the cue pair is interposed, nearer to the readout than the governor.
A RECENCY READER AND A GOVERNANCE READER THEREFORE PREDICT OPPOSITE TOKENS ON EVERY ROW: recency gives ` for` on the
base and ` in` on the donor, exactly inverted, so `g.prepare(valid_only=True)` would drop essentially the whole panel
and the CPU capability screen alone adjudicates before any GPU time is spent.
THE TWO BASELINES THIS CORPUS HOLDS CONFLICT, WHICH IS WHY THIS IS WORTH RUNNING. The standing note that these
circuits are RECENCY readers predicts the collapse; v475, where the model resisted a number attractor in
possessive_number on 32 of 32 rows, predicts it holds. A branch whose TRUE agrees with the baseline tests nothing --
here no single baseline covers both outcomes, so either result is a finding.
Registered in advance: base and donor differ in TWO positions, because a permutation of the pair is the only way to
put the wrong noun nearest while holding the token multiset. That is the design, as in the coordinated-verb cell,
not an oversight. A capability collapse is a REGISTERED OUTCOME here and is the recency finding, not an unperformable
stimulus.
Authored 2026-09-10 from the audit; screened on CPU first.
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
    task_id="noun_preposition_rec_interest.interest_vs_respect",
    vocabulary=(' in', ' for'),
    generator_role="generate_linked_noun_preposition_rec_interest_fit_panels",
    answer_role="score_jointly_tokenized_in_versus_for",
    a1=bs.Family("distractor_frame", "distractor_frame_noun_swap", lambda i, pos: f"The {_agent(i)} showed {'interest' if pos else 'respect'}, despite the ongoing {'respect' if pos else 'interest'}, of course,"),
    a2=bs.Family("dawn_distractor_frame", "dawn_distractor_frame_noun_swap", lambda i, pos: f"Before dawn the {_adj(i)} {_agent(i)} showed {'interest' if pos else 'respect'}, despite the ongoing {'respect' if pos else 'interest'}, of course,"),
    p_donor=lambda i, pos: f"The {_adj2(i)} {_agent(i)} showed {'interest' if pos else 'respect'}, despite the ongoing {'respect' if pos else 'interest'}, of course,",
    a1_suffix=lambda i: ", of course,",
    a2_suffix=lambda i: ", of course,",
    directions=('in_to_for', 'for_to_in'),
    kinds=('interest', 'respect'),
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
