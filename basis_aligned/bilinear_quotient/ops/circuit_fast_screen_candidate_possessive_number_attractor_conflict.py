#!/usr/bin/env python3
"""possessive_number_attractor_conflict.plural_vs_singular -- does the site read the SUBJECT's number, or the NEAREST noun's?

`Near the desk the pilots near the crates lost` obliges ` their`; `... the pilot near the crates lost` obliges ` his`.
The attractor is held PLURAL on both sides, so on the DONOR side the nearest noun (` crates`) carries the opposite
number from the correct answer. A red-team audit of this family found that in all seven existing cells the
intervener's number is held constant and never covaries with the answer, so `number of the subject`, `number of the
nearest noun` and `number of the nearest human noun` have never been separated -- and the one cell that would have
separated them, animate_attractor, is on the board as a donor-side capability failure. This cell is the contract-clean
version: base and donor differ in exactly ONE word, and both `pilot` and `pilots` are single GPT-2 tokens, so the two
sides are length-matched (the shared row builder checks the final token but NOT total length).
A recency reader answers ` their` on both sides, so its signature is a donor-side capability collapse on the CPU
screen, BEFORE any GPU time. That reading is registered here in advance so a null is not written off as unperformable.

Authored through `circuit_fast_screen_behaviour_spec` (Claude, 2026-09-07, v241 batch).
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
    task_id="possessive_number_attractor_conflict.he_vs_they",
    vocabulary=(' their', ' his'),
    generator_role="generate_linked_possessive_number_attractor_conflict_fit_panels",
    answer_role="score_jointly_tokenized_his_versus_their",
    a1=bs.Family("attractor_frame", "attractor_frame_number_swap", lambda i, pos: f"Near the {_obj(i)} the {_agent(i)}{'s' if pos else ''} near the crates lost"),
    a2=bs.Family("notes_attractor_frame", "notes_attractor_frame_number_swap", lambda i, pos: f"In the notes the {_agent(i)}{'s' if pos else ''} near the crates signed"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} the {_agent(i)}{'s' if pos else ''} near the crates lost",
    a1_suffix=lambda i: " near the crates lost",
    a2_suffix=lambda i: " near the crates signed",
    directions=('plural_to_singular', 'singular_to_plural'),
    kinds=('plural', 'singular'),
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
