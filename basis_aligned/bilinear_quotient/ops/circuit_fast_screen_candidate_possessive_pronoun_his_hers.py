#!/usr/bin/env python3
"""possessive_pronoun_his_hers.he_vs_she -- a THIRD independent-pronoun possessive, to test a pattern rather than assert it.

WHY THIS CELL EXISTS. In v633 ten possessive cells were measured on two controls. Eight were selective on both; the
TWO failures were possessive_pronoun_mine_yours and possessive_pronoun_person_ours_theirs, and both failed the
HELD-OUT canonical control while passing the weekly one they were fitted against. Every DETERMINER form -- my/your,
his/their, its/their, our/your, his/her -- passed both. So the failure tracked the syntactic category of the
readout, independent pronoun versus determiner, rather than the axis it encodes. Two cells is not enough to call
that a pattern, and I recorded it at the time as a test to queue rather than a claim to make.
THIS IS THE THIRD CELL. It keeps the independent-pronoun readout and changes the axis to GENDER -- his/hers -- where
the determiner counterpart possessive_gender (his/her) already passes both controls and survives the registered fit.
That pairing is the point: if the independent-pronoun form fails where its own determiner counterpart passes, the
readout category is doing the work, and the two-of-two becomes three-of-three on an axis that controls for the
content. If it passes, the earlier two failures were about person and first/second-person reference, not about
independent pronouns, and I will say so.
AUTHORED THROUGH circuit_fast_screen_behaviour_spec (Claude, 2026-09-12), mirroring possessive_pronoun_mine_yours
exactly -- same frames, same suffixes, same P rewrite -- so the only difference from an already-measured cell is the
cue and the readout pair. Capability is checked on CPU before any battery: rows where the donor does not beat the
base on the donor axis are dropped by g.prepare(valid_only=True) and counted as n_dropped in the receipt.
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
    task_id="possessive_pronoun_his_hers.he_vs_she",
    vocabulary=(' his', ' hers'),
    generator_role="generate_linked_possessive_pronoun_his_hers_fit_panels",
    answer_role="score_jointly_tokenized_his_versus_hers",
    a1=bs.Family("bare_frame", "bare_frame_pronoun_swap", lambda i, pos: f"Near the {_obj(i)} {'he' if pos else 'she'} kept the crate, of course, since it was"),
    a2=bs.Family("paid_frame", "paid_frame_pronoun_swap", lambda i, pos: f"Because {'he' if pos else 'she'} paid the {_adj(i)} {_agent(i)}, of course, the crate became"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} {'he' if pos else 'she'} kept the crate, of course, since it was",
    a1_suffix=lambda i: " kept the crate, of course, since it was",
    a2_suffix=lambda i: f" paid the {_adj(i)} {_agent(i)}, of course, the crate became",
    directions=('male_to_female', 'female_to_male'),
    kinds=('male', 'female'),
    p_generator_role="object_lexical_rewrite",
)

TASK_ID = SPEC.task_id
TASK_SPEC = SPEC.battery_spec()
build_rows, validate_rows, authority_sha256 = SPEC.api()

if __name__ == "__main__":
    rows = build_rows()
    print(TASK_ID, len(rows), authority_sha256()[:12])
