#!/usr/bin/env python3
"""possessive_person_your_their.you_vs_they -- a possessive DETERMINER mapping on an axis the family does not yet cover.

WHY. The clean pool of never-separability-tested cells is finished: sixteen cells gave one surviving candidate, and
two stems remain. New inventory now has to be AUTHORED. The possessive family is where to author it -- ten mapping
variants gave seven clearing four rows and both controls, six mutually distinct, and three surviving the REGISTERED
objective, the best rate of any family measured.
WHAT IS NEW HERE. The family currently covers number (his/their, its/their, hers/theirs), gender (his/her, his/hers)
and first-versus-second person (my/your, our/your, mine/yours, ours/theirs). It does NOT cover person contrasts
against the THIRD person: ' your' versus ' their' pits a second-person possessor against a third-person one.
That is a different cue -> token mapping from every cell in the family, which is what the corpus record says a
behaviour's identity is.
HONEST EXPECTATION, registered before any measurement: this is a CROWDED axis. possessive_person_our_your already
separates on first-versus-second person and possessive_number_his_their on number, so a third-person contrast could
share a direction with either. If it fuses, that is the answer and the cell is retired like both_either was.
Authored through circuit_fast_screen_behaviour_spec (Claude, 2026-09-12), mirroring possessive_pronoun_his_hers
exactly -- same frames, same suffixes, same P rewrite -- so the only difference from a measured cell is cue and
readout. Capability is checked on CPU before any battery.
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
    task_id="possessive_person_your_their.you_vs_they",
    vocabulary=(' your', ' their'),
    generator_role="generate_linked_possessive_person_your_their_fit_panels",
    answer_role="score_jointly_tokenized_your_versus_their",
    a1=bs.Family("bare_frame", "bare_frame_person_swap", lambda i, pos: f"Near the {_obj(i)} {'you' if pos else 'they'} kept the crate, of course, since it was"),
    a2=bs.Family("paid_frame", "paid_frame_person_swap", lambda i, pos: f"Because {'you' if pos else 'they'} paid the {_adj(i)} {_agent(i)}, of course, the crate became"),
    p_donor=lambda i, pos: f"Near the {_adj2(i)} {_obj(i)} {'you' if pos else 'they'} kept the crate, of course, since it was",
    a1_suffix=lambda i: " kept the crate, of course, since it was",
    a2_suffix=lambda i: f" paid the {_adj(i)} {_agent(i)}, of course, the crate became",
    directions=('second_to_third', 'third_to_second'),
    kinds=('second', 'third'),
    p_generator_role="object_lexical_rewrite",
)

TASK_ID = SPEC.task_id
TASK_SPEC = SPEC.battery_spec()
build_rows, validate_rows, authority_sha256 = SPEC.api()

if __name__ == "__main__":
    rows = build_rows()
    print(TASK_ID, len(rows), authority_sha256()[:12])
