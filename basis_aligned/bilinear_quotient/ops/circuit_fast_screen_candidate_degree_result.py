#!/usr/bin/env python3
"""degree_result.too_vs_so -- a degree word licensing its result clause.

`degree_frame` varies the COMPARISON head and its marker (`more` -> ` than`, `as` -> ` as`). This
varies a degree word that licenses a RESULT clause instead: `too` licenses an infinitival result
with ` to`, `so` a finite one with ` that`. Both sit immediately before the same adjective, so
cue distance cannot be blamed for a failure here.

    A1 copular frame
      "The crate was far too heavy"    -> " to"
      "The crate was really so heavy"  -> " that"
    A2 seem frame
      "In the notes the crate seemed far too heavy"   -> " to"
      "In the notes the crate seemed really so heavy" -> " that"

First behaviour authored through `circuit_fast_screen_behaviour_spec`, which reproduces the
`correlative_pair` authority digest exactly and so is known faithful before this uses it.
"""
from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidates as lex

_ADJ = lex._ADJECTIVES
_NOUN = lex._OBJECTS


def _adj(i: int) -> str:
    return _ADJ[i]


def _copular(i: int, infinitival: bool) -> str:
    return f"The {_NOUN[i]} was {'far too' if infinitival else 'really so'} {_adj(i)}"


def _seem(i: int, infinitival: bool) -> str:
    return f"In the notes the {_NOUN[i]} seemed {'far too' if infinitival else 'really so'} {_adj(i)}"


def _p_donor(i: int, infinitival: bool) -> str:
    """Answer-preserving: a different noun, same degree word, same final adjective."""
    other = _NOUN[(i + 7) % len(_NOUN)]
    return f"The {other} was {'far too' if infinitival else 'really so'} {_adj(i)}"


SPEC = bs.BehaviourSpec(
    task_id="degree_result.too_vs_so",
    vocabulary=(" to", " that"),
    generator_role="generate_linked_degree_result_fit_panels",
    answer_role="score_jointly_tokenized_to_versus_that",
    a1=bs.Family("copular_frame", "copular_frame_degree_result_swap", _copular),
    a2=bs.Family("seem_frame", "seem_frame_degree_result_swap", _seem),
    p_donor=_p_donor,
    a1_suffix=lambda i: f" {_adj(i)}",
    a2_suffix=lambda i: f" {_adj(i)}",
    directions=("infinitival_to_finite", "finite_to_infinitival"),
    kinds=("infinitival", "finite"),
    p_generator_role="noun_lexical_rewrite",
)

TASK_ID = SPEC.task_id
TASK_SPEC = SPEC.battery_spec()
build_rows, validate_rows, authority_sha256 = SPEC.api()

if __name__ == "__main__":
    rows = build_rows()
    print("authority:", authority_sha256())
    for family in ("A1", "A2", "P", "C"):
        r = next(x for x in rows if x["family"] == family)
        print(f"  {family} base : {r['base_text']!r} -> {r['base_answer']!r}")
        print(f"     donor: {r['donor_text']!r} -> {r['donor_answer']!r}")
