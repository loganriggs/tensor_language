"""temporal_auxiliary.will_vs_had -- auxiliary choice at the number slot, driven by TIME not number.

This is the specificity control the number family still lacks, and it is authored as a control
rather than for breadth.

Four number behaviours share one rank-1 direction at an auxiliary directly after the subject NP,
and `das_number_token_control_v1` excluded the token-axis reading by varying the answer vocabulary
(0.691 / 0.778, cosine 0.651). But every one of those four behaviours does the same STRUCTURAL
thing: choose between two auxiliary forms at that position. A direction encoding "which auxiliary
goes here" rather than "what number is the subject" would transfer among all four exactly as
observed, and nothing run so far distinguishes those two readings.

This behaviour puts a DIFFERENT variable at the SAME slot. The subject is singular in every row,
so number is constant and carries no information; a fronted temporal adverb decides the auxiliary:

    A1  "Tomorrow the leader near the maple"  -> " will"  /  "Earlier the leader near the maple" -> " had"
    A2  "The reports say that tomorrow the leader beside the forest" -> " will"
    P   swap the head noun lexically; number and the final token are untouched

Vocabulary is disjoint from BOTH number families (' will'/' had' against ' was'/' were' and
' have'/' has'). Read position is identical. So a transfer from the number direction to this
behaviour measures exactly one thing: whether that direction is about number or about auxiliary
selection.

Authored through `circuit_fast_screen_behaviour_spec`.
"""
from __future__ import annotations

import circuit_fast_screen_behaviour_spec as bs
import circuit_fast_screen_candidates as lex

R = lex._REPORTERS
O = lex._OBJECTS
_head = lambda i: R[i][0]
_alt_head = lambda i: R[i][1]
_pp = lambda i: O[(i + 7) % len(O)]
_pp_b = lambda i: O[(i + 17) % len(O)]

SPEC = bs.BehaviourSpec(
    task_id="temporal_auxiliary.will_vs_had",
    vocabulary=(" will", " had"),
    generator_role="generate_linked_temporal_auxiliary_fit_panels",
    answer_role="score_jointly_tokenized_will_versus_had",
    a1=bs.Family("bare_frame", "bare_frame_temporal_adverb_swap",
                 lambda i, pos: f"{'Tomorrow' if pos else 'Earlier'} the {_head(i)} near the {_pp(i)}"),
    a2=bs.Family("report_frame", "report_frame_temporal_adverb_swap",
                 lambda i, pos: f"The reports say that {'tomorrow' if pos else 'earlier'} the {_head(i)} beside the {_pp_b(i)}"),
    p_donor=lambda i, pos: f"{'Tomorrow' if pos else 'Earlier'} the {_alt_head(i)} near the {_pp(i)}",
    a1_suffix=lambda i: f" {_pp(i)}",
    a2_suffix=lambda i: f" {_pp_b(i)}",
    directions=("future_to_anterior", "anterior_to_future"),
    kinds=("future", "anterior"),
    p_generator_role="head_noun_lexical_rewrite",
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
