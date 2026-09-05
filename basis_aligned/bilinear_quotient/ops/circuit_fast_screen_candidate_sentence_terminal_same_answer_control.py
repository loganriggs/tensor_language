#!/usr/bin/env python3
"""sentence_terminal with a SAME-ANSWER control, testing whether the C clause can fail at all.

The context-control screen
(`circuits/fast_screens/sentence_terminal_semantic_choice_context_control_v1_result.json`)
recovered the target perfectly at resid:18 (A1 1.000, A2 1.000, P 0.035) and was refused only
because C also read 1.000. That control is answer-changing at the patched position, so any
site carrying the prediction state carries it in full: it measures "does this site carry the
prediction", not "is this site specific".

Measured across every screen on disk, the C clause is bimodal. Where the control is
answer-changing, C reaches 1.0 and the verdict is a null. Where it is same-answer, C has never
exceeded 0.122 at ANY of 55-64 sites -- never even 35% of its 0.35 bar -- and the verdict is
`selective_causal_site`. Every selectivity verdict we hold rests on the second configuration.

So this candidate holds A1/A2/P byte-identical to the v1 authority and installs a same-answer
control whose base and donor differ substantially in place, subject and reporter. Two readings
are distinguished by the result:

  * C stays near zero at every site, including resid:18, which demonstrably carries the whole
    prediction (it recovers the target at 1.000). Then the clause is close to unfalsifiable and
    our selectivity verdicts are weaker than recorded.
  * C rises at non-specific sites and stays low at specific ones. Then the clause discriminates
    and the concern is answered.

`normalized_same_answer_effect` is `abs(intervened - base) / scale`, a margin-disturbance
measure, so a nonzero C is possible here; that is exactly what makes the test meaningful.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import circuit_fast_screen_candidate_sentence_terminal_context_control as shared

TASK_ID = shared.SAME_ANSWER_TASK_ID
TASK_SPEC = shared.SAME_ANSWER_SPEC
SCHEMA = shared.SCHEMA
SPLIT = shared.SPLIT
DEFAULT_GROUPS = shared.DEFAULT_GROUPS
DEFAULT_SEED = shared.DEFAULT_SEED
CandidateBankError = shared.CandidateBankError


def build_rows(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
               seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    return shared.same_answer_rows(groups, seed)


def validate_rows(rows: Sequence[Mapping[str, object]], *, task_id: str = TASK_ID,
                  groups: int = DEFAULT_GROUPS, seed: int = DEFAULT_SEED) -> str:
    return shared._validate(rows, groups, seed, kind="same_answer",
                            task_id=TASK_ID, spec=TASK_SPEC)


def authority_sha256(task_id: str = TASK_ID, groups: int = DEFAULT_GROUPS,
                     seed: int = DEFAULT_SEED) -> str:
    return shared.same_answer_authority()
