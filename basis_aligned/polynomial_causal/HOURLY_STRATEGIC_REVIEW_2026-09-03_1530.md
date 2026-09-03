# Hourly strategic review — 2026-09-03 15:30 UTC

## Goal and acceptance criteria

The goal is a simpler executable description of bilin18 whose parts correspond to reusable computations. A circuit
record is useful only when it states what information is read, what operation is performed, what is written, and
which later computations use it. Evidence should include genuinely held-out or shifted prediction, sufficiency,
selective removal or interchange, interactions with other circuit variables, and a stable physical realization.

Rank, reconstruction error, parameter count, and quantization are not circuit criteria. They become useful only
after a causal variable has been independently specified and validated.

## What changed this hour

- The old pending-opener rows had no cross-split token-sequence overlap, but they reused prompt pairs within splits.
  Rescoring unique prompts did not change the R538 site result or the R540 selectivity failure. The unopened old
  FINAL/OOD splits were nevertheless retired.
- A replacement dataset now has 240 semantic groups, all five counterfactual families per group, all twelve ordered
  pairs among four closer types, 1,200 unique prompt pairs, and 2,400 unique token sequences. Each ordered pair has
  eight FIT groups and four groups in each held-out split.
- R540 is a strong negative result: fitted directions recover both answer-changing families, but also move the answer
  on pending-state-preserving edits. The rank-one directions have cosine magnitude 0.49--0.57 with a closer-token
  unembedding contrast. The fit learned an answer-margin steering direction, not an isolated pending-opener state.
- The four-closer capability and complete-state gate is now running under the managed GPU runner. It cannot fit a
  projector or inspect FINAL/OOD.
- The generated circuit index now exposes held evidence types, active failed/null/invalid tests, the latest active
  event, and the exact next missing test. This is the minimum anti-duplication view needed before scaling the work.

## Decision and alternatives

The current highest-information step is R544: determine whether both answer-changing constructions and all three
answer-preserving constructions have usable complete-state effects at one frozen site over all four closer values.
If that gate passes, the next fit must not be trained and judged only on the same closer-logit margin. It should use
multiple independent consequences, report overlap with the endpoint readout and downstream-gradient spans, and
require controls to remain unchanged.

The next dataset after pending opener should be induction selector × payload. It directly tests whether the copying
decision and copied content can be interchanged separately and jointly. The equality-score shared subroutine is the
next cross-module record: its existing natural/code positives and downstream-reader evidence are already canonical,
while its failures are preserved.

For hundreds of circuits, every new run must attach to a stable circuit ID and a content-addressed dataset version.
No experiment should start from a prose name alone. The record must say which counterfactual families, sites,
endpoints, controls, splits, and interaction tests have already been run.
