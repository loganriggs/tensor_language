# Hourly circuit and systems review — 2026-09-04 16:00 UTC

## Binding research goal

We want circuits that identify an understandable variable or operation, generalize to held-out and out-of-distribution examples,
can be extracted as an explicit computation, and can be changed or removed without damaging unrelated computations. Shared parts of
different heads or MLPs should be grouped when downstream computation treats them as the same variable; distinct computations inside
one module should be split. Rank, stored bytes, reconstruction error, and activation energy do not establish any of this. They may be
reported later as implementation prices or matched controls.

The current scientific question is narrow: does head 11.3 carry a syntax-general grammatical-number state for subject–verb
agreement, or merely a generic late signal favoring the output token ` is` or ` are`?

## Repository timestamps and actual throughput

The relevant terminal times are:

| UTC | decision |
|---:|---|
| 15:27:52 | corrected exact-state agreement screen: attention 11 and head 11.3 pass |
| 15:45:54 | literal PP ↔ relative-clause validation transfer: pass |
| 15:51:58 | block-11 fixed-component factorial: pass |
| 15:56:40 | unrelated arbitrary-code wording capability: null |

From the 15:30 checkpoint to the last result, three independent decisions landed in about 26 minutes: 8.7 serial minutes per
decision. Their total model execution was only 2.98 seconds. The expensive resource is still human/agent design and audit time, not
GPU time.

Across the broader 70-minute runtime report, the repository recorded 11 landings with a 6.1-minute mean gap and four gaps longer
than five minutes. That count includes engineering commits as well as circuit results, so it is not itself the scientific throughput
number. The more conservative circuit-only interval above is the number to optimize.

## What the last half-hour established

The literal cross-syntax test replaced either all of attention 11's output or only head 11.3's 128 pre-output-projection values.
Across 64 PP-to-relative and relative-to-PP opposite-number pairs, every patch moved the answer margin toward the donor. Mean recovery
was 60.21% for all of attention 11 and 58.93% for head 11.3. This is evidence that the complete head state generalizes across those two
syntactic constructions.

The block-11 experiment cached three vectors at the final token: the incoming state $R$, attention output $A$, and MLP output $M$.
It evaluated all eight recipient/donor sums through the same unchanged model suffix. The exact replay checks passed. On the equal
A1/A2 average, main effects were

$$
R=0.31196,\qquad A=0.55075,\qquad M=0.06383,
$$

while

$$
R+A=0.87416,\qquad R+A+M=0.93693.
$$

All pairwise interaction corrections were about one percentage point and the three-way correction was $-2.09$ percentage points.
Thus, under this fixed-component intervention, most transferred agreement information is an additive sum of the incoming block state
and the attention write. This does not assume the components would remain fixed if the block were allowed to recompute.

The first unrelated-endpoint wording probe was incapable: the model emitted ` is` on all 32 arbitrary-code examples. This is a
developmental prompt null, not evidence about the circuit. One final predeclared natural copy/select wording family is allowed; then
the route closes rather than entering open-ended prompt tuning.

## Engineering failures and corrections

One audit command passed `--dry-run` to a script that did not parse command-line arguments. Python silently ignored the unknown flag,
so the script ran 40 forwards outside the managed queue. The output is preserved under an explicit `INVALID_DIRECT_EXECUTION`
filename and is excluded from evidence. The result path's create-only rule then made the actual queue attempt fail instead of
overwriting it, which exposed the mistake.

The bounded correction took under three minutes:

1. preserve and label the invalid output;
2. add an explicit `--dry-run` argument;
3. reject all unknown arguments before model access;
4. add a regression test;
5. commit and hash the amended source; and
6. rerun only that hash through the managed queue.

This becomes a standing requirement for every new executable experiment. Environment-only dry-run switches are insufficient because
an operator can reasonably try a command-line flag.

A second organization issue also surfaced. Old experiment receipts hash the mutable Task 14 JSON, so appending a new result to that
file invalidates old source checks. Past executed sources must remain immutable. New events therefore go to the append-only
fast-screen ledger or immutable result files; a separately generated current-status view may change, but it must never be an input to
an old receipt. This avoids both duplicated experiments and the false choice between current documentation and reproducibility.

## Parallel work and reuse

Two bounded agent jobs ran concurrently:

- the unrelated-endpoint prompt design took about 15 minutes, almost entirely counterfactual design and tests;
- the head-11.3 projected-interchange adapter took about four minutes and reused the existing DAS projector algebra.

The adapter implements

$$
o'_b=o_b+UU^{\mathsf T}(o_d-o_b)
$$

inside the 128 head values before attention's output projection. It has exact zero-dimensional and full-128-dimensional endpoints,
changes only head 11.3 at the declared token position, and is invariant to rotating the columns of $U$. Twenty-one CPU tests pass.
No frame, dimension, fit, or scientific result has been chosen. This clean interface removes bespoke hook work from the next
experiment without prejudging the answer.

Claude's hourly systems lane also added a dataset lint that checks whether any prompt-derived field other than the declared variable
perfectly predicts the answer. Task 14 passes: subject number is the only such field, while distractor number is decorrelated. This is
a useful positive validation of the counterfactual dataset, not a new model result.

## Frozen next-hour route

1. Run the one-forward natural copy/select capability probe. Stop this prompt-development branch after that result.
2. If capable, freeze independent copy/select text before any activation intervention; then test full head 11.3 as the endpoint-matched
   unrelated control.
3. If incapable, record the control as unavailable and continue with the existing noun-identity and distractor-number controls, while
   keeping the generic-output-token ambiguity explicit.
4. Use the tested head-projector adapter to build a discovery-only fit and frozen validation experiment. The objective is held-out
   donor recovery with low answer-preserving-control movement, not reconstruction or low rank by itself.
5. After a causal projector survives validation, contract it through head 11.3's exact output projection and downstream attention/MLP
   weights, and test the predicted readers. Do not run weight decomposition before the causal variable is frozen.
6. Every terminal result receives an immutable result file, ledger or status-index entry, board note, commit, and next claimed action
   before another circuit starts.

The ten-minute target is currently being met over the last three decisions. The next systems bottleneck to remove is not model
runtime; it is the 10–15 minutes needed to design and audit a fresh counterfactual family. The preferred remedy is a reusable library
of already-frozen, capability-passing task banks and intervention adapters, not weaker review.
