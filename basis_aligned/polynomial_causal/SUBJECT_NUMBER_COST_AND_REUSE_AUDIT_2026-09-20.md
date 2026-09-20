# Subject-number graph: executable baseline and reuse scope

The historical nine-edge graph remains a useful task-specific effect decomposition,
but its edge count is not its executable cost. Its current token executor performs
two native prefixes (blocks 0–10) and ten native suffixes (blocks 11–17), including
normalization, attention and final softcap. That is 92 block evaluations per sequence.
The exact joint effect at the **same five-port intervention boundary** needs the
same two prefixes and only corners 0 and 31: 36 block evaluations. Neither count
includes embedding, readout or bookkeeping; these are not measured FLOPs or latency.
Batching corners can alter latency without eliminating their arithmetic.

This is the required direct-evaluation baseline for a proposed fast aggregate-effect
predictor. It does not supply individual edge attribution, so it is not an equivalent
baseline for an interface that also requires all nine named edge outputs.

## Receipt audit

The export verifier passes its six package hashes and three evidence receipts.
The current token executor also matches the SHA-256 in its original extraction
receipt. This verifies provenance, not an independent rerun of native behavior.
The original receipts report maximum prediction error 6.79% on the original panel
and 5.35% on disjoint fresh nouns/templates. The latter panel is now opened.
Fresh removal is positive on all 32 prompts; target damage is 14.63 times the median
equal-L2 control, with can/will collateral 3.51% of target damage. These remain
historical results under their specific controls, not general syntactic selectivity.

The token executor derives activations internally but still requires the checkpoint,
the frozen number-decoder axis and threshold, subject positions and answer IDs.
It edits the subject position and reads the answer there; the tested templates put
the subject at the prediction endpoint. Broader syntax would require distinct edit
and readout positions. Zero external activation inputs does not mean zero dependency
on native computation.

The manifest's composition/reuse claim needs a narrower reading than our goal.
Its cited behavioral composition receipt is a **three-port** removal/rescue test,
with 18–19% effect error from summing individual damages. The **five-port nine-edge**
predictor additionally includes selected pair interactions and transfers across its
subject-number templates. These are useful composition and same-task transfer
results. The inspected receipts do not establish that a discovered feature is
reused by independently extracted circuits for different tasks. Algebraically
zeroing a graph edge is also distinct from installing its selective native removal.
No original receipt or hashed export was changed by this audit.

## Executed algebraic consequence

Let f_S be the suffix margin after applying the port subset S. The prediction is
minus the sum of five singleton and four pair Mobius dividends. Collecting terms:

```
prediction = f0 - f2 + f4 + 2*f8 + f16 - f9 - f12 - f20 - f24
```

The f1 coefficient cancels. An aggregate-only evaluator therefore needs nine
suffix queries, costing 85 block evaluations under the same execution strategy.
The original ten-query interface is still needed to expose all named edges.
An independent check against the frozen node used all 32 coordinate-basis corner
assignments plus 1,000 random assignments; maximum error was 3.55e-15. Dropping
any of the nine retained queries changes at least one basis response. This proves
minimality only for this linear functional of an arbitrary corner oracle; native
weight identities could permit much greater simplification. The reduced query plan
has not been installed or natively timed.

See [executable CPU control](check_subject_number_corner_cost.py) and
[receipt](SUBJECT_NUMBER_CORNER_COST_AUDIT_2026-09-20.json).

## Consequence for the next folding target

Keep this lineage as a semantic target, not a simplicity success. An aggregate
predictor must be compared to direct exact two-corner evaluation, as well as the
existing attributed graph. A proposed shared feature must specify its consumers
and demonstrate reuse beyond repeated calls to the same checkpoint suffix.
The next meaningful reduction must replace native computation inside the prefix
or suffix, rather than just count fewer named intervention edges. Retain the
exact five-port target and its scoped preservation controls while doing so.

The three-hour mathematical review is not due yet: its last receipt is 04:22 UTC,
and the active cron checks eligibility every five minutes. This CPU audit is not
a substitute for the scheduled primary-literature search.
