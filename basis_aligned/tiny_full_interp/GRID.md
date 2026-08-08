# Training grid — claim a cell by editing this file and pushing BEFORE starting

Status: `unclaimed` / `local:running` / `scale:running` / `done` (+ CE)
Head dim fixed at 16, so heads = width/16. Vocab 4096 primary.
Fresh single-epoch protocol; 3 seeds per cell.

## Primary grid (depths 1-2)

| depth | width | heads | owner | status |
|---|---|---|---|---|
| 1 | 32 | 2 | local | unclaimed |
| 1 | 64 | 4 | local | unclaimed |
| 1 | 128 | 8 | local | unclaimed |
| 1 | 256 | 16 | scale | unclaimed |
| 2 | 32 | 2 | local | unclaimed |
| 2 | 64 | 4 | local | unclaimed |
| 2 | 128 | 8 | local | unclaimed |
| 2 | 256 | 16 | scale | unclaimed |

## Architecture-variant slice (Logan 2026-08-08) — the comparison that motivates the program

Same tiny setting, different explainable architectures, each interpreted in
full. At depth 1 there are 2 modules and at depth 2 there are 4, so the whole
wiring diagram is a 2x2 or 4x4 table that can be written out by hand — the
variants' claims (who reads whom, what is named, what is discrete) become
directly checkable rather than statistical.

Variants: A `vanilla` | B `slots` (partition + per-slot norm + in-loss lasso)
| C `bandwidth` (B + true-small decoders, savings into wider slots)
| D `predicate` (C + named attention terms; parent program's leader)
| E `codebook` (C + variable-k discrete slot content)
| F `shrink` (B + shrinking embedding channel with floor)

**Phase V1 — the comparison slice** (depth 2, width 128, seed 0; 6 models):

| variant | owner | status |
|---|---|---|
| A vanilla | local | unclaimed |
| B slots | local | unclaimed |
| C bandwidth | local | unclaimed |
| D predicate | local | unclaimed |
| E codebook | local | unclaimed |
| F shrink | local | unclaimed |

**Phase V2 — width sweep of the informative variants** (depth 2, seed 0;
which variants carry forward is decided by V1, not pre-committed):

| variant | widths 32/64 | width 256 |
|---|---|---|
| (decided by V1) | local | scale |

**Phase V3 — seeds** (3 seeds on whatever V1/V2 leaves standing; the parent
program's rule is that no structure claim survives a single seed).

The question V1 answers is NOT which variant wins on loss — at this size that
is nearly meaningless. It is: **do the architectures that claim to be more
interpretable actually compute the same thing by different means, or do they
compute something different?** Same-solution-different-encoding and
different-solution are distinguishable here because both models are fully
folded: compare materialized tables, behavioural inventories, and the rung-5
reconstruction remainders directly.

## Depth ladder (after the primary grid has first results)

| depth | width | owner | status |
|---|---|---|---|
| 3 | 64 | scale | unclaimed |
| 3 | 128 | scale | unclaimed |
| 3 | 256 | scale | unclaimed |
| 4 | 64 | scale | unclaimed |
| 4 | 128 | scale | unclaimed |
| 4 | 256 | scale | unclaimed |

## Baselines (matched-optimizer, per width — REQUIRED before quoting any cost)

| kind | widths | owner | status |
|---|---|---|---|
| bigram table (closed form, no training) | n/a | local | unclaimed |
| unigram floor | n/a | local | unclaimed |
| same-size softmax+GELU transformer | 32-256 | local | unclaimed |

The softmax/GELU baseline answers the first question a reviewer will ask:
what does the no-softmax bilinear architecture cost in prediction quality,
at each size? If the gap grows with size, the fold's tractability is being
bought with capability and that must be reported alongside every result.

## Vocab check

| vocab | owner | status |
|---|---|---|
| 8192 at width 128 depth 2 | local | unclaimed |

Answers whether conclusions are vocab-size artifacts.
