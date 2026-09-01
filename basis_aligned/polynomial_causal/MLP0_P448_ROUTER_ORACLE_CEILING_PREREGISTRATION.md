# MLP0 two-state rank-448 router oracle ceiling — preregistration

Date: 2026-09-01 16:28 UTC

## Decision and scope

Rungs 404--406 found document-wave variation among fixed rank-448 input subspaces, but no new global metric beat the
covariance program. Before fitting a router, this rung measures the maximum predictive headroom available from those
already fixed experts and compares it with cheaper single-program controls.

This is an oracle ceiling, not a deployable router, compression claim, or adoption. The oracle may inspect future
target losses. A failed optimistic ceiling closes state search among these experts; a pass only licenses a held-out
prefix-observable router experiment with literal state and compute price.

## Fixed programs and populations

Rebuild from the pinned float32 source weights and the unchanged24 historical program-fitting documents:

- rank448 covariance RRR from rung403;
- rank448 T-active, I-active, and equal-trace TI-active bases from rung405, using the same two seed405 probes;
- rank448 downstream-Fisher basis from rung406, using the same full directional CE gradients;
- covariance RRR rank640 and rank768 controls from the same covariance and source.

No basis, rank, program, pair, threshold, or seed is chosen from this rung's losses. Evaluation is rung404's exact
384-source-document population, one chunk per source, four contiguous96-document waves, positions `[64:256)`, for
73,728 scored next-token positions under the unchanged BF16 model. FINAL remains unopened.

## Literal prices

One shared-input program at rank `r` stores

`1152*r + 2*4608*r + 4608*1152 + 1152` values.

Therefore:

| object | stored MLP0 values |
|---|---:|
| one p448 | 9,954,432 |
| one p640 | 11,945,088 |
| one p768 | 13,272,192 |
| two p448 experts sharing only Down and bias | 14,599,296 |
| native MLP0 | 15,926,400 |

The two-expert price is an optimistic lower bound: it excludes every router parameter, state bit, routing operation,
and cache cost. Both p640 and p768 are cheaper than two p448 experts and are therefore mandatory dominance controls.

## Oracle definitions

Let `loss[d,p,e]` be the physical next-token CE at document `d`, scored position `p`, when the complete sequence is
run with fixed expert `e` at MLP0.

For every one of the `choose(5,2)=10` p448 pairs `(e,f)`:

- document oracle loss is `mean_d min(mean_p loss[d,p,e], mean_p loss[d,p,f])`;
- position oracle loss is `mean_(d,p) min(loss[d,p,e], loss[d,p,f])`.

The document oracle is physically attainable only by a clairvoyant selector that knows the document's future mean
loss before execution. The position oracle is even more generous and need not correspond to one coherent routed
forward pass. Both are upper-bound diagnostics. Also report the analogous oracle over all five p448 experts.

Damage is oracle/program CE minus native CE; lower is better. Record each pair's document winner fractions, per-wave
oracle damage, and gain over covariance p448.

## Frozen predictions

### A — exact fixed programs, population, and measurements

- checkpoint, fit/evaluation rows, hashes, disjointness, waves, BF16 model, scoring positions, and FINAL status are
  exact;
- covariance/T/I/TI/Fisher p448 basis hashes and per-wave physical damages reproduce their valid parent receipts
  within `1e-6`; covariance retained energy reproduces `.9011108875274658`;
- p640/p768 ranks, shapes, prices, p448/two-expert/native prices, all calls, and all per-position losses are finite
  and exact.

### B — two fixed experts have material oracle headroom over covariance p448

For at least one pair:

- document-oracle damage is at most `0.70` times covariance-p448 damage;
- position-oracle damage is at most `0.50` times covariance-p448 damage.

The same pair must satisfy both clauses.

### C — oracle routing can survive its unfavorable storage price

- the best pair's document-oracle damage is at least `0.0005 nat` lower than the cheaper p768 program;
- that pair's position-oracle damage is lower than p768 damage.

### D — the best pair represents a real two-state split rather than one dominant expert

- each expert wins at least 20% of documents globally;
- the pair's document-oracle gain over covariance p448 is positive in all four96-document waves.

## Strong null

The strong null fires if A fails; the all-five position oracle fails to beat p768 by `0.0002 nat`; the best pair
document oracle fails to beat p768 at all; or the best pair improves covariance p448 by less than `0.0002 nat`.

## Decision

- A+B+C+D with no null licenses one two-state held-out router screen. Its state must be computable from the prefix,
  training and evaluation documents must be disjoint, and its literal price must include both experts and routing.
- Position-only headroom without document headroom permits at most a token-state feasibility study; it does not
  license a document router.
- Strong null closes routing among these five experts. Advance direct nonlinear CE fitting or an output-side
  representation change; do not search state definitions or add experts after seeing the oracle.
