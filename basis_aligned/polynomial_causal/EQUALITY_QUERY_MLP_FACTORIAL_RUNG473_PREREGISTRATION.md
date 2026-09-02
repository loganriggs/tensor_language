# Rung 473 preregistration — exact MLP composition at the equality query position

Registered before running any two-MLP query-position intervention. Rung 472's singleton and all-three outcomes are
already open and hash-bound inputs; they are not rescored or changed.

## Question

Rung 472 identified the query position as a causal part of the equality correction across held-out code and natural
text. All three MLPs contribute, but their joint query effect is not the sum of the three individual effects, and the
total interaction changes between code and natural text. Is that non-additivity carried by one consistent MLP pair,
by several pair interactions, or by an irreducible three-MLP interaction?

This is a within-module splitting and cross-module grouping test. It does not reduce rank or claim compression.

## Exact computation

Use exactly rung 472's selected targets, three validation windows, two matcher sources, source states, absent-equality
MLP products, and query-position masks. For every target, run the three missing two-MLP removals:

- MLP8 + MLP9;
- MLP8 + MLP12;
- MLP9 + MLP12.

Let `y(S)` be the target-token CE change when the query-position products of the MLP subset `S` are replaced by their
same-document equality-absent values, with all later computation rerun. Rung 472 supplies `y(8)`, `y(9)`, `y(12)`,
and `y(8,9,12)`. The new run supplies the three pair effects. Define the exact interaction terms

`I(i,j) = y(i,j) - y(i) - y(j)`

and

`I(8,9,12) = y(8,9,12) - y(8) - y(9) - y(12) - I(8,9) - I(8,12) - I(9,12)`.

These terms describe nonlinear downstream composition under the registered removal operation. They are not assumed
to be independently stored features.

## Frozen predictions

### A — valid exact intervention

- all source/preregistration/parent hashes match;
- native replay relative squared error is at most `1e-12`;
- product reconstruction relative squared error is at most `1e-10`;
- an empty query mask changes no logit;
- each requested MLP query patch fires exactly once;
- recombining the main, pair, and triple terms reproduces the rung-472 all-three query effect within `1e-12` in
  float64 analysis;
- observed forwards and patch calls equal the formulas printed before model load;
- no SEALED attention-0 confirmation outcome is opened.

### B — one stable MLP pair carries the interaction

The same pair has the largest pair-interaction norm in all six window × source conditions, supplies at least 40% of
the total interaction norm in every condition, and its N/H four-context vectors have cosine at least `.70` in every
window.

### C — the interaction is register/source dependent

The already observed total-interaction N/H cosine remains at least `.80` on held-out code and is at most `.20` on at
least one natural window, and the factorial localizes that change: either the largest pair changes between code and a
natural condition, or some natural pair's N/H four-context cosine is at most `.20` while its norm is at least `.003`
nat under one source.

### D — pairwise composition is sufficient

After omitting only the triple interaction, the main-plus-pair prediction of the all-three query effect has per-token
Pearson at least `.90` and normalized L2 error at most `.50` in every window and source. This would license the three
pair interactions as the complete composition rule for this intervention.

### E — the decomposition is stable across document halves

For the largest pair in each condition, its signed mean has the same sign in both fixed document halves, and the
pair-interaction norm in each half is at least 20% of its pooled norm. This rejects a grouping caused by a few selected
documents.

## Strong null and routing

The strong null fires if A fails, no pair interaction reaches `.003` nat anywhere, or the triple interaction is at
least as large as the sum of all pair-interaction norms in at least four of six conditions. A+B+D+E would identify one
stable cross-MLP interaction. A+C+E would instead identify separate code/natural composition rules. A plus failure of
B/C/D routes to a downstream-state-conditioned query decomposition rather than another MLP-subset or rank sweep.

## Price

Diagnostic only: zero deployed parameters saved or added. Report model forwards, query patch calls, runtime, and peak
GPU memory. GPU execution is legal only through the managed runner.
