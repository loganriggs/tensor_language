# Attention0 QK common-carrier preregistration

Date: 2026-09-01 19:20 UTC

## Goal and claim boundary

Rung 418 found a nearly uniform cross-head projector overlap (about 0.19) among the 18 exact attention0 QK
head-branch token-function subspaces, but no discrete shared edge. This rung tests the mathematically different
hypothesis that every branch contains one global continuous token-function carrier plus a private remainder.

This is a gauge-invariant identification test. It is not a vocabulary of atoms, a semantic label, a compressed
program, or an adoption result. Private Q/K coordinates are never compared directly.

## Frozen population and exact objects

- Real token IDs: `0,...,50,256`.
- `FIT`: token ID modulo 5 is not 4.
- `SELECT`: token ID modulo 5 is 4.
- `FINAL`: unopened.
- Entries: all 18 `(head, multiplicative-QK-branch)` pairs in attention0.
- Sides: query and key are analyzed separately; their carrier agreement is reported explicitly.

Recompute the exact folded query and key tables from the pinned float32 checkpoint. The established float64 fold
must reproduce a live attention0 score branch to maximum absolute error `<=1e-10`; factor row RMS error must be
`<=1e-6` and all FIT whitening errors `<=2e-4`.

## Gauge-invariant carrier estimator

For entry `e` and side `s`, center its FIT token table and whiten its 128 columns to obtain
`B[e,s]` with `B^T B=I`. Its token-function projector is `P[e,s]=B B^T`, which is unchanged by an invertible
private coordinate change.

For each side separately, define the average projector

`Pbar[s] = (1/18) sum_e P[e,s]`.

The proposed carrier `C[s]` is the top 24-dimensional eigenspace of `Pbar[s]`. Rank 24 is frozen from rung 418's
pre-result estimate: overlap 0.19 times branch rank 128 is about 24. It is not selected from this run. Compute it
with deterministic randomized subspace iteration (`q=48`, eight power iterations, seed 420), and repeat with seed
421 as an approximation-stability check.

For each branch, map `C[s]` into that branch's feature coordinates using `A=C^T B`. The row space of `A` defines a
rank-24 right-coordinate projector `R[e,s]`. This is only a computational representation: the identified object is
the token-function subspace, not those private coordinates.

## Held-out transport and controls

Apply the FIT whitening and frozen `R[e,s]` to SELECT tokens.

- Carrier component: `B_select R`.
- Private remainder: `B_select (I-R)`.

Re-orthonormalize each SELECT component before measuring pairwise projector overlap. Evaluate all 144 cross-head
pairs separately for query and key. Compare with:

1. independent seed-420 token-row permutations before fitting the average projector;
2. independent seed-fixed Haar rank-24 right-coordinate removals in each branch.

Report original, carrier-only, and residual mean/SD/range, plus the residual 99th percentile. A common carrier
predicts mutually aligned carrier components and residual overlap near rung 418's random floor, whereas arbitrary
rank-24 removal should leave most of the original overlap.

## MLP0 connection

Recompute the exact length-one MLP0 token path on the same FIT/SELECT split. Define the bias-free action decomposition
as in rung 397: `F=M+L+Q`, where `M` is the FIT action mean and `L` is the complete degree-one least-squares map from
the exact normalized MLP0 input `z`; `Q` is the remaining token-specific nonlinear part.

On FIT, compare each QK carrier with the leading 64 token-function modes of `L`. Also report comparison with the
leading 64 modes of `z`, `F`, and `Q`; these distinguish a generic input-embedding carrier from an MLP0-output-weighted
linear carrier. The registered null independently permutes the token rows of `L` before extracting its modes.

Because `L=zA`, its complete column space can equal that of `z` when `A` is full rank. Therefore only leading
energy-ranked modes can support a specific MLP0 connection; mere containment in the full degree-one column space is
not evidence.

## Frozen predictions

### A. Exact and reproducible instrument

The fold, row-RMS, split, whitening, rank, finite-value, and shape checks above hold; the recomputed FIT original
overlap means match rung 418 within `0.002` on both sides; and the two randomized carrier estimates have projector
overlap `>=0.95` on both sides.

### B. A global carrier is present on FIT

For both query and key, the mean fraction of each carrier direction captured by a branch is `>=0.65`. The query and
key carriers themselves have rank-24 projector overlap `>=0.25`, indicating at least partial cross-side reuse rather
than two unrelated common spaces.

### C. The carrier transports to unseen tokens and explains the diffuse sharing

On SELECT, for both query and key:

- carrier-only mean pairwise projector overlap is `>=0.70`;
- private-remainder mean overlap is `<=0.03` and at most 25% of the original mean;
- Haar-removal residual mean remains `>=0.12`;
- carrier-only mean exceeds the row-permuted-carrier control by `>=0.40`.

### D. The carrier meets MLP0's degree-one token function

For both Q and K carriers, overlap with the leading 64 modes of `L` is `>=0.25`; at least one is `>=0.40`; and each
exceeds its token-row-permuted `L` control by `>=0.20`. Alignment with `z`, `F`, and `Q` is diagnostic and cannot be
used to change the decision after the run.

## Strong null and next decision

The strong null fires if A fails, or if either side has SELECT carrier-only mean overlap `<0.30`, private-remainder
mean overlap `>0.10`, or `L` alignment margin over the permuted control `<0.10`.

If A-D hold without the null, the result identifies a shared continuous token-function carrier connecting attention0
QK and MLP0's degree-one token computation. The next rung must test native QK score and downstream causal response
after carrier-specific interventions; no compression is licensed. If C holds but D fails, retain an attention-only
carrier. If C fails, reject the average-projector carrier explanation and proceed to a continuous coupled QK-times-OV
block-term model rather than tuning carrier rank.
