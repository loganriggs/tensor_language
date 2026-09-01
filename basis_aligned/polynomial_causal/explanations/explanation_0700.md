# Plain-English update — 2026-09-01 07:00 UTC

**The one-sentence headline:** the best executable program is now a clean staircase obtained by measuring Q/K
rank in the geometry of actual activations, while two plausible alternatives have been sharply separated: value
maps are compressible but expensive, and generic block/tree/DAG structure is not recoverable from MLP0's folded
bilinear weights, even in the full 1,152-dimensional space.

## Our goal

The goal is to compile bilin18 into a substantially smaller tensor program that is simultaneously:

1. **predictive** — low loss on census, fresh windows, and genuinely shifted text;
2. **composable** — separately useful replacements remain useful together;
3. **manipulable** — a named intervention produces the same signed effects as in the native model; and
4. **literally simple** — every tensor, factor, exception, router state, and fallback is counted in the scalar and
   byte bill.

This is stronger than ordinary compression. We are looking for the model's smaller executable structure, with
receipts that distinguish a real program from an approximate output fit.

## The mathematical result that is driving progress

For a linear map `W` receiving vectors `x`, ordinary SVD minimizes raw Frobenius error. The functionally relevant
rank-`r` problem is instead

`min_rank(W_r)<=r E ||(W-W_r)x||^2 = min_rank(W_r)<=r ||(W-W_r) C^(1/2)||_F^2`,

where `C = E[x x^T]` is fitted on a disjoint activation population. We factor `W C^(1/2)`, retain its leading
singular directions, and map the right factor back through `C^(-1/2)`. This gives a reduced-rank-regression
factorization of all 440 replaced query/key maps in layers 2–17.

The literal price decreases by exactly

`440 maps * (128 + 1152) factor coordinates * 8 ranks = 4,505,600 scalars`

at every eight-rank step. The independently fitted and causally gated staircase is now:

| Q/K rank | Standalone scalars | Census CE added | Certificates | Shifted mean / p95 / max | Signed effect cosine |
|---:|---:|---:|---:|---:|---:|
| 96 | 535,089,462 | +.001415 | 61/62 | -.001674 / .007033 / .015147 | .997617 |
| 88 | 530,583,862 | +.002196 | 58/62 | -.003865 / .006220 / .007768 | .996520 |
| 80 | 526,078,262 | +.003336 | 54/62 | -.000879 / .013861 / .026134 | .994961 |
| 72 | 521,572,662 | +.005238 | 54/62 | +.000498 / .020242 / .035112 | .992435 |
| 64 | 517,067,062 | +.008193 | 50/62 | +.004753 / .021539 / .033750 | .988466 |

Negative shifted means are sampling-scale zero, not evidence that compression improves the native model. The
incremental census costs are `+.000781, +.001140, +.001902, +.002955`: the curve is convex, but there is still no
discontinuity. Certificates are not a fixed function of aggregate loss—the 80→72 step costs census loss but no
certificates—so future bars must track both. Rank64's signed gate also passed: normalized effect error `.178663`,
collateral Spearman `.994592`, and own-effect median ratio `1.081668`. Signed fidelity degrades monotonically but
smoothly down the ladder; the current causal bars are catastrophic-failure guards rather than the binding frontier.

## What composition taught us

Three physical Q/K × MLP0 combinations landed at `1.0199x–1.0241x` their additive component prediction. Earlier
compositions within the MLP family repeatedly cost about `1.3x`. The evidence therefore supports two regimes:

- cross-family residuals, at least Q/K versus MLP0, are almost additive;
- within-family MLP residuals align and amplify one another.

This matters operationally: choose the cheapest good point inside each family, then consider cross-family
composition. However, the pure Q/K ladder advanced so efficiently that rank 80 already strictly dominated every
tested Q/K+MLP0 combination. We retired those combinations rather than spending causal gates on interior points.

## The MLP0 full-embedding idea: a useful negative

The proposed route was exactly the right structural question. At position zero, all 50,304 token inputs and the
complete MLP0 weights are available, so one can fold the entire embedding population into the bilinear layer and
ask whether a block, hierarchy, or DAG is present in the function tensor itself.

Raw hidden-unit factors cannot answer this because bilinear CP representations have permutation, scaling, swap,
and more general gauge non-identifiability. We therefore formed gauge-invariant quadratic contractions. For output
direction `q_c`, a contraction has the form

`A_c = sum_u (q_c^T d_u) sym(l_u r_u^T)`,

then it is weighted by the exact covariance of all 50,304 folded MLP0 inputs. If the family `{A_c}` has common
blocks, its joint commutant or an equivalent reference-basis graph should disconnect or nearly disconnect.

Two planted controls recovered their known `(3,4,5)` blocks exactly and remained invariant under common gauge
changes. But the real model failed at both tested scales:

- embedding-PCA32 contraction algebra: real/null structural ratio `1.028`, split overlap `.138`;
- full 1,152D exact fold: real/null graph-connectivity ratio `1.294`, split overlap `.172`.

In the full-space test the real graph was more connected than its spectrum-matched null, so this is not a small
sample or low-rank failure. The generic claim—“MLP0 contains a coordinate system revealing neuron partitions,
tree blocks, or a DAG directly from its folded quadratic weights”—is rejected by a probe that detects such
structure when planted.

The result does **not** rule out task-conditioned structure. A subspace selected by a named output behavior,
intervention, or router variable could reduce even when the full algebra is irreducible. Any reopening must name
that external variable beforehand; an unconstrained search for a visually pleasing partition is closed.

## Top-k is not a finite tensor program; a small MoE router can be

Generic per-token top-k chooses one of `binomial(4608,k)` supports. It is an input-dependent compute policy with a
combinatorial state space, not a small tensor network merely because only `k` units execute. A finite MoE router is
different: if it chooses among a fixed small collection of distinct expert subsets, the router state and each
subset can be stored and priced explicitly.

The first MLP0 router screen found learnable state labels (four-state accuracy `.652` versus `.25` chance), but
even oracle fixed subsets had negative output `R^2`; unconstrained top-k reached `.754`. Thus a small router exists
as a classifier signal but the tested fixed experts do not span the layer's computation. A future router needs a
different expert parameterization, not more optimistic accounting of top-k.

## The third attention family

We applied the same context metric to 144 headwise value maps on top of QK80. The exact program was
522,539,318 scalars. It preserved 46 certificates and transported to the terminal WikiText slice, but census damage
was `+.01438`, a `+.01104` surcharge for only 3.54M additional scalar saving. That is roughly an order of magnitude
worse per saved scalar than the current Q/K steps. Value transport is genuinely compressible—the null did not
fire—but it is not competitive real estate and does not advance.

## Data-hygiene correction

The WikiText-2 test stream contains 286,177 GPT-2 tokens and is now exhausted by the sequence of OOD gates. Earlier
20k skip increments also caused adjacent 120-row evaluations to overlap by about 10,840 tokens. Each gate was fresh
to its own artifact, but consecutive rows were not mutually independent. We now require skip increments at least
the evaluation length or an explicit corpus switch.

WikiText-103's **test** configuration turned out to tokenize to the identical 286,177-token text, so changing its
dataset fingerprint would not create a new population. Rank 56 is instead frozen on WikiText-103 raw **train** rows
`[100000,110000)`: 675,457 tokens, dataset fingerprint `7dabb830ac9ebb0d`, token SHA256 prefix
`4e1ca0fd7f5c6f00`. None of those tokens fits the program.

## Independent paths from here

1. **Finish the Q/K ledge map.** Gate rank64 causally, then test rank56 on the new hashed corpus. The convex
   increments predict about `+.0128` census at rank56; rank48 is likely near or beyond the practical cliff.
2. **Tail-robust functional metrics.** Replace mean covariance with leverage-, loss-gradient-, or minimax-weighted
   covariance. This directly targets the rare-row failures that stop aggressive MLP ranks.
3. **Downstream-aware joint objectives.** Optimize factor directions against suffix loss or the joint residual
   action, with held-out fitting. Sequential refitting already showed that stale inputs do not explain MLP's 1.3x
   tax; the objective must change.
4. **Task-conditioned MLP0 reducing subspaces.** Condition the contraction family on a preregistered behavior or
   intervention and demand planted recovery, split stability, OOD prediction, and literal price. Generic blocks
   are closed; named conditional blocks remain a narrower hypothesis.
5. **Finite-router experts with learned low-dimensional maps.** Keep the number of states small and priced, but
   allow each expert to be a low-rank bilinear map rather than a fixed native-unit subset. Compare against an
   equal-price shared low-rank control.
6. **Error-contract and lower-bound route.** Turn the empirical convex Q/K ladder and certificate margins into a
   prospective stopping rule, then test that rule on rank56/48. This can save GPU sweeps if its interval is both
   calibrated and non-vacuous.

The present best direction is the Q/K functional-rank ladder because it has repeatedly passed prediction, price,
OOD, certificate, and signed-causal gates. The most important independent research direction is tail-robust or
downstream-aware metric learning, because it attacks the specific failure mode that prevents the much larger MLP
tensors from yielding comparable savings.
