# Compilation-mask cut-rank v1 preregistration

Date frozen: 2026-08-28

Status: discovery-only prospective design. No validation or heldout grid cell has
been observed. This does not authorize the final role and cannot move any existing
whole-model, causal, semantic, OOD, or edit ledger.

## Question and mathematical object

S1834 falsified both additive site pricing and one global interaction multiplier:
the latter obtained \(R^2=-1.284\). Layer-1 sites are redundant, deep groups are
super-additive, and compiling MLP5 alone costs 61.2 percentage points of recovered
gap. The next question is whether these context-dependent interactions nevertheless
pass through a very small latent state at the physical cut after layer 5.

For frozen prefix masks \(P_i\) on layers 1--5 and suffix masks \(S_j\) on layers
6--17, define the same-wave behavioral cost matrix

\[
H_{ij}=C(P_i\cup S_j).
\]

If the complete compilation-mask cost function has tensor-train rank \(R\), every
cut matricization has rank at most \(R\). Therefore one measured cut with rank above
\(R\) falsifies the global rank-\(R\) claim. We remove all additive prefix and suffix
effects using

\[
\Delta_{ij}=H_{ij}-H_{i0}-H_{0j}+H_{00}.
\]

This is a left/right linear transform of \(H\), so it cannot increase rank. Additive
site pricing is exactly \(\Delta=0\).

## Frozen 8 by 8 registry

Layer 0 attention and MLP are compiled in every cell. The exact additional masks and
the exact split are executable constants in `compilation_mask_cut_rank_v1.py`.

Prefixes: empty; attention1; MLP1; both layer-1 sites; MLP5; both layer-5 sites;
both layer-1 sites plus MLP5; and the fixed balanced mixture
`{attn2, mlp3, attn4, mlp4}`.

Suffixes: empty; attention15; MLP17; both layer-13 sites; TOP13; the count-matched
scattered set at layers 9/11/13/15/17; TOP9; and the fixed balanced mixture
`{attn6, mlp7, attn8, mlp10, attn12, mlp14, attn16, mlp17}`.

All 15 empty-prefix/empty-suffix anchors are always observed. Of the 49 interaction
cells, 28 frozen cyclic-diagonal cells train the model, 10 select rank/regularization,
and 11 remain held out once. Every nonempty row and column has four training cells;
the training bipartite graph is connected. Validation and heldout both contain MLP5
and non-MLP5 prefixes and dense-deep and sparse-deep suffixes.

## Currency and data

Every grid cell, B0 anchor, live/full control, and bootstrap replicate must use the
same documents, tokens, scored positions, program build, and execution wave. The
primary target is raw percentage-point cost

\[
C(\text{mask})=\operatorname{accuracy}(B0)-
                \operatorname{accuracy}(\text{mask}).
\]

The corresponding per-document correct-count difference must be retained for a
shared source-document cluster bootstrap. Dividing by the live-minus-fully-compiled
stake is descriptive only. Historical S1830--S1834 values selected this design and
provide controls; they receive no train, validation, heldout, or bootstrap credit.

Cross-entropy cost on the identical support is a mandatory secondary target because
the model was trained on CE. A top-1-only pass cannot establish the CE claim.

## Fitted model and baselines

Fit explicit rank \(r\in\{1,2\}\) matrix completion to \(\Delta\), standardized by
train-only RMS. Use a frozen dimensionless ridge grid \(2^{-12},2^{-10},\ldots,2^4\),
fixed ALS restarts, fixed stopping tolerance, and no heldout inspection. Validation
selects \((r,\lambda)\) once by RMSE; exact ties favor smaller rank, then larger
regularization. Refit the selected pipeline within every document-bootstrap draw.

Frozen baselines are: additive anchors (\(\Delta=0\)); literal singleton sum; the
S1834 coefficient \(0.8790830549627247\) times singleton sum; a train-only
count/depth/attention/MLP ridge; and a one-dimensional monotone quadratic saturation
of singleton sum. The rank model must beat the best baseline, not merely the mean.

## Useful-pass conjunction

All conditions are required on the untouched 11 cells:

1. selected rank is at most 2;
2. total-cost RMSE is at most 5 percentage points and maximum absolute error at most
   10 points;
3. interaction NRE is at most 0.50 with 95% bootstrap upper bound at most 0.65;
4. heldout \(R^2\) is at least 0.75 with document-bootstrap lower bound above zero;
5. paired RMSE divided by the best baseline RMSE is at most 0.80, with 95% upper
   bound below 1;
6. RMSE is no worse than the best baseline separately for the frozen MLP5-containing
   versus non-MLP5 groups and dense-deep versus sparse-deep groups;
7. the full-grid best-rank-2 spectral-tail NRE has 95% upper bound at most 0.50;
8. the CE secondary target has positive heldout \(R^2\) and beats the best CE baseline.

Any failure prunes “rank at most 2 across the layer-5 cut.” A pass licenses only a
second independent cut and a larger mask campaign; it does not establish a causal
state, global tensor-train rank, or privileged physical ordering.

## Why a global tensor-train fit is deferred

An inhomogeneous length-17, alphabet-4 tensor train has approximately

\[
8R+44R^2
\]

gauge-adjusted parameters: 52, 192, 420, and 736 for ranks 1--4. The exposed masks,
even after this assay, cannot identify global rank 2. More token rows reduce outcome
noise but do not repair mask undersampling. A shared-transition weighted automaton is
smaller only by imposing a much stronger tied-layer assumption and is therefore a
separate baseline, not evidence for the inhomogeneous tensor train.
