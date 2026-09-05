# Pending-opener semantic-OPEN shared/contrast decomposition

## Evidence boundary and question

The fresh semantic-OPEN zero-removal result is live and necessary on both target constructions, but nonselective:
target normalized damage is 0.978--1.102 and positive in every target cell, while same-state control normalized damage
is 0.9725 with absolute closer-margin damage 2.29. Answers remain preserved. The exact opener term therefore supports
more than delimiter identity. The next question is whether it is the sum of shared pending-closer support and a
delimiter-type contrast.

This is not a rank, PCA, projector, reconstruction, or learned-direction proposal.

## Mathematical object and gauge

For each fresh lexical/template context `g` and construction family `f`, construct a token-aligned natural triplet
whose only target change is the pending opener/closer type `c` in `{parenthesis, square, quote}`. At final query `q`
and semantic opener position `o`, the exact projected L13H8 contribution is

`t[g,f,c] = p[g,f,c](q,o) * W_O,8 v[g,f,c](o) in R^1152`.

All three terms are expressed in the same native model output basis and injected at the same layer, head, and final
position. Define

`mu[g,f] = (t[g,f,paren] + t[g,f,square] + t[g,f,quote]) / 3`

and

`delta[g,f,c] = t[g,f,c] - mu[g,f]`.

The gauge is fixed by the arithmetic mean and the constraint `sum_c delta[g,f,c] = 0`. No fitted metric, direction,
rank, or outcome-selected weighting is allowed. Context `g,f` includes all nonopener tokens, their positions, and the
construction; means are never pooled across lexical contexts or constructions.

## Exact interventions

Let `h` be the native final-position L13H8 projected write containing native term `t_c`.

- Native replay: `h`.
- Natural donor-term swap / contrast swap `c -> c'`: `h - t_c + mu + delta_c' = h - t_c + t_c'`.
  Exact equality to the already-supported natural donor operation is a required implementation identity.
- Contrast removal with shared support preserved: `h - t_c + mu`.
- Shared removal with native contrast preserved, diagnostic only: `h - mu`.

Before interpreting centered arms, every family/direction must have native capability, exact replay, a live term, and
a passing natural donor-term swap. A failed natural swap makes the centered experiment invalid, not null.

## Opposing predictions

Define two live output axes before outcomes. For closer logits `z_c`, the type axis for native type `c` is
`S_c = z_c - mean_d z_d`; the common-support axis is `M = mean_d z_d`. Both can move, and a common-mode perturbation
fails selectivity by moving `M` without moving `S`. The centered contrast arm must have median
`abs(Delta S) / (abs(Delta M) + 1e-6) >= 2` separately by construction and direction. This ratio replaces the generic
screen's fixed same-answer C bar, which subsequent cross-screen audit showed was structurally nondiscriminating.

Same-state surface and punctuation rewrites are invariance controls: their centered type effects should agree with the
matched unrewritten context, not be forced toward zero. Require the matched normalized-effect difference to be <=0.25.

**Shared-support plus type-contrast hypothesis.** Natural/contrast swap transfers delimiter identity. Contrast removal
reduces the centered correct-closer type axis with target/common ratio >=2. Shared removal changes the common closer axis
more strongly than the centered type axis. The contrast effect is stable across matched same-state rewrites.

**Pure shared-support alternative.** Natural contrast swaps and contrast removal have negligible delimiter-specific
effects, while shared removal accounts for essentially all full opener-zero damage.

**Entangled-context alternative.** Natural term swap remains live, but arithmetic centering fails selectivity: contrast
removal moves the common closer axis too strongly, does not survive matched rewrite invariance, or shared removal has
construction/type-specific effects inconsistent with a common support term.

## Cheap preflight and phase boundary

The row authority must contain complete three-type groups, equal token lengths and opener/final positions within each
triplet, balanced delimiter roles, and fresh text disjoint from prior bracket authorities. Unit tests must establish
exact reconstruction, zero-sum centering, permutation equivariance, contrast-swap equality to natural term swap, and
the named complement preserved by each removal. Only after those checks should a small held-out causal run be priced.
OOD remains closed. No GPU or queue action is authorized by this preregistration.
