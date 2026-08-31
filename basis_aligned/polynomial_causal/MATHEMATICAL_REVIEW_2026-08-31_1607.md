# THREE-HOURLY MATHEMATICAL REVIEW — 2026-08-31 16:07Z

Convention (§2135): CE numbers are damage above the real model; LOWER IS BETTER.

## Fresh facts this review answers to
The circuits program's carrier results: rank-32 member-PCA subspaces carry ~0.5 of interchange damage at
~29× per-dimension concentration (§2262), are ~85% family-shared (§2263), PCA-optimal vs learned search
(§2261) — but the complement also damages (0.63), and §2264 just showed the subspace adds NOTHING to
removal selectivity (1.98 vs mean-ablation's 1.94; the famous 425× interchange selectivity was the
member-position gating, not the subspace). Circuits are ~5-component objects (§2260); removal is capped
~2× by substrate sharing regardless of tool.

## Top three mathematical moves

### 1. The variance-vs-causal null battery → EXECUTED (rung 169, queued)
Object: the carrier subspaces themselves. The concentration claim has an untested confound straight from
spiked-covariance thinking: member-PCA directions maximize member VARIANCE by construction, and interchange
damage plausibly scales with patched variance. The decisive three-way control: member-PCA vs OFFSLICE-PCA
(same component's principal variance, no member specificity) vs RANDOM (dimension counting), with each
basis's captured-variance fraction recorded so the damage-vs-variance spectral density comes free.
Assumption that may fail: offslice and member variance structures coincide (substrate sharing suggests they
might — that IS the null). Consequence: decides whether the repertoire's das_subspace column means "causal
variable" or "variance patch." Falsifier: rung 169, ~14 min, zero price.

### 2. Damage spectral density vs eigenvalue spectrum (rides free in 169's receipt)
If per-direction damage density ∝ PCA eigenvalue across ranks 1/8/32 and bases, the interchange functional
is a quadratic form in the patched deviation — the Gauss-Newton picture again, now at component grain; then
carrier choice reduces to variance accounting and DAS-type searches are provably unnecessary here. If
density deviates (specific low-variance directions carry outsized damage), those directions are the true
causal coordinates and become the extraction targets. Scored post-hoc from 169's varfracs + shares.

### 3. Gated removal as the formal object (design; the §2264 lesson)
Removal selectivity is tool-invariant (~2×) because the substrate is shared; interchange achieves 425× only
via member gating. The formal move: removal operators of the form (gate g(x), operator R) where g is a
CHEAP predicate on the local stream (not an oracle) — the deployable analogue of member gating. The
feasibility fact needed first: are members linearly separable from the local stream at the component input?
A probe-AUC measurement per family is the cheapest falsifier (CPU-light GPU capture, one pass) — queued
next wake if 169 upholds the causal reading.

## Pruned
Learned DAS (closed §2261); additive-bias corrections (closed §2252); Hankel/automata (no sequential object
yet); invariant theory (rung 90 gate); MDL (bookkeeping).

## Executed
§2264 written first (the removal-cap lesson). Rung 169 built, preregistered, dryrun-clean, queued behind
rung 167 (family carriers, running).
