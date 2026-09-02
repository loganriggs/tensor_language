# Preregistration — architecture-derived quadratic response falsifier (parallel lane)

Date: 2026-09-02 16:38 UTC
Owner: Claude (parallel probe lane)
Status: frozen before any λ=2.0 outcome exists anywhere

## Question

Rung494 established a regime map phenomenologically: the frozen monotone single-index
readout loses to the additive (λ-linear) predictor at half strength and beats it at
one-and-a-half strength. The 1607 mathematical review derived the network's OWN
second-order theory for the same arms: the patch p → p − λδ injects an exactly
λ-linear write perturbation (linearity of Down), every downstream bilinear MLP is
exactly quadratic in its input perturbation, and CE is analytic — so the
per-occurrence response y_i(λ) should be dominated by y = aλ + bλ² at these
magnitudes. The receipt already shows the quadratic signature in aggregate: effect
RMS grows 1.74–1.86× from λ=.5→1.5 versus 3.0× for linearity (concave, b opposing a).

This rung prices that theory with zero fitted parameters. Codex's 1608 review and
rung495 registration explicitly deferred this falsifier as a distinct object; it is
executed here in the idle slot without touching the main line.

## Exact computation

Reuse rung474/483/494's exact arms, sites (m8, m9, m12), sources (N, H), three
frozen windows, coordinates, 16 position-permutation offsets, and batch structure
unchanged, importing the hash-pinned rung494 module as a library (no frozen file is
modified). Collect in one process, per occurrence and site:

- the seven Möbius subset effects at unit strength (y1 = the singleton effect);
- scaled single-site effects at λ ∈ {0.5, 1.5, 2.0}. **The λ=2.0 outcomes have never
  been measured in any run**; λ∈{0.5,1.5} are re-measured in-run (post-§2599 rule:
  no cross-session numeric bridges).

Solving y(λ)=aλ+bλ² through the two in-run points (0.5, y½) and (1, y1) gives
closed forms with zero free parameters:

- a = 4·y½ − y1, b = 2·y1 − 4·y½;
- prediction at 2.0: y_quad(2) = 6·y1 − 8·y½;
- prediction at 1.5: y_quad(1.5) = 3·(y1 − y½);
- downward direction from (1, y1),(1.5, y1.5): y_quad(0.5) = y1 − y1.5/3.

Competitors, computed identically in-process: the additive predictor λ·y1, and the
rung494 incumbent (per-occurrence isotonic eight-point curve, fit by the identical
frozen code path, evaluated at λ·y1 with its registered clipping; in-range fractions
reported). Controls: the same 16 position-permutation donor rolls, applied to the
quadratic's input pair (donor's (y½,y1), or (y1,y1.5) for the downward direction);
q05 of the 16 permuted median errors is the control bound, as in rung494.

Errors are median absolute error over site-stacked occurrences per cell
(3 windows × 2 sources = 6 cells), exactly rung494's stacking.

## Data scope and evidence grade

Same hash-frozen documents and coordinates as rung474/494; no sealed or final role
is opened. The λ=2.0 clause is prospective intervention-outcome evidence (new
physical outcomes, not new-corpus OOD). The λ=1.5 and λ=0.5 clauses compare
zero-parameter closed forms against outcome TYPES rung494 already opened
(re-measured in-run); they are labeled in-run coherence, not prospective evidence.
The prospective weight of this rung rests on pred_b alone.

## Frozen predictions

### pred_a — exact, lawful, live instrument
All frozen hashes match (rung494 source/result/bundle, rung474 chain via the parent
validator, this preregistration); rung494 has pred_a true, pred_c true, strong_null
true, validation closed. Native replay relative squared ≤ 1e-12; factor
reconstruction ≤ 1e-10; empty-mask patch exactly zero; unit-strength bridge
≤ 3e-5 nat; every scaled intervention live (nonzero RMS) at all three scales;
adjacent-scale effect-difference RMS ≥ 1e-4 for (0.5,1.5) and (1.5,2.0); forwards
exactly 5130 and patch calls exactly 6498 by the stated formulas.

### pred_b — the quadratic law predicts the never-measured λ=2.0 outcomes
In every one of the 6 cells:
- additive median absolute error ≥ 1e-4 nat (materiality floor);
- quadratic median error ≤ .80 × additive median error;
- quadratic median error ≤ .90 × the q05 of the 16 permuted-donor quadratic errors;
- quadratic median error ≤ 1.00 × the incumbent isotonic median error at λ=2.0.

### pred_c — two-sided in-run coherence at the measured scales
In every cell: y_quad(1.5) median error ≤ .85 × additive at 1.5; and the downward
prediction y_quad(0.5) has median error ≤ 1.05 × additive at 0.5 (graceful
degradation where rung494's readout lost by 6.6–21.9%) and ≤ 1.00 × the isotonic
median at 0.5.

### pred_d — document-half stability of the prospective clause
In both document halves of every cell: additive-at-2.0 median ≥ 1e-4 and quadratic
≤ .90 × additive at λ=2.0.

Descriptive (no bars, reported regardless): sign(a·b) opposition fraction per cell,
|b/a| quantiles and the implied linear-regime half-width, isotonic in-range
fractions at each scale, full per-cell error tables, permuted error lists.

## Strong null and interpretation

The strong null fires if any of pred_a–pred_d fails: composition raises the
effective degree, the response curve is not two-point-quadratic-determined, the
(a,b) causal response chart is dead, and rung494's regime map stands exactly as it
left things. Bars will not be relaxed; a partial pass is reported as the null.

A full pass identifies a two-parameter-per-occurrence CAUSAL response chart
(a,b) grounded in the architecture: prediction at unmeasured strengths, selective
partial removal with predictable effect, and a per-occurrence certificate |b/a|
bounding the linear-regime width. It does not license compression, rank reduction,
or any cross-site claim.

## Literal price

72 batches × 71 forwards + 18 bridge forwards = 5,130 full-model forwards; 6,498
product-hook patch calls; one results JSON + one per-token bundle; zero deployed
parameters added or removed. Runs once in the idle slot; no validation phase.
