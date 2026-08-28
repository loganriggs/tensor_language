# Preregistration: typed early-layer observable closure

Date frozen: 2026-08-28 16:30 UTC

Status: prospective mathematical and measurement contract only. This document opens
no model outcome, assigns no data role, and grants no GPU or final-result authority.

## Question

Can the selected MLP0 representation be made into a small **state** rather than merely
a low-rank local approximation? A state must obey a reusable transition law: after an
allowed edit, it must predict what MLP1 produces and what MLP2/the live suffix observes,
including when the two transitions are composed.

For layer-specific observable maps $\phi_i$ and physical edit $a$, fit non-autonomous
operators

$$
\phi_1(x_1)=\psi_0(\phi_0(x_0),a)K_0+e_0,
\qquad
\phi_2(x_2)=\psi_1(\phi_1(x_1),a)K_1+e_1.
$$

“Non-autonomous” means $K_0$ and $K_1$ need not be the same: MLP0 and MLP1 are
different maps. “Typed” means residual coordinates, inverse-RMS scalars, code
coordinates, actions, and downstream response coordinates are never silently treated
as interchangeable vectors.

The central audit is the exact two-step identity

$$
\phi_2-X_0K_0K_1=e_1+e_0K_1,
$$

and therefore

$$
\lVert\phi_2-X_0K_0K_1\rVert_G
\leq \lVert e_1\rVert_G+\lVert e_0K_1\rVert_G,
$$

where $G$ is a registered downstream Fisher/response metric. This is a finite
composition certificate, not an assumption that the two errors are independent.

## Frozen candidate grammar

The experiment must compare nested dictionaries in this order:

1. **Affine:** $[1,c,a]$.
2. **RMS typed:** $[1,c,a,\rho,\rho c,\rho a]$, where $\rho$ is the exact registered
   inverse-RMS scalar at that interface.
3. **Controlled polynomial:** the preceding terms plus $c\otimes a$ and
   $a\otimes a$. Any $c\otimes c$ tier is a separately priced diagnostic because its
   quadratic storage can erase the claimed simplification.
4. **Native-product tier:** only products named by a source-frozen tensor program.
   It may not be selected after viewing validation consequences.

Each interface gets its own dictionary and transition. A single global operator is
not required. Constants and native bias terms are explicit. A random rotated basis and
an affine dictionary with matched column count are mandatory controls.

## Estimation and prices

For source rows $X$, target rows $Y$, positive-semidefinite metric $G$, and registered
rank $r$, solve

$$
\min_{\operatorname{rank}(B)\le r}\lVert(Y-XB)G^{1/2}\rVert_F^2.
$$

The source-closed CPU implementation projects $YG^{1/2}$ into $\operatorname{col}(X)$,
takes the rank-$r$ truncated SVD, and maps back on the support of $G$. This is the
reduced-rank regression optimum for the registered dictionary and metric. It is not a
global optimum over dictionaries.

Report separately:

- stored floating values, nonzero monomials, and multiplications per token;
- dictionary description bits, action metadata, and decoder values;
- one-step weighted closure defects;
- two-step composed defect and its certified triangle upper bound;
- direct source-to-final reduced-rank error at the same rank and at matched stored
  values;
- final CE/KL, finite-edit response, and live-consumer output-norm ratios.

No quantity may be called “simpler” from local MSE alone.

## Data and interventions

Use document-disjoint FIT, VALIDATION, and REPLICATION roles frozen before outcome
access. The exact identity source must come from an independently audited lifecycle;
this document does not authorize opening the existing unpublished role freeze.

The action bank must contain:

- all-on/no-edit controls;
- single registered code directions at small finite positive and negative scales;
- unseen pairs and mixtures held out from fitting;
- matched random directions;
- finite component removals only after small-edit prediction passes.

The live consumer must record its output norm under native and substituted inputs at
every affected layer. This is required because a context-destroying substitute has
already produced 74--153-fold consumer explosions elsewhere in the project.

## Promotion and rejection rules

A dictionary/rank may be promoted from VALIDATION only if all of the following hold,
with simultaneous document-bootstrap 95% intervals:

1. Its two-step composed error is no more than 1.10 times the direct rank-and-cost
   matched source-to-final control.
2. It improves composed downstream error by at least 5% over both the affine and
   column-count-matched rotated controls; the lower confidence bound is positive.
3. On unseen edit mixtures, response $R^2\geq0.75$ and no registered stratum has more
   than twice the in-distribution normalized error.
4. The data-doubling estimate (half FIT versus full FIT) changes the primary normalized
   defect by less than 10% and does not change the selected rank by more than one grid
   step.
5. Every live-consumer median norm ratio lies in $[0.5,2]$ and the 99th percentile is
   below 4. Failing this is an interface failure, regardless of CE.
6. Full executable CE/KL is measured with native bias, RMSNorm, attention context, and
   residual additions present. A missing term invalidates the arm.

The typed-closure hypothesis is rejected in this grammar if the RMS/polynomial tiers
cannot beat the controls, if their apparent gain disappears on REPLICATION, or if
good one-step fits fail the two-step composition gate. Rejection does not imply that
no nonlinear state exists; it prunes this finite observable grammar.

## Source-only work completed

`typed_koopman_closure.py` implements the CPU reduced-rank optimum, explicit constant
augmentation, positive-semidefinite metric support, weighted errors, and the exact
two-step certificate. Its tests use synthetic matrices only. No checkpoint, token,
document role, cached activation, or scientific result was read.
