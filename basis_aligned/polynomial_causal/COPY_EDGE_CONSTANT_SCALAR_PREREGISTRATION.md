# Copy-edge constant-scalar replacement

Status: **exploratory; frozen before any outcome from this runner**.

This experiment reuses the already exposed `selection_natural` cache.  The first
32 documents are a fit split and documents 33--128 are a disjoint evaluation split.
It is not fresh confirmation.  Targets from the evaluation split may be used only
for scoring, never for source selection or coefficient fitting.

## Question

The preceding exact-edge experiment found that, at natural-text copy-positive
destinations, 95.4% of the same-position causal CE damage from L8 H3/H4 is carried
by the single successor edge from the nearest earlier matching token.  Most of that
edge's payload is the shared $\lambda_8 v_1$ token code.  The remaining native
quantity is the attention-pattern scalar

$$
a_h(p,j(p)+1),
$$

computed by four query/key projections and two dot products for each head.  This
experiment asks whether the conditional edge can instead use just one constant
$c_h$ for each of H3 and H4.

For every destination $p$ with a nearest earlier equal-token position $j(p)$ within
128 tokens, the simplified broadcast write is

$$
\widetilde w(p)=\sum_{h\in\{3,4\}}
c_h P_h\bigl(\lambda_8 v_{1,h}(j(p)+1)\bigr).
$$

The equality search and successor operation use input token IDs only.  The physical
replacement subtracts the native **mixed-value** successor-edge write and adds the
simplified write.  Thus a successful broadcast arm replaces both the dynamic
attention scalar and the small context-refined value term, while leaving every
other model computation native.

## Frozen fits

Constants are computed once from native L8 source-pattern scalars on fit documents
1--32, then frozen:

1. `fit_eligible`: per-head arithmetic mean over every input-eligible destination.
   This fit uses no next-token outcomes.
2. `fit_positive`: per-head arithmetic mean over scored fit destinations whose next
   token equals the observed successor token.  This may use fit outcomes, but never
   evaluation outcomes.
3. `historical`: H3 = `-0.119`, H4 = `+0.190`, frozen in the older synthetic-repeat
   payload experiment.  This is a zero-new-fit transfer control.

## Frozen evaluation arms

All interventions apply at every input-eligible destination, including before the
scoring window.  Scoring is on positions 64--255 of evaluation documents 33--128.

1. `native`: no intervention.
2. `edge_removed`: remove the exact native mixed-value successor edge.
3. `native_pattern_broadcast`: retain the native scalar but replace the mixed value
   with its $\lambda_8v_1$ broadcast part.  This isolates payload simplification.
4. `fit_eligible_mixed`: replace only the native scalar by the two input-only fitted
   constants; retain the native mixed value.  This isolates scalar simplification.
5. `fit_eligible_broadcast`: replace both scalar and payload with the input-only fit.
6. `fit_positive_broadcast`: same complete replacement using positive-only fit.
7. `historical_broadcast`: same complete replacement using the historical constants.
8. `wrong_source_fit_broadcast`: use the fitted constants and broadcast value but
   read from $j(p)$ rather than $j(p)+1$.  This directional control must not recover
   the effect.

Cells remain `copy_positive`, `repeat_negative`, `nonrepeat`, and `all_scored` as
defined in the exact-edge preregistration.

## Frozen metrics and decision rules

For replacement arm $r$, define copy-positive causal recovery relative to deletion:

$$
R_r=1-\frac{\mathrm{CE}_r-\mathrm{CE}_{\mathrm{native}}}
{\mathrm{CE}_{\mathrm{edge\ removed}}-\mathrm{CE}_{\mathrm{native}}}.
$$

$R=1$ is native-equivalent on this cell; $R=0$ is no better than deleting the edge.
The primary arm is `fit_eligible_broadcast` because it uses no target-fitted labels.

- C1, useful constant compiler: primary copy-positive recovery $R\ge 0.70$.
- C2, scalar itself is simple: `fit_eligible_mixed` recovery $R\ge 0.85$.
- C3, selective behavior: primary repeat-negative and nonrepeat absolute $\Delta$CE
  are each at most `0.02` nat and no more than 25% of edge-deletion copy damage.
- C4, directional source: wrong-source recovery is at most half the primary recovery.
- C5, fit choice is not outcome-dependent: input-only recovery is within 0.10 of
  positive-only recovery.

All effects also report document means and standard errors.  Passing C1 would give
an executable local copy program with an equality/successor selector, two stored
scalars, the already shared $v_1$ value bus, and two fixed projection slices.  It
would not yet replace the upstream machinery that decides when copy behavior should
affect the final prediction, nor would this exposed split constitute fresh OOD
confirmation.

