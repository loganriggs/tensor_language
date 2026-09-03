# Rung 576 preregistration: compile and actively remove the final-label cached-value path

## Circuit-level question

R573 identified an activation factor for numbered-list successor: the layer-0 cached value of the final visible label,
routed at the final query through L8H7 and L8H3. R576 asks whether this factor can be computed directly from weights and
whether deleting it is necessary for successor behavior while leaving nearby copy behaviors intact. Digit, number-word,
and cross-format comma sequences test whether the same factor is a shared numeric-successor subroutine.

This is not a rank or compression test. It targets exact extraction, selective manipulation, and cross-boundary
grouping/reuse.

## Exact weight computation

Let $z_k^{(0)}$ be the layer-0 attention input at source token $k$. For final query $q$, the compiled contribution is

$$
T(x;q,k)=\sum_{h\in\{3,7\}} p^{(8)}_{h,q,k}(x)\,
W_{O,h}^{(8)}\left(\lambda_8 W_{V,h}^{(0)}z_k^{(0)}\right),
\qquad \lambda_8=4.
$$

$z_k^{(0)}$ is computed directly from the token embedding, block-0 residual coefficients, and RMS normalization. The
compiled donor patch replaces the base $T$ by the donor-value version while retaining the base layer-8 attention score.
The removal intervention subtracts the native $T$ from the layer-8 residual write. It does not call a learned probe,
capture a donor layer-8 activation, update weights, or remove an entire head.

## Frozen rows

R575 mapped a single-token final numeric source and final separator query for 528 R567 FIT/SELECT rows. R576 uses:

- list necessity families: two-line state shift, three-line state shift, surface-preserved list, earlier-middle-label
  edit, and the step-two conflict whose native answer is final-label-plus-one;
- active copy controls: repeated-label lists, repeated-digit comma sequences, and repeated-number-word comma sequences;
- shared-subroutine characterization: digit, number-word, and digit-to-word comma-sequence state shifts.

Each endpoint is evaluated separately for removal. The paired list state shifts are additionally used for direct
compiled-versus-activation donor-patch equivalence. FIT opens first. SELECT opens only if all required FIT gates pass.
FINAL_TEST and OOD remain closed.

## Measurements

For correct answer $a$ and representation-matched candidate set $N$,

$$
m(x,a)=z_a(x)-\max_{b\in N,\,b\ne a}z_b(x).
$$

Removal margin damage and cross-entropy increase are

$$
d_m=m_{\mathrm{native}}-m_{\mathrm{removed}},\qquad
d_{CE}=CE_{\mathrm{removed}}-CE_{\mathrm{native}}.
$$

Full-vocabulary logit RMS measures all output changes. The Euclidean norm of $T$ before deletion verifies that a copy
control received an active intervention rather than a zero tensor.

## Frozen decisions

1. **Exact compilation.** Direct layer-0 cached values must match the value bus observed at layer 8 with relative
   squared error at most $10^{-10}$. The projected source term and the compiled donor-patch logits must match the
   activation implementation with relative squared error at most $10^{-10}$. Native replay must be at most $10^{-12}$.
2. **List necessity.** In every list-family × endpoint cell, at least 75% of rows must have $d_m>0$, with a positive
   2,000-group-bootstrap lower mean $d_m$. The bootstrap lower mean $d_{CE}$ must also be positive.
3. **Active copy preservation.** In every repeated-list, repeated-digit, and repeated-word copy cell, the median
   compiled-term norm must be at least 10% of the median list-target term norm. The correct answer must remain the best
   registered numeric candidate in at least 75% of rows. Mean $d_{CE}$ must be at most 0.1 nat. Median absolute margin
   change and median full-vocabulary logit RMS must each be at most 25% of their corresponding FIT list-target scale.
   FIT target scales are reused unchanged on SELECT.
4. **Shared successor characterization.** The three comma-sequence state-shift families are scored with the same
   necessity inequalities. If all digit, word, and cross-format cells pass on FIT and SELECT, that supports one shared
   cached-value successor subroutine; a partial or null result supports format/task-specific routing. This
   characterization does not rescue a failed list or copy-preservation gate.

The deterministic dry run contains 9 FIT and at most 8 conditional SELECT equivalence batches, plus 34 FIT and at most
19 conditional SELECT removal batches. The maximum price is 210 model forwards, zero backwards, zero fitted
parameters, and zero weight updates. A null blocks adoption of this factor as a selectively removable circuit;
thresholds, families, and the intervention may not be weakened after outcomes.
