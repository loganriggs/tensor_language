# A useful context-dependent scalar, but an unstable fitted coordinate

The original bilinear handoff and pilot motivated a more useful question than
another is/was refinement: can we express an actual computation from the weights,
identify its consumers, and predict edits to it? The latest test supports part
of that aim. The last bilinear MLP produces a scalar whose value needs to match
its context. Replacing it with a constant or another text's value hurts prediction.
Its two uses—vocabulary scores and normalization—have an explicit joint formula.
However, fitting on different samples does not yet recover sufficiently similar
causal effects. This is a conditional computation, not a completed circuit discovery.

The sequence was: derive an exact replacement formula; test it on small numerical
controls; refit the proposed direction on two disjoint fitting samples; run mean
and donor replacements on the real model; then audit uncertainty and the scalar's
constant component on CPU. One enqueue attempt was rejected because the runner's
static gate expected literal prediction keys. That syntax was corrected before
any native execution; the experimental formulas and thresholds did not change.

## What we tested

The preceding [two-consumer result](calibration_scalar_and_two_consumers_2026-09-10.md)
defined an output direction **w**, a vector in the model's 1,152-dimensional
residual space. It was fitted using only 48 FineWeb text rows. The fitting signal
was covariance with the log frequency of the next token in those rows; this is
not a proof that the resulting scalar literally counts token frequency.

The last bilinear MLP has native weights L, R, D and output bias b_M:

\[
M(u)=D[(Lu)\odot(Ru)]+b_M.
\]

Here u is the model's normalized input to that MLP. The symbol \(\odot\)
means elementwise multiplication. Projecting its output onto w gives

\[
q(u)=\frac{w^\top M(u)}{w^\top w}=u^\top Q u+\beta,
\qquad
Q=\operatorname{sym}\!\left(L^\top\operatorname{diag}
\left(\frac{D^\top w}{w^\top w}\right)R\right).
\]

The operator sym averages a matrix and its transpose; beta is the projected bias.
This folds native weights into an explicit quadratic computation. It keeps all
coefficients and makes no low-rank approximation. Q is still a dense matrix, so
this algebra alone does not make its arbitrary numerical content explained.

We made two additional fits, using the first and second 24 fitting rows. Each
used its own token-frequency counts. Evaluation reused the same 42 held-out
FineWeb rows and 16 separate Pile documents as the preceding experiment.
FineWeb document grouping is unknown; the two fits are row-disjoint, not certified
independent-document samples. Pile is a corpus shift, not certified unseen training data.

We then compared six arms: the native model; removal of the original projection;
removal along each separately fitted direction; replacement with the original
scalar's fitting-set mean; and replacement with a donor scalar. Each donor is
the next row in its cohort, cyclically, at the same token position. This rule
was fixed before inspecting effects. It preserves the scalar's empirical
distribution at each position while changing which context receives it.

## Results: matching the scalar to its context matters

Cross-entropy is the model's average negative log probability of the actual next
token. We report its increase in natural-log units, or nats: positive means worse.

| Edit to original scalar | FineWeb loss increase | Pile loss increase |
|---|---:|---:|
| Remove it entirely | 0.338 | 0.380 |
| Replace with fitting-set mean | 0.182 | 0.201 |
| Replace with another row's computed value | 0.287 | 0.395 |

Both replacements worsen average loss in every evaluation row: 42/42 FineWeb
and 16/16 Pile. The registered requirement was at least 0.005 nat for each edit
on each corpus; it passes. A useful constant offset cannot explain all of this
result. Matching the computed value to its recipient matters under these edits.
This does not establish which semantic information it encodes, or exclude a
dependence on token difficulty, context scale, or other upstream quantities.

Post-result 4,000-draw row-bootstrap intervals for mean replacement are
[0.162, 0.202] nats on FineWeb and [0.158, 0.247] on Pile. FineWeb intervals cannot
account for unknown document grouping. Donor effects are reported descriptively:
cyclic donor pairs share rows, so we do not present an ordinary independent-row
bootstrap as document-level uncertainty for those edits.

## The exact math of the two consumers

Let h be the final residual vector and let g=h-qw be the retained background.
Replacing q by s gives h(s)=g+sw. With U the unembedding matrix, define

\[
A=Ug,\quad B=Uw,\quad
c=\operatorname{mean}(g^2)+\epsilon,\quad
d=\operatorname{mean}(g\odot w),\quad e=\operatorname{mean}(w^2).
\]

Then the complete final logits are

\[
z(s)=30\tanh\!\left(
\frac{A+sB}{30\sqrt{c+2ds+es^2}}\right).
\]

The numerator explains the direct vocabulary-score use. The denominator explains
the normalization use. Both consume the same s, and their joint effect generally
does not equal the sum of their separate effects. The native epsilon and softcap
are included; the formula retains the recipient's original upstream background.

For one vocabulary coordinate before the final softcap, the derivative is

\[
\frac{d}{ds}\frac{A+sB}{\sqrt{c+2ds+es^2}}
=\frac{Bc-Ad+s(Bd-Ae)}{(c+2ds+es^2)^{3/2}}.
\]

Its sign can change with s. Even g=(1,0), w=(0,1), U=(1,0) gives a score that
increases toward s=0 and decreases afterward. Thus a globally monotone
"calibration strength" interpretation would be unjustified. The final tanh
multiplies this derivative by a nonnegative saturation factor.

The CPU formula control agrees to 8.9e-16; its derivative agrees with finite
differences to 1.5e-10. On the trained model, native and compiled donor edits
agree with the formula within 4.83e-5 maximum absolute logit error and 6.87e-7
relative error. Those online bridges cover the first four rows of each cohort;
the offline replacement scoring covers all evaluation rows.

This establishes numerical correspondence for a specified scalar intervention.
It does not by itself establish transfer of an independently defined semantic
variable. That distinction is consistent with the intervention correspondence
framework in [Geiger et al., Causal Abstraction](https://jmlr.org/papers/v26/23-0058.html).
The equations here are our direct derivation for this readout.

## Why the stability failure matters

The cosine between the separately fitted directions is 0.953, above the 0.90
geometric threshold. Cosine measures their angular alignment. But the real test
was their causal effect on the entire vocabulary, after subtracting the native
output and removing the irrelevant common logit offset.

| Fitted direction | FineWeb relative effect error | Pile relative effect error |
|---|---:|---:|
| First 24 rows | 0.111 | 0.104 |
| Second 24 rows | 0.340 | 0.308 |

The registered maximum was 0.20 for both directions on both cohorts. Therefore
operational stability fails. Both directions retain the broad rare-token harm /
frequent-token benefit signature, but similar geometry does not ensure the same
computation under interventions. We do not select the better fit after seeing this.
Because each fit used its own histogram, this test combines sensitivity to the
sample with sensitivity to the frequency reference; it does not separate them.

## A further exact check: is this just the constant part of a quadratic?

Write Q=alpha I+Q0, where alpha=trace(Q)/n, n=1,152, and Q0 has zero trace.
For u=RMS(x), the native normalization convention gives the exact identity

\[
q=\beta+\operatorname{tr}(Q)+u^\top Q_0u
-\frac{\operatorname{tr}(Q)\epsilon}{\operatorname{mean}(x^2)+\epsilon}.
\]

The first two terms are constant; the last is the normalization-epsilon correction.
The CPU identity check spans small and large raw inputs and agrees to 2.4e-16
relative error. The constant contribution is approximately **−518**, while the
observed fitting-set mean q is **36,071**. The large positive mean therefore is
not simply the isotropic constant term of Q. It comes through the remaining
quadratic and its actual input distribution. No uniform-input assumption is made.
This is an algebraic separation, not a low-rank discovery or a proof of semantics.

## What this adds to the four-property goal

We can now predict and execute conditional donor edits, and have evidence that
the scalar's contextual value matters on both evaluation corpora. The earlier
two-consumer formula remains valid. Stable identification, independent upstream
extraction, semantic interchange and reduced structural description cost remain
open. All **545,902,902** native parameters are still retained.

The managed test used 32 model-body forwards, 126 length-256 sequence instances,
and 8.72 seconds of executor time. It did not require an hours-long CPU search.
The next identification question is why independently fitted directions produce
different effects; the present result does not license another unregistered
direction sweep or a claim that a unique calibration circuit has been recovered.

Evidence: [preregistration](../../CALIBRATION_STABILITY_CONTEXT_V1_PREREGISTRATION.md),
[native result](../../CALIBRATION_STABILITY_CONTEXT_V1_RESULT.json),
[post-result math audit](../../CALIBRATION_CONTEXT_MATH_AUDIT_V1_RESULT.json),
[replacement formula implementation](../../calibration_scalar_path_v1.py).
