# The direct-token interaction contributes, but does not explain the calibrator

The new weight-based test gives a clearer input decomposition, but no accepted
single-term circuit. The direct token-injection term is very small. Most of the
scalar comes from the rest of the residual, with a smaller token/context
interaction that redistributes prediction quality between frequent and rare
tokens. Neither the token term, the context term, nor the interaction alone
passes the registered fidelity requirements across both corpora.

We derived the three-term formula, verified it and the native input capture,
ran its interventions through the managed GPU runner, then checked the class
effects on CPU. We also implemented controls for a more specific next hypothesis:
whether the quadratic computes vocabulary-score spread. That hypothesis has not
yet been tested against the trained weights.

## The exact input split

The model repeatedly injects the current token's normalized embedding x0 into
the residual. Its direct contribution before the final MLP is T=B*x0, where
the scalar coefficient follows B<-lambda0*B+lambda1 through the eighteen blocks,
starting at1. The native weights give B=145.12154.

Let a be the actual pre-MLP residual and define C=a-T. C contains every remaining
computation, including indirect effects of the current token and numerical
accumulation differences. It is a complement of the **direct injection route**,
not a token-free representation of context.

The previously folded scalar is q(u)=u^TQ u+beta. With
rho²=mean(a²)+epsilon and u=a/rho, its source expansion is

\[
q=\beta+\underbrace{T^TQT/\rho^2}_{\text{direct-token term}}
+\underbrace{2T^TQC/\rho^2}_{\text{interaction term}}
+\underbrace{C^TQC/\rho^2}_{\text{complement term}}.
\]

Even the first term has a contextual denominator. Keeping that term is not a
standalone token lookup. The decomposition is an exact real-arithmetic identity;
native FP32 normalization versus FP64 arithmetic gives a measured source-sum
relative error of about1.0e-7. The captured raw input renormalizes to the native
MLP input exactly, and an independent model replay also agrees exactly.

The interventions replace only the original output projection q*w, keeping its
native background and recomputing final normalization and vocabulary scores.
They correspond to edits of compiled quadratic terms. They do not claim that
deleting native upstream token injections would leave that same background.

## Results and preserved failures

The root-mean-square magnitudes below are relative to the full scalar q.
They are not additive variance fractions: the terms can correlate and cancel.

| Term | FineWeb | Pile |
|---|---:|---:|
| Direct-token term | 0.0068 | 0.0061 |
| Interaction | 0.1003 | 0.0934 |
| Complement | 0.9120 | 0.9261 |

Keeping the direct-token term alone leaves nearly the same prediction damage
as removing q entirely. Keeping only the interaction also loses most of the
effect. The complement is much closer, but misses the fixed0.10 limit on
FineWeb: scalar error0.1056 and full-vocabulary effect error0.1121. It passes
the individual Pile cell, with errors0.0979 and0.0869. The required two-corpus
sufficiency verdict remains **failed**; we do not relax the bar for a near miss.

Removing just the interaction produces these cross-entropy changes, in nats:

| Token class | FineWeb | Pile |
|---|---:|---:|
| Frequent | −0.0454 | −0.0274 |
| Rare | +0.0153 | +0.0195 |
| All tokens | +0.0028 | +0.0080 |

Positive means worse next-token prediction. The registered interaction-necessity
requirement was at least0.005 nat overall damage on both corpora. FineWeb misses
that bar, so this prediction also fails. This does not mean the interaction is
inert: removing it helps frequent tokens and hurts rare ones. The average partly
cancels those changes.

A post-result4,000-draw row bootstrap gives rare-token damage intervals
[0.0126,0.0181] on FineWeb and [0.0134,0.0261] on Pile. The same calculation
reconstructs overall loss as f*frequent_loss+(1-f)*rare_loss, where f is the
fraction of frequent targets. It illustrates why aggregate loss alone cannot
identify a selective computation. These intervals do not change any verdict.

FineWeb document grouping remains unknown; Pile has one row per sampled
document. The42 FineWeb and16 Pile evaluation rows are reused, so this is a
conditional follow-up, not pristine discovery or certified training OOD.

The run used18 model-body forwards and70 sequence instances in9.09 seconds.
Online interaction-removal edits agree with the formula within3.47e-5 maximum
absolute logit error and6.89e-7 relative error. All545,902,902 native parameters
remain. Nor would deleting branches from this expanded expression automatically
reduce cost against the original compact q(u) computation: calculating C itself
requires the direct-route subtraction.

## Next hypothesis: an explicit reused vocabulary reader

"A quadratic of context" is too broad to explain the operation. A more specific
candidate is the spread of vocabulary scores produced by the existing unembedding
U. For scores s=Uu, define their mean squared deviation from their vocabulary mean:

\[
E(u)=\frac1V\sum_{v=1}^V(s_v-\bar s)^2=u^TKu,
\quad
K=\frac1V U^T\left(I-\frac{\mathbf1\mathbf1^T}{V}\right)U.
\]

This is score spread, **not entropy**. It is a proposed input computation,
not a target to preserve activation variance. If the calibration quadratic has
the form Q=alpha*K+gamma*I, then

\[
q(u)=\alpha E(u)+\gamma\|u\|^2+\beta.
\]

That would explain it as reuse of the vocabulary reader plus a norm term and
three scalar coefficients. The reader weights, input dependencies and execution
cost remain charged. It would still need native behavioral and intervention tests.

There is a direct weight-level falsifier. Subtract each matrix's isotropic part:
Q0=Q-trace(Q)/d*I and K0=K-trace(K)/d*I. Exact equality requires Q0=alpha*K0.
The least-squares scalar alpha is the matrix inner product <Q0,K0>/||K0||²,
provided K0 is nonzero. A nonzero residual rules out that exact global form.
On a fixed-radius sphere the isotropic term is constant; native normalization
with epsilon does not justify silently treating the radius as exactly fixed.

The CPU controls verify the score-spread Gram identity to2.7e-15, recover a
planted Q=2.3*K−0.7*I example, and detect a perturbed example with0.145 relative
matrix residual. An isotropic reader is rejected as unable to identify alpha.
The helper's `identified` field denotes a well-defined coefficient fit; equality
requires the residual test. It is not a claim that a circuit has been identified.
No trained-weight comparison or native adoption result exists for this candidate yet.

Evidence: [preregistration](../CALIBRATION_TOKEN_CONTEXT_V1_PREREGISTRATION.md),
[native result](../CALIBRATION_TOKEN_CONTEXT_V1_RESULT.json),
[post-result audit and next-operation controls](../CALIBRATION_TOKEN_CONTEXT_AUDIT_V1_RESULT.json),
[reader-energy certificate](../reader_energy_certificate_v1.py).
