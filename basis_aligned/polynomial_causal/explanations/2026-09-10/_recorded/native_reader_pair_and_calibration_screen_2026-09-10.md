# A native bilinear product with a small causal effect, not the calibrator

We extracted and tested the complete native MLP cross term suggested by the
previous equal-energy counterexample. It is a real weight-defined computation:
read two coordinates, multiply them, and write a fixed vector. Removing it has
a measurable effect far above the random controls. But it fails the registered
calibration role, and its output cannot be adequately summarized by the earlier
calibration direction. This pair is closed as the proposed calibrator; we did
not select another pair or tune its strength after seeing the outcome.

The sequence was: derive the full vector writer, verify the bilinear identity,
test removals on the trained model, and audit natural activation scale and
coordinate changes on CPU. The result explains why a striking algebraic
counterexample need not identify a dominant computation on natural text.

## The component is defined by the native weights

The previous witness fixed reader directions e1 and e2. They are the two
largest-eigenvalue directions of the centered vocabulary-reader Gram matrix,
and were selected by the largest cross coefficient in the calibration quadratic.
That selection used weights, not this experiment's natural-text outcomes.

For normalized input u to the final bilinear MLP, define

\[
a=e_1^Tu,\quad b=e_2^Tu,\qquad
v_{12}=D[(Le_1)\odot(Re_2)+(Le_2)\odot(Re_1)].
\]

The native MLP contains the vector-valued term **a*b*v12**. Its output writer
comes directly from L,R,D; it is not fitted to an output direction. Bias and
all other terms remain in the MLP remainder. Two input vectors and one writer
require3,456 scalar coefficients. Keeping this component explicit does not
eliminate the original model or its remaining opaque weights.

The term is verified by a mixed finite difference: hold the input complement
c fixed and subtract the MLP outputs at c+a*e1 and c+b*e2 from its output at
c+a*e1+b*e2, then add MLP(c). All terms cancel except a*b*v12. The small
FP64 control agrees to1.9e-14. Flipping the sign of a reader changes its writer
sign as well, so the complete term stays unchanged.

These input projections also appear algebraically in the residual-skip
vocabulary numerator, through rho*a*Ue1 and rho*b*Ue2. That is an explicit
shared-input relationship. It does not mean we have independently extracted
their upstream producers or established a semantic meaning for either coordinate.

## The registered calibration screen fails

We removed the full term on the same42 FineWeb rows and16 Pile documents. The
old fitted calibration projection P_w splits its writer into P_w*v12 and
(I-P_w)*v12; we also removed each part separately. Three fixed random reader
pairs provide native-polynomial controls. All edits recompute the complete
final normalization and vocabulary readout.

Cross-entropy changes below are in nats; positive means worse prediction.

| Full-pair removal | FineWeb | Pile |
|---|---:|---:|
| All-token loss increase | 0.0092 | 0.0116 |
| Rare-token loss increase | 0.0155 | 0.0159 |
| Frequent-token loss change | −0.0155 | −0.0016 |
| Mean absolute random-pair rare change | 0.000021 | 0.0000093 |

Random specificity passes, but the calibration requirement was at least0.10
nat rare-token damage and0.02 nat frequent-token improvement on both corpora.
Those bars fail. For comparison, removing the earlier complete q projection
causes approximately0.50/0.57 nat rare-token damage. The native reader pair
explains only a much smaller effect under the tested intervention.

The full-pair mean loss increase has post-result row-bootstrap intervals
[0.0072,0.0113] on FineWeb and [0.0060,0.0169] on Pile. Full removal hurts
average prediction in42/42 and15/16 rows. These are useful descriptive effects,
not a replacement for the failed prospective criterion.

## One output direction misses substantial effects

The writer's projection onto the earlier calibration direction has norm72.5%
of the full writer norm. Yet its removal effect misses83.2%/79.1% of the full
centered vocabulary effect, far above the registered10% error limit.
The complement alone has effect magnitudes82.9%/77.3% of the full effect.
Thus geometric alignment with the fitted direction does not justify collapsing
this native writer to that direction.

Removing the two output parts separately also does not give additive losses:

| All-token CE change | FineWeb | Pile |
|---|---:|---:|
| Calibration-projected part | 0.0126 | 0.0108 |
| Complement part | 0.0113 | 0.0110 |
| Both, as the full term | 0.0092 | 0.0116 |
| Joint change minus sum of separate changes | −0.0148 | −0.0103 |

This interaction is negative in every evaluation row. The full nonlinear
readout predicts the edits correctly; adding separate loss numbers does not.
The complement's effect does not yet tell us which other semantic computations
it serves, so it is not evidence for two identified language circuits.

## Why the weight witness looked much larger

The constructed equal-energy inputs had a*b=±288. On natural inputs, a*b has
standard deviation13.66/13.80, so the witness magnitude is about21 times those
standard deviations. Its large q difference was a valid continuous-domain
counterexample, not a measure of typical behavior.

The pair's projected scalar contribution has only about15.9% of the original
q's root-mean-square magnitude. Its mean is−31 on FineWeb and1,576 on Pile,
versus original q means29,756 and31,080. These are magnitude comparisons,
not additive variance shares. This is why ranking a coefficient by its largest
possible continuous effect is insufficient for ranking natural circuit importance.

## What is established, and what is not

The complete native product and its output split are numerically sound.
Independent model replay is exact; online full/projected/complement edits agree
with the formulas within5.06e-5 maximum absolute logit error and6.92e-7 relative
error. The managed run used22 model-body forwards and86 sequence instances
in10.58 seconds.

A post-result CPU control also transforms the local MLP, readers and writer
consistently under an orthogonal coordinate change. The resulting readout agrees
to3.0e-13. This verifies a local coordinate-invariance property; it does not
permit rotating through untransformed RoPE or prove semantic identification.

The calibration claim is rejected. The exact component remains a useful
recorded native operation with a small causal effect, but all545,902,902 original
parameters and its upstream inputs remain required. Reused evaluation text,
unknown FineWeb document grouping and the limited Pile corpus shift remain
limitations. Independent extraction, semantic specificity, stable task-level
identification and structural savings have not been established.

Evidence: [preregistration](../SIGNED_READER_PAIR_V1_PREREGISTRATION.md),
[native result](../SIGNED_READER_PAIR_V1_RESULT.json),
[post-result audit](../SIGNED_READER_PAIR_AUDIT_V1_RESULT.json),
[native cross-term implementation](../signed_reader_pair_v1.py).
