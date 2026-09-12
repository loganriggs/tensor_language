# Shared-parent rank16 partners fail native fidelity

12 September2026. [Registration](COMPOSED_PARENT_NATIVE_V1_PREREGISTRATION.md),
[result](COMPOSED_PARENT_NATIVE_V1_RESULT.json),
[executed error accounting](COMPOSED_PARENT_SWAP_ERROR_V1.json).

The frozen test completed05:08:50 in2.37seconds. Numerical prediction A passes;
swap B, removal C and write D fail. The exact parent is retained in both arms.
No factors or ranks were selected using these128reused developmental endpoints.

| Family | Write error | Swap relative RMS | Swap sign agreement | Removal-CE disagreement |
|---|---:|---:|---:|---:|
|A1|17.56%|6.92%|100%|0.0450nats|
|A2|12.74%|44.09%|93.75%|0.0629nats|
|Past|13.44%|38.33%|100%|0.0315nats|
|Progressive|12.51%|41.55%|87.5%|0.0359nats|

Every family has16live swaps. The unchanged bars were5%write error,10%swap
error with90%sign agreement, and0.02nat mean absolute removal disagreement.
Overall write error is13.39%. Native inputs reduce the practical approximation
error relative to the broad coefficient tail, but not enough for faithful edits.

The exact star's signed mean removal CE is+0.0205/+0.0230/−0.0605/+0.0345nats
across these families. Positive means removal damages the measured target CE;
negative means it improves it. These mixed signs do not support an unqualified
morphology-helping label or selective causal interpretation. Full reference
swap mean magnitudes are0.225/0.066/0.085/0.065logit-margin units.

The coefficient capture replays to1.28e-14, compiled/native writes to1.59e-15,
and the duplicate exact-reference score has zero disagreement. The compiled
conditional approximation uses10,713,600values: native L16/R16,17folded
product-reader columns and16output writers. Background and normalization remain;
this is not a whole-model saving or an adopted circuit.

## Why the approximation loses swaps

Write the exact endpoint component as$w=sv$, where$s$includes the exact parent
and native MLP17 denominator. Only the partner is approximated, with error
$e=\widehat v-v$. For a donor/base pair,

$$
\Delta(\widehat w-w)=\overline s\,\Delta e+\overline e\,\Delta s.
$$

The first term is changing partner error at the mean scalar; the second is
changing scalar at mean partner error. The CPU check reproduces the actual
write-difference error to7.05e-15. The squared norms of these terms are not
independent percentages: their cross term can cancel strongly, especially in A1.

Projecting these two error vectors through the native answer/foil-margin
derivative at the midpoint between exact and approximate swap states predicts
the actual margin error within0.0035–0.0072%relative error. Partner-change versus
scalar-change contributions have RMS magnitudes:

| Family | Partner-change contribution | Scalar-change contribution | Actual total error |
|---|---:|---:|---:|
|A1|0.0300|0.0163|0.0183|
|A2|0.0263|0.0048|0.0288|
|Past|0.0307|0.0017|0.0307|
|Progressive|0.0291|0.0111|0.0333|

The failing families are primarily sensitive to how the omitted partners change
between endpoints. The exact parent was not lost; final-head nonlinear curvature
does not explain away the discrepancy. No data-guided repair follows from this
diagnostic. Repeating parent optimization or quietly changing the rank would
answer a different question.

The full shared component remains an explicit weight-defined computation with
stable identification under the four fitting starts. Its broad partner and
mixed behavioral effects need interpretation and alias checks before a circuit
claim. The rank16version is not a faithful replacement on this panel. Fresh/OOD
behavior, selective manipulation and useful joint composition remain unproven.
