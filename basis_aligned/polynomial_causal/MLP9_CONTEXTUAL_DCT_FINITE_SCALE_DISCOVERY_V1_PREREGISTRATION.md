# MLP9 contextual DCT finite-scale discovery V1

## Purpose

The downstream-reader discovery found a perfectly sign-stable local derivative
contrast but correctly failed its absolute-effect gate because the frozen DCT
directions have unit norm.  Absolute derivative amplitude is not invariant to
direction normalization.  On the same already opened 64-prefix discovery
panel, evaluate finite diagonal interventions along frozen input direction 2 at
scales `4, 8, 12, 16, 24, 32`.

For each scale, form the exact local MLP9 second difference

`h(z+2av) - 2 h(z+av) + h(z)`

and compare it with `a²` times the packaged Hessian response.  Propagate the
native, additive, packaged-install, and packaged-removal states through the
exact suffix.  The reader is frozen to token contrast `21215 ( -\")` minus
`6165 ( ultimately)` from the immutable discovery receipt.

## Selection rule

A scale is eligible only if aggregate local-state prediction error is at most
`.15`, packaged install predicts the exact local-term contrast effect within
`.20`, and packaged removal leaves at most `.35` of the native finite mixed
contrast.  Select the largest eligible scale.  Serialize every scale whether
or not one is eligible.

This is calibration on opened contexts.  It cannot establish OOD prediction,
selective removal, or behavioral semantics.  A selected scale only authorizes
a prospectively bound fresh-context test with matched random removals and
unrelated-token collateral.

## Price

One checkpoint load; 64 opened prefixes; six scales; seven suffix states per
scale; no gradients, new text, fitting, parameter updates, or fresh outcomes.
