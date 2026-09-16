# Subject-number response-weighted fresh causal test V1

Registered after the fourth-corpus native-only capability license passed and
before any candidate computation on those rows.

The candidate is fully frozen by
`SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_FROZEN_V1_ARTIFACT.json`: two
direction-specific 1,152-vectors, native axis, and four interaction coefficients.
For each fresh recipient/background state $x_b$, choose the frozen vector $p_d$
using the requested singular-to-plural or plural-to-singular intervention direction,
then compute

$$
z=u^TH(x_b),\qquad
\hat s=u^T[H(x_b+p_d)-H(x_b)],\qquad
\hat\alpha=[1,z,\hat s,z\hat s]\beta.
$$

Install the rank-one L11H3 write $\hat\alpha u$ on top of the recipient head.
The candidate reads no fresh opposite-number activation. Its declared ports are
the native recipient/background MLP8 state, frozen head weights, and one categorical
counterfactual-direction selector. The selector is not claimed to be natively
generated in this assay.

Run four primary arms for every row and all 16 E/A/U/W backgrounds:

- base reconstructed recipient head;
- exact row-specific opposite-number head (causal ceiling only);
- frozen direction/cardinality-law write (target compact program); and
- response-weighted candidate write.

Four deterministic ambient random vectors per direction, each exactly matched to
its candidate prototype norm, pass through the identical $H,z,s,zs,\beta$ pipeline.
They are same-site/equal-norm nulls, not direct random output writes.

Instrumentation requires a valid candidate-scoped capability license, 512 rows per
arm, exact decomposed-state closure at most `5e-5`, finite metrics, and no candidate
use of fresh donor activations. The candidate coefficient prediction passes at
cosine `>=.80`, relative L2 `<=.55`, and sign agreement `>=.90`, and must beat the
median equal-norm null coefficient error by `>=.05`.

Behavior relative to the frozen law passes at cosine `>=.85`, relative L2 `<=.50`,
and sign agreement `>=.90`, and must beat median null law-effect error by `>=.05`.
Relative to native exact causal effects it must reach cosine `>=.70`, relative L2
`<=.80`, and sign agreement `>=.75`. Each template and the intermediate-cardinality
subset must separately reach cosine `>=.60`, relative L2 `<=.90`, and sign agreement
`>=.65` against native exact.

A pass establishes prospective OOD prediction and an executable extracted generator
with its ports stated. It does not establish native generation of the direction
selector or selective removal. Those remain separate tests. A valid failure is not
repaired on this authority.
